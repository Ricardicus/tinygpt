import json
import time
import re
import heapq
from collections import Counter, OrderedDict
from datetime import timedelta, datetime


class BPE:
    def __init__(
        self,
        vocab_size=1000,
        verbose=False,
        model_file="bpe_model.json",
        lower_case=True,
        end_of_word_mark="</w>",
        max_unique_words=100_000,
        max_word_chars=4096
    ):
        if max_unique_words < 1 or max_word_chars < 1:
            raise ValueError("BPE word limits must be positive")
        self.max_unique_words = max_unique_words
        self.max_word_chars = max_word_chars
        self.preserve_whitespace = True
        self.alphabet = set(" \t\r\n")
        self.skipped_words = 0
        self.vocab_size = vocab_size
        self._encoding_state = None
        self._word_cache = OrderedDict()
        self._merge_ranks = {}
        self.merges = {}  # Dict of (a, b) => freq_at_merge
        self.vocab = {}  # Token -> ID mapping
        self.rev_vocab = {}  # ID -> Token mapping
        self.word_freq = {}  # {tuple_of_symbols: count} — memory-efficient
        self.verbose = verbose
        self.model_file = model_file  # default save/load path
        self.lower_case = lower_case
        self.end_of_word_mark = end_of_word_mark

    def add_corpus(self, text: str):
        """Add raw text data to the training corpus (frequency-counted)."""
        if self.lower_case:
            text = text.lower()
        self.alphabet.update(text)
        for match in re.finditer(r"\S+", text):
            word = match.group()
            self.alphabet.update(word)
            if len(word) > self.max_word_chars:
                self.skipped_words += 1
                continue
            key = tuple(word)
            if not self.preserve_whitespace:
                key += (self.end_of_word_mark,)
            if key not in self.word_freq and len(self.word_freq) >= self.max_unique_words:
                self.skipped_words += 1
                continue
            self.word_freq[key] = self.word_freq.get(key, 0) + 1

    def get_end_of_word_mark(self):
        return self.end_of_word_mark

    def get_stats(self):
        """Count frequency of symbol pairs, weighted by word frequency."""
        pairs = Counter()
        for word, freq in self.word_freq.items():
            for i in range(len(word) - 1):
                pairs[(word[i], word[i + 1])] += freq
        return pairs

    def merge_vocab(self, pair):
        """Merge a given pair everywhere in the data (frequency-aware)."""
        new_symbol = pair[0] + pair[1]
        new_word_freq = {}
        for word, freq in self.word_freq.items():
            merged = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == pair[0] and word[i + 1] == pair[1]:
                    merged.append(new_symbol)
                    i += 2
                else:
                    merged.append(word[i])
                    i += 1
            key = tuple(merged)
            new_word_freq[key] = new_word_freq.get(key, 0) + freq
        self.word_freq = new_word_freq
        return new_symbol

    def train(self, batch_merges: int = 4):
        """
        Learn BPE merges until vocab size is reached or no more merges.

        Trains faster by merging multiple (batch_merges) pairs per iteration
        instead of only the single best one.
        """

        self._encoding_state = None
        if batch_merges < 1:
            raise ValueError("batch_merges must be positive")
        vocab = self.alphabet | set(ch for word in self.word_freq for ch in word)
        other_chars = ["'", '"', "<", "-", ">", "?", "!", ":"]
        for ch in other_chars:
            vocab.add(ch)
        iteration = 0

        while len(vocab) < self.vocab_size:
            start_time = time.time()

            # Count all pair frequencies once
            pairs = self.get_stats()
            if not pairs:
                break

            # Select top-k pairs to merge this round
            top_k = heapq.nlargest(batch_merges, pairs.items(), key=lambda x: x[1])
            del pairs

            # Merge them sequentially in descending frequency order
            for (a, b), freq in top_k:
                new_symbol = self.merge_vocab((a, b))
                self.merges[(a, b)] = freq
                vocab.add(new_symbol)
                if len(vocab) >= self.vocab_size:
                    break  # stop early if we reached target vocab size

            iteration += 1

            # --- progress output ---
            if self.verbose:
                elapsed = time.time() - start_time
                pct = int(len(vocab) / max(self.vocab_size, 1) * 100)
                print(
                    f"\rBPE training... {pct}% "
                    f"(iteration {iteration}, merged {len(self.merges)} pairs, "
                    f"batch size={batch_merges}, time={elapsed:.2f}s)",
                    end="",
                )

            # stop if we reached vocab target
            if len(vocab) >= self.vocab_size:
                break

        # ---- Build vocab dicts ----
        self.vocab = {tok: i for i, tok in enumerate(sorted(vocab))}
        self.rev_vocab = {i: tok for tok, i in self.vocab.items()}

        if self.verbose:
            print(
                f"\nBPE training complete: {len(self.vocab)} tokens, {len(self.merges)} merges."
            )
        self.word_freq.clear()
        if self.verbose and self.skipped_words:
            print(f"BPE sampling limits skipped {self.skipped_words:,} words.")
        return len(vocab) == self.vocab_size

    def encode_word(self, word):
        """Encode a single word using learned merges."""
        # Start with characters + end-of-word marker
        if self.lower_case:
            word = word.lower()
        state = (id(self.merges), id(self.vocab), self.preserve_whitespace,
                 self.lower_case, self.end_of_word_mark)
        if state != self._encoding_state:
            self._merge_ranks = {pair: rank for rank, pair in enumerate(self.merges)}
            self._word_cache.clear()
            self._encoding_state = state
        cached = self._word_cache.get(word)
        if cached is not None:
            self._word_cache.move_to_end(word)
            return list(cached)

        symbols = list(word)
        if not self.preserve_whitespace:
            symbols.append(self.end_of_word_mark)

        # Only inspect pairs that actually occur in this word. Advance through
        # merge ranks monotonically to preserve the original one-pass semantics,
        # including tokenizers trained with batches of merges.
        last_rank = -1
        while len(symbols) > 1:
            best_rank = len(self._merge_ranks)
            best_pair = None
            for i in range(len(symbols) - 1):
                pair = (symbols[i], symbols[i + 1])
                rank = self._merge_ranks.get(pair)
                if rank is not None and last_rank < rank < best_rank:
                    best_rank, best_pair = rank, pair
            if best_pair is None:
                break
            merged = []
            i = 0
            while i < len(symbols):
                if (i + 1 < len(symbols)
                        and (symbols[i], symbols[i + 1]) == best_pair):
                    merged.append(symbols[i] + symbols[i + 1])
                    i += 2
                else:
                    merged.append(symbols[i])
                    i += 1
            symbols = merged
            last_rank = best_rank

        try:
            ids = tuple(self.vocab[s] for s in symbols)
        except KeyError as e:
            raise ValueError(f"Unknown symbol during encoding: {e.args[0]}") from e
        # Bound both cache entries and per-entry word size.
        if len(word) <= 256:
            self._word_cache[word] = ids
            if len(self._word_cache) > 32768:
                self._word_cache.popitem(last=False)
        return list(ids)

    def encode(self, text: str):
        """Encode text into token IDs."""
        ids = []
        if self.lower_case:
            text = text.lower()
        if not self.preserve_whitespace:
            for word in text.strip().split():
                ids.extend(self.encode_word(word))
            return ids
        for match in re.finditer(r"\s+|\S+", text):
            piece = match.group()
            if piece.isspace():
                for char in piece:
                    if char not in self.vocab:
                        raise ValueError(f"Unknown whitespace during encoding: {char!r}")
                    ids.append(self.vocab[char])
            else:
                ids.extend(self.encode_word(piece))
        return ids

    def decode(self, ids):
        """Decode token IDs back into text."""
        text = "".join(self.rev_vocab[i] for i in ids)
        return text if self.preserve_whitespace else text.rstrip()

    def print_merges_info(self, top_k: int = 10):
        """Print the top-K merges by frequency at merge time."""
        if not self.merges:
            print("No merges learned.")
            return

        # Sort by frequency (descending) and take top_k
        top = sorted(self.merges.items(), key=lambda x: x[1], reverse=True)[:top_k]

        print(f"Total merges learned: {len(self.merges)}")
        print(f"Top {len(top)} merges by frequency at merge time:")
        for i, ((a, b), freq) in enumerate(top, 1):
            print(f"{i:3d}: ({a}, {b}) -> {a+b}, frequency: {freq}")

    # ---------- Persistence ----------
    def save(self, path: str = None):
        """Save the BPE model to a JSON file."""
        out_path = path or self.model_file
        data = {
            "vocab_size": self.vocab_size,
            "lower_case": self.lower_case,
            "preserve_whitespace": self.preserve_whitespace,
            "vocab": self.vocab,
            "end-of-word-mark": self.end_of_word_mark,
            # save merges as list of [a, b, freq] for portability
            "merges": [[a, b, freq] for (a, b), freq in self.merges.items()],
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        if self.verbose:
            print(f"Saved BPE model to {out_path}")

    def load(self, path: str = None):
        """
        Load tokenizer state from JSON (created by save()).
        Rebuilds vocab, rev_vocab, and merges; does NOT load training data.
        """
        in_path = path or self.model_file
        with open(in_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        # Missing flag denotes a legacy tokenizer; keep its token IDs/behavior.
        self._encoding_state = None
        self.preserve_whitespace = payload.get("preserve_whitespace", False)
        self.lower_case = payload.get("lower_case", self.lower_case)
        self.vocab_size = int(payload.get("vocab_size", self.vocab_size))
        self.end_of_word_mark = payload.get("end-of-word-mark", "</w>")

        # rebuild merges as a dict { (a,b): freq }
        raw_merges = payload.get("merges", [])
        self.merges = {(a, b): int(freq) for a, b, freq in raw_merges}

        # vocab: token -> id
        self.vocab = {str(k): int(v) for k, v in payload.get("vocab", {}).items()}
        self.rev_vocab = {v: k for k, v in self.vocab.items()}

        # training data intentionally left empty
        self.word_freq = {}

        if self.verbose:
            print(f"Loaded BPE model from {in_path}")
