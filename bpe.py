import json
import time
from collections import Counter
from datetime import timedelta, datetime


class BPE:
    def __init__(
        self,
        vocab_size=1000,
        verbose=False,
        model_file="bpe_model.json",
        lower_case=True,
    ):
        self.vocab_size = vocab_size
        self.merges = {}  # Dict of (a, b) => freq_at_merge
        self.vocab = {}  # Token -> ID mapping
        self.rev_vocab = {}  # ID -> Token mapping
        self.data = []  # Accumulated training data
        self.verbose = verbose
        self.model_file = model_file  # default save/load path
        self.lower_case = lower_case

    def add_corpus(self, text: str):
        """Add raw text data to the training corpus."""
        if self.lower_case:
            text = text.lower()
        words = text.strip().split()
        for word in words:
            symbols = list(word) + ["</w>"]
            self.data.append(symbols)

    def get_stats(self):
        """Count frequency of symbol pairs."""
        pairs = Counter()
        for word in self.data:
            for i in range(len(word) - 1):
                pairs[(word[i], word[i + 1])] += 1
        return pairs

    def merge_vocab(self, pair):
        """Merge a given pair everywhere in the data."""
        new_symbol = pair[0] + pair[1]
        new_data = []
        for word in self.data:
            merged = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == pair[0] and word[i + 1] == pair[1]:
                    merged.append(new_symbol)
                    i += 2
                else:
                    merged.append(word[i])
                    i += 1
            new_data.append(merged)
        self.data = new_data
        return new_symbol

    def train(self, batch_merges: int = 4):
        """
        Learn BPE merges until vocab size is reached or no more merges.

        Trains faster by merging multiple (batch_merges) pairs per iteration
        instead of only the single best one.
        """

        vocab = set(ch for word in self.data for ch in word)
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
            top_k = sorted(pairs.items(), key=lambda x: x[1], reverse=True)[
                :batch_merges
            ]

            # Merge them sequentially in descending frequency order
            for (a, b), freq in top_k:
                new_symbol = self.merge_vocab((a, b))
                self.merges[(a, b)] = freq
                vocab.add(new_symbol)
                if len(vocab) >= self.vocab_size:
                    break  # stop early if we reached target vocab size

            iteration += 1

            # --- progress output ---
            if self.verbose and iteration % 5 == 0:
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
        return len(vocab) == self.vocab_size

    def encode_word(self, word):
        """Encode a single word using learned merges."""
        # Start with characters + end-of-word marker
        symbols = list(word) + ["</w>"]

        # Apply merges in the order they were learned
        for (a, b), _ in self.merges.items():
            i = 0
            while i < len(symbols) - 1:
                if symbols[i] == a and symbols[i + 1] == b:
                    symbols[i : i + 2] = [a + b]
                else:
                    i += 1

        # Convert symbols to IDs, raising an error if unseen
        try:
            return [self.vocab[s] for s in symbols]
        except KeyError as e:
            raise ValueError(f"Unknown symbol during encoding: {e.args[0]}")

    def encode(self, text: str):
        """Encode text into token IDs."""
        ids = []
        for word in text.strip().split():
            ids.extend(self.encode_word(word))
        return ids

    def decode(self, ids):
        """Decode token IDs back into text."""
        tokens = [self.rev_vocab[i] for i in ids]
        text = ""
        for tok in tokens:
            if tok == "</w>":
                text += " "  # end of word → space
            else:
                text += tok
        return text.strip()

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
            "vocab": self.vocab,
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

        self.vocab_size = int(payload.get("vocab_size", self.vocab_size))

        # rebuild merges as a dict { (a,b): freq }
        raw_merges = payload.get("merges", [])
        self.merges = {(a, b): int(freq) for a, b, freq in raw_merges}

        # vocab: token -> id
        self.vocab = {str(k): int(v) for k, v in payload.get("vocab", {}).items()}
        self.rev_vocab = {v: k for k, v in self.vocab.items()}

        # training data intentionally left empty
        self.data = []

        if self.verbose:
            print(f"Loaded BPE model from {in_path}")
