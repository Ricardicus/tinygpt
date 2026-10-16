import os
import hashlib
import fnmatch
from array import array
from corpus import discover_files, text_chunks
import sys
import json
import random
import re
from pathlib import Path
from typing import Iterator, List, Tuple, Optional, Iterable, TypeVar
from typing import Optional, List, Iterator

T = TypeVar("T")


# -------------------------------------------------------------
# Optional: Generic streaming shuffle wrapper
# -------------------------------------------------------------
def shuffled_stream(reader: Iterable[T], buffer_size: int = 10_000) -> Iterator[T]:
    """
    Yield items from `reader` in (approximately) random order using a bounded buffer.

    - Reads up to `buffer_size` items into a buffer.
    - Each time the buffer is full, randomly pops one item to yield, replacing it with the next input item.
    - At the end, randomly drains the buffer.

    Keeps memory bounded for large datasets.
    """
    buffer: List[T] = []

    for item in reader:
        buffer.append(item)
        if len(buffer) >= buffer_size:
            idx = random.randrange(len(buffer))
            yield buffer.pop(idx)

    while buffer:
        idx = random.randrange(len(buffer))
        yield buffer.pop(idx)


# -------------------------------------------------------------
# TokenizedDataReader
# -------------------------------------------------------------
class TokenizedDataReader:
    """
    Reads all text files from a folder, tokenizes them with a BPE
    tokenizer, and yields a stream of token-ID windows of length
    ``context_size + 1`` (so the caller can split into x = w[:-1],
    y = w[1:]).

    Parameters
    ----------
    folder : str | Path
        Corpus directory, scanned recursively using the suffix configuration.
    bpe : BPE
        A trained BPE tokenizer instance (must have ``.encode()``).
    context_size : int
        Number of tokens per training window.  Each yielded window
        has length ``context_size + 1`` so that (x, y) pairs can be
        formed.
    shuffle : bool
        If *True* (default), windows are yielded in random order.
    lower_case : bool
        Lower-case all text before tokenizing.
    glob_pattern : str
        Optional additional filename filter applied to accepted corpus files.
    verbose : bool
        Print progress while tokenizing.
    cache : bool
        If *True*, save tokenized data to a binary cache file after
        the first tokenization, and load from it on subsequent runs.
        The cache file is stored next to the source folder as
        ``<folder_name>_tokens.bin``.  Default is *False*.
    cache_path : str | Path | None
        Explicit path for the cache file.  Overrides the default
        location derived from *folder*.
    """

    def __init__(
        self,
        folder: str | Path,
        bpe,
        context_size: int,
        shuffle: bool = True,
        lower_case: bool = True,
        glob_pattern: Optional[str] = None,
        verbose: bool = False,
        cache: bool = False,
        cache_path: Optional[str | Path] = None,
        corpus_config=None,
    ):
        self.folder = Path(folder)
        if not self.folder.is_dir():
            raise FileNotFoundError(
                f"Folder {self.folder} does not exist or is not a directory."
            )

        self.files = discover_files(self.folder, corpus_config)
        if glob_pattern:
            self.files = [p for p in self.files if fnmatch.fnmatch(p.name, glob_pattern)]
        if not self.files:
            raise FileNotFoundError("No matching corpus files")
        if context_size < 1:
            raise ValueError("context_size must be positive")
        self.bpe = bpe
        self.context_size = context_size
        self.shuffle = shuffle
        self.lower_case = lower_case
        self.glob_pattern = glob_pattern
        self.verbose = verbose
        self.cache = cache

        # Resolve cache file path.
        if cache_path is not None:
            self._cache_path = Path(cache_path)
        else:
            self._cache_path = self.folder.parent / f"{self.folder.name}_tokens.bin"

        manifest = {
            "version": 3,
            "files": [(str(p.resolve()), p.stat().st_size, p.stat().st_mtime_ns) for p in self.files],
            "vocab": bpe.vocab,
            "merges": list(bpe.merges.items()),
            "lower_case": lower_case,
            "bpe_lower_case": bpe.lower_case,
            "preserve_whitespace": bpe.preserve_whitespace,
            "end_of_word": bpe.end_of_word_mark,
        }
        self._fingerprint = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
        self._meta_path = Path(str(self._cache_path) + ".json")
        self._tokens = array("I")
        valid_cache = False
        if self.cache and self._cache_path.exists() and self._meta_path.exists():
            try:
                meta = json.loads(self._meta_path.read_text())
                valid_cache = (meta.get("fingerprint") == self._fingerprint
                               and meta.get("bytes") == self._cache_path.stat().st_size
                               and self._cache_path.stat().st_size % 4 == 0)
            except (ValueError, OSError):
                pass
        if valid_cache:
            self._load_cache()
        else:
            self._tokenize_folder()
            if self.cache:
                self._save_cache()
        self._n_windows = len(self._tokens) // (self.context_size + 1)

        if self.verbose:
            print(
                f"TokenizedDataReader: {len(self._tokens):,} tokens → "
                f"{self._n_windows:,} windows of size {window}"
            )

    # ---- cache I/O ------------------------------------------------
    def _save_cache(self):
        """Write token IDs to a compact binary file (4 bytes per token)."""
        with open(self._cache_path, "wb") as f:
            if sys.byteorder != "little":
                self._tokens.byteswap()
            try:
                self._tokens.tofile(f)
            finally:
                if sys.byteorder != "little":
                    self._tokens.byteswap()
        self._meta_path.write_text(json.dumps({"fingerprint": self._fingerprint,
                                               "bytes": self._cache_path.stat().st_size}))
        if self.verbose:
            print(
                f"  cache saved → {self._cache_path}  "
                f"({len(self._tokens):,} tokens, "
                f"{self._cache_path.stat().st_size / 1024 / 1024:.1f} MB)"
            )

    def _load_cache(self):
        """Read token IDs from a previously saved binary cache."""
        n_tokens = self._cache_path.stat().st_size // 4
        with open(self._cache_path, "rb") as f:
            self._tokens.fromfile(f, n_tokens)
        if sys.byteorder != "little":
            self._tokens.byteswap()
        if self.verbose:
            print(
                f"  cache loaded ← {self._cache_path}  "
                f"({n_tokens:,} tokens)"
            )

    # ---- internal -------------------------------------------------
    def _tokenize_folder(self):
        files = self.files
        for fp in files:
            for text in text_chunks(fp):
                if self.lower_case:
                    text = text.lower()
                # Report unknown characters instead of silently losing entire files.
                try:
                    self._tokens.extend(self.bpe.encode(text))
                except ValueError as error:
                    raise ValueError(f"Cannot tokenize {fp}: {error}") from error

        if self.verbose:
            print(
                f"  done – {len(files)} files, "
                f"{len(self._tokens):,} tokens total"
            )

    # ---- public API -----------------------------------------------
    def __len__(self):
        """Number of context windows available."""
        return self._n_windows

    def __iter__(self) -> Iterator[List[int]]:
        """
        Yield token windows of length ``context_size + 1``.

        If ``shuffle`` is *True*, windows are yielded in random order
        (compact in-memory index shuffle).
        """
        indices = array("Q", range(self._n_windows))
        if self.shuffle:
            random.shuffle(indices)
        for i in indices:
            yield self[i]

    def __getitem__(self, index: int) -> List[int]:
        if index < 0:
            index += self._n_windows
        if not 0 <= index < self._n_windows:
            raise IndexError(index)
        start = index * (self.context_size + 1)
        return self._tokens[start:start + self.context_size + 1].tolist()

    def __repr__(self):
        return (
            f"<TokenizedDataReader: {len(self._tokens):,} tokens, "
            f"{self._n_windows:,} windows of {self.context_size + 1}, "
            f"shuffle={self.shuffle}>"
        )


# -------------------------------------------------------------
# DataReader (simple eager JSON loader)
# -------------------------------------------------------------
class DataReader:
    def __init__(self, folder="data", lower_case=True):
        self.folder = Path(folder)
        if not self.folder.exists() or not self.folder.is_dir():
            raise FileNotFoundError(
                f"Folder {self.folder} does not exist or is not a directory."
            )

        self.lower_case = lower_case
        self._data = []
        self._load_files()

    def _load_files(self):
        files = sorted(self.folder.glob("*.json"))
        if not files:
            raise FileNotFoundError(f"No JSON files found in {self.folder}")

        for file in files:
            try:
                with open(file, "r", encoding="utf-8") as f:
                    content_list = json.load(f)
                    if not isinstance(content_list, list):
                        raise ValueError(f"File {file} does not contain a JSON list.")
                    for content in content_list:
                        data = content["chat"]
                        if self.lower_case:
                            data = data.lower()
                        self._data.append(data)
            except Exception as e:
                raise RuntimeError(f"Error reading {file}: {e}")

    def __getitem__(self, index):
        return self._data[index]

    def __len__(self):
        return len(self._data)

    def __iter__(self):
        return iter(self._data)

    def iter(self, shuffle=True, buffer_size=10_000):
        """Optional: return a shuffled or sequential iterator."""
        if not shuffle:
            return iter(self._data)
        return shuffled_stream(self._data, buffer_size)

    def __repr__(self):
        return f"<DataReader: {len(self)} entries from '{self.folder}'>"


# -------------------------------------------------------------
# PagedDataReader (lazy JSON reader with caching)
# -------------------------------------------------------------
class PagedDataReader:
    """
    Memory-efficient reader for a folder of JSON files where each file has a list
    of objects with a "chat" field.

    Keeps only a sliding window of chats in memory.
    """

    def __init__(self, folder: str = "data", cache_size: int = 1000, lower_case=True):
        self.folder = Path(folder)
        if not self.folder.exists() or not self.folder.is_dir():
            raise FileNotFoundError(
                f"Folder {self.folder} does not exist or is not a directory."
            )
        if cache_size <= 0:
            raise ValueError("cache_size must be > 0")

        self.cache_size = cache_size
        self.lower_case = lower_case
        self._file_index: List[Tuple[Path, int, int]] = []
        self._total_len: Optional[int] = None

        self._cache_start: int = 0
        self._cache: List[str] = []

        self._build_file_index()

    # ------------------- File indexing -------------------
    def _build_file_index(self):
        files = sorted(self.folder.glob("*.json"))
        if not files:
            raise FileNotFoundError(f"No JSON files found in {self.folder}")

        cum = 0
        for fp in files:
            length = self._count_file_items(fp)
            if length == 0:
                continue
            self._file_index.append((fp, length, cum))
            cum += length

        self._total_len = cum
        if self._total_len == 0:
            raise ValueError("All JSON files are empty (0 items).")

    def _count_file_items(self, path: Path) -> int:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if not isinstance(data, list):
                    raise ValueError(f"File {path} is not a JSON list.")
                return len(data)
        except Exception as e:
            raise RuntimeError(f"Error reading {path}: {e}")

    def __len__(self):
        return int(self._total_len or 0)

    # ------------------- Index lookup -------------------
    def _locate(self, global_index: int) -> Tuple[int, int]:
        if global_index < 0 or global_index >= len(self):
            raise IndexError("Index out of range")

        lo, hi = 0, len(self._file_index) - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            fp, length, start = self._file_index[mid]
            if global_index < start:
                hi = mid - 1
            elif global_index >= start + length:
                lo = mid + 1
            else:
                return mid, global_index - start

        raise IndexError("Failed to locate index")

    # ------------------- Cache loading -------------------
    def _load_window(self, start_index: int):
        start_index = max(0, min(start_index, len(self) - 1))
        end_index = min(start_index + self.cache_size, len(self))

        self._cache = []
        self._cache_start = start_index

        idx = start_index
        while idx < end_index:
            file_idx, local_idx = self._locate(idx)
            fp, length, start = self._file_index[file_idx]

            try:
                with open(fp, "r", encoding="utf-8") as f:
                    items = json.load(f)
                    take = min(length - local_idx, end_index - idx)

                    for j in range(local_idx, local_idx + take):
                        entry = items[j]["chat"]
                        if self.lower_case:
                            entry = entry.lower()
                        self._cache.append(entry)

                    idx += take

            except Exception as e:
                raise RuntimeError(f"Error loading window from {fp}: {e}")

    def _ensure_in_cache(self, index: int):
        if not (self._cache_start <= index < self._cache_start + len(self._cache)):
            window_start = (index // self.cache_size) * self.cache_size
            self._load_window(window_start)

    # ------------------- Public API -------------------
    def __getitem__(self, index: int) -> str:
        self._ensure_in_cache(index)
        return self._cache[index - self._cache_start]

    def __iter__(self) -> Iterator[str]:
        total = len(self)
        pos = 0
        while pos < total:
            self._load_window(pos)
            for item in self._cache:
                yield item
            pos += len(self._cache)

    def iter(self, shuffle=True, buffer_size=10_000):
        """Optional shuffled iterator."""
        base = self.__iter__()
        return shuffled_stream(base, buffer_size) if shuffle else base

    def __repr__(self):
        return (
            f"<PagedDataReader: {len(self)} entries from '{self.folder}', "
            f"cache_size={self.cache_size}, "
            f"cache_window=[{self._cache_start}, "
            f"{self._cache_start + len(self._cache) - 1 if self._cache else '∅'}]>"
        )

    # Extra utilities
    def prefetch_next(self):
        next_start = self._cache_start + self.cache_size
        if next_start < len(self):
            self._load_window(next_start)

    def current_window(self) -> Tuple[int, int]:
        return self._cache_start, self._cache_start + len(self._cache)

    def file_spans(self) -> List[Tuple[str, int, int]]:
        return [(fp.name, start, length) for fp, length, start in self._file_index]


class ChunkedFile:
    def __init__(
        self, file_path, buffer_size, to_new_line=False, round_off_at_whitespace=True
    ):
        self.file_path = file_path
        self.buffer_size = buffer_size
        self.to_new_line = to_new_line
        self.round_off_at_whitespace = round_off_at_whitespace

        # Read entire file content
        with open(file_path, "rb") as f:
            self.data = f.read()

        if self.to_new_line:
            # Split into lines (newline removed)
            self.chunks = self.data.split(b"\n")

            # Remove almost empty lines
            chunks = []
            for chunk in self.chunks:
                if len(chunk) > 2:
                    chunks.append(chunk)
            self.chunks = chunks

            # If enabled, collapse whitespace inside each line
            if self.round_off_at_whitespace:
                collapsed = []
                for line in self.chunks:
                    # Collapse any run of whitespace into one space
                    line = re.sub(rb"\s+", b" ", line.strip())
                    collapsed.append(line)
                self.chunks = collapsed

            self.num_chunks = len(self.chunks)

        elif self.round_off_at_whitespace:
            # 1) Normalize whitespace: collapse runs of whitespace to a single space.
            normalized = re.sub(rb"\s+", b" ", self.data.strip())

            # 2) Build chunks that are ~buffer_size but never cut words in half.
            self.chunks = []
            n = len(normalized)
            pos = 0

            while pos < n:
                # If remaining is smaller than buffer_size, take all of it.
                if pos + self.buffer_size >= n:
                    self.chunks.append(normalized[pos:])
                    break

                window_end = pos + self.buffer_size

                # Try to find last space within [pos, window_end]
                split_pos = normalized.rfind(b" ", pos, window_end + 1)

                if split_pos == -1 or split_pos == pos:
                    # No space in the window (e.g., very long word) or
                    # the only space is at the start; extend to next space after window.
                    next_space = normalized.find(b" ", window_end + 1)
                    if next_space == -1:
                        # No more spaces at all → last chunk
                        self.chunks.append(normalized[pos:])
                        break
                    else:
                        # Chunk up to that next space
                        self.chunks.append(normalized[pos:next_space])
                        pos = next_space + 1
                else:
                    # Found a space inside the window; cut there.
                    self.chunks.append(normalized[pos:split_pos])
                    pos = split_pos + 1

            self.num_chunks = len(self.chunks)

        else:
            # Original fixed-size chunks
            self.num_chunks = (len(self.data) + buffer_size - 1) // buffer_size

    def __getitem__(self, index):
        if not isinstance(index, int) or index < 0:
            raise IndexError("Index must be a non-negative integer.")
        if index >= self.num_chunks:
            raise IndexError("Chunk index out of range.")

        if self.to_new_line or self.round_off_at_whitespace:
            # In both these modes we precomputed self.chunks
            return self.chunks[index]
        else:
            start = index * self.buffer_size
            end = start + self.buffer_size
            return self.data[start:end]

    def __len__(self):
        return self.num_chunks


# -------------------------------------------------------------
# TextCorpusReader (raw txt files)
# -------------------------------------------------------------
class TextCorpusReader:
    """
    Streams accepted corpus files in bounded, whitespace-aligned chunks.

    Designed for very large corpora where loading everything is impossible.
    """

    def __init__(
        self,
        sources: Optional[List[str] | str] = "data",
        lower_case: bool = True,
        delimiter: Optional[str] = None,
        encoding: str = "utf-8",
        buffer_size: int = 240,
        shuffle=True,
        to_new_lines=[],
        round_off_at_whitespace=True,
        corpus_config=None,
    ):
        self.lower_case = lower_case
        self.delimiter = delimiter
        self.encoding = encoding
        self.buffer_size = buffer_size
        self.shuffle = shuffle

        if isinstance(sources, (str, Path)):
            folder = Path(sources)
            if not folder.exists() or not folder.is_dir():
                raise FileNotFoundError(
                    f"Folder {folder} does not exist or is not a directory."
                )
            self.files = [str(p) for p in discover_files(folder, corpus_config)]
        elif isinstance(sources, (list, tuple)):
            self.files = [str(Path(p)) for p in sources]
        else:
            raise TypeError("sources must be a folder path or list of paths")

        if not self.files:
            raise FileNotFoundError("No text files found.")

        self._length = None

    def __iter__(self):
        files = list(self.files)
        if self.shuffle:
            random.shuffle(files)
        for path in files:
            for text in text_chunks(path, chunk_size=max(1024, self.buffer_size)):
                yield text.lower() if self.lower_case else text

    def iter(self, buffer_size=10_000):
        return shuffled_stream(self, buffer_size) if self.shuffle else iter(self)

    def __len__(self):
        if self._length is None:
            self._length = sum(1 for _ in self)
        return self._length

    def __getitem__(self, index):
        if index < 0:
            raise IndexError(index)
        for i, text in enumerate(self):
            if i == index:
                return text
        raise IndexError(index)

    def __repr__(self):
        return f"<TextCorpusReader: {len(self.files)} files>"
