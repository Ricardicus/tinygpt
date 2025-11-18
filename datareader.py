import os
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
    Streams text from .txt files line-by-line or chunk-by-chunk.

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
            self.files = sorted(str(p) for p in folder.glob("*.txt"))
        elif isinstance(sources, (list, tuple)):
            self.files = [str(Path(p)) for p in sources]
        else:
            raise TypeError("sources must be a folder path or list of paths")

        if not self.files:
            raise FileNotFoundError("No text files found.")

        self.chunked_file_readers = []
        for file in self.files:
            if os.path.basename(file) in to_new_lines:
                self.chunked_file_readers.append(
                    ChunkedFile(
                        file,
                        self.buffer_size,
                        to_new_line=True,
                        round_off_at_whitespace=round_off_at_whitespace,
                    )
                )
            else:
                self.chunked_file_readers.append(
                    ChunkedFile(
                        file,
                        self.buffer_size,
                        round_off_at_whitespace=round_off_at_whitespace,
                    )
                )
        self._length: Optional[int] = None

    # Iteration
    def __iter__(self) -> Iterator[str]:
        pairs = []
        for i in range(len(self.chunked_file_readers)):
            size = len(self.chunked_file_readers[i])
            for n in range(size):
                pairs.append((i, n))

        if self.shuffle:
            random.shuffle(pairs)

        for pair in pairs:
            reader, index = pair
            data = self.chunked_file_readers[reader][index].decode(
                self.encoding, errors="ignore"
            )
            if self.lower_case:
                data = data.lower()
            yield data

    def iter(self, buffer_size=10_000):
        """Optional shuffled iteration."""
        return self.__iter__()

    # Length
    def __len__(self) -> int:
        if self._length is None:
            self._length = sum(len(x) for x in self.chunked_file_readers)
        return self._length

    # Indexing (inefficient)
    def __getitem__(self, index: int) -> str:
        for i, text in enumerate(self):
            if i == index:
                return text
        raise IndexError("Index out of range")

    def __repr__(self):
        return f"<TextCorpusReader: {len(self.files)} files, delimiter={'lines' if not self.delimiter else repr(self.delimiter)}>"
