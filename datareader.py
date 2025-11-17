import os
import json
import random
from pathlib import Path
from typing import Iterator, List, Tuple, Optional, Iterable, TypeVar

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
    ):
        self.lower_case = lower_case
        self.delimiter = delimiter
        self.encoding = encoding

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

        self._length: Optional[int] = None

    # Iteration
    def __iter__(self) -> Iterator[str]:
        for path in self.files:
            with open(path, "r", encoding=self.encoding) as f:
                if self.delimiter is None:
                    for line in f:
                        text = line.strip()
                        if text:
                            yield text.lower() if self.lower_case else text
                else:
                    buffer = f.read()
                    for chunk in buffer.split(self.delimiter):
                        text = chunk.strip()
                        if text:
                            yield text.lower() if self.lower_case else text

    def iter(self, shuffle=False, buffer_size=10_000):
        """Optional shuffled iteration."""
        base = self.__iter__()
        return shuffled_stream(base, buffer_size) if shuffle else base

    # Length
    def __len__(self) -> int:
        if self._length is None:
            self._length = sum(1 for _ in self)
        return self._length

    # Indexing (inefficient)
    def __getitem__(self, index: int) -> str:
        for i, text in enumerate(self):
            if i == index:
                return text
        raise IndexError("Index out of range")

    def __repr__(self):
        return f"<TextCorpusReader: {len(self.files)} files, delimiter={'lines' if not self.delimiter else repr(self.delimiter)}>"
