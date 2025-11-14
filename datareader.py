import os
import json
from pathlib import Path
from typing import Iterator, List, Tuple, Optional

class DataReader:
    def __init__(self, folder="data", lower_case=True):
        self.folder = Path(folder)
        if not self.folder.exists() or not self.folder.is_dir():
            raise FileNotFoundError(f"Folder {self.folder} does not exist or is not a directory.")

        self.lower_case = lower_case
        self._data = []
        self._load_files()

    def _load_files(self):
        # Collect all .json files in folder (sorted for consistency)
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
        # Let Python naturally raise IndexError if out of bounds
        return self._data[index]

    def __len__(self):
        return len(self._data)

    def __iter__(self):
        return iter(self._data)

    def __repr__(self):
        return f"<DataReader: {len(self)} entries from '{self.folder}'>"



class PagedDataReader:
    """
    Memory-efficient reader for a folder of JSON files where each file contains a list of
    objects with a "chat" field. Only a sliding window (cache) of N items is kept in memory.

    - cache_size: maximum number of entries to keep in memory at once (default 1000)
    - Assumes each JSON file is a top-level list (e.g., [{...}, {...}, ...])

    Access pattern optimizations:
      * __getitem__ loads the window containing the requested index
      * __iter__ streams data block-by-block to avoid holding all items

    Housekeeping:
      * Builds a lightweight index of files with their list lengths
      * Maps global indices -> (file_idx, local_idx) using cumulative counts
    """

    def __init__(self, folder: str = "data", cache_size: int = 1000, lower_case=True):
        self.folder = Path(folder)
        if not self.folder.exists() or not self.folder.is_dir():
            raise FileNotFoundError(f"Folder {self.folder} does not exist or is not a directory.")
        if cache_size <= 0:
            raise ValueError("cache_size must be > 0")

        self.cache_size = cache_size
        self.lower_case = lower_case

        # file_index: List of (path, length, cum_start)
        self._file_index: List[Tuple[Path, int, int]] = []
        self._total_len: Optional[int] = None

        # Sliding cache state
        self._cache_start: int = 0          # global index of first item in cache
        self._cache: List[str] = []         # cached chats only

        self._build_file_index()

    # --------------------- Housekeeping & indexing ---------------------
    def _build_file_index(self):
        files = sorted(self.folder.glob("*.json"))
        if not files:
            raise FileNotFoundError(f"No JSON files found in {self.folder}")

        cum = 0
        self._file_index.clear()
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
        """Return the number of list elements in a JSON file without keeping the data.
        This uses json.load (which parses into memory). For huge files, consider using
        a streaming parser like `ijson`.
        """
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if not isinstance(data, list):
                    raise ValueError(f"File {path} does not contain a JSON list.")
                return len(data)
        except Exception as e:
            raise RuntimeError(f"Error reading {path}: {e}")

    def __len__(self) -> int:
        return int(self._total_len or 0)

    def _locate(self, global_index: int) -> Tuple[int, int]:
        """Map a global index -> (file_idx, local_idx)."""
        if global_index < 0 or global_index >= len(self):
            raise IndexError("Index out of range")
        # Binary search over cumulative ranges
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
        # Should never get here if inputs are valid
        raise IndexError("Failed to locate index")

    # --------------------- Cache loading ---------------------
    def _load_window(self, start_index: int):
        """Load a window [start_index, start_index + cache_size) into cache.
        Spans multiple files if necessary. Stores only the `chat` strings.
        """
        start_index = max(0, min(start_index, max(0, len(self) - 1)))
        end_index = min(start_index + self.cache_size, len(self))

        self._cache = []
        self._cache_start = start_index

        # Walk indices and collect chats across files
        idx = start_index
        while idx < end_index:
            file_idx, local_idx = self._locate(idx)
            fp, length, start = self._file_index[file_idx]

            # Load this file into memory once, then slice the needed portion
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    items = json.load(f)
                    # Determine how many from this file belong to the window
                    take_from_file = min(length - local_idx, end_index - idx)
                    # Append chats only
                    for j in range(local_idx, local_idx + take_from_file):
                        data = items[j]["chat"]
                        if self.lower_case:
                            data = data.lower()
                        self._cache.append(data)# raises if missing
                    idx += take_from_file
            except Exception as e:
                raise RuntimeError(f"Error loading window from {fp}: {e}")

    def _ensure_in_cache(self, index: int):
        if not (self._cache_start <= index < self._cache_start + len(self._cache)):
            # Align window to a multiple of cache_size for predictable paging
            window_start = (index // self.cache_size) * self.cache_size
            self._load_window(window_start)

    # --------------------- Public API ---------------------
    def __getitem__(self, index: int) -> str:
        self._ensure_in_cache(index)
        offset = index - self._cache_start
        return self._cache[offset]

    def __iter__(self) -> Iterator[str]:
        """Stream items block-by-block using the paging cache."""
        total = len(self)
        if total == 0:
            return
        pos = 0
        while pos < total:
            self._load_window(pos)
            for item in self._cache:
                yield item
            pos += len(self._cache)

    def __repr__(self):
        return (
            f"<PagedDataReader: {len(self)} entries from '{self.folder}', "
            f"cache_size={self.cache_size}, cache_window=[{self._cache_start},"
            f"{self._cache_start + len(self._cache) - 1 if self._cache else '∅'}]>"
        )

    # -------- Optional utilities --------
    def prefetch_next(self):
        """Prefetch the next cache window (useful for strictly linear access)."""
        next_start = self._cache_start + self.cache_size
        if next_start < len(self):
            self._load_window(next_start)

    def current_window(self) -> Tuple[int, int]:
        """Return the [start, end) global index range currently cached."""
        start = self._cache_start
        end = start + len(self._cache)
        return start, end

    def file_spans(self) -> List[Tuple[str, int, int]]:
        """Return a compact summary of file ranges: (filename, start_index, length)."""
        return [(fp.name, start, length) for fp, length, start in self._file_index]


class TextCorpusReader:
    """
    Efficient reader for large raw text corpora.

    - Can read from a folder (all .txt files) or an explicit list of file paths.
    - Streams lines or paragraphs (depending on `delimiter`).
    - Designed for very large datasets that cannot fit in memory.

    Parameters
    ----------
    sources : str | List[str]
        Folder containing text files or explicit list of paths.
    lower_case : bool, default=True
        Whether to convert text to lowercase.
    delimiter : Optional[str], default=None
        Split text on this delimiter. If None, yields one line per entry.
    encoding : str, default="utf-8"
        Encoding for reading text files.
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

        # Resolve sources
        if isinstance(sources, (str, Path)):
            folder = Path(sources)
            if not folder.exists() or not folder.is_dir():
                raise FileNotFoundError(f"Folder {folder} does not exist or is not a directory.")
            self.files = sorted(str(p) for p in folder.glob("*.txt"))
        elif isinstance(sources, (list, tuple)):
            self.files = [str(Path(p)) for p in sources]
        else:
            raise TypeError("sources must be a folder path or list of file paths")

        if not self.files:
            raise FileNotFoundError("No text files found.")

        # Optionally precompute line counts (lazy by default)
        self._length: Optional[int] = None

    # --------------------- Iteration ---------------------
    def __iter__(self) -> Iterator[str]:
        for path in self.files:
            with open(path, "r", encoding=self.encoding) as f:
                if self.delimiter is None:
                    for line in f:
                        text = line.strip()
                        if not text:
                            continue
                        yield text.lower() if self.lower_case else text
                else:
                    buffer = f.read()
                    for chunk in buffer.split(self.delimiter):
                        text = chunk.strip()
                        if text:
                            yield text.lower() if self.lower_case else text

    # --------------------- Optional utilities ---------------------
    def __len__(self) -> int:
        """Count total text entries (lazy; reads file headers if not cached)."""
        if self._length is None:
            self._length = sum(1 for _ in self)
        return self._length

    def __getitem__(self, index: int) -> str:
        """Access by index (inefficient for very large corpora)."""
        for i, text in enumerate(self):
            if i == index:
                return text
        raise IndexError("Index out of range")

    def __repr__(self):
        return f"<TextCorpusReader: {len(self.files)} files, delimiter={'lines' if not self.delimiter else repr(self.delimiter)}>"
