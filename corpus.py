"""Shared corpus discovery and bounded text streaming."""
import json
import os
import re
import time
from pathlib import Path

DEFAULT_CONFIG = Path(__file__).with_name("training_data.json")


def load_config(path=None):
    with open(path or DEFAULT_CONFIG, encoding="utf-8") as source:
        config = json.load(source)
    suffixes = config.get("suffixes")
    if not isinstance(suffixes, list) or not suffixes or any(
        not isinstance(s, str) or not s.startswith(".") or len(s) < 2
        for s in suffixes
    ):
        raise ValueError("Corpus config requires a nonempty 'suffixes' list (e.g. ['.txt']).")
    excluded = config.get("exclude_dirs", [])
    if not isinstance(excluded, list) or any(not isinstance(s, str) for s in excluded):
        raise ValueError("'exclude_dirs' must be a list of directory names.")
    return {"suffixes": [s.lower() for s in suffixes], "exclude_dirs": excluded}


def discover_files(folder, config=None, verbose=False):
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"Not a corpus directory: {folder}")
    config = config or load_config()
    suffixes = tuple(config["suffixes"])
    excluded = set(config["exclude_dirs"])
    files = []
    last_report = time.monotonic()
    if verbose:
        print(f"Discovering corpus files in {folder}...", flush=True)
    def onerror(error):
        raise error
    for root, dirs, names in os.walk(folder, followlinks=False, onerror=onerror):
        dirs[:] = sorted(d for d in dirs if d not in excluded)
        for name in sorted(names):
            path = Path(root) / name
            if name.lower().endswith(suffixes) and not path.is_symlink() and path.is_file():
                files.append(path)
            if verbose and time.monotonic() - last_report >= 2:
                print(f"Corpus discovery: {len(files):,} accepted files found...", flush=True)
                last_report = time.monotonic()
    if not files:
        raise FileNotFoundError(f"No accepted corpus files in {folder}")
    if verbose:
        print(f"Corpus discovery complete: {len(files):,} files.", flush=True)
    return files


def text_chunks(path, chunk_size=65536, max_word_chars=4096):
    """Stream text with exact whitespace; discard oversized words (e.g. blobs).

    Only a partial word crosses read boundaries. Invalid UTF-8 is replaced.
    NUL-containing chunks are treated as binary and rejected explicitly.
    """
    if chunk_size < 1 or max_word_chars < 1:
        raise ValueError("Chunk and word limits must be positive")
    with open(path, encoding="utf-8", errors="replace", newline="") as source:
        pending = ""
        dropping = False
        while True:
            block = source.read(chunk_size)
            if not block:
                break
            if "\x00" in block:
                raise ValueError(f"NUL bytes in accepted text file: {path}")
            if dropping:
                boundary = re.search(r"\s", block)
                if boundary is None:
                    continue
                block = block[boundary.start():]
                dropping = False
            text = pending + block
            boundary = list(re.finditer(r"\s+", text))
            if boundary:
                end = boundary[-1].end()
                yield re.sub(r"\S+", lambda m: m.group() if len(m.group()) <= max_word_chars else "", text[:end])
                pending = text[end:]
            else:
                pending = text
            if len(pending) > max_word_chars:
                pending = ""
                dropping = True
        if pending:
            yield pending
