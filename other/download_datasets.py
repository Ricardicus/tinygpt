#!/usr/bin/env python3
"""Interactively select and download a mixed TinyGPT training corpus.

Run without arguments for a checkbox TUI, or use --plain for a line-based wizard.
GB limits count saved UTF-8 text (1 GB = 1,000,000,000 bytes), not RAM or traffic.
Gutenberg uses a soft limit: finish the whole book that reaches the target.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import csv
from functools import lru_cache
import gzip
import io
import hashlib
import json
import os
from pathlib import Path
import shlex
import sqlite3
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

if __package__:
    from . import prepare_wikipedia_dataset as wikipedia
else:
    import prepare_wikipedia_dataset as wikipedia

# A small registry keeps adding sources independent of the selection UI.
SOURCES = {
    "wikipedia": {"label": "Wikipedia (English)", "default": "0.25",
                  "dataset": wikipedia.DATASET, "config": wikipedia.CONFIG,
                  "revision": wikipedia.REVISION,
                  "url": "https://huggingface.co/datasets/wikimedia/wikipedia"},
    "dolly": {"label": "Dolly 15k (instructions + answers)", "default": "full",
              "dataset": "databricks/databricks-dolly-15k", "config": None,
              "url": "https://huggingface.co/datasets/databricks/databricks-dolly-15k"},
    "fineweb-edu": {"label": "FineWeb-Edu (complete collection)", "default": "1.0",
                    "dataset": "HuggingFaceFW/fineweb-edu", "config": "default",
                    "url": "https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu"},
    "gutenberg": {"label": "Gutenberg books (full catalog)", "default": "0.25",
                  "url": "https://www.gutenberg.org/cache/epub/feeds/pg_catalog.csv.gz"},
}


def parse_limit(value):
    """Return an exact byte cap, or None for the full source."""
    if value.strip().lower() == "full":
        return None
    try:
        number = Decimal(value.strip())
        if not number.is_finite() or number <= 0:
            raise ValueError
        result = int(number * 1_000_000_000)
        if result < 1:
            raise ValueError
        return result
    except (InvalidOperation, ValueError, OverflowError):
        raise ValueError("Enter a positive GB amount (e.g. 0.5) or full") from None


def default_folder(selected, languages=("en",)):
    parts = [f"{name}-{('full' if cap is None else format(Decimal(cap) / 1_000_000_000, 'f') + 'gb')}"
             for name, cap in selected.items()]
    if "gutenberg" in selected:
        parts.append("-".join(languages))
    return "datasets/" + "_".join(parts)


def validate_output(value):
    root = Path(value).expanduser().resolve()
    if root.exists() and (not root.is_dir() or any(root.iterdir())):
        raise ValueError("Choose a new or empty folder; existing files are never overwritten")
    return root


def plain_plan():
    names = list(SOURCES)
    print("Choose datasets (comma-separated numbers, or all):")
    for index, source in enumerate(SOURCES.values(), 1):
        print(f"  {index}. {source['label']}")
    while True:
        answer = input("Sources [1,2,3,4]: ").strip() or "all"
        try:
            indices = list(range(1, len(names) + 1)) if answer.lower() == "all" else [int(i.strip()) for i in answer.split(",")]
            if not indices or any(i < 1 or i > len(names) for i in indices):
                raise ValueError
            selected_names = [name for i, name in enumerate(names, 1) if i in indices]
            break
        except ValueError:
            print("Enter numbers from 1 to 4 separated by commas, or all.")
    print("Limits are saved text in decimal GB, not RAM, network traffic or caches.")
    print("Gutenberg's GB target is soft: the final book is saved in full, even above the target.")
    print("Full FineWeb-Edu is a very large collection. Gutenberg full = all selected-language books.")
    languages = plain_languages() if "gutenberg" in selected_names else ("en",)
    selected = {}
    for name in selected_names:
        while True:
            value = input(f"{SOURCES[name]['label']} - GB or full [{SOURCES[name]['default']}]: ")
            try:
                selected[name] = parse_limit(value or SOURCES[name]["default"])
                break
            except ValueError as error:
                print(error)
    while True:
        try:
            root = validate_output(input(f"Output folder [{default_folder(selected, languages)}]: ").strip() or default_folder(selected, languages))
            break
        except ValueError as error:
            print(error)
    print_plan(selected, root, languages)
    return (selected, root, languages) if input("Start download? [y/N]: ").strip().lower() in ("y", "yes") else None


def tui_plan():
    import curses

    def screen(stdscr):
        height, width = stdscr.getmaxyx()
        if height < 19 or width < 75:
            raise curses.error("Terminal too small")
        curses.curs_set(0)
        stdscr.keypad(True)
        names = list(SOURCES)
        checked = set(names)
        limits = {name: source["default"] for name, source in SOURCES.items()}
        cursor = 0
        folder = ""
        message = ""
        reviewing = False
        languages = ("en",)

        def put(row, value, attr=0):
            height, width = stdscr.getmaxyx()
            if 0 <= row < height - 1 and width > 1:
                stdscr.addnstr(row, 0, value, width - 1, attr)

        def edit(prompt, initial):
            value = initial
            curses.curs_set(1)
            try:
                while True:
                    height, width = stdscr.getmaxyx()
                    stdscr.move(height - 2, 0)
                    stdscr.clrtoeol()
                    shown = prompt + value
                    stdscr.addnstr(height - 2, 0, shown[-max(1, width - 2):], max(1, width - 2))
                    key = stdscr.get_wch()
                    if key in ("\n", "\r"):
                        return value
                    if key == "\x1b":
                        return initial
                    if key in (curses.KEY_BACKSPACE, "\x7f", "\b"):
                        value = value[:-1]
                    elif key == "\x15":
                        value = ""
                    elif isinstance(key, str) and key.isprintable():
                        value += key
            finally:
                curses.curs_set(0)

        while True:
            stdscr.erase()
            selected = {name: parse_limit(limits[name]) for name in names if name in checked}
            put(0, "TinyGPT | Dataset downloader", curses.A_BOLD)
            put(2, "Up/Down: move   Space: select   A: all/none   E: edit GB   F: full")
            put(3, "L: Gutenberg languages   O: output folder   Enter: review/start   Q: cancel")
            for index, name in enumerate(names):
                attr = curses.A_REVERSE if cursor == index else 0
                put(5 + index, f" [{'x' if name in checked else ' '}] {SOURCES[name]['label']:<45} {limits[name]}" + (" GB" if limits[name] != "full" else ""), attr)
            put(10, "GB = saved UTF-8 text; RAM, network traffic and caches are extra.")
            put(11, "Gutenberg languages: " + ", ".join(languages) + " (L to change; default English)")
            put(12, "Gutenberg GB is a soft target: finish the final book, then stop.")
            put(13, "Folder: " + (folder or (default_folder(selected, languages) if selected else "(select a source)")))
            put(15, "REVIEW: Enter starts downloading; B returns to editing." if reviewing else message,
                curses.A_BOLD)
            stdscr.refresh()
            key = stdscr.get_wch()
            if key in ("q", "Q", "\x1b"):
                return None
            if reviewing:
                if key in ("b", "B"):
                    reviewing = False
                elif key in ("\n", "\r", curses.KEY_ENTER):
                    try:
                        return selected, validate_output(folder or default_folder(selected, languages)), languages
                    except ValueError as error:
                        reviewing, message = False, str(error)
                continue
            if key in (curses.KEY_UP, "k"):
                cursor = (cursor - 1) % len(names)
            elif key in (curses.KEY_DOWN, "j"):
                cursor = (cursor + 1) % len(names)
            elif key == " ":
                name = names[cursor]
                checked.symmetric_difference_update({name})
            elif key in ("a", "A"):
                checked = set() if len(checked) == len(names) else set(names)
            elif key in ("f", "F"):
                limits[names[cursor]] = "full"
            elif key in ("e", "E"):
                name = names[cursor]
                value = edit("GB or full (Ctrl-U clears): ", limits[name])
                try:
                    cap = parse_limit(value)
                    limits[name] = "full" if cap is None else format(Decimal(cap) / 1_000_000_000, "f")
                    message = ""
                except ValueError as error:
                    message = str(error)
            elif key in ("l", "L"):
                put(15, "Loading Gutenberg catalog to find its top 10 languages...", curses.A_BOLD)
                stdscr.refresh()
                try:
                    choices = gutenberg_languages()
                    languages = language_screen(stdscr, choices, languages)
                    message = ""
                except (OSError, ValueError) as error:
                    message = f"Catalog unavailable: {error}"
            elif key in ("o", "O"):
                folder = edit("Folder (blank = automatic, Ctrl-U clears): ", folder).strip()
            elif key in ("\n", "\r", curses.KEY_ENTER):
                if not selected:
                    message = "Select at least one dataset with Space."
                else:
                    try:
                        validate_output(folder or default_folder(selected, languages))
                        reviewing = True
                    except ValueError as error:
                        message = str(error)

    return curses.wrapper(screen)


def print_plan(selected, root, languages=("en",)):
    print(f"\nOutput: {root}")
    for name, limit in selected.items():
        print(f"  {SOURCES[name]['label']}: {'full' if limit is None else f'{limit / 1e9:g} GB'}")

    if "gutenberg" in selected:
        print("  Gutenberg languages: " + ", ".join(languages))
        print("  Gutenberg uses a soft GB target; the final book is always saved in full.")


LANGUAGE_NAMES = {
    "en": "English", "fr": "French", "fi": "Finnish", "de": "German",
    "nl": "Dutch", "it": "Italian", "es": "Spanish", "pt": "Portuguese",
    "hu": "Hungarian", "zh": "Chinese", "sv": "Swedish", "el": "Greek",
    "eo": "Esperanto", "la": "Latin", "ca": "Catalan", "da": "Danish",
}


def fetch_gutenberg(url):
    for attempt in range(3):
        try:
            request = Request(url, headers={"User-Agent": "TinyGPT-DatasetDownloader/1.0"})
            with urlopen(request, timeout=90) as response:
                return response.read(), response.url
        except (URLError, TimeoutError) as error:
            if isinstance(error, HTTPError) and error.code < 500 and error.code != 429:
                raise
            if attempt == 2:
                raise
            time.sleep(2 * (attempt + 1))


@lru_cache(maxsize=1)
def gutenberg_catalog():
    """Load the machine-readable catalog once per run, including for the UI."""
    data, url = fetch_gutenberg(SOURCES["gutenberg"]["url"])
    with gzip.GzipFile(fileobj=io.BytesIO(data)) as compressed:
        with io.TextIOWrapper(compressed, encoding="utf-8-sig", newline="") as text:
            reader = csv.DictReader(text)
            if not {"Text#", "Type", "Title", "Language", "Authors"}.issubset(reader.fieldnames or []):
                raise ValueError("Unrecognized Gutenberg CSV catalog columns")
            books = []
            for row in reader:
                if row["Type"] == "Text" and row["Text#"].isdigit():
                    books.append({"id": row["Text#"], "title": row["Title"],
                                  "authors": row["Authors"],
                                  "languages": [part.strip() for part in row["Language"].split(";") if part.strip()]})
    if not books:
        raise ValueError("Gutenberg catalog contains no text books")
    metadata = {"catalog_url": url, "catalog_sha256": hashlib.sha256(data).hexdigest(),
                "catalog_retrieved_utc": datetime.now(timezone.utc).isoformat()}
    return books, metadata


def gutenberg_languages():
    books, _ = gutenberg_catalog()
    counts = Counter(lang for book in books for lang in set(book["languages"]))
    return sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))[:10]


def language_screen(stdscr, choices, selected):
    import curses
    checked = set(selected)
    cursor = 0
    while True:
        stdscr.erase()
        height, width = stdscr.getmaxyx()
        lines = ["Gutenberg | Top 10 languages by catalog book count",
                 "Up/Down: move   Space: toggle   Enter: apply   Esc: back", ""]
        for code, count in choices:
            lines.append(f" [{'x' if code in checked else ' '}] {LANGUAGE_NAMES.get(code, code):<15} {count:,} books")
        lines.extend(["", "Select at least one language. The GB limit covers all selected languages together."])
        for row, line in enumerate(lines):
            if row < height - 1:
                stdscr.addnstr(row, 0, line, width - 1, curses.A_REVERSE if row == cursor + 3 else 0)
        stdscr.refresh()
        key = stdscr.get_wch()
        if key in ("\x1b", "q", "Q"):
            return selected
        if key in (curses.KEY_DOWN, "j"):
            cursor = (cursor + 1) % len(choices)
        elif key in (curses.KEY_UP, "k"):
            cursor = (cursor - 1) % len(choices)
        elif key == " ":
            checked.symmetric_difference_update({choices[cursor][0]})
        elif key in ("\n", "\r", curses.KEY_ENTER) and checked:
            return tuple(code for code, _ in choices if code in checked)


def plain_languages():
    print("Loading Gutenberg catalog for the top 10 languages...")
    choices = gutenberg_languages()
    for index, (code, count) in enumerate(choices, 1):
        print(f"  {index}. {LANGUAGE_NAMES.get(code, code)} ({count:,} books)")
    english = next(i for i, (code, _) in enumerate(choices, 1) if code == "en")
    while True:
        value = input(f"Gutenberg languages (comma-separated numbers) [{english}: English]: ").strip() or str(english)
        try:
            numbers = {int(part.strip()) for part in value.split(",")}
            if not numbers or any(i < 1 or i > len(choices) for i in numbers):
                raise ValueError
            return tuple(code for i, (code, _) in enumerate(choices, 1) if i in numbers)
        except ValueError:
            print("Select one or more of the listed language numbers.")


def gutenberg_records(languages=("en",), provenance=None, completed_ids=()):
    """Iterate the full catalog in book-ID order, without a book-count cap."""
    books, metadata = gutenberg_catalog()
    matching = [book for book in books if set(book["languages"]) & set(languages)]
    if provenance is not None:
        provenance.update(metadata, languages=list(languages), matching_books=len(matching),
                          order="ascending book ID",
                          skipped_unavailable=provenance.get("skipped_unavailable", 0))
    if not matching:
        raise ValueError("No Gutenberg books match the selected languages")
    print(f"Gutenberg: {len(matching):,} catalog books match {', '.join(languages)}", flush=True)
    for book in sorted(matching, key=lambda book: int(book["id"])):
        book_id = book["id"]
        if book_id in completed_ids:
            continue
        time.sleep(1)
        download_url = f"https://www.gutenberg.org/ebooks/{book_id}.txt.utf-8"
        try:
            try:
                data, actual_url = fetch_gutenberg(download_url)
            except HTTPError as error:
                if error.code not in (404, 406, 410):
                    raise
                # Format negotiation can fail even when the generated text exists.
                fallback = f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"
                time.sleep(1)
                data, actual_url = fetch_gutenberg(fallback)
            text = data.decode("utf-8-sig")
            if "*** START OF" not in text.upper():
                raise UnicodeError("Missing Gutenberg text marker")
        except (HTTPError, UnicodeError) as error:
            if isinstance(error, HTTPError) and error.code not in (404, 406, 410):
                raise
            if provenance is not None:
                provenance["skipped_unavailable"] += 1
            print(f"Gutenberg: skipping {book_id} ({book['title']}), no usable UTF-8 text edition", flush=True)
            continue
        yield {**book, "url": f"https://www.gutenberg.org/ebooks/{book_id}",
               "download_url": actual_url, "text": text}


def source_records(name, provenance, languages=("en",), completed_ids=()):
    if name == "gutenberg":
        provenance["languages"] = list(languages)
        return gutenberg_records(languages, provenance, completed_ids)
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    from datasets import load_dataset
    from huggingface_hub import HfApi
    source = SOURCES[name]
    revision = provenance.get("revision") or source.get("revision") or HfApi().dataset_info(source["dataset"]).sha
    provenance.update(dataset=source["dataset"], config=source["config"], revision=revision)
    return iter(load_dataset(source["dataset"], name=source["config"], split="train",
                             revision=revision, streaming=True))


def record_text(name, record):
    if name == "dolly":
        # Preserve short answers and code; prose filters would destroy these pairs.
        if not record.get("instruction", "").strip() or not record.get("response", "").strip():
            return ""
        parts = ["### Instruction\n" + record["instruction"].strip()]
        if record.get("context", "").strip():
            parts.append("### Context\n" + record["context"].strip())
        parts.append("### Response\n" + record["response"].strip())
        return "\n\n".join(parts) + "\n\n"
    text = record.get("text", "")
    if name == "gutenberg":
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        # Keep the complete book, including its Gutenberg license/notice.
        return text.strip() + "\n\n" if text.strip() else ""
    return wikipedia.clean_text(text)


def restore_download(root, summary, hashes):
    """Verify manifest documents before appending; rebuild dedup from actual files."""
    print("Verifying saved documents and rebuilding deduplication metadata...", flush=True)
    print("This reads the existing corpus from disk; downloads resume after verification.", flush=True)
    last_report = time.monotonic()
    verified_files = verified_bytes = 0
    selected = summary["limits_bytes"]
    if not selected or any(name not in SOURCES for name in selected):
        raise ValueError("Unrecognized dataset sources in the saved summary")
    counts = {name: Counter(summary.get("counts", {}).get(name, {})) for name in selected}
    for count in counts.values():
        count.update(files=-count["files"], bytes=-count["bytes"])
    completed_books = set()
    hashes.execute("DELETE FROM hashes")
    hashes.execute("CREATE TEMP TABLE restored_paths (path TEXT PRIMARY KEY)")
    manifest_path = root / "metadata/files.jsonl"
    with manifest_path.open(encoding="utf-8") as manifest:
        for line in manifest:
            record = json.loads(line)
            rel = Path(record["path"])
            name = record["kind"]
            path = root / rel
            if (name not in selected or rel.is_absolute() or ".." in rel.parts
                    or rel.parts[:2] != ("corpus", name) or rel.suffix != ".txt"
                    or path.is_symlink() or not path.resolve().is_relative_to((root / "corpus").resolve())):
                raise ValueError(f"Invalid saved corpus path: {rel}")
            index = counts[name]["files"]
            expected = Path("corpus") / name / f"{index // 1000:05d}" / f"{index:08d}.txt"
            if rel != expected:
                raise ValueError(f"Unexpected saved document sequence: {rel}")
            data = path.read_bytes()
            if len(data) != record["bytes"] or hashlib.sha256(data).hexdigest() != record["sha256"]:
                raise ValueError(f"Saved document hash/size mismatch: {rel}")
            if not data.endswith(b"\n\n") or "\x00" in data.decode("utf-8"):
                raise ValueError(f"Invalid saved document text: {rel}")
            try:
                hashes.execute("INSERT INTO restored_paths VALUES (?)", (rel.as_posix(),))
                hashes.execute("INSERT INTO hashes VALUES (?)", (record["sha256"],))
            except sqlite3.IntegrityError as error:
                raise ValueError(f"Repeated saved path or document: {rel}") from error
            counts[name]["files"] += 1
            counts[name]["bytes"] += len(data)
            verified_files += 1
            verified_bytes += len(data)
            if time.monotonic() - last_report >= 2:
                print(f"Verification: {verified_files:,} documents | {verified_bytes / 1e9:.4f} GB checked | {name}", flush=True)
                last_report = time.monotonic()
            if name == "gutenberg" and "id" in record:
                completed_books.add(str(record["id"]))
    if manifest_path.stat().st_size:
        with manifest_path.open("rb") as manifest:
            manifest.seek(-1, 2)
            if manifest.read(1) != b"\n":
                raise ValueError("Saved manifest is missing its final newline; preserve and repair it before resuming")
    print(f"Verified document contents: {verified_files:,} documents, {verified_bytes / 1e9:.4f} GB. "
          "Checking for unlisted corpus files...", flush=True)
    last_report = time.monotonic()
    scanned = 0
    for path in (root / "corpus").rglob("*.txt"):
        rel = path.relative_to(root).as_posix()
        if not hashes.execute("SELECT 1 FROM restored_paths WHERE path = ?", (rel,)).fetchone():
            raise ValueError(f"Unlisted corpus file: {rel}; preserve it outside corpus before resuming")
        scanned += 1
        if time.monotonic() - last_report >= 2:
            print(f"Corpus scan: {scanned:,} / {verified_files:,} files checked", flush=True)
            last_report = time.monotonic()
    hashes.execute("DROP TABLE restored_paths")
    summary["counts"] = counts
    print(f"Verified {sum(c['files'] for c in counts.values()):,} saved documents; continuing download.", flush=True)
    return completed_books


def prepare(selected, root, languages=("en",), resume=False):
    root = Path(root).expanduser().resolve() if resume else validate_output(root)
    if not resume:
        (root / "metadata").mkdir(parents=True)
        (root / "corpus").mkdir()
    summary = {"created_utc": datetime.now(timezone.utc).isoformat(), "status": "downloading",
               "limits_bytes": selected, "sources": {}, "counts": {},
               "gutenberg_languages": list(languages),
               "limit_modes": {name: "soft_whole_book" if name == "gutenberg" else "hard_whole_document"
                               for name in selected}}
    summary_path = root / "metadata/summary.json"
    if resume:
        print(f"Resuming dataset: {root}", flush=True)
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("status") == "complete":
            raise ValueError("Dataset is already complete")
        selected = summary["limits_bytes"]
        languages = tuple(summary.get("gutenberg_languages") or
                          summary.get("sources", {}).get("gutenberg", {}).get("languages", ["en"]))
    # Disk-backed hashes keep full-dataset deduplication from growing a RAM set.
    with sqlite3.connect(root / "metadata/dedup.sqlite3") as hashes, \
            (root / "metadata/files.jsonl").open("a" if resume else "w", encoding="utf-8") as manifest:
        hashes.execute("CREATE TABLE IF NOT EXISTS hashes (digest TEXT PRIMARY KEY)")
        completed_books = restore_download(root, summary, hashes) if resume else set()
        hashes.commit()
        summary["status"] = "downloading"
        summary.pop("error", None)
        try:
            for name, limit in selected.items():
                counts = Counter(summary["counts"].get(name, {}))
                counts.update(files=0, bytes=0, examined=0)
                if counts["files"] and (counts["stopped_at_byte_limit"] or counts["source_exhausted"]
                                        or (limit is not None and counts["bytes"] >= limit)):
                    print(f"{name}: already finished; reusing {counts['files']:,} verified documents.", flush=True)
                    continue
                summary["counts"][name] = counts
                provenance = summary["sources"].get(name, {"url": SOURCES[name]["url"]})
                summary["sources"][name] = provenance
                wikipedia.write_json(summary_path, summary)
                print(f"\n{SOURCES[name]['label']}: connecting...", flush=True)
                records = (source_records(name, provenance, languages, completed_books) if name == "gutenberg" and resume
                           else source_records(name, provenance, languages) if name == "gutenberg"
                           else source_records(name, provenance))
                wikipedia.write_json(summary_path, summary)
                last_report = 0
                try:
                    for record in records:
                        counts["examined"] += 1
                        if "\x00" in record.get("text", ""):
                            counts["invalid"] += 1
                            continue
                        text = record_text(name, record)
                        if not text.strip() or "\x00" in text:
                            counts["filtered"] += 1
                            continue
                        data = text.encode("utf-8")
                        digest = hashlib.sha256(data).hexdigest()
                        if hashes.execute("SELECT 1 FROM hashes WHERE digest = ?", (digest,)).fetchone():
                            counts["duplicates"] += 1
                            continue
                        if name != "gutenberg" and limit is not None and counts["bytes"] + len(data) > limit:
                            counts["stopped_at_byte_limit"] = 1
                            break
                        index = counts["files"]
                        rel = Path("corpus") / name / f"{index // 1000:05d}" / f"{index:08d}.txt"
                        (root / rel).parent.mkdir(parents=True, exist_ok=True)
                        (root / rel).write_bytes(data)
                        meta = {key: record[key] for key in
                                ("id", "url", "title", "authors", "languages", "download_url", "category") if key in record}
                        manifest.write(json.dumps({**meta, "kind": name, "path": rel.as_posix(),
                                                   "bytes": len(data), "sha256": digest}, ensure_ascii=False) + "\n")
                        manifest.flush()
                        hashes.execute("INSERT INTO hashes VALUES (?)", (digest,))
                        counts["files"] += 1
                        counts["bytes"] += len(data)
                        if time.monotonic() - last_report >= 2:
                            target = "full" if limit is None else f"{limit / 1e9:g} GB"
                            print(f"{name}: {counts['files']:,} documents | {counts['bytes'] / 1e9:.4f} GB / {target}", flush=True)
                            last_report = time.monotonic()
                            hashes.commit()
                            wikipedia.write_json(summary_path, summary)
                        if limit is not None and counts["bytes"] >= limit:
                            counts["stopped_at_byte_limit"] = 1
                            break
                    else:
                        counts["source_exhausted"] = 1
                finally:
                    close = getattr(records, "close", None)
                    if close:
                        close()
                if not counts["files"]:
                    raise ValueError(f"No usable {name} documents fit; select a larger limit")
                hashes.commit()
                print(f"{name}: finished, {counts['files']:,} documents, {counts['bytes'] / 1e9:.4f} GB", flush=True)
            wikipedia.write_json(root / "training_data.json", {"suffixes": [".txt"], "exclude_dirs": []})
            wikipedia.write_helpers(root)
            (root / "README.md").write_text(
                "# TinyGPT selected datasets\n\nTrain with `bash train.sh`. Preparation does not start training.\n"
                "Override Python using `PYTHON=/path/to/python bash train.sh`.\n"
                "This launcher requires the original TinyGPT checkout and its dependencies.\n\n"
                "Each corpus file is a whole document. GB caps count saved UTF-8 text only;\n"
                "network traffic, caches, filesystem overhead and training RAM are extra.\n"
                "Gutenberg uses a soft GB target: the last book is saved in full even if\n"
                "it crosses the target; other sources keep hard whole-document caps.\n"
                "Full means all available records at the recorded source revision/catalog,\n"
                "after prose filtering and exact deduplication. Gutenberg keeps book notices.\n"
                "Dolly retains instruction/context/response sections, including short answers.\n"
                "Training uses ordinary next-token prediction over the mixed corpus.\n\n"
                "See metadata/summary.json for source revisions, limits and actual totals,\n"
                "and metadata/files.jsonl for document URLs, IDs and hashes. Keep metadata\n"
                "for attribution; source cards describe dataset and underlying content licenses.\n" +
                "\n".join(f"- {SOURCES[name]['label']}: {SOURCES[name]['url']}" for name in selected) + "\n",
                encoding="utf-8")
            summary["status"] = "complete"
        except BaseException as error:
            summary["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
            summary["error"] = str(error)
            raise
        finally:
            summary["files"] = sum(c["files"] for c in summary["counts"].values())
            summary["bytes"] = sum(c["bytes"] for c in summary["counts"].values())
            wikipedia.write_json(summary_path, summary)
    print(f"\nReady. Train with: bash {shlex.quote(str(root / 'train.sh'))}")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plain", action="store_true", help="Use line-based prompts instead of the checkbox TUI")
    parser.add_argument("--resume", type=Path, metavar="FOLDER", help="Verify and continue a failed/interrupted download using its saved selections")
    args = parser.parse_args(argv)
    if not args.resume and not sys.stdin.isatty():
        parser.error("Interactive terminal required. Run this command directly in your terminal.")
    try:
        if args.resume:
            prepare({}, args.resume, resume=True)
            return
        if args.plain:
            plan = plain_plan()
        else:
            try:
                import curses
            except ImportError:
                print("Checkbox UI unavailable; using line-based selection.")
                plan = plain_plan()
            else:
                try:
                    plan = tui_plan()
                except curses.error:
                    print("Terminal cannot display the checkbox UI; using line-based selection.")
                    plan = plain_plan()
        if plan is None:
            print("Cancelled; nothing downloaded.")
            return
        print_plan(*plan)
        prepare(*plan)
    except (KeyboardInterrupt, EOFError):
        print("\nCancelled. Any downloaded text and metadata remain in the selected folder.")
        raise SystemExit(130)
    except (ValueError, OSError, ImportError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
