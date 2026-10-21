#!/usr/bin/env python3
"""Interactively select and download a mixed TinyGPT training corpus.

Run without arguments for a checkbox TUI, or use --plain for a line-based wizard.
GB limits count saved UTF-8 text (1 GB = 1,000,000,000 bytes), not RAM or traffic.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import hashlib
from html.parser import HTMLParser
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
    "gutenberg": {"label": "Gutenberg top 100 books (last 7 days)", "default": "full",
                  "url": "https://www.gutenberg.org/browse/scores/top"},
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


def default_folder(selected):
    parts = [f"{name}-{('full' if cap is None else format(Decimal(cap) / 1_000_000_000, 'f') + 'gb')}"
             for name, cap in selected.items()]
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
    print("Full FineWeb-Edu is a very large collection. Gutenberg full = all top 100.")
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
            root = validate_output(input(f"Output folder [{default_folder(selected)}]: ").strip() or default_folder(selected))
            break
        except ValueError as error:
            print(error)
    print_plan(selected, root)
    return (selected, root) if input("Start download? [y/N]: ").strip().lower() in ("y", "yes") else None


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
            put(3, "O: output folder   Enter: review/start   Q/Esc: cancel")
            for index, name in enumerate(names):
                attr = curses.A_REVERSE if cursor == index else 0
                put(5 + index, f" [{'x' if name in checked else ' '}] {SOURCES[name]['label']:<45} {limits[name]}" + (" GB" if limits[name] != "full" else ""), attr)
            put(10, "GB = saved UTF-8 text; RAM, network traffic and caches are extra.")
            put(11, "Full FineWeb-Edu = entire collection; full Gutenberg = the top 100.")
            put(13, "Folder: " + (folder or (default_folder(selected) if selected else "(select a source)")))
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
                        return selected, validate_output(folder or default_folder(selected))
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
            elif key in ("o", "O"):
                folder = edit("Folder (blank = automatic, Ctrl-U clears): ", folder).strip()
            elif key in ("\n", "\r", curses.KEY_ENTER):
                if not selected:
                    message = "Select at least one dataset with Space."
                else:
                    try:
                        validate_output(folder or default_folder(selected))
                        reviewing = True
                    except ValueError as error:
                        message = str(error)

    return curses.wrapper(screen)


def print_plan(selected, root):
    print(f"\nOutput: {root}")
    for name, limit in selected.items():
        print(f"  {SOURCES[name]['label']}: {'full' if limit is None else f'{limit / 1e9:g} GB'}")


class GutenbergChart(HTMLParser):
    """Read only the seven-day book list, preserving ranking and title text."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.found = False
        self.in_list = False
        self.finished = False
        self.current = None
        self.links = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if attrs.get("id") == "books-last7":
            self.found = True
        if self.found and not self.finished and tag == "ol":
            self.in_list = True
        if self.in_list and tag == "a" and attrs.get("href", "").startswith("/ebooks/"):
            self.current = [attrs["href"], ""]

    def handle_data(self, data):
        if self.current is not None:
            self.current[1] += data

    def handle_endtag(self, tag):
        if tag == "a" and self.current is not None:
            self.links.append(tuple(self.current))
            self.current = None
        if tag == "ol" and self.in_list:
            self.in_list = False
            self.finished = True


def gutenberg_records():
    """Fetch the live seven-day chart and preserve a snapshot for provenance."""
    from urllib.parse import urljoin

    def get(url):
        for attempt in range(3):
            try:
                request = Request(url, headers={"User-Agent": "TinyGPT-DatasetDownloader/1.0"})
                with urlopen(request, timeout=90) as response:
                    return response.read().decode("utf-8-sig"), response.url
            except (URLError, TimeoutError) as error:
                if isinstance(error, HTTPError) and error.code < 500 and error.code != 429:
                    raise
                if attempt == 2:
                    raise
                time.sleep(2 * (attempt + 1))

    page = GutenbergChart()
    page.feed(get(SOURCES["gutenberg"]["url"])[0])
    if len(page.links) != 100:
        raise ValueError("Gutenberg seven-day chart did not contain 100 books")
    for rank, (href, title) in enumerate(page.links, 1):
        url = urljoin("https://www.gutenberg.org", href)
        book_id = href.rstrip("/").split("/")[-1]
        time.sleep(1)
        download_url = f"https://www.gutenberg.org/ebooks/{book_id}.txt.utf-8"
        text, actual_url = get(download_url)
        if "*** START OF" not in text.upper():
            raise ValueError(f"Missing Gutenberg text marker in {download_url}")
        yield {"id": book_id, "rank": rank, "title": title.strip(),
               "url": url, "download_url": actual_url, "text": text}


def source_records(name, provenance):
    if name == "gutenberg":
        provenance["chart_retrieved_utc"] = datetime.now(timezone.utc).isoformat()
        provenance["chart"] = "Top 100 EBooks last 7 days"
        return gutenberg_records()
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    from datasets import load_dataset
    from huggingface_hub import HfApi
    source = SOURCES[name]
    revision = source.get("revision") or HfApi().dataset_info(source["dataset"]).sha
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


def prepare(selected, root):
    root = validate_output(root)
    (root / "metadata").mkdir(parents=True)
    (root / "corpus").mkdir()
    summary = {"created_utc": datetime.now(timezone.utc).isoformat(), "status": "downloading",
               "limits_bytes": selected, "sources": {}, "counts": {}}
    summary_path = root / "metadata/summary.json"
    # Disk-backed hashes keep full-dataset deduplication from growing a RAM set.
    with sqlite3.connect(root / "metadata/dedup.sqlite3") as hashes, \
            (root / "metadata/files.jsonl").open("w", encoding="utf-8") as manifest:
        hashes.execute("CREATE TABLE hashes (digest TEXT PRIMARY KEY)")
        try:
            for name, limit in selected.items():
                counts = Counter(files=0, bytes=0, examined=0)
                summary["counts"][name] = counts
                provenance = {"url": SOURCES[name]["url"]}
                summary["sources"][name] = provenance
                wikipedia.write_json(summary_path, summary)
                print(f"\n{SOURCES[name]['label']}: connecting...", flush=True)
                records = source_records(name, provenance)
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
                        if limit is not None and counts["bytes"] + len(data) > limit:
                            counts["stopped_at_byte_limit"] = 1
                            break
                        index = counts["files"]
                        rel = Path("corpus") / name / f"{index // 1000:05d}" / f"{index:08d}.txt"
                        (root / rel).parent.mkdir(parents=True, exist_ok=True)
                        (root / rel).write_bytes(data)
                        meta = {key: record[key] for key in
                                ("id", "url", "title", "rank", "download_url", "category") if key in record}
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
                        if limit is not None and counts["bytes"] == limit:
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
                "Full means all available records at the recorded source revision/chart,\n"
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
    args = parser.parse_args(argv)
    if not sys.stdin.isatty():
        parser.error("Interactive terminal required. Run this command directly in your terminal.")
    try:
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
