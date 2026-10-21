#!/usr/bin/env python3
"""Combine downloaded English Wikipedia with up to 1 GiB of FineWeb-Edu prose.

Reuse a completed prepare_wikipedia_dataset.py output, or download a 256 MiB
Wikipedia subset if absent. FineWeb-Edu downloads in cached shards; the size cap
counts cleaned UTF-8 training text, not compressed downloads or token IDs.
Install requirements.txt first. Preparation never starts training.
"""
import argparse
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
import random
from pathlib import Path
import shlex
import threading
import time

if __package__:
    from . import prepare_wikipedia_dataset as wikipedia
else:
    import prepare_wikipedia_dataset as wikipedia

PROJECT = Path(__file__).resolve().parent.parent
FINEWEB = "HuggingFaceFW/fineweb-edu"
FINEWEB_CONFIG = "sample-10BT"


def download_fineweb(seed, revision=None):
    # Prefer the HTTP downloader to native Xet download workers on this path.
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    try:
        import pyarrow.parquet as parquet
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError as error:
        raise SystemExit("Install dependencies: python -m pip install -r requirements.txt") from error
    api = HfApi()
    revision = revision or api.dataset_info(FINEWEB).sha
    shards = sorted(name for name in api.list_repo_files(FINEWEB, repo_type="dataset", revision=revision)
                    if name.startswith("sample/10BT/") and name.endswith(".parquet"))
    if not shards:
        raise ValueError("No FineWeb-Edu sample-10BT shards found")
    rng = random.Random(seed)
    rng.shuffle(shards)

    def records():
        for index, name in enumerate(shards, 1):
            print(f"FineWeb-Edu shard {index}/{len(shards)}: downloading/reusing {name}", flush=True)
            for attempt in range(1, 4):
                try:
                    path = hf_hub_download(FINEWEB, name, repo_type="dataset", revision=revision)
                    break
                except Exception as error:
                    if attempt == 3:
                        raise RuntimeError("Shard download failed. Rerun with --resume to keep prepared text.") from error
                    print(f"Shard download failed ({error}); retry {attempt}/2...", flush=True)
                    time.sleep(attempt)
            print(f"FineWeb-Edu shard {index}: reading local Parquet", flush=True)
            # No remote range reads or threaded multi-shard Parquet prefetching.
            with parquet.ParquetFile(path) as source:
                for batch in source.iter_batches(batch_size=64, use_threads=False):
                    rows = batch.to_pylist()
                    rng.shuffle(rows)
                    yield from rows

    return records(), revision


def wikipedia_source(folder, max_mib, min_chars, seed):
    if not folder.exists() or (folder.is_dir() and not any(folder.iterdir())):
        print(f"Downloading up to {max_mib} MiB of Wikipedia into {folder}", flush=True)
        wikipedia.main(["--output", str(folder), "--max-mib", str(max_mib),
                        "--min-chars", str(min_chars), "--seed", str(seed)])
    summary_path = folder / "metadata/summary.json"
    manifest = folder / "metadata/files.jsonl"
    if not summary_path.is_file() or not manifest.is_file():
        raise ValueError(f"Not a completed Wikipedia download: {folder}. Choose another --wikipedia-dir.")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    source = summary.get("source", {})
    if source.get("dataset") != wikipedia.DATASET or not source.get("config", "").endswith(".en"):
        raise ValueError(f"Expected downloaded English Wikipedia at {folder}; fullEnglish exports are not used.")
    print(f"Reusing downloaded English Wikipedia: {folder}", flush=True)
    return summary


class CorpusWriter:
    """Write one document at a time, deduplicating both sources by content."""
    def __init__(self, root, manifest):
        self.root = root
        self.manifest = manifest
        self.seen = set()
        self.counts = {name: Counter(files=0, bytes=0) for name in ("wikipedia", "fineweb-edu")}

    def restore(self):
        """Verify interrupted output; preserve incomplete tails/orphans as metadata."""
        manifest_path = self.root / "metadata/files.jsonl"
        paths = set()
        valid_end = 0
        last_line = b""
        with manifest_path.open("rb") as source:
            while line := source.readline():
                try:
                    record = json.loads(line)
                except (ValueError, UnicodeDecodeError):
                    if source.read(1):
                        raise ValueError("Invalid manifest row before the final line; refusing to repair")
                    last_line = line
                    break
                rel = Path(record["path"])
                kind = record["kind"]
                path = self.root / rel
                if (kind not in self.counts or rel.is_absolute() or ".." in rel.parts
                        or rel.parts[:2] != ("corpus", kind) or rel.suffix != ".txt"
                        or path.is_symlink() or not path.resolve().is_relative_to((self.root / "corpus").resolve())
                        or rel in paths):
                    raise ValueError(f"Invalid/repeated manifest path: {rel}")
                data = path.read_bytes()
                digest = hashlib.sha256(data).hexdigest()
                if digest != record["sha256"] or len(data) != record["bytes"]:
                    raise ValueError(f"Hash/size mismatch in interrupted output: {path}")
                if not data.endswith(b"\n\n") or "\x00" in data.decode("utf-8"):
                    raise ValueError(f"Invalid prepared text: {path}")
                paths.add(rel)
                self.seen.add(digest)
                self.counts[kind]["files"] += 1
                self.counts[kind]["bytes"] += len(data)
                valid_end = source.tell()
                if len(paths) % 5000 == 0:
                    print(f"Verified {len(paths):,} existing documents...", flush=True)
        if last_line:
            backup = self.root / "metadata" / f"interrupted-manifest-tail-{time.time_ns()}.bin"
            backup.write_bytes(last_line)
            with manifest_path.open("r+b") as output:
                output.truncate(valid_end)
        elif valid_end:
            with manifest_path.open("r+b") as output:
                output.seek(valid_end - 1)
                if output.read(1) != b"\n":
                    output.seek(0, 2)
                    output.write(b"\n")
        # A crash can leave a document written before its buffered manifest row.
        # Move those documents out of training, preserving them for inspection.
        for path in (self.root / "corpus").rglob("*.txt"):
            rel = path.relative_to(self.root)
            if rel not in paths:
                backup = self.root / "metadata/recovered-orphans" / rel
                backup.parent.mkdir(parents=True, exist_ok=True)
                if backup.exists():
                    raise ValueError(f"Orphan backup already exists: {backup}")
                path.rename(backup)
        print(f"Verified {len(paths):,} existing documents; continuing without overwriting them.", flush=True)

    def add(self, kind, data, metadata):
        digest = hashlib.sha256(data).hexdigest()
        counts = self.counts[kind]
        if digest in self.seen:
            counts["duplicates"] += 1
            return False
        index = counts["files"]
        path = Path("corpus") / kind / f"{index // 1000:05d}" / f"{index:08d}.txt"
        target = self.root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        self.manifest.write(json.dumps({**metadata, "path": path.as_posix(), "kind": kind,
                                       "sha256": digest, "bytes": len(data)}, ensure_ascii=False) + "\n")
        self.manifest.flush()
        self.seen.add(digest)
        counts["files"] += 1
        counts["bytes"] += len(data)
        return True


def copy_wikipedia(folder, summary, writer):
    observed = Counter(files=0, bytes=0)
    with (folder / "metadata/files.jsonl").open(encoding="utf-8") as manifest:
        for line in manifest:
            record = json.loads(line)
            rel = Path(record["path"])
            path = folder / rel
            if (rel.is_absolute() or ".." in rel.parts or not rel.parts
                    or rel.parts[0] != "corpus" or rel.suffix != ".txt" or path.is_symlink()
                    or not path.resolve().is_relative_to((folder / "corpus").resolve())):
                raise ValueError(f"Invalid Wikipedia corpus path: {rel}")
            data = path.read_bytes()
            if len(data) != record["bytes"] or hashlib.sha256(data).hexdigest() != record["sha256"]:
                raise ValueError(f"Wikipedia hash/size mismatch: {path}")
            text = data.decode("utf-8")
            if "\x00" in text or not text.endswith("\n\n"):
                raise ValueError(f"Invalid prepared Wikipedia text: {path}")
            observed["files"] += 1
            observed["bytes"] += len(data)
            metadata = {key: record[key] for key in ("id", "url", "title") if key in record}
            writer.add("wikipedia", data, {**metadata, "source_path": str(path)})
    if observed["files"] != summary["files"] or observed["bytes"] != summary["bytes"]:
        raise ValueError("Wikipedia manifest totals do not match its summary")
    if not writer.counts["wikipedia"]["files"]:
        raise ValueError("Wikipedia source contains no usable files")


@contextmanager
def report_progress(writer, max_bytes, interval=15):
    """Report even while the streaming iterator waits for its next network read."""
    stopped = threading.Event()

    def report():
        while not stopped.wait(interval):
            counts = writer.counts["fineweb-edu"]
            remaining = max(0, max_bytes - counts["bytes"]) / 2**20
            print(f"FineWeb-Edu status: {counts['files']:,} documents; "
                  f"{counts['bytes'] / 2**20:.1f} / {max_bytes / 2**20:.1f} MiB; "
                  f"{remaining:.1f} MiB remaining; {counts['examined']:,} records examined. "
                  "Streaming/filtering is still running; unchanged counts may mean a network wait.",
                  flush=True)

    thread = threading.Thread(target=report, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stopped.set()
        thread.join()


def prepare_fineweb(records, writer, max_bytes, min_chars):
    counts = writer.counts["fineweb-edu"]
    if counts["bytes"] >= max_bytes:
        counts["stopped_at_byte_limit"] += 1
        return
    for record in records:
        counts["examined"] += 1
        if record.get("language") != "en" or record.get("language_score", 0) < 0.9:
            counts["language_filtered"] += 1
            continue
        raw = record.get("text", "")
        if "\x00" in raw:
            counts["invalid_text"] += 1
            continue
        text = wikipedia.clean_text(raw)
        if len(text.strip()) < min_chars:
            counts["too_short_after_filtering"] += 1
            continue
        data = text.encode("utf-8")
        if hashlib.sha256(data).hexdigest() in writer.seen:
            counts["duplicates"] += 1
            continue
        # Preserve whole documents; stop just below the cap rather than cutting one.
        if counts["bytes"] + len(data) > max_bytes:
            counts["stopped_at_byte_limit"] += 1
            break
        metadata = {key: record[key] for key in
                    ("id", "url", "language", "language_score", "score", "int_score") if key in record}
        writer.add("fineweb-edu", data, metadata)
        if counts["files"] % 1000 == 0:
            print(f"FineWeb-Edu: {counts['files']:,} documents; {counts['bytes'] / 2**20:.1f} MiB", flush=True)
        if counts["bytes"] == max_bytes:
            counts["stopped_at_byte_limit"] += 1
            break
    if not counts["files"]:
        raise ValueError("No FineWeb-Edu documents fit the byte limit after filtering")
    if not counts["stopped_at_byte_limit"]:
        raise ValueError("FineWeb-Edu stream ended before reaching the requested size; output is incomplete")


def write_helpers(root):
    launcher = f'''#!/usr/bin/env bash
set -euo pipefail
prepared_dir="$(cd -- "$(dirname -- "${{BASH_SOURCE[0]}}")" && pwd)"
python_bin="${{PYTHON:-{PROJECT}/venv/bin/python}}"
mkdir -p "$prepared_dir/artifacts"
exec "$python_bin" {shlex.quote(str(PROJECT / 'main.py'))} \\
  --rawdata "$prepared_dir/corpus" \\
  --data-config "$prepared_dir/training_data.json" \\
  --tokenizer "$prepared_dir/artifacts/tokenizer.json" \\
  --model "$prepared_dir/artifacts/model.pt" \\
  --lower-case false --vocab_size 8192 --bpe-part 1 --bpe-max-words 100000 \\
  --d-model 512 --num-layers 8 --n-heads 8 \\
  --context-length 512 --batch-size 4 --lr 0.0003 --epochs 3 \\
  --validation-fraction 0.05 --validation-every-steps 1000 \\
  --log-progress "$prepared_dir/artifacts/progress.csv" "$@"
'''
    (root / "train.sh").write_text(launcher, encoding="utf-8")
    (root / "README.md").write_text('''# English Wikipedia + FineWeb-Edu

Train with `bash train.sh`. Override with e.g. `bash train.sh --batch-size 2 --epochs 1`.
Select Python with `PYTHON=/path/to/python`. This requires the original TinyGPT
checkout and its dependencies. Preparation never starts training.

Fresh artifacts: artifacts/tokenizer.json, artifacts/model.pt, artifacts/progress.csv.
Defaults: 8,192-token case-sensitive vocabulary, width 512, 8 layers, 8 heads,
context 512, batch size 4, learning rate 0.0003, 3 epochs: about 29.7M parameters.
Adjust batch size to the device. Validation holds out 5% of files; use its loss to
judge additional epochs. Do not reuse the old checkpoint with the new tokenizer.

All of the cached Wikipedia subset is copied after checking hashes. If absent,
a 256 MiB subset is downloaded by default, rather than all English Wikipedia.
FineWeb-Edu downloads one cached sample-10BT shard at a time and reads it locally
in small batches, without threaded remote Parquet reads. Only needed shards are
fetched; an entire compressed shard can exceed the remaining text quota.
Its size cap counts cleaned UTF-8 text, keeping whole documents just below the
limit. Network traffic, Hub caches and filesystem/training overhead are extra.

FineWeb documents require English language metadata and language score >= 0.9.
Both sources use the Wikipedia prose filters and are exactly deduplicated across
sources. Filtering is heuristic; code snippets or foreign quotations may remain.
Each document is a .txt file. Training globally shuffles windows and uses sources
in proportion to token count, without source balancing.

metadata/files.jsonl stores original URLs, source kinds, hashes and byte counts.
metadata/summary.json records actual source sizes and downloaded revisions.
Keep metadata for provenance/attribution. Wikipedia lists CC BY-SA 3.0/GFDL;
FineWeb-Edu lists ODC-By and underlying website terms. Dataset cards:
https://huggingface.co/datasets/wikimedia/wikipedia
https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu

Token IDs and shuffle indices reside in RAM during training. The corpus cap is
not a total RAM/disk limit. After interruption, rerun the builder with --resume.
It verifies existing documents, replays cached shards and skips known documents;
replaying is not an exact iterator-position resume. Previously downloaded shards
are reused. --finalize-existing verifies and uses the collected subset offline,
even below the target size. Recovery preserves partial final manifest rows and
unlisted documents under metadata/ instead of including unverified text in training.
Old builds did not save source revisions before completion; use --fineweb-revision
to supply one if known. Otherwise metadata explicitly marks the old revision unknown.
''', encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("wikipedia-fineweb-data"))
    parser.add_argument("--wikipedia-dir", type=Path, default=Path("wikipedia-downloaded-data"),
                        help="Completed Wikipedia dataset to reuse, or destination for missing download")
    parser.add_argument("--wikipedia-mib", type=int, default=256,
                        help="Wikipedia limit only when downloading missing data (default: 256 MiB)")
    parser.add_argument("--fineweb-mib", type=int, default=1024,
                        help="Cleaned FineWeb text limit (default: 1024 MiB = 1 GiB)")
    parser.add_argument("--min-chars", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    recovery = parser.add_mutually_exclusive_group()
    recovery.add_argument("--resume", action="store_true", help="Verify and continue an interrupted output")
    recovery.add_argument("--finalize-existing", action="store_true",
                          help="Verify and finalize the existing partial corpus without downloading more")
    parser.add_argument("--fineweb-revision", help="Pin a Hub revision; useful when recovering an older build")
    args = parser.parse_args(argv)
    if min(args.wikipedia_mib, args.fineweb_mib, args.min_chars) < 1:
        parser.error("Size limits and --min-chars must be positive")
    root = args.output.expanduser().resolve()
    wiki_dir = args.wikipedia_dir.expanduser().resolve()
    if root == wiki_dir or root in wiki_dir.parents or wiki_dir in root.parents:
        parser.error("Output and Wikipedia directories must not contain each other")
    recovering = args.resume or args.finalize_existing
    if recovering:
        if not (root / "metadata/files.jsonl").is_file():
            parser.error("Recovery requires an existing metadata/files.jsonl")
        if (root / "metadata/summary.json").exists():
            parser.error("Dataset is already complete; choose a new output to change it")
    elif root.exists() and (not root.is_dir() or any(root.iterdir())):
        parser.error("Output must be new or empty; existing datasets are never overwritten")
    state_path = root / "metadata/build.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    if recovering and state:
        for name in ("seed", "min_chars", "fineweb_mib"):
            setattr(args, name, state[name])
        stored_revision = state["sources"]["fineweb-edu"]["revision"]
        if args.fineweb_revision and stored_revision and args.fineweb_revision != stored_revision:
            parser.error("--fineweb-revision differs from the interrupted build's recorded revision")
        args.fineweb_revision = args.fineweb_revision or stored_revision
    if recovering:
        cached_summary = wiki_dir / "metadata/summary.json"
        wiki_source = state.get("sources", {}).get("wikipedia")
        if wiki_source is None:
            source = json.loads(cached_summary.read_text())["source"] if cached_summary.exists() else {
                "dataset": wikipedia.DATASET, "config": None, "revision": None,
                "note": "Original Wikipedia snapshot not recorded by interrupted builder"}
            wiki_source = {"directory": str(wiki_dir), **source}
        writer = CorpusWriter(root, None)
        writer.restore()
    else:
        summary = wikipedia_source(wiki_dir, args.wikipedia_mib, args.min_chars, args.seed)
        wiki_source = {"directory": str(wiki_dir), **summary["source"]}
        (root / "metadata").mkdir(parents=True)
        with (root / "metadata/files.jsonl").open("w", encoding="utf-8") as manifest:
            writer = CorpusWriter(root, manifest)
            copy_wikipedia(wiki_dir, summary, writer)
    revision = args.fineweb_revision
    previous_revision_unknown = recovering and not state and revision is None
    if not args.finalize_existing:
        print(f"Collecting up to {args.fineweb_mib} MiB of cleaned FineWeb-Edu...", flush=True)
        records, revision = download_fineweb(args.seed, revision=revision)
        state = {"seed": args.seed, "min_chars": args.min_chars, "fineweb_mib": args.fineweb_mib,
                 "previous_revision_unknown": state.get("previous_revision_unknown", previous_revision_unknown),
                 "sources": {"wikipedia": wiki_source,
                             "fineweb-edu": {"dataset": FINEWEB, "config": FINEWEB_CONFIG, "revision": revision}}}
        wikipedia.write_json(state_path, state)
        try:
            with (root / "metadata/files.jsonl").open("a", encoding="utf-8") as manifest:
                writer.manifest = manifest
                if recovering:
                    # Reconcile any partially copied Wikipedia section as well.
                    summary = wikipedia_source(wiki_dir, args.wikipedia_mib, args.min_chars, args.seed)
                    copy_wikipedia(wiki_dir, summary, writer)
                with report_progress(writer, args.fineweb_mib * 2**20):
                    prepare_fineweb(records, writer, args.fineweb_mib * 2**20, args.min_chars)
        except KeyboardInterrupt:
            print("\nPreparation interrupted; verified text is kept. Rerun with --resume, "
                  "or --finalize-existing to use the collected subset.", flush=True)
            raise SystemExit(130)
        finally:
            close = getattr(records, "close", None)
            if close:
                close()
    if not all(counts["files"] for counts in writer.counts.values()):
        raise ValueError("Both Wikipedia and FineWeb-Edu need at least one verified document")
    print(f"FineWeb-Edu collected: {writer.counts['fineweb-edu']['bytes'] / 2**20:.1f} MiB. "
          "Finalizing metadata and training launcher...", flush=True)
    wikipedia.write_json(root / "training_data.json", {"suffixes": [".txt"], "exclude_dirs": []})
    totals = {"created_utc": datetime.now(timezone.utc).isoformat(),
              "files": sum(c["files"] for c in writer.counts.values()),
              "bytes": sum(c["bytes"] for c in writer.counts.values()),
              "sources": {"wikipedia": wiki_source,
                          "fineweb-edu": {"dataset": FINEWEB, "config": FINEWEB_CONFIG, "revision": revision,
                                          "previous_revision_unknown": state.get("previous_revision_unknown", previous_revision_unknown)}},
              "finalized_existing": args.finalize_existing,
              "limits": {"fineweb_bytes": args.fineweb_mib * 2**20, "seed": args.seed,
                         "min_chars": args.min_chars, "fineweb_language_score": 0.9},
              "counts": {name: dict(counts) for name, counts in writer.counts.items()}}
    write_helpers(root)
    # The summary marks completion; write it after all corpus files and helpers.
    wikipedia.write_json(root / "metadata/summary.json", totals)
    print(json.dumps(totals, indent=2), flush=True)
    print(f"Ready. Train with: bash {shlex.quote(str(root / 'train.sh'))}")


if __name__ == "__main__":
    main()
