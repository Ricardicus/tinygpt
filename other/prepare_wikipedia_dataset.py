#!/usr/bin/env python3
"""Prepare English Wikipedia prose for TinyGPT, without a source-code corpus.

Downloads the cleaned English Wikipedia snapshot from Wikimedia's dataset on
Hugging Face. Requires `pip install datasets`; no local source data is needed.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shlex

PROJECT = Path(__file__).resolve().parent.parent
DATASET = "wikimedia/wikipedia"
CONFIG = "20231101.en"
# Pin the snapshot independently of future changes to the Hub's main branch.
REVISION = "b04c8d1ceb2f5cd4588862100d08de323dccfbaa"
CODE = re.compile(
    r"```|~~~|<\s*/?\s*(?:code|pre|syntaxhighlight|source)\b|"
    r"^\s*(?:#include\b|def\s+\w+\s*\(|class\s+\w+\s*[:({]|"
    r"(?:pub\s+)?fn\s+\w+\s*\(|(?:int|void)\s+main\s*\(|"
    r"import\s+\w+|from\s+\w+\s+import\b|\$\s+\w+)|"
    r"[{};]\s*$",
    re.I | re.M,
)
MARKUP = re.compile(r"<[^>]+>|\{\{|\}\}|\[\[")


def clean_text(text):
    """Keep prose paragraphs; conservatively drop code/markup/list fragments.

    English is selected at the dataset level, not inferred from ASCII characters.
    These filters cannot guarantee the absence of every foreign quote or snippet.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if "\x00" in text:
        raise ValueError("NUL byte in Wikipedia text")
    # Remove entire fenced/preformatted blocks before paragraph filtering.
    text = re.sub(r"```.*?```|~~~.*?~~~", "", text, flags=re.S)
    text = re.sub(
        r"<(pre|code|syntaxhighlight|source)\b[^>]*>.*?</\1\s*>",
        "", text, flags=re.I | re.S,
    )
    kept = []
    for paragraph in re.split(r"\n\s*\n", text):
        paragraph = paragraph.strip()
        if not paragraph or CODE.search(paragraph) or MARKUP.search(paragraph):
            continue
        if re.search(r"\S{4097}", paragraph):
            continue
        # Prose, rather than headings, tables, bibliographies or bare URLs.
        if len(re.findall(r"\b[^\W\d_]+\b", paragraph)) < 20:
            continue
        if sum(c.isalpha() for c in paragraph) / len(paragraph) < 0.65:
            continue
        kept.append(" ".join(paragraph.split()))
    return "\n\n".join(kept) + "\n\n" if kept else ""


def download_records(seed):
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise SystemExit("Download mode requires: python -m pip install datasets") from error
    dataset = load_dataset(DATASET, CONFIG, split="train", streaming=True,
                           revision=REVISION)
    # Bounded shuffle also shuffles Parquet shards, avoiding only early articles.
    return dataset.shuffle(seed=seed, buffer_size=1000)


def write_json(path, payload):
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def prepare(records, root, max_bytes, max_files, min_chars):
    (root / "corpus").mkdir(parents=True)
    (root / "metadata").mkdir()
    counts = Counter()
    seen = set()
    with (root / "metadata/files.jsonl").open("w", encoding="utf-8") as manifest:
        for record in records:
            counts["examined"] += 1
            text = clean_text(record["text"])
            if len(text.strip()) < min_chars:
                counts["too_short_after_filtering"] += 1
                continue
            data = text.encode("utf-8")
            digest = hashlib.sha256(data).hexdigest()
            if digest in seen:
                counts["duplicates"] += 1
                continue
            # Preserve whole source units and enforce a hard output-byte limit.
            if counts["bytes"] + len(data) > max_bytes:
                counts["over_byte_budget"] += 1
                break
            index = counts["files"]
            path = Path("corpus") / f"{index // 1000:05d}" / f"{index:08d}.txt"
            (root / path).parent.mkdir(parents=True, exist_ok=True)
            (root / path).write_bytes(data)
            seen.add(digest)
            metadata = {key: record.get(key, "") for key in ("id", "title", "url", "source")}
            metadata.update(path=path.as_posix(), bytes=len(data), sha256=digest)
            manifest.write(json.dumps(metadata, ensure_ascii=False) + "\n")
            counts["files"] += 1
            counts["bytes"] += len(data)
            if counts["files"] % 100 == 0:
                print(f"Prepared {counts['files']:,} files; {counts['bytes'] / 2**20:.1f} MiB", flush=True)
            if max_files and counts["files"] >= max_files:
                break
    if counts["files"] < 2:
        raise ValueError("Fewer than two usable files; increase limits or check the source. Choose a fresh output folder to retry.")
    write_json(root / "training_data.json", {"suffixes": [".txt"], "exclude_dirs": []})
    return dict(counts)


def write_helpers(root):
    # The trainer learns BPE on its training split and scans for character coverage.
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
  --lower-case false --vocab_size 8192 --bpe-part 1 --bpe-max-words 50000 \\
  --d-model 256 --num-layers 6 --n-heads 4 \\
  --context-length 512 --batch-size 8 --epochs 3 \\
  --validation-fraction 0.05 \\
  --log-progress "$prepared_dir/artifacts/progress.csv" "$@"
'''
    (root / "train.sh").write_text(launcher, encoding="utf-8")
    (root / "README.md").write_text(f'''# English Wikipedia prose

Run `bash train.sh` to train a fresh tokenizer and model, or
`PYTHON=/path/to/python bash train.sh --epochs 1` to override settings.
Training is not started by preparation. Artifacts stay under `artifacts/`.
The launcher requires the TinyGPT checkout at {PROJECT} and its dependencies.

Only `corpus/*.txt` (recursively) is used for training. Metadata is outside it.
The trainer holds out 5% of files for validation before learning BPE merges.
Each downloaded article is one file.
Exact cleaned-file duplicates are omitted; repeated passages are not deduplicated.

The source is the English Wikipedia edition. Paragraph filters remove recognizable
code, markup and short/list-like material. This is heuristic cleaning: foreign
quotes and unrecognized snippets may remain.
Unicode names and accents are preserved. See `metadata/summary.json` for source,
limits and filtering counts, and `metadata/files.jsonl` for file hashes and URLs.

Downloaded data: https://huggingface.co/datasets/wikimedia/wikipedia
Snapshot: {CONFIG}, revision {REVISION}.
The dataset card lists CC BY-SA 3.0 and GFDL; retain article URLs and attribution
when redistributing.

Use new tokenizer/model paths when changing the corpus. Token IDs and shuffled
window indices use RAM during training; start with a subset before scaling up.
''', encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("wikipedia-downloaded-data"))
    parser.add_argument("--max-mib", type=int, default=256, help="Hard output limit in MiB (default: 256)")
    parser.add_argument("--max-files", type=int, default=0, help="Optional accepted-file limit; 0 means no file limit")
    parser.add_argument("--min-chars", type=int, default=500, help="Minimum cleaned characters per file")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if args.max_mib < 1 or args.max_files < 0 or args.max_files == 1 or args.min_chars < 1:
        parser.error("Require positive --max-mib/--min-chars and --max-files of 0 or at least 2")
    root = args.output.expanduser().resolve()
    if root.exists() and (not root.is_dir() or any(root.iterdir())):
        parser.error("Output must be a new or empty directory; existing datasets are never overwritten")
    print(f"Downloading English Wikipedia ({CONFIG}) into {root}", flush=True)
    records = download_records(args.seed)
    provenance = {"dataset": DATASET, "config": CONFIG, "revision": REVISION, "unit": "article"}
    counts = prepare(records, root, args.max_mib * 2**20, args.max_files, args.min_chars)
    summary = {"created_utc": datetime.now(timezone.utc).isoformat(), "source": provenance,
               "options": {k: v for k, v in vars(args).items() if not isinstance(v, Path)}, **counts}
    write_json(root / "metadata/summary.json", summary)
    write_helpers(root)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Ready. Train with: bash {shlex.quote(str(root / 'train.sh'))}")


if __name__ == "__main__":
    main()
