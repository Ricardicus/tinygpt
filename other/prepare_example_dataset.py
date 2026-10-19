#!/usr/bin/env python3
"""Download and prepare books plus Rust/C/C++ repositories, including Linux.

Run without arguments to choose a destination interactively. Only Python's
standard library and git are required; no source code is built or executed.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import time
import unicodedata
from urllib.request import Request, urlopen
import zlib

PROJECT = Path(__file__).resolve().parent.parent
ROOT = PROJECT / "codex-prepared-data"
BOOKS_DIR = PROJECT / "rawdata"
LINUX_SOURCE = PROJECT / "code-bases/linux"
REPOS = [
    ("BurntSushi/ripgrep", "Rust"),
    ("rust-lang/regex", "Rust"),
    ("serde-rs/serde", "Rust"),
    ("clap-rs/clap", "Rust"),
    ("tokio-rs/bytes", "Rust"),
    ("rust-lang/rust", "Rust"),
    ("tokio-rs/tokio", "Rust"),
    ("rust-lang/rust-analyzer", "Rust"),
    ("bevyengine/bevy", "Rust"),
    ("pola-rs/polars", "Rust"),
    ("apache/datafusion", "Rust"),
    ("apache/arrow-rs", "Rust"),
    ("nushell/nushell", "Rust"),
    ("denoland/deno", "Rust"),
    ("bytecodealliance/wasmtime", "Rust"),
    ("astral-sh/ruff", "Rust"),
    ("helix-editor/helix", "Rust"),
    ("sharkdp/fd", "Rust"),
    ("sharkdp/bat", "Rust"),
    ("uutils/coreutils", "Rust"),
    ("madler/zlib", "C"),
    ("libarchive/libarchive", "C"),
    ("fmtlib/fmt", "C++"),
    ("nlohmann/json", "C++"),
    ("catchorg/Catch2", "C++"),
    ("torvalds/linux", "C/Rust"),
]
SUFFIXES = {".rs", ".c", ".h", ".cc", ".cpp", ".cxx", ".hh", ".hpp", ".hxx", ".inl", ".ipp", ".tpp"}
EXCLUDED = {".git", ".github", "target", "build", "dist", "node_modules", "vendor", "vendored", "third_party", "third-party", "external", "deps", "gtest", "googletest", "single_include", "extras", "unicode_tables", "ucd", "corpus", "corpora", "testdata", "test-data", "fixtures", "snapshots", "__pycache__"}
START = re.compile(r"\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG.*?\*\*\*", re.I)
END = re.compile(r"\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG", re.I)
GENERATED = re.compile(r"(?:automatically generated|auto[- ]generated|generated (?:by|from)|do not edit (?:this file|directly))", re.I)


BOOK_DOWNLOADS = [('10_The_Complete_Works_of_William_Shakespeare_by_William_Shakespeare_12838.txt', 100), ('kjv.txt', 10), ('39_The_Count_of_Monte_Cristo_by_Alexandre_Dumas_and_Auguste_Maquet_7241.txt', 1184), ('28_History_of_Tom_Jones_a_Foundling_by_Henry_Fielding_9039.txt', 6593), ('42_The_Brothers_Karamazov_by_Fyodor_Dostoyevsky_6823.txt', 28054), ('11_Middlemarch_by_George_Eliot_12406.txt', 145), ('32_Twenty_years_after_by_Alexandre_Dumas_and_Auguste_Maquet_8792.txt', 1259), ('33_My_Life__Volume_1_by_Richard_Wagner_8620.txt', 5197), ('02_Moby_Dick_Or_The_Whale_by_Herman_Melville_27809.txt', 2701), ('40_Leviathan_by_Thomas_Hobbes_7141.txt', 3207), ('18_Crime_and_Punishment_by_Fyodor_Dostoyevsky_10172.txt', 2554), ('49_The_Iliad_by_Homer_6172.txt', 6130), ('31_The_Adventures_of_Roderick_Random_by_T_Smollett_8868.txt', 4085), ('14_Little_Women_Or_Meg_Jo_Beth_and_Amy_by_Louisa_May_Alcott_11207.txt', 37106), ('15_Jane_Eyre_An_Autobiography_by_Charlotte_Bront_10771.txt', 1260), ('45_Great_Expectations_by_Charles_Dickens_6594.txt', 1400), ('30_The_Adventures_of_Ferdinand_Count_Fathom__Complete_by_T_Smollett_8950.txt', 6761), ('29_The_Expedition_of_Humphry_Clinker_by_T_Smollett_9015.txt', 2160), ('13_Dracula_by_Bram_Stoker_11278.txt', 345), ('37_A_Tale_of_Two_Cities_by_Charles_Dickens_8044.txt', 98), ('03_Pride_and_Prejudice_by_Jane_Austen_21103.txt', 1342), ('20_Wuthering_Heights_by_Emily_Bront_9992.txt', 768), ('48_Thus_Spake_Zarathustra_A_Book_for_All_and_None_by_Friedrich_Wilhelm_Nietzsche_63.txt', 1998), ('36_Adventures_of_Huckleberry_Finn_by_Mark_Twain_8217.txt', 76), ('41_The_Adventures_of_Sherlock_Holmes_by_Arthur_Conan_Doyle_6921.txt', 1661), ('47_Grimms_Fairy_Tales_by_Jacob_Grimm_and_Wilhelm_Grimm_6452.txt', 2591), ('09_The_Scarlet_Letter_by_Nathaniel_Hawthorne_13594.txt', 25344), ('24_The_Enchanted_April_by_Elizabeth_Von_Arnim_9599.txt', 16389), ('16_The_Picture_of_Dorian_Gray_by_Oscar_Wilde_10447.txt', 174), ('01_Frankenstein_Or_The_Modern_Prometheus_by_Mary_Wollstonecraft_Shelley_93616.txt', 84), ('04_The_King_in_Yellow_by_Robert_W_Chambers_19645.txt', 8492), ('26_Cranford_by_Elizabeth_Cleghorn_Gaskell_9229.txt', 394), ('19_The_Blue_Castle_a_novel_by_L_M_Montgomery_10163.txt', 67979), ('12_A_Room_with_a_View_by_E_M_Forster_12203.txt', 2641), ('44_The_Prince_by_Niccol_Machiavelli_6633.txt', 1232), ('21_Beowulf_An_Anglo-Saxon_Epic_Poem_9882.txt', 16328), ('25_The_Great_Gatsby_by_F_Scott_Fitzgerald_9363.txt', 64317), ('50_Doctrina_Christiana_6132.txt', 16119), ('23_A_Christmas_Carol_in_Prose_Being_a_Ghost_Story_of_Christmas_by_Charles_Dickens_9.txt', 46), ('08_Alices_Adventures_in_Wonderland_by_Lewis_Carroll_14963.txt', 11), ('17_A_Dolls_House__a_play_by_Henrik_Ibsen_10433.txt', 2542), ('07_The_Strange_Case_of_Dr_Jekyll_and_Mr_Hyde_by_Robert_Louis_Stevenson_16192.txt', 43), ('46_Metamorphosis_by_Franz_Kafka_6489.txt', 5200), ('22_The_Importance_of_Being_Earnest_A_Trivial_Comedy_for_Serious_People_by_Oscar_Wil.txt', 844), ('35_Fifteen_Thousand_Useful_Phrases_by_Grenville_Kleiser_8331.txt', 18362), ('27_A_Modest_Proposal_by_Jonathan_Swift_9203.txt', 1080), ('04_Spoon_River_Anthology_by_Edgar_Lee_Masters_20157.txt', 1280), ('43_Three_Hundred_Tang_Poems_by_Hengtangtuishi.txt', 52323), ('34_The_Junior_Classics_Volume_1_Fairy_and_wonder_tales_by_William_Allan_Neilson_848.txt', 3152)]
SUPPLEMENTAL_FILES = {
    'verify.py': r'''
#!/usr/bin/env python3
"""Verify corpus integrity and save full character coverage for tokenizer training."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from corpus import discover_files, load_config


def main():
    paths = set()
    digests = set()
    alphabet = set()
    counts = Counter()
    with (ROOT / "metadata/files.jsonl").open(encoding="utf-8") as manifest:
        for line in manifest:
            record = json.loads(line)
            path = ROOT / record["path"]
            assert path not in paths, f"Repeated path: {path}"
            paths.add(path)
            assert not path.is_symlink(), f"Symlink: {path}"
            data = path.read_bytes()
            digest = hashlib.sha256(data).hexdigest()
            assert digest == record["sha256"], f"Hash mismatch: {path}"
            assert len(data) == record["bytes"], f"Size mismatch: {path}"
            assert digest not in digests, f"Exact duplicate: {path}"
            digests.add(digest)
            text = data.decode("utf-8")
            assert "\x00" not in text, f"NUL byte: {path}"
            assert re.search(r"\S{4097}", text) is None, f"Oversized word: {path}"
            assert text.endswith("\n\n"), f"Missing boundary whitespace: {path}"
            alphabet.update(text)
            counts[record["kind"]] += 1
            counts["bytes"] += len(data)
    found = set(discover_files(ROOT / "corpus", load_config(ROOT / "training_data.json")))
    assert paths == found, "Loader discovery differs from the manifest"
    summary = json.loads((ROOT / "metadata/summary.json").read_text())
    assert len(paths) == summary["files"]
    assert counts["bytes"] == summary["bytes"]
    assert counts["book"] == summary["books"]["files"]
    assert counts["code"] == summary["code"]["files"]
    (ROOT / "metadata/alphabet.json").write_text(json.dumps(sorted(alphabet), ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Verified {len(paths):,} files; {counts['bytes']:,} bytes; {len(alphabet):,} distinct characters; no exact duplicate files.")
    return counts


if __name__ == "__main__":
    main()
''',
    'prepare_tokenizer.py': r'''
#!/usr/bin/env python3
"""Train a fresh tokenizer on a balanced sample, covering all corpus characters."""
from collections import defaultdict
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from bpe import BPE


def main():
    output = ROOT / "artifacts/tokenizer.json"
    if output.exists():
        payload = json.loads(output.read_text(encoding="utf-8"))
        vocab = payload.get("vocab", {})
        if payload.get("merges") and max(vocab.values(), default=-1) < payload["vocab_size"]:
            print(f"Using existing tokenizer: {output}", flush=True)
            return
        if (ROOT / "artifacts/model.pt").exists():
            raise ValueError("Invalid tokenizer has an associated model.pt; use fresh tokenizer/model paths before retraining.")
        print(f"Rebuilding tokenizer with no merges or an invalid vocabulary size: {output}", flush=True)
    output.parent.mkdir(exist_ok=True)
    groups = defaultdict(list)
    with (ROOT / "metadata/files.jsonl").open(encoding="utf-8") as manifest:
        for line in manifest:
            record = json.loads(line)
            groups[record.get("repository", "books")].append(ROOT / record["path"])
    rng = random.Random(20261019)
    for paths in groups.values():
        rng.shuffle(paths)
    tokenizer = BPE(vocab_size=4096, lower_case=False, verbose=True, max_unique_words=50_000)
    # Interleave groups before reaching the vocabulary word cap. Sample up to
    # 256 KiB per group, rather than learning merges from alphabetically first books.
    budget = {name: 256 * 1024 for name in groups}
    for index in range(max(map(len, groups.values()))):
        for name, paths in sorted(groups.items()):
            if index >= len(paths) or budget[name] <= 0:
                continue
            text = paths[index].read_text(encoding="utf-8")
            width = min(8192, budget[name])
            start = rng.randrange(max(1, len(text) - width + 1))
            sample = text[start:start + width]
            tokenizer.add_corpus(sample)
            budget[name] -= len(sample)
    # Alphabet was measured across every prepared file, not just the sample.
    alphabet = json.loads((ROOT / "metadata/alphabet.json").read_text(encoding="utf-8"))
    tokenizer.alphabet.update(alphabet)
    # Reserve at least 2,048 tokens for merges beyond the complete alphabet.
    base = tokenizer.alphabet | set("'\"<->?!:")
    tokenizer.vocab_size = max(4096, 1 << (len(base) + 2048 - 1).bit_length())
    print(f"Training tokenizer: {len(tokenizer.word_freq):,} unique sampled words, {len(tokenizer.alphabet):,} characters, target {tokenizer.vocab_size:,} tokens", flush=True)
    tokenizer.train()
    if not tokenizer.merges:
        raise ValueError("Tokenizer learned no merges; check that the corpus sample contains words.")
    temporary = output.with_suffix(".tmp")
    tokenizer.save(str(temporary))
    temporary.replace(output)
    print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()
''',
    'train.sh': r'''
#!/usr/bin/env bash
set -euo pipefail
prepared_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(dirname -- "$prepared_dir")"
python_bin="${PYTHON:-$project_dir/venv/bin/python}"
mkdir -p "$prepared_dir/artifacts"
"$python_bin" "$prepared_dir/prepare_tokenizer.py"
exec "$python_bin" "$project_dir/main.py" \
  --rawdata "$prepared_dir/corpus" \
  --data-config "$prepared_dir/training_data.json" \
  --tokenizer "$prepared_dir/artifacts/tokenizer.json" \
  --model "$prepared_dir/artifacts/model.pt" \
  --lower-case false --vocab_size 4096 \
  --d-model 256 --num-layers 6 --n-heads 4 \
  --context-length 512 --batch-size 8 --lr 0.0003 --epochs 3 \
  --validation-fraction 0.05 \
  --log-progress "$prepared_dir/artifacts/progress.csv" "$@"
''',
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def shingles(text):
    words = re.findall(r"\w+", unicodedata.normalize("NFKC", text).lower().replace("_", " "))
    # Deterministic 1/16 sample of seven-word shingles; punctuation/case independent.
    result = set()
    for i in range(len(words) - 6):
        h = zlib.crc32(" ".join(words[i:i + 7]).encode())
        if h % 16 == 0:
            result.add(h)
    return result


def books(records, omitted):
    candidates = []
    for source in sorted(BOOKS_DIR.glob("*.txt")):
        if source.name == "trump.txt":
            omitted.append({"source": str(source), "reason": "social media collection, not a book"})
            continue
        raw = source.read_bytes()
        text = raw.decode("utf-8-sig").replace("\r\n", "\n").replace("\r", "\n")
        notices = []
        start = START.search(text)
        if start:
            notices.append(text[:start.end()])
            text = text[start.end():]
        end = END.search(text)
        if end:
            notices.append(text[end.start():])
            text = text[:end.start()]
        text = text.strip() + "\n\n"
        candidates.append((source, raw, text, notices))
    # Prefer the most complete edition when a shorter copy is contained in it.
    candidates.sort(key=lambda item: (-len(item[2]), item[0].name))
    kept = []
    titles = {}
    for source, raw, text, notices in candidates:
        title = re.sub(r"_\d+$", "", re.sub(r"^\d+_", "", source.stem)).lower()
        if title in titles:
            omitted.append({"source": str(source), "reason": "alternate edition of same title", "kept": titles[title]})
            continue
        signature = shingles(text)
        duplicate = None
        for name, previous in kept:
            containment = len(signature & previous) / max(1, len(signature))
            if containment >= 0.90:
                duplicate = {"source": str(source), "reason": "duplicate/overlapping edition", "kept": name, "sampled_shingle_containment": round(containment, 6)}
                break
        if duplicate:
            omitted.append(duplicate)
            continue
        kept.append((source.name, signature))
        titles[title] = source.name
        dest = ROOT / "corpus/books" / source.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        data = text.encode("utf-8")
        dest.write_bytes(data)
        if notices:
            notice = ROOT / "metadata/book-notices" / source.name
            notice.parent.mkdir(parents=True, exist_ok=True)
            notice.write_text("\n\n".join(notices), encoding="utf-8")
        records.append({"path": str(dest.relative_to(ROOT)), "kind": "book", "source": str(source), "source_sha256": sha(raw), "sha256": sha(data), "bytes": len(data)})
        print("Book:", source.name, flush=True)


def code(clones, records, omitted, repositories=None):
    repositories = list(repositories or [])
    completed = {r["repository"] for r in repositories}
    seen = {}
    for record in records:
        if record["kind"] == "code":
            normalized = record.get("normalized_sha256")
            if normalized is None:
                text = (ROOT / record["path"]).read_text(encoding="utf-8")
                normalized = sha("\n".join(line.rstrip() for line in text.splitlines()).strip().encode("utf-8"))
                record["normalized_sha256"] = normalized
            seen[normalized] = record["path"]
    for repo, language in REPOS:
        if repo in completed:
            continue
        key = repo.replace("/", "--")
        source = LINUX_SOURCE if repo == "torvalds/linux" else clones / key
        commit = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
        tracked = subprocess.check_output(["git", "-C", str(source), "ls-files", "-z"]).decode().split("\0")
        dirty = subprocess.check_output(["git", "-C", str(source), "status", "--porcelain", "-uno"], text=True)
        if dirty:
            raise RuntimeError(f"Source checkout has tracked modifications: {source}")
        counts = Counter()
        license_paths = []
        for name in sorted(filter(None, tracked)):
            rel = Path(name)
            path = source / rel
            # Never follow source symlinks, including symlinked parents.
            if any(p.is_symlink() for p in [path, *list(path.parents)[:len(rel.parts) - 1]]):
                counts["symlink"] += 1
                continue
            if not path.is_file():
                continue
            if (rel.name.upper().startswith(("LICENSE", "LICENCE", "COPYING", "NOTICE", "COPYRIGHT")) or "LICENSES" in rel.parts) and ".git" not in rel.parts:
                dest = ROOT / "metadata/licenses" / key / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, dest)
                license_paths.append(str(dest.relative_to(ROOT)))
            if rel.suffix.lower() not in SUFFIXES:
                continue
            reason = None
            if language == "Rust" and rel.suffix.lower() != ".rs":
                reason = "non-Rust file in Rust repository"
            elif repo == "rust-lang/rust" and rel.parts[0] == "tests":
                reason = "compiler test fixture"
            elif set(rel.parts[:-1]) & (EXCLUDED | {"ui", "compile-fail", "compile_fail", "test_data"}):
                reason = "excluded directory"
            elif path.stat().st_size > 1024 * 1024:
                reason = "over 1 MiB"
            else:
                raw = path.read_bytes()
                try:
                    text = raw.decode("utf-8-sig")
                except UnicodeDecodeError:
                    reason = "invalid UTF-8"
                if reason is None:
                    if "\x00" in text:
                        reason = "NUL byte"
                    elif not text.strip():
                        reason = "empty"
                    elif GENERATED.search(text[:4096]):
                        reason = "generated-file marker in first 4096 characters"
                    elif re.search(r"\S{4097}", text):
                        reason = "oversized word/blob"
            if reason:
                counts[reason] += 1
                omitted.append({"repository": repo, "source": name, "reason": reason})
                continue
            # Preserve comments, indentation and case; normalize line endings and
            # add blank lines so the existing concatenating loader has a boundary.
            text = text.replace("\r\n", "\n").replace("\r", "\n")
            data = (text.rstrip("\n") + "\n\n").encode("utf-8")
            digest = sha(data)
            # Also catch copies differing only in trailing whitespace per line.
            normalized = sha("\n".join(line.rstrip() for line in text.splitlines()).strip().encode("utf-8"))
            if normalized in seen:
                counts["duplicate"] += 1
                omitted.append({"repository": repo, "source": name, "reason": "duplicate", "kept": seen[normalized]})
                continue
            dest = ROOT / "corpus/code" / key / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
            relative = str(dest.relative_to(ROOT))
            seen[normalized] = relative
            records.append({"path": relative, "kind": "code", "repository": repo, "commit": commit, "source": name, "source_sha256": sha(raw), "sha256": digest, "normalized_sha256": normalized, "bytes": len(data)})
            counts["kept"] += 1
            counts["bytes"] += len(data)
        repositories.append({"repository": repo, "url": f"https://github.com/{repo}", "commit": commit, "language": language, "licenses": license_paths, "counts": dict(counts)})
        print(f"Repository: {repo}: {counts['kept']:,} files, {counts['bytes'] / 1e6:.1f} MB", flush=True)
    return repositories


def download_books():
    BOOKS_DIR.mkdir(parents=True, exist_ok=True)
    downloads = []
    for name, ebook_id in BOOK_DOWNLOADS:
        dest = BOOKS_DIR / name
        # Use the HTTPS text file directly; the ebook shortcut can redirect to HTTP.
        url = f"https://www.gutenberg.org/cache/epub/{ebook_id}/pg{ebook_id}.txt"
        if not dest.exists():
            print(f"Downloading book {ebook_id}: {name}", flush=True)
            for attempt in range(3):
                try:
                    request = Request(url, headers={"User-Agent": "TinyGPT-dataset-preparer/1.0"})
                    with urlopen(request, timeout=90) as response:
                        data = response.read()
                    text = data.decode("utf-8-sig")
                    if not START.search(text) or not END.search(text):
                        raise ValueError(f"Missing Gutenberg text markers: {url}")
                    temporary = dest.with_suffix(".download")
                    temporary.write_bytes(data)
                    temporary.replace(dest)
                    break
                except Exception:
                    if attempt == 2:
                        raise
                    time.sleep(2 ** attempt)
        downloads.append({"file": name, "ebook_id": ebook_id, "url": url, "sha256": sha(dest.read_bytes())})
    write_json(ROOT / "metadata/book-downloads.json", downloads)


def fetch_repositories(clones, completed):
    clones.mkdir(parents=True, exist_ok=True)
    for repo, _ in REPOS:
        if repo in completed:
            continue
        dest = LINUX_SOURCE if repo == "torvalds/linux" else clones / repo.replace("/", "--")
        if dest.exists():
            expected = f"https://github.com/{repo}.git"
            origin = subprocess.check_output(["git", "-C", str(dest), "remote", "get-url", "origin"], text=True).strip()
            if origin.rstrip("/").removesuffix(".git") != expected.removesuffix(".git"):
                raise ValueError(f"Unexpected origin for {dest}: {origin}")
            print(f"Reusing checkout: {repo}", flush=True)
            continue
        print(f"Cloning {repo}...", flush=True)
        subprocess.run(["git", "clone", "--progress", "--depth", "1", "--single-branch", "--no-tags", f"https://github.com/{repo}.git", str(dest)], check=True)


def write_helpers(summary):
    for name, template in SUPPLEMENTAL_FILES.items():
        if name == "train.sh":
            template = template.replace('project_dir="$(dirname -- "$prepared_dir")"', "project_dir=" + shlex.quote(str(PROJECT)))
        else:
            template = template.replace('sys.path.insert(0, str(ROOT.parent))', "sys.path.insert(0, " + repr(str(PROJECT)) + ")")
        (ROOT / name).write_text(template, encoding="utf-8")
    # Verification only needs corpus discovery, not the training environment.
    verify_source = (ROOT / "verify.py").read_text()
    verify_source = verify_source.replace('from corpus import discover_files, load_config', '''def load_config(path):
    return json.loads(Path(path).read_text())

def discover_files(folder, config):
    suffixes = set(config["suffixes"])
    return [p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in suffixes]''')
    (ROOT / "verify.py").write_text(verify_source)
    (ROOT / "README.md").write_text(f'''# Prepared transformer training data

{summary["files"]:,} files, {summary["bytes"] / 1e9:.3f} GB of prepared UTF-8 text.
{summary["books"]["files"]} books and {summary["repositories"]} repositories, including Linux.

## Start training

From this dataset folder, run:

```bash
bash train.sh
```

The launcher trains a case-sensitive tokenizer using an interleaved
sample of books and repositories, with character coverage measured over the full
corpus. Its vocabulary target is at least 4,096 tokens and grows to leave room
for at least 2,048 merges beyond the full alphabet. The model uses the saved
tokenizer's vocabulary size. It then trains a transformer: 6 layers, width 256,
4 heads, context 512, batch size 8, and 3 epochs. Tokenizer/model/logs are saved
under `artifacts/`. It does not start training during dataset preparation.

The latest model and optimizer are saved every 1,000 optimizer steps and at epoch
end. Override with `bash train.sh --save-every-steps 500`. Each checkpoint replaces
the previous one atomically. Resuming a mid-epoch checkpoint restarts that epoch
with a new data shuffle, while retaining weights and optimizer state.

Override settings with, for example, `bash train.sh --batch-size 4 --epochs 1`.
`PYTHON=/path/to/python bash train.sh` selects another Python environment.
Training requires this TinyGPT project and its dependencies. Existing model and
tokenizer artifacts are reused; the trainer treats `--epochs` as the total epoch
target on resume. Use a fresh model/tokenizer path after changing the corpus.

## Contents and preparation

- `corpus/books/`: books with Gutenberg headers/footers separated into metadata.
- `corpus/code/`: filtered source trees; comments, case and indentation retained.
- `metadata/repositories.json`: URLs, commit hashes, licenses and per-repository counts.
- `metadata/files.jsonl`: source paths, original/prepared hashes and byte sizes.
- `metadata/omitted.json`: duplicate and filtering decisions.
- `metadata/licenses/` and `metadata/book-notices/`: retained upstream notices.
- `metadata/summary.json`: sizes by suffix and category.
- `metadata/alphabet.json`: character coverage used by the tokenizer helper.
- `sources/`: download cache, if sources were downloaded instead of reused locally.

Books are compared using sampled seven-word shingles after normalizing case and
punctuation; a shorter copy is omitted when at least 90% of its sampled shingles
occur in a retained book. Named alternate editions are also deduplicated. This
includes removing standalone Romeo and Juliet because it is already in the
Shakespeare collection. Social-media text is excluded. This is document-level
deduplication; it does not remove every repeated paragraph.

Code is deduplicated across repositories, including copies differing only in
trailing whitespace. Generated-file markers, common vendor/build directories,
amalgamations, compiler error fixtures, binary/invalid UTF-8 files, files over
1 MiB, and words over 4,096 characters are filtered. These are heuristics, not
a guarantee that every generated or repeated code fragment is removed. Rust
repositories contribute `.rs` files; C/C++ repositories contribute source and
headers. Linux contributes C/C++ and Rust; assembly/build scripts are not selected.

Line endings become LF and each file ends in two newlines. No special EOF token
is added: the current trainer concatenates files with that whitespace boundary.

The full corpus retains Linux, which still strongly influences the mixture.
See `metadata/summary.json` for actual Rust/C/C++ byte shares. The tokenizer sample
is balanced across sources; model training visits all prepared data without
source reweighting. The launcher holds out 5% of whole files with a fixed seed
and reports validation loss and perplexity before training, every 1,000 steps,
and at epoch end. Each check samples up to 100 batches using the same windows.
Scores are written to `artifacts/progress.csv.validation.csv`. Override with
`--validation-fraction`, `--validation-every-steps`, and `--validation-batches`;
`--validation-fraction 0` disables validation. The fraction is by file count,
not bytes; it is not a held-out repository test. The prepared tokenizer is shared
across both sets. For an unseen-data evaluation, train a fresh model with the
split enabled; a resumed model may already have seen the held-out files.

## Verify and rebuild

```bash
python3 verify.py
```

Run `{PROJECT / "other/prepare_example_dataset.py"}` to prepare a fresh dataset. It prompts for
a folder name; `--output PATH` chooses one noninteractively. The standalone script
requires Python 3.10+ and Git. It uses local `rawdata` and the local Linux checkout
when available; `--fresh-downloads` downloads everything. For downloaded books,
the KJV Bible uses Gutenberg ebook 10 rather than the project's local edition.
Downloads use current upstream revisions and record hashes/commits; future builds
can differ. Original local files and checkouts are never modified.
''', encoding="utf-8")


def main(argv=None):
    global ROOT, BOOKS_DIR, LINUX_SOURCE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Destination; prompts when omitted")
    parser.add_argument("--clones-root", type=Path, help="Reuse/cache repository checkouts here")
    parser.add_argument("--books-dir", type=Path, help="Use local .txt books instead of downloading")
    parser.add_argument("--linux-source", type=Path, help="Reuse an existing Linux checkout")
    parser.add_argument("--fresh-downloads", action="store_true", help="Ignore automatic local book/Linux reuse")
    parser.add_argument("--extend", action="store_true", help="Add missing repositories to a completed dataset")
    parser.add_argument("--resume", action="store_true", help="Reuse an interrupted download directory before corpus creation")
    parser.add_argument("--refresh-helpers", action="store_true", help="Update helper scripts in a completed dataset without rebuilding its corpus")
    args = parser.parse_args(argv)
    if args.output is None:
        default = "codex-prepared-data"
        index = 2
        while Path(default).exists():
            default = f"codex-prepared-data-{index}"
            index += 1
        try:
            answer = input(f"Dataset folder [{default}]: ").strip()
        except EOFError:
            parser.error("Pass --output PATH when running without interactive input")
        args.output = Path(answer or default)
    ROOT = args.output.expanduser().resolve()
    if ROOT == PROJECT or ROOT == Path.home() or ROOT == Path(ROOT.anchor):
        parser.error("Choose a dedicated dataset directory")
    if ROOT.exists() and not ROOT.is_dir():
        parser.error("Destination exists and is not a directory")
    if args.refresh_helpers:
        summary = ROOT / "metadata/summary.json"
        if not summary.is_file():
            parser.error("--refresh-helpers requires a completed dataset")
        write_helpers(json.loads(summary.read_text(encoding="utf-8")))
        print(f"Updated dataset helpers in {ROOT}", flush=True)
        return
    if args.extend:
        if not (ROOT / "metadata/files.jsonl").is_file():
            parser.error("--extend requires a completed dataset")
    elif ROOT.exists() and any(ROOT.iterdir()) and not args.resume:
        parser.error("Destination is not empty; choose another folder, --extend, or --resume")
    if (ROOT / "corpus").exists() and not args.extend:
        parser.error("A corpus already exists; use --extend or choose another folder")
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "metadata").mkdir(exist_ok=True)
    clones = (args.clones_root or ROOT / "sources/repos").expanduser().resolve()
    local_linux = args.linux_source or (PROJECT / "code-bases/linux" if not args.fresh_downloads else clones / "torvalds--linux")
    LINUX_SOURCE = local_linux.expanduser().resolve() if local_linux.exists() else clones / "torvalds--linux"
    records, omitted, repos = [], [], []
    if args.extend:
        print("Checking existing corpus before extending...", flush=True)
        subprocess.run([sys.executable, str(ROOT / "verify.py")], check=True)
        records = [json.loads(line) for line in (ROOT / "metadata/files.jsonl").read_text().splitlines()]
        omitted = json.loads((ROOT / "metadata/omitted.json").read_text())
        repos = json.loads((ROOT / "metadata/repositories.json").read_text())
    fetch_repositories(clones, {r["repository"] for r in repos})
    if not args.extend:
        local_books = args.books_dir or (PROJECT / "rawdata" if not args.fresh_downloads else None)
        if local_books is not None and local_books.is_dir():
            BOOKS_DIR = local_books.expanduser().resolve()
            if not list(BOOKS_DIR.glob("*.txt")):
                raise ValueError(f"No .txt books in {BOOKS_DIR}")
        elif args.books_dir is not None:
            raise FileNotFoundError(args.books_dir)
        else:
            BOOKS_DIR = ROOT / "sources/books"
            download_books()
        books(records, omitted)
    repos = code(clones, records, omitted, repos)
    write_json(ROOT / "metadata/repositories.json", repos)
    with (ROOT / "metadata/files.jsonl").open("w", encoding="utf-8") as out:
        for record in records:
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
    write_json(ROOT / "metadata/omitted.json", omitted)
    summary = {"created_utc": datetime.now(timezone.utc).isoformat(), "repositories": len(repos), "files": len(records), "bytes": sum(r["bytes"] for r in records), "books": {"files": sum(r["kind"] == "book" for r in records), "bytes": sum(r["bytes"] for r in records if r["kind"] == "book")}, "code": {"files": sum(r["kind"] == "code" for r in records), "bytes": sum(r["bytes"] for r in records if r["kind"] == "code")}, "by_suffix": dict(Counter(Path(r["path"]).suffix for r in records))}
    suffix_bytes = Counter()
    for record in records:
        suffix_bytes[Path(record["path"]).suffix] += record["bytes"]
    summary["bytes_by_suffix"] = dict(suffix_bytes)
    write_json(ROOT / "metadata/summary.json", summary)
    write_json(ROOT / "training_data.json", {"suffixes": [".txt", *sorted(SUFFIXES)], "exclude_dirs": ["metadata", ".git", "__pycache__", "sources", "artifacts"]})
    write_helpers(summary)
    print(json.dumps(summary, indent=2), flush=True)
    subprocess.run([sys.executable, str(ROOT / "verify.py")], check=True)


if __name__ == "__main__":
    main()
