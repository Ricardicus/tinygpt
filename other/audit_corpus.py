#!/usr/bin/env python3
"""Audit character suitability for English training without modifying the corpus.

Accept a prepared dataset folder or its corpus/ folder. Reports percentages per
file, character examples, source totals and whole-file removal estimates. Character
rules are not language identification or evidence of malicious data poisoning.
"""
import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
from functools import lru_cache
import json
import math
import shlex
from pathlib import Path
import sys
import time
import unicodedata

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from corpus import discover_files, load_config, relative_corpus_path

# Common English typography is acceptable under the English policy. ASCII mode
# instead matches a tokenizer restricted to printable ASCII plus tab/CR/LF.
TYPOGRAPHY = set("\u00a0\u2010\u2011\u2012\u2013\u2014\u2015\u2018\u2019\u201a\u201b\u201c\u201d\u201e\u201f\u2026")


@lru_cache(maxsize=65536)
def character_info(char, policy):
    code = ord(char)
    category = unicodedata.category(char)
    letter = category.startswith("L")
    if char in "\t\r\n" or 32 <= code <= 126:
        return True, "", letter
    if 0xDC80 <= code <= 0xDCFF:
        reason = "invalid_utf8_byte"
    elif char == "\ufffd":
        reason = "replacement_character"
    elif category.startswith("C"):
        reason = "control_or_format"
    elif letter and "LATIN" not in unicodedata.name(char, ""):
        reason = "non_latin_letter"
    elif letter:
        reason = "accented_latin_letter"
    elif char in TYPOGRAPHY:
        reason = "common_typography"
    elif category.startswith("M"):
        reason = "combining_mark"
    else:
        reason = "other_symbol_or_punctuation"
    allowed = policy == "english" and reason in ("accented_latin_letter", "common_typography", "combining_mark")
    return allowed, reason, letter


def char_record(char, count):
    code = ord(char)
    invalid = 0xDC80 <= code <= 0xDCFF
    display = f"\\x{code - 0xDC00:02x}" if invalid else char
    if unicodedata.category(char).startswith("C") and not invalid:
        display = ascii(char)[1:-1]
    return {"character": display, "codepoint": f"U+{code:04X}" if not invalid else f"BYTE 0x{code - 0xDC00:02X}",
            "name": unicodedata.name(char, "UNNAMED" if not invalid else "INVALID UTF-8 BYTE"), "count": count}


def percentage(part, total):
    return 100 * part / total if total else 0.0


def parse_thresholds(value):
    try:
        result = sorted(set(float(item) for item in value.split(",")))
    except ValueError:
        raise argparse.ArgumentTypeError("Use comma-separated percentages, e.g. 0,0.1,1,5") from None
    if not result or any(not math.isfinite(item) or not 0 <= item <= 100 for item in result):
        raise argparse.ArgumentTypeError("Thresholds must be finite percentages between 0 and 100")
    return result


def exclusion_config(report, threshold, output=None):
    """Select exclusions from an existing audit without scanning corpus text again."""
    if not math.isfinite(threshold) or not 0 <= threshold <= 100:
        raise ValueError("Exclusion threshold must be between 0 and 100")
    report = Path(report).expanduser().resolve()
    if report.is_file():
        report = report.parent
    summary = json.loads((report / "summary.json").read_text(encoding="utf-8"))
    output = Path(output).expanduser().resolve() if output else report / f"exclusions-gt-{threshold:g}"
    corpus = Path(summary["corpus"]).resolve()
    if output == corpus or output.is_relative_to(corpus):
        raise ValueError("Exclusion reports/config must be outside the corpus folder")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("Exclusion output must be a new or empty directory")
    selected = {}
    print(f"Selecting files above {threshold:g}% from {report / 'files.csv'}...", flush=True)
    last_report = time.monotonic()
    with (report / "files.csv").open(encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        if not {"path", "bytes", "outside_policy_percent"}.issubset(reader.fieldnames or []):
            raise ValueError("Audit CSV is missing required columns")
        for index, row in enumerate(reader, 1):
            percent = float(row["outside_policy_percent"])
            size = int(row["bytes"])
            if not math.isfinite(percent) or not 0 <= percent <= 100 or size < 0:
                raise ValueError("Invalid audit percentage or file size")
            if percent > threshold:
                path = relative_corpus_path(row["path"])
                if path in selected:
                    raise ValueError(f"Repeated file in audit CSV: {path}")
                selected[path] = size
            if time.monotonic() - last_report >= 2:
                print(f"Exclusion selection: {index:,} file records checked, {len(selected):,} selected...", flush=True)
                last_report = time.monotonic()
    # New audits snapshot their selection config. Legacy reports used prepared
    # .txt corpora, so keep a conservative fallback rather than require old paths.
    base = summary.get("corpus_config", {"suffixes": [".txt"], "exclude_dirs": []})
    paths = sorted(set(selected) | {relative_corpus_path(path) for path in base.get("exclude_files", [])})
    output.mkdir(parents=True, exist_ok=True)
    (output / "excluded-files.txt").write_text(
        "# Exact paths relative to the corpus root; edit this list to review exclusions.\n" +
        "".join(path + "\n" for path in paths), encoding="utf-8")
    config = {"suffixes": base["suffixes"], "exclude_dirs": base.get("exclude_dirs", []),
              "exclude_files_from": ["excluded-files.txt"]}
    config_path = output / "training_data.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    lost = sum(selected.values())
    decision = {"audit": str(report), "policy": summary["policy"], "exclude_above_percent": threshold,
                "rule": "outside_policy_percent > exclude_above_percent",
                "newly_excluded_files": len(selected), "excluded_paths_including_prior_rules": len(paths),
                "bytes_removed": lost, "MB_removed": lost / 1e6, "GB_removed": lost / 1e9,
                "corpus_percent_removed": percentage(lost, summary["totals"]["bytes"]),
                "config": str(config_path)}
    (output / "selection.json").write_text(json.dumps(decision, indent=2) + "\n", encoding="utf-8")
    print(f"\nExclusion candidate: {len(selected):,} files above {threshold:g}% | "
          f"{lost / 1e6:.2f} MB / {lost / 1e9:.4f} GB excluded.\n"
          f"Review: {output / 'excluded-files.txt'}\n"
          f"Apply to training with: --data-config {shlex.quote(str(config_path))}\n"
          "Original files and dataset config were not changed.", flush=True)
    return decision


def scan_file(path, relative, policy, examples, progress=None, block_size=65536):
    before = path.stat()
    counts = Counter(bytes=0, characters=0, letters=0, outside_policy=0,
                     non_ascii=0, non_latin_letters=0, outside_policy_bytes=0)
    unusual = Counter()
    bad = Counter()
    reasons = Counter()
    samples = []
    line_offset = offset = 0
    with path.open(encoding="utf-8", errors="surrogateescape", newline="") as source:
        while block := source.read(block_size):
            frequencies = Counter(block)
            block_bad = Counter()
            counts["bytes"] += len(block.encode("utf-8", errors="surrogateescape"))
            counts["characters"] += len(block)
            for char, number in frequencies.items():
                allowed, reason, letter = character_info(char, policy)
                counts["letters"] += number if letter else 0
                counts["non_ascii"] += number if ord(char) > 127 else 0
                counts["non_latin_letters"] += number if reason == "non_latin_letter" else 0
                if ord(char) > 127 or not allowed:
                    unusual[char] += number
                if not allowed:
                    counts["outside_policy"] += number
                    counts["outside_policy_bytes"] += number * len(char.encode("utf-8", errors="surrogateescape"))
                    block_bad[char] += number
                    bad[char] += number
                    reasons[reason] += number
            for char, _ in block_bad.most_common(max(0, examples - len(samples))):
                position = block.find(char)
                example = char_record(char, bad[char])
                example.pop("count")
                samples.append({**example, "line": line_offset + block.count("\n", 0, position) + 1,
                                "character_offset": offset + position,
                                "context": block[max(0, position - 60):position + 61]})
            offset += len(block)
            line_offset += block.count("\n")
            if progress:
                progress(counts["bytes"])
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns) or counts["bytes"] != after.st_size:
        raise ValueError(f"File changed during audit: {path}")
    result = {"path": relative.as_posix(), "source": relative.parts[0] if len(relative.parts) > 1 else "(root)",
              **counts, "outside_policy_percent": percentage(counts["outside_policy"], counts["characters"]),
              "outside_policy_byte_percent": percentage(counts["outside_policy_bytes"], counts["bytes"]),
              "size_mb": counts["bytes"] / 1e6, "size_gb": counts["bytes"] / 1e9,
              "non_ascii_percent": percentage(counts["non_ascii"], counts["characters"]),
              "non_latin_percent_of_letters": percentage(counts["non_latin_letters"], counts["letters"]),
              "reasons": dict(reasons), "examples": samples,
              "top_outside_policy_characters": [char_record(char, number) for char, number in bad.most_common(10)]}
    return result, unusual


def audit(root, output=None, policy="english", thresholds=(0, 0.1, 1, 5, 10), examples=3):
    root = Path(root).expanduser().resolve()
    corpus = root / "corpus" if (root / "corpus").is_dir() else root
    dataset = corpus.parent
    config_path = dataset / "training_data.json"
    config = load_config(config_path) if config_path.is_file() else {"suffixes": [".txt"], "exclude_dirs": []}
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
    output = Path(output).expanduser().resolve() if output else dataset / "audits" / f"{policy}-{stamp}"
    if output == corpus or output.is_relative_to(corpus):
        raise ValueError("Audit reports must be outside the corpus folder")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("Report output must be a new or empty directory")
    print(f"Auditing {corpus}\nPolicy: {policy}; reports: {output}", flush=True)
    files = discover_files(corpus, config, verbose=True)
    output.mkdir(parents=True, exist_ok=True)
    sources = defaultdict(Counter)
    scenarios = {str(limit): {"files_removed": 0, "bytes_removed": 0, "sources": {}} for limit in thresholds}
    all_characters = Counter()
    character_files = Counter()
    total = Counter(files=0, bytes=0, characters=0, letters=0, outside_policy=0,
                    non_ascii=0, non_latin_letters=0, outside_policy_bytes=0, flagged_files=0, flagged_file_bytes=0)
    started = last_report = time.monotonic()
    def progress(current_bytes=0):
        nonlocal last_report
        now = time.monotonic()
        if now - last_report >= 2:
            print(f"Audit: {total['files']:,} / {len(files):,} files | "
                  f"{(total['bytes'] + current_bytes) / 1e9:.3f} GB scanned | "
                  f"{total['flagged_files']:,} files outside policy | {now - started:.0f}s", flush=True)
            last_report = now
    fields = ["path", "source", "bytes", "size_mb", "size_gb", "characters", "outside_policy", "outside_policy_percent",
              "outside_policy_bytes", "outside_policy_byte_percent", "non_ascii", "non_ascii_percent", "letters", "non_latin_letters", "non_latin_percent_of_letters"]
    with (output / "files.csv").open("w", encoding="utf-8", newline="") as csv_output, \
            (output / "flagged-files.jsonl").open("w", encoding="utf-8") as details:
        writer = csv.DictWriter(csv_output, fieldnames=fields)
        writer.writeheader()
        for path in files:
            record, unusual = scan_file(path, path.relative_to(corpus), policy, examples, progress)
            writer.writerow({field: record[field] for field in fields})
            if record["outside_policy"]:
                details.write(json.dumps(record, ensure_ascii=True) + "\n")
            group = sources[record["source"]]
            for target in (total, group):
                target["files"] += 1
                target["flagged_files"] += bool(record["outside_policy"])
                target["flagged_file_bytes"] += record["bytes"] if record["outside_policy"] else 0
                for field in ("bytes", "characters", "letters", "outside_policy", "outside_policy_bytes", "non_ascii", "non_latin_letters"):
                    target[field] += record[field]
            all_characters.update(unusual)
            character_files.update(unusual.keys())
            for threshold in thresholds:
                if record["outside_policy_percent"] > threshold:
                    scenario = scenarios[str(threshold)]
                    scenario["files_removed"] += 1
                    scenario["bytes_removed"] += record["bytes"]
                    source = scenario["sources"].setdefault(record["source"], {"files_removed": 0, "bytes_removed": 0})
                    source["files_removed"] += 1
                    source["bytes_removed"] += record["bytes"]
            progress()
    with (output / "characters.csv").open("w", encoding="utf-8", newline="") as character_output:
        writer = csv.DictWriter(character_output, fieldnames=["character", "codepoint", "name", "count", "files", "allowed", "category"])
        writer.writeheader()
        for char, number in all_characters.most_common():
            allowed, reason, _ = character_info(char, policy)
            writer.writerow({**char_record(char, number), "files": character_files[char], "allowed": allowed, "category": reason})
    for scenario in scenarios.values():
        scenario.update(MB_removed=scenario["bytes_removed"] / 1e6, GB_removed=scenario["bytes_removed"] / 1e9,
                        corpus_percent_removed=percentage(scenario["bytes_removed"], total["bytes"]),
                        bytes_retained=total["bytes"] - scenario["bytes_removed"])
        for name, removed in scenario["sources"].items():
            removed.update(MB_removed=removed["bytes_removed"] / 1e6, GB_removed=removed["bytes_removed"] / 1e9,
                           source_percent_removed=percentage(removed["bytes_removed"], sources[name]["bytes"]))
        whole_source_bytes = sum(sources[name]["bytes"] for name in scenario["sources"])
        scenario["if_entire_affected_sources_removed"] = {
            "sources": sorted(scenario["sources"]), "bytes_removed": whole_source_bytes,
            "GB_removed": whole_source_bytes / 1e9,
            "corpus_percent_removed": percentage(whole_source_bytes, total["bytes"])}
    for counts in [total, *sources.values()]:
        counts.update(size_mb=counts["bytes"] / 1e6, size_gb=counts["bytes"] / 1e9,
                      flagged_file_mb=counts["flagged_file_bytes"] / 1e6,
                      flagged_file_gb=counts["flagged_file_bytes"] / 1e9,
                      outside_policy_percent=percentage(counts["outside_policy"], counts["characters"]),
                      outside_policy_byte_percent=percentage(counts["outside_policy_bytes"], counts["bytes"]))
    summary = {"created_utc": datetime.now(timezone.utc).isoformat(), "corpus": str(corpus), "policy": policy,
               "report_directory": str(output), "corpus_config": config,
               "units": "decimal MB/GB; percentages count characters (including whitespace), not bytes",
               "rule": "Remove whole files whose outside_policy_percent strictly exceeds the threshold",
               "limitations": ["Character rules do not identify the document language or prove data poisoning.",
                               "English prose can legitimately quote foreign names, scripts, equations and symbols.",
                               "Latin-script non-English text can pass the English character policy.",
                               "Malicious or low-quality English content can pass either character policy.",
                               "English policy accepts printable ASCII, tab/CR/LF, Latin letters, combining marks and common typography.",
                               "ASCII policy accepts only printable ASCII and tab/CR/LF."],
               "totals": dict(total), "sources": {name: dict(counts) for name, counts in sorted(sources.items())},
               "removal_scenarios": scenarios, "elapsed_seconds": time.monotonic() - started}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"\nScanned {total['files']:,} files, {total['bytes'] / 1e9:.3f} GB. "
          f"{total['flagged_files']:,} files contain characters outside policy.", flush=True)
    print("\nSource                  Total GB    Flagged GB    Flagged files    Outside-policy chars %")
    for name, counts in sorted(sources.items()):
        print(f"{name:<23} {counts['bytes'] / 1e9:>8.3f} {counts['flagged_file_bytes'] / 1e9:>13.3f} {counts['flagged_files']:>16,} "
              f"{percentage(counts['outside_policy'], counts['characters']):>25.4f}")
    print("\nDrop files exceeding     Files removed       MB lost       GB lost    Corpus % lost")
    for threshold, scenario in scenarios.items():
        print(f"{threshold + '%':<24} {scenario['files_removed']:>13,} {scenario['MB_removed']:>13.2f} "
              f"{scenario['GB_removed']:>13.4f} {scenario['corpus_percent_removed']:>16.2f}")
    print("\nMost frequent outside-policy characters:")
    top_bad = [(char, number) for char, number in all_characters.most_common() if not character_info(char, policy)[0]][:10]
    for char, number in top_bad:
        info = char_record(char, number)
        print(f"  {info['codepoint']:<12} {info['name']:<42} {number:>12,} occurrences in {character_files[char]:,} files")
    print(f"\nReports: {output}\nNo corpus files were changed. Findings indicate character suitability, not proof of poisoning.")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, nargs="?", help="Prepared dataset folder or corpus folder")
    parser.add_argument("--from-report", type=Path, help="Generate exclusions from an existing audit directory or summary.json, without rescanning")
    parser.add_argument("--exclude-above", type=float, metavar="PERCENT", help="Generate a candidate exclusion list/config for files above this outside-policy character percentage")
    parser.add_argument("--output", type=Path, help="New directory outside corpus; default: timestamped audit folder, or exclusions-gt-PERCENT under --from-report")
    parser.add_argument("--policy", choices=("english", "ascii"),
                        help="New audit policy (default: english); english allows accented Latin letters and common typography, ascii is strict ASCII")
    parser.add_argument("--thresholds", type=parse_thresholds, default=parse_thresholds("0,0.1,1,5,10"),
                        help="Estimate removing files exceeding these character percentages (default: 0,0.1,1,5,10)")
    parser.add_argument("--examples", type=int, default=3, help="Context examples per flagged file (default: 3)")
    args = parser.parse_args(argv)
    if args.exclude_above is not None and (not math.isfinite(args.exclude_above) or not 0 <= args.exclude_above <= 100):
        parser.error("--exclude-above must be a finite percentage between 0 and 100")
    if args.from_report and (args.dataset or args.exclude_above is None):
        parser.error("--from-report requires --exclude-above and cannot be combined with a dataset argument")
    if args.from_report and args.policy is not None:
        parser.error("--from-report uses its saved character policy; run a new audit to change --policy")
    if not args.from_report and args.dataset is None:
        parser.error("Provide a dataset folder or --from-report")
    if args.examples < 0:
        parser.error("--examples must be nonnegative")
    try:
        if args.from_report:
            exclusion_config(args.from_report, args.exclude_above, args.output)
        else:
            summary = audit(args.dataset, args.output, args.policy or "english", args.thresholds, args.examples)
            if args.exclude_above is not None:
                exclusion_config(summary["report_directory"], args.exclude_above)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
