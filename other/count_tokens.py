#!/usr/bin/env python3
"""Report the saved pretraining token count, or recount a corpus with its tokenizer."""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bpe import BPE
from corpus import discover_files, encoded_chunks, load_config, split_validation_files


def training_cache(folder):
    candidates = [folder.with_name(folder.name + suffix)
                  for suffix in ("_train_tokens.bin", "_tokens.bin")]
    found = [path for path in candidates if path.is_file()]
    if not found:
        raise ValueError(f"No training token cache for {folder}. Pass --tokenizer and the run's corpus options to recount.")
    if len(found) > 1:
        raise ValueError(f"Both split and unsplit caches exist. Pass the desired file: {found[0]} or {found[1]}")
    return found[0]


def cached_count(cache):
    size = cache.stat().st_size
    if size % 4:
        raise ValueError(f"Incomplete four-byte token ID in {cache}")
    metadata = Path(str(cache) + ".json")
    if metadata.exists():
        manifest = json.loads(metadata.read_text())
        if not isinstance(manifest, dict) or manifest.get("bytes") != size:
            raise ValueError(f"Cache size disagrees with {metadata}. Use --tokenizer on the corpus to recount.")
    else:
        print("No cache metadata. This count assumes the .bin file is complete.", file=sys.stderr)
    return size // 4


def recount(folder, args):
    config_path = args.data_config
    if config_path is None and folder.name == "corpus" and (folder.parent / "training_data.json").is_file():
        config_path = folder.parent / "training_data.json"
    lower_case = args.lower_case != "false"
    tokenizer = BPE(lower_case=lower_case)
    tokenizer.load(args.tokenizer)
    files = discover_files(folder, load_config(config_path), verbose=True)
    if args.validation_fraction:
        seed = args.validation_seed if args.validation_seed is not None else 42
        files, _ = split_validation_files(files, folder, args.validation_fraction, seed)
    print(f"Counting {len(files):,} training files with {args.tokenizer} "
          f"(validation fraction {args.validation_fraction or 0:g}, lower-case {lower_case}).", flush=True)
    total = 0
    last_report = time.monotonic()
    for index, path in enumerate(files, 1):
        for ids in encoded_chunks(path, tokenizer, lower_case):
            total += len(ids)
            now = time.monotonic()
            if now - last_report >= 2:
                print(f"File {index:,}/{len(files):,} | {total:,} tokens", flush=True)
                last_report = now
    return total


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="Prepared dataset directory, corpus directory, or token cache .bin file")
    parser.add_argument("--tokenizer", type=Path, help="Recount current corpus text using this saved tokenizer")
    parser.add_argument("--data-config", type=Path, help="Recount with these corpus selection rules")
    parser.add_argument("--validation-fraction", type=float, help="Fraction of files held out by this run (recount default: 0)")
    parser.add_argument("--validation-seed", type=int, help="File split seed (recount default: 42)")
    parser.add_argument("--lower-case", choices=("true", "false"), help="Lowercase before encoding (recount default: true)")
    parser.add_argument("--context-length", type=int, help="Also show complete windows and target tokens per epoch")
    args = parser.parse_args(argv)
    if args.context_length is not None and args.context_length < 1:
        parser.error("--context-length must be positive")
    if args.validation_fraction is not None and not 0 <= args.validation_fraction < 1:
        parser.error("--validation-fraction must be in [0, 1)")
    if args.tokenizer is None and any(value is not None for value in (
            args.validation_fraction, args.validation_seed, args.data_config, args.lower_case)):
        parser.error("Corpus selection options require --tokenizer for a fresh recount")

    path = args.path.expanduser().absolute()
    try:
        folder = path / "corpus" if (path / "corpus").is_dir() else path
        if args.tokenizer is not None:
            if not folder.is_dir():
                parser.error("Pass the corpus or dataset directory when recounting with --tokenizer")
            total = recount(folder, args)
        else:
            cache = training_cache(folder) if folder.is_dir() else path
            if not cache.is_file() or cache.suffix != ".bin":
                parser.error("Pass a dataset or corpus directory, or an existing token cache .bin file")
            total = cached_count(cache)
            print(f"Saved cache: {cache}")
    except (OSError, ValueError) as error:
        parser.error(str(error))

    print(f"Total tokens: {total:,}")
    if args.context_length is not None:
        windows = total // (args.context_length + 1)
        print(f"Complete windows: {windows:,}")
        print(f"Target tokens per epoch: {windows * args.context_length:,}")


if __name__ == "__main__":
    main()
