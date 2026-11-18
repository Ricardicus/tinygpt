#!/usr/bin/env python3
"""Remove Gutenberg headers and footers from text files in place."""
import argparse
import os
from pathlib import Path
import re
import tempfile


START = re.compile(rb'^(?:\xef\xbb\xbf)?[ \t]*\*\*\*[ \t]*START OF (?:THE|THIS) PROJECT GUTENBERG[^\r\n]*\*\*[ \t]*(?:\r?\n|$)', re.IGNORECASE | re.MULTILINE)
END = re.compile(rb'^[ \t]*\*\*\*[ \t]*END OF (?:THE|THIS) PROJECT GUTENBERG[^\r\n]*\*\*\*[^\r\n]*(?:\r?\n|$)', re.IGNORECASE | re.MULTILINE)


def trim_text(data):
    start = START.search(data)
    if not start:
        return None
    end = END.search(data, start.end())
    if not end:
        return None
    return re.sub(rb'\A(?:[ \t]*(?:\r\n|\n|\r))+', b'', data[start.end():end.start()])


def replace_file(path, data):
    descriptor, temporary = tempfile.mkstemp(prefix='.gutenberg-trim-', dir=path.parent)
    try:
        with os.fdopen(descriptor, 'wb') as output:
            output.write(data)
        os.chmod(temporary, path.stat().st_mode & 0o777)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gutenberg-folder', type=Path, required=True, help='Directory to scan recursively')
    parser.add_argument('--dry-run', action='store_true', help='Report changes without writing files')
    args = parser.parse_args(argv)
    if not args.gutenberg_folder.is_dir():
        parser.error('--gutenberg-folder must be an existing directory')
    trimmed = skipped = errors = 0
    for path in sorted(args.gutenberg_folder.rglob('*.txt')):
        if path.is_symlink() or not path.is_file():
            continue
        try:
            original = path.read_bytes()
            result = trim_text(original)
            if result is None:
                skipped += 1
                print(f'Skipped (missing ordered markers): {path}')
                continue
            if not result.strip():
                skipped += 1
                print(f'Skipped (empty content): {path}')
                continue
            if not args.dry_run:
                replace_file(path, result)
            trimmed += 1
            print(f'{"Would trim" if args.dry_run else "Trimmed"}: {path} ({len(original):,} → {len(result):,} bytes)')
        except OSError as error:
            errors += 1
            print(f'Error: {path}: {error}')
    print(f'{"Would trim" if args.dry_run else "Trimmed"} {trimmed} files; skipped {skipped}; errors {errors}.')
    return 1 if errors else 0


if __name__ == '__main__':
    raise SystemExit(main())
