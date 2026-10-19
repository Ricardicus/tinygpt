import os
import argparse

def parse_extensions(ext_string):
    return {ext.strip() for ext in ext_string.split(",") if ext.strip()}

def collect_files(root_dir, extensions):
    matched_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            _, ext = os.path.splitext(filename)
            if ext in extensions:
                matched_files.append(os.path.join(dirpath, filename))
    return matched_files

def merge_files(file_list, output_file):
    with open(output_file, "w", encoding="utf-8") as out:
        for filepath in file_list:
            try:
                with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                    out.write(f.read())
            except Exception as e:
                print(f"Skipping {filepath}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Merge files by extension")
    parser.add_argument(
        "--input",
        required=True,
        help='Comma-separated file extensions (e.g. ".cpp,.h,.hpp")'
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output file (e.g. collection.txt)"
    )
    parser.add_argument(
        "--root",
        default=".",
        help="Root directory to start from (default: current directory)"
    )

    args = parser.parse_args()

    extensions = parse_extensions(args.input)
    files = collect_files(args.root, extensions)

    print(f"Found {len(files)} matching files.")
    merge_files(files, args.output)
    print(f"Merged into {args.output}")

if __name__ == "__main__":
    main()
