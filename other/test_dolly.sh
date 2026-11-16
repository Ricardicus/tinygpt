#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON:-$repo_dir/venv/bin/python}"
instruction="${1:-}"
if [[ -z "$instruction" ]]; then
  read -r -p 'Instruction: ' instruction
fi
if [[ -z "$instruction" ]]; then
  echo 'Instruction must not be empty.' >&2
  exit 1
fi
if (( $# > 0 )); then shift; fi
prompt=$(printf '### Instruction\n%s\n\n### Response\n' "$instruction")
prompt+=$'\n'
exec "$python_bin" "$repo_dir/other/inference.py" \
  --model "$repo_dir/dolly15k/artifacts/dolly-sft.pt" \
  --tokenizer "$repo_dir/artifacts-2026-11-02/tokenizer.json" \
  --prompt "$prompt" "$@"
