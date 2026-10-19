#!/usr/bin/env python3
"""Generate text from a saved TinyGPT checkpoint and its matching tokenizer."""
import argparse
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bpe import BPE
from main import generate
from tinygpt import TinyGPT


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="Saved model checkpoint.")
    parser.add_argument("--tokenizer", type=Path,
                        help="Matching tokenizer JSON (default: tokenizer.json beside the model).")
    parser.add_argument("--prompt", help="Input prompt; prompts interactively when omitted.")
    parser.add_argument("--max-new-tokens", type=int, default=100,
                        help="Number of tokens to generate (default: 100).")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, help="Random seed for repeatable sampling.")
    args = parser.parse_args(argv)
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be positive")
    model_path = args.model.expanduser()
    tokenizer_path = (args.tokenizer or model_path.with_name("tokenizer.json")).expanduser()
    for path in (model_path, tokenizer_path):
        if not path.is_file():
            parser.error(f"File not found: {path}")
    prompt = args.prompt
    if prompt is None:
        try:
            prompt = input("Prompt: ")
        except EOFError:
            parser.error("Pass --prompt TEXT when running without interactive input")
    if not prompt:
        parser.error("Prompt must not be empty")

    checkpoint = torch.load(model_path, map_location=args.device, weights_only=True)
    if not isinstance(checkpoint, dict) or not checkpoint.get("config") or "model_state_dict" not in checkpoint:
        parser.error("Expected a TinyGPT checkpoint with saved config and model_state_dict")
    config = checkpoint["config"]
    tokenizer = BPE()
    tokenizer.load(str(tokenizer_path))
    if (not tokenizer.vocab or tokenizer.vocab_size != config["vocab_size"]
            or min(tokenizer.vocab.values()) < 0
            or max(tokenizer.vocab.values()) >= config["vocab_size"]):
        parser.error("Tokenizer vocabulary does not match the model; use the tokenizer used during training")
    try:
        if not tokenizer.encode(prompt):
            parser.error("Prompt contains no encodable tokens")
    except ValueError as error:
        parser.error(str(error))
    model = TinyGPT(**{name: config[name] for name in
                       ("vocab_size", "context_length", "d_model", "n_heads", "num_layers")}).to(args.device)
    model.load_state_dict(checkpoint["model_state_dict"])
    if args.seed is not None:
        torch.manual_seed(args.seed)
    result = generate(model, tokenizer, prompt, args.max_new_tokens, args.device)
    if not tokenizer.preserve_whitespace:
        result = result.replace(tokenizer.get_end_of_word_mark(), " ")
    print(result)


if __name__ == "__main__":
    main()
