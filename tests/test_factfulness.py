from pathlib import Path
from contextlib import redirect_stdout
import io
import subprocess
import sys
import tempfile
import os
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bpe import BPE, getBPE
from main import generate, main, parse_args, save_checkpoint
from tinygpt import TinyGPT

import argparse
from pathlib import Path

def str2bool(v):
    if isinstance(v, bool):
        return v
    v = v.strip().lower()
    if v in ("yes", "true", "t", "1", "y"):
        return True
    if v in ("no", "false", "f", "0", "n"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected for --verbose")

def default_device():
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def parseArguments():
    def existing_file(path_str: str) -> Path:
        path = Path(path_str)

        if not path.exists():
            raise argparse.ArgumentTypeError(
                f"File does not exist: {path}"
            )

        if not path.is_file():
            raise argparse.ArgumentTypeError(
                f"Expected a file but got: {path}"
            )

        return path

    parser = argparse.ArgumentParser(
        description="Load model and tokenizer files"
    )

    parser.add_argument(
        "--verbose",
        type=str2bool,
        default=True,
        help="Enable verbose output (default: true).",
    )
    parser.add_argument(
        "--lower-case",
        type=str2bool,
        default=True,
        help="Only handle lower-case (default: true).",
    )
    parser.add_argument(
        "--bpe-part",
        type=float,
        default=default_bpe_part,
        help="Fraction of dataset used to train BPE.",
    )

    parser.add_argument(
        "--tokenizer",
        required=True,
        type=existing_file,
        help="Path to tokenizer file",
    )

    parser.add_argument(
        "--model",
        required=True,
        type=existing_file,
        help="Path to model file",
    )

    parser.add_argument(
        "--device",
        type=str,
        default=default_device(),
        help="Compute device.",
    )

    return parser.parse_args()

def prompt_response(prompt, model, bpe):
    result = generate(model, bpe, prompt, 30, args.device,
                      temperature=0.01, top_k=0)
    if not bpe.preserve_whitespace:
        result = result.replace(bpe.get_end_of_word_mark(), " ")
    return result

# ---------------------------------------------------------
# Default parameters
# ---------------------------------------------------------
default_tokenizer = "tokenizer.json"
default_bpe_part = 0.01
default_model_n_heads = 5
default_model_d_dim = 240
default_vocab_size = 1920
default_model_context_length = 64

# Actual testing data
# Country - Capital
tests = [
    {
        "name": "Country -> Capitals",
        "basePrompt": "The capital of ",
        "variations": [
            "Norway",
            "Sweden",
            "Denmark",
            "Finland",
            "Germany",
            "France",
            "Spain",
            "Italy",
            "Portugal",
            "the Netherlands",
            "Belgium",
            "Austria",
            "Poland",
            "the Czech Republic",
            "Hungary",
            "Ireland",
            "the United Kingdom",
            "Greece",
            "Canada",
            "Japan",
            "Australia"
        ],
        "responses": [
            "Oslo",
            "Stockholm",
            "Copenhagen",
            "Helsinki",
            "Berlin",
            "Paris",
            "Madrid",
            "Rome",
            "Lisbon",
            "Amsterdam",
            "Brussels",
            "Vienna",
            "Warsaw",
            "Prague",
            "Budapest",
            "Dublin",
            "London",
            "Athens",
            "Ottawa",
            "Tokyo",
            "Canberra"
        ]
    },
    {
        "name": "Capital -> Countries",
        "basePrompt": "",
        "variations": [
            "Oslo is the capital of the country ",
            "Stockholm is the capital of the country ",
            "Copenhagen is the capital of the country ",
            "Helsinki is the capital of the country",
            "Berlin is the capital of the country",
            "Paris is the capital of the country",
            "Madrid is the capital of the country",
            "Rome is the capital of the country",
            "Lisbon is the capital of the country",
            "Amsterdam is the capital of the country"
        ],
        "responses": [
            "Norway",
            "Sweden",
            "Denmark",
            "Finland",
            "Germany",
            "France",
            "Spain",
            "Italy",
            "Portugal",
            "Netherlands"
        ]
    },
    {
        "name": "Opposites",
        "basePrompt": "The opposite of ",
        "variations": [
            "hot, is",
            "up, is",
            "big, is",
            "happy, is",
            "fast, is",
            "light, is",
            "young, is",
            "rich, is"
        ],
        "responses": [
            "cold",
            "down",
            "small",
            "sad",
            "slow",
            "dark",
            "old",
            "poor"
        ]
    }
]


def doTests(model, bpe):
    total = 0
    successes = 0
    fails = 0

    category_stats = []

    for test in tests:
        test_successes = 0
        test_total = len(test["variations"])

        for variation, expected in zip(
            test["variations"],
            test["responses"]
        ):
            total += 1

            prompt = test["basePrompt"] + variation
            response = prompt_response(prompt, model, bpe)
            response = response.split(".")[0] + "."

            success = expected.lower() in response.lower()

            if success:
                successes += 1
                test_successes += 1
            else:
                fails += 1

                print("FAIL")
                print(f"  Prompt   : {prompt}")
                print(f"  Expected : {expected}")
                print(f"  Actual   : {response}")
                print()

        category_stats.append(
            (
                test["name"],
                test_successes,
                test_total
            )
        )

    print()
    print("=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    for name, passed, count in category_stats:
        pct = 100.0 * passed / count
        print(
            f"{name:20} : "
            f"{passed:3}/{count:<3} "
            f"({pct:6.2f}%)"
        )

    print("-" * 60)

    accuracy = (
        100.0 * successes / total
        if total > 0 else 0.0
    )

    print(f"Total Tests : {total}")
    print(f"Passed      : {successes}")
    print(f"Failed      : {fails}")
    print(f"Accuracy    : {accuracy:.2f}%")

    return {
        "total": total,
        "passed": successes,
        "failed": fails,
        "accuracy": accuracy
    }


if __name__ == "__main__":
    args = parseArguments()

    print(f"Loading model from '{args.model}' for inference...")
    checkpoint = torch.load(args.model, map_location=args.device)

    cfg = checkpoint.get("config") if isinstance(checkpoint, dict) else None
    if cfg is not None:
        # Use hyperparameters from checkpoint config
        if args.verbose:
            print("Using model config from checkpoint:")
            print(cfg)

        # Tokenizer: still driven by CLI for now, but vocab_size from cfg
        bpe = getBPE(
            [],
            vocab_size=cfg["vocab_size"],
            verbose=args.verbose,
            tokenizer_path=args.tokenizer,
            bpe_part=args.bpe_part,
            lower_case=args.lower_case,
        )

        # Build model from saved config
        model = TinyGPT(
            vocab_size=cfg["vocab_size"],
            context_length=cfg["context_length"],
            d_model=cfg["d_model"],
            n_heads=cfg["n_heads"],
            num_layers=cfg["num_layers"],
        ).to(args.device)

        model.load_state_dict(checkpoint["model_state_dict"])

        doTests(model, bpe)
