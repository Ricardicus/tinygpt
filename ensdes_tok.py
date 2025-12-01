"""
Just a script to test a tokenizer
"""
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import os
import sys
from itertools import islice
import time
from bpe import BPE

def getData(verbose=True, lower_case=True, rawdata_folder="rawdata", oneliners=[]):
    if verbose:
        print("Initializing data reader...")
    datareader = TextCorpusReader(
        rawdata_folder, lower_case=lower_case, to_new_lines=oneliners
    )
    if verbose:
        print(f"DataReader ready: {len(datareader)} entries")
    return datareader


def getBPE(
    tokenizer_path,
    vocab_size=1920,
    verbose=True,
    bpe_part=0.01,
    lower_case=True,
):
    """Load existing tokenizer or train a new one."""
    if tokenizer_path and os.path.exists(tokenizer_path):
        bpe = BPE(
            vocab_size=vocab_size,
            verbose=verbose,

            model_file=tokenizer_path,
            lower_case=lower_case,
        )
        bpe.load()
        if verbose:
            print(f"Using existing tokenizer from '{tokenizer_path}'")
        return bpe

    raise ValueError("Invalid tokenizer path")

# ---------------------------------------------------------
# Argument parsing
# ------------------------------------self.chunked_file_readers.append(ChunkedFile(file, self.buffer_size))---------------------
def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Use a tokenizer to encode something."
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        required=True,
        help=f"Path to tokenizer file to load. Required."
    )
    parser.add_argument(
        "--encode",
        type=str,
        default=None,
        help="Text to encode."
    )
    parser.add_argument(
        "--decode",
        type=str,
        default=None,
        help="Tokens to decode. Use a comma-separated numbers e.g.: --input-decode 32,421,23",
    )
    return parser.parse_args(argv)

def main(argv=None):
    args = parse_args(argv)

    tokenizer = args.tokenizer
    encode = args.encode
    decode = args.decode

    bpe = getBPE(tokenizer)

    print(f"Loaded BPE: {tokenizer}")
    print(f"  vocab size:       {bpe.vocab_size}")
    print(f"  end-of-word-mark: {bpe.get_end_of_word_mark()}")

    if encode is not None:
        result = bpe.encode(encode)
        print(f"{encode} -> {result}")

    if decode is not None:
        inputs = [ int(x) for x in decode.split(",") ]
        result = bpe.decode(inputs)
        print(f"{inputs} -> {result}")

# ---------------------------------------------------------
if __name__ == "__main__":
    main()
