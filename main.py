import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import csv
import os
import sys
from itertools import islice
import time
import tempfile
import hashlib
import math
import random
from contextlib import nullcontext
from pathlib import Path

from datareader import PagedDataReader, TextCorpusReader, TokenizedDataReader
from bpe import BPE, getBPE
from corpus import load_config, text_chunks
from tinygpt import TinyGPT
from lr_schedule import resolve_schedule


# ---------------------------------------------------------
# Default parameters
# ---------------------------------------------------------
default_tokenizer = "tokenizer.json"
default_bpe_part = 0.01
default_model_n_heads = 5
default_model_d_dim = 240
default_vocab_size = 1920
default_model_context_length = 64


# ---------------------------------------------------------
# Helper functions
# ---------------------------------------------------------
def str2bool(v):
    if isinstance(v, bool):
        return v
    v = v.strip().lower()
    if v in ("yes", "true", "t", "1", "y"):
        return True
    if v in ("no", "false", "f", "0", "n"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected for --verbose")


def split_validation_files(files, folder, fraction, seed):
    """Deterministically hold out whole files without changing the global RNG."""
    if len(files) < 2:
        raise ValueError("Validation requires at least two corpus files; use --validation-fraction 0 to disable it.")
    root = Path(folder).resolve()
    ranked = sorted(files, key=lambda p: hashlib.sha256(
        f"{seed}:{Path(p).resolve().relative_to(root).as_posix()}".encode()
    ).digest())
    count = min(len(files) - 1, max(1, round(len(files) * fraction)))
    held_out = set(ranked[:count])
    return ([p for p in files if p not in held_out],
            [p for p in files if p in held_out])


def training_dtype(device, precision):
    """Keep checkpoint weights FP32; use BF16 operations on capable CUDA GPUs."""
    if precision == "float32":
        return None
    cuda_device = torch.device(device)
    supported = False
    if cuda_device.type == "cuda" and torch.cuda.is_available():
        with torch.cuda.device(cuda_device):
            supported = torch.cuda.is_bf16_supported(including_emulation=False)
    if precision == "bfloat16" and not supported:
        raise ValueError("--precision bfloat16 requires a CUDA GPU with native BF16 support; use auto or float32")
    return torch.bfloat16 if supported else None


def precision_context(device, dtype):
    return torch.autocast(device_type=torch.device(device).type, dtype=dtype) if dtype is not None else nullcontext()


def write_layer_norms(path, model, epoch, global_step):
    """Record block norms and selected individual tensors before clipping/update."""
    def norm(tensors):
        return torch.stack([t.detach().float().norm() for t in tensors]).norm().item()

    exists = os.path.exists(path) and os.path.getsize(path) > 0
    with open(path, "a", newline="") as output:
        writer = csv.writer(output)
        if not exists:
            writer.writerow(["epoch", "global_step", "layer", "parameter_norm", "gradient_norm"])
        for index, layer in enumerate(model.layers):
            parameters = list(layer.parameters())
            gradients = [p.grad for p in parameters if p.grad is not None]
            writer.writerow([epoch, global_step, index, norm(parameters),
                             norm(gradients) if gradients else ""])

    detail_path = path + ".parameters.csv"
    exists = os.path.exists(detail_path) and os.path.getsize(detail_path) > 0
    with open(detail_path, "a", newline="") as output:
        writer = csv.writer(output)
        if not exists:
            writer.writerow(["epoch", "global_step", "parameter", "parameter_norm", "gradient_norm"])
        for name, parameter in model.named_parameters():
            if (name in ("E", "pos_E") or name.startswith("final_ln.")
                    or name.startswith(("layers.0.", "layers.1.", "layers.2."))):
                writer.writerow([epoch, global_step, name, norm([parameter]),
                                 norm([parameter.grad]) if parameter.grad is not None else ""])


@torch.inference_mode()
def evaluate(model, datareader, batch_size, max_batches, device, seed=42, dtype=None):
    """Token-weighted loss over a fixed sample of held-out windows."""
    count = min(len(datareader), batch_size * max_batches)
    if not count:
        raise ValueError("Validation corpus has no complete context windows; use more files or a shorter context length.")
    indices = random.Random(seed).sample(range(len(datareader)), count)
    was_training = model.training
    total_loss = total_tokens = 0
    model.eval()
    try:
        for start in range(0, count, batch_size):
            windows = torch.tensor([datareader[i] for i in indices[start:start + batch_size]],
                                   dtype=torch.long, device=device)
            with precision_context(device, dtype):
                logits = model(windows[:, :-1])
                targets = windows[:, 1:]
                loss = nn.functional.cross_entropy(
                    logits.reshape(-1, logits.size(-1)), targets.reshape(-1), reduction="sum"
                )
            total_loss += loss.item()
            total_tokens += targets.numel()
    finally:
        model.train(was_training)
    loss = total_loss / total_tokens
    return loss, math.exp(loss) if loss < 700 else float("inf"), total_tokens


def save_checkpoint(path, model, optimizer, epoch, loss, global_step, step=0, lr_schedule=None):
    """Atomically replace the latest checkpoint; epoch counts completed epochs.

    A mid-epoch checkpoint resumes weights/optimizer and restarts that epoch
    with a new shuffle. It does not promise an exact data-position resume.
    """
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "step": step,
        "global_step": global_step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
        "lr_schedule": lr_schedule.state_dict() if lr_schedule is not None else None,
        "config": {
            "vocab_size": model.vocab_size,
            "context_length": model.context_length,
            "d_model": model.d_model,
            "n_heads": model.n_heads,
            "num_layers": model.num_layers,
        },
    }
    descriptor, temporary = tempfile.mkstemp(prefix="checkpoint-", suffix=".tmp", dir=directory)
    os.close(descriptor)
    try:
        torch.save(checkpoint, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


PROGRESS_COLUMNS = ["epoch", "step", "loss", "mean_loss", "validation_loss", "lr", "grad_norm"]


def prepare_progress_log(path):
    """Add missing metric columns to legacy logs, leaving old values unknown."""
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        with path.open("w", newline="") as output:
            csv.writer(output).writerow(PROGRESS_COLUMNS)
        return
    with path.open(newline="") as source:
        reader = csv.reader(source)
        header = next(reader)
        if header == PROGRESS_COLUMNS:
            return
        if header not in (PROGRESS_COLUMNS[:4], PROGRESS_COLUMNS[:5], PROGRESS_COLUMNS[:6]):
            raise ValueError(f"Unexpected progress CSV columns in {path}: {header}")
        descriptor, temporary = tempfile.mkstemp(prefix="progress-", suffix=".tmp", dir=path.parent)
        try:
            with os.fdopen(descriptor, "w", newline="") as output:
                writer = csv.writer(output)
                writer.writerow(PROGRESS_COLUMNS)
                for row in reader:
                    if len(row) != len(header):
                        raise ValueError(f"Malformed progress CSV row in {path}: {row}")
                    writer.writerow([*row, *([""] * (len(PROGRESS_COLUMNS) - len(header)))])
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)


def write_progress_row(path, row, position=None):
    """Append a batch, or refresh its final validation value after epoch end."""
    with open(path, "r+", newline="") as output:
        if position is None:
            output.seek(0, os.SEEK_END)
        else:
            output.seek(position)
        position = output.tell()
        csv.writer(output).writerow(row)
        output.truncate()
    return position


def getData(verbose=True, lower_case=True, rawdata_folder="rawdata", oneliners=[], corpus_config=None):
    if verbose:
        print("Initializing data reader...", flush=True)
    datareader = TextCorpusReader(
        rawdata_folder, lower_case=lower_case, to_new_lines=oneliners,
        corpus_config=corpus_config, verbose=verbose
    )
    if verbose:
        print(f"DataReader ready: {len(datareader.files):,} corpus files", flush=True)
    return datareader



# ---------------------------------------------------------
# Training pipeline (streaming, token-packed)
# ---------------------------------------------------------
def stream_batches(datareader, bpe, context_length, batch_size):
    """
    Generator that yields batches of (x, y) token tensors.

    If `datareader` is a TokenizedDataReader it already yields
    pre-tokenized windows of length context_length + 1.
    Otherwise falls back to on-the-fly encoding.
    """
    batch_x, batch_y = [], []

    def flush():
        nonlocal batch_x, batch_y
        if batch_x:
            x_t = torch.tensor(batch_x, dtype=torch.long)
            y_t = torch.tensor(batch_y, dtype=torch.long)
            batch_x, batch_y = [], []
            return x_t, y_t
        return None

    if isinstance(datareader, TokenizedDataReader):
        # Fast path: windows are already tokenised
        for window in datareader:
            batch_x.append(window[:-1])
            batch_y.append(window[1:])
            if len(batch_x) >= batch_size:
                out = flush()
                if out is not None:
                    yield out
        out = flush()
        if out is not None:
            yield out
        return

    # Legacy path: read raw text, tokenise on the fly
    token_buffer = []
    start_idx = 0

    for text in datareader:
        try:
            ids = bpe.encode(text)
        except ValueError:
            continue
        if not ids:
            continue

        token_buffer.extend(ids)

        while len(token_buffer) - start_idx >= context_length + 1:
            window = token_buffer[start_idx : start_idx + context_length + 1]
            batch_x.append(window[:-1])
            batch_y.append(window[1:])
            start_idx += context_length

            if len(batch_x) >= batch_size:
                out = flush()
                if out is not None:
                    yield out

        if start_idx > 10 * context_length:
            token_buffer = token_buffer[start_idx:]
            start_idx = 0

    while len(token_buffer) - start_idx >= context_length + 1:
        window = token_buffer[start_idx : start_idx + context_length + 1]
        batch_x.append(window[:-1])
        batch_y.append(window[1:])
        start_idx += context_length
        if len(batch_x) >= batch_size:
            out = flush()
            if out is not None:
                yield out

    out = flush()
    if out is not None:
        yield out


@torch.no_grad()
def generate(model, bpe, prompt, max_new_tokens=50, device="cpu", temperature=1.0, top_k=0):
    """Generate text autoregressively from a prompt."""
    if not math.isfinite(temperature) or temperature < 0 or top_k < 0:
        raise ValueError("Temperature must be finite and nonnegative; top_k must be nonnegative")
    model.eval()

    # Encode prompt → token IDs
    ids = bpe.encode(prompt)
    ids = torch.tensor([ids], dtype=torch.long).to(device)  # (1, C)

    for _ in range(max_new_tokens):
        # Trim to context length if too long
        x = ids[:, -model.context_length :]
        logits = model(x)  # (1, C, d_model)
        next_token_logits = logits[:, -1, :]  # last position
        if temperature == 0:
            next_id = next_token_logits.argmax(dim=-1, keepdim=True)
        else:
            next_token_logits = next_token_logits / temperature
            if top_k:
                cutoff = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1))).values[:, -1:]
                next_token_logits = next_token_logits.masked_fill(next_token_logits < cutoff, float("-inf"))
            probs = torch.softmax(next_token_logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
        ids = torch.cat([ids, next_id], dim=1)

    # Decode all tokens back to text
    generated = bpe.decode(ids[0].tolist())
    return generated


# ---------------------------------------------------------
# Argument parsing
# ------------------------------------self.chunked_file_readers.append(ChunkedFile(file, self.buffer_size))---------------------
def default_device():
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Train TinyGPT using a BPE tokenizer with streaming data."
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=default_tokenizer,
        help=f"Path to tokenizer file (will load or create). Default: {default_tokenizer}.",
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=1920,
        help="Requested BPE vocabulary size (default: 1920); oversized alphabets are limited by default.",
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
        "--epochs", type=int, default=3, help="Training epochs for TinyGPT."
    )
    parser.add_argument(
        "--context-length",
        type=int,
        default=default_model_context_length,
        help="Context length (sequence length). Default: {default_model_context_length}.",
    )
    parser.add_argument(
        "--d-model",
        type=int,
        default=default_model_d_dim,
        help=f"Embedding size (model width). Default {default_model_d_dim}",
    )
    parser.add_argument(
        "--n-heads",
        type=int,
        default=default_model_n_heads,
        help=f"Number of heads in tranformer multi head self attention block. Default {default_model_n_heads}",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=6,
        help="Number of transformer layers. Default 6.",
    )
    parser.add_argument(
        "--lr", type=float, default=argparse.SUPPRESS,
        help="Learning rate (cosine peak): 0.0003 for new runs; omitted on resume preserves the saved rate/schedule, explicitly supplied overrides it."
    )
    parser.add_argument("--lr-schedule", choices=("constant", "cosine"), default=None,
                        help="Default: constant for new runs, saved schedule on resume. Cosine enables linear warmup then cosine decay.")
    parser.add_argument("--warmup-steps", type=int, default=None,
                        help="Cosine warmup updates; 0 disables warmup. Default: min(1000, total schedule updates // 10).")
    parser.add_argument("--lr-decay-steps", type=int, default=None,
                        help="Total schedule updates INCLUDING warmup; default: planned remaining training updates. Saved on resume.")
    parser.add_argument("--min-lr-ratio", type=float, default=None,
                        help="Cosine floor as a fraction of peak --lr (default: 0.1).")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size (approximate, streaming). Default 16.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/model.pt",
        help="Where to store the model data.",
    )
    parser.add_argument("--data-config", default=None,
                        help="JSON suffix/excluded-directory config (default: training_data.json beside main.py).")
    parser.add_argument("--bpe-max-words", type=int, default=100_000,
                        help="Maximum unique words retained for BPE merges (default: 100000).")
    parser.add_argument(
        "--rawdata",
        type=str,
        default="rawdata",
        help="Folder where raw text corpus data resides.",
    )
    parser.add_argument(
        "--oneliners",
        type=str,
        default="",
        help="Files that are best read one at a time as text corpus",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=default_device(),
        help="Compute device.",
    )
    parser.add_argument("--precision", choices=("auto", "float32", "bfloat16"), default="auto",
                        help="Training precision: auto uses native CUDA BF16 when supported, otherwise FP32. Checkpoint weights stay FP32.")
    parser.add_argument("--bpe-alphabet-policy", choices=("limit", "expand", "error"), default="limit",
                        help="New tokenizer behavior when characters exceed vocabulary capacity: limit rare characters to <unk> (default), expand vocabulary, or error.")
    parser.add_argument("--bpe-max-chars", type=int, default=0,
                        help="Optional base alphabet token cap, including reserved symbols and <unk>, for policy limit; 0 caps to half the vocabulary only when it overflows.")
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Text prompt to generate from. If set together with --model, training is skipped.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=50,
        help="Maximum number of tokens to generate during inference.",
    )
    parser.add_argument(
        "--log-progress",
        type=str,
        default=None,
        help="Path to CSV file to append training progress (iteration, loss, mean_loss).",
    )
    parser.add_argument(
        "--monitor-layer-norms", metavar="CSV", default=None,
        help="Append block norms and selected per-parameter norms to CSV files during training.",
    )
    parser.add_argument(
        "--monitor-layer-norms-every", type=int, default=1, metavar="N",
        help="Record layer norms every N optimizer steps (default: 1).",
    )
    parser.add_argument(
        "--save-every-steps", type=int, default=1000,
        help="Save the latest model every N optimizer steps (default: 1000; 0 disables interval saves). Epoch-end saves always run.",
    )
    parser.add_argument("--validation-fraction", type=float, default=0,
                        help="Fraction of whole files held out from training (default: 0, disabled).")
    parser.add_argument("--validation-every-steps", type=int, default=1000,
                        help="Evaluate every N optimizer steps; 0 evaluates only at start and epoch end.")
    parser.add_argument("--validation-batches", type=int, default=100,
                        help="Maximum batches per evaluation, using the same sampled windows (default: 100).")
    parser.add_argument("--validation-seed", type=int, default=42,
                        help="Seed for the file split and validation window sample (default: 42).")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling temperature; lower is less random, 0 is greedy (default: 1).")
    parser.add_argument("--top-k", type=int, default=0,
                        help="Sample from the top K tokens; 0 uses the full vocabulary (default: 0).")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for repeatable inference.")
    args = parser.parse_args(argv)
    if args.monitor_layer_norms_every < 1:
        parser.error("--monitor-layer-norms-every must be positive")
    args.lr_explicit = hasattr(args, "lr")
    if not args.lr_explicit:
        args.lr = 3e-4
    if not math.isfinite(args.lr) or args.lr < 0:
        parser.error("--lr must be finite and nonnegative")
    if args.warmup_steps is not None and args.warmup_steps < 0:
        parser.error("--warmup-steps must be nonnegative")
    if args.lr_decay_steps is not None and args.lr_decay_steps < 1:
        parser.error("--lr-decay-steps must be positive")
    if args.min_lr_ratio is not None and (not math.isfinite(args.min_lr_ratio) or not 0 <= args.min_lr_ratio <= 1):
        parser.error("--min-lr-ratio must be between 0 and 1")
    if args.warmup_steps is not None and args.lr_decay_steps is not None and args.warmup_steps >= args.lr_decay_steps:
        parser.error("--warmup-steps must be smaller than --lr-decay-steps")
    if args.lr_schedule == "constant" and any(value is not None for value in
                                              (args.warmup_steps, args.lr_decay_steps, args.min_lr_ratio)):
        parser.error("Schedule options require --lr-schedule cosine")
    if args.batch_size < 1 or args.epochs < 1:
        parser.error("--batch-size and --epochs must be positive")
    if args.bpe_max_chars < 0 or (args.bpe_max_chars and args.bpe_alphabet_policy != "limit"):
        parser.error("--bpe-max-chars must be nonnegative and requires --bpe-alphabet-policy limit")
    if args.bpe_max_chars >= args.vocab_size:
        parser.error("--bpe-max-chars must be smaller than --vocab_size")
    if args.bpe_max_chars and args.bpe_max_chars < 13:
        parser.error("--bpe-max-chars must be at least 13 for reserved symbols and <unk>")
    if not math.isfinite(args.temperature) or args.temperature < 0 or args.top_k < 0:
        parser.error("--temperature must be finite and nonnegative; --top-k must be nonnegative")
    if args.max_new_tokens < 0:
        parser.error("--max-new-tokens must be nonnegative")
    if args.save_every_steps < 0:
        parser.error("--save-every-steps must be nonnegative")
    if not 0 <= args.validation_fraction < 1:
        parser.error("--validation-fraction must be at least 0 and less than 1")
    if args.validation_every_steps < 0 or args.validation_batches < 1:
        parser.error("Validation interval must be nonnegative and validation batches must be positive")
    return args

def main(argv=None):
    args = parse_args(argv)

    # ---------------------------------------------------------
    # --- Inference-only mode (skip training) ---
    # ---------------------------------------------------------
    if args.prompt is not None:
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

        else:
            # Fallback: old checkpoints without a config, use CLI settings
            if args.verbose:
                print("No config found in checkpoint; falling back to CLI hyperparameters.")

            bpe = getBPE(
                [],
                vocab_size=args.vocab_size,
                verbose=args.verbose,
                tokenizer_path=args.tokenizer,
                bpe_part=args.bpe_part,
                lower_case=args.lower_case,
            )
            args.vocab_size = bpe.vocab_size

            model = TinyGPT(
                vocab_size=args.vocab_size,
                context_length=args.context_length,
                d_model=args.d_model,
                n_heads=args.n_heads,
                num_layers=args.num_layers,
            ).to(args.device)

            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            else:
                # Allow plain state_dict as well
                model.load_state_dict(checkpoint)

        prompt = args.prompt.lower() if bpe.lower_case else args.prompt
        if args.seed is not None:
            torch.manual_seed(args.seed)
        result = generate(model, bpe, prompt, args.max_new_tokens, args.device,
                          temperature=args.temperature, top_k=args.top_k)
        print("\n--- Generated Text ---")
        if not bpe.preserve_whitespace:
            result = result.replace(bpe.get_end_of_word_mark(), " ")
        print(result)
        return  # skip training

    # ---------------------------------------------------------
    # --- Training setup ---
    # ---------------------------------------------------------
    dtype = training_dtype(args.device, args.precision)
    print(f"Training precision: {'BF16 mixed precision' if dtype is not None else 'FP32'}", flush=True)
    # --- Data & tokenizer ---
    # ---------------------------------------------------------
    oneliners = []
    for arg in args.oneliners.split(","):
        if len(arg) > 0:
            oneliners.append(arg)

    # datareader is created after BPE is ready (see below)


    start_epoch = 0
    mean_loss = 0.0

    # ---------------------------------------------------------
    # Try to load checkpoint if args.model exists
    # ---------------------------------------------------------
    ckpt_state = None
    if os.path.exists(args.model):
        print(f"Found existing model checkpoint at: {args.model}")
        try:
            checkpoint = torch.load(args.model, map_location=args.device)

            if isinstance(checkpoint, dict):
                ckpt_state = checkpoint.get("model_state_dict")
                ckpt_optim = checkpoint.get("optimizer_state_dict")
                ckpt_cfg   = checkpoint.get("config")
            else:
                ckpt_state = None
                ckpt_optim = None
                ckpt_cfg   = None

            if ckpt_state is not None:
                if ckpt_cfg is not None:
                    # ✅ Use checkpoint config as source of truth
                    if args.verbose:
                        print("Using model hyperparameters from checkpoint:")
                        for key in ckpt_cfg:
                            print(f"{key} : {ckpt_cfg[key]}")

                    args.vocab_size     = ckpt_cfg.get("vocab_size", args.vocab_size)
                    args.context_length = ckpt_cfg.get("context_length", args.context_length)
                    args.d_model        = ckpt_cfg.get("d_model", args.d_model)
                    args.n_heads        = ckpt_cfg.get("n_heads", args.n_heads)
                    args.num_layers     = ckpt_cfg.get("num_layers", args.num_layers)

                    start_epoch = checkpoint.get("epoch", 0)
                    mean_loss   = checkpoint.get("loss", 0.0)
                    print(
                        f"✅ Will resume from epoch {start_epoch}, "
                        f"mean loss={mean_loss:.4f} using checkpoint config."
                    )

                else:
                    # ⚠️ Old checkpoint without config → do the heuristic mismatch check
                    mismatch_reasons = []

                    if args.verbose:
                        print("No config in checkpoint; inferring architecture from weights.")

                    ckpt_d_model = (
                        next(iter(ckpt_state.values())).shape[-1] if ckpt_state else None
                    )
                    ckpt_num_layers = sum(
                        1
                        for k in ckpt_state.keys()
                        if "transformer_blocks" in k and "attn" in k
                    )

                    if ckpt_d_model and ckpt_d_model != args.d_model:
                        mismatch_reasons.append(
                            f"d_model mismatch (checkpoint={ckpt_d_model}, current={args.d_model})"
                        )
                    if ckpt_num_layers and ckpt_num_layers != args.num_layers:
                        mismatch_reasons.append(
                            f"num_layers mismatch (checkpoint={ckpt_num_layers}, current={args.num_layers})"
                        )

                    if mismatch_reasons:
                        print("⚠️ Checkpoint will NOT be loaded due to architecture mismatch:")
                        for reason in mismatch_reasons:
                            print(f"   - {reason}")
                        print("A new model will be trained and overwrite this checkpoint.\n")
                        ckpt_state = None
                        ckpt_optim = None
                    else:
                        start_epoch = checkpoint.get("epoch", 0)
                        mean_loss   = checkpoint.get("loss", 0.0)
                        print(
                            f"✅ Will resume from epoch {start_epoch}, "
                            f"mean loss={mean_loss:.4f} (no config, inferred match)."
                        )
            else:
                print("⚠️ Checkpoint found but missing model_state_dict — skipping load.")

        except Exception as e:
            print(f"⚠️ Failed to load checkpoint: {e}")
            print("A new model will be trained and overwrite the existing file.\n")
            ckpt_state = None
            ckpt_optim = None
    else:
        print("No existing checkpoint found. Training from scratch.\n")

    # For BPE training we still need raw text if no tokenizer exists yet
    corpus_config = load_config(args.data_config)
    raw_datareader = getData(
        verbose=args.verbose,
        lower_case=args.lower_case,
        rawdata_folder=args.rawdata,
        oneliners=oneliners,
        corpus_config=corpus_config,
    )
    validation_files = None
    reader_options = {}
    if args.validation_fraction:
        training_files, validation_files = split_validation_files(
            raw_datareader.files, args.rawdata, args.validation_fraction, args.validation_seed)
        raw_datareader.files = training_files
        raw_datareader._length = None
        folder = Path(args.rawdata)
        reader_options = {"files": training_files,
                          "cache_path": folder.parent / f"{folder.name}_train_tokens.bin"}
        print(f"File split: {len(training_files):,} training, {len(validation_files):,} validation (seed {args.validation_seed}).", flush=True)
        if ckpt_state is not None:
            print("Validation on a resumed model may include files seen before this split was enabled.", flush=True)
    bpe = getBPE(
        raw_datareader,
        vocab_size=args.vocab_size,
        verbose=args.verbose,
        tokenizer_path=args.tokenizer,
        bpe_part=args.bpe_part,
        lower_case=args.lower_case,
        bpe_max_words=args.bpe_max_words,
        character_sources=validation_files or (),
        bpe_alphabet_policy=args.bpe_alphabet_policy,
        bpe_max_chars=args.bpe_max_chars,
    )
    args.vocab_size = bpe.vocab_size

    # Build the tokenized datareader (reads folder, tokenizes, yields windows)
    datareader = TokenizedDataReader(
        folder=args.rawdata,
        bpe=bpe,
        context_size=args.context_length,
        shuffle=True,
        lower_case=args.lower_case,
        verbose=args.verbose,
        cache=True,
        corpus_config=corpus_config,
        **reader_options,
    )

    validation_reader = None
    if validation_files is not None:
        folder = Path(args.rawdata)
        validation_reader = TokenizedDataReader(
            folder=args.rawdata, files=validation_files, bpe=bpe,
            context_size=args.context_length, shuffle=False,
            lower_case=args.lower_case, verbose=args.verbose, cache=True,
            cache_path=folder.parent / f"{folder.name}_validation_tokens.bin",
            corpus_config=corpus_config)
        if not len(datareader) or not len(validation_reader):
            raise ValueError("Training and validation each need at least one complete context window; adjust the split or context length.")

    if args.verbose:
        print(args)
        print(f"Tokenizer path: {args.tokenizer}")
        print(f"Vocab size (CLI): {args.vocab_size}")
        print(f"Context length (CLI): {args.context_length}")
        print(f"Num layers (CLI): {args.num_layers}")
        print(f"Embedding dimension (CLI): {args.d_model}")
        print(f"Heads (CLI): {args.n_heads}")
        print(f"Device: {args.device}")

    # Build model from CLI (we may verify against checkpoint config below)
    model = TinyGPT(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        n_heads=args.n_heads,
        num_layers=args.num_layers,
    ).to(args.device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    lr_source = "--lr" if args.lr_explicit else "default"
    global_step = 0
    if ckpt_state is not None:
        model.load_state_dict(ckpt_state)
        if ckpt_optim is not None:
            optimizer.load_state_dict(ckpt_optim)
            lr_source = "checkpoint"
            if args.lr_explicit:
                for group in optimizer.param_groups:
                    group["lr"] = args.lr
                lr_source = "--lr overrides checkpoint"
        print("✅ Loaded model and optimizer state from checkpoint.")
        global_step = checkpoint.get("global_step", 0)
        if checkpoint.get("step", 0):
            print("Resuming a mid-epoch checkpoint: restarting the saved epoch with a new data shuffle.")

    saved_schedule = checkpoint.get("lr_schedule") if ckpt_state is not None else None
    steps_per_epoch = (len(datareader) + args.batch_size - 1) // args.batch_size
    lr_schedule = resolve_schedule(args, saved_schedule, optimizer.param_groups[0]["lr"],
                                   global_step, steps_per_epoch * max(0, args.epochs - start_epoch))
    if lr_schedule is not None:
        print(f"Learning rate schedule: linear warmup {lr_schedule.warmup_steps:,} updates, "
              f"cosine to update {lr_schedule.decay_steps:,}, peak {lr_schedule.peak_lr:g}, "
              f"floor {lr_schedule.peak_lr * lr_schedule.min_lr_ratio:g}, "
              f"start at global step {lr_schedule.start_step:,}; "
              f"next LR {lr_schedule.rate(global_step):g}", flush=True)
    else:
        print(f"Learning rate ({lr_source}): " + ", ".join(f"{group['lr']:g}" for group in optimizer.param_groups), flush=True)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {num_params:,}")

    loss_fn = nn.CrossEntropyLoss()
    model.train()

    last_validation_step = None
    latest_validation_loss = None
    if args.log_progress:
        prepare_progress_log(args.log_progress)

    def report_validation(epoch):
        nonlocal last_validation_step, latest_validation_loss
        val_loss, perplexity, tokens = evaluate(
            model, validation_reader, args.batch_size, args.validation_batches,
            args.device, args.validation_seed, dtype=dtype)
        print(f"Validation | Step {global_step:,} | Loss: {val_loss:.4f} | Perplexity: {perplexity:.2f} | Tokens: {tokens:,}", flush=True)
        if args.log_progress:
            path = args.log_progress + ".validation.csv"
            exists = os.path.exists(path) and os.path.getsize(path) > 0
            with open(path, "a", newline="") as output:
                writer = csv.writer(output)
                if not exists:
                    writer.writerow(["epoch", "global_step", "validation_loss", "perplexity", "tokens"])
                writer.writerow([epoch, global_step, val_loss, perplexity, tokens])
        last_validation_step = global_step
        latest_validation_loss = val_loss

    if validation_reader is not None:
        report_validation(start_epoch)

    # ---------------------------------------------------------
    # --- Training loop ---
    # ---------------------------------------------------------
    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        total_loss, step = 0.0, 0
        mean_loss = 0.0
        last_progress_row = None
        last_progress_position = None
        timing_steps = timing_tokens = 0
        timing_seconds = 0.0

        for x, y in stream_batches(
            datareader, bpe, args.context_length, args.batch_size
        ):
            if lr_schedule is not None:
                for group in optimizer.param_groups:
                    group["lr"] = lr_schedule.rate(global_step)
            current_lr = optimizer.param_groups[0]["lr"]
            start_time = time.perf_counter()
            x, y = x.to(args.device), y.to(args.device)
            optimizer.zero_grad(set_to_none=True)
            with precision_context(args.device, dtype):
                logits = model(x)
                loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
            loss.backward()
            if args.monitor_layer_norms and (global_step + 1) % args.monitor_layer_norms_every == 0:
                # Measure gradients before clipping, and parameters before the update.
                write_layer_norms(args.monitor_layer_norms, model, epoch + 1, global_step + 1)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=1.0, error_if_nonfinite=True
            )
            optimizer.step()
            global_step += 1

            # Reading loss waits for GPU completion, so this includes forward,
            # backward and optimizer work without extra per-forward synchronizations.
            loss_value = loss.item()
            timing_seconds += time.perf_counter() - start_time
            timing_tokens += y.numel()
            timing_steps += 1
            total_loss += loss_value
            step += 1
            avg_loss = total_loss / step

            if (validation_reader is not None and args.validation_every_steps
                    and global_step % args.validation_every_steps == 0):
                report_validation(epoch + 1)

            if args.save_every_steps and global_step % args.save_every_steps == 0:
                save_checkpoint(args.model, model, optimizer, epoch, avg_loss, global_step, step, lr_schedule)
                print(f"Checkpoint saved at optimizer step {global_step:,}: {args.model}", flush=True)

            # Log progress to CSV if requested
            if args.log_progress:
                last_progress_row = [epoch + 1, step, loss_value, avg_loss, latest_validation_loss, current_lr, grad_norm.item()]
                last_progress_position = write_progress_row(args.log_progress, last_progress_row)

            if args.verbose and step % 10 == 0:
                print(
                    f"Epoch {epoch + 1}/{args.epochs} | Step {step:6d} | Avg Loss: {avg_loss:.4f} | "
                    f"LR: {current_lr:.3g} | "
                    f"Train step: {timing_seconds / timing_steps:.4f}s | "
                    f"{timing_tokens / max(timing_seconds, 1e-9):,.0f} tokens/s",
                    flush=True,
                )
                timing_steps = timing_tokens = 0
                timing_seconds = 0.0

        mean_loss = total_loss / max(1, step)
        print(f"Epoch {epoch+1} complete | Mean loss: {mean_loss:.4f}")
        if validation_reader is not None and last_validation_step != global_step:
            report_validation(epoch + 1)
            if last_progress_row is not None:
                last_progress_row[4] = latest_validation_loss
                write_progress_row(args.log_progress, last_progress_row, last_progress_position)

        # --- Save model checkpoint ---
        save_checkpoint(args.model, model, optimizer, epoch + 1, mean_loss, global_step, lr_schedule=lr_schedule)

        print(f"💾 Checkpoint saved to {args.model}")


# ---------------------------------------------------------
if __name__ == "__main__":
    main()
