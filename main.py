import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import csv
import os
from itertools import islice
import time
import tempfile
import hashlib
import math
import random
import threading
from contextlib import nullcontext, contextmanager
from pathlib import Path

from datareader import PagedDataReader, TextCorpusReader, TokenizedDataReader
from bpe import BPE, getBPE
from corpus import load_config, text_chunks
from tinygpt import TinyGPT, MOE_DEFAULTS
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


@contextmanager
def compilation_progress(stage, interval=15):
    """Report first-call work without inventing compiler completion percentages."""
    started = time.perf_counter()
    stopped = threading.Event()
    print(f"Compile warmup: starting {stage}. This may take several minutes.", flush=True)

    def heartbeat():
        while not stopped.wait(interval):
            print(f"Compile warmup: {stage} still running ({time.perf_counter() - started:.0f}s elapsed; compilation/execution).", flush=True)

    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    try:
        yield
    except BaseException:
        print(f"Compile warmup: {stage} failed or was interrupted.", flush=True)
        raise
    else:
        print(f"Compile warmup: finished {stage} in {time.perf_counter() - started:.1f}s.", flush=True)
    finally:
        stopped.set()
        thread.join()


def configure_compilation(model, args):
    if args.compile_model:
        print("Setting up whole-model compilation (includes QK normalization when enabled)...", flush=True)
        model.compile()
    elif args.compile_qk_norm:
        print("Setting up QK normalization compilation...", flush=True)
        model.compile_qk_norms()
    else:
        return False
    print("Compiler configured. Compilation is lazy: first validation, training forward and backward calls warm up separately. Further shapes may recompile.", flush=True)
    return True


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


def save_checkpoint(path, model, optimizer, epoch, loss, global_step, step=0, lr_schedule=None, response_end_marker=None):
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
            "qk_norm": model.qk_norm,
            **{name: getattr(model, name) for name in MOE_DEFAULTS},
        },
    }
    if response_end_marker is not None:
        checkpoint['response_end_marker'] = response_end_marker
    descriptor, temporary = tempfile.mkstemp(prefix="checkpoint-", suffix=".tmp", dir=directory)
    os.close(descriptor)
    try:
        torch.save(checkpoint, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def minify_checkpoint(args, parser):
    if args.model.resolve() == args.output.resolve():
        parser.error('--output must be distinct from --model')
    if not args.model.is_file():
        parser.error('--model must be an existing file')
    if args.output.exists():
        parser.error('--output must not already exist')
    try:
        checkpoint = torch.load(args.model, map_location='cpu', weights_only=True, mmap=True)
    except RuntimeError as error:
        if 'mmap can only be used' not in str(error):
            raise
        checkpoint = torch.load(args.model, map_location='cpu', weights_only=True)
    if not isinstance(checkpoint, dict):
        parser.error('Expected a checkpoint dictionary or model state dictionary')
    if 'model_state_dict' in checkpoint:
        if not isinstance(checkpoint['model_state_dict'], dict):
            parser.error('Invalid model_state_dict in checkpoint')
        keys = ('model_state_dict', 'config', 'response_end_marker', 'epoch', 'step', 'global_step', 'loss')
        result = {key: checkpoint[key] for key in keys if key in checkpoint}
    elif checkpoint and all(isinstance(value, torch.Tensor) for value in checkpoint.values()):
        result = checkpoint
    else:
        parser.error('No model state found in checkpoint')
    try:
        with args.output.open('xb') as output:
            try:
                torch.save(result, output)
            except BaseException:
                output.close()
                args.output.unlink()
                raise
    except FileExistsError:
        parser.error('--output must not already exist')
    print(f'Saved {args.output} ({args.output.stat().st_size:,} bytes; source {args.model.stat().st_size:,} bytes).')
    print('Optimizer state and learning-rate schedule are not included; resuming training uses a fresh optimizer.')


PROGRESS_COLUMNS = ["epoch", "step", "loss", "mean_loss", "validation_loss", "lr", "grad_norm"]


def prepare_progress_log(path, columns=PROGRESS_COLUMNS):
    """Add missing metric columns to legacy logs, leaving old values unknown."""
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        with path.open("w", newline="") as output:
            csv.writer(output).writerow(columns)
        return
    with path.open(newline="") as source:
        reader = csv.reader(source)
        header = next(reader)
        if header == columns:
            with path.open("rb+") as output:
                output.seek(-1, os.SEEK_END)
                if output.read(1) not in (b"\n", b"\r"):
                    output.write(b"\n")
            return
        if len(header) < 4 or header != columns[:len(header)]:
            raise ValueError(f"Unexpected progress CSV columns in {path}: {header}")
        descriptor, temporary = tempfile.mkstemp(prefix="progress-", suffix=".tmp", dir=path.parent)
        try:
            with os.fdopen(descriptor, "w", newline="") as output:
                writer = csv.writer(output)
                writer.writerow(columns)
                for row in reader:
                    if len(row) != len(header):
                        raise ValueError(f"Malformed progress CSV row in {path}: {row}")
                    writer.writerow([*row, *([""] * (len(columns) - len(header)))])
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
def generate(model, bpe, prompt, max_new_tokens=50, device="cpu", temperature=1.0, top_k=0, response_end_marker=None, show_end_marker=False):
    """Generate text autoregressively from a prompt."""
    if not math.isfinite(temperature) or temperature < 0 or top_k < 0:
        raise ValueError("Temperature must be finite and nonnegative; top_k must be nonnegative")
    model.eval()

    # Encode prompt → token IDs
    ids = bpe.encode(prompt)
    ids = torch.tensor([ids], dtype=torch.long).to(device)  # (1, C)
    prompt_length = ids.size(1)

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
        if response_end_marker and response_end_marker in bpe.decode(ids[0, prompt_length:].tolist()):
            break

    # Decode all tokens back to text
    generated = bpe.decode(ids[0].tolist())
    if response_end_marker:
        prefix = bpe.decode(ids[0, :prompt_length].tolist())
        response = bpe.decode(ids[0, prompt_length:].tolist())
        before, separator, _ = response.partition(response_end_marker)
        response = before + (separator if show_end_marker else '')
        return prefix + response
    return generated


# ---------------------------------------------------------
# Argument parsing
# ------------------------------------self.chunked_file_readers.append(ChunkedFile(file, self.buffer_size))---------------------
def default_device():
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def add_compile_arguments(parser):
    parser.add_argument("--compile-qk-norm", action="store_true",
                        help="Compile Q/K RMSNorm forward/backward with Inductor (training only; requires QK normalization). Execution option, not saved in checkpoints.")
    parser.add_argument("--compile-model", action="store_true",
                        help="Compile the whole model with Inductor during training; takes precedence over --compile-qk-norm. Not saved in checkpoints.")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Pretrain, fine-tune, or run inference with TinyGPT.",
        epilog="Use finetune --help or minify --help for command-specific options.",
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
    parser.add_argument("--qk-norm", action=argparse.BooleanOptionalAction, default=False,
                        help="Enable per-head Q/K RMSNorm for fresh models. On resume/inference, the saved architecture takes precedence; old checkpoints disable it.")
    parser.add_argument("--moe", action=argparse.BooleanOptionalAction, default=False, help="Use sparse MoE MLPs for fresh models; saved architecture wins on resume.")
    parser.add_argument("--num-experts", type=int, default=4)
    parser.add_argument("--moe-top-k", type=int, default=2)
    parser.add_argument("--moe-aux-loss-weight", type=float, default=0.01)
    add_compile_arguments(parser)
    parser.add_argument("--beta1", type=float, default=None,
                        help="AdamW beta1: default 0.9 for new runs; omitted on resume preserves the saved value.")
    parser.add_argument("--beta2", type=float, default=None,
                        help="AdamW beta2: default 0.999 for new runs; omitted on resume preserves the saved value.")
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
    parser.add_argument("--bpe-keep-chars", default="", metavar="TEXT",
                        help="Characters to guarantee in a newly trained tokenizer, even when limiting the alphabet.")
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

    commands = parser.add_subparsers(title="commands", metavar="{finetune,minify}")
    finetune_parser = commands.add_parser(
        "finetune", help="Fine-tune a pretrained model on instruction/response examples.",
        description="Supervised fine-tuning on formatted instruction/response files.",
    )
    finetune_parser.add_argument('--data', type=Path, required=True, help='Directory containing training examples (.txt)')
    finetune_parser.add_argument('--base-model', type=Path, required=True, help='Pretrained checkpoint')
    finetune_parser.add_argument('--tokenizer', type=Path, required=True, help='Exact tokenizer used for pretraining')
    finetune_parser.add_argument('--output', type=Path, required=True, help='New output checkpoint')
    finetune_parser.add_argument('--log-progress', type=Path, default=None, help='Training CSV path (defaults to output path plus .progress.csv)')
    finetune_parser.add_argument('--response-end-marker', default=DEFAULT_RESPONSE_END_MARKER,
                                 help='Text appended to every response and used to stop generation')
    finetune_parser.add_argument('--device', default=default_device(), help='Compute device')
    finetune_parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs (default: %(default)s)')
    finetune_parser.add_argument('--batch-size', type=int, default=2, help='Training windows per batch (default: %(default)s)')
    finetune_parser.add_argument('--lr', type=float, default=2e-5, help='AdamW learning rate (default: %(default)s)')
    finetune_parser.add_argument('--validation-fraction', type=float, default=0.05,
                                 help='Fraction of files held out for validation (default: %(default)s)')
    finetune_parser.add_argument('--validation-every-steps', type=int, default=0,
                                 help='Validate every N optimizer steps and at epoch end (default: epoch end only)')
    finetune_parser.add_argument('--save-every-steps', type=int, default=0,
                                 help='Save the latest checkpoint every N optimizer steps and at epoch end (default: epoch end only)')
    finetune_parser.add_argument('--seed', type=int, default=42, help='Seed for splitting and shuffling files (default: %(default)s)')
    add_compile_arguments(finetune_parser)
    finetune_parser.set_defaults(command_handler=finetune, command_parser=finetune_parser)

    minify_parser = commands.add_parser("minify", help="Export a checkpoint without optimizer state.")
    minify_parser.add_argument('--model', type=Path, required=True, help='Existing checkpoint')
    minify_parser.add_argument('--output', type=Path, required=True, help='New output file; never overwrites an existing file')
    minify_parser.set_defaults(command_handler=minify_checkpoint, command_parser=minify_parser)

    args = parser.parse_args(argv)
    if hasattr(args, "command_handler"):
        return args
    if args.num_experts < 1 or not 1 <= args.moe_top_k <= args.num_experts:
        parser.error("Require 1 <= --moe-top-k <= --num-experts")
    if not math.isfinite(args.moe_aux_loss_weight) or args.moe_aux_loss_weight < 0:
        parser.error("--moe-aux-loss-weight must be finite and nonnegative")
    for name in ("beta1", "beta2"):
        value = getattr(args, name)
        if value is not None and (not math.isfinite(value) or not 0 <= value < 1):
            parser.error(f"--{name} must be finite and in [0, 1)")
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

RESPONSE_MARKER = '\n\n### Response\n'
DEFAULT_RESPONSE_END_MARKER = '\n\n### End Response\n'


def finetune_examples(path, tokenizer, context_length, response_end_marker=DEFAULT_RESPONSE_END_MARKER):
    text = path.read_text(encoding='utf-8')
    if RESPONSE_MARKER not in text:
        raise ValueError(f'Invalid Dolly format: {path}')
    prompt, response = text.split(RESPONSE_MARKER, 1)
    if prompt.startswith('### Instruction\n'):
        if '\n\n### Context\n' in prompt:
            instruction, context = prompt.split('\n\n### Context\n', 1)
            prompt = '### Context\n' + context + '\n\n' + instruction
    elif not (prompt.startswith('### Context\n') and '\n\n### Instruction\n' in prompt):
        raise ValueError(f'Invalid Dolly format: {path}')
    prompt += RESPONSE_MARKER
    if not response.strip():
        raise ValueError(f'Empty response: {path}')
    # The marker ends in whitespace, so the tokenizer cannot merge across the boundary.
    prompt_ids = tokenizer.encode(prompt)
    response_ids = tokenizer.encode(response) + tokenizer.encode(response_end_marker)
    if not prompt_ids or not response_ids:
        raise ValueError(f'Empty tokenized prompt or response: {path}')
    tokens = prompt_ids + response_ids
    # Cover every answer target once, keeping up to half the window for context.
    stride = max(1, context_length // 2)
    for first in range(len(prompt_ids), len(tokens), stride):
        end = min(first + stride, len(tokens))
        start = max(0, end - context_length - 1)
        x = tokens[start:end - 1]
        y = tokens[start + 1:end]
        y = [-100 if start + i + 1 < first else token for i, token in enumerate(y)]
        yield x, y


def finetune_batches(files, tokenizer, context_length, batch_size, response_end_marker=DEFAULT_RESPONSE_END_MARKER):
    pending = []
    for path in files:
        for pair in finetune_examples(path, tokenizer, context_length, response_end_marker):
            pending.append(pair)
            if len(pending) == batch_size:
                yield finetune_collate(pending)
                pending = []
    if pending:
        yield finetune_collate(pending)


def finetune_collate(items):
    width = max(len(x) for x, _ in items)
    x = torch.tensor([x + [0] * (width - len(x)) for x, _ in items], dtype=torch.long)
    y = torch.tensor([y + [-100] * (width - len(y)) for _, y in items], dtype=torch.long)
    return x, y


def finetune(args, parser):
    if args.output.exists() or args.output.resolve() == args.base_model.resolve():
        parser.error('--output must be a new path distinct from the base model')
    progress_path = args.log_progress or Path(str(args.output) + '.progress.csv')
    validation_path = Path(str(progress_path) + '.validation.csv')
    log_paths = {progress_path.resolve(), validation_path.resolve()}
    protected_paths = {args.output.resolve(), args.base_model.resolve(), args.tokenizer.resolve()}
    if len(log_paths) != 2 or log_paths & protected_paths:
        parser.error('Progress and validation logs must be distinct from each other and from model and tokenizer paths')
    if args.epochs < 1 or args.batch_size < 1 or not 0 < args.lr or not 0 < args.validation_fraction < 1:
        parser.error('epochs, batch size, learning rate and validation fraction must be positive; validation fraction must be below 1')
    if args.validation_every_steps < 0:
        parser.error('--validation-every-steps must be nonnegative')
    if args.save_every_steps < 0:
        parser.error('--save-every-steps must be nonnegative')
    files = sorted(args.data.rglob('*.txt'))
    if len(files) < 2:
        parser.error('Fine-tuning data must contain at least two .txt files')
    rng = random.Random(args.seed)
    rng.shuffle(files)
    nval = min(len(files) - 1, max(1, round(len(files) * args.validation_fraction)))
    validation, training = files[:nval], files[nval:]
    checkpoint = torch.load(args.base_model, map_location='cpu', weights_only=True)
    cfg = checkpoint['config']
    if args.compile_qk_norm and not args.compile_model and not cfg.get('qk_norm', False):
        parser.error('--compile-qk-norm requires a QK-normalized base checkpoint')
    tok = BPE()
    tok.load(str(args.tokenizer))
    if not tok.preserve_whitespace:
        parser.error('Fine-tuning requires a whitespace-preserving tokenizer')
    if not args.response_end_marker or tok.decode(tok.encode(args.response_end_marker)) != args.response_end_marker:
        parser.error('Response end marker must be nonempty and exactly representable by the tokenizer')
    if tok.vocab_size != cfg['vocab_size'] or not tok.vocab or max(tok.vocab.values()) >= cfg['vocab_size']:
        parser.error('Tokenizer vocabulary does not match checkpoint; use the exact pretraining tokenizer')
    model = TinyGPT(qk_norm=cfg.get('qk_norm', False),
                    **{name: cfg.get(name, value) for name, value in MOE_DEFAULTS.items()},
                    **{name: cfg[name] for name in ('vocab_size', 'context_length', 'd_model', 'n_heads', 'num_layers')}).to(args.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    num_params = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    print(
        f'Base checkpoint: {args.base_model}\n'
        f'Tokenizer path: {args.tokenizer}\n'
        f'Vocab size: {model.vocab_size:,}\n'
        f'Context length: {model.context_length:,}\n'
        f'Num layers: {model.num_layers}\n'
        f'Embedding dimension: {model.d_model:,}\n'
        f'Heads: {model.n_heads}\n'
        f'Device: {args.device}\n'
        f'QK normalization: {model.qk_norm}\n'
        f'MoE: {model.moe}; experts={model.num_experts}, top-k={model.moe_top_k}, auxiliary weight={model.moe_aux_loss_weight}\n'
        f'Total trainable parameters: {num_params:,}',
        flush=True,
    )
    context_length = cfg['context_length']
    print(f'Counting training windows across {len(training):,} examples...', flush=True)
    training_windows = sum(
        1 for path in training
        for _ in finetune_examples(path, tok, context_length, args.response_end_marker)
    )
    steps_per_epoch = (training_windows + args.batch_size - 1) // args.batch_size
    compile_training_pending = configure_compilation(model, args)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    prepare_progress_log(progress_path, [*PROGRESS_COLUMNS, 'tokens_per_second'])
    prepare_progress_log(validation_path, ['epoch', 'global_step', 'validation_loss', 'perplexity', 'tokens'])
    print(f'Training progress: {progress_path}; validation: {validation_path}', flush=True)

    @torch.no_grad()
    def report_validation(epoch, step):
        model.eval()
        val_total = val_count = 0
        try:
            for x, y in finetune_batches(validation, tok, context_length, args.batch_size, args.response_end_marker):
                x, y = x.to(args.device), y.to(args.device)
                logits = model(x)
                loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction='sum')
                val_total += loss.item()
                val_count += (y != -100).sum().item()
        finally:
            model.train()
        val_loss = val_total / val_count
        perplexity = math.exp(val_loss)
        print(f'Validation | Step {step:,} | Response loss: {val_loss:.4f} | Perplexity: {perplexity:.2f}',
              flush=True)
        with validation_path.open('a', newline='') as output:
            csv.writer(output).writerow([epoch, step, val_loss, perplexity, val_count])
        return val_loss

    step = 0
    print(f'Fine-tuning {len(training):,} examples ({steps_per_epoch:,} steps per epoch); '
          f'validating on {len(validation):,}.', flush=True)
    for epoch in range(args.epochs):
        rng.shuffle(training)
        model.train()
        total = count = 0
        timing_tokens = 0
        timing_seconds = 0.0
        batches = finetune_batches(training, tok, context_length, args.batch_size, args.response_end_marker)
        for epoch_step, (x, y) in enumerate(batches, start=1):
            start_time = time.perf_counter()
            x, y = x.to(args.device), y.to(args.device)
            optimizer.zero_grad(set_to_none=True)
            with compilation_progress("first training forward") if compile_training_pending else nullcontext():
                if model.moe:
                    logits, aux = model(x, return_aux=True)
                else:
                    logits, aux = model(x), 0
                loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            with compilation_progress("first training backward") if compile_training_pending else nullcontext():
                (loss + model.moe_aux_loss_weight * aux).backward()
            compile_training_pending = False
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            n = (y != -100).sum().item()
            loss_value = loss.item()
            step_seconds = max(time.perf_counter() - start_time, 1e-9)
            timing_seconds += step_seconds
            timing_tokens += x.numel()
            total += loss_value * n
            count += n
            step += 1
            epoch_end = epoch_step == steps_per_epoch
            val_loss = None
            if epoch_end or (args.validation_every_steps and step % args.validation_every_steps == 0):
                val_loss = report_validation(epoch + 1, step)
            write_progress_row(progress_path, [epoch + 1, step, loss_value, total / count, val_loss,
                                               args.lr, grad_norm.item(), x.numel() / step_seconds])
            if not epoch_end and args.save_every_steps and step % args.save_every_steps == 0:
                checkpoint_loss = val_loss if val_loss is not None else total / count
                save_checkpoint(args.output, model, optimizer, epoch, checkpoint_loss, step, epoch_step,
                                response_end_marker=args.response_end_marker)
                print(f'Checkpoint saved at optimizer step {step:,}: {args.output}', flush=True)
            if step % 100 == 0 or epoch_end:
                epoch_progress = epoch_step / steps_per_epoch
                print(f'Epoch {epoch + 1}/{args.epochs} | Step {step:6d} ({epoch_progress:7.2%}) | '
                      f'Response loss: {total / count:.4f} | {timing_tokens / timing_seconds:,.0f} tokens/s',
                      flush=True)
                timing_tokens = 0
                timing_seconds = 0.0
        print(f'Epoch {epoch + 1}: train response loss {total / count:.4f}; validation response loss {val_loss:.4f}', flush=True)
        save_checkpoint(args.output, model, optimizer, epoch + 1, val_loss, step,
                        response_end_marker=args.response_end_marker)
        print(f'Saved {args.output}', flush=True)


def main(argv=None):
    args = parse_args(argv)
    if hasattr(args, "command_handler"):
        return args.command_handler(args, args.command_parser)

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
                qk_norm=cfg.get("qk_norm", False),
                **{name: cfg.get(name, value) for name, value in MOE_DEFAULTS.items()},
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

    if ckpt_state is not None:
        saved_qk_norm = (ckpt_cfg or {}).get("qk_norm", False)
        if args.qk_norm != saved_qk_norm:
            print(f"Using checkpoint QK normalization setting: {saved_qk_norm} (overrides CLI).")
        args.qk_norm = saved_qk_norm
        for name, default in MOE_DEFAULTS.items():
            setattr(args, name, (ckpt_cfg or {}).get(name, default))

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
        bpe_keep_chars=args.bpe_keep_chars,
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
        qk_norm=args.qk_norm,
        **{name: getattr(args, name) for name in MOE_DEFAULTS},
    ).to(args.device)
    print(f"QK normalization: {model.qk_norm}", flush=True)
    print(f"MoE: {model.moe}; experts={model.num_experts}, top-k={model.moe_top_k}, auxiliary weight={model.moe_aux_loss_weight}", flush=True)
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

    # Apply explicit overrides after restoration, preserving omitted betas and moments.
    for group in optimizer.param_groups:
        beta1, beta2 = group["betas"]
        group["betas"] = (args.beta1 if args.beta1 is not None else beta1,
                          args.beta2 if args.beta2 is not None else beta2)
    print("AdamW betas: " + ", ".join(str(group["betas"]) for group in optimizer.param_groups), flush=True)

    compile_enabled = configure_compilation(model, args)

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
        with compilation_progress("initial validation (includes all validation batches)") if compile_enabled else nullcontext():
            report_validation(start_epoch)

    # ---------------------------------------------------------
    # --- Training loop ---
    # ---------------------------------------------------------
    compile_training_pending = compile_enabled
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
            with compilation_progress("first training forward") if compile_training_pending else nullcontext():
                with precision_context(args.device, dtype):
                    if model.moe:
                        logits, aux_loss = model(x, return_aux=True)
                    else:
                        logits, aux_loss = model(x), 0.0
                    loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
                    objective = loss + model.moe_aux_loss_weight * aux_loss
            with compilation_progress("first training backward") if compile_training_pending else nullcontext():
                objective.backward()
            if compile_training_pending:
                print("Initial training compile warmup finished; continuing training. New shapes or modes may trigger further compilation.", flush=True)
                compile_training_pending = False
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
                epoch_progress = step / steps_per_epoch
                print(
                    f"Epoch {epoch + 1}/{args.epochs} | Step {step:6d} ({epoch_progress:7.2%}) | "
                    f"Avg Loss: {avg_loss:.4f} | "
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
