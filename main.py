import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import os
import sys
from itertools import islice
import time

from datareader import PagedDataReader, TextCorpusReader
from bpe import BPE
from tinygpt import TinyGPT


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


def getData(verbose=True, lower_case=True):
    if verbose:
        print("Initializing data reader...")
    datareader = TextCorpusReader("rawdata", lower_case=lower_case)
    if verbose:
        print(f"DataReader ready: {len(datareader)} entries")
    return datareader


def getBPE(
    data,
    vocab_size=1920,
    verbose=True,
    tokenizer_path=None,
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

    bpe = BPE(
        vocab_size=vocab_size,
        verbose=verbose,
        model_file=(tokenizer_path or default_tokenizer),
        lower_case=lower_case,
    )
    if verbose:
        if tokenizer_path:
            print(
                f"Creating new tokenizer; will save to '{tokenizer_path}' after training"
            )
        else:
            print(
                "Creating new tokenizer (no model path specified; using default inside BPE)"
            )

    total = len(data)
    if verbose:
        print("Feeding data to BPE...")
    for i, row in enumerate(data):
        bpe.add_corpus(row)
        if total > 0 and i / total >= bpe_part:
            if verbose:
                print(f"\nTraining BPE on {int(i / total * 100)}% of the dataset.")
            break
    bpe.train()

    if tokenizer_path:
        bpe.save(tokenizer_path)
        if verbose:
            print(f"Saved new tokenizer to '{tokenizer_path}'")

    if verbose:
        print("BPE ready.")
    return bpe


# ---------------------------------------------------------
# Training pipeline (streaming, token-packed)
# ---------------------------------------------------------
def stream_batches(datareader, bpe, context_length, batch_size):
    """
    Generator that yields batches of (x, y) token tensors.

    Strategy:
    - Read sequentially from `datareader` (lines/files).
    - Encode to token IDs using `bpe.encode`.
    - Maintain a *global token buffer* across texts.
    - Pack that continuous token stream into fixed-length
      context windows of size `context_length` tokens.
    - For each window, x is tokens[0:C], y is tokens[1:C+1].
    """
    token_buffer = []
    start_idx = 0  # sliding window start within token_buffer

    batch_x, batch_y = [], []

    def flush_batches():
        nonlocal batch_x, batch_y
        if batch_x:
            x_tensor = torch.tensor(batch_x, dtype=torch.long)
            y_tensor = torch.tensor(batch_y, dtype=torch.long)
            batch_x, batch_y = [], []
            return x_tensor, y_tensor
        return None

    for text in datareader:
        try:
            ids = bpe.encode(text)
        except ValueError as e:
            # Skip text with unknown symbols
            print(f"[Discarded text: unknown symbol] {text[:80]!r} ({e})")
            continue

        if not ids:
            continue

        # Extend global token stream
        token_buffer.extend(ids)

        # While we have enough tokens for at least one full (x,y) pair
        while len(token_buffer) - start_idx >= context_length + 1:
            window = token_buffer[start_idx : start_idx + context_length + 1]
            x = window[:-1]
            y = window[1:]

            batch_x.append(x)
            batch_y.append(y)

            start_idx += context_length  # non-overlapping windows

            # Yield batch when full
            if len(batch_x) >= batch_size:
                out = flush_batches()
                if out is not None:
                    yield out

        # Periodically compact buffer to avoid unbounded growth
        # (drop tokens we've already stepped over)
        if start_idx > 10 * context_length:
            token_buffer = token_buffer[start_idx:]
            start_idx = 0

    # After we exhaust the datareader, still try to use remaining tokens
    while len(token_buffer) - start_idx >= context_length + 1:
        window = token_buffer[start_idx : start_idx + context_length + 1]
        x = window[:-1]
        y = window[1:]
        batch_x.append(x)
        batch_y.append(y)
        start_idx += context_length

        if len(batch_x) >= batch_size:
            out = flush_batches()
            if out is not None:
                yield out

    # Flush any remaining partial batch
    out = flush_batches()
    if out is not None:
        yield out


@torch.no_grad()
def generate(model, bpe, prompt, max_new_tokens=50, device="cpu"):
    """Generate text autoregressively from a prompt."""
    model.eval()

    # Encode prompt → token IDs
    ids = bpe.encode(prompt)
    ids = torch.tensor([ids], dtype=torch.long).to(device)  # (1, C)

    for _ in range(max_new_tokens):
        # Trim to context length if too long
        x = ids[:, -model.context_length :]
        logits = model(x)  # (1, C, d_model)
        next_token_logits = logits[:, -1, :]  # last position
        probs = torch.softmax(next_token_logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        ids = torch.cat([ids, next_id], dim=1)

    # Decode all tokens back to text
    generated = bpe.decode(ids[0].tolist())
    return generated


# ---------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------
def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Train TinyGPT using a BPE tokenizer with streaming data."
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=default_tokenizer,
        help="Path to tokenizer file (will load or create).",
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=1920,
        help="Vocabulary size for BPE training (default: 1920).",
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
        help="Context length (sequence length).",
    )
    parser.add_argument(
        "--d-model",
        type=int,
        default=default_model_d_dim,
        help="Embedding size (model width).",
    )
    parser.add_argument(
        "--n-heads",
        type=int,
        default=default_model_n_heads,
        help="Number of heads in tranformer multi head self attention block.",
    )
    parser.add_argument(
        "--num-layers", type=int, default=6, help="Number of transformer layers."
    )
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate.")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size (approximate, streaming).",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/model.pt",
        help="Where to store the model data.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Compute device.",
    )

    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to a trained model (.pt) for inference.",
    )
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
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    if args.verbose:
        print(args)
        print(f"Tokenizer path: {args.tokenizer}")
        print(f"Vocab size: {args.vocab_size}")
        print(f"Context length: {args.context_length}")
        print(f"Num layers: {args.num_layers}")
        print(f"Embedding dimension: {args.d_model}")
        print(f"Heads: {args.n_heads}")
        print(f"Device: {args.device}")

    # ---------------------------------------------------------
    # --- Inference-only mode (skip training) ---
    # ---------------------------------------------------------
    if args.model and args.prompt:
        # Initialize model with CLI hyperparameters
        model = TinyGPT(
            vocab_size=args.vocab_size,
            context_length=args.context_length,
            d_model=args.d_model,
            n_heads=args.n_heads,
            num_layers=args.num_layers,
        ).to(args.device)

        # Load checkpoint from the provided model path
        print(f"Loading model from '{args.model}' for inference...")
        checkpoint = torch.load(args.model, map_location=args.device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            # Allow plain state_dict as well
            model.load_state_dict(checkpoint)

        # Load existing tokenizer
        bpe = getBPE(
            [],
            vocab_size=args.vocab_size,
            verbose=args.verbose,
            tokenizer_path=args.tokenizer,
            bpe_part=args.bpe_part,
            lower_case=args.lower_case,
        )

        prompt = args.prompt.lower() if args.lower_case else args.prompt
        result = generate(model, bpe, prompt, args.max_new_tokens, args.device)
        print("\n--- Generated Text ---")
        result = result.replace("</w>", " ")
        print(result)
        return  # skip training

    # ---------------------------------------------------------
    # --- Training setup ---
    # ---------------------------------------------------------
    model = TinyGPT(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        n_heads=args.n_heads,
        num_layers=args.num_layers,
    ).to(args.device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {num_params:,}")

    start_epoch = 0
    mean_loss = 0.0

    # ---------------------------------------------------------
    # Try to load checkpoint if args.model_path exists
    # ---------------------------------------------------------
    if os.path.exists(args.model_path):
        print(f"Found existing model checkpoint at: {args.model_path}")
        try:
            checkpoint = torch.load(args.model_path, map_location=args.device)

            # Validate key hyperparameters
            ckpt_model = checkpoint.get("model_state_dict", None)
            if ckpt_model is not None:
                # Attempt to infer shapes from the saved weights
                ckpt_d_model = (
                    next(iter(ckpt_model.values())).shape[-1] if ckpt_model else None
                )
                ckpt_num_layers = sum(
                    1
                    for k in ckpt_model.keys()
                    if "transformer_blocks" in k and "attn" in k
                )
                ckpt_n_heads = (
                    None  # can’t reliably infer from weights unless saved manually
                )

                # Compare only those we can check reliably
                mismatch_reasons = []
                if ckpt_d_model and ckpt_d_model != args.d_model:
                    mismatch_reasons.append(
                        f"d_model mismatch (checkpoint={ckpt_d_model}, current={args.d_model})"
                    )
                if ckpt_num_layers and ckpt_num_layers != args.num_layers:
                    mismatch_reasons.append(
                        f"num_layers mismatch (checkpoint={ckpt_num_layers}, current={args.num_layers})"
                    )

                if mismatch_reasons:
                    print("⚠️ Checkpoint not loaded due to architecture mismatch:")
                    for reason in mismatch_reasons:
                        print(f"   - {reason}")
                    print(
                        "A new model will be trained and overwrite this checkpoint.\n"
                    )
                else:
                    model.load_state_dict(checkpoint["model_state_dict"])
                    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                    start_epoch = checkpoint.get("epoch", 0)
                    mean_loss = checkpoint.get("loss", 0.0)
                    print(
                        f"✅ Loaded checkpoint from epoch {start_epoch}, mean loss={mean_loss:.4f}"
                    )
            else:
                print(
                    "⚠️ Checkpoint found but missing model_state_dict — skipping load."
                )
        except Exception as e:
            print(f"⚠️ Failed to load checkpoint: {e}")
            print("A new model will be trained and overwrite the existing file.\n")
    else:
        print("No existing checkpoint found. Training from scratch.\n")

    # ---------------------------------------------------------
    # --- Data & tokenizer ---
    # ---------------------------------------------------------
    datareader = getData(verbose=args.verbose, lower_case=args.lower_case)
    bpe = getBPE(
        datareader,
        vocab_size=args.vocab_size,
        verbose=args.verbose,
        tokenizer_path=args.tokenizer,
        bpe_part=args.bpe_part,
        lower_case=args.lower_case,
    )

    loss_fn = nn.CrossEntropyLoss()
    model.train()

    # ---------------------------------------------------------
    # --- Training loop ---
    # ---------------------------------------------------------
    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        total_loss, step = 0.0, 0
        mean_loss = 0.0

        for x, y in stream_batches(
            datareader, bpe, args.context_length, args.batch_size
        ):
            x, y = x.to(args.device), y.to(args.device)

            # --- Measure inference (forward) time ---
            if args.device.startswith("cuda"):
                torch.cuda.synchronize()
            start_time = time.time()

            logits = model(x)  # forward pass

            if args.device.startswith("cuda"):
                torch.cuda.synchronize()
            end_time = time.time()
            inference_time = end_time - start_time

            # --- Compute loss and backprop ---
            loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            step += 1

            if args.verbose and step % 10 == 0:
                avg_loss = total_loss / step
                print(
                    f"Step {step:6d} | Avg Loss: {avg_loss:.4f} | Inference: {inference_time:.4f}s"
                )

        mean_loss = total_loss / max(1, step)
        print(f"Epoch {epoch+1} complete | Mean loss: {mean_loss:.4f}")

        # --- Save model checkpoint ---
        os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": mean_loss,
            },
            args.model_path,
        )

        print(f"💾 Checkpoint saved to {args.model_path}")


# ---------------------------------------------------------
if __name__ == "__main__":
    main()
