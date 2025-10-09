import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import os
import sys
from itertools import islice

from datareader import PagedDataReader
from bpe import BPE
from tinygpt import TinyGPT


# ---------------------------------------------------------
# Default parameters
# ---------------------------------------------------------
default_tokenizer = "tokenizer.json"
default_bpe_part = 0.01
default_model_n_heads = 8
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
    datareader = PagedDataReader("data", lower_case=lower_case)
    if verbose:
        print(f"DataReader ready: {len(datareader)} entries")
    return datareader


def getBPE(data, vocab_size=1920, verbose=True, tokenizer_path=None, bpe_part=0.01, lower_case=True):
    """Load existing tokenizer or train a new one."""
    if tokenizer_path and os.path.exists(tokenizer_path):
        bpe = BPE(vocab_size=vocab_size, verbose=verbose, model_file=tokenizer_path, lower_case=lower_case)
        bpe.load()
        if verbose:
            print(f"Using existing tokenizer from '{tokenizer_path}'")
        return bpe

    bpe = BPE(vocab_size=vocab_size, verbose=verbose, model_file=(tokenizer_path or default_tokenizer), lower_case=lower_case)
    if verbose:
        if tokenizer_path:
            print(f"Creating new tokenizer; will save to '{tokenizer_path}' after training")
        else:
            print("Creating new tokenizer (no model path specified; using default inside BPE)")

    total = len(data)
    if verbose:
        print("Feeding data to BPE...")
    for i, row in enumerate(data):
        bpe.add_corpus(row)
        if i / total >= bpe_part:
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
# Training pipeline (streaming)
# ---------------------------------------------------------
def stream_batches(datareader, bpe, context_length, batch_size):
    """
    Generator that yields batches of (x, y) token tensors.
    It reads sequentially from datareader -> encodes -> splits into chunks -> yields mini-batches.
    """
    buffer_x, buffer_y = [], []

    for text in datareader:
        try:
            ids = bpe.encode(text)
        except ValueError as e:
            # Skip text with unknown symbols
            print(f"[Discarded text: unknown symbol] {text[:80]!r} ({e})")
            continue

        if len(ids) < 2:
            continue

        # break into overlapping chunks
        for i in range(0, len(ids) - context_length, context_length):
            x = ids[i : i + context_length]
            y = ids[i + 1 : i + 1 + context_length]
            buffer_x.append(x)
            buffer_y.append(y)

            # yield a batch when full
            if len(buffer_x) >= batch_size:
                x_tensor = torch.tensor(buffer_x, dtype=torch.long)
                y_tensor = torch.tensor(buffer_y, dtype=torch.long)
                yield x_tensor, y_tensor
                buffer_x.clear()
                buffer_y.clear()

    # flush remainder
    if buffer_x:
        x_tensor = torch.tensor(buffer_x, dtype=torch.long)
        y_tensor = torch.tensor(buffer_y, dtype=torch.long)
        yield x_tensor, y_tensor

@torch.no_grad()
def generate(model, bpe, prompt, max_new_tokens=50, device="cpu"):
    """Generate text autoregressively from a prompt."""
    model.eval()

    # Encode prompt → token IDs
    ids = bpe.encode(prompt)
    ids = torch.tensor([ids], dtype=torch.long).to(device)  # (1, C)

    for _ in range(max_new_tokens):
        # Trim to context length if too long
        x = ids[:, -model.context_length:]
        out = model(x)  # (1, C, d_model)
        logits = out @ model.E.T  # project to vocab
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
    parser = argparse.ArgumentParser(description="Train TinyGPT using a BPE tokenizer with streaming data.")
    parser.add_argument("--tokenizer", type=str, default=default_tokenizer, help="Path to tokenizer file (will load or create).")
    parser.add_argument("--vocab_size", type=int, default=1920, help="Vocabulary size for BPE training (default: 1920).")
    parser.add_argument("--verbose", type=str2bool, default=True, help="Enable verbose output (default: true).")
    parser.add_argument("--lower-case", type=str2bool, default=True, help="Only handle lower-case (default: true).")
    parser.add_argument("--bpe-part", type=float, default=default_bpe_part, help="Fraction of dataset used to train BPE.")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs for TinyGPT.")
    parser.add_argument("--context-length", type=int, default=default_model_context_length, help="Context length (sequence length).")
    parser.add_argument("--d-model", type=int, default=default_model_d_dim, help="Embedding size (model width).")
    parser.add_argument("--n-heads", type=int, default=default_model_n_heads, help="Number of heads in tranformer multi head self attention block.")
    parser.add_argument("--num-layers", type=int, default=6, help="Number of transformer layers.")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate.")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size (approximate, streaming).")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Compute device.")

    parser.add_argument("--model", type=str, default=None,
                    help="Path to a trained model (.pt) for inference.")
    parser.add_argument("--prompt", type=str, default=None,
                    help="Text prompt to generate from. If set together with --model, training is skipped.")
    parser.add_argument("--max-new-tokens", type=int, default=50,
                    help="Maximum number of tokens to generate during inference.")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    if args.verbose:
        print(f"Tokenizer path: {args.tokenizer}")
        print(f"Vocab size: {args.vocab_size}")
        print(f"Context length: {args.context_length}")
        print(f"Num layers: {args.num_layers}")
        print(f"Device: {args.device}")

    # --- Data + tokenizer ---
    datareader = getData(verbose=args.verbose, lower_case=args.lower_case)
    bpe = getBPE(datareader,
                 vocab_size=args.vocab_size,
                 verbose=args.verbose,
                 tokenizer_path=args.tokenizer,
                 bpe_part=args.bpe_part,
                 lower_case=args.lower_case)

    # --- Inference mode ---
    if args.model and args.prompt:
        print(f"Loading model from {args.model} for inference...")
        model = TinyGPT(vocab_size=args.vocab_size,
                        context_length=args.context_length,
                        d_model=args.d_model,
                        n_heads=args.n_heads,
                        num_layers=args.num_layers).to(args.device)
        model.load_state_dict(torch.load(args.model, map_location=args.device))
        print("Model loaded successfully.")
        result = generate(model, bpe, args.prompt, args.max_new_tokens, args.device)
        print("\n--- Generated Text ---")
        print(result)
        return  # skip training

    # --- Model ---
    model = TinyGPT(vocab_size=args.vocab_size,
                    context_length=args.context_length,
                    d_model=args.d_model,
                    num_layers=args.num_layers).to(args.device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {num_params:,}")

    # --- Training ---
    model.train()
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        total_loss, step = 0.0, 0

        for x, y in stream_batches(datareader, bpe, args.context_length, args.batch_size):
            x, y = x.to(args.device), y.to(args.device)

            out = model(x)  # (B, C, d_model)
            logits = out @ model.E.T  # tied weights: (B, C, vocab_size)
            loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            step += 1

            if args.verbose and step % 10 == 0:
                avg_loss = total_loss / step
                print(f"Step {step:6d} | Avg Loss: {avg_loss:.4f}")

        print(f"Epoch {epoch+1} complete | Mean loss: {total_loss / max(1, step):.4f}")

    # --- Save model ---
    torch.save(model.state_dict(), "tinygpt.pt")
    if args.verbose:
        print("Model saved to tinygpt.pt")


# ---------------------------------------------------------
if __name__ == "__main__":
    main()
