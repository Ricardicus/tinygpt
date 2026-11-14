#!/usr/bin/env python3
"""Supervised fine-tuning on the downloaded Dolly instruction/response files."""
import argparse
import random
from pathlib import Path

import torch
import torch.nn.functional as F

from bpe import BPE
from main import default_device, save_checkpoint
from tinygpt import TinyGPT, MOE_DEFAULTS

RESPONSE_MARKER = '\n\n### Response\n'


def examples(path, tokenizer, context_length):
    text = path.read_text(encoding='utf-8')
    if not text.startswith('### Instruction\n') or RESPONSE_MARKER not in text:
        raise ValueError(f'Invalid Dolly format: {path}')
    prompt, response = text.split(RESPONSE_MARKER, 1)
    prompt += RESPONSE_MARKER
    if not response.strip():
        raise ValueError(f'Empty response: {path}')
    # The marker ends in whitespace, so the tokenizer cannot merge across the boundary.
    prompt_ids = tokenizer.encode(prompt)
    response_ids = tokenizer.encode(response)
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


def batches(files, tokenizer, context_length, batch_size):
    pending = []
    for path in files:
        for pair in examples(path, tokenizer, context_length):
            pending.append(pair)
            if len(pending) == batch_size:
                yield collate(pending)
                pending = []
    if pending:
        yield collate(pending)


def collate(items):
    width = max(len(x) for x, _ in items)
    x = torch.tensor([x + [0] * (width - len(x)) for x, _ in items], dtype=torch.long)
    y = torch.tensor([y + [-100] * (width - len(y)) for _, y in items], dtype=torch.long)
    return x, y


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True, help='Dolly corpus directory')
    p.add_argument('--base-model', type=Path, required=True)
    p.add_argument('--tokenizer', type=Path, required=True, help='Exact tokenizer used for pretraining')
    p.add_argument('--output', type=Path, required=True, help='New output checkpoint; must not already exist')
    p.add_argument('--device', default=default_device())
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--batch-size', type=int, default=2)
    p.add_argument('--lr', type=float, default=2e-5)
    p.add_argument('--validation-fraction', type=float, default=0.05)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args(argv)
    if args.output.exists() or args.output.resolve() == args.base_model.resolve():
        p.error('--output must be a new path distinct from the base model')
    if args.epochs < 1 or args.batch_size < 1 or not 0 < args.lr or not 0 < args.validation_fraction < 1:
        p.error('epochs, batch size, learning rate and validation fraction must be positive; validation fraction must be below 1')
    files = sorted(args.data.rglob('*.txt'))
    if len(files) < 2:
        p.error('Dolly corpus must contain at least two .txt files')
    rng = random.Random(args.seed)
    rng.shuffle(files)
    nval = min(len(files) - 1, max(1, round(len(files) * args.validation_fraction)))
    validation, training = files[:nval], files[nval:]
    checkpoint = torch.load(args.base_model, map_location='cpu', weights_only=True)
    cfg = checkpoint['config']
    tok = BPE()
    tok.load(str(args.tokenizer))
    if not tok.preserve_whitespace:
        p.error('Fine-tuning requires a whitespace-preserving tokenizer')
    if tok.vocab_size != cfg['vocab_size'] or not tok.vocab or max(tok.vocab.values()) >= cfg['vocab_size']:
        p.error('Tokenizer vocabulary does not match checkpoint; use the exact pretraining tokenizer')
    model = TinyGPT(qk_norm=cfg.get('qk_norm', False),
                    **{name: cfg.get(name, value) for name, value in MOE_DEFAULTS.items()},
                    **{name: cfg[name] for name in ('vocab_size', 'context_length', 'd_model', 'n_heads', 'num_layers')}).to(args.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    step = 0
    print(f'Fine-tuning {len(training):,} examples; validating on {len(validation):,}.', flush=True)
    for epoch in range(args.epochs):
        rng.shuffle(training)
        model.train()
        total = count = 0
        for x, y in batches(training, tok, cfg['context_length'], args.batch_size):
            x, y = x.to(args.device), y.to(args.device)
            optimizer.zero_grad(set_to_none=True)
            if model.moe:
                logits, aux = model(x, return_aux=True)
            else:
                logits, aux = model(x), 0
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            (loss + model.moe_aux_loss_weight * aux).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            n = (y != -100).sum().item()
            total += loss.item() * n
            count += n
            step += 1
            if step % 100 == 0:
                print(f'Epoch {epoch + 1} step {step}: response loss {total / count:.4f}', flush=True)
        model.eval()
        val_total = val_count = 0
        with torch.no_grad():
            for x, y in batches(validation, tok, cfg['context_length'], args.batch_size):
                x, y = x.to(args.device), y.to(args.device)
                logits = model(x)
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction='sum')
                val_total += loss.item()
                val_count += (y != -100).sum().item()
        print(f'Epoch {epoch + 1}: train response loss {total / count:.4f}; validation response loss {val_total / val_count:.4f}', flush=True)
        save_checkpoint(str(args.output), model, optimizer, epoch + 1, val_total / val_count, step)
        print(f'Saved {args.output}', flush=True)


if __name__ == '__main__':
    main()
