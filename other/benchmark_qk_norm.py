"""CUDA forward/backward benchmark for the Q/K shapes used during training."""
import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tinygpt import QKRMSNorm


class NativeQKRMSNorm(QKRMSNorm):
    def forward(self, x):
        # Match the reference precision and epsilon explicitly. Native RMSNorm
        # autocast policies vary by PyTorch release.
        with torch.autocast(device_type=x.device.type, enabled=False):
            return F.rms_norm(x.float(), (x.shape[-1],), self.weight.float(), self.eps).to(x.dtype)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--heads', type=int, default=18)
    parser.add_argument('--context-length', type=int, default=1024)
    parser.add_argument('--head-dim', type=int, default=64)
    parser.add_argument('--warmup', type=int, default=20)
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--precision', choices=['bfloat16', 'float32'], default='bfloat16')
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error('This benchmark requires a CUDA GPU')
    if min(args.batch_size, args.heads, args.context_length, args.head_dim, args.warmup, args.steps) < 1:
        parser.error('Dimensions, warmup and steps must be positive')
    dtype = getattr(torch, args.precision)
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        parser.error('This GPU does not support BF16')
    torch.manual_seed(42)
    print(f'PyTorch {torch.__version__}; {torch.cuda.get_device_name()}; {args.precision}')
    # Match the noncontiguous views returned by the actual combined QKV projection.
    packed = torch.randn(args.batch_size, args.context_length, 3, args.heads,
                         args.head_dim, device='cuda', dtype=dtype)
    x = packed.permute(2, 0, 3, 1, 4)[0].detach().requires_grad_()
    upstream = torch.randn_like(x)
    state = QKRMSNorm(args.head_dim).state_dict()
    reference = None
    for name in ('eager', 'compiled', 'native'):
        module = (NativeQKRMSNorm if name == 'native' else QKRMSNorm)(args.head_dim).cuda()
        module.load_state_dict(state)
        if name == 'compiled':
            module.compile(fullgraph=True)

        def run():
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=dtype == torch.bfloat16):
                y = module(x)
            grads = torch.autograd.grad(y, (x, module.weight), upstream)
            return y, *grads

        result = run()
        if reference is None:
            reference = tuple(t.detach().clone() for t in result)
        else:
            # BF16 rounding and fused reductions are not bitwise identical.
            for label, actual, expected in zip(('output', 'input gradient', 'weight gradient'), result, reference):
                diff = (actual.float() - expected.float()).norm()
                rel = (diff / expected.float().norm().clamp_min(1e-12)).item()
                limit = 0.02 if dtype == torch.bfloat16 else 1e-4
                print(f'{name}: {label} relative L2 error={rel:.3g}')
                if not torch.isfinite(actual).all() or rel > limit:
                    raise RuntimeError(f'{name}: {label} failed numerical check')
        del result
        for _ in range(args.warmup):
            run()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        baseline = torch.cuda.memory_allocated()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(args.steps):
            run()
        end.record()
        end.synchronize()
        extra = (torch.cuda.max_memory_allocated() - baseline) / 1024**2
        print(f'{name}: {start.elapsed_time(end) / args.steps:.3f} ms/forward+backward; extra peak allocated {extra:.1f} MiB')
        del module
    print('Microbenchmark only: confirm any gain with full training-step tokens/s after compilation warmup.')


if __name__ == '__main__':
    main()
