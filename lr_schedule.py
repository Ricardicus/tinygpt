"""Learning rates indexed by completed optimizer updates, independent of epochs."""
from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class WarmupCosine:
    peak_lr: float
    warmup_steps: int
    decay_steps: int
    min_lr_ratio: float = 0.1
    start_step: int = 0

    def __post_init__(self):
        if not math.isfinite(self.peak_lr) or self.peak_lr < 0:
            raise ValueError("Schedule peak learning rate must be finite and nonnegative")
        if not math.isfinite(self.min_lr_ratio) or not 0 <= self.min_lr_ratio <= 1:
            raise ValueError("--min-lr-ratio must be between 0 and 1")
        if self.start_step < 0 or not 0 <= self.warmup_steps < self.decay_steps:
            raise ValueError("Cosine schedule requires 0 <= --warmup-steps < --lr-decay-steps")

    def rate(self, completed_steps):
        """Rate for the next update: warm up to peak, then decay to a floor.

        The first warmup update uses peak/warmup_steps; the final planned update
        uses the floor. Further updates remain at the floor. Without warmup,
        the first update uses the peak (a one-update schedule uses the floor).
        """
        update = max(1, completed_steps - self.start_step + 1)
        if self.warmup_steps and update <= self.warmup_steps:
            return self.peak_lr * update / self.warmup_steps
        first_decay = max(1, self.warmup_steps)
        span = self.decay_steps - first_decay
        progress = min(1.0, max(0.0, (update - first_decay) / span)) if span else 1.0
        factor = self.min_lr_ratio + (1 - self.min_lr_ratio) * (1 + math.cos(math.pi * progress)) / 2
        return self.peak_lr * factor

    def state_dict(self):
        return asdict(self)


def resolve_schedule(args, saved, current_lr, global_step, remaining_steps):
    """Restore the saved timeline; explicit flags override its configuration."""
    mode = args.lr_schedule or ("cosine" if saved else "constant")
    overrides = (args.warmup_steps, args.lr_decay_steps, args.min_lr_ratio)
    if mode == "constant":
        if any(value is not None for value in overrides):
            raise ValueError("Schedule options require --lr-schedule cosine (or a saved cosine schedule)")
        return None
    if saved:
        config = dict(saved)
    else:
        total = args.lr_decay_steps if args.lr_decay_steps is not None else remaining_steps
        config = dict(peak_lr=current_lr, warmup_steps=min(1000, total // 10),
                      decay_steps=total, min_lr_ratio=0.1, start_step=global_step)
    if args.lr_explicit:
        config["peak_lr"] = args.lr
    for flag, key in (("warmup_steps", "warmup_steps"), ("lr_decay_steps", "decay_steps"),
                      ("min_lr_ratio", "min_lr_ratio")):
        value = getattr(args, flag)
        if value is not None:
            config[key] = value
    return WarmupCosine(**config)
