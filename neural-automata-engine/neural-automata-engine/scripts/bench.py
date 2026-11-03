from __future__ import annotations

import argparse
import time
from typing import Callable, Optional, Sequence

from nae import Engine, NeuralWeights


def measure(make: Callable[[], Engine], steps: int, warmup: int = 3) -> float:
    with make() as engine:
        engine.step(warmup)
        start = time.perf_counter()
        engine.step(steps)
        return time.perf_counter() - start


def life_factory(size: int) -> Callable[[], Engine]:
    def make() -> Engine:
        engine = Engine.life(size, size)
        engine.seed_random(7, 0.3)
        return engine
    return make


def neural_factory(size: int, channels: int, hidden: int) -> Callable[[], Engine]:
    weights = NeuralWeights.random(channels, hidden, seed=3)

    def make() -> Engine:
        engine = Engine.neural(size, size, weights)
        engine.seed_random(7, 0.3)
        return engine
    return make


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Throughput benchmark")
    p.add_argument("--sizes", type=int, nargs="+", default=[64, 128, 256, 512, 1024])
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--channels", type=int, default=8)
    p.add_argument("--hidden", type=int, default=32)
    p.add_argument("--rule", choices=("life", "neural", "both"), default="both")
    args = p.parse_args(argv)

    print(f"{'rule':<8}{'size':>8}{'steps':>8}{'seconds':>10}{'Mcells/s':>12}")
    for size in args.sizes:
        jobs = []
        if args.rule in ("life", "both"):
            jobs.append(("life", life_factory(size)))
        if args.rule in ("neural", "both"):
            jobs.append(("neural", neural_factory(size, args.channels, args.hidden)))
        for name, make in jobs:
            elapsed = measure(make, args.steps)
            rate = size * size * args.steps / elapsed / 1e6
            print(f"{name:<8}{size:>8}{args.steps:>8}{elapsed:>10.3f}{rate:>12.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
