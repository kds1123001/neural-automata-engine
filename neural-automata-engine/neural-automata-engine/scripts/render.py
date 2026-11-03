from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List, Optional, Sequence

from nae import Engine, NeuralWeights

SHADES = " .:-=+*#%@"
HALF_BLOCKS = {(0, 0): " ", (1, 0): "\u2580", (0, 1): "\u2584", (1, 1): "\u2588"}


def plane(frame: Sequence[float], channels: int, channel: int) -> List[float]:
    return list(frame[channel::channels])


def shade(value: float) -> str:
    if value != value:
        return SHADES[0]
    top = len(SHADES) - 1
    return SHADES[min(top, max(0, int(value * top + 0.5)))]


def render_blocks(values: Sequence[float], width: int, height: int) -> str:
    lines = []
    for y in range(0, height, 2):
        upper = values[y * width:(y + 1) * width]
        lower = values[(y + 1) * width:(y + 2) * width] if y + 1 < height else [0.0] * width
        lines.append("".join(HALF_BLOCKS[(int(a > 0.5), int(b > 0.5))] for a, b in zip(upper, lower)))
    return "\n".join(lines)


def render_shades(values: Sequence[float], width: int, height: int) -> str:
    rows = (values[y * width:(y + 1) * width] for y in range(height))
    return "\n".join("".join(shade(v) for v in row) for row in rows)


def write_pgm(path: Path, values: Sequence[float], width: int, height: int) -> None:
    pixels = bytes(
        min(255, max(0, int(v * 255.0 + 0.5))) if v == v else 0 for v in values
    )
    with open(path, "wb") as fh:
        fh.write(f"P5\n{width} {height}\n255\n".encode("ascii"))
        fh.write(pixels)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run a neural-automata-engine simulation")
    p.add_argument("--rule", choices=("life", "neural"), default="life")
    p.add_argument("--width", type=int, default=96)
    p.add_argument("--height", type=int, default=48)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--delay", type=float, default=0.04)
    p.add_argument("--boundary", choices=("wrap", "zero", "clamp"), default="wrap")
    p.add_argument("--kernel", choices=("moore", "von_neumann"), default="moore")
    p.add_argument("--radius", type=int, default=1)
    p.add_argument("--density", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--weights", type=Path)
    p.add_argument("--channels", type=int, default=4)
    p.add_argument("--hidden", type=int, default=16)
    p.add_argument("--channel", type=int, default=0)
    p.add_argument("--pgm", type=Path)
    p.add_argument("--every", type=int, default=1)
    p.add_argument("--quiet", action="store_true")
    return p.parse_args(argv)


def build_engine(args: argparse.Namespace) -> Engine:
    if args.rule == "life":
        engine = Engine.life(
            args.width, args.height,
            boundary=args.boundary, kernel=args.kernel, radius=args.radius,
        )
        engine.seed_random(args.seed, args.density)
        return engine

    if args.weights:
        weights = NeuralWeights.load(args.weights)
    else:
        weights = NeuralWeights.random(args.channels, args.hidden, seed=args.seed)
    engine = Engine.neural(
        args.width, args.height, weights,
        boundary=args.boundary, kernel=args.kernel, radius=args.radius,
    )
    engine.seed_point(args.width // 2, args.height // 2, 1.0)
    return engine


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    engine = build_engine(args)
    draw = render_blocks if args.rule == "life" else render_shades
    channel = min(max(args.channel, 0), engine.channels - 1)
    out_dir = args.pgm
    live = out_dir is None and not args.quiet

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
    if live:
        sys.stdout.write("\x1b[?25l\x1b[2J")

    try:
        for step in range(args.steps + 1):
            if step:
                engine.step()
            values = plane(engine.read(), engine.channels, channel)
            if out_dir is not None and step % max(args.every, 1) == 0:
                write_pgm(out_dir / f"frame_{step:05d}.pgm", values, args.width, args.height)
            if live:
                sys.stdout.write("\x1b[H" + draw(values, args.width, args.height))
                sys.stdout.write(f"\ngeneration {engine.generation}\n")
                sys.stdout.flush()
                time.sleep(args.delay)
    except KeyboardInterrupt:
        pass
    finally:
        if live:
            sys.stdout.write("\x1b[?25h")
            sys.stdout.flush()
        engine.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
