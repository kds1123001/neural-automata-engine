# neural-automata-engine

Tensor-free spatial state evolution in C99, with a thin stdlib-only Python layer for visualization and harness scripts.

## Build

    make lib            build/libnae.a and build/libnae.so
    make test           C unit tests
    make pytest         Python binding tests
    make check          both
    make bench          throughput table

Options: `OMP=1` enables row-chunk parallelism via OpenMP, `SAN=1` builds with ASan and UBSan (run `make clean` when switching), `ARCH=` overrides `-march=native`.

## Layout

    include/nae/    public headers (types, grid, kernel, rule, rules, engine)
    src/            grid, kernel, rule registry, engine
    src/rules/      life.c, neural.c
    bindings/       flat C API consumed by ctypes
    scripts/nae/    Python package (Engine, NeuralWeights)
    scripts/        render.py, bench.py
    tests/          C and Python tests

## Design notes

- Grids are one flat row-major float buffer, channels interleaved per cell, with a halo of `kernel.radius` cells and a 64-byte aligned origin.
- Boundary conditions (wrap, zero, clamp) fill the halo once per step, so rule loops never branch on edges.
- The engine owns two grids and swaps them each step.
- Kernel offsets are converted once into linear deltas for the grid strides; rules index neighbours with `cell[deltas[k]]`.
- On a successful `nae_engine_init` the engine takes ownership of `cfg.kernel`. On failure the caller keeps it.
- Rule params are copied shallowly into the engine. For the neural rule, the caller keeps the weight storage alive (`nae_neural_params_free`).

## Adding a rule

1. Implement `void apply(const nae_rule_ctx *, uint32_t row_begin, uint32_t row_end)`, reading `ctx->src` and writing `ctx->dst`.
2. Define a `nae_rule` with name, params size and channel requirements, and an optional validator.
3. Declare the getter in `rule.h` and add it to the registry in `src/rule.c`.

## Neural rule

Perception per channel is `[self, sum(w*v), sum(w*dx*v), sum(w*dy*v)]` over the kernel, so any kernel doubles as a perception filter. A ReLU hidden layer and a linear output produce a residual update: `next = state + step_size * delta`, optionally clipped to `[-state_clip, state_clip]`.

## Python

    python3 scripts/render.py --rule life
    python3 scripts/render.py --rule neural --channels 4 --hidden 16 --pgm out/
    python3 scripts/bench.py --sizes 256 512 1024

Weights load from JSON with keys `channels`, `hidden`, `step_size`, `state_clip`, `w1`, `b1`, `w2`, `b2`.

## Credits

  everythig create by me ye
  no me use ai, ai bug ai bad i no like ai 
