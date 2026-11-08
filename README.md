# neural-automata-engine

A cellular automaton engine written in C99. It runs Conway's Game of Life and a small neural update rule on a 2D grid. There is also a thin Python layer for drawing frames and running benchmarks, and it uses only the Python standard library. The C code does not depend on a tensor library or any other outside library.

## Build

```
make lib       build/libnae.a and build/libnae.so
make test      C unit tests
make pytest    Python binding tests
make check     both test suites
make bench     prints a throughput table
```

Options:

- `OMP=1` turns on row-chunk parallelism with OpenMP.
- `SAN=1` builds with AddressSanitizer and UndefinedBehaviorSanitizer. Run `make clean` when you switch between sanitizer and normal builds.
- `ARCH=` overrides `-march=native`. By default the code is built for the CPU you compile on, so the result may not run on older CPUs.

The Makefile builds `libnae.so`, which is the Linux shared library format. Other platforms would need changes.

## Layout

```
include/nae/    public headers (types, grid, kernel, rule, rules, engine)
src/            grid, kernel, rule registry, engine
src/rules/      life.c, neural.c
bindings/       flat C API that Python calls through ctypes
scripts/nae/    Python package (Engine, NeuralWeights)
scripts/        render.py, bench.py
tests/          C and Python tests
```

## Design notes

- A grid is one flat row-major float buffer. Channels are interleaved per cell. There is a border (halo) of `kernel.radius` cells around the grid, and the first real cell is 64-byte aligned.
- Edge handling (wrap, zero, or clamp) fills the halo once per step. Because of that, the rule loops never have to check for edges.
- The engine owns two grids and swaps them every step.
- Kernel offsets are turned into linear index offsets once, based on the grid strides. A rule reads its neighbors with `cell[deltas[k]]`.
- If `nae_engine_init` succeeds, the engine takes ownership of `cfg.kernel`. If it fails, the caller keeps it.
- Rule parameters are copied into the engine shallowly. For the neural rule, that means the caller has to keep the weight memory alive until it is done, and free it with `nae_neural_params_free`.

## Adding a rule

1. Write `void apply(const nae_rule_ctx *, uint32_t row_begin, uint32_t row_end)`. It reads `ctx->src` and writes `ctx->dst`.
2. Define a `nae_rule` with a name, the size of its params, its channel requirements, and optionally a validator.
3. Declare the getter in `rule.h` and add it to the registry in `src/rule.c`.

## The neural rule

For each channel, a cell's perception is `[self, sum(w*v), sum(w*dx*v), sum(w*dy*v)]`, summed over the kernel. Here `v` is the neighbor's value, `w` is the kernel weight, and `dx` and `dy` are that neighbor's offset. This means any kernel can be used as a perception filter.

The perception goes through a hidden layer with ReLU and then a linear output layer. The output is an update, applied like this:

```
next = state + step_size * delta
```

The result can optionally be clipped to `[-state_clip, state_clip]`.

## Python

```
python3 scripts/render.py --rule life
python3 scripts/render.py --rule neural --channels 4 --hidden 16 --pgm out/
python3 scripts/bench.py --sizes 256 512 1024
```

Weights for the neural rule load from a JSON file with the keys `channels`, `hidden`, `step_size`, `state_clip`, `w1`, `b1`, `w2`, and `b2`.

## What this does not do

- **It does not train anything.** The neural rule runs a network, it doesn't learn one. The weights have to come from somewhere else and be saved as JSON in the format above.
- **No benchmark numbers are published here.** `make bench` prints a table for your own machine. I make no claims about speed compared to other libraries.
- **The README doesn't report test results.** The C and Python tests exist, and you can run them with `make check`, and with `SAN=1` for the sanitizer build.
- **Platforms.** The build is set up for Linux, as described above.

## References

- Conway's Game of Life.
- Mordvintsev, Randazzo, Niklasson, and Levin, "Growing Neural Cellular Automata" (Distill, 2020). The neural rule is in the same family of models.

## Credits

Made by kds1123001.
okay with this readme updating and allat dont yall think i have stopped making jokes typeshit in my code being professional and allat?
## License

MIT. See `LICENSE`.
