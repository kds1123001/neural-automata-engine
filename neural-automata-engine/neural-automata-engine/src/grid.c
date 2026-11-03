#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "nae/grid.h"

static int checked_mul(size_t a, size_t b, size_t *out)
{
    if (a != 0 && b > SIZE_MAX / a)
        return 0;
    *out = a * b;
    return 1;
}

static int32_t wrap_index(int32_t i, int32_t n)
{
    int32_t m = i % n;
    return m < 0 ? m + n : m;
}

static int32_t clamp_index(int32_t i, int32_t n)
{
    if (i < 0)
        return 0;
    if (i >= n)
        return n - 1;
    return i;
}

static int32_t map_index(int32_t i, int32_t n, nae_boundary mode)
{
    switch (mode) {
    case NAE_BOUNDARY_WRAP:
        return wrap_index(i, n);
    case NAE_BOUNDARY_CLAMP:
        return clamp_index(i, n);
    default:
        return -1;
    }
}

static float *cell_at(const nae_grid *g, int32_t x, int32_t y)
{
    return g->origin + (ptrdiff_t)y * (ptrdiff_t)g->stride_y +
           (ptrdiff_t)x * (ptrdiff_t)g->stride_x;
}

nae_status nae_grid_init(nae_grid *g, uint32_t width, uint32_t height,
                         uint32_t channels, uint32_t halo)
{
    size_t span_x, rows, stride_y, total, bytes, origin_off;
    uintptr_t origin_addr;
    void *raw;

    if (!g)
        return NAE_ERR_ARG;
    memset(g, 0, sizeof *g);

    if (width == 0 || height == 0 || channels == 0)
        return NAE_ERR_ARG;
    if ((uint64_t)width + 2u * (uint64_t)halo > INT32_MAX ||
        (uint64_t)height + 2u * (uint64_t)halo > INT32_MAX)
        return NAE_ERR_ARG;

    span_x = (size_t)width + 2u * (size_t)halo;
    rows = (size_t)height + 2u * (size_t)halo;

    if (!checked_mul(span_x, channels, &stride_y) ||
        !checked_mul(stride_y, rows, &total) ||
        !checked_mul(total, sizeof(float), &bytes) ||
        bytes > SIZE_MAX - NAE_ALIGNMENT)
        return NAE_ERR_OVERFLOW;

    origin_off = ((size_t)halo * stride_y + (size_t)halo * channels) * sizeof(float);

    raw = calloc(1, bytes + NAE_ALIGNMENT);
    if (!raw)
        return NAE_ERR_ALLOC;

    origin_addr = ((uintptr_t)raw + origin_off + (NAE_ALIGNMENT - 1u)) &
                  ~(uintptr_t)(NAE_ALIGNMENT - 1u);

    g->width = width;
    g->height = height;
    g->channels = channels;
    g->halo = halo;
    g->stride_x = channels;
    g->stride_y = stride_y;
    g->total = total;
    g->raw = raw;
    g->base = (float *)(origin_addr - origin_off);
    g->origin = (float *)origin_addr;
    return NAE_OK;
}

void nae_grid_free(nae_grid *g)
{
    if (!g)
        return;
    free(g->raw);
    memset(g, 0, sizeof *g);
}

void nae_grid_fill(nae_grid *g, float value)
{
    size_t i;

    for (i = 0; i < g->total; ++i)
        g->base[i] = value;
}

void nae_grid_apply_boundary(nae_grid *g, nae_boundary mode)
{
    const int32_t w = (int32_t)g->width;
    const int32_t h = (int32_t)g->height;
    const int32_t hl = (int32_t)g->halo;
    const size_t cell_bytes = (size_t)g->channels * sizeof(float);
    int32_t x, y;

    for (y = -hl; y < h + hl; ++y) {
        const int inside_y = y >= 0 && y < h;
        const int32_t sy = inside_y ? y : map_index(y, h, mode);

        for (x = -hl; x < w + hl; ++x) {
            int32_t sx;
            float *dst;

            if (inside_y && x == 0) {
                x = w - 1;
                continue;
            }

            sx = map_index(x, w, mode);
            dst = cell_at(g, x, y);
            if (sx < 0 || sy < 0)
                memset(dst, 0, cell_bytes);
            else
                memcpy(dst, cell_at(g, sx, sy), cell_bytes);
        }
    }
}

nae_status nae_grid_copy(nae_grid *dst, const nae_grid *src)
{
    if (!dst || !src)
        return NAE_ERR_ARG;
    if (!nae_grid_same_shape(dst, src))
        return NAE_ERR_SHAPE;
    memcpy(dst->base, src->base, src->total * sizeof(float));
    return NAE_OK;
}
