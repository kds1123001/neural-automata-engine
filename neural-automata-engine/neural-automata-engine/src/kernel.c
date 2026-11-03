#include <stdlib.h>
#include <string.h>

#include "nae/kernel.h"

#define MAX_TAPS (1u << 24)

typedef int (*shape_fn)(int32_t dx, int32_t dy, int32_t radius);

static int in_moore(int32_t dx, int32_t dy, int32_t radius)
{
    return abs(dx) <= radius && abs(dy) <= radius;
}

static int in_von_neumann(int32_t dx, int32_t dy, int32_t radius)
{
    return abs(dx) + abs(dy) <= radius;
}

static nae_status alloc_arrays(nae_kernel *k, uint32_t count)
{
    k->offsets = malloc((size_t)count * sizeof *k->offsets);
    k->weights = malloc((size_t)count * sizeof *k->weights);
    if (!k->offsets || !k->weights) {
        free(k->offsets);
        free(k->weights);
        memset(k, 0, sizeof *k);
        return NAE_ERR_ALLOC;
    }
    k->count = count;
    return NAE_OK;
}

static int skip_cell(int32_t dx, int32_t dy, int include_center)
{
    return dx == 0 && dy == 0 && !include_center;
}

static nae_status build_shape(nae_kernel *k, uint32_t radius, int include_center,
                              shape_fn inside)
{
    const int32_t r = (int32_t)radius;
    uint32_t count = 0;
    uint32_t n = 0;
    nae_status st;
    int32_t dx, dy;

    if (!k)
        return NAE_ERR_ARG;
    memset(k, 0, sizeof *k);
    if (radius == 0 || radius > NAE_KERNEL_MAX_RADIUS)
        return NAE_ERR_ARG;

    for (dy = -r; dy <= r; ++dy)
        for (dx = -r; dx <= r; ++dx)
            if (inside(dx, dy, r) && !skip_cell(dx, dy, include_center))
                count++;

    st = alloc_arrays(k, count);
    if (st != NAE_OK)
        return st;

    for (dy = -r; dy <= r; ++dy) {
        for (dx = -r; dx <= r; ++dx) {
            if (!inside(dx, dy, r) || skip_cell(dx, dy, include_center))
                continue;
            k->offsets[n].dx = (int16_t)dx;
            k->offsets[n].dy = (int16_t)dy;
            k->weights[n] = 1.0f;
            n++;
        }
    }
    k->radius = radius;
    return NAE_OK;
}

nae_status nae_kernel_moore(nae_kernel *k, uint32_t radius, int include_center)
{
    return build_shape(k, radius, include_center, in_moore);
}

nae_status nae_kernel_von_neumann(nae_kernel *k, uint32_t radius, int include_center)
{
    return build_shape(k, radius, include_center, in_von_neumann);
}

nae_status nae_kernel_custom(nae_kernel *k, const nae_offset *offsets,
                             const float *weights, uint32_t count)
{
    uint32_t i, reach;
    nae_status st;

    if (!k)
        return NAE_ERR_ARG;
    memset(k, 0, sizeof *k);
    if (!offsets || !weights || count == 0 || count > MAX_TAPS)
        return NAE_ERR_ARG;

    st = alloc_arrays(k, count);
    if (st != NAE_OK)
        return st;

    memcpy(k->offsets, offsets, (size_t)count * sizeof *offsets);
    memcpy(k->weights, weights, (size_t)count * sizeof *weights);

    k->radius = 0;
    for (i = 0; i < count; ++i) {
        reach = (uint32_t)abs(offsets[i].dx);
        if ((uint32_t)abs(offsets[i].dy) > reach)
            reach = (uint32_t)abs(offsets[i].dy);
        if (reach > k->radius)
            k->radius = reach;
    }
    return NAE_OK;
}

void nae_kernel_free(nae_kernel *k)
{
    if (!k)
        return;
    free(k->offsets);
    free(k->weights);
    memset(k, 0, sizeof *k);
}
