#ifndef NAE_GRID_H
#define NAE_GRID_H

#include "nae/types.h"

typedef struct nae_grid {
    uint32_t width;
    uint32_t height;
    uint32_t channels;
    uint32_t halo;
    size_t stride_x;
    size_t stride_y;
    size_t total;
    float *origin;
    float *base;
    void *raw;
} nae_grid;

nae_status nae_grid_init(nae_grid *g, uint32_t width, uint32_t height,
                         uint32_t channels, uint32_t halo);
void nae_grid_free(nae_grid *g);
void nae_grid_fill(nae_grid *g, float value);
void nae_grid_apply_boundary(nae_grid *g, nae_boundary mode);
nae_status nae_grid_copy(nae_grid *dst, const nae_grid *src);

static inline float *nae_grid_at(const nae_grid *g, uint32_t x, uint32_t y)
{
    return g->origin + (size_t)y * g->stride_y + (size_t)x * g->stride_x;
}

static inline int nae_grid_same_shape(const nae_grid *a, const nae_grid *b)
{
    return a->width == b->width && a->height == b->height &&
           a->channels == b->channels && a->halo == b->halo;
}

#endif
