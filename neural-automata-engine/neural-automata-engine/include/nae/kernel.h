#ifndef NAE_KERNEL_H
#define NAE_KERNEL_H

#include "nae/types.h"

#define NAE_KERNEL_MAX_RADIUS 1024u

typedef struct nae_offset {
    int16_t dx;
    int16_t dy;
} nae_offset;

typedef struct nae_kernel {
    nae_offset *offsets;
    float *weights;
    uint32_t count;
    uint32_t radius;
} nae_kernel;

nae_status nae_kernel_moore(nae_kernel *k, uint32_t radius, int include_center);
nae_status nae_kernel_von_neumann(nae_kernel *k, uint32_t radius, int include_center);
nae_status nae_kernel_custom(nae_kernel *k, const nae_offset *offsets,
                             const float *weights, uint32_t count);
void nae_kernel_free(nae_kernel *k);

#endif
