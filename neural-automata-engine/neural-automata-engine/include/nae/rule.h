#ifndef NAE_RULE_H
#define NAE_RULE_H

#include "nae/grid.h"
#include "nae/kernel.h"

typedef struct nae_rule_ctx {
    const nae_grid *src;
    nae_grid *dst;
    const nae_kernel *kernel;
    const ptrdiff_t *deltas;
    const void *params;
} nae_rule_ctx;

typedef void (*nae_rule_fn)(const nae_rule_ctx *ctx, uint32_t row_begin, uint32_t row_end);
typedef nae_status (*nae_rule_validate_fn)(const void *params, uint32_t channels);

typedef struct nae_rule {
    const char *name;
    nae_rule_fn apply;
    nae_rule_validate_fn validate;
    size_t params_size;
    uint32_t in_channels;
    uint32_t out_channels;
} nae_rule;

const nae_rule *nae_rule_find(const char *name);
const nae_rule *nae_rule_life(void);
const nae_rule *nae_rule_neural(void);

#endif
