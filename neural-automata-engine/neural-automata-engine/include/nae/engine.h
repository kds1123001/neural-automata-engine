#ifndef NAE_ENGINE_H
#define NAE_ENGINE_H

#include "nae/rule.h"

typedef struct nae_engine {
    nae_grid buffers[2];
    uint32_t front;
    nae_kernel kernel;
    ptrdiff_t *deltas;
    const nae_rule *rule;
    void *params;
    nae_boundary boundary;
    uint64_t generation;
} nae_engine;

typedef struct nae_engine_config {
    uint32_t width;
    uint32_t height;
    uint32_t channels;
    nae_boundary boundary;
    const nae_rule *rule;
    const void *params;
    nae_kernel kernel;
} nae_engine_config;

nae_status nae_engine_init(nae_engine *e, const nae_engine_config *cfg);
void nae_engine_free(nae_engine *e);
void nae_engine_step(nae_engine *e);
void nae_engine_run(nae_engine *e, uint64_t steps);
nae_grid *nae_engine_state(nae_engine *e);
const nae_grid *nae_engine_state_const(const nae_engine *e);

#endif
