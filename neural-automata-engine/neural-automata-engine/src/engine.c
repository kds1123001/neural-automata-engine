#include <stdlib.h>
#include <string.h>

#include "nae/engine.h"

#define ROWS_PER_CHUNK 16u

static nae_status check_config(const nae_engine_config *cfg)
{
    const nae_rule *rule;

    if (!cfg || !cfg->rule)
        return NAE_ERR_ARG;
    rule = cfg->rule;

    if (cfg->width == 0 || cfg->height == 0 || cfg->channels == 0)
        return NAE_ERR_ARG;
    if ((unsigned)cfg->boundary > (unsigned)NAE_BOUNDARY_CLAMP)
        return NAE_ERR_ARG;
    if (!cfg->kernel.offsets || !cfg->kernel.weights || cfg->kernel.count == 0)
        return NAE_ERR_ARG;
    if (rule->in_channels && rule->in_channels != cfg->channels)
        return NAE_ERR_SHAPE;
    if (rule->out_channels && rule->out_channels != cfg->channels)
        return NAE_ERR_SHAPE;
    if (rule->params_size && !cfg->params)
        return NAE_ERR_ARG;
    return NAE_OK;
}

static nae_status copy_params(nae_engine *e, const nae_engine_config *cfg)
{
    const nae_rule *rule = cfg->rule;
    nae_status st;

    if (rule->validate) {
        st = rule->validate(cfg->params, cfg->channels);
        if (st != NAE_OK)
            return st;
    }
    if (rule->params_size == 0)
        return NAE_OK;

    e->params = malloc(rule->params_size);
    if (!e->params)
        return NAE_ERR_ALLOC;
    memcpy(e->params, cfg->params, rule->params_size);
    return NAE_OK;
}

static nae_status build_deltas(nae_engine *e, const nae_kernel *k)
{
    const nae_grid *g = &e->buffers[0];
    uint32_t i;

    e->deltas = malloc((size_t)k->count * sizeof *e->deltas);
    if (!e->deltas)
        return NAE_ERR_ALLOC;

    for (i = 0; i < k->count; ++i) {
        e->deltas[i] = (ptrdiff_t)k->offsets[i].dy * (ptrdiff_t)g->stride_y +
                       (ptrdiff_t)k->offsets[i].dx * (ptrdiff_t)g->stride_x;
    }
    return NAE_OK;
}

nae_status nae_engine_init(nae_engine *e, const nae_engine_config *cfg)
{
    nae_status st;

    if (!e)
        return NAE_ERR_ARG;
    st = check_config(cfg);
    if (st != NAE_OK)
        return st;

    memset(e, 0, sizeof *e);

    st = nae_grid_init(&e->buffers[0], cfg->width, cfg->height, cfg->channels,
                       cfg->kernel.radius);
    if (st != NAE_OK)
        goto fail;
    st = nae_grid_init(&e->buffers[1], cfg->width, cfg->height, cfg->channels,
                       cfg->kernel.radius);
    if (st != NAE_OK)
        goto fail;
    st = copy_params(e, cfg);
    if (st != NAE_OK)
        goto fail;
    st = build_deltas(e, &cfg->kernel);
    if (st != NAE_OK)
        goto fail;

    e->kernel = cfg->kernel;
    e->rule = cfg->rule;
    e->boundary = cfg->boundary;
    return NAE_OK;

fail:
    nae_engine_free(e);
    return st;
}

void nae_engine_free(nae_engine *e)
{
    if (!e)
        return;
    nae_grid_free(&e->buffers[0]);
    nae_grid_free(&e->buffers[1]);
    nae_kernel_free(&e->kernel);
    free(e->deltas);
    free(e->params);
    memset(e, 0, sizeof *e);
}

void nae_engine_step(nae_engine *e)
{
    nae_grid *src = &e->buffers[e->front];
    nae_grid *dst = &e->buffers[e->front ^ 1u];
    const int chunks = (int)((src->height + ROWS_PER_CHUNK - 1u) / ROWS_PER_CHUNK);
    nae_rule_ctx ctx;
    int c;

    nae_grid_apply_boundary(src, e->boundary);

    ctx.src = src;
    ctx.dst = dst;
    ctx.kernel = &e->kernel;
    ctx.deltas = e->deltas;
    ctx.params = e->params;

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (c = 0; c < chunks; ++c) {
        uint32_t begin = (uint32_t)c * ROWS_PER_CHUNK;
        uint32_t end = begin + ROWS_PER_CHUNK;

        if (end > src->height)
            end = src->height;
        e->rule->apply(&ctx, begin, end);
    }

    e->front ^= 1u;
    e->generation++;
}

void nae_engine_run(nae_engine *e, uint64_t steps)
{
    uint64_t i;

    for (i = 0; i < steps; ++i)
        nae_engine_step(e);
}

nae_grid *nae_engine_state(nae_engine *e)
{
    return &e->buffers[e->front];
}

const nae_grid *nae_engine_state_const(const nae_engine *e)
{
    return &e->buffers[e->front];
}
