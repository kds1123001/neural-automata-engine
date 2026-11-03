#include <stdlib.h>
#include <string.h>

#include "nae/engine.h"
#include "nae/rules.h"

typedef struct nae_handle {
    nae_engine engine;
    nae_neural_params neural;
} nae_handle;

enum {
    KERNEL_MOORE = 0,
    KERNEL_VON_NEUMANN = 1
};

static nae_status make_kernel(nae_kernel *k, int kind, uint32_t radius, int include_center)
{
    switch (kind) {
    case KERNEL_MOORE:
        return nae_kernel_moore(k, radius, include_center);
    case KERNEL_VON_NEUMANN:
        return nae_kernel_von_neumann(k, radius, include_center);
    default:
        return NAE_ERR_ARG;
    }
}

static void report(int *status, nae_status value)
{
    if (status)
        *status = (int)value;
}

static int valid_boundary(int boundary)
{
    return boundary >= (int)NAE_BOUNDARY_WRAP && boundary <= (int)NAE_BOUNDARY_CLAMP;
}

static nae_handle *boot(nae_handle *h, nae_engine_config *cfg, int kind, uint32_t radius,
                        int include_center, int *status)
{
    nae_status st = make_kernel(&cfg->kernel, kind, radius, include_center);

    if (st == NAE_OK) {
        st = nae_engine_init(&h->engine, cfg);
        if (st != NAE_OK)
            nae_kernel_free(&cfg->kernel);
    }
    if (st != NAE_OK) {
        nae_neural_params_free(&h->neural);
        free(h);
        report(status, st);
        return NULL;
    }
    report(status, NAE_OK);
    return h;
}

static uint64_t next_u64(uint64_t *state)
{
    uint64_t x = *state;

    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *state = x;
    return x * 0x2545F4914F6CDD1Dull;
}

static float next_unit(uint64_t *state)
{
    return (float)(next_u64(state) >> 40) * (1.0f / 16777216.0f);
}

static size_t frame_count(const nae_handle *h)
{
    const nae_grid *g = nae_engine_state_const(&h->engine);

    return (size_t)g->width * g->height * g->channels;
}

nae_handle *nae_capi_create_life(uint32_t width, uint32_t height, int boundary,
                                 int kernel_kind, uint32_t radius, uint32_t birth_mask,
                                 uint32_t survive_mask, int *status)
{
    nae_life_params params;
    nae_engine_config cfg;
    nae_handle *h;

    if (!valid_boundary(boundary)) {
        report(status, NAE_ERR_ARG);
        return NULL;
    }
    h = calloc(1, sizeof *h);
    if (!h) {
        report(status, NAE_ERR_ALLOC);
        return NULL;
    }

    params.birth_mask = birth_mask;
    params.survive_mask = survive_mask;

    memset(&cfg, 0, sizeof cfg);
    cfg.width = width;
    cfg.height = height;
    cfg.channels = 1;
    cfg.boundary = (nae_boundary)boundary;
    cfg.rule = nae_rule_life();
    cfg.params = &params;
    return boot(h, &cfg, kernel_kind, radius, 0, status);
}

nae_handle *nae_capi_create_neural(uint32_t width, uint32_t height, uint32_t channels,
                                   int boundary, int kernel_kind, uint32_t radius,
                                   int include_center, uint32_t hidden, float step_size,
                                   float state_clip, const float *w1, const float *b1,
                                   const float *w2, const float *b2, int *status)
{
    nae_engine_config cfg;
    nae_status st;
    nae_handle *h;
    size_t n_w1, n_w2;

    if (!valid_boundary(boundary) || !w1 || !b1 || !w2 || !b2) {
        report(status, NAE_ERR_ARG);
        return NULL;
    }
    h = calloc(1, sizeof *h);
    if (!h) {
        report(status, NAE_ERR_ALLOC);
        return NULL;
    }

    st = nae_neural_params_init(&h->neural, channels, hidden);
    if (st != NAE_OK) {
        free(h);
        report(status, st);
        return NULL;
    }

    n_w1 = (size_t)hidden * channels * NAE_NEURAL_PERCEPTION;
    n_w2 = (size_t)channels * hidden;
    memcpy(h->neural.w1, w1, n_w1 * sizeof *w1);
    memcpy(h->neural.b1, b1, (size_t)hidden * sizeof *b1);
    memcpy(h->neural.w2, w2, n_w2 * sizeof *w2);
    memcpy(h->neural.b2, b2, (size_t)channels * sizeof *b2);
    h->neural.step_size = step_size;
    h->neural.state_clip = state_clip;

    memset(&cfg, 0, sizeof cfg);
    cfg.width = width;
    cfg.height = height;
    cfg.channels = channels;
    cfg.boundary = (nae_boundary)boundary;
    cfg.rule = nae_rule_neural();
    cfg.params = &h->neural;
    return boot(h, &cfg, kernel_kind, radius, include_center, status);
}

void nae_capi_destroy(nae_handle *h)
{
    if (!h)
        return;
    nae_engine_free(&h->engine);
    nae_neural_params_free(&h->neural);
    free(h);
}

void nae_capi_step(nae_handle *h, uint64_t steps)
{
    nae_engine_run(&h->engine, steps);
}

uint64_t nae_capi_generation(const nae_handle *h)
{
    return h->engine.generation;
}

uint32_t nae_capi_width(const nae_handle *h)
{
    return nae_engine_state_const(&h->engine)->width;
}

uint32_t nae_capi_height(const nae_handle *h)
{
    return nae_engine_state_const(&h->engine)->height;
}

uint32_t nae_capi_channels(const nae_handle *h)
{
    return nae_engine_state_const(&h->engine)->channels;
}

int nae_capi_read(const nae_handle *h, float *out, size_t count)
{
    const nae_grid *g = nae_engine_state_const(&h->engine);
    const size_t row = (size_t)g->width * g->channels;
    uint32_t y;

    if (!out)
        return (int)NAE_ERR_ARG;
    if (count != frame_count(h))
        return (int)NAE_ERR_SHAPE;

    for (y = 0; y < g->height; ++y)
        memcpy(out + (size_t)y * row, nae_grid_at(g, 0, y), row * sizeof *out);
    return (int)NAE_OK;
}

int nae_capi_write(nae_handle *h, const float *in, size_t count)
{
    nae_grid *g = nae_engine_state(&h->engine);
    const size_t row = (size_t)g->width * g->channels;
    uint32_t y;

    if (!in)
        return (int)NAE_ERR_ARG;
    if (count != frame_count(h))
        return (int)NAE_ERR_SHAPE;

    for (y = 0; y < g->height; ++y)
        memcpy(nae_grid_at(g, 0, y), in + (size_t)y * row, row * sizeof *in);
    return (int)NAE_OK;
}

void nae_capi_seed_random(nae_handle *h, uint64_t seed, float density)
{
    nae_grid *g = nae_engine_state(&h->engine);
    const size_t row = (size_t)g->width * g->channels;
    uint64_t state = seed ? seed : 0x9E3779B97F4A7C15ull;
    uint32_t y;
    size_t i;

    for (y = 0; y < g->height; ++y) {
        float *line = nae_grid_at(g, 0, y);

        for (i = 0; i < row; ++i)
            line[i] = next_unit(&state) < density ? 1.0f : 0.0f;
    }
}

int nae_capi_seed_point(nae_handle *h, uint32_t x, uint32_t y, float value)
{
    nae_grid *g = nae_engine_state(&h->engine);
    uint32_t row, c;

    if (x >= g->width || y >= g->height)
        return (int)NAE_ERR_ARG;

    for (row = 0; row < g->height; ++row)
        memset(nae_grid_at(g, 0, row), 0, (size_t)g->width * g->channels * sizeof(float));
    for (c = 0; c < g->channels; ++c)
        nae_grid_at(g, x, y)[c] = value;
    return (int)NAE_OK;
}

const char *nae_capi_status_name(int status)
{
    switch (status) {
    case NAE_OK:
        return "ok";
    case NAE_ERR_ARG:
        return "invalid argument";
    case NAE_ERR_ALLOC:
        return "allocation failed";
    case NAE_ERR_SHAPE:
        return "shape mismatch";
    case NAE_ERR_OVERFLOW:
        return "size overflow";
    default:
        return "unknown error";
    }
}
