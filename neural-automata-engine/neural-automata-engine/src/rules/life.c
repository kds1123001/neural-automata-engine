#include "nae/rules.h"

static void life_apply(const nae_rule_ctx *ctx, uint32_t row_begin, uint32_t row_end)
{
    const nae_life_params *params = ctx->params;
    const nae_grid *src = ctx->src;
    nae_grid *dst = ctx->dst;
    const float *weights = ctx->kernel->weights;
    const ptrdiff_t *deltas = ctx->deltas;
    const uint32_t count = ctx->kernel->count;
    uint32_t x, y, k;

    for (y = row_begin; y < row_end; ++y) {
        const float *in = nae_grid_at(src, 0, y);
        float *out = nae_grid_at(dst, 0, y);

        for (x = 0; x < src->width; ++x) {
            float acc = 0.0f;
            uint32_t n, mask;

            for (k = 0; k < count; ++k)
                acc += in[deltas[k]] > 0.5f ? weights[k] : 0.0f;

            n = acc > 0.0f ? (uint32_t)(acc + 0.5f) : 0u;
            mask = in[0] > 0.5f ? params->survive_mask : params->birth_mask;
            out[0] = (n < 32u && ((mask >> n) & 1u)) ? 1.0f : 0.0f;

            in += src->stride_x;
            out += dst->stride_x;
        }
    }
}

static const nae_rule life_rule = {
    "life",
    life_apply,
    NULL,
    sizeof(nae_life_params),
    1u,
    1u,
};

const nae_rule *nae_rule_life(void)
{
    return &life_rule;
}
