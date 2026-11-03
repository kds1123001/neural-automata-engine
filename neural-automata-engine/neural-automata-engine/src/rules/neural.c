#include <stdlib.h>

#include "nae/rules.h"

static void gather_perception(float *out, const float *cell, const nae_rule_ctx *ctx,
                              uint32_t channels)
{
    const nae_kernel *kernel = ctx->kernel;
    uint32_t k, c;

    for (c = 0; c < channels; ++c) {
        float *slot = out + (size_t)c * NAE_NEURAL_PERCEPTION;
        slot[0] = cell[c];
        slot[1] = 0.0f;
        slot[2] = 0.0f;
        slot[3] = 0.0f;
    }

    for (k = 0; k < kernel->count; ++k) {
        const float *nb = cell + ctx->deltas[k];
        const float w = kernel->weights[k];
        const float wx = w * (float)kernel->offsets[k].dx;
        const float wy = w * (float)kernel->offsets[k].dy;

        for (c = 0; c < channels; ++c) {
            float *slot = out + (size_t)c * NAE_NEURAL_PERCEPTION;
            const float v = nb[c];

            slot[1] += w * v;
            slot[2] += wx * v;
            slot[3] += wy * v;
        }
    }
}

static void neural_apply(const nae_rule_ctx *ctx, uint32_t row_begin, uint32_t row_end)
{
    const nae_neural_params *p = ctx->params;
    const nae_grid *src = ctx->src;
    nae_grid *dst = ctx->dst;
    const uint32_t channels = src->channels;
    const uint32_t pdim = channels * NAE_NEURAL_PERCEPTION;
    const uint32_t hidden = p->hidden;
    float perception[NAE_NEURAL_MAX_CHANNELS * NAE_NEURAL_PERCEPTION];
    float act[NAE_NEURAL_MAX_HIDDEN];
    uint32_t x, y, c, i, j;

    for (y = row_begin; y < row_end; ++y) {
        const float *in = nae_grid_at(src, 0, y);
        float *out = nae_grid_at(dst, 0, y);

        for (x = 0; x < src->width; ++x) {
            gather_perception(perception, in, ctx, channels);

            for (j = 0; j < hidden; ++j) {
                const float *row = p->w1 + (size_t)j * pdim;
                float acc = p->b1[j];

                for (i = 0; i < pdim; ++i)
                    acc += row[i] * perception[i];
                act[j] = acc > 0.0f ? acc : 0.0f;
            }

            for (c = 0; c < channels; ++c) {
                const float *row = p->w2 + (size_t)c * hidden;
                float acc = p->b2[c];
                float v;

                for (j = 0; j < hidden; ++j)
                    acc += row[j] * act[j];

                v = in[c] + p->step_size * acc;
                if (p->state_clip > 0.0f) {
                    if (v > p->state_clip)
                        v = p->state_clip;
                    else if (v < -p->state_clip)
                        v = -p->state_clip;
                }
                out[c] = v;
            }

            in += src->stride_x;
            out += dst->stride_x;
        }
    }
}

static nae_status neural_validate(const void *params, uint32_t channels)
{
    const nae_neural_params *p = params;

    if (!p)
        return NAE_ERR_ARG;
    if (p->channels != channels)
        return NAE_ERR_SHAPE;
    if (channels == 0 || channels > NAE_NEURAL_MAX_CHANNELS)
        return NAE_ERR_SHAPE;
    if (p->hidden == 0 || p->hidden > NAE_NEURAL_MAX_HIDDEN)
        return NAE_ERR_SHAPE;
    if (!p->w1 || !p->b1 || !p->w2 || !p->b2)
        return NAE_ERR_ARG;
    return NAE_OK;
}

nae_status nae_neural_params_init(nae_neural_params *p, uint32_t channels, uint32_t hidden)
{
    size_t n_w1, n_w2, total;
    float *block;

    if (!p)
        return NAE_ERR_ARG;
    p->channels = 0;
    p->hidden = 0;
    p->step_size = 0.0f;
    p->state_clip = 0.0f;
    p->w1 = p->b1 = p->w2 = p->b2 = NULL;
    p->storage = NULL;

    if (channels == 0 || channels > NAE_NEURAL_MAX_CHANNELS)
        return NAE_ERR_ARG;
    if (hidden == 0 || hidden > NAE_NEURAL_MAX_HIDDEN)
        return NAE_ERR_ARG;

    n_w1 = (size_t)hidden * channels * NAE_NEURAL_PERCEPTION;
    n_w2 = (size_t)channels * hidden;
    total = n_w1 + hidden + n_w2 + channels;

    block = calloc(total, sizeof *block);
    if (!block)
        return NAE_ERR_ALLOC;

    p->channels = channels;
    p->hidden = hidden;
    p->step_size = 1.0f;
    p->w1 = block;
    p->b1 = p->w1 + n_w1;
    p->w2 = p->b1 + hidden;
    p->b2 = p->w2 + n_w2;
    p->storage = block;
    return NAE_OK;
}

void nae_neural_params_free(nae_neural_params *p)
{
    if (!p)
        return;
    free(p->storage);
    p->channels = 0;
    p->hidden = 0;
    p->w1 = p->b1 = p->w2 = p->b2 = NULL;
    p->storage = NULL;
}

static const nae_rule neural_rule = {
    "neural",
    neural_apply,
    neural_validate,
    sizeof(nae_neural_params),
    0u,
    0u,
};

const nae_rule *nae_rule_neural(void)
{
    return &neural_rule;
}
