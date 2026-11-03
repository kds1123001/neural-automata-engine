#ifndef NAE_RULES_H
#define NAE_RULES_H

#include "nae/rule.h"

#define NAE_LIFE_CONWAY_BIRTH   0x008u
#define NAE_LIFE_CONWAY_SURVIVE 0x00Cu

#define NAE_NEURAL_PERCEPTION   4u
#define NAE_NEURAL_MAX_CHANNELS 32u
#define NAE_NEURAL_MAX_HIDDEN   256u

typedef struct nae_life_params {
    uint32_t birth_mask;
    uint32_t survive_mask;
} nae_life_params;

typedef struct nae_neural_params {
    uint32_t channels;
    uint32_t hidden;
    float step_size;
    float state_clip;
    float *w1;
    float *b1;
    float *w2;
    float *b2;
    void *storage;
} nae_neural_params;

nae_status nae_neural_params_init(nae_neural_params *p, uint32_t channels, uint32_t hidden);
void nae_neural_params_free(nae_neural_params *p);

#endif
