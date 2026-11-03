#include <string.h>

#include "check.h"
#include "nae/engine.h"
#include "nae/rules.h"

static int make_life(nae_engine *e, uint32_t w, uint32_t h, nae_boundary boundary)
{
    nae_life_params params = { NAE_LIFE_CONWAY_BIRTH, NAE_LIFE_CONWAY_SURVIVE };
    nae_engine_config cfg;

    memset(&cfg, 0, sizeof cfg);
    cfg.width = w;
    cfg.height = h;
    cfg.channels = 1;
    cfg.boundary = boundary;
    cfg.rule = nae_rule_life();
    cfg.params = &params;
    if (nae_kernel_moore(&cfg.kernel, 1, 0) != NAE_OK)
        return 0;
    if (nae_engine_init(e, &cfg) != NAE_OK) {
        nae_kernel_free(&cfg.kernel);
        return 0;
    }
    return 1;
}

static void put(nae_engine *e, uint32_t x, uint32_t y)
{
    *nae_grid_at(nae_engine_state(e), x, y) = 1.0f;
}

static int alive(nae_engine *e, uint32_t x, uint32_t y)
{
    return *nae_grid_at(nae_engine_state(e), x, y) > 0.5f;
}

static uint32_t population(nae_engine *e)
{
    const nae_grid *g = nae_engine_state(e);
    uint32_t x, y, n = 0;

    for (y = 0; y < g->height; ++y)
        for (x = 0; x < g->width; ++x)
            n += *nae_grid_at(g, x, y) > 0.5f;
    return n;
}

static int near(float a, float b)
{
    float d = a - b;

    return d < 1e-5f && d > -1e-5f;
}

static int make_neural(nae_engine *e, nae_neural_params *p, uint32_t w, uint32_t h,
                       nae_boundary boundary, int include_center)
{
    nae_engine_config cfg;

    memset(&cfg, 0, sizeof cfg);
    cfg.width = w;
    cfg.height = h;
    cfg.channels = p->channels;
    cfg.boundary = boundary;
    cfg.rule = nae_rule_neural();
    cfg.params = p;
    if (nae_kernel_moore(&cfg.kernel, 1, include_center) != NAE_OK)
        return 0;
    if (nae_engine_init(e, &cfg) != NAE_OK) {
        nae_kernel_free(&cfg.kernel);
        return 0;
    }
    return 1;
}

static int test_blinker(void)
{
    nae_engine e;

    CHECK(make_life(&e, 5, 5, NAE_BOUNDARY_WRAP));
    put(&e, 1, 2);
    put(&e, 2, 2);
    put(&e, 3, 2);

    nae_engine_step(&e);
    CHECK(population(&e) == 3);
    CHECK(alive(&e, 2, 1) && alive(&e, 2, 2) && alive(&e, 2, 3));

    nae_engine_step(&e);
    CHECK(alive(&e, 1, 2) && alive(&e, 2, 2) && alive(&e, 3, 2));
    CHECK(e.generation == 2);
    nae_engine_free(&e);
    return 0;
}

static int test_glider(void)
{
    static const uint32_t cells[5][2] = { { 1, 0 }, { 2, 1 }, { 0, 2 }, { 1, 2 }, { 2, 2 } };
    nae_engine e;
    int i;

    CHECK(make_life(&e, 8, 8, NAE_BOUNDARY_WRAP));
    for (i = 0; i < 5; ++i)
        put(&e, cells[i][0], cells[i][1]);

    nae_engine_run(&e, 4);
    CHECK(population(&e) == 5);
    for (i = 0; i < 5; ++i)
        CHECK(alive(&e, cells[i][0] + 1, cells[i][1] + 1));

    nae_engine_run(&e, 28);
    CHECK(population(&e) == 5);
    for (i = 0; i < 5; ++i)
        CHECK(alive(&e, cells[i][0], cells[i][1]));
    nae_engine_free(&e);
    return 0;
}

static int test_zero_boundary(void)
{
    nae_engine e;

    CHECK(make_life(&e, 6, 6, NAE_BOUNDARY_ZERO));
    put(&e, 0, 0);
    nae_engine_step(&e);
    CHECK(population(&e) == 0);
    nae_engine_free(&e);

    CHECK(make_life(&e, 6, 6, NAE_BOUNDARY_ZERO));
    put(&e, 0, 0);
    put(&e, 1, 0);
    put(&e, 0, 1);
    put(&e, 1, 1);
    nae_engine_run(&e, 3);
    CHECK(population(&e) == 4);
    nae_engine_free(&e);
    return 0;
}

static int test_neural_drift(void)
{
    nae_neural_params p;
    nae_engine e;
    const nae_grid *g;

    CHECK(nae_neural_params_init(&p, 2, 1) == NAE_OK);
    p.b1[0] = 1.0f;
    p.w2[0] = 0.5f;
    p.w2[1] = 0.25f;
    CHECK(make_neural(&e, &p, 5, 4, NAE_BOUNDARY_WRAP, 0));

    nae_engine_run(&e, 3);
    g = nae_engine_state_const(&e);
    CHECK(near(nae_grid_at(g, 2, 2)[0], 1.5f));
    CHECK(near(nae_grid_at(g, 4, 3)[1], 0.75f));
    nae_engine_free(&e);
    nae_neural_params_free(&p);
    return 0;
}

static int test_neural_smooth(void)
{
    nae_neural_params p;
    nae_engine e;
    uint32_t x, y;

    CHECK(nae_neural_params_init(&p, 1, 1) == NAE_OK);
    p.w1[1] = 1.0f;
    p.w2[0] = 1.0f;
    CHECK(make_neural(&e, &p, 6, 6, NAE_BOUNDARY_WRAP, 0));

    for (y = 0; y < 6; ++y)
        for (x = 0; x < 6; ++x)
            *nae_grid_at(nae_engine_state(&e), x, y) = 1.0f;

    nae_engine_step(&e);
    for (y = 0; y < 6; ++y)
        for (x = 0; x < 6; ++x)
            CHECK(near(*nae_grid_at(nae_engine_state(&e), x, y), 9.0f));
    nae_engine_free(&e);
    nae_neural_params_free(&p);
    return 0;
}

static int test_neural_gradient(void)
{
    nae_neural_params p;
    nae_engine e;
    uint32_t x, y;

    CHECK(nae_neural_params_init(&p, 1, 2) == NAE_OK);
    p.w1[2] = 1.0f;
    p.w1[4 + 2] = -1.0f;
    p.w2[0] = 1.0f;
    p.w2[1] = -1.0f;
    CHECK(make_neural(&e, &p, 6, 6, NAE_BOUNDARY_WRAP, 0));

    for (y = 0; y < 6; ++y)
        for (x = 0; x < 6; ++x)
            *nae_grid_at(nae_engine_state(&e), x, y) = (float)x;

    nae_engine_step(&e);
    CHECK(near(*nae_grid_at(nae_engine_state(&e), 2, 3), 8.0f));
    CHECK(near(*nae_grid_at(nae_engine_state(&e), 3, 2), 9.0f));
    nae_engine_free(&e);
    nae_neural_params_free(&p);
    return 0;
}

static int test_init_rejects_bad_input(void)
{
    nae_life_params params = { NAE_LIFE_CONWAY_BIRTH, NAE_LIFE_CONWAY_SURVIVE };
    nae_engine_config cfg;
    nae_engine e;

    memset(&cfg, 0, sizeof cfg);
    cfg.width = 4;
    cfg.height = 4;
    cfg.channels = 2;
    cfg.rule = nae_rule_life();
    cfg.params = &params;
    CHECK(nae_kernel_moore(&cfg.kernel, 1, 0) == NAE_OK);
    CHECK(nae_engine_init(&e, &cfg) == NAE_ERR_SHAPE);

    cfg.channels = 1;
    cfg.params = NULL;
    CHECK(nae_engine_init(&e, &cfg) == NAE_ERR_ARG);
    nae_kernel_free(&cfg.kernel);

    CHECK(nae_rule_find("life") == nae_rule_life());
    CHECK(nae_rule_find("neural") == nae_rule_neural());
    CHECK(nae_rule_find("missing") == NULL);
    return 0;
}

int main(void)
{
    CHECK(test_blinker() == 0);
    CHECK(test_glider() == 0);
    CHECK(test_zero_boundary() == 0);
    CHECK(test_neural_drift() == 0);
    CHECK(test_neural_smooth() == 0);
    CHECK(test_neural_gradient() == 0);
    CHECK(test_init_rejects_bad_input() == 0);
    return 0;
}
