#include <stdint.h>

#include "check.h"
#include "nae/grid.h"

static float peek(const nae_grid *g, int x, int y)
{
    return g->origin[(ptrdiff_t)y * (ptrdiff_t)g->stride_y + (ptrdiff_t)x * (ptrdiff_t)g->stride_x];
}

static void fill_ramp(nae_grid *g)
{
    uint32_t x, y;

    nae_grid_fill(g, 7.0f);
    for (y = 0; y < g->height; ++y)
        for (x = 0; x < g->width; ++x)
            *nae_grid_at(g, x, y) = (float)(x + 10u * y);
}

int main(void)
{
    nae_grid g, other;

    CHECK(nae_grid_init(&g, 4, 3, 3, 2) == NAE_OK);
    CHECK(g.stride_x == 3);
    CHECK(g.stride_y == 24);
    CHECK(g.total == 168);
    CHECK(((uintptr_t)g.origin % NAE_ALIGNMENT) == 0);
    CHECK(nae_grid_at(&g, 1, 1) == g.origin + 24 + 3);
    nae_grid_free(&g);
    CHECK(g.origin == NULL);

    CHECK(nae_grid_init(&g, 4, 3, 1, 2) == NAE_OK);

    fill_ramp(&g);
    nae_grid_apply_boundary(&g, NAE_BOUNDARY_WRAP);
    CHECK(peek(&g, -1, 0) == 3.0f);
    CHECK(peek(&g, -2, -1) == 22.0f);
    CHECK(peek(&g, 5, 3) == 1.0f);
    CHECK(peek(&g, 2, 1) == 12.0f);

    fill_ramp(&g);
    nae_grid_apply_boundary(&g, NAE_BOUNDARY_CLAMP);
    CHECK(peek(&g, -2, -1) == 0.0f);
    CHECK(peek(&g, 5, 4) == 23.0f);
    CHECK(peek(&g, -1, 1) == 10.0f);

    fill_ramp(&g);
    nae_grid_apply_boundary(&g, NAE_BOUNDARY_ZERO);
    CHECK(peek(&g, -1, 0) == 0.0f);
    CHECK(peek(&g, 0, -1) == 0.0f);
    CHECK(peek(&g, 5, 4) == 0.0f);
    CHECK(peek(&g, 2, 1) == 12.0f);

    CHECK(nae_grid_init(&other, 4, 3, 1, 2) == NAE_OK);
    fill_ramp(&g);
    CHECK(nae_grid_copy(&other, &g) == NAE_OK);
    CHECK(peek(&other, 3, 2) == 23.0f);
    nae_grid_free(&other);

    CHECK(nae_grid_init(&other, 5, 3, 1, 2) == NAE_OK);
    CHECK(nae_grid_copy(&other, &g) == NAE_ERR_SHAPE);
    nae_grid_free(&other);
    nae_grid_free(&g);

    CHECK(nae_grid_init(&g, 2, 2, 1, 3) == NAE_OK);
    fill_ramp(&g);
    nae_grid_apply_boundary(&g, NAE_BOUNDARY_WRAP);
    CHECK(peek(&g, -3, 0) == 1.0f);
    CHECK(peek(&g, 4, -3) == 10.0f);
    nae_grid_free(&g);

    CHECK(nae_grid_init(&g, 0, 4, 1, 1) == NAE_ERR_ARG);
    CHECK(nae_grid_init(&g, UINT32_MAX, UINT32_MAX, UINT32_MAX, 1) != NAE_OK);
    CHECK(nae_grid_init(NULL, 1, 1, 1, 1) == NAE_ERR_ARG);
    return 0;
}
