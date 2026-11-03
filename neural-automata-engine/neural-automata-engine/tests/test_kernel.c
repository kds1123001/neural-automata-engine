#include <stdlib.h>

#include "check.h"
#include "nae/kernel.h"

static int within(const nae_kernel *k, int metric_von_neumann)
{
    uint32_t i;

    for (i = 0; i < k->count; ++i) {
        int dx = abs(k->offsets[i].dx);
        int dy = abs(k->offsets[i].dy);
        int reach = metric_von_neumann ? dx + dy : (dx > dy ? dx : dy);

        if (reach > (int)k->radius)
            return 0;
    }
    return 1;
}

int main(void)
{
    nae_kernel k;
    nae_offset offsets[3] = { { -2, 1 }, { 0, 0 }, { 3, -1 } };
    float weights[3] = { 0.5f, 1.0f, 0.25f };

    CHECK(nae_kernel_moore(&k, 1, 0) == NAE_OK);
    CHECK(k.count == 8);
    CHECK(k.radius == 1);
    CHECK(k.offsets[0].dx == -1 && k.offsets[0].dy == -1);
    CHECK(within(&k, 0));
    nae_kernel_free(&k);

    CHECK(nae_kernel_moore(&k, 2, 1) == NAE_OK);
    CHECK(k.count == 25);
    nae_kernel_free(&k);

    CHECK(nae_kernel_von_neumann(&k, 1, 0) == NAE_OK);
    CHECK(k.count == 4);
    CHECK(within(&k, 1));
    nae_kernel_free(&k);

    CHECK(nae_kernel_von_neumann(&k, 3, 1) == NAE_OK);
    CHECK(k.count == 25);
    nae_kernel_free(&k);

    CHECK(nae_kernel_von_neumann(&k, 2, 0) == NAE_OK);
    CHECK(k.count == 12);
    nae_kernel_free(&k);

    CHECK(nae_kernel_custom(&k, offsets, weights, 3) == NAE_OK);
    CHECK(k.count == 3);
    CHECK(k.radius == 3);
    CHECK(k.weights[2] == 0.25f);
    nae_kernel_free(&k);
    CHECK(k.offsets == NULL && k.weights == NULL);

    CHECK(nae_kernel_moore(&k, 0, 1) == NAE_ERR_ARG);
    CHECK(nae_kernel_moore(&k, NAE_KERNEL_MAX_RADIUS + 1, 1) == NAE_ERR_ARG);
    CHECK(nae_kernel_custom(&k, NULL, weights, 3) == NAE_ERR_ARG);
    CHECK(nae_kernel_custom(&k, offsets, weights, 0) == NAE_ERR_ARG);
    return 0;
}
