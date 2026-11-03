#ifndef NAE_TEST_CHECK_H
#define NAE_TEST_CHECK_H

#include <stdio.h>

#define CHECK(expr)                                                              \
    do {                                                                         \
        if (!(expr)) {                                                           \
            fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #expr); \
            return 1;                                                            \
        }                                                                        \
    } while (0)

#endif
