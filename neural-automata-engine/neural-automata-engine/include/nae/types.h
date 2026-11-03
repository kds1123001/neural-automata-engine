#ifndef NAE_TYPES_H
#define NAE_TYPES_H

#include <stddef.h>
#include <stdint.h>

#define NAE_ALIGNMENT 64u

typedef enum nae_status {
    NAE_OK = 0,
    NAE_ERR_ARG,
    NAE_ERR_ALLOC,
    NAE_ERR_SHAPE,
    NAE_ERR_OVERFLOW
} nae_status;

typedef enum nae_boundary {
    NAE_BOUNDARY_WRAP = 0,
    NAE_BOUNDARY_ZERO,
    NAE_BOUNDARY_CLAMP
} nae_boundary;

#endif
