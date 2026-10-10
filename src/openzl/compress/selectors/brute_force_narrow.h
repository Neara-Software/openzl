#ifndef ZSTRONG_COMPRESS_SELECTORS_BRUTE_FORCE_NARROW_H
#define ZSTRONG_COMPRESS_SELECTORS_BRUTE_FORCE_NARROW_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "openzl/shared/portability.h"
#include "openzl/zl_opaque_types.h"

ZL_BEGIN_C_DECLS

/// Per-CCtx memory of brute force selectors that narrow their successor list
/// across compressions (see ZL_BRUTE_FORCE_NARROW_PARAM_BASE). One entry per
/// selector instance, identified by its successor list.
typedef struct BFNarrow_s BFNarrow;

BFNarrow* BFNarrow_create(void);
void BFNarrow_free(BFNarrow* state);
void BFNarrow_reset(BFNarrow* state);

typedef struct {
    uint64_t key;
    size_t nbGraphs;
    size_t nbInputs;    // distinct inputs ranked so far
    uint64_t lastInput; // hash of the last ranked input
    uint64_t* totals;   // summed compressed size per successor
    bool* kept;         // successors still tried
    bool* keptForLast;  // successors tried on lastInput
} BFNarrow_Entry;

/// @returns the entry for the selector whose successors are @p graphs, created
/// with every successor kept when new; NULL on allocation failure.
BFNarrow_Entry*
BFNarrow_get(BFNarrow* state, const ZL_GraphID* graphs, size_t nbGraphs);

ZL_END_C_DECLS

#endif
