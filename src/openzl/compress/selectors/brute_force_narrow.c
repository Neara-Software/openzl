#include "openzl/compress/selectors/brute_force_narrow.h"

#include <string.h>

#include "openzl/common/allocation.h"
#include "openzl/shared/xxhash.h"

struct BFNarrow_s {
    BFNarrow_Entry* entries;
    size_t nbEntries;
};

BFNarrow* BFNarrow_create(void)
{
    return (BFNarrow*)ZL_calloc(sizeof(BFNarrow));
}

void BFNarrow_reset(BFNarrow* state)
{
    if (state == NULL) {
        return;
    }
    for (size_t i = 0; i < state->nbEntries; ++i) {
        ZL_free(state->entries[i].totals);
        ZL_free(state->entries[i].kept);
        ZL_free(state->entries[i].keptForLast);
    }
    ZL_free(state->entries);
    state->entries   = NULL;
    state->nbEntries = 0;
}

void BFNarrow_free(BFNarrow* state)
{
    BFNarrow_reset(state);
    ZL_free(state);
}

BFNarrow_Entry*
BFNarrow_get(BFNarrow* state, const ZL_GraphID* graphs, size_t nbGraphs)
{
    const uint64_t key = XXH3_64bits(graphs, nbGraphs * sizeof(*graphs));
    for (size_t i = 0; i < state->nbEntries; ++i) {
        BFNarrow_Entry* e = &state->entries[i];
        if (e->key == key && e->nbGraphs == nbGraphs) {
            return e;
        }
    }
    BFNarrow_Entry* entries = (BFNarrow_Entry*)ZL_realloc(
            state->entries, (state->nbEntries + 1) * sizeof(BFNarrow_Entry));
    if (entries == NULL) {
        return NULL;
    }
    state->entries    = entries;
    BFNarrow_Entry* e = &entries[state->nbEntries];
    *e                = (BFNarrow_Entry){ .key = key, .nbGraphs = nbGraphs };
    e->totals         = (uint64_t*)ZL_calloc(nbGraphs * sizeof(uint64_t));
    e->kept           = (bool*)ZL_malloc(nbGraphs * sizeof(bool));
    e->keptForLast    = (bool*)ZL_malloc(nbGraphs * sizeof(bool));
    if (e->totals == NULL || e->kept == NULL || e->keptForLast == NULL) {
        ZL_free(e->totals);
        ZL_free(e->kept);
        ZL_free(e->keptForLast);
        return NULL;
    }
    memset(e->kept, 1, nbGraphs * sizeof(bool));
    state->nbEntries++;
    return e;
}
