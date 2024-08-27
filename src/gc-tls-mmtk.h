// This file is a part of Julia. License is MIT: https://julialang.org/license

#ifdef MMTK_GC

#include <assert.h>
#include "mmtkMutator.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    // variable for tracking weak references
    small_arraylist_t weak_refs;
    // live tasks started on this thread
    // that are holding onto a stack from the pool
    small_arraylist_t live_tasks;

    // variables for tracking malloc'd arrays
    struct _mallocmemory_t *mallocarrays;
    struct _mallocmemory_t *mafreelist;

#define JL_N_STACK_POOLS 16
    small_arraylist_t free_stacks[JL_N_STACK_POOLS];
} jl_thread_heap_t;

typedef struct {
    _Atomic(int64_t) allocd;
    _Atomic(int64_t) pool_live_bytes;
    _Atomic(uint64_t) malloc;
    _Atomic(uint64_t) realloc;
    _Atomic(uint64_t) poolalloc;
    _Atomic(uint64_t) bigalloc;
    _Atomic(int64_t) free_acc;
    _Atomic(uint64_t) alloc_acc;
} jl_thread_gc_num_t;

typedef struct {
    jl_thread_heap_t heap;
    jl_thread_gc_num_t gc_num;
    MMTkMutatorContext mmtk_mutator;
    size_t malloc_sz_since_last_poll;
} jl_gc_tls_states_t;

#ifdef __cplusplus
}
#endif

#endif // MMTK_GC
