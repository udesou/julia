#ifdef MMTK_GC

#include "mmtk_julia.h"
#include "gc-common.h"
#include "mmtkMutator.h"
#include "gc-mmtk.h"
#include "threading.h"

#ifdef __cplusplus
extern "C" {
#endif

// FIXME: Does it make sense for MMTk to implement something similar
// for now, just ignoring this.

// Must be kept in sync with `base/timing.jl`
#define FULL_SWEEP_REASON_SWEEP_ALWAYS_FULL (0)
#define FULL_SWEEP_REASON_FORCED_FULL_SWEEP (1)
#define FULL_SWEEP_REASON_USER_MAX_EXCEEDED (2)
#define FULL_SWEEP_REASON_LARGE_PROMOTION_RATE (3)
#define FULL_SWEEP_NUM_REASONS (4)

// Table recording number of full GCs due to each reason
JL_DLLEXPORT uint64_t jl_full_sweep_reasons[FULL_SWEEP_NUM_REASONS];

// FIXME: Should the values below be shared between both GC's?
// Note that MMTk uses a hard max heap limit, which is set by default
// as 70% of the free available memory. The min heap is set as the
// default_collect_interval variable below.

// max_total_memory is a suggestion.  We try very hard to stay
// under this limit, but we will go above it rather than halting.
#ifdef _P64
typedef uint64_t memsize_t;
static const size_t default_collect_interval = 5600 * 1024 * sizeof(void*);
// We expose this to the user/ci as jl_gc_set_max_memory
static memsize_t max_total_memory = (memsize_t) 2 * 1024 * 1024 * 1024 * 1024 * 1024;
#else
typedef uint32_t memsize_t;
static const size_t default_collect_interval = 3200 * 1024 * sizeof(void*);
// Work really hard to stay within 2GB
// Alternative is to risk running out of address space
// on 32 bit architectures.
#define MAX32HEAP 1536 * 1024 * 1024
static memsize_t max_total_memory = (memsize_t) MAX32HEAP;
#endif

void jl_gc_init(void) {
    // TODO: use jl_options.heap_size_hint to set MMTk's fixed heap size? (see issue: https://github.com/mmtk/mmtk-julia/issues/167)
    JL_MUTEX_INIT(&finalizers_lock, "finalizers_lock");

    arraylist_new(&to_finalize, 0);
    arraylist_new(&finalizer_list_marked, 0);

    gc_num.allocd = 0;
    gc_num.max_pause = 0;
    gc_num.max_memory = 0;

    long long min_heap_size;
    long long max_heap_size;
    char* min_size_def = getenv("MMTK_MIN_HSIZE");
    char* min_size_gb = getenv("MMTK_MIN_HSIZE_G");

    char* max_size_def = getenv("MMTK_MAX_HSIZE");
    char* max_size_gb = getenv("MMTK_MAX_HSIZE_G");

    // default min heap currently set as Julia's default_collect_interval
    if (min_size_def != NULL) {
        char *p;
        double min_size = strtod(min_size_def, &p);
        min_heap_size = (long) 1024 * 1024 * min_size;
    } else if (min_size_gb != NULL) {
        char *p;
        double min_size = strtod(min_size_gb, &p);
        min_heap_size = (long) 1024 * 1024 * 1024 * min_size;
    } else {
        min_heap_size = default_collect_interval;
    }

    // default max heap currently set as 70% the free memory in the system
    if (max_size_def != NULL) {
        char *p;
        double max_size = strtod(max_size_def, &p);
        max_heap_size = (long) 1024 * 1024 * max_size;
    } else if (max_size_gb != NULL) {
        char *p;
        double max_size = strtod(max_size_gb, &p);
        max_heap_size = (long) 1024 * 1024 * 1024 * max_size;
    } else {
        max_heap_size = uv_get_free_memory() * 70 / 100;
    }

    // Assert that the number of stock GC threads is 0; MMTK uses the number of threads in jl_options.ngcthreads
    assert(jl_n_gcthreads == 0);

    // Check that the julia_copy_stack rust feature has been defined when the COPY_STACK has been defined
    int copy_stacks;

#ifdef COPY_STACKS
    copy_stacks = 1;
#else
    copy_stacks = 0;
#endif

    mmtk_julia_copy_stack_check(copy_stacks);

    // if only max size is specified initialize MMTk with a fixed size heap
    // TODO: We just assume mark threads means GC threads, and ignore the number of concurrent sweep threads.
    // If the two values are the same, we can use either. Otherwise, we need to be careful.
    uintptr_t gcthreads = jl_options.nmarkthreads;
    if (max_size_def != NULL || (max_size_gb != NULL && (min_size_def == NULL && min_size_gb == NULL))) {
        mmtk_gc_init(0, max_heap_size, gcthreads, &mmtk_upcalls, (sizeof(jl_taggedvalue_t)), jl_buff_tag);
    } else {
        mmtk_gc_init(min_heap_size, max_heap_size, gcthreads, &mmtk_upcalls, (sizeof(jl_taggedvalue_t)), jl_buff_tag);
    }
}

void jl_start_gc_threads(void) {
    jl_ptls_t ptls = jl_current_task->ptls;
    mmtk_initialize_collection((void *)ptls);
}

void jl_init_thread_heap(struct _jl_tls_states_t *ptls) JL_NOTSAFEPOINT {
    jl_thread_heap_common_t *heap = &ptls->gc_tls_common.heap;
    small_arraylist_new(&heap->weak_refs, 0);
    small_arraylist_new(&heap->live_tasks, 0);
    for (int i = 0; i < JL_N_STACK_POOLS; i++)
        small_arraylist_new(&heap->free_stacks[i], 0);
    heap->mallocarrays = NULL;
    heap->mafreelist = NULL;
    arraylist_new(&ptls->finalizers, 0);
    // Clear the malloc sz count
    jl_atomic_store_relaxed(&ptls->gc_tls.malloc_sz_since_last_poll, 0);
    // Create mutator
    MMTk_Mutator mmtk_mutator = mmtk_bind_mutator((void *)ptls, ptls->tid);
    // Copy the mutator to the thread local storage
    memcpy(&ptls->gc_tls.mmtk_mutator, mmtk_mutator, sizeof(MMTkMutatorContext));
    // Call post_bind to maintain a list of active mutators and to reclaim the old mutator (which is no longer needed)
    mmtk_post_bind_mutator(&ptls->gc_tls.mmtk_mutator, mmtk_mutator);
    memset(&ptls->gc_tls_common.gc_num, 0, sizeof(ptls->gc_tls_common.gc_num));
}

void jl_free_thread_gc_state(struct _jl_tls_states_t *ptls) {
    mmtk_destroy_mutator(&ptls->gc_tls.mmtk_mutator);
}

JL_DLLEXPORT void jl_gc_set_max_memory(uint64_t max_mem) {
    // MMTk currently does not allow setting the heap size at runtime
}


inline void maybe_collect(jl_ptls_t ptls)
{
    // Just do a safe point for general maybe_collect
    jl_gc_safepoint_(ptls);
}

// This is only used for malloc. We need to know if we need to do GC. However, keeping checking with MMTk (mmtk_gc_poll),
// is expensive. So we only check for every few allocations.
static inline void malloc_maybe_collect(jl_ptls_t ptls, size_t sz)
{
    // We do not need to carefully maintain malloc_sz_since_last_poll. We just need to
    // avoid using mmtk_gc_poll too frequently, and try to be precise on our heap usage
    // as much as we can.
    if (ptls->gc_tls.malloc_sz_since_last_poll > 4096) {
        jl_atomic_store_relaxed(&ptls->gc_tls.malloc_sz_since_last_poll, 0);
        mmtk_gc_poll(ptls);
    } else {
        jl_atomic_fetch_add_relaxed(&ptls->gc_tls.malloc_sz_since_last_poll, sz);
        jl_gc_safepoint_(ptls);
    }
}

JL_DLLEXPORT void jl_gc_collect(jl_gc_collection_t collection) {
    jl_task_t *ct = jl_current_task;
    jl_ptls_t ptls = ct->ptls;
    if (jl_atomic_load_acquire(&jl_gc_disable_counter)) {
        size_t localbytes = jl_atomic_load_relaxed(&ptls->gc_tls_common.gc_num.allocd) + gc_num.interval;
        jl_atomic_store_relaxed(&ptls->gc_tls_common.gc_num.allocd, -(int64_t)gc_num.interval);
        static_assert(sizeof(_Atomic(uint64_t)) == sizeof(gc_num.deferred_alloc), "");
        jl_atomic_fetch_add_relaxed((_Atomic(uint64_t)*)&gc_num.deferred_alloc, localbytes);
        return;
    }
    mmtk_handle_user_collection_request(ptls, collection);
}

// FIXME: The functions combine_thread_gc_counts and reset_thread_gc_counts
// are currently nearly identical for mmtk and for stock. However, the stats
// are likely different (e.g., MMTk doesn't track the bytes allocated in the fastpath,
// but only when the slowpath is called). We might need to adapt these later so that
// the statistics are the same or as close as possible for each GC.

static void combine_thread_gc_counts(jl_gc_num_t *dest, int update_heap) JL_NOTSAFEPOINT
{
    int gc_n_threads;
    jl_ptls_t* gc_all_tls_states;
    gc_n_threads = jl_atomic_load_acquire(&jl_n_threads);
    gc_all_tls_states = jl_atomic_load_relaxed(&jl_all_tls_states);
    for (int i = 0; i < gc_n_threads; i++) {
        jl_ptls_t ptls = gc_all_tls_states[i];
        if (ptls) {
            dest->allocd += (jl_atomic_load_relaxed(&ptls->gc_tls_common.gc_num.allocd) + gc_num.interval);
            dest->malloc += jl_atomic_load_relaxed(&ptls->gc_tls_common.gc_num.malloc);
            dest->realloc += jl_atomic_load_relaxed(&ptls->gc_tls_common.gc_num.realloc);
            dest->poolalloc += jl_atomic_load_relaxed(&ptls->gc_tls_common.gc_num.poolalloc);
            dest->bigalloc += jl_atomic_load_relaxed(&ptls->gc_tls_common.gc_num.bigalloc);
            dest->freed += jl_atomic_load_relaxed(&ptls->gc_tls_common.gc_num.free_acc);
            if (update_heap) {
                jl_atomic_store_relaxed(&ptls->gc_tls_common.gc_num.alloc_acc, 0);
                jl_atomic_store_relaxed(&ptls->gc_tls_common.gc_num.free_acc, 0);
            }
        }
    }
}


void reset_thread_gc_counts(void) JL_NOTSAFEPOINT
{
    int gc_n_threads;
    jl_ptls_t* gc_all_tls_states;
    gc_n_threads = jl_atomic_load_acquire(&jl_n_threads);
    gc_all_tls_states = jl_atomic_load_relaxed(&jl_all_tls_states);
    for (int i = 0; i < gc_n_threads; i++) {
        jl_ptls_t ptls = gc_all_tls_states[i];
        if (ptls != NULL) {
            // don't reset `pool_live_bytes` here
            jl_atomic_store_relaxed(&ptls->gc_tls_common.gc_num.allocd, -(int64_t)gc_num.interval);
            jl_atomic_store_relaxed(&ptls->gc_tls_common.gc_num.malloc, 0);
            jl_atomic_store_relaxed(&ptls->gc_tls_common.gc_num.realloc, 0);
            jl_atomic_store_relaxed(&ptls->gc_tls_common.gc_num.poolalloc, 0);
            jl_atomic_store_relaxed(&ptls->gc_tls_common.gc_num.bigalloc, 0);
            jl_atomic_store_relaxed(&ptls->gc_tls_common.gc_num.alloc_acc, 0);
            jl_atomic_store_relaxed(&ptls->gc_tls_common.gc_num.free_acc, 0);
        }
    }
}

// Retrieves Julia's `GC_Num` (structure that stores GC statistics).
JL_DLLEXPORT jl_gc_num_t jl_gc_num(void) {
    jl_gc_num_t num = gc_num;
    combine_thread_gc_counts(&num, 0);
    return num;
}

int64_t last_gc_total_bytes = 0;
int64_t last_live_bytes = 0; // live_bytes at last collection
int64_t live_bytes = 0;

JL_DLLEXPORT int64_t jl_gc_diff_total_bytes(void) JL_NOTSAFEPOINT {
    int64_t oldtb = last_gc_total_bytes;
    int64_t newtb;
    jl_gc_get_total_bytes(&newtb);
    last_gc_total_bytes = newtb;
    return newtb - oldtb;
}

JL_DLLEXPORT int64_t jl_gc_sync_total_bytes(int64_t offset) JL_NOTSAFEPOINT
{
    int64_t oldtb = last_gc_total_bytes;
    int64_t newtb;
    jl_gc_get_total_bytes(&newtb);
    last_gc_total_bytes = newtb - offset;
    return newtb - oldtb;
}

JL_DLLEXPORT int64_t jl_gc_pool_live_bytes(void) {
    return 0;
}

void jl_gc_count_allocd(size_t sz) JL_NOTSAFEPOINT
{
    jl_ptls_t ptls = jl_current_task->ptls;
    jl_atomic_store_relaxed(&ptls->gc_tls_common.gc_num.allocd,
        jl_atomic_load_relaxed(&ptls->gc_tls_common.gc_num.allocd) + sz);
}

void jl_gc_count_freed(size_t sz) JL_NOTSAFEPOINT
{
}

int64_t inc_live_bytes(int64_t inc) JL_NOTSAFEPOINT
{
    jl_timing_counter_inc(JL_TIMING_COUNTER_HeapSize, inc);
    return live_bytes += inc;
}

void jl_gc_reset_alloc_count(void) JL_NOTSAFEPOINT
{
    combine_thread_gc_counts(&gc_num, 0);
    inc_live_bytes(gc_num.deferred_alloc + gc_num.allocd);
    gc_num.allocd = 0;
    gc_num.deferred_alloc = 0;
    reset_thread_gc_counts();
}

JL_DLLEXPORT int64_t jl_gc_live_bytes(void) {
    return last_live_bytes;
}

JL_DLLEXPORT void jl_gc_get_total_bytes(int64_t *bytes) JL_NOTSAFEPOINT
{
    jl_gc_num_t num = gc_num;
    combine_thread_gc_counts(&num, 0);
    // Sync this logic with `base/util.jl:GC_Diff`
    *bytes = (num.total_allocd + num.deferred_alloc + num.allocd);
}

JL_DLLEXPORT uint64_t jl_gc_get_max_memory(void)
{
    // FIXME: should probably return MMTk's heap size
    return max_total_memory;
}

// weak references
// ---
JL_DLLEXPORT jl_weakref_t *jl_gc_new_weakref_th(jl_ptls_t ptls, jl_value_t *value)
{
    jl_weakref_t *wr = (jl_weakref_t*)jl_gc_alloc(ptls, sizeof(void*), jl_weakref_type);
    wr->value = value;  // NOTE: wb not needed here
    mmtk_add_weak_candidate(wr);
    return wr;
}

// allocation

extern void mmtk_object_reference_write_post(void* mutator, const void* parent, const void* ptr);
extern void mmtk_object_reference_write_slow(void* mutator, const void* parent, const void* ptr);
extern void* mmtk_alloc(void* mutator, size_t size, size_t align, size_t offset, int allocator);
extern void mmtk_post_alloc(void* mutator, void* refer, size_t bytes, int allocator);
extern const void* MMTK_SIDE_LOG_BIT_BASE_ADDRESS;
extern const void* MMTK_SIDE_VO_BIT_BASE_ADDRESS;
extern void mmtk_store_obj_size_c(void* obj, size_t size);

#define MMTK_DEFAULT_IMMIX_ALLOCATOR (0)
#define MMTK_IMMORTAL_BUMP_ALLOCATOR (0)


int jl_gc_classify_pools(size_t sz, int *osize)
{
    if (sz > GC_MAX_SZCLASS)
        return -1; // call big alloc function
    size_t allocsz = sz + sizeof(jl_taggedvalue_t);
    *osize = LLT_ALIGN(allocsz, 16);
    return 0; // use MMTk's fastpath logic
}
#define MMTK_MIN_ALIGNMENT 4
// MMTk assumes allocation size is aligned to min alignment.
inline size_t mmtk_align_alloc_sz(size_t sz) JL_NOTSAFEPOINT
{
    return (sz + MMTK_MIN_ALIGNMENT - 1) & ~(MMTK_MIN_ALIGNMENT - 1);
}

inline void* bump_alloc_fast(MMTkMutatorContext* mutator, uintptr_t* cursor, uintptr_t limit, size_t size, size_t align, size_t offset, int allocator) {
    intptr_t delta = (-offset - *cursor) & (align - 1);
    uintptr_t result = *cursor + (uintptr_t)delta;

    if (__unlikely(result + size > limit)) {
        return (void*) mmtk_alloc(mutator, size, align, offset, allocator);
    } else{
        *cursor = result + size;
        return (void*)result;
    }
}

inline void* mmtk_immix_alloc_fast(MMTkMutatorContext* mutator, size_t size, size_t align, size_t offset) {
    ImmixAllocator* allocator = &mutator->allocators.immix[MMTK_DEFAULT_IMMIX_ALLOCATOR];
    return bump_alloc_fast(mutator, (uintptr_t*)&allocator->cursor, (intptr_t)allocator->limit, size, align, offset, 0);
}

inline void mmtk_immix_post_alloc_slow(MMTkMutatorContext* mutator, void* obj, size_t size) {
    mmtk_post_alloc(mutator, obj, size, 0);
}

inline void mmtk_immix_post_alloc_fast(MMTkMutatorContext* mutator, void* obj, size_t size) {
    // FIXME: for now, we do nothing
    // but when supporting moving, this is where we set the valid object (VO) bit
}

inline void* mmtk_immortal_alloc_fast(MMTkMutatorContext* mutator, size_t size, size_t align, size_t offset) {
    BumpAllocator* allocator = &mutator->allocators.bump_pointer[MMTK_IMMORTAL_BUMP_ALLOCATOR];
    return bump_alloc_fast(mutator, (uintptr_t*)&allocator->cursor, (uintptr_t)allocator->limit, size, align, offset, 1);
}

inline void mmtk_immortal_post_alloc_fast(MMTkMutatorContext* mutator, void* obj, size_t size) {
    // FIXME: Similarly, for now, we do nothing
    // but when supporting moving, this is where we set the valid object (VO) bit
    // and log (old gen) bit
}

// allocation wrappers that track allocation and let collection run
JL_DLLEXPORT void *jl_gc_counted_malloc(size_t sz)
{
    jl_gcframe_t **pgcstack = jl_get_pgcstack();
    jl_task_t *ct = jl_current_task;
    void *data = malloc(sz);
    if (data != NULL && pgcstack != NULL && ct->world_age) {
        jl_ptls_t ptls = ct->ptls;
        malloc_maybe_collect(ptls, sz);
        jl_atomic_fetch_add_relaxed(&JULIA_MALLOC_BYTES, sz);
    }
    return data;
}

JL_DLLEXPORT void *jl_gc_counted_calloc(size_t nm, size_t sz)
{
    jl_gcframe_t **pgcstack = jl_get_pgcstack();
    jl_task_t *ct = jl_current_task;
    void *data = calloc(nm, sz);
    if (data != NULL && pgcstack != NULL && ct->world_age) {
        jl_ptls_t ptls = ct->ptls;
        malloc_maybe_collect(ptls, nm * sz);
        jl_atomic_fetch_add_relaxed(&JULIA_MALLOC_BYTES, nm * sz);
    }
    return data;
}

JL_DLLEXPORT void jl_gc_counted_free_with_size(void *p, size_t sz)
{
    jl_gcframe_t **pgcstack = jl_get_pgcstack();
    jl_task_t *ct = jl_current_task;
    free(p);
    if (pgcstack != NULL && ct->world_age) {
        jl_atomic_fetch_add_relaxed(&JULIA_MALLOC_BYTES, -sz);
    }
}

JL_DLLEXPORT void *jl_gc_counted_realloc_with_old_size(void *p, size_t old, size_t sz)
{
    jl_gcframe_t **pgcstack = jl_get_pgcstack();
    jl_task_t *ct = jl_current_task;
    if (pgcstack && ct->world_age) {
        jl_ptls_t ptls = ct->ptls;
        malloc_maybe_collect(ptls, sz);
        if (sz < old)
            jl_atomic_fetch_add_relaxed(&JULIA_MALLOC_BYTES, old - sz);
        else
            jl_atomic_fetch_add_relaxed(&JULIA_MALLOC_BYTES, sz - old);
    }
    return realloc(p, sz);
}

void *jl_gc_perm_alloc_nolock(size_t sz, int zero, unsigned align, unsigned offset)
{
    jl_ptls_t ptls = jl_current_task->ptls;
    size_t allocsz = mmtk_align_alloc_sz(sz);
    void* addr = mmtk_immortal_alloc_fast(&ptls->gc_tls.mmtk_mutator, allocsz, align, offset);
    return addr;
}

void *jl_gc_perm_alloc(size_t sz, int zero, unsigned align, unsigned offset)
{
    return jl_gc_perm_alloc_nolock(sz, zero, align, offset);
}

jl_value_t *jl_gc_permobj(size_t sz, void *ty) JL_NOTSAFEPOINT
{
    const size_t allocsz = sz + sizeof(jl_taggedvalue_t);
    unsigned align = (sz == 0 ? sizeof(void*) : (allocsz <= sizeof(void*) * 2 ?
                                                 sizeof(void*) * 2 : 16));
    jl_taggedvalue_t *o = (jl_taggedvalue_t*)jl_gc_perm_alloc(allocsz, 0, align,
                                                              sizeof(void*) % align);

    jl_ptls_t ptls = jl_current_task->ptls;
    mmtk_immortal_post_alloc_fast(&ptls->gc_tls.mmtk_mutator, jl_valueof(o), allocsz);
    o->header = (uintptr_t)ty;
    return jl_valueof(o);
}

JL_DLLEXPORT jl_value_t *jl_mmtk_gc_alloc_default(jl_ptls_t ptls, int osize, size_t align, void *ty)
{
    // safepoint
    jl_gc_safepoint_(ptls);

    jl_value_t *v;
    if ((uintptr_t)ty != jl_buff_tag) {
        // v needs to be 16 byte aligned, therefore v_tagged needs to be offset accordingly to consider the size of header
        jl_taggedvalue_t *v_tagged = (jl_taggedvalue_t *)mmtk_immix_alloc_fast(&ptls->gc_tls.mmtk_mutator, LLT_ALIGN(osize, align), align, sizeof(jl_taggedvalue_t));
        v = jl_valueof(v_tagged);
        mmtk_immix_post_alloc_fast(&ptls->gc_tls.mmtk_mutator, v, LLT_ALIGN(osize, align));
    } else {
        // allocating an extra word to store the size of buffer objects
        jl_taggedvalue_t *v_tagged = (jl_taggedvalue_t *)mmtk_immix_alloc_fast(&ptls->gc_tls.mmtk_mutator, LLT_ALIGN(osize+sizeof(jl_taggedvalue_t), align), align, 0);
        jl_value_t* v_tagged_aligned = ((jl_value_t*)((char*)(v_tagged) + sizeof(jl_taggedvalue_t)));
        v = jl_valueof(v_tagged_aligned);
        mmtk_store_obj_size_c(v, LLT_ALIGN(osize+sizeof(jl_taggedvalue_t), align));
        mmtk_immix_post_alloc_fast(&ptls->gc_tls.mmtk_mutator, v, LLT_ALIGN(osize+sizeof(jl_taggedvalue_t), align));
    }

    ptls->gc_tls_common.gc_num.allocd += osize;
    ptls->gc_tls_common.gc_num.poolalloc++;

    return v;
}

JL_DLLEXPORT jl_value_t *jl_mmtk_gc_alloc_big(jl_ptls_t ptls, size_t sz)
{
    // safepoint
    jl_gc_safepoint_(ptls);

    size_t offs = offsetof(bigval_t, header);
    assert(sz >= sizeof(jl_taggedvalue_t) && "sz must include tag");
    static_assert(offsetof(bigval_t, header) >= sizeof(void*), "Empty bigval header?");
    static_assert(sizeof(bigval_t) % JL_HEAP_ALIGNMENT == 0, "");
    size_t allocsz = LLT_ALIGN(sz + offs, JL_CACHE_BYTE_ALIGNMENT);
    if (allocsz < sz) { // overflow in adding offs, size was "negative"
        assert(0 && "Error when allocating big object");
        jl_throw(jl_memory_exception);
    }

    bigval_t *v = (bigval_t*)mmtk_alloc_large(&ptls->gc_tls.mmtk_mutator, allocsz, JL_CACHE_BYTE_ALIGNMENT, 0, 2);

    if (v == NULL) {
        assert(0 && "Allocation failed");
        jl_throw(jl_memory_exception);
    }
    v->sz = allocsz;

    ptls->gc_tls_common.gc_num.allocd += allocsz;
    ptls->gc_tls_common.gc_num.bigalloc++;

    jl_value_t *result = jl_valueof(&v->header);
    mmtk_post_alloc(&ptls->gc_tls.mmtk_mutator, result, allocsz, 2);

    return result;
}

// Instrumented version of jl_gc_small_alloc_inner, called into by LLVM-generated code.
JL_DLLEXPORT jl_value_t *jl_gc_small_alloc(jl_ptls_t ptls, int offset, int osize, jl_value_t* type)
{
    assert(jl_atomic_load_relaxed(&ptls->gc_state) == 0);

    jl_value_t *val = jl_mmtk_gc_alloc_default(ptls, osize, 16, NULL);
    maybe_record_alloc_to_profile(val, osize, (jl_datatype_t*)type);
    return val;
}

// Instrumented version of jl_gc_big_alloc_inner, called into by LLVM-generated code.
JL_DLLEXPORT jl_value_t *jl_gc_big_alloc(jl_ptls_t ptls, size_t sz, jl_value_t *type)
{
    // TODO: assertion needed here?
    assert(jl_atomic_load_relaxed(&ptls->gc_state) == 0);

    jl_value_t *val = jl_mmtk_gc_alloc_big(ptls, sz);
    maybe_record_alloc_to_profile(val, sz, (jl_datatype_t*)type);
    return val;
}

inline jl_value_t *jl_gc_alloc_(jl_ptls_t ptls, size_t sz, void *ty)
{
    jl_value_t *v;
    const size_t allocsz = sz + sizeof(jl_taggedvalue_t);
    if (sz <= GC_MAX_SZCLASS) {
        v = jl_mmtk_gc_alloc_default(ptls, allocsz, 16, ty);
    }
    else {
        if (allocsz < sz) // overflow in adding offs, size was "negative"
            jl_throw(jl_memory_exception);
        v = jl_mmtk_gc_alloc_big(ptls, allocsz);
    }
    jl_set_typeof(v, ty);
    maybe_record_alloc_to_profile(v, sz, (jl_datatype_t*)ty);
    return v;
}

JL_DLLEXPORT void *jl_gc_managed_malloc(size_t sz)
{
    jl_ptls_t ptls = jl_current_task->ptls;
    maybe_collect(ptls);
    size_t allocsz = LLT_ALIGN(sz, JL_CACHE_BYTE_ALIGNMENT);
    if (allocsz < sz)  // overflow in adding offs, size was "negative"
        jl_throw(jl_memory_exception);

    int last_errno = errno;
#ifdef _OS_WINDOWS_
    DWORD last_error = GetLastError();
#endif
    void *b = malloc_cache_align(allocsz);
    if (b == NULL)
        jl_throw(jl_memory_exception);

    jl_atomic_store_relaxed(&ptls->gc_tls_common.gc_num.allocd,
        jl_atomic_load_relaxed(&ptls->gc_tls_common.gc_num.allocd) + allocsz);
    jl_atomic_store_relaxed(&ptls->gc_tls_common.gc_num.malloc,
        jl_atomic_load_relaxed(&ptls->gc_tls_common.gc_num.malloc) + 1);
    // FIXME: Should these be part of mmtk's heap?
    // malloc_maybe_collect(ptls, sz);
    // jl_atomic_fetch_add_relaxed(&JULIA_MALLOC_BYTES, allocsz);
#ifdef _OS_WINDOWS_
    SetLastError(last_error);
#endif
    errno = last_errno;
    // jl_gc_managed_malloc is currently always used for allocating array buffers.
    maybe_record_alloc_to_profile((jl_value_t*)b, sz, (jl_datatype_t*)jl_buff_tag);
    return b;
}

void jl_gc_notify_image_load(const char* img_data, size_t len)
{
    mmtk_set_vm_space((void*)img_data, len);
}

// mutex for page profile
uv_mutex_t page_profile_lock;

JL_DLLEXPORT void jl_gc_take_page_profile(ios_t *stream)
{
    uv_mutex_lock(&page_profile_lock);
    const char *str = "Page profiler in unsupported in MMTk.";
    ios_write(stream, str, strlen(str));
    uv_mutex_unlock(&page_profile_lock);
}

// this seems to be needed by the gc tests
#define JL_GC_N_MAX_POOLS 51
JL_DLLEXPORT double jl_gc_page_utilization_stats[JL_GC_N_MAX_POOLS];

STATIC_INLINE void gc_dump_page_utilization_data(void) JL_NOTSAFEPOINT
{
    // FIXME: MMTk would have to provide its own stats
}

#define MMTK_GC_PAGE_SZ (1 << 12) // MMTk's page size is defined in mmtk-core constants

JL_DLLEXPORT uint64_t jl_get_pg_size(void)
{
    return MMTK_GC_PAGE_SZ;
}

// Not used by mmtk
// Number of GC threads that may run parallel marking
int jl_n_markthreads;
// Number of GC threads that may run concurrent sweeping (0 or 1)
int jl_n_sweepthreads;
// `tid` of first GC thread
int gc_first_tid;

JL_DLLEXPORT void jl_gc_queue_root(const struct _jl_value_t *ptr) JL_NOTSAFEPOINT
{
    mmtk_unreachable();
}

JL_DLLEXPORT void jl_gc_queue_multiroot(const struct _jl_value_t *root, const void *stored,
                                        struct _jl_datatype_t *dt) JL_NOTSAFEPOINT
{
    mmtk_unreachable();
}

// marking
// ---

JL_DLLEXPORT int jl_gc_mark_queue_obj(jl_ptls_t ptls, jl_value_t *obj)
{
    mmtk_unreachable();
    return 0;
}
JL_DLLEXPORT void jl_gc_mark_queue_objarray(jl_ptls_t ptls, jl_value_t *parent,
                                            jl_value_t **objs, size_t nobjs)
{
    mmtk_unreachable();
}

JL_DLLEXPORT size_t jl_gc_max_internal_obj_size(void)
{
    // TODO: meaningful for MMTk?
    return GC_MAX_SZCLASS;
}

JL_DLLEXPORT void jl_gc_schedule_foreign_sweepfunc(jl_ptls_t ptls, jl_value_t *obj)
{
    // FIXME: do we need to implement this?
}

// gc-debug functions
// ---

JL_DLLEXPORT jl_taggedvalue_t *jl_gc_find_taggedvalue_pool(char *p, size_t *osize_p)
{
    return NULL;
}

void jl_gc_debug_critical_error(void) JL_NOTSAFEPOINT
{
}

int gc_is_collector_thread(int tid) JL_NOTSAFEPOINT
{
    return 0;
}

void jl_gc_debug_print_status(void) JL_NOTSAFEPOINT
{
    // May not be accurate but should be helpful enough
    uint64_t pool_count = gc_num.poolalloc;
    uint64_t big_count = gc_num.bigalloc;
    jl_safe_printf("Allocations: %" PRIu64 " "
                   "(Pool: %" PRIu64 "; Big: %" PRIu64 "); GC: %d\n",
                   pool_count + big_count, pool_count, big_count, gc_num.pause);
}

JL_DLLEXPORT size_t jl_gc_external_obj_hdr_size(void)
{
    return sizeof(bigval_t);
}

void jl_print_gc_stats(JL_STREAM *s)
{
}

JL_DLLEXPORT int jl_gc_enable_conservative_gc_support(void)
{
    return 0;
}

JL_DLLEXPORT int jl_gc_conservative_gc_support_enabled(void)
{
    return 0;
}

// TODO: if this is needed, it can be added in MMTk
JL_DLLEXPORT jl_value_t *jl_gc_internal_obj_base_ptr(void *p)
{
    return NULL;
}

#ifdef __cplusplus
}
#endif

#endif // MMTK_GC
