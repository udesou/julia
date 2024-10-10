#ifdef MMTK_GC

#ifdef __cplusplus
extern "C" {
#endif

extern jl_mutex_t finalizers_lock;
extern arraylist_t to_finalize;
extern arraylist_t finalizer_list_marked;

JL_EXTENSION typedef struct _bigval_t {
    size_t sz;
#ifdef _P64 // Add padding so that the value is 64-byte aligned
    // (8 pointers of 8 bytes each) - (2 other pointers in struct)
    void *_padding[8 - 2];
#else
    // (16 pointers of 4 bytes each) - (2 other pointers in struct)
    void *_padding[16 - 2];
#endif
    //struct jl_taggedvalue_t <>;
    union {
        uintptr_t header;
        struct {
            uintptr_t gc:2;
        } bits;
    };
    // must be 64-byte aligned here, in 32 & 64 bit modes
} bigval_t;

#ifdef __cplusplus
}
#endif

#endif // MMTK_GC
