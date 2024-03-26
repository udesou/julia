// This file is a part of Julia. License is MIT: https://julialang.org/license

#ifndef JULIA_H
#define JULIA_H

#ifdef LIBRARY_EXPORTS

#ifdef __cplusplus
extern "C" {
#endif

extern int mmtk_object_is_managed_by_mmtk(void* addr);
extern unsigned char mmtk_pin_object(void* obj);
// FIXME: Pinning objects that get hashed in the ptrhash table
// until we implement address space hashing.
#ifdef MMTK_GC
#define PTRHASH_PIN(key) mmtk_pin_object(key);
#else
#define PTRHASH_PIN(key)
#endif

#ifdef __cplusplus
}
#endif

// Generated file, needs to be searched in include paths so that the builddir
// retains priority
#include <jl_internal_funcs.inc>
#undef jl_setjmp
#undef jl_longjmp
#undef jl_egal
#endif

#include "julia_fasttls.h"
#include "libsupport.h"
#include <stdint.h>
#include <string.h>

#include "htable.h"
#include "arraylist.h"
#include "analyzer_annotations.h"

#include <setjmp.h>
#ifndef _OS_WINDOWS_
#  define jl_jmp_buf sigjmp_buf
#  if defined(_CPU_ARM_) || defined(_CPU_PPC_) || defined(_CPU_WASM_)
#    define MAX_ALIGN 8
#  elif defined(_CPU_AARCH64_)
// int128 is 16 bytes aligned on aarch64
#    define MAX_ALIGN 16
#  elif defined(_P64)
// Generically we assume MAX_ALIGN is sizeof(void*)
#    define MAX_ALIGN 8
#  else
#    define MAX_ALIGN 4
#  endif
#else
#  include "win32_ucontext.h"
#  define jl_jmp_buf jmp_buf
#  define MAX_ALIGN 8
#endif

// Define the largest size (bytes) of a properly aligned object that the
// processor family and compiler typically supports without a lock
// (assumed to be at least a pointer size). Since C is bad at handling 16-byte
// types, we currently use 8 here as the default.
#define MAX_ATOMIC_SIZE 8
#define MAX_POINTERATOMIC_SIZE 8

#ifdef _P64
#define NWORDS(sz) (((sz)+7)>>3)
#else
#define NWORDS(sz) (((sz)+3)>>2)
#endif

#if defined(__GNUC__)
#  define JL_NORETURN __attribute__ ((noreturn))
#  define JL_CONST_FUNC __attribute__((const))
#  define JL_USED_FUNC __attribute__((used))
#else
#  define JL_NORETURN
#  define JL_CONST_FUNC
#  define JL_USED_FUNC
#endif

#define container_of(ptr, type, member) \
    ((type *) ((char *)(ptr) - offsetof(type, member)))

typedef struct _jl_taggedvalue_t jl_taggedvalue_t;
typedef struct _jl_tls_states_t *jl_ptls_t;

#ifdef LIBRARY_EXPORTS
#include "uv.h"
#endif
#include "julia_atomics.h"
#include "julia_threads.h"
#include "julia_assert.h"

#ifdef __cplusplus
extern "C" {
#endif

// core data types ------------------------------------------------------------

// the common fields are hidden before the pointer, but the following macro is
// used to indicate which types below are subtypes of jl_value_t
#define JL_DATA_TYPE

typedef struct _jl_value_t jl_value_t;

struct _jl_taggedvalue_bits {
    uintptr_t gc:2;
    uintptr_t in_image:1;
};

JL_EXTENSION struct _jl_taggedvalue_t {
    union {
        uintptr_t header;
        jl_taggedvalue_t *next;
        jl_value_t *type; // 16-byte aligned
        struct _jl_taggedvalue_bits bits;
    };
    // jl_value_t value;
};

#ifdef __clang_gcanalyzer__
JL_DLLEXPORT jl_taggedvalue_t *_jl_astaggedvalue(jl_value_t *v JL_PROPAGATES_ROOT) JL_NOTSAFEPOINT;
#define jl_astaggedvalue(v) _jl_astaggedvalue((jl_value_t*)(v))
jl_value_t *_jl_valueof(jl_taggedvalue_t *tv JL_PROPAGATES_ROOT) JL_NOTSAFEPOINT;
#define jl_valueof(v) _jl_valueof((jl_taggedvalue_t*)(v))
JL_DLLEXPORT jl_value_t *_jl_typeof(jl_value_t *v JL_PROPAGATES_ROOT) JL_NOTSAFEPOINT;
#define jl_typeof(v) _jl_typeof((jl_value_t*)(v))
#else
#define jl_astaggedvalue(v)                                             \
    ((jl_taggedvalue_t*)((char*)(v) - sizeof(jl_taggedvalue_t)))
#define jl_valueof(v)                                           \
    ((jl_value_t*)((char*)(v) + sizeof(jl_taggedvalue_t)))
#define jl_typeof(v)                                                    \
    ((jl_value_t*)(jl_astaggedvalue(v)->header & ~(uintptr_t)15))
#endif
static inline void jl_set_typeof(void *v, void *t) JL_NOTSAFEPOINT
{
    // Do not call this on a value that is already initialized.
    jl_taggedvalue_t *tag = jl_astaggedvalue(v);
    jl_atomic_store_relaxed((_Atomic(jl_value_t*)*)&tag->type, (jl_value_t*)t);
}
#define jl_typeis(v,t) (jl_typeof(v)==(jl_value_t*)(t))

// Symbols are interned strings (hash-consed) stored as an invasive binary tree.
// The string data is nul-terminated and hangs off the end of the struct.
typedef struct _jl_sym_t {
    JL_DATA_TYPE
    _Atomic(struct _jl_sym_t*) left;
    _Atomic(struct _jl_sym_t*) right;
    uintptr_t hash;    // precomputed hash value
    // JL_ATTRIBUTE_ALIGN_PTRSIZE(char name[]);
} jl_sym_t;

// A numbered SSA value, for optimized code analysis and generation
// the `id` is a unique, small number
typedef struct _jl_ssavalue_t {
    JL_DATA_TYPE
    ssize_t id;
} jl_ssavalue_t;

// A SimpleVector is an immutable pointer array
// Data is stored at the end of this variable-length struct.
typedef struct {
    JL_DATA_TYPE
    size_t length;
    // pointer size aligned
    // jl_value_t *data[];
} jl_svec_t;

typedef struct {
    /*
      how - allocation style
      0 = data is inlined, or a foreign pointer we don't manage
      1 = julia-allocated buffer that needs to be marked
      2 = malloc-allocated pointer this array object manages
      3 = has a pointer to the object that owns the data
    */
    uint16_t how:2;
    uint16_t ndims:9;
    uint16_t pooled:1;
    uint16_t ptrarray:1; // representation is pointer array
    uint16_t hasptr:1; // representation has embedded pointers
    uint16_t isshared:1; // data is shared by multiple Arrays
    uint16_t isaligned:1; // data allocated with memalign
} jl_array_flags_t;

JL_EXTENSION typedef struct {
    JL_DATA_TYPE
    void *data;
    size_t length;
    jl_array_flags_t flags;
    uint16_t elsize;  // element size including alignment (dim 1 memory stride)
    uint32_t offset;  // for 1-d only. does not need to get big.
    size_t nrows;
    union {
        // 1d
        size_t maxsize;
        // Nd
        size_t ncols;
    };
    // other dim sizes go here for ndims > 2

    // followed by alignment padding and inline data, or owner pointer
} jl_array_t;

// compute # of extra words needed to store dimensions
STATIC_INLINE int jl_array_ndimwords(uint32_t ndims) JL_NOTSAFEPOINT
{
    return (ndims < 3 ? 0 : ndims-2);
}

typedef struct _jl_datatype_t jl_tupletype_t;
struct _jl_code_instance_t;

// TypeMap is an implicitly defined type
// that can consist of any of the following nodes:
//   typedef TypeMap Union{TypeMapLevel, TypeMapEntry, Nothing}
// it forms a roughly tree-shaped structure, consisting of nodes of TypeMapLevels
// which split the tree when possible, for example based on the key into the tuple type at `offs`
// when key is a leaftype, (but only when the tree has enough entries for this to be
// more efficient than storing them sorted linearly)
// otherwise the leaf entries are stored sorted, linearly
typedef jl_value_t jl_typemap_t;

typedef jl_value_t *(jl_call_t)(jl_value_t*, jl_value_t**, uint32_t, struct _jl_code_instance_t*);
typedef jl_call_t *jl_callptr_t;

// "speccall" calling convention signatures.
// This describes some of the special ABI used by compiled julia functions.
extern jl_call_t jl_fptr_args;
JL_DLLEXPORT extern jl_callptr_t jl_fptr_args_addr;
typedef jl_value_t *(*jl_fptr_args_t)(jl_value_t*, jl_value_t**, uint32_t);

extern jl_call_t jl_fptr_const_return;
JL_DLLEXPORT extern jl_callptr_t jl_fptr_const_return_addr;

extern jl_call_t jl_fptr_sparam;
JL_DLLEXPORT extern jl_callptr_t jl_fptr_sparam_addr;
typedef jl_value_t *(*jl_fptr_sparam_t)(jl_value_t*, jl_value_t**, uint32_t, jl_svec_t*);

extern jl_call_t jl_fptr_interpret_call;
JL_DLLEXPORT extern jl_callptr_t jl_fptr_interpret_call_addr;

typedef struct _jl_method_instance_t jl_method_instance_t;

typedef struct _jl_line_info_node_t {
    struct _jl_module_t *module;
    jl_value_t *method;
    jl_sym_t *file;
    int32_t line;
    int32_t inlined_at;
} jl_line_info_node_t;

// the following mirrors `struct EffectsOverride` in `base/compiler/effects.jl`
typedef union __jl_purity_overrides_t {
    struct {
        uint8_t ipo_consistent          : 1;
        uint8_t ipo_effect_free         : 1;
        uint8_t ipo_nothrow             : 1;
        uint8_t ipo_terminates_globally : 1;
        // Weaker form of `terminates` that asserts
        // that any control flow syntactically in the method
        // is guaranteed to terminate, but does not make
        // assertions about any called functions.
        uint8_t ipo_terminates_locally  : 1;
        uint8_t ipo_notaskstate         : 1;
        uint8_t ipo_inaccessiblememonly : 1;
    } overrides;
    uint8_t bits;
} _jl_purity_overrides_t;

// This type describes a single function body
typedef struct _jl_code_info_t {
    // ssavalue-indexed arrays of properties:
    jl_array_t *code;  // Any array of statements
    jl_value_t *codelocs; // Int32 array of indices into the line table
    jl_value_t *ssavaluetypes; // types of ssa values (or count of them)
    jl_array_t *ssaflags; // flags associated with each statement:
        // 0 = inbounds
        // 1 = inline
        // 2 = noinline
        // 3 = <reserved> strict-ieee (strictfp)
        // 4 = effect-free (may be deleted if unused)
        // 5-6 = <unused>
        // 7 = has out-of-band info
    // miscellaneous data:
    jl_value_t *method_for_inference_limit_heuristics; // optional method used during inference
    jl_value_t *linetable; // Table of locations [TODO: make this volatile like slotnames]
    jl_array_t *slotnames; // names of local variables
    jl_array_t *slotflags;  // local var bit flags
    // the following are optional transient properties (not preserved by compression--as they typically get stored elsewhere):
    jl_value_t *slottypes; // inferred types of slots
    jl_value_t *rettype;
    jl_method_instance_t *parent; // context (optionally, if available, otherwise nothing)
    jl_value_t *edges; // forward edges to method instances that must be invalidated
    size_t min_world;
    size_t max_world;
    // various boolean properties:
    uint8_t inferred;
    uint16_t inlining_cost;
    uint8_t propagate_inbounds;
    uint8_t pure;
    uint8_t has_fcall;
    // uint8 settings
    uint8_t constprop; // 0 = use heuristic; 1 = aggressive; 2 = none
    _jl_purity_overrides_t purity;
} jl_code_info_t;

// This type describes a single method definition, and stores data
// shared by the specializations of a function.
typedef struct _jl_method_t {
    JL_DATA_TYPE
    jl_sym_t *name;  // for error reporting
    struct _jl_module_t *module;
    jl_sym_t *file;
    int32_t line;
    size_t primary_world;
    size_t deleted_world;

    // method's type signature. redundant with TypeMapEntry->specTypes
    jl_value_t *sig;

    // table of all jl_method_instance_t specializations we have
    _Atomic(jl_svec_t*) specializations; // allocated as [hashable, ..., NULL, linear, ....]
    _Atomic(jl_array_t*) speckeyset; // index lookup by hash into specializations

    jl_value_t *slot_syms; // compacted list of slot names (String)
    jl_value_t *external_mt; // reference to the method table this method is part of, null if part of the internal table
    jl_value_t *source;  // original code template (jl_code_info_t, but may be compressed), null for builtins
    _Atomic(struct _jl_method_instance_t*) unspecialized;  // unspecialized executable method instance, or null
    jl_value_t *generator;  // executable code-generating function if available
    jl_array_t *roots;  // pointers in generated code (shared to reduce memory), or null
    // Identify roots by module-of-origin. We only track the module for roots added during incremental compilation.
    // May be NULL if no external roots have been added, otherwise it's a Vector{UInt64}
    jl_array_t *root_blocks;   // RLE (build_id.lo, offset) pairs (even/odd indexing)
    int32_t nroots_sysimg;     // # of roots stored in the system image
    jl_svec_t *ccallable; // svec(rettype, sig) if a ccallable entry point is requested for this

    // cache of specializations of this method for invoke(), i.e.
    // cases where this method was called even though it was not necessarily
    // the most specific for the argument types.
    _Atomic(jl_typemap_t*) invokes;

    // A function that compares two specializations of this method, returning
    // `true` if the first signature is to be considered "smaller" than the
    // second for purposes of recursion analysis. Set to NULL to use
    // the default recursion relation.
    jl_value_t *recursion_relation;

    uint32_t nargs;
    uint32_t called;        // bit flags: whether each of the first 8 arguments is called
    uint32_t nospecialize;  // bit flags: which arguments should not be specialized
    uint32_t nkw;           // # of leading arguments that are actually keyword arguments
                            // of another method.
    uint8_t isva;
    uint8_t pure;
    uint8_t is_for_opaque_closure;
    // uint8 settings
    uint8_t constprop;     // 0x00 = use heuristic; 0x01 = aggressive; 0x02 = none

    // Override the conclusions of inter-procedural effect analysis,
    // forcing the conclusion to always true.
    _jl_purity_overrides_t purity;

// hidden fields:
    // lock for modifications to the method
    jl_mutex_t writelock;
} jl_method_t;

// This type is a placeholder to cache data for a specType signature specialization of a Method
// can can be used as a unique dictionary key representation of a call to a particular Method
// with a particular set of argument types
struct _jl_method_instance_t {
    JL_DATA_TYPE
    union {
        jl_value_t *value; // generic accessor
        struct _jl_module_t *module; // this is a toplevel thunk
        jl_method_t *method; // method this is specialized from
    } def; // pointer back to the context for this code
    jl_value_t *specTypes;  // argument types this was specialized for
    jl_svec_t *sparam_vals; // static parameter values, indexed by def.method->sparam_syms
    jl_value_t *uninferred; // cached uncompressed code, for generated functions, top-level thunks, or the interpreter
    jl_array_t *backedges; // list of method-instances which call this method-instance; `invoke` records (invokesig, caller) pairs
    jl_array_t *callbacks; // list of callback functions to inform external caches about invalidations
    _Atomic(struct _jl_code_instance_t*) cache;
    uint8_t inInference; // flags to tell if inference is running on this object
    uint8_t cache_with_orig; // !cache_with_specTypes
    uint8_t precompiled; // true if this instance was generated by an explicit `precompile(...)` call
};

// OpaqueClosure
typedef struct jl_opaque_closure_t {
    JL_DATA_TYPE
    jl_value_t *captures;
    size_t world;
    jl_method_t *source;
    jl_fptr_args_t invoke;
    void *specptr;
} jl_opaque_closure_t;

// This type represents an executable operation
typedef struct _jl_code_instance_t {
    JL_DATA_TYPE
    jl_method_instance_t *def; // method this is specialized from
    _Atomic(struct _jl_code_instance_t*) next; // pointer to the next cache entry

    // world range for which this object is valid to use
    size_t min_world;
    size_t max_world;

    // inference state cache
    jl_value_t *rettype; // return type for fptr
    jl_value_t *rettype_const; // inferred constant return value, or null
    _Atomic(jl_value_t *) inferred; // inferred jl_code_info_t, or jl_nothing, or null
    //TODO: jl_array_t *edges; // stored information about edges from this object
    //TODO: uint8_t absolute_max; // whether true max world is unknown

    // purity results
#ifdef JL_USE_ANON_UNIONS_FOR_PURITY_FLAGS
    // see also encode_effects() and decode_effects() in `base/compiler/effects.jl`,
    union {
        uint32_t ipo_purity_bits;
        struct {
            uint8_t ipo_consistent          : 2;
            uint8_t ipo_effect_free         : 2;
            uint8_t ipo_nothrow             : 2;
            uint8_t ipo_terminates          : 2;
            uint8_t ipo_nonoverlayed        : 1;
            uint8_t ipo_notaskstate         : 2;
            uint8_t ipo_inaccessiblememonly : 2;
        } ipo_purity_flags;
    };
    union {
        uint32_t purity_bits;
        struct {
            uint8_t consistent          : 2;
            uint8_t effect_free         : 2;
            uint8_t nothrow             : 2;
            uint8_t terminates          : 2;
            uint8_t nonoverlayed        : 1;
            uint8_t notaskstate         : 2;
            uint8_t inaccessiblememonly : 2;
        } purity_flags;
    };
#else
    uint32_t ipo_purity_bits;
    _Atomic(uint32_t) purity_bits;
#endif
    jl_value_t *argescapes; // escape information of call arguments

    // compilation state cache
    _Atomic(uint8_t) specsigflags; // & 0b001 == specptr is a specialized function signature for specTypes->rettype
                                   // & 0b010 == invokeptr matches specptr
                                   // & 0b100 == From image
    _Atomic(uint8_t) precompile;  // if set, this will be added to the output system image
    uint8_t relocatability;  // nonzero if all roots are built into sysimg or tagged by module key
    _Atomic(jl_callptr_t) invoke; // jlcall entry point
    union _jl_generic_specptr_t {
        _Atomic(void*) fptr;
        _Atomic(jl_fptr_args_t) fptr1;
        // 2 constant
        _Atomic(jl_fptr_sparam_t) fptr3;
        // 4 interpreter
    } specptr; // private data for `jlcall entry point
} jl_code_instance_t;

// all values are callable as Functions
typedef jl_value_t jl_function_t;

typedef struct {
    JL_DATA_TYPE
    jl_sym_t *name;
    jl_value_t *lb;   // lower bound
    jl_value_t *ub;   // upper bound
} jl_tvar_t;

// UnionAll type (iterated union over all values of a variable in certain bounds)
// written `body where lb<:var<:ub`
typedef struct {
    JL_DATA_TYPE
    jl_tvar_t *var;
    jl_value_t *body;
} jl_unionall_t;

// represents the "name" part of a DataType, describing the syntactic structure
// of a type and storing all data common to different instantiations of the type,
// including a cache for hash-consed allocation of DataType objects.
typedef struct {
    JL_DATA_TYPE
    jl_sym_t *name;
    struct _jl_module_t *module;
    jl_svec_t *names;  // field names
    const uint32_t *atomicfields; // if any fields are atomic, we record them here
    const uint32_t *constfields; // if any fields are const, we record them here
    // `wrapper` is either the only instantiation of the type (if no parameters)
    // or a UnionAll accepting parameters to make an instantiation.
    jl_value_t *wrapper;
    _Atomic(jl_value_t*) Typeofwrapper;  // cache for Type{wrapper}
    _Atomic(jl_svec_t*) cache;        // sorted array
    _Atomic(jl_svec_t*) linearcache;  // unsorted array
    struct _jl_methtable_t *mt;
    jl_array_t *partial;     // incomplete instantiations of this type
    intptr_t hash;
    int32_t n_uninitialized;
    // type properties
    uint8_t abstract:1;
    uint8_t mutabl:1;
    uint8_t mayinlinealloc:1;
    uint8_t _reserved:5;
    uint8_t max_methods; // override for inference's max_methods setting (0 = no additional limit or relaxation)
} jl_typename_t;

typedef struct {
    JL_DATA_TYPE
    jl_value_t *a;
    jl_value_t *b;
} jl_uniontype_t;

// in little-endian, isptr is always the first bit, avoiding the need for a branch in computing isptr
typedef struct {
    uint8_t isptr:1;
    uint8_t size:7;
    uint8_t offset;   // offset relative to data start, excluding type tag
} jl_fielddesc8_t;

typedef struct {
    uint16_t isptr:1;
    uint16_t size:15;
    uint16_t offset;   // offset relative to data start, excluding type tag
} jl_fielddesc16_t;

typedef struct {
    uint32_t isptr:1;
    uint32_t size:31;
    uint32_t offset;   // offset relative to data start, excluding type tag
} jl_fielddesc32_t;

typedef struct {
    uint32_t size;
    uint32_t nfields;
    uint32_t npointers; // number of pointers embedded inside
    int32_t first_ptr; // index of the first pointer (or -1)
    uint16_t alignment; // strictest alignment over all fields
    uint16_t haspadding : 1; // has internal undefined bytes
    uint16_t fielddesc_type : 2; // 0 -> 8, 1 -> 16, 2 -> 32, 3 -> foreign type
    // union {
    //     jl_fielddesc8_t field8[nfields];
    //     jl_fielddesc16_t field16[nfields];
    //     jl_fielddesc32_t field32[nfields];
    // };
    // union { // offsets relative to data start in words
    //     uint8_t ptr8[npointers];
    //     uint16_t ptr16[npointers];
    //     uint32_t ptr32[npointers];
    // };
} jl_datatype_layout_t;

typedef struct _jl_datatype_t {
    JL_DATA_TYPE
    jl_typename_t *name;
    struct _jl_datatype_t *super;
    jl_svec_t *parameters;
    jl_svec_t *types;
    jl_value_t *instance;  // for singletons
    const jl_datatype_layout_t *layout;
    // memoized properties
    uint32_t hash;
    uint8_t hasfreetypevars:1; // majority part of isconcrete computation
    uint8_t isconcretetype:1; // whether this type can have instances
    uint8_t isdispatchtuple:1; // aka isleaftupletype
    uint8_t isbitstype:1; // relevant query for C-api and type-parameters
    uint8_t zeroinit:1; // if one or more fields requires zero-initialization
    uint8_t has_concrete_subtype:1; // If clear, no value will have this datatype
    uint8_t cached_by_hash:1; // stored in hash-based set cache (instead of linear cache)
    uint8_t isprimitivetype:1; // whether this is declared with 'primitive type' keyword (sized, no fields, and immutable)
} jl_datatype_t;

typedef struct _jl_vararg_t {
    JL_DATA_TYPE
    jl_value_t *T;
    jl_value_t *N;
} jl_vararg_t;

typedef struct {
    JL_DATA_TYPE
    jl_value_t *value;
} jl_weakref_t;

typedef struct {
    // not first-class
    jl_sym_t *name;
    _Atomic(jl_value_t*) value;
    _Atomic(jl_value_t*) globalref;  // cached GlobalRef for this binding
    struct _jl_module_t* owner;  // for individual imported bindings -- TODO: make _Atomic
    _Atomic(jl_value_t*) ty;  // binding type
    uint8_t constp:1;
    uint8_t exportp:1;
    uint8_t imported:1;
    uint8_t deprecated:2; // 0=not deprecated, 1=renamed, 2=moved to another package
} jl_binding_t;

typedef struct {
    uint64_t hi;
    uint64_t lo;
} jl_uuid_t;

typedef struct _jl_module_t {
    JL_DATA_TYPE
    jl_sym_t *name;
    struct _jl_module_t *parent;
    // hidden fields:
    htable_t bindings;
    arraylist_t usings;  // modules with all bindings potentially imported
    jl_uuid_t build_id;
    jl_uuid_t uuid;
    size_t primary_world;
    _Atomic(uint32_t) counter;
    int32_t nospecialize;  // global bit flags: initialization for new methods
    int8_t optlevel;
    int8_t compile;
    int8_t infer;
    uint8_t istopmod;
    int8_t max_methods;
    jl_mutex_t lock;
} jl_module_t;

typedef struct {
    jl_module_t *mod;
    jl_sym_t *name;
    // Not serialized. Caches the value of jl_get_binding(mod, name).
    jl_binding_t *bnd_cache;
} jl_globalref_t;

// one Type-to-Value entry
typedef struct _jl_typemap_entry_t {
    JL_DATA_TYPE
    _Atomic(struct _jl_typemap_entry_t*) next; // invasive linked list
    jl_tupletype_t *sig; // the type signature for this entry
    jl_tupletype_t *simplesig; // a simple signature for fast rejection
    jl_svec_t *guardsigs;
    size_t min_world;
    size_t max_world;
    union {
        jl_value_t *value; // generic accessor
        jl_method_instance_t *linfo; // [nullable] for guard entries
        jl_method_t *method;
    } func;
    // memoized properties of sig:
    int8_t isleafsig; // isleaftype(sig) & !any(isType, sig) : unsorted and very fast
    int8_t issimplesig; // all(isleaftype | isAny | isType | isVararg, sig) : sorted and fast
    int8_t va; // isVararg(sig)
} jl_typemap_entry_t;

// one level in a TypeMap tree (each level splits on a type at a given offset)
typedef struct _jl_typemap_level_t {
    JL_DATA_TYPE
    // these vectors contains vectors of more levels in their intended visit order
    // with an index that gives the functionality of a sorted dict.
    // next split may be on Type{T} as LeafTypes then TypeName's parents up to Any
    // next split may be on LeafType
    // next split may be on TypeName
    _Atomic(jl_array_t*) arg1; // contains LeafType
    _Atomic(jl_array_t*) targ; // contains Type{LeafType}
    _Atomic(jl_array_t*) name1; // contains non-abstract TypeName, for parents up to (excluding) Any
    _Atomic(jl_array_t*) tname; // contains a dict of Type{TypeName}, for parents up to Any
    // next a linear list of things too complicated at this level for analysis (no more levels)
    _Atomic(jl_typemap_entry_t*) linear;
    // finally, start a new level if the type at offs is Any
    _Atomic(jl_typemap_t*) any;
} jl_typemap_level_t;

// contains the TypeMap for one Type
typedef struct _jl_methtable_t {
    JL_DATA_TYPE
    jl_sym_t *name; // sometimes a hack used by serialization to handle kwsorter
    _Atomic(jl_typemap_t*) defs;
    _Atomic(jl_array_t*) leafcache;
    _Atomic(jl_typemap_t*) cache;
    intptr_t max_args;  // max # of non-vararg arguments in a signature
    jl_module_t *module; // used for incremental serialization to locate original binding
    jl_array_t *backedges; // (sig, caller::MethodInstance) pairs
    jl_mutex_t writelock;
    uint8_t offs;  // 0, or 1 to skip splitting typemap on first (function) argument
    uint8_t frozen; // whether this accepts adding new methods
} jl_methtable_t;

typedef struct {
    JL_DATA_TYPE
    jl_sym_t *head;
    jl_array_t *args;
} jl_expr_t;

typedef struct {
    JL_DATA_TYPE
    jl_tupletype_t *spec_types;
    jl_svec_t *sparams;
    jl_method_t *method;
    // A bool on the julia side, but can be temporarily 0x2 as a sentinel
    // during construction.
    uint8_t fully_covers;
} jl_method_match_t;

// constants and type objects -------------------------------------------------

// kinds
extern JL_DLLIMPORT jl_datatype_t *jl_typeofbottom_type JL_GLOBALLY_ROOTED;
extern JL_DLLIMPORT jl_datatype_t *jl_datatype_type JL_GLOBALLY_ROOTED;
extern JL_DLLIMPORT jl_datatype_t *jl_uniontype_type JL_GLOBALLY_ROOTED;
extern JL_DLLIMPORT jl_datatype_t *jl_unionall_type JL_GLOBALLY_ROOTED;
extern JL_DLLIMPORT jl_datatype_t *jl_tvar_type JL_GLOBALLY_ROOTED;

extern JL_DLLIMPORT jl_datatype_t *jl_any_type JL_GLOBALLY_ROOTED;
extern JL_DLLIMPORT jl_unionall_t *jl_type_type JL_GLOBALLY_ROOTED;
extern JL_DLLIMPORT jl_datatype_t *jl_typename_type JL_GLOBALLY_ROOTED;
extern JL_DLLIMPORT jl_typename_t *jl_type_typename JL_GLOBALLY_ROOTED;
extern JL_DLLIMPORT jl_datatype_t *jl_symbol_type JL_GLOBALLY_ROOTED;
extern JL_DLLIMPORT jl_datatype_t *jl_ssavalue_type JL_GLOBALLY_ROOTED;
extern JL_DLLIMPORT jl_datatype_t *jl_abstractslot_type JL_GLOBALLY_ROOTED;
extern JL_DLLIMPORT jl_datatype_t *jl_slotnumber_type JL_GLOBALLY_ROOTED;
extern JL_DLLIMPORT jl_datatype_t *jl_typedslot_type JL_GLOBALLY_ROOTED;
extern JL_DLLIMPORT jl_datatype_t *jl_argument_type JL_GLOBALLY_ROOTED;
extern JL_DLLIMPORT jl_datatype_t *jl_const_type JL_GLOBALLY_ROOTED;
extern JL_DLLIMPORT jl_datatype_t *jl_partial_struct_type JL_GLOBALLY_ROOTED;
extern JL_DLLIMPORT jl_datatype_t *jl_partial_opaque_type JL_GLOBALLY_ROOTED;
extern JL_DLLIMPORT jl_datatype_t *jl_interconditional_type JL_GLOBALLY_ROOTED;
extern JL_DLLIMPORT jl_datatype_t *jl_method_match_type JL_GLOBALLY_ROOTED;
extern JL_DLLIMPORT jl_datatype_t *jl_simplevector_type JL_GLOBALLY_ROOTED;
extern JL_DLLIMPORT jl_typename_t *jl_tuple_typename JL_GLOBALLY_ROOTED;
extern JL_DLLIMPORT jl_typename_t *jl_vecelement_typename JL_GLOBALLY_ROOTED;
extern JL_DLLIMPORT jl_datatype_t *jl_anytuple_type JL_GLOBALLY_ROOTED;
extern JL_DLLIMPORT jl_datatype_t *jl_emptytuple_type JL_GLOBALLY_ROOTED;
#define jl_tuple_type jl_anytuple_type
extern JL_DLLIMPORT jl_unionall_t *jl_anytuple_type_type JL_GLOBALLY_ROOTED;
extern JL_DLLIMPORT jl_datatype_t *jl_vararg_type JL_GLOBALLY_ROOTED;
extern JL_DLLIMPORT jl_datatype_t *jl_function_type JL_GLOBALLY_ROOTED;
extern JL_DLLIMPORT jl_datatype_t *jl_builtin_type JL_GLOBALLY_ROOTED;
extern JL_DLLIMPORT jl_unionall_t *jl_opaque_closure_type JL_GLOBALLY_ROOTED;
extern JL_DLLIMPORT jl_typename_t *jl_opaque_closure_typename JL_GLOBALLY_ROOTED;

extern JL_DLLIMPORT jl_value_t *jl_bottom_type JL_GLOBALLY_ROOTED;
extern JL_DLLIMPORT jl_datatype_t *jl_method_instance_type JL_GLOBALLY_ROOTED;
extern JL_DLLIMPORT jl_datatype_t *jl_code_instance_type JL_GLOBALLY_ROOTED;
extern JL_DLLIMPORT jl_datatype_t *jl_code_info_type JL_GLOBALLY_ROOTED;
extern JL_DLLIMPORT jl_datatype_t *jl_method_type JL_GLOBALLY_ROOTED;
extern JL_DLLIMPORT jl_datatype_t *jl_module_type JL_GLOBALLY_ROOTED;
extern JL_DLLIMPORT jl_unionall_t *jl_abstractarray_type JL_GLOBALLY_ROOTED;
extern JL_DLLIMPORT jl_unionall_t *jl_densearray_type JL_GLOBALLY_ROOTED;
extern JL_DLLIMPORT jl_unionall_t *jl_array_type JL_GLOBALLY_ROOTED;
extern JL_DLLIMPORT jl_typename_t *jl_array_typename JL_GLOBALLY_ROOTED;
extern JL_DLLIMPORT jl_datatype_t *jl_weakref_type JL_GLOBALLY_ROOTED;
extern JL_DLLIMPORT jl_datatype_t *jl_abstractstring_type JL_GLOBALLY_ROOTED;
extern JL_DLLIMPORT jl_datatype_t *jl_string_type JL_GLOBALLY_ROOTED;
extern JL_DLLIMPORT jl_datatype_t *jl_errorexception_type JL_GLOBALLY_ROOTED;
extern JL_DLLIMPORT jl_datatype_t *jl_argumenterror_type JL_GLOBALLY_ROOTED;
extern JL_DLLIMPORT jl_datatype_t *jl_loaderror_type JL_GLOBALLY_ROOTED;
extern JL_DLLIMPORT jl_datatype_t *jl_initerror_type JL_GLOBALLY_ROOTED;
extern JL_DLLIMPORT jl_datatype_t *jl_typeerror_type JL_GLOBALLY_ROOTED;
extern JL_DLLIMPORT jl_datatype_t *jl_methoderror_type JL_GLOBALLY_ROOTED;
extern JL_DLLIMPORT jl_datatype_t *jl_undefvarerror_type JL_GLOBALLY_ROOTED;
extern JL_DLLIMPORT jl_datatype_t *jl_atomicerror_type JL_GLOBALLY_ROOTED;
extern JL_DLLIMPORT jl_datatype_t *jl_lineinfonode_type JL_GLOBALLY_ROOTED;
extern JL_DLLIMPORT jl_value_t *jl_stackovf_exception JL_GLOBALLY_ROOTED;
extern JL_DLLIMPORT jl_value_t *jl_memory_exception JL_GLOBALLY_ROOTED;
extern JL_DLLIMPORT jl_value_t *jl_readonlymemory_exception JL_GLOBALLY_ROOTED;
extern JL_DLLIMPORT jl_value_t *jl_diverror_exception JL_GLOBALLY_ROOTED;
extern JL_DLLIMPORT jl_value_t *jl_undefref_exception JL_GLOBALLY_ROOTED;
extern JL_DLLIMPORT jl_value_t *jl_interrupt_exception JL_GLOBALLY_ROOTED;
extern JL_DLLIMPORT jl_datatype_t *jl_boundserror_type JL_GLOBALLY_ROOTED;
extern JL_DLLIMPORT jl_value_t *jl_an_empty_vec_any JL_GLOBALLY_ROOTED;
extern JL_DLLIMPORT jl_value_t *jl_an_empty_string JL_GLOBALLY_ROOTED;

extern JL_DLLIMPORT jl_datatype_t *jl_bool_type JL_GLOBALLY_ROOTED;
extern JL_DLLIMPORT jl_datatype_t *jl_char_type JL_GLOBALLY_ROOTED;
extern JL_DLLIMPORT jl_datatype_t *jl_int8_type JL_GLOBALLY_ROOTED;
extern JL_DLLIMPORT jl_datatype_t *jl_uint8_type JL_GLOBALLY_ROOTED;
extern JL_DLLIMPORT jl_datatype_t *jl_int16_type JL_GLOBALLY_ROOTED;
extern JL_DLLIMPORT jl_datatype_t *jl_uint16_type JL_GLOBALLY_ROOTED;
extern JL_DLLIMPORT jl_datatype_t *jl_int32_type JL_GLOBALLY_ROOTED;
extern JL_DLLIMPORT jl_datatype_t *jl_uint32_type JL_GLOBALLY_ROOTED;
extern JL_DLLIMPORT jl_datatype_t *jl_int64_type JL_GLOBALLY_ROOTED;
extern JL_DLLIMPORT jl_datatype_t *jl_uint64_type JL_GLOBALLY_ROOTED;
extern JL_DLLIMPORT jl_datatype_t *jl_float16_type JL_GLOBALLY_ROOTED;
extern JL_DLLIMPORT jl_datatype_t *jl_float32_type JL_GLOBALLY_ROOTED;
extern JL_DLLIMPORT jl_datatype_t *jl_float64_type JL_GLOBALLY_ROOTED;
extern JL_DLLIMPORT jl_datatype_t *jl_floatingpoint_type JL_GLOBALLY_ROOTED;
extern JL_DLLIMPORT jl_datatype_t *jl_number_type JL_GLOBALLY_ROOTED;
extern JL_DLLIMPORT jl_datatype_t *jl_void_type JL_GLOBALLY_ROOTED;  // deprecated
extern JL_DLLIMPORT jl_datatype_t *jl_nothing_type JL_GLOBALLY_ROOTED;
extern JL_DLLIMPORT jl_datatype_t *jl_signed_type JL_GLOBALLY_ROOTED;
extern JL_DLLIMPORT jl_datatype_t *jl_voidpointer_type JL_GLOBALLY_ROOTED;
extern JL_DLLIMPORT jl_datatype_t *jl_uint8pointer_type JL_GLOBALLY_ROOTED;
extern JL_DLLIMPORT jl_unionall_t *jl_pointer_type JL_GLOBALLY_ROOTED;
extern JL_DLLIMPORT jl_unionall_t *jl_llvmpointer_type JL_GLOBALLY_ROOTED;
extern JL_DLLIMPORT jl_unionall_t *jl_ref_type JL_GLOBALLY_ROOTED;
extern JL_DLLIMPORT jl_typename_t *jl_pointer_typename JL_GLOBALLY_ROOTED;
extern JL_DLLIMPORT jl_typename_t *jl_llvmpointer_typename JL_GLOBALLY_ROOTED;
extern JL_DLLIMPORT jl_typename_t *jl_namedtuple_typename JL_GLOBALLY_ROOTED;
extern JL_DLLIMPORT jl_unionall_t *jl_namedtuple_type JL_GLOBALLY_ROOTED;
extern JL_DLLIMPORT jl_datatype_t *jl_task_type JL_GLOBALLY_ROOTED;
extern JL_DLLIMPORT jl_value_t *jl_pair_type JL_GLOBALLY_ROOTED;

extern JL_DLLIMPORT jl_value_t *jl_array_uint8_type JL_GLOBALLY_ROOTED;
extern JL_DLLIMPORT jl_value_t *jl_array_any_type JL_GLOBALLY_ROOTED;
extern JL_DLLIMPORT jl_value_t *jl_array_symbol_type JL_GLOBALLY_ROOTED;
extern JL_DLLIMPORT jl_value_t *jl_array_int32_type JL_GLOBALLY_ROOTED;
extern JL_DLLIMPORT jl_value_t *jl_array_uint64_type JL_GLOBALLY_ROOTED;
extern JL_DLLIMPORT jl_datatype_t *jl_expr_type JL_GLOBALLY_ROOTED;
extern JL_DLLIMPORT jl_datatype_t *jl_globalref_type JL_GLOBALLY_ROOTED;
extern JL_DLLIMPORT jl_datatype_t *jl_linenumbernode_type JL_GLOBALLY_ROOTED;
extern JL_DLLIMPORT jl_datatype_t *jl_gotonode_type JL_GLOBALLY_ROOTED;
extern JL_DLLIMPORT jl_datatype_t *jl_gotoifnot_type JL_GLOBALLY_ROOTED;
extern JL_DLLIMPORT jl_datatype_t *jl_returnnode_type JL_GLOBALLY_ROOTED;
extern JL_DLLIMPORT jl_datatype_t *jl_phinode_type JL_GLOBALLY_ROOTED;
extern JL_DLLIMPORT jl_datatype_t *jl_pinode_type JL_GLOBALLY_ROOTED;
extern JL_DLLIMPORT jl_datatype_t *jl_phicnode_type JL_GLOBALLY_ROOTED;
extern JL_DLLIMPORT jl_datatype_t *jl_upsilonnode_type JL_GLOBALLY_ROOTED;
extern JL_DLLIMPORT jl_datatype_t *jl_quotenode_type JL_GLOBALLY_ROOTED;
extern JL_DLLIMPORT jl_datatype_t *jl_newvarnode_type JL_GLOBALLY_ROOTED;
extern JL_DLLIMPORT jl_datatype_t *jl_intrinsic_type JL_GLOBALLY_ROOTED;
extern JL_DLLIMPORT jl_datatype_t *jl_methtable_type JL_GLOBALLY_ROOTED;
extern JL_DLLIMPORT jl_datatype_t *jl_typemap_level_type JL_GLOBALLY_ROOTED;
extern JL_DLLIMPORT jl_datatype_t *jl_typemap_entry_type JL_GLOBALLY_ROOTED;

extern JL_DLLIMPORT jl_svec_t *jl_emptysvec JL_GLOBALLY_ROOTED;
extern JL_DLLIMPORT jl_value_t *jl_emptytuple JL_GLOBALLY_ROOTED;
extern JL_DLLIMPORT jl_value_t *jl_true JL_GLOBALLY_ROOTED;
extern JL_DLLIMPORT jl_value_t *jl_false JL_GLOBALLY_ROOTED;
extern JL_DLLIMPORT jl_value_t *jl_nothing JL_GLOBALLY_ROOTED;
extern JL_DLLIMPORT jl_value_t *jl_kwcall_func JL_GLOBALLY_ROOTED;

// gc -------------------------------------------------------------------------

struct _jl_gcframe_t {
    size_t nroots;
    struct _jl_gcframe_t *prev;
    // actual roots go here
    // final value indicates id from where it's been pushed from
};

// NOTE: it is the caller's responsibility to make sure arguments are
// rooted such that the gc can see them on the stack.
// `foo(f(), g())` is not safe,
// since the result of `f()` is not rooted during the call to `g()`,
// and the arguments to foo are not gc-protected during the call to foo.
// foo can't do anything about it, so the caller must do:
// jl_value_t *x=NULL, *y=NULL; JL_GC_PUSH2(&x, &y);
// x = f(); y = g(); foo(x, y)

#define jl_pgcstack (jl_current_task->gcstack)

#ifndef MMTK_GC
#define JL_GC_ENCODE_PUSHARGS(n)   (((size_t)(n))<<2)
#define JL_GC_ENCODE_PUSH(n)       ((((size_t)(n))<<2)|1)

#define JL_GC_ENCODE_PUSHARGS_NO_TPIN(n)  JL_GC_ENCODE_PUSHARGS(n)
#define JL_GC_ENCODE_PUSH_NO_TPIN(n)      JL_GC_ENCODE_PUSH(n)
#else
// these are transitively pinning
#define JL_GC_ENCODE_PUSHARGS(n)   (((size_t)(n))<<3)
#define JL_GC_ENCODE_PUSH(n)       ((((size_t)(n))<<3)|1)

// these only pin the root object itself
#define JL_GC_ENCODE_PUSHARGS_NO_TPIN(n)   (((size_t)(n))<<3|4)
#define JL_GC_ENCODE_PUSH_NO_TPIN(n)       ((((size_t)(n))<<3)|5)
#endif

#ifdef __clang_gcanalyzer__

// When running with the analyzer make these real function calls, that are
// easier to detect in the analyzer
extern void JL_GC_PUSH1(void *) JL_NOTSAFEPOINT;
extern void JL_GC_PUSH2(void *, void *) JL_NOTSAFEPOINT;
extern void JL_GC_PUSH3(void *, void *, void *)  JL_NOTSAFEPOINT;
extern void JL_GC_PUSH4(void *, void *, void *, void *)  JL_NOTSAFEPOINT;
extern void JL_GC_PUSH5(void *, void *, void *, void *, void *)  JL_NOTSAFEPOINT;
extern void JL_GC_PUSH7(void *, void *, void *, void *, void *, void *, void *)  JL_NOTSAFEPOINT;
extern void JL_GC_PUSH8(void *, void *, void *, void *, void *, void *, void *, void *)  JL_NOTSAFEPOINT;
extern void _JL_GC_PUSHARGS(jl_value_t **, size_t) JL_NOTSAFEPOINT;
// This is necessary, because otherwise the analyzer considers this undefined
// behavior and terminates the exploration
#define JL_GC_PUSHARGS(rts_var, n)     \
  rts_var = (jl_value_t **)alloca(sizeof(void*) * (n)); \
  memset(rts_var, 0, sizeof(void*) * (n)); \
  _JL_GC_PUSHARGS(rts_var, (n));

extern void JL_GC_POP() JL_NOTSAFEPOINT;

#else

#define JL_GC_PUSH1(arg1, locid)                                                                               \
  void *__gc_stkf[] = {(void*)JL_GC_ENCODE_PUSH(1), jl_pgcstack, arg1, locid};                                 \
  jl_pgcstack = (jl_gcframe_t*)__gc_stkf;

#define JL_GC_PUSH2(arg1, arg2, locid)                                                                         \
  void *__gc_stkf[] = {(void*)JL_GC_ENCODE_PUSH(2), jl_pgcstack, arg1, arg2, locid};                           \
  jl_pgcstack = (jl_gcframe_t*)__gc_stkf;

#define JL_GC_PUSH3(arg1, arg2, arg3, locid)                                                                   \
  void *__gc_stkf[] = {(void*)JL_GC_ENCODE_PUSH(3), jl_pgcstack, arg1, arg2, arg3, locid};                     \
  jl_pgcstack = (jl_gcframe_t*)__gc_stkf;

#define JL_GC_PUSH4(arg1, arg2, arg3, arg4, locid)                                                             \
  void *__gc_stkf[] = {(void*)JL_GC_ENCODE_PUSH(4), jl_pgcstack, arg1, arg2, arg3, arg4, locid};               \
  jl_pgcstack = (jl_gcframe_t*)__gc_stkf;

#define JL_GC_PUSH5(arg1, arg2, arg3, arg4, arg5, locid)                                                       \
  void *__gc_stkf[] = {(void*)JL_GC_ENCODE_PUSH(5), jl_pgcstack, arg1, arg2, arg3, arg4, arg5, locid};         \
  jl_pgcstack = (jl_gcframe_t*)__gc_stkf;

#define JL_GC_PUSH6(arg1, arg2, arg3, arg4, arg5, arg6, locid)                                                 \
  void *__gc_stkf[] = {(void*)JL_GC_ENCODE_PUSH(6), jl_pgcstack, arg1, arg2, arg3, arg4, arg5, arg6, locid};   \
  jl_pgcstack = (jl_gcframe_t*)__gc_stkf;

#define JL_GC_PUSH7(arg1, arg2, arg3, arg4, arg5, arg6, arg7, locid)                                           \
  void *__gc_stkf[] = {(void*)JL_GC_ENCODE_PUSH(7), jl_pgcstack, arg1, arg2, arg3, arg4, arg5, arg6, arg7, locid}; \
  jl_pgcstack = (jl_gcframe_t*)__gc_stkf;

#define JL_GC_PUSH8(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, locid)                                     \
  void *__gc_stkf[] = {(void*)JL_GC_ENCODE_PUSH(8), jl_pgcstack, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, locid}; \
  jl_pgcstack = (jl_gcframe_t*)__gc_stkf;

#define JL_GC_PUSHARGS(rts_var,n, locid)                                                                       \
  rts_var = ((jl_value_t**)alloca(((n+1)+2)*sizeof(jl_value_t*)))+2;                                      \
  ((void**)rts_var)[-2] = (void*)JL_GC_ENCODE_PUSHARGS(n);                                              \
  ((void**)rts_var)[-1] = jl_pgcstack;                                                                  \
  memset((void*)rts_var, 0, (n+1)*sizeof(jl_value_t*));                                                   \
  ((void**)rts_var)[n] = locid;                                                                       \
  jl_pgcstack = (jl_gcframe_t*)&(((void**)rts_var)[-2])

#define JL_GC_POP() (jl_pgcstack = jl_pgcstack->prev)

#endif

#ifdef MMTK_GC
// these are pinning roots: only the root object needs to be pinned as opposed to
// the functions above which are transitively pinning
#define JL_GC_PUSH1_NO_TPIN(arg1)                                                                                     \
  void *__gc_stkf[] = {(void*)JL_GC_ENCODE_PUSH_NO_TPIN(1), jl_pgcstack, arg1};                                  \
  jl_pgcstack = (jl_gcframe_t*)__gc_stkf;

#define JL_GC_PUSH2_NO_TPIN(arg1, arg2)                                                                               \
  void *__gc_stkf[] = {(void*)JL_GC_ENCODE_PUSH_NO_TPIN(2), jl_pgcstack, arg1, arg2};                            \
  jl_pgcstack = (jl_gcframe_t*)__gc_stkf;

#define JL_GC_PUSH3_NO_TPIN(arg1, arg2, arg3)                                                                         \
  void *__gc_stkf[] = {(void*)JL_GC_ENCODE_PUSH_NO_TPIN(3), jl_pgcstack, arg1, arg2, arg3};                       \
  jl_pgcstack = (jl_gcframe_t*)__gc_stkf;

#define JL_GC_PUSH4_NO_TPIN(arg1, arg2, arg3, arg4)                                                                   \
  void *__gc_stkf[] = {(void*)JL_GC_ENCODE_PUSH_NO_TPIN(4), jl_pgcstack, arg1, arg2, arg3, arg4};                \
  jl_pgcstack = (jl_gcframe_t*)__gc_stkf;

#define JL_GC_PUSH5_NO_TPIN(arg1, arg2, arg3, arg4, arg5)                                                             \
  void *__gc_stkf[] = {(void*)JL_GC_ENCODE_PUSH_NO_TPIN(5), jl_pgcstack, arg1, arg2, arg3, arg4, arg5};           \
  jl_pgcstack = (jl_gcframe_t*)__gc_stkf;

#define JL_GC_PUSH6_NO_TPIN(arg1, arg2, arg3, arg4, arg5, arg6)                                                       \
  void *__gc_stkf[] = {(void*)JL_GC_ENCODE_PUSH_NO_TPIN(6), jl_pgcstack, arg1, arg2, arg3, arg4, arg5, arg6};     \
  jl_pgcstack = (jl_gcframe_t*)__gc_stkf;

#define JL_GC_PUSH7_NO_TPIN(arg1, arg2, arg3, arg4, arg5, arg6, arg7)                                                    \
  void *__gc_stkf[] = {(void*)JL_GC_ENCODE_PUSH_NO_TPIN(7), jl_pgcstack, arg1, arg2, arg3, arg4, arg5, arg6, arg7};  \
  jl_pgcstack = (jl_gcframe_t*)__gc_stkf;

#define JL_GC_PUSH8_NO_TPIN(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8)                                     \
  void *__gc_stkf[] = {(void*)JL_GC_ENCODE_PUSH_NO_TPIN(8), jl_pgcstack, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8}; \
  jl_pgcstack = (jl_gcframe_t*)__gc_stkf;

#define JL_GC_PUSHARGS_NO_TPIN(rts_var,n)                                                                       \
  rts_var = ((jl_value_t**)alloca(((n)+2)*sizeof(jl_value_t*)))+2;                                      \
  ((void**)rts_var)[-2] = (void*)JL_GC_ENCODE_PUSHARGS_NO_TPIN(n);                                              \
  ((void**)rts_var)[-1] = jl_pgcstack;                                                                  \
  memset((void*)rts_var, 0, (n)*sizeof(jl_value_t*));                                                   \
  jl_pgcstack = (jl_gcframe_t*)&(((void**)rts_var)[-2])
#else
// When not using MMTk, default to the stock functions
#define JL_GC_PUSH1_NO_TPIN(arg1) JL_GC_PUSH1(arg1)

#define JL_GC_PUSH2_NO_TPIN(arg1, arg2) JL_GC_PUSH2(arg1, arg2)

#define JL_GC_PUSH3_NO_TPIN(arg1, arg2, arg3) JL_GC_PUSH3(arg1, arg2, arg3)

#define JL_GC_PUSH4_NO_TPIN(arg1, arg2, arg3, arg4) JL_GC_PUSH4(arg1, arg2, arg3, arg4)

#define JL_GC_PUSH5_NO_TPIN(arg1, arg2, arg3, arg4, arg5) JL_GC_PUSH5(arg1, arg2, arg3, arg4, arg5)

#define JL_GC_PUSH6_NO_TPIN(arg1, arg2, arg3, arg4, arg5, arg6) JL_GC_PUSH6(arg1, arg2, arg3, arg4, arg5, arg6)

#define JL_GC_PUSH7_NO_TPIN(arg1, arg2, arg3, arg4, arg5, arg6, arg7) JL_GC_PUSH7(arg1, arg2, arg3, arg4, arg5, arg6, arg7)

#define JL_GC_PUSH8_NO_TPIN(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8) JL_GC_PUSH8(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8)

#define JL_GC_PUSHARGS_NO_TPIN(rts_var,n) JL_GC_PUSHARGS(rts_var,n)
#endif

JL_DLLEXPORT int jl_gc_enable(int on);
JL_DLLEXPORT int jl_gc_is_enabled(void);

typedef enum {
    JL_GC_AUTO = 0,         // use heuristics to determine the collection type
    JL_GC_FULL = 1,         // force a full collection
    JL_GC_INCREMENTAL = 2,  // force an incremental collection
} jl_gc_collection_t;

JL_DLLEXPORT void jl_gc_collect(jl_gc_collection_t);

JL_DLLEXPORT void jl_gc_add_finalizer(jl_value_t *v, jl_function_t *f) JL_NOTSAFEPOINT;
JL_DLLEXPORT void jl_gc_add_ptr_finalizer(jl_ptls_t ptls, jl_value_t *v, void *f) JL_NOTSAFEPOINT;
JL_DLLEXPORT void jl_gc_add_quiescent(jl_ptls_t ptls, void **v, void *f) JL_NOTSAFEPOINT;
JL_DLLEXPORT void jl_finalize(jl_value_t *o);
JL_DLLEXPORT jl_weakref_t *jl_gc_new_weakref(jl_value_t *value);
JL_DLLEXPORT jl_value_t *jl_gc_alloc_0w(void);
JL_DLLEXPORT jl_value_t *jl_gc_alloc_1w(void);
JL_DLLEXPORT jl_value_t *jl_gc_alloc_2w(void);
JL_DLLEXPORT jl_value_t *jl_gc_alloc_3w(void);
JL_DLLEXPORT jl_value_t *jl_gc_allocobj(size_t sz);
JL_DLLEXPORT void *jl_malloc_stack(size_t *bufsz, struct _jl_task_t *owner) JL_NOTSAFEPOINT;
JL_DLLEXPORT void jl_free_stack(void *stkbuf, size_t bufsz);
JL_DLLEXPORT void jl_gc_use(jl_value_t *a);
JL_DLLEXPORT uint64_t jl_gc_get_max_memory(void);

JL_DLLEXPORT void jl_clear_malloc_data(void);

// GC write barriers
JL_DLLEXPORT void jl_gc_queue_root(const jl_value_t *root) JL_NOTSAFEPOINT;
JL_DLLEXPORT void jl_gc_queue_multiroot(const jl_value_t *root, const jl_value_t *stored) JL_NOTSAFEPOINT;

#ifndef MMTK_GC
STATIC_INLINE void jl_gc_wb(const void *parent, const void *ptr) JL_NOTSAFEPOINT
{
    // parent and ptr isa jl_value_t*
    if (__unlikely(jl_astaggedvalue(parent)->bits.gc == 3 && // parent is old and not in remset
                   (jl_astaggedvalue(ptr)->bits.gc & 1) == 0)) // ptr is young
        jl_gc_queue_root((jl_value_t*)parent);
}

STATIC_INLINE void jl_gc_wb_back(const void *ptr) JL_NOTSAFEPOINT // ptr isa jl_value_t*
{
    // if ptr is old
    if (__unlikely(jl_astaggedvalue(ptr)->bits.gc == 3)) {
        jl_gc_queue_root((jl_value_t*)ptr);
    }
}

STATIC_INLINE void jl_gc_multi_wb(const void *parent, const jl_value_t *ptr) JL_NOTSAFEPOINT
{
    // ptr is an immutable object
    if (__likely(jl_astaggedvalue(parent)->bits.gc != 3))
        return; // parent is young or in remset
    if (__likely(jl_astaggedvalue(ptr)->bits.gc == 3))
        return; // ptr is old and not in remset (thus it does not point to young)
    jl_datatype_t *dt = (jl_datatype_t*)jl_typeof(ptr);
    const jl_datatype_layout_t *ly = dt->layout;
    if (ly->npointers)
        jl_gc_queue_multiroot((jl_value_t*)parent, ptr);
}
#else  // MMTK_GC
STATIC_INLINE void mmtk_gc_wb(const void *parent, const void *ptr) JL_NOTSAFEPOINT;
STATIC_INLINE void mmtk_gc_wb_binding(const void *parent, const void *ptr) JL_NOTSAFEPOINT;

STATIC_INLINE void jl_gc_wb(const void *parent, const void *ptr) JL_NOTSAFEPOINT
{
    mmtk_gc_wb(parent, ptr);
}

STATIC_INLINE void jl_gc_wb_back(const void *ptr) JL_NOTSAFEPOINT // ptr isa jl_value_t*
{
    mmtk_gc_wb(ptr, (void*)0);
}

STATIC_INLINE void jl_gc_multi_wb(const void *parent, const jl_value_t *ptr) JL_NOTSAFEPOINT
{
    mmtk_gc_wb(parent, (void*)0);
}
#endif // MMTK_GC

JL_DLLEXPORT void *jl_gc_managed_malloc(size_t sz);
JL_DLLEXPORT void *jl_gc_managed_realloc(void *d, size_t sz, size_t oldsz,
                                         int isaligned, jl_value_t *owner);
JL_DLLEXPORT void jl_gc_safepoint(void);

void *mtarraylist_get(small_arraylist_t *_a, size_t idx) JL_NOTSAFEPOINT;
size_t mtarraylist_length(small_arraylist_t *_a) JL_NOTSAFEPOINT;
void mtarraylist_add(small_arraylist_t *_a, void *elt, size_t idx) JL_NOTSAFEPOINT;
void mtarraylist_push(small_arraylist_t *_a, void *elt) JL_NOTSAFEPOINT;

// object accessors -----------------------------------------------------------

#define jl_svec_len(t)              (((jl_svec_t*)(t))->length)
#define jl_svec_set_len_unsafe(t,n) (((jl_svec_t*)(t))->length=(n))
#define jl_svec_data(t) ((jl_value_t**)((char*)(t) + sizeof(jl_svec_t)))

#ifdef __clang_gcanalyzer__
STATIC_INLINE jl_value_t *jl_svecref(void *t JL_PROPAGATES_ROOT, size_t i) JL_NOTSAFEPOINT;
STATIC_INLINE jl_value_t *jl_svecset(
    void *t JL_ROOTING_ARGUMENT JL_PROPAGATES_ROOT,
    size_t i, void *x JL_ROOTED_ARGUMENT) JL_NOTSAFEPOINT;
#else
STATIC_INLINE jl_value_t *jl_svecref(void *t JL_PROPAGATES_ROOT, size_t i) JL_NOTSAFEPOINT
{
    assert(jl_typeis(t,jl_simplevector_type));
    assert(i < jl_svec_len(t));
    // while svec is supposedly immutable, in practice we sometimes publish it first
    // and set the values lazily
    return jl_atomic_load_relaxed((_Atomic(jl_value_t*)*)jl_svec_data(t) + i);
}
STATIC_INLINE jl_value_t *jl_svecset(
    void *t JL_ROOTING_ARGUMENT JL_PROPAGATES_ROOT,
    size_t i, void *x JL_ROOTED_ARGUMENT) JL_NOTSAFEPOINT
{
    assert(jl_typeis(t,jl_simplevector_type));
    assert(i < jl_svec_len(t));
    // TODO: while svec is supposedly immutable, in practice we sometimes publish it first
    // and set the values lazily. Those users should be using jl_atomic_store_release here.
    jl_svec_data(t)[i] = (jl_value_t*)x;
    jl_gc_wb(t, x);
    return (jl_value_t*)x;
}
#endif

#define jl_array_len(a)   (((jl_array_t*)(a))->length)
#define jl_array_data(a)  ((void*)((jl_array_t*)(a))->data)
#define jl_array_dim(a,i) ((&((jl_array_t*)(a))->nrows)[i])
#define jl_array_dim0(a)  (((jl_array_t*)(a))->nrows)
#define jl_array_nrows(a) (((jl_array_t*)(a))->nrows)
#define jl_array_ndims(a) ((int32_t)(((jl_array_t*)a)->flags.ndims))
#define jl_array_data_owner_offset(ndims) (offsetof(jl_array_t,ncols) + sizeof(size_t)*(1+jl_array_ndimwords(ndims))) // in bytes
#define jl_array_data_owner(a) (*((jl_value_t**)((char*)a + jl_array_data_owner_offset(jl_array_ndims(a)))))

JL_DLLEXPORT char *jl_array_typetagdata(jl_array_t *a) JL_NOTSAFEPOINT;

#ifdef __clang_gcanalyzer__
jl_value_t **jl_array_ptr_data(jl_array_t *a JL_PROPAGATES_ROOT) JL_NOTSAFEPOINT;
STATIC_INLINE jl_value_t *jl_array_ptr_ref(void *a JL_PROPAGATES_ROOT, size_t i) JL_NOTSAFEPOINT;
STATIC_INLINE jl_value_t *jl_array_ptr_set(
    void *a JL_ROOTING_ARGUMENT, size_t i,
    void *x JL_ROOTED_ARGUMENT) JL_NOTSAFEPOINT;
#else
#define jl_array_ptr_data(a)  ((jl_value_t**)((jl_array_t*)(a))->data)
STATIC_INLINE jl_value_t *jl_array_ptr_ref(void *a JL_PROPAGATES_ROOT, size_t i) JL_NOTSAFEPOINT
{
    assert(((jl_array_t*)a)->flags.ptrarray);
    assert(i < jl_array_len(a));
    return jl_atomic_load_relaxed(((_Atomic(jl_value_t*)*)(jl_array_data(a))) + i);
}
STATIC_INLINE jl_value_t *jl_array_ptr_set(
    void *a JL_ROOTING_ARGUMENT, size_t i,
    void *x JL_ROOTED_ARGUMENT) JL_NOTSAFEPOINT
{
    assert(((jl_array_t*)a)->flags.ptrarray);
    assert(i < jl_array_len(a));
    jl_atomic_store_release(((_Atomic(jl_value_t*)*)(jl_array_data(a))) + i, (jl_value_t*)x);
    if (x) {
        if (((jl_array_t*)a)->flags.how == 3) {
            a = jl_array_data_owner(a);
        }
        jl_gc_wb(a, x);
    }
    return (jl_value_t*)x;
}
#endif

STATIC_INLINE uint8_t jl_array_uint8_ref(void *a, size_t i) JL_NOTSAFEPOINT
{
    assert(i < jl_array_len(a));
    assert(jl_typeis(a, jl_array_uint8_type));
    return ((uint8_t*)(jl_array_data(a)))[i];
}
STATIC_INLINE void jl_array_uint8_set(void *a, size_t i, uint8_t x) JL_NOTSAFEPOINT
{
    assert(i < jl_array_len(a));
    assert(jl_typeis(a, jl_array_uint8_type));
    ((uint8_t*)(jl_array_data(a)))[i] = x;
}

#define jl_exprarg(e,n) jl_array_ptr_ref(((jl_expr_t*)(e))->args, n)
#define jl_exprargset(e, n, v) jl_array_ptr_set(((jl_expr_t*)(e))->args, n, v)
#define jl_expr_nargs(e) jl_array_len(((jl_expr_t*)(e))->args)

#define jl_fieldref(s,i) jl_get_nth_field(((jl_value_t*)(s)),i)
#define jl_fieldref_noalloc(s,i) jl_get_nth_field_noalloc(((jl_value_t*)(s)),i)
#define jl_nfields(v)    jl_datatype_nfields(jl_typeof(v))

// Not using jl_fieldref to avoid allocations
#define jl_linenode_line(x) (((intptr_t*)(x))[0])
#define jl_linenode_file(x) (((jl_value_t**)(x))[1])
#define jl_slot_number(x) (((intptr_t*)(x))[0])
#define jl_typedslot_get_type(x) (((jl_value_t**)(x))[1])
#define jl_gotonode_label(x) (((intptr_t*)(x))[0])
#define jl_gotoifnot_cond(x) (((jl_value_t**)(x))[0])
#define jl_gotoifnot_label(x) (((intptr_t*)(x))[1])
#define jl_globalref_mod(s) (*(jl_module_t**)(s))
#define jl_globalref_name(s) (((jl_sym_t**)(s))[1])
#define jl_quotenode_value(x) (((jl_value_t**)x)[0])
#define jl_returnnode_value(x) (((jl_value_t**)x)[0])

#define jl_nparams(t)  jl_svec_len(((jl_datatype_t*)(t))->parameters)
#define jl_tparam0(t)  jl_svecref(((jl_datatype_t*)(t))->parameters, 0)
#define jl_tparam1(t)  jl_svecref(((jl_datatype_t*)(t))->parameters, 1)
#define jl_tparam(t,i) jl_svecref(((jl_datatype_t*)(t))->parameters, i)

// get a pointer to the data in a datatype
#define jl_data_ptr(v)  ((jl_value_t**)v)

#define jl_string_data(s) ((char*)s + sizeof(void*))
#define jl_string_len(s)  (*(size_t*)s)

#define jl_gf_mtable(f) (((jl_datatype_t*)jl_typeof(f))->name->mt)
#define jl_gf_name(f)   (jl_gf_mtable(f)->name)

// struct type info
JL_DLLEXPORT jl_svec_t *jl_compute_fieldtypes(jl_datatype_t *st JL_PROPAGATES_ROOT, void *stack);
#define jl_get_fieldtypes(st) ((st)->types ? (st)->types : jl_compute_fieldtypes((st), NULL))
STATIC_INLINE jl_svec_t *jl_field_names(jl_datatype_t *st) JL_NOTSAFEPOINT
{
    return st->name->names;
}
STATIC_INLINE jl_value_t *jl_field_type(jl_datatype_t *st JL_PROPAGATES_ROOT, size_t i)
{
    return jl_svecref(jl_get_fieldtypes(st), i);
}
STATIC_INLINE jl_value_t *jl_field_type_concrete(jl_datatype_t *st JL_PROPAGATES_ROOT, size_t i) JL_NOTSAFEPOINT
{
    assert(st->types);
    return jl_svecref(st->types, i);
}

#define jl_datatype_size(t)    (((jl_datatype_t*)t)->layout->size)
#define jl_datatype_align(t)   (((jl_datatype_t*)t)->layout->alignment)
#define jl_datatype_nbits(t)   ((((jl_datatype_t*)t)->layout->size)*8)
#define jl_datatype_nfields(t) (((jl_datatype_t*)(t))->layout->nfields)

JL_DLLEXPORT void *jl_symbol_name(jl_sym_t *s);
// inline version with strong type check to detect typos in a `->name` chain
STATIC_INLINE char *jl_symbol_name_(jl_sym_t *s) JL_NOTSAFEPOINT
{
    return (char*)s + LLT_ALIGN(sizeof(jl_sym_t), sizeof(void*));
}
#define jl_symbol_name(s) jl_symbol_name_(s)

static inline uint32_t jl_fielddesc_size(int8_t fielddesc_type) JL_NOTSAFEPOINT
{
    assert(fielddesc_type >= 0 && fielddesc_type <= 2);
    return 2 << fielddesc_type;
    //if (fielddesc_type == 0) {
    //    return sizeof(jl_fielddesc8_t);
    //}
    //else if (fielddesc_type == 1) {
    //    return sizeof(jl_fielddesc16_t);
    //}
    //else {
    //    return sizeof(jl_fielddesc32_t);
    //}
}

#define jl_dt_layout_fields(d) ((const char*)(d) + sizeof(jl_datatype_layout_t))
static inline const char *jl_dt_layout_ptrs(const jl_datatype_layout_t *l) JL_NOTSAFEPOINT
{
    return jl_dt_layout_fields(l) + jl_fielddesc_size(l->fielddesc_type) * l->nfields;
}

#define DEFINE_FIELD_ACCESSORS(f)                                             \
    static inline uint32_t jl_field_##f(jl_datatype_t *st,                    \
                                        int i) JL_NOTSAFEPOINT                \
    {                                                                         \
        const jl_datatype_layout_t *ly = st->layout;                          \
        assert(i >= 0 && (size_t)i < ly->nfields);                            \
        if (ly->fielddesc_type == 0) {                                        \
            return ((const jl_fielddesc8_t*)jl_dt_layout_fields(ly))[i].f;    \
        }                                                                     \
        else if (ly->fielddesc_type == 1) {                                   \
            return ((const jl_fielddesc16_t*)jl_dt_layout_fields(ly))[i].f;   \
        }                                                                     \
        else {                                                                \
            assert(ly->fielddesc_type == 2);                                  \
            return ((const jl_fielddesc32_t*)jl_dt_layout_fields(ly))[i].f;   \
        }                                                                     \
    }                                                                         \

DEFINE_FIELD_ACCESSORS(offset)
DEFINE_FIELD_ACCESSORS(size)
#undef DEFINE_FIELD_ACCESSORS

static inline int jl_field_isptr(jl_datatype_t *st, int i) JL_NOTSAFEPOINT
{
    const jl_datatype_layout_t *ly = st->layout;
    assert(i >= 0 && (size_t)i < ly->nfields);
    return ((const jl_fielddesc8_t*)(jl_dt_layout_fields(ly) + jl_fielddesc_size(ly->fielddesc_type) * i))->isptr;
}

static inline uint32_t jl_ptr_offset(jl_datatype_t *st, int i) JL_NOTSAFEPOINT
{
    const jl_datatype_layout_t *ly = st->layout;
    assert(i >= 0 && (size_t)i < ly->npointers);
    const void *ptrs = jl_dt_layout_ptrs(ly);
    if (ly->fielddesc_type == 0) {
        return ((const uint8_t*)ptrs)[i];
    }
    else if (ly->fielddesc_type == 1) {
        return ((const uint16_t*)ptrs)[i];
    }
    else {
        assert(ly->fielddesc_type == 2);
        return ((const uint32_t*)ptrs)[i];
    }
}

static inline int jl_field_isatomic(jl_datatype_t *st, int i) JL_NOTSAFEPOINT
{
    const uint32_t *atomicfields = st->name->atomicfields;
    if (atomicfields != NULL) {
        if (atomicfields[i / 32] & (1 << (i % 32)))
            return 1;
    }
    return 0;
}

static inline int jl_field_isconst(jl_datatype_t *st, int i) JL_NOTSAFEPOINT
{
    jl_typename_t *tn = st->name;
    if (!tn->mutabl)
        return 1;
    const uint32_t *constfields = tn->constfields;
    if (constfields != NULL) {
        if (constfields[i / 32] & (1 << (i % 32)))
            return 1;
    }
    return 0;
}


static inline int jl_is_layout_opaque(const jl_datatype_layout_t *l) JL_NOTSAFEPOINT
{
    return l->nfields == 0 && l->npointers > 0;
}

// basic predicates -----------------------------------------------------------
#define jl_is_nothing(v)     (((jl_value_t*)(v)) == ((jl_value_t*)jl_nothing))
#define jl_is_tuple(v)       (((jl_datatype_t*)jl_typeof(v))->name == jl_tuple_typename)
#define jl_is_namedtuple(v)  (((jl_datatype_t*)jl_typeof(v))->name == jl_namedtuple_typename)
#define jl_is_svec(v)        jl_typeis(v,jl_simplevector_type)
#define jl_is_simplevector(v) jl_is_svec(v)
#define jl_is_datatype(v)    jl_typeis(v,jl_datatype_type)
#define jl_is_mutable(t)     (((jl_datatype_t*)t)->name->mutabl)
#define jl_is_mutable_datatype(t) (jl_is_datatype(t) && (((jl_datatype_t*)t)->name->mutabl))
#define jl_is_immutable(t)   (!((jl_datatype_t*)t)->name->mutabl)
#define jl_is_immutable_datatype(t) (jl_is_datatype(t) && (!((jl_datatype_t*)t)->name->mutabl))
#define jl_is_uniontype(v)   jl_typeis(v,jl_uniontype_type)
#define jl_is_typevar(v)     jl_typeis(v,jl_tvar_type)
#define jl_is_unionall(v)    jl_typeis(v,jl_unionall_type)
#define jl_is_typename(v)    jl_typeis(v,jl_typename_type)
#define jl_is_int8(v)        jl_typeis(v,jl_int8_type)
#define jl_is_int16(v)       jl_typeis(v,jl_int16_type)
#define jl_is_int32(v)       jl_typeis(v,jl_int32_type)
#define jl_is_int64(v)       jl_typeis(v,jl_int64_type)
#define jl_is_uint8(v)       jl_typeis(v,jl_uint8_type)
#define jl_is_uint16(v)      jl_typeis(v,jl_uint16_type)
#define jl_is_uint32(v)      jl_typeis(v,jl_uint32_type)
#define jl_is_uint64(v)      jl_typeis(v,jl_uint64_type)
#define jl_is_bool(v)        jl_typeis(v,jl_bool_type)
#define jl_is_symbol(v)      jl_typeis(v,jl_symbol_type)
#define jl_is_ssavalue(v)    jl_typeis(v,jl_ssavalue_type)
#define jl_is_slot(v)        (jl_typeis(v,jl_slotnumber_type) || jl_typeis(v,jl_typedslot_type))
#define jl_is_expr(v)        jl_typeis(v,jl_expr_type)
#define jl_is_globalref(v)   jl_typeis(v,jl_globalref_type)
#define jl_is_gotonode(v)    jl_typeis(v,jl_gotonode_type)
#define jl_is_gotoifnot(v)   jl_typeis(v,jl_gotoifnot_type)
#define jl_is_returnnode(v)  jl_typeis(v,jl_returnnode_type)
#define jl_is_argument(v)    jl_typeis(v,jl_argument_type)
#define jl_is_pinode(v)      jl_typeis(v,jl_pinode_type)
#define jl_is_phinode(v)     jl_typeis(v,jl_phinode_type)
#define jl_is_phicnode(v)    jl_typeis(v,jl_phicnode_type)
#define jl_is_upsilonnode(v) jl_typeis(v,jl_upsilonnode_type)
#define jl_is_quotenode(v)   jl_typeis(v,jl_quotenode_type)
#define jl_is_newvarnode(v)  jl_typeis(v,jl_newvarnode_type)
#define jl_is_linenode(v)    jl_typeis(v,jl_linenumbernode_type)
#define jl_is_method_instance(v) jl_typeis(v,jl_method_instance_type)
#define jl_is_code_instance(v) jl_typeis(v,jl_code_instance_type)
#define jl_is_code_info(v)   jl_typeis(v,jl_code_info_type)
#define jl_is_method(v)      jl_typeis(v,jl_method_type)
#define jl_is_module(v)      jl_typeis(v,jl_module_type)
#define jl_is_mtable(v)      jl_typeis(v,jl_methtable_type)
#define jl_is_task(v)        jl_typeis(v,jl_task_type)
#define jl_is_string(v)      jl_typeis(v,jl_string_type)
#define jl_is_cpointer(v)    jl_is_cpointer_type(jl_typeof(v))
#define jl_is_pointer(v)     jl_is_cpointer_type(jl_typeof(v))
#define jl_is_uint8pointer(v)jl_typeis(v,jl_uint8pointer_type)
#define jl_is_llvmpointer(v) (((jl_datatype_t*)jl_typeof(v))->name == jl_llvmpointer_typename)
#define jl_is_intrinsic(v)   jl_typeis(v,jl_intrinsic_type)
#define jl_array_isbitsunion(a) (!(((jl_array_t*)(a))->flags.ptrarray) && jl_is_uniontype(jl_tparam0(jl_typeof(a))))

JL_DLLEXPORT int jl_subtype(jl_value_t *a, jl_value_t *b);

STATIC_INLINE int jl_is_kind(jl_value_t *v) JL_NOTSAFEPOINT
{
    return (v==(jl_value_t*)jl_uniontype_type || v==(jl_value_t*)jl_datatype_type ||
            v==(jl_value_t*)jl_unionall_type || v==(jl_value_t*)jl_typeofbottom_type);
}

STATIC_INLINE int jl_is_type(jl_value_t *v) JL_NOTSAFEPOINT
{
    return jl_is_kind(jl_typeof(v));
}

STATIC_INLINE int jl_is_primitivetype(void *v) JL_NOTSAFEPOINT
{
    return (jl_is_datatype(v) && ((jl_datatype_t*)(v))->isprimitivetype);
}

STATIC_INLINE int jl_is_structtype(void *v) JL_NOTSAFEPOINT
{
    return (jl_is_datatype(v) &&
            !((jl_datatype_t*)(v))->name->abstract &&
            !((jl_datatype_t*)(v))->isprimitivetype);
}

STATIC_INLINE int jl_isbits(void *t) JL_NOTSAFEPOINT // corresponding to isbitstype() in julia
{
    return (jl_is_datatype(t) && ((jl_datatype_t*)t)->isbitstype);
}

STATIC_INLINE int jl_is_datatype_singleton(jl_datatype_t *d) JL_NOTSAFEPOINT
{
    return (d->instance != NULL);
}

STATIC_INLINE int jl_is_abstracttype(void *v) JL_NOTSAFEPOINT
{
    return (jl_is_datatype(v) && ((jl_datatype_t*)(v))->name->abstract);
}

STATIC_INLINE int jl_is_array_type(void *t) JL_NOTSAFEPOINT
{
    return (jl_is_datatype(t) &&
            ((jl_datatype_t*)(t))->name == jl_array_typename);
}

STATIC_INLINE int jl_is_array(void *v) JL_NOTSAFEPOINT
{
    jl_value_t *t = jl_typeof(v);
    return jl_is_array_type(t);
}

STATIC_INLINE jl_value_t *jl_array_owner(jl_array_t *a JL_PROPAGATES_ROOT) JL_NOTSAFEPOINT
{
    if (a->flags.how == 3) {
        a = (jl_array_t*)jl_array_data_owner(a);
        assert(jl_is_string(a) || a->flags.how != 3);
    }
    return (jl_value_t*)a;
}

STATIC_INLINE int jl_is_opaque_closure_type(void *t) JL_NOTSAFEPOINT
{
    return (jl_is_datatype(t) &&
            ((jl_datatype_t*)(t))->name == jl_opaque_closure_typename);
}

STATIC_INLINE int jl_is_opaque_closure(void *v) JL_NOTSAFEPOINT
{
    jl_value_t *t = jl_typeof(v);
    return jl_is_opaque_closure_type(t);
}

STATIC_INLINE int jl_is_cpointer_type(jl_value_t *t) JL_NOTSAFEPOINT
{
    return (jl_is_datatype(t) &&
            ((jl_datatype_t*)(t))->name == ((jl_datatype_t*)jl_pointer_type->body)->name);
}

STATIC_INLINE int jl_is_llvmpointer_type(jl_value_t *t) JL_NOTSAFEPOINT
{
    return (jl_is_datatype(t) &&
            ((jl_datatype_t*)(t))->name == jl_llvmpointer_typename);
}

STATIC_INLINE int jl_is_abstract_ref_type(jl_value_t *t) JL_NOTSAFEPOINT
{
    return (jl_is_datatype(t) &&
            ((jl_datatype_t*)(t))->name == ((jl_datatype_t*)jl_ref_type->body)->name);
}

STATIC_INLINE int jl_is_tuple_type(void *t) JL_NOTSAFEPOINT
{
    return (jl_is_datatype(t) &&
            ((jl_datatype_t*)(t))->name == jl_tuple_typename);
}

STATIC_INLINE int jl_is_namedtuple_type(void *t) JL_NOTSAFEPOINT
{
    return (jl_is_datatype(t) &&
            ((jl_datatype_t*)(t))->name == jl_namedtuple_typename);
}

STATIC_INLINE int jl_is_vecelement_type(jl_value_t* t) JL_NOTSAFEPOINT
{
    return (jl_is_datatype(t) &&
            ((jl_datatype_t*)(t))->name == jl_vecelement_typename);
}

STATIC_INLINE int jl_is_type_type(jl_value_t *v) JL_NOTSAFEPOINT
{
    return (jl_is_datatype(v) &&
            ((jl_datatype_t*)(v))->name == ((jl_datatype_t*)jl_type_type->body)->name);
}

STATIC_INLINE int jl_is_array_zeroinit(jl_array_t *a) JL_NOTSAFEPOINT
{
    if (a->flags.ptrarray || a->flags.hasptr)
        return 1;
    jl_value_t *elty = jl_tparam0(jl_typeof(a));
    return jl_is_datatype(elty) && ((jl_datatype_t*)elty)->zeroinit;
}

// object identity
JL_DLLEXPORT int jl_egal(const jl_value_t *a JL_MAYBE_UNROOTED, const jl_value_t *b JL_MAYBE_UNROOTED) JL_NOTSAFEPOINT;
JL_DLLEXPORT int jl_egal__bits(const jl_value_t *a JL_MAYBE_UNROOTED, const jl_value_t *b JL_MAYBE_UNROOTED, jl_datatype_t *dt) JL_NOTSAFEPOINT;
JL_DLLEXPORT int jl_egal__special(const jl_value_t *a JL_MAYBE_UNROOTED, const jl_value_t *b JL_MAYBE_UNROOTED, jl_datatype_t *dt) JL_NOTSAFEPOINT;
JL_DLLEXPORT int jl_egal__unboxed(const jl_value_t *a JL_MAYBE_UNROOTED, const jl_value_t *b JL_MAYBE_UNROOTED, jl_datatype_t *dt) JL_NOTSAFEPOINT;
JL_DLLEXPORT uintptr_t jl_object_id(jl_value_t *v) JL_NOTSAFEPOINT;

STATIC_INLINE int jl_egal__unboxed_(const jl_value_t *a JL_MAYBE_UNROOTED, const jl_value_t *b JL_MAYBE_UNROOTED, jl_datatype_t *dt) JL_NOTSAFEPOINT
{
    if (dt->name->mutabl) {
        if (dt == jl_simplevector_type || dt == jl_string_type || dt == jl_datatype_type)
            return jl_egal__special(a, b, dt);
        return 0;
    }
    return jl_egal__bits(a, b, dt);
}

STATIC_INLINE int jl_egal_(const jl_value_t *a JL_MAYBE_UNROOTED, const jl_value_t *b JL_MAYBE_UNROOTED) JL_NOTSAFEPOINT
{
    if (a == b)
        return 1;
    jl_datatype_t *dt = (jl_datatype_t*)jl_typeof(a);
    if (dt != (jl_datatype_t*)jl_typeof(b))
        return 0;
    return jl_egal__unboxed_(a, b, dt);
}
#define jl_egal(a, b) jl_egal_((a), (b))

// type predicates and basic operations
JL_DLLEXPORT int jl_type_equality_is_identity(jl_value_t *t1, jl_value_t *t2) JL_NOTSAFEPOINT;
JL_DLLEXPORT int jl_has_free_typevars(jl_value_t *v) JL_NOTSAFEPOINT;
JL_DLLEXPORT int jl_has_typevar(jl_value_t *t, jl_tvar_t *v) JL_NOTSAFEPOINT;
JL_DLLEXPORT int jl_has_typevar_from_unionall(jl_value_t *t, jl_unionall_t *ua);
JL_DLLEXPORT int jl_subtype_env_size(jl_value_t *t);
JL_DLLEXPORT int jl_subtype_env(jl_value_t *x, jl_value_t *y, jl_value_t **env, int envsz);
JL_DLLEXPORT int jl_isa(jl_value_t *a, jl_value_t *t);
JL_DLLEXPORT int jl_types_equal(jl_value_t *a, jl_value_t *b);
JL_DLLEXPORT int jl_is_not_broken_subtype(jl_value_t *a, jl_value_t *b);
JL_DLLEXPORT jl_value_t *jl_type_union(jl_value_t **ts, size_t n);
JL_DLLEXPORT jl_value_t *jl_type_intersection(jl_value_t *a, jl_value_t *b);
JL_DLLEXPORT int jl_has_empty_intersection(jl_value_t *x, jl_value_t *y);
JL_DLLEXPORT jl_value_t *jl_type_unionall(jl_tvar_t *v, jl_value_t *body);
JL_DLLEXPORT const char *jl_typename_str(jl_value_t *v) JL_NOTSAFEPOINT;
JL_DLLEXPORT const char *jl_typeof_str(jl_value_t *v) JL_NOTSAFEPOINT;
JL_DLLEXPORT int jl_type_morespecific(jl_value_t *a, jl_value_t *b);

STATIC_INLINE int jl_is_dispatch_tupletype(jl_value_t *v) JL_NOTSAFEPOINT
{
    return jl_is_datatype(v) && ((jl_datatype_t*)v)->isdispatchtuple;
}

STATIC_INLINE int jl_is_concrete_type(jl_value_t *v) JL_NOTSAFEPOINT
{
    return jl_is_datatype(v) && ((jl_datatype_t*)v)->isconcretetype;
}

JL_DLLEXPORT int jl_isa_compileable_sig(jl_tupletype_t *type, jl_svec_t *sparams, jl_method_t *definition);

// type constructors
JL_DLLEXPORT jl_typename_t *jl_new_typename_in(jl_sym_t *name, jl_module_t *inmodule, int abstract, int mutabl);
JL_DLLEXPORT jl_tvar_t *jl_new_typevar(jl_sym_t *name, jl_value_t *lb, jl_value_t *ub);
JL_DLLEXPORT jl_value_t *jl_instantiate_unionall(jl_unionall_t *u, jl_value_t *p);
JL_DLLEXPORT jl_value_t *jl_apply_type(jl_value_t *tc, jl_value_t **params, size_t n);
JL_DLLEXPORT jl_value_t *jl_apply_type1(jl_value_t *tc, jl_value_t *p1);
JL_DLLEXPORT jl_value_t *jl_apply_type2(jl_value_t *tc, jl_value_t *p1, jl_value_t *p2);
JL_DLLEXPORT jl_datatype_t *jl_apply_modify_type(jl_value_t *dt);
JL_DLLEXPORT jl_datatype_t *jl_apply_cmpswap_type(jl_value_t *dt);
JL_DLLEXPORT jl_tupletype_t *jl_apply_tuple_type(jl_svec_t *params);
JL_DLLEXPORT jl_tupletype_t *jl_apply_tuple_type_v(jl_value_t **p, size_t np);
JL_DLLEXPORT jl_datatype_t *jl_new_datatype(jl_sym_t *name,
                                            jl_module_t *module,
                                            jl_datatype_t *super,
                                            jl_svec_t *parameters,
                                            jl_svec_t *fnames,
                                            jl_svec_t *ftypes,
                                            jl_svec_t *fattrs,
                                            int abstract, int mutabl,
                                            int ninitialized);
JL_DLLEXPORT jl_datatype_t *jl_new_primitivetype(jl_value_t *name,
                                                 jl_module_t *module,
                                                 jl_datatype_t *super,
                                                 jl_svec_t *parameters, size_t nbits);

// constructors
JL_DLLEXPORT jl_value_t *jl_new_bits(jl_value_t *bt, const void *src);
JL_DLLEXPORT jl_value_t *jl_atomic_new_bits(jl_value_t *dt, const char *src);
JL_DLLEXPORT void jl_atomic_store_bits(char *dst, const jl_value_t *src, int nb);
JL_DLLEXPORT jl_value_t *jl_atomic_swap_bits(jl_value_t *dt, char *dst, const jl_value_t *src, int nb);
JL_DLLEXPORT int jl_atomic_bool_cmpswap_bits(char *dst, const jl_value_t *expected, const jl_value_t *src, int nb);
JL_DLLEXPORT jl_value_t *jl_atomic_cmpswap_bits(jl_datatype_t *dt, jl_datatype_t *rettype, char *dst, const jl_value_t *expected, const jl_value_t *src, int nb);
JL_DLLEXPORT jl_value_t *jl_new_struct(jl_datatype_t *type, ...);
JL_DLLEXPORT jl_value_t *jl_new_structv(jl_datatype_t *type, jl_value_t **args, uint32_t na);
JL_DLLEXPORT jl_value_t *jl_new_structt(jl_datatype_t *type, jl_value_t *tup);
JL_DLLEXPORT jl_value_t *jl_new_struct_uninit(jl_datatype_t *type);
JL_DLLEXPORT jl_method_instance_t *jl_new_method_instance_uninit(void);
JL_DLLEXPORT jl_svec_t *jl_svec(size_t n, ...) JL_MAYBE_UNROOTED;
JL_DLLEXPORT jl_svec_t *jl_svec1(void *a);
JL_DLLEXPORT jl_svec_t *jl_svec2(void *a, void *b);
JL_DLLEXPORT jl_svec_t *jl_alloc_svec(size_t n);
JL_DLLEXPORT jl_svec_t *jl_alloc_svec_uninit(size_t n);
JL_DLLEXPORT jl_svec_t *jl_svec_copy(jl_svec_t *a);
JL_DLLEXPORT jl_svec_t *jl_svec_fill(size_t n, jl_value_t *x);
JL_DLLEXPORT jl_value_t *jl_tupletype_fill(size_t n, jl_value_t *v);
JL_DLLEXPORT jl_sym_t *jl_symbol(const char *str) JL_NOTSAFEPOINT;
JL_DLLEXPORT jl_sym_t *jl_symbol_lookup(const char *str) JL_NOTSAFEPOINT;
JL_DLLEXPORT jl_sym_t *jl_symbol_n(const char *str, size_t len) JL_NOTSAFEPOINT;
JL_DLLEXPORT jl_sym_t *jl_gensym(void);
JL_DLLEXPORT jl_sym_t *jl_tagged_gensym(const char *str, size_t len);
JL_DLLEXPORT jl_sym_t *jl_get_root_symbol(void);
JL_DLLEXPORT jl_value_t *jl_generic_function_def(jl_sym_t *name,
                                                 jl_module_t *module,
                                                 _Atomic(jl_value_t*) *bp, jl_value_t *bp_owner,
                                                 jl_binding_t *bnd);
JL_DLLEXPORT jl_method_t *jl_method_def(jl_svec_t *argdata, jl_methtable_t *mt, jl_code_info_t *f, jl_module_t *module);
JL_DLLEXPORT jl_code_info_t *jl_code_for_staged(jl_method_instance_t *linfo);
JL_DLLEXPORT jl_code_info_t *jl_copy_code_info(jl_code_info_t *src);
JL_DLLEXPORT size_t jl_get_world_counter(void) JL_NOTSAFEPOINT;
JL_DLLEXPORT jl_value_t *jl_box_bool(int8_t x) JL_NOTSAFEPOINT;
JL_DLLEXPORT jl_value_t *jl_box_int8(int8_t x) JL_NOTSAFEPOINT;
JL_DLLEXPORT jl_value_t *jl_box_uint8(uint8_t x) JL_NOTSAFEPOINT;
JL_DLLEXPORT jl_value_t *jl_box_int16(int16_t x);
JL_DLLEXPORT jl_value_t *jl_box_uint16(uint16_t x);
JL_DLLEXPORT jl_value_t *jl_box_int32(int32_t x);
JL_DLLEXPORT jl_value_t *jl_box_uint32(uint32_t x);
JL_DLLEXPORT jl_value_t *jl_box_char(uint32_t x);
JL_DLLEXPORT jl_value_t *jl_box_int64(int64_t x);
JL_DLLEXPORT jl_value_t *jl_box_uint64(uint64_t x);
JL_DLLEXPORT jl_value_t *jl_box_float32(float x);
JL_DLLEXPORT jl_value_t *jl_box_float64(double x);
JL_DLLEXPORT jl_value_t *jl_box_voidpointer(void *x);
JL_DLLEXPORT jl_value_t *jl_box_uint8pointer(uint8_t *x);
JL_DLLEXPORT jl_value_t *jl_box_ssavalue(size_t x);
JL_DLLEXPORT jl_value_t *jl_box_slotnumber(size_t x);
JL_DLLEXPORT int8_t jl_unbox_bool(jl_value_t *v) JL_NOTSAFEPOINT;
JL_DLLEXPORT int8_t jl_unbox_int8(jl_value_t *v) JL_NOTSAFEPOINT;
JL_DLLEXPORT uint8_t jl_unbox_uint8(jl_value_t *v) JL_NOTSAFEPOINT;
JL_DLLEXPORT int16_t jl_unbox_int16(jl_value_t *v) JL_NOTSAFEPOINT;
JL_DLLEXPORT uint16_t jl_unbox_uint16(jl_value_t *v) JL_NOTSAFEPOINT;
JL_DLLEXPORT int32_t jl_unbox_int32(jl_value_t *v) JL_NOTSAFEPOINT;
JL_DLLEXPORT uint32_t jl_unbox_uint32(jl_value_t *v) JL_NOTSAFEPOINT;
JL_DLLEXPORT int64_t jl_unbox_int64(jl_value_t *v) JL_NOTSAFEPOINT;
JL_DLLEXPORT uint64_t jl_unbox_uint64(jl_value_t *v) JL_NOTSAFEPOINT;
JL_DLLEXPORT float jl_unbox_float32(jl_value_t *v) JL_NOTSAFEPOINT;
JL_DLLEXPORT double jl_unbox_float64(jl_value_t *v) JL_NOTSAFEPOINT;
JL_DLLEXPORT void *jl_unbox_voidpointer(jl_value_t *v) JL_NOTSAFEPOINT;
JL_DLLEXPORT uint8_t *jl_unbox_uint8pointer(jl_value_t *v) JL_NOTSAFEPOINT;

JL_DLLEXPORT int jl_get_size(jl_value_t *val, size_t *pnt);

#ifdef _P64
#define jl_box_long(x)   jl_box_int64(x)
#define jl_box_ulong(x)  jl_box_uint64(x)
#define jl_unbox_long(x) jl_unbox_int64(x)
#define jl_unbox_ulong(x) jl_unbox_uint64(x)
#define jl_is_long(x)    jl_is_int64(x)
#define jl_is_ulong(x)   jl_is_uint64(x)
#define jl_long_type     jl_int64_type
#define jl_ulong_type    jl_uint64_type
#else
#define jl_box_long(x)   jl_box_int32(x)
#define jl_box_ulong(x)  jl_box_uint32(x)
#define jl_unbox_long(x) jl_unbox_int32(x)
#define jl_unbox_ulong(x) jl_unbox_uint32(x)
#define jl_is_long(x)    jl_is_int32(x)
#define jl_is_ulong(x)   jl_is_uint32(x)
#define jl_long_type     jl_int32_type
#define jl_ulong_type    jl_uint32_type
#endif

// structs
JL_DLLEXPORT int         jl_field_index(jl_datatype_t *t, jl_sym_t *fld, int err);
JL_DLLEXPORT jl_value_t *jl_get_nth_field(jl_value_t *v, size_t i);
// Like jl_get_nth_field above, but asserts if it needs to allocate
JL_DLLEXPORT jl_value_t *jl_get_nth_field_noalloc(jl_value_t *v JL_PROPAGATES_ROOT, size_t i) JL_NOTSAFEPOINT;
JL_DLLEXPORT jl_value_t *jl_get_nth_field_checked(jl_value_t *v, size_t i);
JL_DLLEXPORT void        jl_set_nth_field(jl_value_t *v, size_t i, jl_value_t *rhs) JL_NOTSAFEPOINT;
JL_DLLEXPORT int         jl_field_isdefined(jl_value_t *v, size_t i) JL_NOTSAFEPOINT;
JL_DLLEXPORT int         jl_field_isdefined_checked(jl_value_t *v, size_t i);
JL_DLLEXPORT jl_value_t *jl_get_field(jl_value_t *o, const char *fld);
JL_DLLEXPORT jl_value_t *jl_value_ptr(jl_value_t *a);
int jl_uniontype_size(jl_value_t *ty, size_t *sz);
JL_DLLEXPORT int jl_islayout_inline(jl_value_t *eltype, size_t *fsz, size_t *al);

// arrays
JL_DLLEXPORT jl_array_t *jl_new_array(jl_value_t *atype, jl_value_t *dims);
JL_DLLEXPORT jl_array_t *jl_reshape_array(jl_value_t *atype, jl_array_t *data,
                                          jl_value_t *dims);
JL_DLLEXPORT jl_array_t *jl_ptr_to_array_1d(jl_value_t *atype, void *data,
                                            size_t nel, int own_buffer);
JL_DLLEXPORT jl_array_t *jl_ptr_to_array(jl_value_t *atype, void *data,
                                         jl_value_t *dims, int own_buffer);

JL_DLLEXPORT jl_array_t *jl_alloc_array_1d(jl_value_t *atype, size_t nr);
JL_DLLEXPORT jl_array_t *jl_alloc_array_2d(jl_value_t *atype, size_t nr,
                                           size_t nc);
JL_DLLEXPORT jl_array_t *jl_alloc_array_3d(jl_value_t *atype, size_t nr,
                                           size_t nc, size_t z);
JL_DLLEXPORT jl_array_t *jl_pchar_to_array(const char *str, size_t len);
JL_DLLEXPORT jl_value_t *jl_pchar_to_string(const char *str, size_t len);
JL_DLLEXPORT jl_value_t *jl_cstr_to_string(const char *str);
JL_DLLEXPORT jl_value_t *jl_alloc_string(size_t len);
JL_DLLEXPORT jl_value_t *jl_array_to_string(jl_array_t *a);
JL_DLLEXPORT jl_array_t *jl_alloc_vec_any(size_t n);
JL_DLLEXPORT jl_value_t *jl_arrayref(jl_array_t *a, size_t i);  // 0-indexed
JL_DLLEXPORT jl_value_t *jl_ptrarrayref(jl_array_t *a JL_PROPAGATES_ROOT, size_t i) JL_NOTSAFEPOINT;  // 0-indexed
JL_DLLEXPORT void jl_arrayset(jl_array_t *a JL_ROOTING_ARGUMENT, jl_value_t *v JL_ROOTED_ARGUMENT JL_MAYBE_UNROOTED, size_t i);  // 0-indexed
JL_DLLEXPORT void jl_arrayunset(jl_array_t *a, size_t i);  // 0-indexed
JL_DLLEXPORT int jl_array_isassigned(jl_array_t *a, size_t i);  // 0-indexed
JL_DLLEXPORT void jl_array_grow_end(jl_array_t *a, size_t inc);
JL_DLLEXPORT void jl_array_del_end(jl_array_t *a, size_t dec);
JL_DLLEXPORT void jl_array_grow_beg(jl_array_t *a, size_t inc);
JL_DLLEXPORT void jl_array_del_beg(jl_array_t *a, size_t dec);
JL_DLLEXPORT void jl_array_sizehint(jl_array_t *a, size_t sz);
JL_DLLEXPORT void jl_array_ptr_1d_push(jl_array_t *a, jl_value_t *item);
JL_DLLEXPORT void jl_array_ptr_1d_append(jl_array_t *a, jl_array_t *a2);
JL_DLLEXPORT jl_value_t *jl_apply_array_type(jl_value_t *type, size_t dim);
JL_DLLEXPORT int jl_array_validate_dims(size_t *nel, size_t *tot, uint32_t ndims, size_t *dims, size_t elsz);
// property access
JL_DLLEXPORT void *jl_array_ptr(jl_array_t *a);
JL_DLLEXPORT void *jl_array_eltype(jl_value_t *a);
JL_DLLEXPORT int jl_array_rank(jl_value_t *a);
JL_DLLEXPORT size_t jl_array_size(jl_value_t *a, int d);

// strings
JL_DLLEXPORT const char *jl_string_ptr(jl_value_t *s);

// modules and global variables
extern JL_DLLEXPORT jl_module_t *jl_main_module JL_GLOBALLY_ROOTED;
extern JL_DLLEXPORT jl_module_t *jl_core_module JL_GLOBALLY_ROOTED;
extern JL_DLLEXPORT jl_module_t *jl_base_module JL_GLOBALLY_ROOTED;
extern JL_DLLEXPORT jl_module_t *jl_top_module JL_GLOBALLY_ROOTED;
JL_DLLEXPORT jl_module_t *jl_new_module(jl_sym_t *name);
JL_DLLEXPORT void jl_set_module_nospecialize(jl_module_t *self, int on);
JL_DLLEXPORT void jl_set_module_optlevel(jl_module_t *self, int lvl);
JL_DLLEXPORT int jl_get_module_optlevel(jl_module_t *m);
JL_DLLEXPORT void jl_set_module_compile(jl_module_t *self, int value);
JL_DLLEXPORT int jl_get_module_compile(jl_module_t *m);
JL_DLLEXPORT void jl_set_module_infer(jl_module_t *self, int value);
JL_DLLEXPORT int jl_get_module_infer(jl_module_t *m);
JL_DLLEXPORT void jl_set_module_max_methods(jl_module_t *self, int value);
JL_DLLEXPORT int jl_get_module_max_methods(jl_module_t *m);
// get binding for reading
JL_DLLEXPORT jl_binding_t *jl_get_binding(jl_module_t *m JL_PROPAGATES_ROOT, jl_sym_t *var);
JL_DLLEXPORT jl_binding_t *jl_get_binding_or_error(jl_module_t *m, jl_sym_t *var);
JL_DLLEXPORT jl_binding_t *jl_get_binding_if_bound(jl_module_t *m, jl_sym_t *var);
JL_DLLEXPORT jl_value_t *jl_module_globalref(jl_module_t *m, jl_sym_t *var);
JL_DLLEXPORT jl_value_t *jl_binding_type(jl_module_t *m, jl_sym_t *var);
// get binding for assignment
JL_DLLEXPORT jl_binding_t *jl_get_binding_wr(jl_module_t *m JL_PROPAGATES_ROOT, jl_sym_t *var, int alloc);
JL_DLLEXPORT jl_binding_t *jl_get_binding_wr_or_error(jl_module_t *m JL_PROPAGATES_ROOT, jl_sym_t *var);
JL_DLLEXPORT jl_binding_t *jl_get_binding_for_method_def(jl_module_t *m JL_PROPAGATES_ROOT, jl_sym_t *var);
JL_DLLEXPORT int jl_boundp(jl_module_t *m, jl_sym_t *var);
JL_DLLEXPORT int jl_defines_or_exports_p(jl_module_t *m, jl_sym_t *var);
JL_DLLEXPORT int jl_binding_resolved_p(jl_module_t *m, jl_sym_t *var);
JL_DLLEXPORT int jl_is_const(jl_module_t *m, jl_sym_t *var);
JL_DLLEXPORT int jl_binding_is_const(jl_binding_t *b);
JL_DLLEXPORT int jl_binding_boundp(jl_binding_t *b);
JL_DLLEXPORT jl_value_t *jl_get_global(jl_module_t *m JL_PROPAGATES_ROOT, jl_sym_t *var);
JL_DLLEXPORT void jl_set_global(jl_module_t *m JL_ROOTING_ARGUMENT, jl_sym_t *var, jl_value_t *val JL_ROOTED_ARGUMENT);
JL_DLLEXPORT void jl_set_const(jl_module_t *m JL_ROOTING_ARGUMENT, jl_sym_t *var, jl_value_t *val JL_ROOTED_ARGUMENT);
JL_DLLEXPORT void jl_checked_assignment(jl_binding_t *b, jl_value_t *rhs JL_MAYBE_UNROOTED);
JL_DLLEXPORT void jl_declare_constant(jl_binding_t *b);
JL_DLLEXPORT void jl_module_using(jl_module_t *to, jl_module_t *from);
JL_DLLEXPORT void jl_module_use(jl_module_t *to, jl_module_t *from, jl_sym_t *s);
JL_DLLEXPORT void jl_module_use_as(jl_module_t *to, jl_module_t *from, jl_sym_t *s, jl_sym_t *asname);
JL_DLLEXPORT void jl_module_import(jl_module_t *to, jl_module_t *from, jl_sym_t *s);
JL_DLLEXPORT void jl_module_import_as(jl_module_t *to, jl_module_t *from, jl_sym_t *s, jl_sym_t *asname);
JL_DLLEXPORT void jl_module_export(jl_module_t *from, jl_sym_t *s);
JL_DLLEXPORT int jl_is_imported(jl_module_t *m, jl_sym_t *s);
JL_DLLEXPORT int jl_module_exports_p(jl_module_t *m, jl_sym_t *var);
JL_DLLEXPORT void jl_add_standard_imports(jl_module_t *m);
STATIC_INLINE jl_function_t *jl_get_function(jl_module_t *m, const char *name)
{
    return (jl_function_t*)jl_get_global(m, jl_symbol(name));
}

// eq hash tables
JL_DLLEXPORT jl_array_t *jl_eqtable_put(jl_array_t *h JL_ROOTING_ARGUMENT, jl_value_t *key, jl_value_t *val JL_ROOTED_ARGUMENT, int *inserted);
JL_DLLEXPORT jl_value_t *jl_eqtable_get(jl_array_t *h JL_PROPAGATES_ROOT, jl_value_t *key, jl_value_t *deflt) JL_NOTSAFEPOINT;
JL_DLLEXPORT jl_value_t *jl_eqtable_pop(jl_array_t *h, jl_value_t *key, jl_value_t *deflt, int *found);
jl_value_t *jl_eqtable_getkey(jl_array_t *h JL_PROPAGATES_ROOT, jl_value_t *key, jl_value_t *deflt) JL_NOTSAFEPOINT;

// system information
JL_DLLEXPORT int jl_errno(void) JL_NOTSAFEPOINT;
JL_DLLEXPORT void jl_set_errno(int e) JL_NOTSAFEPOINT;
JL_DLLEXPORT int32_t jl_stat(const char *path, char *statbuf) JL_NOTSAFEPOINT;
JL_DLLEXPORT int jl_cpu_threads(void) JL_NOTSAFEPOINT;
JL_DLLEXPORT int jl_effective_threads(void) JL_NOTSAFEPOINT;
JL_DLLEXPORT long jl_getpagesize(void) JL_NOTSAFEPOINT;
JL_DLLEXPORT long jl_getallocationgranularity(void) JL_NOTSAFEPOINT;
JL_DLLEXPORT int jl_is_debugbuild(void) JL_NOTSAFEPOINT;
JL_DLLEXPORT jl_sym_t *jl_get_UNAME(void) JL_NOTSAFEPOINT;
JL_DLLEXPORT jl_sym_t *jl_get_ARCH(void) JL_NOTSAFEPOINT;
JL_DLLEXPORT jl_value_t *jl_get_libllvm(void) JL_NOTSAFEPOINT;
extern JL_DLLIMPORT int jl_n_threadpools;
extern JL_DLLIMPORT _Atomic(int) jl_n_threads;
extern JL_DLLIMPORT int jl_n_gcthreads;
extern JL_DLLIMPORT int *jl_n_threads_per_pool;

// environment entries
JL_DLLEXPORT jl_value_t *jl_environ(int i);

// throwing common exceptions
JL_DLLEXPORT jl_value_t *jl_vexceptionf(jl_datatype_t *exception_type,
                                        const char *fmt, va_list args);
JL_DLLEXPORT void JL_NORETURN jl_error(const char *str);
JL_DLLEXPORT void JL_NORETURN jl_errorf(const char *fmt, ...);
JL_DLLEXPORT void JL_NORETURN jl_exceptionf(jl_datatype_t *ty,
                                            const char *fmt, ...);
JL_DLLEXPORT void JL_NORETURN jl_too_few_args(const char *fname, int min);
JL_DLLEXPORT void JL_NORETURN jl_too_many_args(const char *fname, int max);
JL_DLLEXPORT void JL_NORETURN jl_type_error(const char *fname,
                                            jl_value_t *expected JL_MAYBE_UNROOTED,
                                            jl_value_t *got JL_MAYBE_UNROOTED);
JL_DLLEXPORT void JL_NORETURN jl_type_error_rt(const char *fname,
                                               const char *context,
                                               jl_value_t *ty JL_MAYBE_UNROOTED,
                                               jl_value_t *got JL_MAYBE_UNROOTED);
JL_DLLEXPORT void JL_NORETURN jl_undefined_var_error(jl_sym_t *var);
JL_DLLEXPORT void JL_NORETURN jl_has_no_field_error(jl_sym_t *type_name, jl_sym_t *var);
JL_DLLEXPORT void JL_NORETURN jl_atomic_error(char *str);
JL_DLLEXPORT void JL_NORETURN jl_bounds_error(jl_value_t *v JL_MAYBE_UNROOTED,
                                              jl_value_t *t JL_MAYBE_UNROOTED);
JL_DLLEXPORT void JL_NORETURN jl_bounds_error_v(jl_value_t *v JL_MAYBE_UNROOTED,
                                                jl_value_t **idxs, size_t nidxs);
JL_DLLEXPORT void JL_NORETURN jl_bounds_error_int(jl_value_t *v JL_MAYBE_UNROOTED,
                                                  size_t i);
JL_DLLEXPORT void JL_NORETURN jl_bounds_error_tuple_int(jl_value_t **v,
                                                        size_t nv, size_t i);
JL_DLLEXPORT void JL_NORETURN jl_bounds_error_unboxed_int(void *v, jl_value_t *vt, size_t i);
JL_DLLEXPORT void JL_NORETURN jl_bounds_error_ints(jl_value_t *v JL_MAYBE_UNROOTED,
                                                   size_t *idxs, size_t nidxs);
JL_DLLEXPORT void JL_NORETURN jl_eof_error(void);

// Return the exception currently being handled, or `jl_nothing`.
//
// The catch scope is determined dynamically so this works in functions called
// from a catch block.  The returned value is gc rooted until we exit the
// enclosing JL_CATCH.
// FIXME: Teach the static analyzer about this rather than using
// JL_GLOBALLY_ROOTED which is far too optimistic.
JL_DLLEXPORT jl_value_t *jl_current_exception(void) JL_GLOBALLY_ROOTED JL_NOTSAFEPOINT;
JL_DLLEXPORT jl_value_t *jl_exception_occurred(void);
JL_DLLEXPORT void jl_exception_clear(void) JL_NOTSAFEPOINT;

#define JL_NARGS(fname, min, max)                               \
    if (nargs < min) jl_too_few_args(#fname, min);              \
    else if (nargs > max) jl_too_many_args(#fname, max);

#define JL_NARGSV(fname, min)                           \
    if (nargs < min) jl_too_few_args(#fname, min);

#define JL_TYPECHK(fname, type, v)                                 \
    if (!jl_is_##type(v)) {                                        \
        jl_type_error(#fname, (jl_value_t*)jl_##type##_type, (v)); \
    }
#define JL_TYPECHKS(fname, type, v)                                     \
    if (!jl_is_##type(v)) {                                             \
        jl_type_error(fname, (jl_value_t*)jl_##type##_type, (v));       \
    }

// initialization functions
typedef enum {
    JL_IMAGE_CWD = 0,
    JL_IMAGE_JULIA_HOME = 1,
    //JL_IMAGE_LIBJULIA = 2,
} JL_IMAGE_SEARCH;

JL_DLLEXPORT const char *jl_get_libdir(void);
JL_DLLEXPORT void julia_init(JL_IMAGE_SEARCH rel);
JL_DLLEXPORT void jl_init(void);
JL_DLLEXPORT void jl_init_with_image(const char *julia_bindir,
                                     const char *image_path);
JL_DLLEXPORT const char *jl_get_default_sysimg_path(void);
JL_DLLEXPORT int jl_is_initialized(void);
JL_DLLEXPORT void jl_atexit_hook(int status);
JL_DLLEXPORT void jl_postoutput_hook(void);
JL_DLLEXPORT void JL_NORETURN jl_exit(int status);
JL_DLLEXPORT void JL_NORETURN jl_raise(int signo);
JL_DLLEXPORT const char *jl_pathname_for_handle(void *handle);
JL_DLLEXPORT jl_gcframe_t **jl_adopt_thread(void);

JL_DLLEXPORT int jl_deserialize_verify_header(ios_t *s);
JL_DLLEXPORT void jl_preload_sysimg_so(const char *fname);
JL_DLLEXPORT void jl_set_sysimg_so(void *handle);
JL_DLLEXPORT void jl_create_system_image(void **, jl_array_t *worklist, bool_t emit_split, ios_t **s, ios_t **z, jl_array_t **udeps, int64_t *srctextpos);
JL_DLLEXPORT void jl_restore_system_image(const char *fname);
JL_DLLEXPORT void jl_restore_system_image_data(const char *buf, size_t len);
JL_DLLEXPORT jl_value_t *jl_restore_incremental(const char *fname, jl_array_t *depmods, int complete);

JL_DLLEXPORT void jl_set_newly_inferred(jl_value_t *newly_inferred);
JL_DLLEXPORT void jl_push_newly_inferred(jl_value_t *ci);
JL_DLLEXPORT void jl_write_compiler_output(void);

// parsing
JL_DLLEXPORT jl_value_t *jl_parse_all(const char *text, size_t text_len,
                                      const char *filename, size_t filename_len, size_t lineno);
JL_DLLEXPORT jl_value_t *jl_parse_string(const char *text, size_t text_len,
                                         int offset, int greedy);
// lowering
JL_DLLEXPORT jl_value_t *jl_expand(jl_value_t *expr, jl_module_t *inmodule);
JL_DLLEXPORT jl_value_t *jl_expand_with_loc(jl_value_t *expr, jl_module_t *inmodule,
                                            const char *file, int line);
JL_DLLEXPORT jl_value_t *jl_expand_with_loc_warn(jl_value_t *expr, jl_module_t *inmodule,
                                                 const char *file, int line);
JL_DLLEXPORT jl_value_t *jl_expand_in_world(jl_value_t *expr, jl_module_t *inmodule,
                                            const char *file, int line, size_t world);
JL_DLLEXPORT jl_value_t *jl_expand_stmt(jl_value_t *expr, jl_module_t *inmodule);
JL_DLLEXPORT jl_value_t *jl_expand_stmt_with_loc(jl_value_t *expr, jl_module_t *inmodule,
                                                 const char *file, int line);
// deprecated; use jl_parse_all
JL_DLLEXPORT jl_value_t *jl_parse_input_line(const char *text, size_t text_len,
                                             const char *filename, size_t filename_len);

// external libraries
enum JL_RTLD_CONSTANT {
     JL_RTLD_LOCAL=1U,
     JL_RTLD_GLOBAL=2U,
     JL_RTLD_LAZY=4U,
     JL_RTLD_NOW=8U,
     /* Linux/glibc and MacOS X: */
     JL_RTLD_NODELETE=16U,
     JL_RTLD_NOLOAD=32U,
     /* Linux/glibc: */
     JL_RTLD_DEEPBIND=64U,
     /* MacOS X 10.5+: */
     JL_RTLD_FIRST=128U
};
#define JL_RTLD_DEFAULT (JL_RTLD_LAZY | JL_RTLD_DEEPBIND)

typedef void *jl_libhandle; // compatible with dlopen (void*) / LoadLibrary (HMODULE)
JL_DLLEXPORT jl_libhandle jl_load_dynamic_library(const char *fname, unsigned flags, int throw_err);
JL_DLLEXPORT jl_libhandle jl_dlopen(const char *filename, unsigned flags) JL_NOTSAFEPOINT;
JL_DLLEXPORT int jl_dlclose(jl_libhandle handle) JL_NOTSAFEPOINT;
JL_DLLEXPORT int jl_dlsym(jl_libhandle handle, const char *symbol, void ** value, int throw_err) JL_NOTSAFEPOINT;

// evaluation
JL_DLLEXPORT jl_value_t *jl_toplevel_eval(jl_module_t *m, jl_value_t *v);
JL_DLLEXPORT jl_value_t *jl_toplevel_eval_in(jl_module_t *m, jl_value_t *ex);
// code loading (parsing + evaluation)
JL_DLLEXPORT jl_value_t *jl_eval_string(const char *str); // embedding interface
JL_DLLEXPORT jl_value_t *jl_load_file_string(const char *text, size_t len,
                                             char *filename, jl_module_t *module);
JL_DLLEXPORT jl_value_t *jl_load(jl_module_t *module, const char *fname);

JL_DLLEXPORT jl_module_t *jl_base_relative_to(jl_module_t *m JL_PROPAGATES_ROOT);

// tracing
JL_DLLEXPORT void jl_register_newmeth_tracer(void (*callback)(jl_method_t *tracee));

// AST access
JL_DLLEXPORT jl_value_t *jl_copy_ast(jl_value_t *expr JL_MAYBE_UNROOTED);

// IR representation
JL_DLLEXPORT jl_array_t *jl_compress_ir(jl_method_t *m, jl_code_info_t *code);
JL_DLLEXPORT jl_code_info_t *jl_uncompress_ir(jl_method_t *m, jl_code_instance_t *metadata, jl_array_t *data);
JL_DLLEXPORT uint8_t jl_ir_flag_inferred(jl_array_t *data) JL_NOTSAFEPOINT;
JL_DLLEXPORT uint8_t jl_ir_flag_pure(jl_array_t *data) JL_NOTSAFEPOINT;
JL_DLLEXPORT uint16_t jl_ir_inlining_cost(jl_array_t *data) JL_NOTSAFEPOINT;
JL_DLLEXPORT ssize_t jl_ir_nslots(jl_array_t *data) JL_NOTSAFEPOINT;
JL_DLLEXPORT uint8_t jl_ir_slotflag(jl_array_t *data, size_t i) JL_NOTSAFEPOINT;
JL_DLLEXPORT jl_value_t *jl_compress_argnames(jl_array_t *syms);
JL_DLLEXPORT jl_array_t *jl_uncompress_argnames(jl_value_t *syms);
JL_DLLEXPORT jl_value_t *jl_uncompress_argname_n(jl_value_t *syms, size_t i);


JL_DLLEXPORT int jl_is_operator(char *sym);
JL_DLLEXPORT int jl_is_unary_operator(char *sym);
JL_DLLEXPORT int jl_is_unary_and_binary_operator(char *sym);
JL_DLLEXPORT int jl_is_syntactic_operator(char *sym);
JL_DLLEXPORT int jl_operator_precedence(char *sym);

STATIC_INLINE int jl_vinfo_sa(uint8_t vi)
{
    return (vi&16)!=0;
}

STATIC_INLINE int jl_vinfo_usedundef(uint8_t vi)
{
    return (vi&32)!=0;
}

// calling into julia ---------------------------------------------------------

JL_DLLEXPORT jl_value_t *jl_apply_generic(jl_value_t *F, jl_value_t **args, uint32_t nargs);
JL_DLLEXPORT jl_value_t *jl_invoke(jl_value_t *F, jl_value_t **args, uint32_t nargs, jl_method_instance_t *meth);
JL_DLLEXPORT int32_t jl_invoke_api(jl_code_instance_t *linfo);

STATIC_INLINE jl_value_t *jl_apply(jl_value_t **args, uint32_t nargs)
{
    return jl_apply_generic(args[0], &args[1], nargs - 1);
}

JL_DLLEXPORT jl_value_t *jl_call(jl_function_t *f JL_MAYBE_UNROOTED, jl_value_t **args, uint32_t nargs);
JL_DLLEXPORT jl_value_t *jl_call0(jl_function_t *f JL_MAYBE_UNROOTED);
JL_DLLEXPORT jl_value_t *jl_call1(jl_function_t *f JL_MAYBE_UNROOTED, jl_value_t *a JL_MAYBE_UNROOTED);
JL_DLLEXPORT jl_value_t *jl_call2(jl_function_t *f JL_MAYBE_UNROOTED, jl_value_t *a JL_MAYBE_UNROOTED, jl_value_t *b JL_MAYBE_UNROOTED);
JL_DLLEXPORT jl_value_t *jl_call3(jl_function_t *f JL_MAYBE_UNROOTED, jl_value_t *a JL_MAYBE_UNROOTED,
                                  jl_value_t *b JL_MAYBE_UNROOTED, jl_value_t *c JL_MAYBE_UNROOTED);

// interfacing with Task runtime
JL_DLLEXPORT void jl_yield(void);

// async signal handling ------------------------------------------------------

JL_DLLEXPORT void jl_install_sigint_handler(void);
JL_DLLEXPORT void jl_sigatomic_begin(void);
JL_DLLEXPORT void jl_sigatomic_end(void);

// tasks and exceptions -------------------------------------------------------

typedef struct _jl_timing_block_t jl_timing_block_t;
typedef struct _jl_excstack_t jl_excstack_t;

// info describing an exception handler
typedef struct _jl_handler_t {
    jl_jmp_buf eh_ctx;
    jl_gcframe_t *gcstack;
    struct _jl_handler_t *prev;
    int8_t gc_state;
    size_t locks_len;
    sig_atomic_t defer_signal;
    jl_timing_block_t *timing_stack;
    size_t world_age;
} jl_handler_t;

typedef struct _jl_task_t {
    JL_DATA_TYPE
    jl_value_t *next; // invasive linked list for scheduler
    jl_value_t *queue; // invasive linked list for scheduler
    jl_value_t *tls;
    jl_value_t *donenotify;
    jl_value_t *result;
    jl_value_t *logstate;
    jl_function_t *start;
    uint64_t rngState[4];
    _Atomic(uint8_t) _state;
    uint8_t sticky; // record whether this Task can be migrated to a new thread
    _Atomic(uint8_t) _isexception; // set if `result` is an exception to throw or that we exited with
    // multiqueue priority
    uint16_t priority;

// hidden state:
    // id of owning thread - does not need to be defined until the task runs
    _Atomic(int16_t) tid;
    // threadpool id
    int8_t threadpoolid;
    // saved gc stack top for context switches
    jl_gcframe_t *gcstack;
    size_t world_age;
    // quick lookup for current ptls
    jl_ptls_t ptls; // == jl_all_tls_states[tid]
    // saved exception stack
    jl_excstack_t *excstack;
    // current exception handler
    jl_handler_t *eh;
    // saved thread state
    jl_ucontext_t ctx;
    void *stkbuf; // malloc'd memory (either copybuf or stack)
    size_t bufsz; // actual sizeof stkbuf
    uint64_t inference_start_time; // time when inference started
    uint16_t reentrant_inference; // How many times we've reentered inference
    uint16_t reentrant_timing; // How many times we've reentered timing
    unsigned int copy_stack:31; // sizeof stack for copybuf
    unsigned int started:1;
} jl_task_t;

#define JL_TASK_STATE_RUNNABLE 0
#define JL_TASK_STATE_DONE     1
#define JL_TASK_STATE_FAILED   2

JL_DLLEXPORT jl_task_t *jl_new_task(jl_function_t*, jl_value_t*, size_t);
JL_DLLEXPORT void jl_switchto(jl_task_t **pt);
JL_DLLEXPORT int jl_set_task_tid(jl_task_t *task, int16_t tid) JL_NOTSAFEPOINT;
JL_DLLEXPORT int jl_set_task_threadpoolid(jl_task_t *task, int8_t tpid) JL_NOTSAFEPOINT;
JL_DLLEXPORT void JL_NORETURN jl_throw(jl_value_t *e JL_MAYBE_UNROOTED);
JL_DLLEXPORT void JL_NORETURN jl_rethrow(void);
JL_DLLEXPORT void JL_NORETURN jl_sig_throw(void);
JL_DLLEXPORT void JL_NORETURN jl_rethrow_other(jl_value_t *e JL_MAYBE_UNROOTED);
JL_DLLEXPORT void JL_NORETURN jl_no_exc_handler(jl_value_t *e, jl_task_t *ct);
JL_DLLEXPORT JL_CONST_FUNC jl_gcframe_t **(jl_get_pgcstack)(void) JL_GLOBALLY_ROOTED JL_NOTSAFEPOINT;
#define jl_current_task (container_of(jl_get_pgcstack(), jl_task_t, gcstack))

extern JL_DLLIMPORT int jl_task_gcstack_offset;
extern JL_DLLIMPORT int jl_task_ptls_offset;

#include "julia_locks.h"   // requires jl_task_t definition

JL_DLLEXPORT void jl_enter_handler(jl_handler_t *eh);
JL_DLLEXPORT void jl_eh_restore_state(jl_handler_t *eh);
JL_DLLEXPORT void jl_pop_handler(int n);
JL_DLLEXPORT size_t jl_excstack_state(void) JL_NOTSAFEPOINT;
JL_DLLEXPORT void jl_restore_excstack(size_t state) JL_NOTSAFEPOINT;

#if defined(_OS_WINDOWS_)
#if defined(_COMPILER_GCC_)
JL_DLLEXPORT int __attribute__ ((__nothrow__,__returns_twice__)) (jl_setjmp)(jmp_buf _Buf);
__declspec(noreturn) __attribute__ ((__nothrow__)) void (jl_longjmp)(jmp_buf _Buf, int _Value);
JL_DLLEXPORT int __attribute__ ((__nothrow__,__returns_twice__)) (ijl_setjmp)(jmp_buf _Buf);
__declspec(noreturn) __attribute__ ((__nothrow__)) void (ijl_longjmp)(jmp_buf _Buf, int _Value);
#else
JL_DLLEXPORT int (jl_setjmp)(jmp_buf _Buf);
void (jl_longjmp)(jmp_buf _Buf, int _Value);
JL_DLLEXPORT int (ijl_setjmp)(jmp_buf _Buf);
void (ijl_longjmp)(jmp_buf _Buf, int _Value);
#endif
#ifdef LIBRARY_EXPORTS
#define jl_setjmp_f ijl_setjmp
#define jl_setjmp_name "ijl_setjmp"
#define jl_setjmp(a,b) ijl_setjmp(a)
#define jl_longjmp(a,b) ijl_longjmp(a,b)
#else
#define jl_setjmp_f jl_setjmp
#define jl_setjmp_name "jl_setjmp"
#define jl_setjmp(a,b) jl_setjmp(a)
#define jl_longjmp(a,b) jl_longjmp(a,b)
#endif
#elif defined(_OS_EMSCRIPTEN_)
#define jl_setjmp(a,b) setjmp(a)
#define jl_longjmp(a,b) longjmp(a,b)
#define jl_setjmp_f    setjmp
#define jl_setjmp_name "setjmp"
#else
// determine actual entry point name
#if defined(sigsetjmp)
#define jl_setjmp_f    __sigsetjmp
#define jl_setjmp_name "__sigsetjmp"
#else
#define jl_setjmp_f    sigsetjmp
#define jl_setjmp_name "sigsetjmp"
#endif
#define jl_setjmp(a,b) sigsetjmp(a,b)
#if defined(_COMPILER_ASAN_ENABLED_) && __GLIBC__
// Bypass the ASAN longjmp wrapper - we're unpoisoning the stack ourselves.
extern int __attribute__ ((nothrow)) (__libc_siglongjmp)(jl_jmp_buf buf, int val);
#define jl_longjmp(a,b) __libc_siglongjmp(a,b)
#else
#define jl_longjmp(a,b) siglongjmp(a,b)
#endif
#endif


#ifdef __clang_gcanalyzer__

// This is hard. Ideally we'd teach the static analyzer about the extra control
// flow edges. But for now, just hide this as best we can
extern int had_exception;
#define JL_TRY if (1)
#define JL_CATCH if (had_exception)

#else

#define JL_TRY                                                    \
    int i__tr, i__ca; jl_handler_t __eh;                          \
    size_t __excstack_state = jl_excstack_state();                \
    jl_enter_handler(&__eh);                                      \
    if (!jl_setjmp(__eh.eh_ctx,0))                                \
        for (i__tr=1; i__tr; i__tr=0, jl_eh_restore_state(&__eh))

#define JL_CATCH                                                \
    else                                                        \
        for (i__ca=1, jl_eh_restore_state(&__eh); i__ca; i__ca=0, jl_restore_excstack(__excstack_state))

#endif

// I/O system -----------------------------------------------------------------

struct uv_loop_s;
struct uv_handle_s;
struct uv_stream_s;
#ifdef _OS_WINDOWS_
typedef HANDLE jl_uv_os_fd_t;
#else
typedef int jl_uv_os_fd_t;
#endif
#define JL_STREAM struct uv_stream_s
#define JL_STDOUT jl_uv_stdout
#define JL_STDERR jl_uv_stderr
#define JL_STDIN  jl_uv_stdin

JL_DLLEXPORT int jl_process_events(void);

JL_DLLEXPORT struct uv_loop_s *jl_global_event_loop(void);

JL_DLLEXPORT void jl_close_uv(struct uv_handle_s *handle);

JL_DLLEXPORT jl_array_t *jl_take_buffer(ios_t *s);

typedef struct {
    void *data;
    struct uv_loop_s *loop;
    int type; // enum uv_handle_type
    jl_uv_os_fd_t file;
} jl_uv_file_t;

#ifdef __GNUC__
#define _JL_FORMAT_ATTR(type, str, arg) \
    __attribute__((format(type, str, arg)))
#else
#define _JL_FORMAT_ATTR(type, str, arg)
#endif

JL_DLLEXPORT void jl_uv_puts(struct uv_stream_s *stream, const char *str, size_t n);
JL_DLLEXPORT int jl_printf(struct uv_stream_s *s, const char *format, ...)
    _JL_FORMAT_ATTR(printf, 2, 3);
JL_DLLEXPORT int jl_vprintf(struct uv_stream_s *s, const char *format, va_list args)
    _JL_FORMAT_ATTR(printf, 2, 0);
JL_DLLEXPORT void jl_safe_printf(const char *str, ...) JL_NOTSAFEPOINT
    _JL_FORMAT_ATTR(printf, 1, 2);

extern JL_DLLEXPORT JL_STREAM *JL_STDIN;
extern JL_DLLEXPORT JL_STREAM *JL_STDOUT;
extern JL_DLLEXPORT JL_STREAM *JL_STDERR;

JL_DLLEXPORT JL_STREAM *jl_stdout_stream(void);
JL_DLLEXPORT JL_STREAM *jl_stdin_stream(void);
JL_DLLEXPORT JL_STREAM *jl_stderr_stream(void);
JL_DLLEXPORT int jl_termios_size(void);

// showing and std streams
JL_DLLEXPORT void jl_flush_cstdio(void) JL_NOTSAFEPOINT;
JL_DLLEXPORT jl_value_t *jl_stdout_obj(void) JL_NOTSAFEPOINT;
JL_DLLEXPORT jl_value_t *jl_stderr_obj(void) JL_NOTSAFEPOINT;
JL_DLLEXPORT size_t jl_static_show(JL_STREAM *out, jl_value_t *v) JL_NOTSAFEPOINT;
JL_DLLEXPORT size_t jl_static_show_func_sig(JL_STREAM *s, jl_value_t *type) JL_NOTSAFEPOINT;
JL_DLLEXPORT void jl_print_backtrace(void) JL_NOTSAFEPOINT;
JL_DLLEXPORT void jlbacktrace(void) JL_NOTSAFEPOINT; // deprecated
// Mainly for debugging, use `void*` so that no type cast is needed in C++.
JL_DLLEXPORT void jl_(void *jl_value) JL_NOTSAFEPOINT;

// julia options -----------------------------------------------------------

#include "jloptions.h"

extern JL_DLLIMPORT jl_options_t jl_options;

JL_DLLEXPORT ssize_t jl_sizeof_jl_options(void);

// Parse an argc/argv pair to extract general julia options, passing back out
// any arguments that should be passed on to the script.
JL_DLLEXPORT void jl_parse_opts(int *argcp, char ***argvp);
JL_DLLEXPORT char *jl_format_filename(const char *output_pattern);

// Set julia-level ARGS array according to the arguments provided in
// argc/argv
JL_DLLEXPORT void jl_set_ARGS(int argc, char **argv);

JL_DLLEXPORT int jl_generating_output(void) JL_NOTSAFEPOINT;

// Settings for code_coverage and malloc_log
// NOTE: if these numbers change, test/cmdlineargs.jl will have to be updated
#define JL_LOG_NONE 0
#define JL_LOG_USER 1
#define JL_LOG_ALL  2
#define JL_LOG_PATH 3

#define JL_OPTIONS_CHECK_BOUNDS_DEFAULT 0
#define JL_OPTIONS_CHECK_BOUNDS_ON 1
#define JL_OPTIONS_CHECK_BOUNDS_OFF 2

#define JL_OPTIONS_COMPILE_DEFAULT 1
#define JL_OPTIONS_COMPILE_OFF 0
#define JL_OPTIONS_COMPILE_ON  1
#define JL_OPTIONS_COMPILE_ALL 2
#define JL_OPTIONS_COMPILE_MIN 3

#define JL_OPTIONS_COLOR_AUTO 0
#define JL_OPTIONS_COLOR_ON 1
#define JL_OPTIONS_COLOR_OFF 2

#define JL_OPTIONS_HISTORYFILE_ON 1
#define JL_OPTIONS_HISTORYFILE_OFF 0

#define JL_OPTIONS_STARTUPFILE_ON 1
#define JL_OPTIONS_STARTUPFILE_OFF 2

#define JL_LOGLEVEL_BELOWMIN -1000001
#define JL_LOGLEVEL_DEBUG    -1000
#define JL_LOGLEVEL_INFO      0
#define JL_LOGLEVEL_WARN      1000
#define JL_LOGLEVEL_ERROR     2000
#define JL_LOGLEVEL_ABOVEMAX  1000001

#define JL_OPTIONS_DEPWARN_OFF 0
#define JL_OPTIONS_DEPWARN_ON 1
#define JL_OPTIONS_DEPWARN_ERROR 2

#define JL_OPTIONS_WARN_OVERWRITE_OFF 0
#define JL_OPTIONS_WARN_OVERWRITE_ON 1

#define JL_OPTIONS_WARN_SCOPE_OFF 0
#define JL_OPTIONS_WARN_SCOPE_ON 1

#define JL_OPTIONS_POLLY_ON 1
#define JL_OPTIONS_POLLY_OFF 0

#define JL_OPTIONS_FAST_MATH_ON 1
#define JL_OPTIONS_FAST_MATH_OFF 2
#define JL_OPTIONS_FAST_MATH_DEFAULT 0

#define JL_OPTIONS_HANDLE_SIGNALS_ON 1
#define JL_OPTIONS_HANDLE_SIGNALS_OFF 0

#define JL_OPTIONS_USE_SYSIMAGE_NATIVE_CODE_YES 1
#define JL_OPTIONS_USE_SYSIMAGE_NATIVE_CODE_NO 0

#define JL_OPTIONS_USE_COMPILED_MODULES_YES 1
#define JL_OPTIONS_USE_COMPILED_MODULES_NO 0

#define JL_OPTIONS_USE_PKGIMAGES_YES 1
#define JL_OPTIONS_USE_PKGIMAGES_NO 0

// Version information
#include <julia_version.h> // Generated file

JL_DLLEXPORT extern int jl_ver_major(void);
JL_DLLEXPORT extern int jl_ver_minor(void);
JL_DLLEXPORT extern int jl_ver_patch(void);
JL_DLLEXPORT extern int jl_ver_is_release(void);
JL_DLLEXPORT extern const char *jl_ver_string(void);
JL_DLLEXPORT const char *jl_git_branch(void);
JL_DLLEXPORT const char *jl_git_commit(void);

// nullable struct representations
typedef struct {
    uint8_t hasvalue;
    double value;
} jl_nullable_float64_t;

typedef struct {
    uint8_t hasvalue;
    float value;
} jl_nullable_float32_t;

#define jl_root_task (jl_current_task->ptls->root_task)

JL_DLLEXPORT jl_task_t *jl_get_current_task(void) JL_GLOBALLY_ROOTED JL_NOTSAFEPOINT;

// TODO: we need to pin the task while using this (set pure bit)
JL_DLLEXPORT jl_jmp_buf *jl_get_safe_restore(void) JL_NOTSAFEPOINT;
JL_DLLEXPORT void jl_set_safe_restore(jl_jmp_buf *) JL_NOTSAFEPOINT;

// codegen interface ----------------------------------------------------------
// The root propagation here doesn't have to be literal, but callers should
// ensure that the return value outlives the MethodInstance
typedef jl_value_t *(*jl_codeinstance_lookup_t)(jl_method_instance_t *mi JL_PROPAGATES_ROOT,
    size_t min_world, size_t max_world);
typedef struct {
    int track_allocations;  // can we track allocations?
    int code_coverage;      // can we measure coverage?
    int prefer_specsig;     // are specialized function signatures preferred?

    // controls the emission of debug-info. mirrors the clang options
    int gnu_pubnames;       // can we emit the gnu pubnames debuginfo
    int debug_info_kind;    // Enum for line-table-only, line-directives-only,
                            // limited, standalone

    int safepoint_on_entry; // Emit a safepoint on entry to each function

    // Cache access. Default: jl_rettype_inferred.
    jl_codeinstance_lookup_t lookup;

    // If not `nothing`, rewrite all generic calls to call
    // generic_context(f, args...) instead of f(args...).
    jl_value_t *generic_context;
} jl_cgparams_t;
extern JL_DLLEXPORT int jl_default_debug_info_kind;

#ifdef MMTK_GC
extern void mmtk_object_reference_write_post(void* mutator, const void* parent, const void* ptr);
extern void mmtk_object_reference_write_slow(void* mutator, const void* parent, const void* ptr);
extern void* mmtk_alloc(void* mutator, size_t size, size_t align, size_t offset, int allocator);

extern const void* MMTK_SIDE_LOG_BIT_BASE_ADDRESS;

// These need to be constants.

#define MMTK_OBJECT_BARRIER (1)
// Stickyimmix needs write barrier. Immix does not need write barrier.
#ifdef MMTK_PLAN_IMMIX
#define MMTK_NEEDS_WRITE_BARRIER (0)
#endif
#ifdef MMTK_PLAN_STICKYIMMIX
#define MMTK_NEEDS_WRITE_BARRIER (1)
#endif

#define MMTK_DEFAULT_IMMIX_ALLOCATOR (0)
#define MMTK_IMMORTAL_BUMP_ALLOCATOR (0)

// Directly call into MMTk for write barrier (debugging only)
STATIC_INLINE void mmtk_gc_wb_full(const void *parent, const void *ptr) JL_NOTSAFEPOINT
{
    jl_task_t *ct = jl_current_task;
    jl_ptls_t ptls = ct->ptls;
    mmtk_object_reference_write_post(&ptls->mmtk_mutator, parent, ptr);
}

// Fastpath. Return 1 if we should go to slowpath
STATIC_INLINE int mmtk_gc_wb_fast_check(const void *parent, const void *ptr) JL_NOTSAFEPOINT
{
    if (MMTK_NEEDS_WRITE_BARRIER == MMTK_OBJECT_BARRIER) {
        intptr_t addr = (intptr_t) (void*) parent;
        uint8_t* meta_addr = (uint8_t*) (MMTK_SIDE_LOG_BIT_BASE_ADDRESS) + (addr >> 6);
        intptr_t shift = (addr >> 3) & 0b111;
        uint8_t byte_val = *meta_addr;
        return ((byte_val >> shift) & 1) == 1;
    } else {
        return 0;
    }
}

// Slowpath.
STATIC_INLINE void mmtk_gc_wb_slow(const void *parent, const void *ptr) JL_NOTSAFEPOINT
{
    if (MMTK_NEEDS_WRITE_BARRIER == MMTK_OBJECT_BARRIER) {
        jl_task_t *ct = jl_current_task;
        jl_ptls_t ptls = ct->ptls;
        mmtk_object_reference_write_slow(&ptls->mmtk_mutator, parent, ptr);
    }
}

STATIC_INLINE void mmtk_gc_wb(const void *parent, const void *ptr) JL_NOTSAFEPOINT
{
    if (mmtk_gc_wb_fast_check(parent, ptr)) {
        mmtk_gc_wb_slow(parent, ptr);
    }
}

STATIC_INLINE void mmtk_gc_wb_binding(const void *bnd, const void *val) JL_NOTSAFEPOINT
{
    if (mmtk_gc_wb_fast_check(bnd, val)) {
        jl_astaggedvalue(bnd)->bits.gc = 2; // to indicate that the buffer is a binding
        mmtk_gc_wb_slow(bnd, val);
    }
}

#define MMTK_MIN_ALIGNMENT 4
// MMTk assumes allocation size is aligned to min alignment.
STATIC_INLINE size_t mmtk_align_alloc_sz(size_t sz) JL_NOTSAFEPOINT
{
    return (sz + MMTK_MIN_ALIGNMENT - 1) & ~(MMTK_MIN_ALIGNMENT - 1);
}

STATIC_INLINE void* bump_alloc_fast(MMTkMutatorContext* mutator, uintptr_t* cursor, uintptr_t limit, size_t size, size_t align, size_t offset, int allocator) {
    intptr_t delta = (-offset - *cursor) & (align - 1);
    uintptr_t result = *cursor + (uintptr_t)delta;

    if (__unlikely(result + size > limit)) {
        return (void*) mmtk_alloc(mutator, size, align, offset, allocator);
    } else{
        *cursor = result + size;
        return (void*)result;
    }
}

STATIC_INLINE void* mmtk_immix_alloc_fast(MMTkMutatorContext* mutator, size_t size, size_t align, size_t offset) {
    ImmixAllocator* allocator = &mutator->allocators.immix[MMTK_DEFAULT_IMMIX_ALLOCATOR];
    return bump_alloc_fast(mutator, (uintptr_t*)&allocator->cursor, (intptr_t)allocator->limit, size, align, offset, 0);
}

STATIC_INLINE void mmtk_immix_post_alloc_fast(MMTkMutatorContext* mutator, void* obj, size_t size) {
    // We do not need post alloc for immix objects in immix/stickyimmix
}

STATIC_INLINE void* mmtk_immortal_alloc_fast(MMTkMutatorContext* mutator, size_t size, size_t align, size_t offset) {
    BumpAllocator* allocator = &mutator->allocators.bump_pointer[MMTK_IMMORTAL_BUMP_ALLOCATOR];
    return bump_alloc_fast(mutator, (uintptr_t*)&allocator->cursor, (uintptr_t)allocator->limit, size, align, offset, 1);
}

STATIC_INLINE void mmtk_immortal_post_alloc_fast(MMTkMutatorContext* mutator, void* obj, size_t size) {
    if (MMTK_NEEDS_WRITE_BARRIER == MMTK_OBJECT_BARRIER) {
        intptr_t addr = (intptr_t) obj;
        uint8_t* meta_addr = (uint8_t*) (MMTK_SIDE_LOG_BIT_BASE_ADDRESS) + (addr >> 6);
        intptr_t shift = (addr >> 3) & 0b111;
        while(1) {
            uint8_t old_val = *meta_addr;
            uint8_t new_val = old_val | (1 << shift);
            if (jl_atomic_cmpswap((_Atomic(uint8_t)*)meta_addr, &old_val, new_val)) {
                break;
            }
        }
    }
}

#endif

#ifdef __cplusplus
}
#endif

#endif
