#ifndef BALLISTIC_BAL_X86_TIER1_COMPILER_H
#define BALLISTIC_BAL_X86_TIER1_COMPILER_H

#include "bal_attributes.h"
#include "bal_jit_debug.h"
#include "bal_memory.h"
#include "bal_x86_assembler.h"
#include "bal_x86_sliding_window.h"
#include <stdbool.h>

#ifdef __cplusplus
extern "C"
{
#endif /* __cplusplus */

    BAL_ALIGNED(64) typedef struct
    {
        bal_x86_assembler_t  assembler;
        bal_sliding_window_t window;
        int8_t               arm_to_x86[32];
        int8_t               x86_to_arm[16];
        uint32_t             is_dirty;
        uint8_t              eviction_cursor;
        /// Whether the block being translated has switched MXCSR to the guest floating-point
        /// environment, which every later exit must undo.
        bool is_guest_mxcsr_active;
        /// Bitmask of the x86 registers allocated for the guest instruction being translated.
        /// They are never chosen for eviction, so an instruction cannot evict its own operands.
        uint16_t                 pinned_registers;
        bal_jit_debug_context_t *debug_context;
        bal_error_t              status;
        /// Whether the host has FMA3, which the fused multiply-add fast paths need.
        bool has_fma;
        /// Whether the exits of the block being translated may be chained to the next block.
        /// Blocks that raise a guest exception or end on ISB always return to the engine.
        bool    is_linkable;
        uint8_t pad1[2];
    } bal_tier1_compiler_t;

    static_assert(256 == sizeof(bal_tier1_compiler_t), "Struct size mismatch");

    /// Initializes the Tier 1 Compiler.
    ///
    /// # Safety
    ///
    /// The `executable_buffer` must be memory allocated with execution permissions.
    ///
    /// # Errors
    ///
    /// Returns [`BAL_SUCCESS`] on success.
    ///
    /// Returns [`BAL_ERROR_INVALID_ARGUMENT`] if passed pointers are NULL or `buffer_size` is 0.
    BAL_COLD bal_error_t bal_tier1_compiler_init(bal_tier1_compiler_t    *compiler,
                                                 bal_executable_buffer_t  executable_buffer,
                                                 size_t                   buffer_size,
                                                 bal_jit_debug_context_t *debug_context);

    BAL_HOT void bal_tier1_compiler_reset(bal_tier1_compiler_t *compiler);

    /// The size of the prologue every translated block starts with. A chained jump enters a block
    /// this many bytes after its start, where the registers the prologue saves already are.
#define BAL_TIER1_BLOCK_PROLOGUE_SIZE 12U

    /// A translated block that indirect branches (BR, BLR, RET and their authenticated forms)
    /// can jump to without returning to the engine: its guest address and the host address just
    /// past its prologue. The table is direct-mapped on bits 13:2 of the guest address; an
    /// empty entry has a NULL `host_code`.
    typedef struct
    {
        uint64_t guest_address;
        void    *host_code;
    } bal_tier1_lookup_entry_t;

#define BAL_TIER1_LOOKUP_ENTRIES 4096U

    /// A guest page that loads and stores reach without calling the memory interface: its page
    /// number (the guest address shifted right by 12) and the value to add to a guest address in
    /// it to get the host address. The table is direct-mapped on the low bits of the page number;
    /// an empty entry has a page number no address has.
    typedef struct
    {
        uint64_t page;
        uint64_t host_offset;
    } bal_tier1_tlb_entry_t;

#define BAL_TIER1_TLB_ENTRIES     256U
#define BAL_TIER1_TLB_EMPTY_PAGE  0xFFFFFFFFFFFFFFFFULL
#define BAL_TIER1_GUEST_PAGE_SIZE 4096U

    /// The MXCSR translated code runs under: round to nearest, every exception masked, no
    /// flush-to-zero, and no flags raised. It is the guest floating-point environment when FPCR is
    /// zero, the only one inline floating point implements.
#define BAL_TIER1_GUEST_MXCSR 0x1F80U

    /// The FPSR cumulative flags the MXCSR exception flags in `mxcsr` stand for: IE, ZE, OE, UE
    /// and PE (bits 0 and 2 to 5) become IOC, DZC, OFC, UFC and IXC (bits 0 to 4).
    uint64_t bal_tier1_fpsr_flags_from_mxcsr(uint32_t mxcsr);

    /// Empties a software TLB of BAL_TIER1_TLB_ENTRIES entries.
    void bal_tier1_tlb_clear(bal_tier1_tlb_entry_t *tlb);

    BAL_HOT void *bal_tier1_compiler_translate(bal_tier1_compiler_t         *compiler,
                                               const bal_memory_interface_t *memory_interface,
                                               bal_guest_address_t           guest_address,
                                               size_t                        max_instructions,
                                               uint32_t                      engine_flags);

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif // BALLISTIC_BAL_X86_TIER1_COMPILER_H

/*** end of file ***/
