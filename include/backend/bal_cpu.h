//! Holds the state of the ARM guest CPU.

#ifndef BALLISTIC_BAL_CPU_H
#define BALLISTIC_BAL_CPU_H

#include "bal_attributes.h"
#include "bal_jit_debug.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif // __cplusplus

    /// The generic timer frequency of the Nintendo Switch, 19.2 MHz, which CNTFRQ_EL0 reports
    /// when bal_cpu_t::counter_frequency is zero.
#define BAL_DEFAULT_COUNTER_FREQUENCY 19200000ULL

    /// Guest exceptions reported through [`bal_cpu_t::exception`].
    typedef enum
    {
        BAL_EXCEPTION_NONE = 0,

        /// A load or store accessed an address the memory interface does not translate, or was
        /// misaligned where the architecture requires alignment. `pc` is the faulting
        /// instruction.
        BAL_EXCEPTION_DATA_ABORT = 1,

        /// SVC: a call into the guest operating system. `pc` is the instruction after the SVC,
        /// where execution resumes, and `exception_immediate` its 16-bit immediate.
        BAL_EXCEPTION_SUPERVISOR_CALL = 2,

        /// BRK: a software breakpoint. `pc` is the BRK and `exception_immediate` its immediate.
        BAL_EXCEPTION_BREAKPOINT = 3,

        /// UDF, and HLT, HVC and SMC, which are undefined at EL0. `pc` is the instruction and
        /// `exception_immediate` its immediate.
        BAL_EXCEPTION_UNDEFINED_INSTRUCTION = 4,

        /// An instruction tier 1 does not translate: one Ballistic does not implement yet, such as
        /// an access to an unknown system register. The guest should see an undefined
        /// instruction exception. It is raised when the instruction is reached, after the ones
        /// before it have run; `pc` is the instruction and `exception_immediate` its encoding,
        /// and bal_engine_run_thread() returns BAL_ERROR_UNKNOWN_INSTRUCTION.
        BAL_EXCEPTION_UNSUPPORTED_INSTRUCTION = 5,

        /// A guarded control stack exception: GCSSTR or GCSSTTR, while the guarded control stack
        /// stores are disabled at EL0 (GCSCRE0_EL1.STREn = 0), as they always are here. Linux
        /// reports it as SIGSEGV. `pc` is the instruction and `exception_immediate` zero.
        BAL_EXCEPTION_GUARDED_CONTROL_STACK = 6,
    } bal_exception_t;

    /// The pointer authentication keys in bal_cpu_t::pac_keys.
    typedef enum
    {
        BAL_PAC_KEY_IA,
        BAL_PAC_KEY_IB,
        BAL_PAC_KEY_DA,
        BAL_PAC_KEY_DB,
        BAL_PAC_KEY_GA,
    } bal_pac_key_t;

    BAL_ALIGNED(64) typedef struct
    {
        uint64_t                 x[32];
        uint64_t                 pc;
        bal_jit_debug_context_t *debug_context;

        // ARM's Carry flag is inverted compared to x86's Carry during subtraction operations.
        // Compiler's must handle this.
        uint8_t flag_c; // ARM C -> x86 CF
        uint8_t flag_z; // ARM Z -> x86 ZF
        uint8_t flag_n; // ARM N -> x86 SF
        uint8_t flag_v; // ARM V -> x86 OF

        /// Non-zero when the last block stopped on a guest exception instead of finishing, see
        /// [`bal_exception_t`]. `pc` then holds the address of the instruction that raised it. It
        /// is zero while a block runs, which translators rely on when they store the flags as one
        /// 64-bit word, and the host must clear it before resuming execution.
        uint32_t exception;

        /// Tracks executed guest instructions.
        ///
        /// This only increments if the engine flag [`BAL_ENGINE_FLAG_INSTRUCTION_COUNTING`] is set.
        uint64_t instruction_count;

        /// EL0 thread pointer, read and written by the guest with MRS/MSR TPIDR_EL0.
        uint64_t tpidr_el0;

        /// Read-only EL0 thread pointer. The guest can only read it; the host sets it.
        uint64_t tpidrro_el0;

        /// Floating-point control and status registers, as seen through MRS/MSR.
        uint64_t fpcr;
        uint64_t fpsr;

        /// The SIMD&FP registers V0-V31. v[n][0] holds bits 63:0 and v[n][1] bits 127:64. The
        /// scalar B, H, S and D views are the low bits of v[n][0]; writing one clears the rest of
        /// the register.
        uint64_t v[32][2];

        /// The local exclusive monitor. A load-exclusive sets `exclusive_state` to 1 and records
        /// the address and the value it read; a store-exclusive, CLREX, or a host that wants to
        /// break the reservation sets it back to 0. Zero-initialised state has no reservation.
        uint64_t exclusive_state;
        uint64_t exclusive_address;
        uint64_t exclusive_value[2];

        /// The immediate of the SVC, BRK or undefined instruction that raised `exception`.
        uint64_t exception_immediate;

        /// CNTFRQ_EL0: the frequency of the generic timer in Hz. Zero selects
        /// BAL_DEFAULT_COUNTER_FREQUENCY.
        uint64_t counter_frequency;

        /// Returns the generic timer count that CNTPCT_EL0 and CNTVCT_EL0 read, ticking at the
        /// counter frequency, given `counter_context`. When NULL, the count comes from the host's
        /// monotonic clock.
        uint64_t (*read_counter)(void *context);
        void *counter_context;

        /// Used by translated code that runs floating point inline: the host's MXCSR while the
        /// block runs, and a word for loading and storing MXCSR.
        uint32_t jit_host_mxcsr;
        uint32_t jit_mxcsr;

        /// The pointer authentication keys, indexed by bal_pac_key_t: [key][0] holds the
        /// APxxKeyLo_EL1 half and [key][1] the APxxKeyHi_EL1 half.
        uint64_t pac_keys[5][2];

        /// The SCTLR_EL1 bits EnIA (31), EnIB (30), EnDA (27) and EnDB (13), which enable the
        /// instruction and data keys. With a key disabled, the instructions that sign or
        /// authenticate with it leave the pointer unchanged; zero disables pointer authentication,
        /// and PACIASP and the other hint-space forms then execute as NOP.
        uint64_t sctlr_el1;

        /// The TCR_EL1 fields that place the pointer authentication code: T0SZ (5:0) and T1SZ
        /// (21:16) for addresses with bit 55 clear and set, TBI0 (37), TBI1 (38), TBID0 (51) and
        /// TBID1 (52). A TxSZ below 16 counts as 16, so zero gives 48-bit addresses.
        uint64_t tcr_el1;

        /// Set by IC IVAU, which says the guest has written instructions, and by ISB, which
        /// makes them visible to the instruction stream. When a block ends on an ISB after an
        /// IC IVAU, the engine discards its translations before running on, so the guest
        /// executes the code it wrote. Other engines sharing the memory need
        /// bal_engine_clear_cache() from the host.
        uint32_t instruction_cache_invalidated;
        uint32_t instruction_barrier;

        /// PSTATE.DIT (bit 24) and PSTATE.SSBS (bit 12), as MRS DIT and MRS SSBS read them.
        /// They are kept for the guest to read back only: Ballistic does not make translated
        /// code data-independent in timing when DIT is set, and the host CPU decides how loads
        /// speculate past stores whatever SSBS says.
        uint64_t dit;
        uint64_t ssbs;

        /// Used by the engine to chain translated blocks: the host address of the jump a block
        /// left through, which the engine points at the next block, and a flag the host sets,
        /// through bal_engine_request_exit(), to make chained blocks return to the engine.
        uint64_t jit_link_site;
        uint32_t jit_exit_requested;
        uint32_t jit_pad;

        /// Used by the engine: the table indirect branches look their target block up in, an
        /// array of BAL_TIER1_LOOKUP_ENTRIES bal_tier1_lookup_entry_t.
        uint64_t jit_lookup_table;

        /// Used by the engine: the software TLB loads and stores look guest pages up in, an
        /// array of BAL_TIER1_TLB_ENTRIES bal_tier1_tlb_entry_t.
        uint64_t jit_memory_tlb;

        /// Used by the engine: the bal_memory_interface_t a TLB miss asks.
        uint64_t jit_memory_interface;

        /// Used by the engine: the pointer authentication memo translated code signs and
        /// authenticates through, the entries of a bal_pac_memo_t.
        uint64_t jit_pac_memo;

        /// RCWSMASK_EL1: the bits RCWSCAS, RCWSCLR, RCWSSET and RCWSSWP may change in a value whose
        /// bit 0 is set. Bit 17 also stands for bits 49:18, as the architecture replicates it.
        uint64_t rcwsmask_el1;

        /// PSTATE.TCO (bit 25), the MTE tag check override, as MSR TCO writes it and MRS TCO
        /// reads it. Tag checks are not performed, so it has no other effect.
        uint64_t tco;

        /// The SVE predicate registers P0-P15 and the first-fault register FFR. The vector
        /// length is 128 bits, so a predicate has one bit per byte of a Z register, 16 bits.
        uint16_t p[16];
        uint16_t ffr;

        /// Keeps the SVE state a whole number of the 64-byte alignment.
        uint8_t reserved[30];

        /// The SME state, with a streaming vector length of 128 bits like the SVE one: the ZA
        /// array as its 16 horizontal byte slices of 16 bytes, and the SME2 lookup table ZT0.
        uint8_t za[16][16];
        uint8_t zt0[64];
        /// SVCR: PSTATE.SM (bit 0), streaming mode, and PSTATE.ZA (bit 1), ZA storage. Changing SM
        /// zeroes the Z and P registers and FFR and sets FPSR to 0x0800009F; enabling ZA zeroes ZA
        /// and ZT0. SME instructions are undefined while the mode they need is off.
        uint64_t svcr;
        /// TPIDR2_EL0, the thread register SME software uses for lazy saving of ZA.
        uint64_t tpidr2_el0;

        /// Keeps the structure a whole number of its 64-byte alignment.
        uint8_t reserved_sme[48];
    } bal_cpu_t;

    /// The function signature of a JIT-compiled basic block.
    typedef void (*bal_jit_block_t)(bal_cpu_t *cpu);

#ifdef __cplusplus
}
#endif // __cplusplus

#endif // BALLISTIC_BAL_CPU_H

/*** end of file ***/
