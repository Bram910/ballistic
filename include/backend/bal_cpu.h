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
    } bal_exception_t;

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

        /// Keeps the structure a whole number of its 64-byte alignment.
        uint8_t reserved[56];
    } bal_cpu_t;

    /// The function signature of a JIT-compiled basic block.
    typedef void (*bal_jit_block_t)(bal_cpu_t *cpu);

#ifdef __cplusplus
}
#endif // __cplusplus

#endif // BALLISTIC_BAL_CPU_H

/*** end of file ***/
