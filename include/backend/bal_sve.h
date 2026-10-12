//! Runtime support for the guest's SVE, SVE2 and SME instructions.
//!
//! The guest CPU implements SVE with the smallest vector length, 128 bits, so the Z registers are
//! the SIMD&FP registers in bal_cpu_t::v and the predicate registers P0-P15 and FFR are 16 bits
//! each, in bal_cpu_t::p and bal_cpu_t::ffr. SME has the same streaming vector length, and with
//! FEAT_SME_FA64 every instruction is also legal in streaming mode. bal_execute_sve() executes
//! one SVE or SME instruction on that state; translated code calls it with the guest registers
//! flushed to memory.
#ifndef BALLISTIC_BAL_SVE_H
#define BALLISTIC_BAL_SVE_H

#include "backend/bal_cpu.h"
#include "bal_memory.h"
#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif /* __cplusplus */

    /// The vector length in bytes.
#define BAL_SVE_VECTOR_BYTES 16U

    /// Returns true if bal_execute_sve() implements `instruction`.
    bool bal_is_sve_supported(uint32_t instruction);

    /// Executes one SVE or SME instruction against `cpu`. Returns zero when the instruction raised
    /// an exception, which it records in bal_cpu_t::exception: BAL_EXCEPTION_DATA_ABORT, after a
    /// first-faulting load's partial work if any, when an access is not mapped by
    /// `memory_interface`, or BAL_EXCEPTION_UNDEFINED_INSTRUCTION, with a zero
    /// bal_cpu_t::exception_immediate, for an SME instruction while the streaming mode or ZA
    /// storage it needs is off. Returns non-zero otherwise.
    ///
    /// # Safety
    ///
    /// `instruction` must be accepted by bal_is_sve_supported().
    uint64_t bal_execute_sve(bal_cpu_t                    *cpu,
                             const bal_memory_interface_t *memory_interface,
                             uint32_t                      instruction);

    /// SMSTART and SMSTOP (MSR SVCRSM, SVCRZA and SVCRSMZA, immediate) and MSR SVCR, Xt: sets
    /// PSTATE.SM and PSTATE.ZA in bal_cpu_t::svcr. Changing SM zeroes the Z and P registers and
    /// FFR and sets FPSR to 0x0800009F; enabling ZA zeroes ZA and ZT0. Returns zero.
    ///
    /// # Safety
    ///
    /// `instruction` must be one of those forms.
    uint64_t bal_execute_svcr_write(bal_cpu_t *cpu, uint32_t instruction);

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* BALLISTIC_BAL_SVE_H */

/*** end of file ***/
