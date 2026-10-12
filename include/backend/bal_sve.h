//! Runtime support for the guest's SVE and SVE2 instructions.
//!
//! The guest CPU implements SVE with the smallest vector length, 128 bits, so the Z registers are
//! the SIMD&FP registers in bal_cpu_t::v and the predicate registers P0-P15 and FFR are 16 bits
//! each, in bal_cpu_t::p and bal_cpu_t::ffr. bal_execute_sve() executes one instruction on that
//! state; translated code calls it with the guest registers flushed to memory.
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

    /// Executes one SVE instruction against `cpu`. Returns zero, after a first-faulting load's
    /// partial work if any, when an access is not mapped by `memory_interface`, and non-zero
    /// otherwise.
    ///
    /// # Safety
    ///
    /// `instruction` must be accepted by bal_is_sve_supported().
    uint64_t bal_execute_sve(bal_cpu_t                    *cpu,
                             const bal_memory_interface_t *memory_interface,
                             uint32_t                      instruction);

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* BALLISTIC_BAL_SVE_H */

/*** end of file ***/
