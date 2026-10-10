//! Runtime support for the guest's Advanced SIMD instructions.
//!
//! bal_execute_simd() executes one Advanced SIMD instruction, vector or scalar, on the SIMD&FP
//! registers in the cpu structure. Floating-point lanes use the element operations of
//! bal_float.h, so they follow the AArch64 rules for NaNs, FPCR and FPSR exactly; the saturating
//! integer operations set FPSR.QC.
#ifndef BALLISTIC_BAL_SIMD_H
#define BALLISTIC_BAL_SIMD_H

#include "backend/bal_cpu.h"
#include "bal_memory.h"
#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif /* __cplusplus */

    /// Returns true if bal_execute_simd() implements `instruction`.
    bool bal_is_simd_supported(uint32_t instruction);

    /// Executes one Advanced SIMD instruction against `cpu`. Always returns 1.
    ///
    /// # Safety
    ///
    /// `instruction` must be accepted by bal_is_simd_supported().
    uint64_t bal_execute_simd(bal_cpu_t *cpu, uint32_t instruction);

    /// Returns true if bal_execute_simd_memory() implements `instruction`.
    bool bal_is_simd_memory_supported(uint32_t instruction);

    /// Executes one Advanced SIMD structure load or store: LD1-LD4 and ST1-ST4 of multiple
    /// structures or a single lane, and LD1R-LD4R, with optional post-index writeback.
    ///
    /// Elements are accessed one at a time in architectural order, translating each address.
    /// Returns zero when an access is not mapped by `memory_interface`; the elements before it
    /// have then been transferred and the base register is not written back, and the caller
    /// raises a data abort. Returns non-zero otherwise.
    ///
    /// # Safety
    ///
    /// `instruction` must be accepted by bal_is_simd_memory_supported().
    uint64_t bal_execute_simd_memory(bal_cpu_t                    *cpu,
                                     const bal_memory_interface_t *memory_interface,
                                     uint32_t                      instruction);

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* BALLISTIC_BAL_SIMD_H */

/*** end of file ***/
