//! Runtime support for the guest's Advanced SIMD instructions.
//!
//! bal_execute_simd() executes one Advanced SIMD instruction, vector or scalar, on the SIMD&FP
//! registers in the cpu structure. Floating-point lanes use the element operations of
//! bal_float.h, so they follow the AArch64 rules for NaNs, FPCR and FPSR exactly; the saturating
//! integer operations set FPSR.QC.
#ifndef BALLISTIC_BAL_SIMD_H
#define BALLISTIC_BAL_SIMD_H

#include "backend/bal_cpu.h"
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

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* BALLISTIC_BAL_SIMD_H */

/*** end of file ***/
