//! Runtime support for the guest's scalar floating-point instructions.
//!
//! bal_execute_float() executes one scalar floating-point instruction with the exact AArch64
//! semantics: NaN propagation and the default NaN, FPCR.FZ flushing, the FPCR rounding mode,
//! saturating conversions to integers, and the cumulative FPSR exception flags. Translated code
//! calls it with the guest state flushed to memory.
#ifndef BALLISTIC_BAL_FLOAT_H
#define BALLISTIC_BAL_FLOAT_H

#include "backend/bal_cpu.h"
#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif /* __cplusplus */

    /// Returns true if bal_execute_float() implements `instruction`.
    bool bal_is_float_supported(uint32_t instruction);

    /// Executes one scalar floating-point instruction against `cpu`, updating its registers,
    /// flags and FPSR. Always returns 1.
    ///
    /// # Safety
    ///
    /// `instruction` must be accepted by bal_is_float_supported().
    uint64_t bal_execute_float(bal_cpu_t *cpu, uint32_t instruction);

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* BALLISTIC_BAL_FLOAT_H */

/*** end of file ***/
