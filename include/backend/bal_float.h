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

    /// Element operations of bal_float_element(), with the AArch64 semantics of the
    /// instruction they are named after.
    typedef enum
    {
        BAL_FLOAT_ADD,
        BAL_FLOAT_SUBTRACT,
        BAL_FLOAT_MULTIPLY,
        BAL_FLOAT_DIVIDE,
        BAL_FLOAT_MAXIMUM,
        BAL_FLOAT_MINIMUM,
        BAL_FLOAT_MAXIMUM_NUMBER,
        BAL_FLOAT_MINIMUM_NUMBER,
        /// FMULX: a multiply where infinity times zero is 2 instead of a NaN.
        BAL_FLOAT_MULTIPLY_EXTENDED,
        /// FMLA: c + a * b with a single rounding; c is the addend for NaN priority.
        BAL_FLOAT_MULTIPLY_ADD,
        /// FABD: the absolute value of a - b.
        BAL_FLOAT_ABSOLUTE_DIFFERENCE,
        /// FRECPS: 2 - a * b.
        BAL_FLOAT_RECIPROCAL_STEP,
        /// FRSQRTS: (3 - a * b) / 2.
        BAL_FLOAT_RECIPROCAL_SQUARE_ROOT_STEP,
        BAL_FLOAT_SQUARE_ROOT,
        /// FRECPE and FRSQRTE: the architected 8-bit estimates of 1/a and 1/sqrt(a).
        BAL_FLOAT_RECIPROCAL_ESTIMATE,
        BAL_FLOAT_RECIPROCAL_SQUARE_ROOT_ESTIMATE,
        /// FRECPX: a power of two near 1/a, from the inverted exponent.
        BAL_FLOAT_RECIPROCAL_EXPONENT,
        /// Comparisons return all ones when they hold and zero otherwise, as FCMEQ, FCMGE,
        /// FCMGT, FACGE and FACGT do.
        BAL_FLOAT_COMPARE_EQUAL,
        BAL_FLOAT_COMPARE_GREATER_EQUAL,
        BAL_FLOAT_COMPARE_GREATER,
        BAL_FLOAT_ABSOLUTE_COMPARE_GREATER_EQUAL,
        BAL_FLOAT_ABSOLUTE_COMPARE_GREATER,
    } bal_float_operation_t;

    /// Applies `operation` to half, single or double precision operands (`esize` 16, 32 or 64)
    /// under `fpcr`, accumulating the FPSR flags it raises into `fpsr`. Unary operations use `a`.
    uint64_t bal_float_element(uint64_t              fpcr,
                               uint64_t             *fpsr,
                               uint32_t              esize,
                               bal_float_operation_t operation,
                               uint64_t              a,
                               uint64_t              b,
                               uint64_t              c);

    /// FRINT* on one element: `rmode` 0-3 are to nearest, +infinity, -infinity and zero, 4 is
    /// to nearest with ties away; `exact` raises Inexact (FRINTX).
    uint64_t bal_float_round_element(
        uint64_t fpcr, uint64_t *fpsr, uint32_t esize, uint64_t value, uint32_t rmode, bool exact);

    /// FCVT[NPMZA][SU] on one element, to a `bits`-bit integer scaled by 2^`fraction_bits`.
    uint64_t bal_float_to_integer_element(uint64_t  fpcr,
                                          uint64_t *fpsr,
                                          uint32_t  esize,
                                          uint64_t  value,
                                          uint32_t  fraction_bits,
                                          uint32_t  rmode,
                                          bool      is_unsigned,
                                          uint32_t  bits);

    /// SCVTF and UCVTF on one `bits`-bit element, divided by 2^`fraction_bits`.
    uint64_t bal_float_from_integer_element(uint64_t  fpcr,
                                            uint64_t *fpsr,
                                            uint32_t  esize,
                                            uint64_t  value,
                                            uint32_t  fraction_bits,
                                            bool      is_unsigned,
                                            uint32_t  bits);

    /// FCVTL and FCVTN on one element, from `source_esize` to `target_esize` bits.
    uint64_t bal_float_convert_element(uint64_t  fpcr,
                                       uint64_t *fpsr,
                                       uint32_t  source_esize,
                                       uint32_t  target_esize,
                                       uint64_t  value);

    /// FCVTXN on one element: double to single precision, rounding to odd.
    uint64_t bal_float_narrow_odd_element(uint64_t fpcr, uint64_t *fpsr, uint64_t value);

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* BALLISTIC_BAL_FLOAT_H */

/*** end of file ***/
