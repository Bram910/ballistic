//! Runtime support for rarely used integer and system instructions.
//!
//! bal_execute_integer_helper() executes CRC32/CRC32C and the flag manipulation instructions
//! CFINV, RMIF, SETF8, SETF16, AXFLAG and XAFLAG, and the FEAT_CSSC ABS, CNT, CTZ, SMAX, SMIN,
//! UMAX and UMIN on the guest state; bal_execute_cache_zero()
//! executes DC ZVA. Translated code calls them with the guest state flushed to memory.
#ifndef BALLISTIC_BAL_SYSTEM_H
#define BALLISTIC_BAL_SYSTEM_H

#include "backend/bal_cpu.h"
#include "bal_memory.h"
#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif /* __cplusplus */

    /// The DC ZVA block size: 4 << DCZID_EL0.BS bytes, with the BS value that MRS DCZID_EL0
    /// reports.
#define BAL_CACHE_ZERO_BLOCK_BYTES 64U

    /// Returns true if bal_execute_integer_helper() implements `instruction`.
    bool bal_is_integer_helper_supported(uint32_t instruction);

    /// Executes CRC32, CRC32C, CFINV, RMIF, SETF8 or SETF16 against `cpu`. Always returns 1.
    ///
    /// # Safety
    ///
    /// `instruction` must be accepted by bal_is_integer_helper_supported().
    uint64_t bal_execute_integer_helper(bal_cpu_t *cpu, uint32_t instruction);

    /// Executes DC ZVA: zeroes the BAL_CACHE_ZERO_BLOCK_BYTES-byte block holding the address in
    /// the instruction's Xt. Returns zero, writing nothing, when the block is not mapped by
    /// `memory_interface`, and non-zero otherwise.
    /// Executes MRS of CNTFRQ_EL0, CNTPCT_EL0 or CNTVCT_EL0 (`instruction`) against `cpu`, using
    /// bal_cpu_t::counter_frequency and read_counter. Always returns 1.
    uint64_t bal_execute_counter_read(bal_cpu_t *cpu, uint32_t instruction);

    uint64_t bal_execute_cache_zero(bal_cpu_t                    *cpu,
                                    const bal_memory_interface_t *memory_interface,
                                    uint32_t                      instruction);

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* BALLISTIC_BAL_SYSTEM_H */

/*** end of file ***/
