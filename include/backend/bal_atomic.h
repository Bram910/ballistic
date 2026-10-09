//! Runtime support for the guest's synchronisation instructions.
//!
//! Exclusives, load-acquire/store-release, compare-and-swap and the LSE atomic memory operations
//! are executed by bal_execute_atomic(), which translated code calls with the guest state flushed
//! to memory. Each instruction becomes one host atomic operation on the translated address, so the
//! guarantees hold when several guest threads share memory.
#ifndef BALLISTIC_BAL_ATOMIC_H
#define BALLISTIC_BAL_ATOMIC_H

#include "backend/bal_cpu.h"
#include "bal_memory.h"
#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif /* __cplusplus */

    /// Returns true if bal_execute_atomic() implements `instruction`.
    bool bal_is_atomic_supported(uint32_t instruction);

    /// Executes one synchronisation instruction against `cpu`.
    ///
    /// Returns non-zero on success. Returns zero, leaving every register unchanged, when the
    /// access is not mapped by `memory_interface` or is misaligned where the architecture requires
    /// alignment; the caller then raises a data abort.
    ///
    /// The exclusive monitor compares values, as QEMU does: a store-exclusive succeeds when it
    /// targets the address of the last load-exclusive and memory still holds the value that load
    /// returned.
    ///
    /// # Safety
    ///
    /// `instruction` must be accepted by bal_is_atomic_supported().
    uint64_t bal_execute_atomic(bal_cpu_t                    *cpu,
                                const bal_memory_interface_t *memory_interface,
                                uint32_t                      instruction);

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* BALLISTIC_BAL_ATOMIC_H */

/*** end of file ***/
