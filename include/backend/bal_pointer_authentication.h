//! Runtime support for pointer authentication (FEAT_PAuth, as in the Cortex-A78C).
//!
//! bal_execute_pointer_authentication() executes PACIA and the other PAC*, AUT* and XPAC*
//! instructions, PACGA, their hint-space forms such as PACIASP and AUTIASP, and the
//! authenticated branches BRAA, BLRAA, RETAA and their variants.
//! bal_execute_authenticated_load() executes LDRAA and LDRAB. Translated code calls them with
//! the guest state flushed to memory.
//!
//! The pointer authentication codes are computed with the QARMA5 block cipher the architecture
//! specifies, from the keys in bal_cpu_t::pac_keys, as an EL0 process sees them: SCTLR_EL1
//! enables the keys and TCR_EL1 sets the address size and the top-byte-ignore rules.
#ifndef BALLISTIC_BAL_POINTER_AUTHENTICATION_H
#define BALLISTIC_BAL_POINTER_AUTHENTICATION_H

#include "backend/bal_cpu.h"
#include "bal_memory.h"
#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif /* __cplusplus */

    /// Returns true if bal_execute_pointer_authentication() implements `instruction`.
    bool bal_is_pointer_authentication_supported(uint32_t instruction);

    /// Returns true if `instruction` is one of the authenticated branches, which set
    /// bal_cpu_t::pc.
    bool bal_is_authenticated_branch(uint32_t instruction);

    /// Returns true if `instruction` is LDRAA or LDRAB.
    bool bal_is_authenticated_load(uint32_t instruction);

    /// Executes a pointer authentication instruction against `cpu`. The authenticated branches
    /// store their target to bal_cpu_t::pc; BLRAA and BLRAB leave the link register to the
    /// caller. Always returns 1.
    ///
    /// # Safety
    ///
    /// `instruction` must be accepted by bal_is_pointer_authentication_supported().
    uint64_t bal_execute_pointer_authentication(bal_cpu_t *cpu, uint32_t instruction);

    /// Executes LDRAA or LDRAB. Returns zero, changing nothing, when the authenticated address
    /// is not mapped by `memory_interface`, and non-zero otherwise.
    uint64_t bal_execute_authenticated_load(bal_cpu_t                    *cpu,
                                            const bal_memory_interface_t *memory_interface,
                                            uint32_t                      instruction);

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* BALLISTIC_BAL_POINTER_AUTHENTICATION_H */

/*** end of file ***/
