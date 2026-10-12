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

    /// A pointer and its signed form under one key and modifier, as the hint-space forms and
    /// RETAA and RETAB produced them. Translated code signs a pointer that matches `plain` by
    /// replacing it with `signed_pointer`, and authenticates one that matches `signed_pointer` by
    /// replacing it with `plain`, without calling the helper. Only pairs where signing and
    /// authenticating are exact inverses are recorded.
    typedef struct
    {
        /// The key plus one, so that an empty entry matches nothing.
        uint64_t key;
        uint64_t modifier;
        uint64_t plain;
        uint64_t signed_pointer;
    } bal_pac_memo_entry_t;

#define BAL_PAC_MEMO_ENTRIES 256U

    /// The entry of a pointer and modifier. A pointer's bits below 25 are never part of its code,
    /// so the plain and signed forms share it.
#define BAL_PAC_MEMO_INDEX(pointer, modifier) \
    ((((modifier) >> 4U) ^ ((pointer) >> 2U)) & (BAL_PAC_MEMO_ENTRIES - 1U))

    /// The memo, with the keys and TCR_EL1 its entries were computed under.
    typedef struct
    {
        uint64_t             configuration[8];
        bal_pac_memo_entry_t entries[BAL_PAC_MEMO_ENTRIES];
    } bal_pac_memo_t;

    /// Points bal_cpu_t::jit_pac_memo at `memo`, emptying it first if the instruction keys or
    /// TCR_EL1 changed since it was filled.
    void bal_pac_memo_attach(bal_pac_memo_t *memo, bal_cpu_t *cpu);

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
