#ifndef BALLISTIC_BAL_X86_ASSEMBLER_H
#define BALLISTIC_BAL_X86_ASSEMBLER_H

#include "bal_errors.h"
#include "bal_memory.h"
#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif /* __cplusplus */

    typedef enum
    {
        BAL_X86_INVALID = -1,

        /// Accumulator Register.
        BAL_X86_RAX = 0,

        /// Counter Register.
        BAL_X86_RCX,

        /// Data Register.
        BAL_X86_RDX,

        /// Base register.
        BAL_X86_RBX,

        /// Stack pointer.
        BAL_X86_RSP,

        /// Base pointer.
        ///
        /// We use this register holds the base [`bal_cpu_t`] context pointer.
        BAL_X86_RBP,

        /// Source Index.
        BAL_X86_RSI,

        /// Destination Index.
        BAL_X86_RDI,

        /// Extended Register 8.
        BAL_X86_R8,

        /// Extended Register 9.
        BAL_X86_R9,

        /// Extended Register 10.
        BAL_X86_R10,

        /// Extended Register 11.
        BAL_X86_R11,

        /// Extended Register 12.
        BAL_X86_R12,

        /// Extended Register 13.
        BAL_X86_R13,

        /// Extended Register 14.
        BAL_X86_R14,

        /// Extended Register 15.
        BAL_X86_R15,
    } bal_x86_register_t;

    /// x86 Condition Codes for SETcc and Jcc instructions.
    typedef enum
    {
        BAL_X86_COND_O  = 0x0, // Overflow (OF = 1)
        BAL_X86_COND_NO = 0x1, // No Overflow (OF = 0)
        BAL_X86_COND_B  = 0x2, // Below / Carry (CF = 1)
        BAL_X86_COND_AE = 0x3, // Above or Equal / No Carry (CF = 0)
        BAL_X86_COND_E  = 0x4, // Equal / Zero (ZF = 1)
        BAL_X86_COND_NE = 0x5, // Not Equal / Not Zero (ZF = 0)
        BAL_X86_COND_BE = 0x6, // Below or Equal (CF = 1 or ZF = 1)
        BAL_X86_COND_A  = 0x7, // Above (CF = 0 and ZF = 0)
        BAL_X86_COND_S  = 0x8, // Sign (SF = 1)
        BAL_X86_COND_NS = 0x9, // No Sign (SF = 0)
        BAL_X86_COND_P  = 0xA, // Parity (PF = 1)
        BAL_X86_COND_NP = 0xB, // No Parity (PF = 0)
        BAL_X86_COND_L  = 0xC, // Less (SF != OF)
        BAL_X86_COND_GE = 0xD, // Greater or Equal (SF == OF)
        BAL_X86_COND_LE = 0xE, // Less or Equal (ZF = 1 or SF != OF)
        BAL_X86_COND_G  = 0xF, // Greater (ZF = 0 and SF == OF)
    } bal_x86_condition_t;

    static_assert(0xF == BAL_X86_COND_G,
                  "BAL_X86_COND_G Must be the last element in the enum and equals to 0xF to cast "
                  "to uint8_t safely");

    /// The x86-64 assembler state.
    typedef struct
    {
        /// Memory buffer where machine code bytes are written.
        uint8_t *buffer;

        /// Pointer to the executable view of the memory buffer.
        uint8_t *rx_buffer;

        /// Maximum number of bytes this assembler can write.
        size_t capacity;

        /// Current byte index in the buffer.
        size_t offset;

        /// Current error state. If this is not [`BAL_SUCCESS`], all emit calls will fail.
        bal_error_t status;

        uint32_t pad;
    } bal_x86_assembler_t;

    static_assert(40 == sizeof(bal_x86_assembler_t), "Struct size mismatch");

    /// Initializes the x86-64 assembler with `executable_buffer`.
    ///
    /// # Safety
    ///
    /// `executable_buffer` must point to a valid allocation of at least `size` bytes. The buffer
    /// must also be configured with executable page permissions by the host OS.
    ///
    /// # Errors
    ///
    /// Returns [`BAL_SUCCESS`] on success.
    ///
    /// Returns [`BAL_ERROR_INVALID_ARGUMENT`] if `assembler` or `executable_buffer` is `NULL` or
    /// `size` is 0.
    bal_error_t bal_x86_assembler_init(bal_x86_assembler_t    *assembler,
                                       bal_executable_buffer_t executable_buffer,
                                       size_t                  size);

    /// Resets the assembler back to its initial state.
    ///
    /// # Safety
    ///
    /// `assembler` must be valid.
    void bal_x86_assembler_reset(bal_x86_assembler_t *assembler);

    /// Emits an add immediate value to a memory location pointed to by RBP + offset.
    ///
    /// Assembly equivalent: `add qword ptr [rbp + offset], immediate`
    ///
    /// # Warning
    ///
    /// This function fails if:
    ///
    /// - `assembler` is `NULL`
    /// - `assembler->status` != [`BAL_SUCCESS`]
    /// - `assembler->buffer` is full.
    void bal_x86_emit_add_mem64_rbp_offset_imm(bal_x86_assembler_t *assembler,
                                               int32_t              offset,
                                               int32_t              immediate);

    /// Emits an add 32-bit immediate value to a 64-bit register.
    ///
    /// Assembly equivalent: `add destination, immediate`
    ///
    /// # Warning
    ///
    /// This function fails if:
    ///
    /// - `assembler` is `NULL`
    /// - `assembler->status` != [`BAL_SUCCESS`]
    /// - `assembler->buffer` is full.
    void bal_x86_emit_add_r64_imm32(bal_x86_assembler_t *assembler,
                                    bal_x86_register_t   destination,
                                    int32_t              immediate);

    /// Emits an add 32-bit immediate value to a 32-bit register. The result is zero extended into
    /// the full 64-bit register.
    ///
    /// Assembly equivalent: `add destination32, immediate`
    ///
    /// # Warning
    ///
    /// This function fails if:
    ///
    /// - `assembler` is `NULL`
    /// - `assembler->status` != [`BAL_SUCCESS`]
    /// - `assembler->buffer` is full.
    void bal_x86_emit_add_r32_imm32(bal_x86_assembler_t *assembler,
                                    bal_x86_register_t   destination,
                                    int32_t              immediate);

    /// Emits an add 64-bit register to another 64-bit register.
    ///
    /// Assembly equivalent: `add destination, source`
    ///
    /// # Warning
    ///
    /// This function fails if:
    ///
    /// - `assembler` is `NULL`
    /// - `assembler->status` != [`BAL_SUCCESS`]
    /// - `assembler->buffer` is full.
    void bal_x86_emit_add_r64_r64(bal_x86_assembler_t *assembler,
                                  bal_x86_register_t   destination,
                                  bal_x86_register_t   source);

    /// Emits an add 32-bit register to another 32-bit register. The result is zero extended
    /// into the full 64-bit register.
    ///
    /// Assembly equivalent: `add destination32, source32`
    ///
    /// # Warning
    ///
    /// This function fails if:
    ///
    /// - `assembler` is `NULL`
    /// - `assembler->status` != [`BAL_SUCCESS`]
    /// - `assembler->buffer` is full.
    void bal_x86_emit_add_r32_r32(bal_x86_assembler_t *assembler,
                                  bal_x86_register_t   destination,
                                  bal_x86_register_t   source);

    /// Emits a compare byte instruction between a memory location pointed to by RBP + offset and
    /// an 8-bit immediate.
    ///
    /// Assembly equivalent: `cmp byte ptr [rbp + offset], immediate`
    ///
    /// # Warning
    ///
    /// This function fails if:
    ///
    /// - `assembler` is `NULL`
    /// - `assembler->status` != [`BAL_SUCCESS`]
    /// - `assembler->buffer` is full.
    void bal_x86_emit_cmp_mem8_rbp_offset_imm(bal_x86_assembler_t *assembler,
                                              int32_t              offset,
                                              int8_t               immediate);

    /// Emits a subtract 32-bit immediate from a 64-bit register.
    ///
    /// Assembly equivalent: `sub destination, immediate`
    ///
    /// # Warning
    ///
    /// This function fails if:
    ///
    /// - `assembler` is `NULL`
    /// - `assembler->status` != [`BAL_SUCCESS`]
    /// - `assembler->buffer` is full.
    void bal_x86_emit_sub_r64_imm32(bal_x86_assembler_t *assembler,
                                    bal_x86_register_t   destination,
                                    int32_t              immediate);

    /// Emits a subtract 32-bit immediate from a 32-bit register. The result is zero extended
    /// into the full 64-bit register.
    ///
    /// Assembly equivalent: `sub destination32, immediate`
    ///
    /// # Warning
    ///
    /// This function fails if:
    ///
    /// - `assembler` is `NULL`
    /// - `assembler->status` != [`BAL_SUCCESS`]
    /// - `assembler->buffer` is full.
    void bal_x86_emit_sub_r32_imm32(bal_x86_assembler_t *assembler,
                                    bal_x86_register_t   destination,
                                    int32_t              immediate);

    /// Emits a subtract 64-bit register from another 64-bit register.
    ///
    /// Assembly equivalent: `sub destination, source`
    ///
    /// # Warning
    ///
    /// This function fails if:
    ///
    /// - `assembler` is `NULL`
    /// - `assembler->status` != [`BAL_SUCCESS`]
    /// - `assembler->buffer` is full.
    void bal_x86_emit_sub_r64_r64(bal_x86_assembler_t *assembler,
                                  bal_x86_register_t   destination,
                                  bal_x86_register_t   source);

    /// Emits a subtract 32-bit register from another 32-bit register. The result is zero extended
    /// into the full 64-bit register.
    ///
    /// Assembler equivalent: `sub destination32, source32`
    ///
    /// # Warning
    ///
    /// This function fails if:
    ///
    /// - `assembler` is `NULL`
    /// - `assembler->status` != [`BAL_SUCCESS`]
    /// - `assembler->buffer` is full.
    void bal_x86_emit_sub_r32_r32(bal_x86_assembler_t *assembler,
                                  bal_x86_register_t   destination,
                                  bal_x86_register_t   source);

    /// Emits a bitwise AND instruction between two 64-bit registers.
    ///
    /// Assembly equivalent: `and destination, source`.
    ///
    /// # Safety
    ///
    /// This function fails if:
    ///
    /// - `assembler` is `NULL`.
    /// - `assembler->status` != [`BAL_SUCCESS`]
    /// - `assembler->buffer` is full.
    void bal_x86_emit_and_r64_r64(bal_x86_assembler_t *assembler,
                                  bal_x86_register_t   destination,
                                  bal_x86_register_t   source);

    /// Emits a bitwise AND instruction between two 32-bit registers. The upper 32 bits of
    /// `destination` are zeroed.
    ///
    /// Assembly equivalent: `and destination_32, source_32`.
    ///
    /// # Safety
    ///
    /// This function fails if:
    ///
    /// - `assembler` is `NULL`.
    /// - `assembler->status` != [`BAL_SUCCESS`]
    /// - `assembler->buffer` is full.
    void bal_x86_emit_and_r32_r32(bal_x86_assembler_t *assembler,
                                  bal_x86_register_t   destination,
                                  bal_x86_register_t   source);

    /// Emits a conditional jump to a relative 32-bit offset.
    ///
    /// Assembler equivalent : `jcc offset`
    ///
    /// # Warning
    ///
    /// This function fails if:
    ///
    /// - `assembler` is `NULL`.
    /// - `assembler->status` != [`BAL_SUCCESS`].
    /// - `assembler->buffer` is full.
    void bal_x86_emit_jcc_rel32(bal_x86_assembler_t *assembler,
                                bal_x86_condition_t  condition,
                                int32_t              offset);

    /// Emits an unconditional jump to the address in the specified 64-bit register.
    ///
    /// Assembly equivalent: `jmp reg64`
    ///
    /// # Warning
    ///
    /// This function fails if:
    ///
    /// - `assembler` is `NULL`.
    /// - `assembler->status` != [`BAL_SUCCESS`].
    /// - `assembler->buffer` is full.
    void bal_x86_emit_jmp_r64(bal_x86_assembler_t *assembler, bal_x86_register_t reg);

    /// Emits an unconditional jump to a relative 32-bit offset.
    ///
    /// Assembly equivalent: `jmp offset`
    ///
    /// # Warning
    ///
    /// This function fails if:
    ///
    /// - `assembler` is `NULL`.
    /// - `assembler->status` != [`BAL_SUCCESS`]
    /// - `assembler->buffer` is full.
    void bal_x86_emit_jmp_rel32(bal_x86_assembler_t *assembler, int32_t offset);

    /// Emits a memory load using the 32-bit displacement `offset`.
    ///
    /// Assembler equivalent: `mov destination, [rbp + offset]`
    ///
    /// # Warning
    ///
    /// This function fails if:
    ///
    /// - `assembler` is `NULL`.
    /// - `assembler->status` != [`BAL_SUCCESS`]
    /// - `assembler->buffer` is full.
    void bal_x86_emit_load_r64_rbp_offset(bal_x86_assembler_t *assembler,
                                          bal_x86_register_t   destination,
                                          int32_t              offset);

    /// Emits a memory load from the address in a 64-bit register.
    ///
    /// Assembly equivalent: `mov destination, [base]`
    ///
    /// Handles all 16 x86-64 registers including RSP and RBP.
    ///
    /// # Warning
    ///
    /// This function fails if:
    ///
    /// - `assembler` is `NULL`.
    /// - `assembler->status` != [`BAL_SUCCESS`]
    /// - `assembler->buffer` is full.
    void bal_x86_emit_load_r64_from_r64(bal_x86_assembler_t *assembler,
                                        bal_x86_register_t   destination,
                                        bal_x86_register_t   base);

    /// Emits an instruction to move a 64-bit immediate value into a 64-bit register.
    ///
    /// Assembly equivalent: `mov destination, immediate`
    ///
    /// # Warning
    ///
    /// This function fails if:
    ///
    /// - `assembler` is `NULL`.
    /// - `assembler->status` != [`BAL_SUCCESS`]
    /// - `assembler->buffer` is full.
    void bal_x86_emit_mov_r64_imm64(bal_x86_assembler_t *assembler,
                                    bal_x86_register_t   destination,
                                    uint64_t             immediate);

    /// Emits an instruction to move 64-bit register `source` into `destination`.
    ///
    /// Assembly equivalent: `mov reg64, reg64`
    ///
    /// # Warning
    ///
    /// This function fails if:
    ///
    /// - `assembler` is `NULL`.
    /// - `assembler->status` != [`BAL_SUCCESS`]
    /// - `assembler->buffer` is full.
    void bal_x86_emit_mov_r64_r64(bal_x86_assembler_t *assembler,
                                  bal_x86_register_t   destination,
                                  bal_x86_register_t   source);

    /// Emits a bitwise OR instruction between two 64-bit registers.
    ///
    /// Assembly equivalent: `or destination, source`.
    ///
    /// # Safety
    ///
    /// This function fails if:
    ///
    /// - `assembler` is `NULL`.
    /// - `assembler->status` != [`BAL_SUCCESS`]
    /// - `assembler->buffer` is full.
    void bal_x86_emit_or_r64_r64(bal_x86_assembler_t *assembler,
                                 bal_x86_register_t   destination,
                                 bal_x86_register_t   source);

    /// Emits a bitwise OR between two 32-bit registers. The upper 32 bits of `destination` are
    /// zeroed.
    ///
    /// Assembly equivalent: `or destination_32, source_32`
    ///
    /// # Warning
    ///
    /// This function fails if:
    ///
    /// - `assembler` is `NULL`.
    /// - `assembler->status` != [`BAL_SUCCESS`]
    /// - `assembler->buffer` is full.
    void bal_x86_emit_or_r32_r32(bal_x86_assembler_t *assembler,
                                 bal_x86_register_t   destination,
                                 bal_x86_register_t   source);

    /// Emits a bitwise NOT of a 64-bit register.
    ///
    /// Assembly equivalent: `not destination`
    ///
    /// # Warning
    ///
    /// This function fails if:
    ///
    /// - `assembler` is `NULL`.
    /// - `assembler->status` != [`BAL_SUCCESS`]
    /// - `assembler->buffer` is full.
    void bal_x86_emit_not_r64(bal_x86_assembler_t *assembler, bal_x86_register_t destination);

    /// Emits a bitwise NOT of a 32-bit register. The upper 32 bits of `destination` are zeroed.
    ///
    /// Assembly equivalent: `not destination_32`
    ///
    /// # Warning
    ///
    /// This function fails if:
    ///
    /// - `assembler` is `NULL`.
    /// - `assembler->status` != [`BAL_SUCCESS`]
    /// - `assembler->buffer` is full.
    void bal_x86_emit_not_r32(bal_x86_assembler_t *assembler, bal_x86_register_t destination);

    /// Emits a two's complement negation of a 64-bit register.
    ///
    /// Assembly equivalent: `neg destination`
    ///
    /// # Warning
    ///
    /// This function fails if:
    ///
    /// - `assembler` is `NULL`.
    /// - `assembler->status` != [`BAL_SUCCESS`]
    /// - `assembler->buffer` is full.
    void bal_x86_emit_neg_r64(bal_x86_assembler_t *assembler, bal_x86_register_t destination);

    /// Emits a two's complement negation of a 32-bit register. The upper 32 bits of `destination`
    /// are zeroed.
    ///
    /// Assembly equivalent: `neg destination_32`
    ///
    /// # Warning
    ///
    /// This function fails if:
    ///
    /// - `assembler` is `NULL`.
    /// - `assembler->status` != [`BAL_SUCCESS`]
    /// - `assembler->buffer` is full.
    void bal_x86_emit_neg_r32(bal_x86_assembler_t *assembler, bal_x86_register_t destination);

    /// Emits a 64-bit unsigned division of RDX:RAX by `source`, leaving the quotient in RAX and the
    /// remainder in RDX. The CPU raises #DE when `source` is zero or the quotient overflows.
    ///
    /// Assembly equivalent: `div source`
    ///
    /// # Warning
    ///
    /// This function fails if:
    ///
    /// - `assembler` is `NULL`.
    /// - `assembler->status` != [`BAL_SUCCESS`]
    /// - `assembler->buffer` is full.
    void bal_x86_emit_div_r64(bal_x86_assembler_t *assembler, bal_x86_register_t source);

    /// Emits a 32-bit unsigned division of EDX:EAX by `source`. The CPU raises #DE when `source` is
    /// zero or the quotient overflows.
    ///
    /// Assembly equivalent: `div source_32`
    ///
    /// # Warning
    ///
    /// This function fails if:
    ///
    /// - `assembler` is `NULL`.
    /// - `assembler->status` != [`BAL_SUCCESS`]
    /// - `assembler->buffer` is full.
    void bal_x86_emit_div_r32(bal_x86_assembler_t *assembler, bal_x86_register_t source);

    /// Emits a 64-bit signed division of RDX:RAX by `source`, leaving the quotient in RAX and the
    /// remainder in RDX. The CPU raises #DE when `source` is zero or the quotient overflows.
    ///
    /// Assembly equivalent: `idiv source`
    ///
    /// # Warning
    ///
    /// This function fails if:
    ///
    /// - `assembler` is `NULL`.
    /// - `assembler->status` != [`BAL_SUCCESS`]
    /// - `assembler->buffer` is full.
    void bal_x86_emit_idiv_r64(bal_x86_assembler_t *assembler, bal_x86_register_t source);

    /// Emits a 32-bit signed division of EDX:EAX by `source`. The CPU raises #DE when `source` is
    /// zero or the quotient overflows.
    ///
    /// Assembly equivalent: `idiv source_32`
    ///
    /// # Warning
    ///
    /// This function fails if:
    ///
    /// - `assembler` is `NULL`.
    /// - `assembler->status` != [`BAL_SUCCESS`]
    /// - `assembler->buffer` is full.
    void bal_x86_emit_idiv_r32(bal_x86_assembler_t *assembler, bal_x86_register_t source);

    /// Emits a sign extension of RAX into RDX:RAX.
    ///
    /// Assembly equivalent: `cqo`
    ///
    /// # Warning
    ///
    /// This function fails if:
    ///
    /// - `assembler` is `NULL`.
    /// - `assembler->status` != [`BAL_SUCCESS`]
    /// - `assembler->buffer` is full.
    void bal_x86_emit_cqo(bal_x86_assembler_t *assembler);

    /// Emits a sign extension of EAX into EDX:EAX.
    ///
    /// Assembly equivalent: `cdq`
    ///
    /// # Warning
    ///
    /// This function fails if:
    ///
    /// - `assembler` is `NULL`.
    /// - `assembler->status` != [`BAL_SUCCESS`]
    /// - `assembler->buffer` is full.
    void bal_x86_emit_cdq(bal_x86_assembler_t *assembler);

    /// Emits a 64-bit conditional move: `destination` = `source` when `condition` holds.
    ///
    /// Assembly equivalent: `cmov<condition> destination, source`
    ///
    /// # Warning
    ///
    /// This function fails if:
    ///
    /// - `assembler` is `NULL`.
    /// - `assembler->status` != [`BAL_SUCCESS`]
    /// - `assembler->buffer` is full.
    void bal_x86_emit_cmov_r64_r64(bal_x86_assembler_t *assembler,
                                   bal_x86_condition_t  condition,
                                   bal_x86_register_t   destination,
                                   bal_x86_register_t   source);

    /// Emits a 32-bit conditional move. The upper 32 bits of `destination` are zeroed whether or
    /// not the condition holds.
    ///
    /// Assembly equivalent: `cmov<condition> destination_32, source_32`
    ///
    /// # Warning
    ///
    /// This function fails if:
    ///
    /// - `assembler` is `NULL`.
    /// - `assembler->status` != [`BAL_SUCCESS`]
    /// - `assembler->buffer` is full.
    void bal_x86_emit_cmov_r32_r32(bal_x86_assembler_t *assembler,
                                   bal_x86_condition_t  condition,
                                   bal_x86_register_t   destination,
                                   bal_x86_register_t   source);

    /// Emits a 64-bit signed multiply keeping the low 64 bits of the product, which are the same
    /// for unsigned operands.
    ///
    /// Assembly equivalent: `imul destination, source`
    ///
    /// # Warning
    ///
    /// This function fails if:
    ///
    /// - `assembler` is `NULL`.
    /// - `assembler->status` != [`BAL_SUCCESS`]
    /// - `assembler->buffer` is full.
    void bal_x86_emit_imul_r64_r64(bal_x86_assembler_t *assembler,
                                   bal_x86_register_t   destination,
                                   bal_x86_register_t   source);

    /// Emits a 32-bit multiply keeping the low 32 bits of the product. The upper 32 bits of
    /// `destination` are zeroed.
    ///
    /// Assembly equivalent: `imul destination_32, source_32`
    ///
    /// # Warning
    ///
    /// This function fails if:
    ///
    /// - `assembler` is `NULL`.
    /// - `assembler->status` != [`BAL_SUCCESS`]
    /// - `assembler->buffer` is full.
    void bal_x86_emit_imul_r32_r32(bal_x86_assembler_t *assembler,
                                   bal_x86_register_t   destination,
                                   bal_x86_register_t   source);

    /// Emits a sign extension of the low 32 bits of `source` into the 64-bit `destination`.
    ///
    /// Assembly equivalent: `movsxd destination, source_32`
    ///
    /// # Warning
    ///
    /// This function fails if:
    ///
    /// - `assembler` is `NULL`.
    /// - `assembler->status` != [`BAL_SUCCESS`]
    /// - `assembler->buffer` is full.
    void bal_x86_emit_movsxd_r64_r32(bal_x86_assembler_t *assembler,
                                     bal_x86_register_t   destination,
                                     bal_x86_register_t   source);

    /// Emits a 32-bit register move, which zero-extends the low 32 bits of `source` into
    /// `destination`.
    ///
    /// Assembly equivalent: `mov destination_32, source_32`
    ///
    /// # Warning
    ///
    /// This function fails if:
    ///
    /// - `assembler` is `NULL`.
    /// - `assembler->status` != [`BAL_SUCCESS`]
    /// - `assembler->buffer` is full.
    void bal_x86_emit_mov_r32_r32(bal_x86_assembler_t *assembler,
                                  bal_x86_register_t   destination,
                                  bal_x86_register_t   source);

    /// Emits a 64-bit logical shift-left by CL, modulo 64.
    ///
    /// Assembly equivalent: `shl destination, cl`
    ///
    /// # Warning
    ///
    /// This function fails if:
    ///
    /// - `assembler` is `NULL`.
    /// - `assembler->status` != [`BAL_SUCCESS`]
    /// - `assembler->buffer` is full.
    void bal_x86_emit_shl_r64_cl(bal_x86_assembler_t *assembler, bal_x86_register_t destination);

    /// Emits a 32-bit logical shift-left by CL, modulo 32. The upper 32 bits of `destination` are
    /// zeroed.
    ///
    /// Assembly equivalent: `shl destination_32, cl`
    ///
    /// # Warning
    ///
    /// This function fails if:
    ///
    /// - `assembler` is `NULL`.
    /// - `assembler->status` != [`BAL_SUCCESS`]
    /// - `assembler->buffer` is full.
    void bal_x86_emit_shl_r32_cl(bal_x86_assembler_t *assembler, bal_x86_register_t destination);

    /// Emits a 64-bit logical shift-right by CL, modulo 64.
    ///
    /// Assembly equivalent: `shr destination, cl`
    ///
    /// # Warning
    ///
    /// This function fails if:
    ///
    /// - `assembler` is `NULL`.
    /// - `assembler->status` != [`BAL_SUCCESS`]
    /// - `assembler->buffer` is full.
    void bal_x86_emit_shr_r64_cl(bal_x86_assembler_t *assembler, bal_x86_register_t destination);

    /// Emits a 32-bit logical shift-right by CL, modulo 32. The upper 32 bits of `destination` are
    /// zeroed.
    ///
    /// Assembly equivalent: `shr destination_32, cl`
    ///
    /// # Warning
    ///
    /// This function fails if:
    ///
    /// - `assembler` is `NULL`.
    /// - `assembler->status` != [`BAL_SUCCESS`]
    /// - `assembler->buffer` is full.
    void bal_x86_emit_shr_r32_cl(bal_x86_assembler_t *assembler, bal_x86_register_t destination);

    /// Emits a 64-bit arithmetic shift-right by CL, modulo 64.
    ///
    /// Assembly equivalent: `sar destination, cl`
    ///
    /// # Warning
    ///
    /// This function fails if:
    ///
    /// - `assembler` is `NULL`.
    /// - `assembler->status` != [`BAL_SUCCESS`]
    /// - `assembler->buffer` is full.
    void bal_x86_emit_sar_r64_cl(bal_x86_assembler_t *assembler, bal_x86_register_t destination);

    /// Emits a 32-bit arithmetic shift-right by CL, modulo 32. The upper 32 bits of `destination`
    /// are zeroed.
    ///
    /// Assembly equivalent: `sar destination_32, cl`
    ///
    /// # Warning
    ///
    /// This function fails if:
    ///
    /// - `assembler` is `NULL`.
    /// - `assembler->status` != [`BAL_SUCCESS`]
    /// - `assembler->buffer` is full.
    void bal_x86_emit_sar_r32_cl(bal_x86_assembler_t *assembler, bal_x86_register_t destination);

    /// Emits a 64-bit rotate-right by CL, modulo 64.
    ///
    /// Assembly equivalent: `ror destination, cl`
    ///
    /// # Warning
    ///
    /// This function fails if:
    ///
    /// - `assembler` is `NULL`.
    /// - `assembler->status` != [`BAL_SUCCESS`]
    /// - `assembler->buffer` is full.
    void bal_x86_emit_ror_r64_cl(bal_x86_assembler_t *assembler, bal_x86_register_t destination);

    /// Emits a 32-bit rotate-right by CL, modulo 32. The upper 32 bits of `destination` are zeroed.
    ///
    /// Assembly equivalent: `ror destination_32, cl`
    ///
    /// # Warning
    ///
    /// This function fails if:
    ///
    /// - `assembler` is `NULL`.
    /// - `assembler->status` != [`BAL_SUCCESS`]
    /// - `assembler->buffer` is full.
    void bal_x86_emit_ror_r32_cl(bal_x86_assembler_t *assembler, bal_x86_register_t destination);

    /// Emits a 64-bit add with carry: `destination` + `source` and CF.
    ///
    /// Assembly equivalent: `adc destination, source`
    ///
    /// # Warning
    ///
    /// This function fails if:
    ///
    /// - `assembler` is `NULL`.
    /// - `assembler->status` != [`BAL_SUCCESS`]
    /// - `assembler->buffer` is full.
    void bal_x86_emit_adc_r64_r64(bal_x86_assembler_t *assembler,
                                  bal_x86_register_t   destination,
                                  bal_x86_register_t   source);

    /// Emits a 32-bit add with carry. The upper 32 bits of `destination` are zeroed.
    ///
    /// Assembly equivalent: `adc destination_32, source_32`
    ///
    /// # Warning
    ///
    /// This function fails if:
    ///
    /// - `assembler` is `NULL`.
    /// - `assembler->status` != [`BAL_SUCCESS`]
    /// - `assembler->buffer` is full.
    void bal_x86_emit_adc_r32_r32(bal_x86_assembler_t *assembler,
                                  bal_x86_register_t   destination,
                                  bal_x86_register_t   source);

    /// Emits a 64-bit subtract with borrow: `destination` - `source` and CF.
    ///
    /// Assembly equivalent: `sbb destination, source`
    ///
    /// # Warning
    ///
    /// This function fails if:
    ///
    /// - `assembler` is `NULL`.
    /// - `assembler->status` != [`BAL_SUCCESS`]
    /// - `assembler->buffer` is full.
    void bal_x86_emit_sbb_r64_r64(bal_x86_assembler_t *assembler,
                                  bal_x86_register_t   destination,
                                  bal_x86_register_t   source);

    /// Emits a 32-bit subtract with borrow. The upper 32 bits of `destination` are zeroed.
    ///
    /// Assembly equivalent: `sbb destination_32, source_32`
    ///
    /// # Warning
    ///
    /// This function fails if:
    ///
    /// - `assembler` is `NULL`.
    /// - `assembler->status` != [`BAL_SUCCESS`]
    /// - `assembler->buffer` is full.
    void bal_x86_emit_sbb_r32_r32(bal_x86_assembler_t *assembler,
                                  bal_x86_register_t   destination,
                                  bal_x86_register_t   source);

    /// Emits a bit test, copying bit `bit` of `destination` into CF.
    ///
    /// Assembly equivalent: `bt destination, bit`
    ///
    /// # Warning
    ///
    /// This function fails if:
    ///
    /// - `assembler` is `NULL`.
    /// - `assembler->status` != [`BAL_SUCCESS`]
    /// - `assembler->buffer` is full.
    /// - `bit` is greater than 63.
    void bal_x86_emit_bt_r64_imm8(bal_x86_assembler_t *assembler,
                                  bal_x86_register_t   destination,
                                  uint8_t              bit);

    /// Emits a complement of the carry flag.
    ///
    /// Assembly equivalent: `cmc`
    ///
    /// # Warning
    ///
    /// This function fails if:
    ///
    /// - `assembler` is `NULL`.
    /// - `assembler->status` != [`BAL_SUCCESS`]
    /// - `assembler->buffer` is full.
    void bal_x86_emit_cmc(bal_x86_assembler_t *assembler);

    /// Emits a 64-bit rotate-right by an 8-bit immediate.
    ///
    /// Assembly equivalent: `ror destination, immediate`
    ///
    /// # Warning
    ///
    /// This function fails if:
    ///
    /// - `assembler` is `NULL`.
    /// - `assembler->status` != [`BAL_SUCCESS`]
    /// - `assembler->buffer` is full.
    /// - `immediate` is greater than 63.
    void bal_x86_emit_ror_r64_imm8(bal_x86_assembler_t *assembler,
                                   bal_x86_register_t   destination,
                                   uint8_t              immediate);

    /// Emits a 32-bit rotate-right by an 8-bit immediate. The upper 32 bits of `destination` are
    /// zeroed.
    ///
    /// Assembly equivalent: `ror destination_32, immediate`
    ///
    /// # Warning
    ///
    /// This function fails if:
    ///
    /// - `assembler` is `NULL`.
    /// - `assembler->status` != [`BAL_SUCCESS`]
    /// - `assembler->buffer` is full.
    /// - `immediate` is greater than 31.
    void bal_x86_emit_ror_r32_imm8(bal_x86_assembler_t *assembler,
                                   bal_x86_register_t   destination,
                                   uint8_t              immediate);

    /// Emits a 64-bit arithmetic shift-right by an 8-bit immediate.
    ///
    /// Assembly equivalent: `sar destination, immediate`
    ///
    /// # Warning
    ///
    /// This function fails if:
    ///
    /// - `assembler` is `NULL`.
    /// - `assembler->status` != [`BAL_SUCCESS`]
    /// - `assembler->buffer` is full.
    /// - `immediate` is greater than 63.
    void bal_x86_emit_sar_r64_imm8(bal_x86_assembler_t *assembler,
                                   bal_x86_register_t   destination,
                                   uint8_t              immediate);

    /// Emits a 32-bit arithmetic shift-right by an 8-bit immediate. The upper 32 bits of
    /// `destination` are zeroed.
    ///
    /// Assembly equivalent: `sar destination_32, immediate`
    ///
    /// # Warning
    ///
    /// This function fails if:
    ///
    /// - `assembler` is `NULL`.
    /// - `assembler->status` != [`BAL_SUCCESS`]
    /// - `assembler->buffer` is full.
    /// - `immediate` is greater than 31.
    void bal_x86_emit_sar_r32_imm8(bal_x86_assembler_t *assembler,
                                   bal_x86_register_t   destination,
                                   uint8_t              immediate);

    /// Emits a 64-bit logical shift-left by an 8-bit immediate.
    ///
    /// Assembly equivalent: `shl destination, immediate`
    ///
    /// # Warning
    ///
    /// This function fails if:
    ///
    /// - `assembler` is `NULL`.
    /// - `assembler->status` != [`BAL_SUCCESS`]
    /// - `assembler->buffer` is full.
    /// - `immediate` is greater than 63.
    void bal_x86_emit_shl_r64_imm8(bal_x86_assembler_t *assembler,
                                   bal_x86_register_t   destination,
                                   uint8_t              immediate);

    /// Emits a 32-bit logical shift-left by an 8-bit immediate. The upper 32 bits of `destination`
    /// are zeroed.
    ///
    /// Assembly equivalent: `shl destination_32, immediate`
    ///
    /// # Warning
    ///
    /// This function fails if:
    ///
    /// - `assembler` is `NULL`.
    /// - `assembler->status` != [`BAL_SUCCESS`]
    /// - `assembler->buffer` is full.
    /// - `immediate` is greater than 31.
    void bal_x86_emit_shl_r32_imm8(bal_x86_assembler_t *assembler,
                                   bal_x86_register_t   destination,
                                   uint8_t              immediate);

    /// Emits a 64-bit logical shift-right by an 8-bit immediate.
    ///
    /// Assembly equivalent: `shr destination, immediate`
    ///
    /// # Warning
    ///
    /// This function fails if:
    ///
    /// - `assembler` is `NULL`.
    /// - `assembler->status` != [`BAL_SUCCESS`]
    /// - `assembler->buffer` is full.
    /// - `immediate` is greater than 63.
    void bal_x86_emit_shr_r64_imm8(bal_x86_assembler_t *assembler,
                                   bal_x86_register_t   destination,
                                   uint8_t              immediate);

    /// Emits a 32-bit logical shift-right by an 8-bit immediate. The upper 32 bits of `destination`
    /// are zeroed.
    ///
    /// Assembly equivalent: `shr destination_32, immediate`
    ///
    /// # Warning
    ///
    /// This function fails if:
    ///
    /// - `assembler` is `NULL`.
    /// - `assembler->status` != [`BAL_SUCCESS`]
    /// - `assembler->buffer` is full.
    /// - `immediate` is greater than 31.
    void bal_x86_emit_shr_r32_imm8(bal_x86_assembler_t *assembler,
                                   bal_x86_register_t   destination,
                                   uint8_t              immediate);

    /// Emits a Stack Pop instruction.
    ///
    /// Assembly equivalent: `push reg64`
    ///
    /// # Warning
    ///
    /// This function fails if:
    ///
    /// - `assembler` is `NULL`.
    /// - `assembler->status` != [`BAL_SUCCESS`]
    /// - `assembler->buffer` is full.
    void bal_x86_emit_pop_r64(bal_x86_assembler_t *assembler, bal_x86_register_t reg);

    /// Emits a Stack Push instruction.
    ///
    /// Assembly equivalent: `push reg64`
    ///
    /// # Warning
    ///
    /// This function fails if:
    ///
    /// - `assembler` is `NULL`.
    /// - `assembler->status` != [`BAL_SUCCESS`]
    /// - `assembler->buffer` is full.
    void bal_x86_emit_push_r64(bal_x86_assembler_t *assembler, bal_x86_register_t reg);

    /// Emits a near return instruction.
    ///
    /// Assembly equivalent: `ret`
    ///
    /// # Warning
    ///
    /// This function fails if:
    ///
    /// - `assembler` is `NULL`.
    /// - `assembler->status` != [`BAL_SUCCESS`]
    /// - `assembler->buffer` is full.
    void bal_x86_emit_ret(bal_x86_assembler_t *assembler);

    /// Emits a SETcc instruction to a memory location pointed to by RBP + offset.
    ///
    /// Assembly equivalent: `setcc byte ptr [rbp + offset]
    ///
    /// # Warning
    ///
    /// This function fails if:
    ///
    /// - `assembler` is `NULL`.
    /// - `assembler->status` != [`BAL_SUCCESS`]
    /// - `assembler->buffer` is full.
    void bal_x86_emit_setcc_mem8_rbp_offset(bal_x86_assembler_t *assembler,
                                            bal_x86_condition_t  condition,
                                            int32_t              offset);

    /// Emits a memory store using a 32-bit displacement.
    ///
    /// Assembly equivalent: `mov[rbp + offset], reg64`
    ///
    /// # Warning
    ///
    /// This function fails if:
    ///
    /// - `assembler` is `NULL`.
    /// - `assembler->status` != [`BAL_SUCCESS`]
    /// - `assembler->buffer` is full.
    void bal_x86_emit_store_r64_rbp_offset(bal_x86_assembler_t *assembler,
                                           bal_x86_register_t   source,
                                           int32_t              offset);

    /// Emit a bitwise TEST instruction between two 64-bit registers.
    ///
    /// Assembly equivalent: `test destination, source`.
    ///
    /// # Safety
    ///
    /// This function fails if:
    ///
    /// - `assembler` is `NULL`.
    /// - `assembler->status` != [`BAL_SUCCESS`]
    /// - `assembler->buffer` is full.
    void bal_x86_emit_test_r64_r64(bal_x86_assembler_t *assembler,
                                   bal_x86_register_t   destination,
                                   bal_x86_register_t   source);

    /// Emits a bitwise AND test between two 32-bit registers, setting the flags without storing
    /// the result.
    ///
    /// Assembly equivalent: `test destination_32, source_32`
    ///
    /// # Warning
    ///
    /// This function fails if:
    ///
    /// - `assembler` is `NULL`.
    /// - `assembler->status` != [`BAL_SUCCESS`]
    /// - `assembler->buffer` is full.
    void bal_x86_emit_test_r32_r32(bal_x86_assembler_t *assembler,
                                   bal_x86_register_t   destination,
                                   bal_x86_register_t   source);

    /// Emits a bitwise XOR instruction between two 64-bit registers.
    ///
    /// Assembly equivalent: `xor destination, source`.
    ///
    /// # Warning
    ///
    /// This function fails if:
    ///
    /// - `assembler` is `NULL`.
    /// - `assembler->status` != [`BAL_SUCCESS`]
    /// - `assembler->buffer` is full.
    void bal_x86_emit_xor_r64_r64(bal_x86_assembler_t *assembler,
                                  bal_x86_register_t   destination,
                                  bal_x86_register_t   source);

    /// Emits a bitwise XOR between two 32-bit registers. The upper 32 bits of `destination` are
    /// zeroed.
    ///
    /// Assembly equivalent: `xor destination_32, source_32`
    ///
    /// # Warning
    ///
    /// This function fails if:
    ///
    /// - `assembler` is `NULL`.
    /// - `assembler->status` != [`BAL_SUCCESS`]
    /// - `assembler->buffer` is full.
    void bal_x86_emit_xor_r32_r32(bal_x86_assembler_t *assembler,
                                  bal_x86_register_t   destination,
                                  bal_x86_register_t   source);

    /// Emits an undefined instruction.
    ///
    /// Assembly equivalent: `ud2`
    ///
    /// # Warning
    ///
    /// This function fails if:
    ///
    /// - `assembler` is `NULL`.
    /// - `assembler->status` != [`BAL_SUCCESS`]
    /// - `assembler->buffer` is full.
    void bal_x86_emit_ud2(bal_x86_assembler_t *assembler);

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif // BALLISTIC_BAL_X86_ASSEMBLER_H

/*** end of file ***/
