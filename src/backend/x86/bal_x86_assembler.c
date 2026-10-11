#include "backend/x86/bal_x86_assembler.h"
#include "bal_assert.h"
#include <stdbool.h>
#include <string.h>

#ifndef NDEBUG

// Strings MUST match the enum value order in bal_x86_macro_t.
static const char *const BAL_X86_CONDITION_CODES[] = {
    "o", "no", "b", "ae", "e", "ne", "be", "a", "s", "ns", "p", "np", "l", "ge", "le", "g",
};

static inline const char *
condition_code_to_string(const bal_x86_condition_t condition)
{
    const size_t num_names = sizeof(BAL_X86_CONDITION_CODES) / sizeof(BAL_X86_CONDITION_CODES[0]);

    if ((size_t)condition < num_names && BAL_X86_CONDITION_CODES[condition] != NULL)
    {
        return BAL_X86_CONDITION_CODES[condition];
    }

    return "???";
}

#endif // NDEBUG

BAL_HOT static bool is_valid_register(int reg);
BAL_HOT static bool can_emit(bal_x86_assembler_t *assembler, size_t size);
BAL_HOT static void emit8(uint8_t *buffer, size_t *offset, uint8_t value);
BAL_HOT static void emit32(uint8_t *buffer, size_t *offset, uint32_t value);
BAL_HOT static void emit64(uint8_t *buffer, size_t *offset, uint64_t value);

/// Emits the REX prefix.
/// w = 1 for 64-bit operands.
/// r = extension for the ModR/W `reg` field.
/// b = extension for the ModR/W `r/m` field or opcode register.
BAL_HOT static void emit_rex(uint8_t *buffer, size_t *offset, uint8_t w, uint8_t r, uint8_t b);

/// Emits the ModR/M byte for Register-to-Register operations.
BAL_HOT static void emit_modrm_register(uint8_t *buffer, size_t *offset, int reg, int rm);

/// Emits the ModR/M byte for Memory Addressing: [RBP + disp32].
BAL_HOT static void emit_modrm_memory_disp32_rbp(uint8_t *buffer, size_t *offset, int reg);

/// Emits `REX C1 /opcode_extension imm8` for shifts and rotates. REX.W is set when `is_64bit` is
/// true, otherwise the operation is 32-bit and zeroes the upper half of `destination`.
BAL_HOT static void emit_shift_imm8(bal_x86_assembler_t *assembler,
                                    bal_x86_register_t   destination,
                                    uint8_t              immediate,
                                    uint8_t              opcode_extension,
                                    bool                 is_64bit,
                                    const char          *mnemonic);

/// Emits `REX opcode ModR/M` for a register to register instruction in the `op reg, r/m` direction,
/// with `destination` in the ModR/M reg field and `source` in the r/m field. REX.W is set when
/// `is_64bit` is true.
BAL_HOT static void emit_register_register(bal_x86_assembler_t *assembler,
                                           uint8_t              opcode,
                                           bool                 is_64bit,
                                           bal_x86_register_t   destination,
                                           bal_x86_register_t   source,
                                           const char          *mnemonic);

/// Emits `[66] REX opcode ModR/M [SIB] [disp8]` for a load or store between `reg` and the memory at
/// the address in `base`. `opcode_size` bytes of `opcode` are emitted.
BAL_HOT static void emit_memory_access(bal_x86_assembler_t *assembler,
                                       uint8_t              operand_size_prefix,
                                       uint8_t              rex_w,
                                       const uint8_t       *opcode,
                                       size_t               opcode_size,
                                       bal_x86_register_t   reg,
                                       bal_x86_register_t   base,
                                       const char          *mnemonic);

/// Emits `REX D3 /opcode_extension` for shifts and rotates by CL. REX.W is set when `is_64bit` is
/// true, otherwise the operation is 32-bit and zeroes the upper half of `destination`.
BAL_HOT static void emit_shift_cl(bal_x86_assembler_t *assembler,
                                  bal_x86_register_t   destination,
                                  uint8_t              opcode_extension,
                                  bool                 is_64bit,
                                  const char          *mnemonic);

/// Emits `REX 0F opcode ModR/M` for a two-byte-opcode register to register instruction in the
/// `op reg, r/m` direction. REX.W is set when `is_64bit` is true.
BAL_HOT static void emit_register_register_0f(bal_x86_assembler_t *assembler,
                                              uint8_t              opcode,
                                              bool                 is_64bit,
                                              bal_x86_register_t   destination,
                                              bal_x86_register_t   source,
                                              const char          *mnemonic);

/// Emits `REX F7 /opcode_extension` for the single operand group 3 instructions such as NOT.
/// REX.W is set when `is_64bit` is true.
BAL_HOT static void emit_group3_register(bal_x86_assembler_t *assembler,
                                         uint8_t              opcode_extension,
                                         bool                 is_64bit,
                                         bal_x86_register_t   destination,
                                         const char          *mnemonic);

bal_error_t
bal_x86_assembler_init(bal_x86_assembler_t          *assembler,
                       const bal_executable_buffer_t executable_buffer,
                       const size_t                  size)
{
    const bal_error_t error = BAL_ERROR_INVALID_ARGUMENT;

    if (BAL_UNLIKELY(NULL == assembler))
    {
        BAL_LOG_ERROR(&bal_thread_logger, "Aborting function: Assembler is NULL");
        return error;
    }

    if (BAL_UNLIKELY(NULL == executable_buffer.rx_pointer)
        || BAL_UNLIKELY(NULL == executable_buffer.rw_pointer))
    {
        BAL_LOG_ERROR(&bal_thread_logger, "Aborting function: Buffer is NULL");
        return error;
    }

    if (BAL_UNLIKELY(0 == size))
    {
        BAL_LOG_ERROR(&bal_thread_logger, "Aborting function: Size is 0");
        return error;
    }

    // WARNING: Assumes executable_buffer.rw_pointer represents a valid writable host memory
    // allocation of at least `size` bytes.
    assembler->buffer = (uint8_t *)executable_buffer.rw_pointer;

    // WARNING: Assumes executable_buffer.rx_pointer represents a valid executable host memory
    // allocation of at least `size` bytes.
    assembler->rx_buffer = (uint8_t *)executable_buffer.rx_pointer;
    assembler->capacity  = size;
    assembler->offset    = 0;
    assembler->status    = BAL_SUCCESS;

    BAL_LOG_INFO(&bal_thread_logger,
                 "x86 Assembler initialized. RW Buffer: %p, RX Buffer: %p, Capacity: %zu bytes",
                 executable_buffer.rw_pointer,
                 executable_buffer.rx_pointer,
                 size);
    return BAL_SUCCESS;
}

void
bal_x86_assembler_reset(bal_x86_assembler_t *assembler)
{
    if (BAL_UNLIKELY(NULL == assembler))
    {
        return;
    }

    assembler->offset = 0;
    assembler->status = BAL_SUCCESS;
    // rx_buffer views the same memory as buffer, and may never be writable.
    (void)memset(assembler->buffer, 0, assembler->capacity);
}

void
bal_x86_emit_add_mem64_rbp_offset_imm(bal_x86_assembler_t *assembler,
                                      const int32_t        offset,
                                      const int32_t        immediate)
{
    if (BAL_UNLIKELY(NULL == assembler))
    {
        return;
    }

    if (BAL_UNLIKELY(assembler->status != BAL_SUCCESS))
    {
        BAL_LOG_ERROR(&bal_thread_logger, "Aborting function: assembler status != BAL_SUCCESS");
        return;
    }

    const bool   fit_in_8_bits          = (immediate >= -128) & (immediate <= 127);
    const size_t instruction_size_bytes = true == fit_in_8_bits ? 8 : 11;
    const bool   can_emit_status        = can_emit(assembler, instruction_size_bytes);

    if (BAL_UNLIKELY(false == can_emit_status))
    {
        return;
    }

    BAL_LOG_DEBUG(&bal_thread_logger,
                  "[0x%04zx] add qword ptr [rbp + 0x%X], 0x%X",
                  assembler->offset,
                  offset,
                  immediate);

    const uint8_t w          = 1;
    const uint8_t r          = 0;
    const uint8_t b          = 0;
    const uint8_t opcode     = fit_in_8_bits ? 0x83 : 0x81;
    const size_t  old_offset = assembler->offset;
    emit_rex(assembler->buffer, &assembler->offset, w, r, b);
    emit8(assembler->buffer, &assembler->offset, opcode);
    emit_modrm_memory_disp32_rbp(assembler->buffer, &assembler->offset, BAL_X86_RAX);

    // WARNING: Bit pattern of int32_t offset is preserved.
    emit32(assembler->buffer, &assembler->offset, (uint32_t)offset);

    if (true == fit_in_8_bits)
    {
        // WARNING: Cast is safe because it is only reached if fit_in_bits is true.
        emit8(assembler->buffer, &assembler->offset, (uint8_t)immediate);
    }
    else
    {
        // WARNING: Bit pattern of int32_t immediate is preserved.
        emit32(assembler->buffer, &assembler->offset, (uint32_t)immediate);
    }

    const size_t bytes_emitted = assembler->offset - old_offset;
    BAL_ASSERT_MSG(bytes_emitted == instruction_size_bytes,
                   "Bytes emitted %d does not match instruction size %d",
                   bytes_emitted,
                   instruction_size_bytes);
}

void
bal_x86_emit_and_r64_r64(bal_x86_assembler_t     *assembler,
                         const bal_x86_register_t destination,
                         const bal_x86_register_t source)
{
    if (BAL_UNLIKELY(NULL == assembler))
    {
        return;
    }

    if (BAL_UNLIKELY(assembler->status != BAL_SUCCESS))
    {
        BAL_LOG_ERROR(&bal_thread_logger, "Aborting function: assembler status != BAL_SUCCESS");
        return;
    }

    bool is_valid_register_result = is_valid_register(destination);

    if (BAL_UNLIKELY(false == is_valid_register_result))
    {
        BAL_LOG_ERROR(&bal_thread_logger, "Invalid destination register: %d", destination);
        assembler->status = BAL_ERROR_INVALID_ARGUMENT;
        return;
    }

    is_valid_register_result = is_valid_register(source);

    if (BAL_UNLIKELY(false == is_valid_register_result))
    {
        BAL_LOG_ERROR(&bal_thread_logger, "Invalid source register: %d", destination);
        assembler->status = BAL_ERROR_INVALID_ARGUMENT;
        return;
    }

    const size_t instruction_size_bytes = 3;

    const bool can_emit_status = can_emit(assembler, instruction_size_bytes);

    if (BAL_UNLIKELY(false == can_emit_status))
    {
        return;
    }

    BAL_LOG_DEBUG(
        &bal_thread_logger, "[0x%04zx] and r%d, r%d", assembler->offset, destination, source);

    const uint8_t w = 1;

    // WARNING: Destination is verified by is_valid_register() to fall within the safe enum range
    // [0, 15].
    const uint8_t r = (uint8_t)destination >> 3;

    // WARNING: Source is verified by is_valid_register() to fall within the safe enum range
    // [0, 15].
    const uint8_t b = (uint8_t)source >> 3;

    const uint8_t opcode     = 0x23;
    const size_t  old_offset = assembler->offset;
    emit_rex(assembler->buffer, &assembler->offset, w, r, b);
    emit8(assembler->buffer, &assembler->offset, opcode);
    emit_modrm_register(assembler->buffer, &assembler->offset, destination, source);
    const size_t bytes_emitted = assembler->offset - old_offset;
    BAL_ASSERT_MSG(bytes_emitted == instruction_size_bytes,
                   "Bytes emitted %d does not match instruction size %d",
                   bytes_emitted,
                   instruction_size_bytes);
}

void
bal_x86_emit_and_r32_r32(bal_x86_assembler_t     *assembler,
                         const bal_x86_register_t destination,
                         const bal_x86_register_t source)
{
    if (BAL_UNLIKELY(NULL == assembler))
    {
        return;
    }

    if (BAL_UNLIKELY(assembler->status != BAL_SUCCESS))
    {
        BAL_LOG_ERROR(&bal_thread_logger, "Aborting function: assembler status != BAL_SUCCESS");
        return;
    }

    bool is_valid_register_result = is_valid_register(destination);

    if (BAL_UNLIKELY(false == is_valid_register_result))
    {
        BAL_LOG_ERROR(&bal_thread_logger, "Invalid destination register: %d", destination);
        assembler->status = BAL_ERROR_INVALID_ARGUMENT;
        return;
    }

    is_valid_register_result = is_valid_register(source);

    if (BAL_UNLIKELY(false == is_valid_register_result))
    {
        BAL_LOG_ERROR(&bal_thread_logger, "Invalid source register: %d", destination);
        assembler->status = BAL_ERROR_INVALID_ARGUMENT;
        return;
    }

    const size_t instruction_size_bytes = 3;

    const bool can_emit_status = can_emit(assembler, instruction_size_bytes);

    if (BAL_UNLIKELY(false == can_emit_status))
    {
        return;
    }

    BAL_LOG_DEBUG(
        &bal_thread_logger, "[0x%04zx] and r%dd, r%dd", assembler->offset, destination, source);

    const uint8_t w = 0;

    // WARNING: Destination is verified by is_valid_register() to fall within the safe enum range
    // [0, 15].
    const uint8_t r = (uint8_t)destination >> 3;

    // WARNING: Source is verified by is_valid_register() to fall within the safe enum range
    // [0, 15].
    const uint8_t b = (uint8_t)source >> 3;

    const uint8_t opcode     = 0x23;
    const size_t  old_offset = assembler->offset;
    emit_rex(assembler->buffer, &assembler->offset, w, r, b);
    emit8(assembler->buffer, &assembler->offset, opcode);
    emit_modrm_register(assembler->buffer, &assembler->offset, destination, source);
    const size_t bytes_emitted = assembler->offset - old_offset;
    BAL_ASSERT_MSG(bytes_emitted == instruction_size_bytes,
                   "Bytes emitted %d does not match instruction size %d",
                   bytes_emitted,
                   instruction_size_bytes);
}

void
bal_x86_emit_jmp_r64(bal_x86_assembler_t *assembler, const bal_x86_register_t reg)
{
    if (BAL_UNLIKELY(NULL == assembler))
    {
        return;
    }

    if (BAL_UNLIKELY(assembler->status != BAL_SUCCESS))
    {
        BAL_LOG_ERROR(&bal_thread_logger, "Aborting function: assembler status != BAL_SUCCESS");
        return;
    }

    const bool is_valid_register_result = is_valid_register(reg);

    if (BAL_UNLIKELY(false == is_valid_register_result))
    {
        BAL_LOG_ERROR(&bal_thread_logger, "Invalid register: %d", reg);
        assembler->status = BAL_ERROR_INVALID_ARGUMENT;
        return;
    }

    // Registers R8 to R15 need a REX prefix in front of the two-byte instruction.
    const size_t instruction_size_bytes = reg > 7 ? 3U : 2U;
    const bool   can_emit_status        = can_emit(assembler, instruction_size_bytes);

    if (BAL_UNLIKELY(false == can_emit_status))
    {
        return;
    }

    BAL_LOG_DEBUG(&bal_thread_logger, "[+0x%04zx] jmp r%d", assembler->offset, reg);
    const size_t old_offset = assembler->offset;

    if (reg > 7)
    {
        const uint8_t w = 0;
        const uint8_t r = 0;

        // WARNING: reg is verified by is_valid_register() to fall within the safe enum range.
        const uint8_t b = (uint8_t)reg >> 3;
        emit_rex(assembler->buffer, &assembler->offset, w, r, b);
    }

    const uint8_t opcode = 0xFF;
    emit8(assembler->buffer, &assembler->offset, opcode);

    const uint8_t safe_reg = (uint8_t)reg & 7;
    const uint8_t modrm    = (uint8_t)(0xC0U | 4U << 3U | safe_reg);
    emit8(assembler->buffer, &assembler->offset, modrm);
    const size_t bytes_emitted = assembler->offset - old_offset;
    BAL_ASSERT_MSG(bytes_emitted == instruction_size_bytes,
                   "Bytes emitted %d does not match instruction size %d",
                   bytes_emitted,
                   instruction_size_bytes);
}

void
bal_x86_emit_jmp_rel32(bal_x86_assembler_t *assembler, const int32_t offset)
{
    if (BAL_UNLIKELY(NULL == assembler))
    {
        return;
    }

    if (BAL_UNLIKELY(assembler->status != BAL_SUCCESS))
    {
        BAL_LOG_ERROR(&bal_thread_logger, "Aborting function: assembler status != BAL_SUCCESS");
        return;
    }

    const size_t instruction_size_bytes = 5;
    const bool   can_emit_status        = can_emit(assembler, instruction_size_bytes);

    if (BAL_UNLIKELY(false == can_emit_status))
    {
        return;
    }

    BAL_LOG_DEBUG(&bal_thread_logger, "[+0x%04zx] jmp 0x%X", assembler->offset, offset);
    const size_t  old_offset = assembler->offset;
    const uint8_t opcode     = 0xE9;
    emit8(assembler->buffer, &assembler->offset, opcode);

    // WARNING: Bit pattern of int32_t offset is preserved in uint32_t.
    emit32(assembler->buffer, &assembler->offset, (uint32_t)offset);

    const size_t bytes_emitted = assembler->offset - old_offset;
    BAL_ASSERT_MSG(bytes_emitted == instruction_size_bytes,
                   "Bytes emitted %d does not match instruction size %d",
                   bytes_emitted,
                   instruction_size_bytes);
}

void
bal_x86_emit_call_rel32(bal_x86_assembler_t *assembler, const int32_t offset)
{
    if (BAL_UNLIKELY(NULL == assembler))
    {
        return;
    }

    if (BAL_UNLIKELY(assembler->status != BAL_SUCCESS))
    {
        BAL_LOG_ERROR(&bal_thread_logger, "Aborting function: assembler status != BAL_SUCCESS");
        return;
    }

    const size_t instruction_size_bytes = 5;

    if (BAL_UNLIKELY(false == can_emit(assembler, instruction_size_bytes)))
    {
        return;
    }

    BAL_LOG_DEBUG(&bal_thread_logger, "[+0x%04zx] call 0x%X", assembler->offset, offset);
    emit8(assembler->buffer, &assembler->offset, 0xE8);

    // WARNING: Bit pattern of int32_t offset is preserved in uint32_t.
    emit32(assembler->buffer, &assembler->offset, (uint32_t)offset);
}

void
bal_x86_emit_load_r64_rbp_offset(bal_x86_assembler_t     *assembler,
                                 const bal_x86_register_t destination,
                                 const int32_t            offset)
{
    if (BAL_UNLIKELY(NULL == assembler))
    {
        return;
    }

    if (BAL_UNLIKELY(assembler->status != BAL_SUCCESS))
    {
        BAL_LOG_ERROR(&bal_thread_logger, "Aborting function: assembler status != BAL_SUCCESS");
        return;
    }

    const bool is_valid_register_result = is_valid_register(destination);

    if (BAL_UNLIKELY(false == is_valid_register_result))
    {
        BAL_LOG_ERROR(&bal_thread_logger, "Invalid destination register: %d", destination);
        assembler->status = BAL_ERROR_INVALID_ARGUMENT;
        return;
    }

    const size_t instruction_size_bytes = 7;
    const bool   can_emit_status        = can_emit(assembler, instruction_size_bytes);

    if (BAL_UNLIKELY(false == can_emit_status))
    {
        return;
    }

    BAL_LOG_DEBUG(&bal_thread_logger,
                  "[0x%04zx] mov r%d, [rbp + 0x%X]",
                  assembler->offset,
                  destination,
                  offset);

    const uint8_t w = 1;

    // WARNING: Destination is verified by is_valid_register() to fall within the safe enum range
    // [0, 15].
    const uint8_t r = (uint8_t)destination >> 3;

    const uint8_t b          = 0;
    const uint8_t opcode     = 0X8B;
    const size_t  old_offset = assembler->offset;
    emit_rex(assembler->buffer, &assembler->offset, w, r, b);
    emit8(assembler->buffer, &assembler->offset, opcode);
    emit_modrm_memory_disp32_rbp(assembler->buffer, &assembler->offset, destination);

    // WARNING: Bit pattern of int32_t offset is preserved in uint32_t.
    emit32(assembler->buffer, &assembler->offset, (uint32_t)offset);

    const size_t bytes_emitted = assembler->offset - old_offset;
    BAL_ASSERT_MSG(bytes_emitted == instruction_size_bytes,
                   "Bytes emitted %d does not match instruction size %d",
                   bytes_emitted,
                   instruction_size_bytes);
}

void
bal_x86_emit_load_r64_from_r64(bal_x86_assembler_t     *assembler,
                               const bal_x86_register_t destination,
                               const bal_x86_register_t base)
{
    if (BAL_UNLIKELY(NULL == assembler))
    {
        BAL_LOG_ERROR(&bal_thread_logger, "Aborting function: assembler is NULL.");
        return;
    }

    if (BAL_UNLIKELY(assembler->status != BAL_SUCCESS))
    {
        BAL_LOG_ERROR(&bal_thread_logger, "Aborting function: assembler status != BAL_SUCCESS");
        return;
    }

    bool is_valid_register_result = is_valid_register(destination);

    if (BAL_UNLIKELY(false == is_valid_register_result))
    {
        BAL_LOG_ERROR(
            &bal_thread_logger, "Aborting function: invalid destination register: %d", destination);
        assembler->status = BAL_ERROR_INVALID_ARGUMENT;
        return;
    }

    is_valid_register_result = is_valid_register(base);

    if (BAL_UNLIKELY(false == is_valid_register_result))
    {
        BAL_LOG_ERROR(&bal_thread_logger, "Aborting function: invalid base register: %d", base);
        assembler->status = BAL_ERROR_INVALID_ARGUMENT;
        return;
    }

    // WARNING: Verified by is_valid_register() to fall within [0, 15].
    const uint8_t base_low = (uint8_t)base & 7U;

    size_t instruction_size_bytes = 3;
    bool   needs_sib              = false;
    bool   needs_disp8            = false;

    if (4U == base_low)
    {
        needs_sib              = true;
        instruction_size_bytes = 4;
    }
    else if (5U == base_low)
    {
        needs_disp8            = true;
        instruction_size_bytes = 4;
    }

    const bool can_emit_result = can_emit(assembler, instruction_size_bytes);

    if (BAL_UNLIKELY(false == can_emit_result))
    {
        return;
    }

    BAL_LOG_DEBUG(
        &bal_thread_logger, "[0x%04zx] mov r%d, [r%d]", assembler->offset, destination, base);

    const uint8_t w          = 1;
    const uint8_t r          = (uint8_t)destination >> 3;
    const uint8_t b          = (uint8_t)base >> 3;
    const uint8_t opcode     = 0x8BU;
    const size_t  old_offset = assembler->offset;

    emit_rex(assembler->buffer, &assembler->offset, w, r, b);
    emit8(assembler->buffer, &assembler->offset, opcode);
    const uint8_t safe_destination = (uint8_t)destination & 7U;

    if (true == needs_sib)
    {
        const uint8_t modrm = (uint8_t)((safe_destination << 3U) | 4);
        emit8(assembler->buffer, &assembler->offset, modrm);
        const uint8_t sib = 0x24U;
        emit8(assembler->buffer, &assembler->offset, sib);
    }
    else if (true == needs_disp8)
    {
        const uint8_t modrm = (uint8_t)(0x40U | (uint8_t)(safe_destination << 3U) | 5U);
        emit8(assembler->buffer, &assembler->offset, modrm);

        const uint8_t disp8 = 0x00U;
        emit8(assembler->buffer, &assembler->offset, disp8);
    }
    else
    {
        const uint8_t modrm = (uint8_t)((safe_destination << 3U) | base_low);
        emit8(assembler->buffer, &assembler->offset, modrm);
    }

    const size_t bytes_emitted = assembler->offset - old_offset;
    BAL_ASSERT_MSG(bytes_emitted == instruction_size_bytes,
                   "Bytes emitted %zu does not match instruction size %zu",
                   bytes_emitted,
                   instruction_size_bytes);
}

void
bal_x86_emit_store_r64_rbp_offset(bal_x86_assembler_t     *assembler,
                                  const bal_x86_register_t source,
                                  const int32_t            offset)
{
    if (BAL_UNLIKELY(NULL == assembler))
    {
        return;
    }

    if (BAL_UNLIKELY(assembler->status != BAL_SUCCESS))
    {
        BAL_LOG_ERROR(&bal_thread_logger, "Aborting function: assembler status != BAL_SUCCESS");
        return;
    }

    const bool is_valid_register_result = is_valid_register(source);

    if (BAL_UNLIKELY(false == is_valid_register_result))
    {
        BAL_LOG_ERROR(&bal_thread_logger, "Invalid source register: %d", source);
        assembler->status = BAL_ERROR_INVALID_ARGUMENT;
        return;
    }

    const size_t instruction_size_bytes = 7;
    const bool   can_emit_status        = can_emit(assembler, instruction_size_bytes);

    if (BAL_UNLIKELY(false == can_emit_status))
    {
        return;
    }

    BAL_LOG_DEBUG(
        &bal_thread_logger, "[0x%04zx] mov[rbp + 0x%X], r%d", assembler->offset, offset, source);
    const uint8_t w = 1;

    // WARNING: Source is verified by is_valid_register() to fall within the safe enum range
    // [0, 15].
    const uint8_t r = (uint8_t)source >> 3;

    const uint8_t b          = 0;
    const uint8_t opcode     = 0X89;
    const size_t  old_offset = assembler->offset;
    emit_rex(assembler->buffer, &assembler->offset, w, r, b);
    emit8(assembler->buffer, &assembler->offset, opcode);
    emit_modrm_memory_disp32_rbp(assembler->buffer, &assembler->offset, source);

    // WARNING: Bit pattern of int32_t offset is preserved in uint32_t.
    emit32(assembler->buffer, &assembler->offset, (uint32_t)offset);

    const size_t bytes_emitted = assembler->offset - old_offset;
    BAL_ASSERT_MSG(bytes_emitted == instruction_size_bytes,
                   "Bytes emitted %d does not match instruction size %d",
                   bytes_emitted,
                   instruction_size_bytes);
}

void
bal_x86_emit_mov_r64_r64(bal_x86_assembler_t     *assembler,
                         const bal_x86_register_t destination,
                         const bal_x86_register_t source)
{
    if (BAL_UNLIKELY(NULL == assembler))
    {
        return;
    }

    if (BAL_UNLIKELY(assembler->status != BAL_SUCCESS))
    {
        BAL_LOG_ERROR(&bal_thread_logger, "Assembler status != BAL_SUCCESS, aborting function");
        assembler->status = BAL_ERROR_INVALID_ARGUMENT;
        return;
    }

    bool is_valid_register_result = is_valid_register(source);

    if (BAL_UNLIKELY(false == is_valid_register_result))
    {
        BAL_LOG_ERROR(&bal_thread_logger, "Invalid source register: %d", source);
        assembler->status = BAL_ERROR_INVALID_ARGUMENT;
        return;
    }

    is_valid_register_result = is_valid_register(destination);

    if (BAL_UNLIKELY(false == is_valid_register_result))
    {
        BAL_LOG_ERROR(&bal_thread_logger, "Invalid destination register: %d", destination);
        assembler->status = BAL_ERROR_INVALID_ARGUMENT;
        return;
    }

    const size_t instruction_size_bytes = 3;
    const bool   can_emit_status        = can_emit(assembler, instruction_size_bytes);

    if (BAL_UNLIKELY(false == can_emit_status))
    {
        return;
    }

    BAL_LOG_DEBUG(
        &bal_thread_logger, "[+0x%04zx] mov r%d, r%d", assembler->offset, destination, source);
    const uint8_t w = 1;

    // WARNING: Destination is verified by is_valid_register() to fall within the safe enum range
    // [0, 15].
    const uint8_t r = (uint8_t)destination >> 3;

    // WARNING: Source is verified by is_valid_register() to fall within the safe enum range
    // [0, 15].
    const uint8_t b = (uint8_t)source >> 3;

    const uint8_t opcode     = 0x8BU;
    const size_t  old_offset = assembler->offset;
    emit_rex(assembler->buffer, &assembler->offset, w, r, b);
    emit8(assembler->buffer, &assembler->offset, opcode);
    emit_modrm_register(assembler->buffer, &assembler->offset, destination, source);
    const size_t bytes_emitted = assembler->offset - old_offset;
    BAL_ASSERT_MSG(bytes_emitted == instruction_size_bytes,
                   "Bytes emitted %d does not match instruction size %d",
                   bytes_emitted,
                   instruction_size_bytes);
}

void
bal_x86_emit_mov_r64_imm64(bal_x86_assembler_t     *assembler,
                           const bal_x86_register_t destination,
                           const uint64_t           immediate)
{
    if (BAL_UNLIKELY(NULL == assembler))
    {
        return;
    }

    if (BAL_UNLIKELY(assembler->status != BAL_SUCCESS))
    {
        BAL_LOG_ERROR(&bal_thread_logger, "Assembler status != BAL_SUCCESS, aborting function");
        assembler->status = BAL_ERROR_INVALID_ARGUMENT;
        return;
    }

    const bool is_valid_register_result = is_valid_register(destination);

    if (BAL_UNLIKELY(false == is_valid_register_result))
    {
        BAL_LOG_ERROR(&bal_thread_logger, "Invalid destination register: %d", destination);
        assembler->status = BAL_ERROR_INVALID_ARGUMENT;
        return;
    }

    const bool fits_in_32_bits = immediate <= 0xFFFFFFFFULL;

    // WARNING: Destination is verified by is_valid_register() to fall within the safe enum range
    // [0, 15].
    const uint8_t b = (uint8_t)destination >> 3;
    size_t        instruction_size_bytes;

    if (true == fits_in_32_bits)
    {
        // 1 byte opcode + 4 bytes imm32 = 5 bytes. +1 byte REX if destination is R8-R15.
        instruction_size_bytes = b > 0 ? 6 : 5;
    }
    else
    {
        // 1 byte REX + 1 byte opcode + 8 bytes imm64 = 10 bytes.
        instruction_size_bytes = 10;
    }

    const bool can_emit_status = can_emit(assembler, instruction_size_bytes);

    if (BAL_UNLIKELY(false == can_emit_status))
    {
        return;
    }

    BAL_LOG_DEBUG(&bal_thread_logger,
                  "[+0x%04zx] mov r%d, 0x%llX",
                  assembler->offset,
                  destination,
                  (unsigned long long)immediate);
    const size_t old_offset = assembler->offset;

    if (true == fits_in_32_bits)
    {
        if (b > 0)
        {
            // We only need REX.B if the register is R8-R15. No REX.W needed.
            //
            const uint8_t w = 0;
            const uint8_t r = 0;
            emit_rex(assembler->buffer, &assembler->offset, w, r, b);
        }

        emit8(assembler->buffer, &assembler->offset, 0xB8 + (destination & 7));

        // WARNING: Cast from uint64_t to uint32_t is safe because it is only reached if
        // `fits_in_32_bits` is true.
        emit32(assembler->buffer, &assembler->offset, (uint32_t)immediate);
    }
    else
    {
        const uint8_t w = 1;
        const uint8_t r = 0;
        emit_rex(assembler->buffer, &assembler->offset, w, r, b);
        emit8(assembler->buffer, &assembler->offset, 0xB8 + (destination & 7));
        emit64(assembler->buffer, &assembler->offset, immediate);
    }
    const size_t bytes_emitted = assembler->offset - old_offset;
    BAL_ASSERT_MSG(bytes_emitted == instruction_size_bytes,
                   "Bytes emitted %d does not match instruction size %d",
                   bytes_emitted,
                   instruction_size_bytes);
}

void
bal_x86_emit_ret(bal_x86_assembler_t *assembler)
{
    if (BAL_UNLIKELY(NULL == assembler))
    {
        return;
    }

    if (BAL_UNLIKELY(assembler->status != BAL_SUCCESS))
    {
        BAL_LOG_ERROR(&bal_thread_logger, "Assembler status != BAL_SUCCESS, aborting function");
        assembler->status = BAL_ERROR_INVALID_ARGUMENT;
        return;
    }

    const size_t instruction_size_bytes = 1;
    const bool   can_emit_status        = can_emit(assembler, instruction_size_bytes);

    if (BAL_UNLIKELY(false == can_emit_status))
    {
        return;
    }

    BAL_LOG_DEBUG(&bal_thread_logger, "[+0x%04zx] ret", assembler->offset);
    emit8(assembler->buffer, &assembler->offset, 0XC3);
}

void
bal_x86_emit_or_r64_r64(bal_x86_assembler_t     *assembler,
                        const bal_x86_register_t destination,
                        const bal_x86_register_t source)
{
    if (BAL_UNLIKELY(NULL == assembler))
    {
        return;
    }

    if (BAL_UNLIKELY(assembler->status != BAL_SUCCESS))
    {
        BAL_LOG_ERROR(&bal_thread_logger, "Assembler status != BAL_SUCCESS, aborting function");
        assembler->status = BAL_ERROR_INVALID_ARGUMENT;
        return;
    }

    bool is_valid_register_result = is_valid_register(source);

    if (BAL_UNLIKELY(false == is_valid_register_result))
    {
        BAL_LOG_ERROR(&bal_thread_logger, "Invalid source register: %d", source);
        assembler->status = BAL_ERROR_INVALID_ARGUMENT;
        return;
    }

    is_valid_register_result = is_valid_register(destination);

    if (BAL_UNLIKELY(false == is_valid_register_result))
    {
        BAL_LOG_ERROR(&bal_thread_logger, "Invalid destination register: %d", destination);
        assembler->status = BAL_ERROR_INVALID_ARGUMENT;
        return;
    }

    const size_t instruction_size_bytes = 3;
    const bool   can_emit_status        = can_emit(assembler, instruction_size_bytes);

    if (BAL_UNLIKELY(false == can_emit_status))
    {
        return;
    }

    BAL_LOG_DEBUG(
        &bal_thread_logger, "[+0x%04zx] or r%d, r%d", assembler->offset, destination, source);
    const uint8_t w = 1;

    // WARNING: Destination is verified by is_valid_register() to fall within the safe enum range
    // [0, 15].
    const uint8_t r = (uint8_t)destination >> 3;

    // WARNING: Source is verified by is_valid_register() to fall within the safe enum range
    // [0, 15].
    const uint8_t b = (uint8_t)source >> 3;

    const uint8_t opcode     = 0x0B;
    const size_t  old_offset = assembler->offset;
    emit_rex(assembler->buffer, &assembler->offset, w, r, b);
    emit8(assembler->buffer, &assembler->offset, opcode);
    emit_modrm_register(assembler->buffer, &assembler->offset, destination, source);
    const size_t bytes_emitted = assembler->offset - old_offset;
    BAL_ASSERT_MSG(bytes_emitted == instruction_size_bytes,
                   "Bytes emitted %d does not match instruction size %d",
                   bytes_emitted,
                   instruction_size_bytes);
}

void
bal_x86_emit_ror_r64_imm8(bal_x86_assembler_t     *assembler,
                          const bal_x86_register_t destination,
                          const uint8_t            immediate)
{
    const uint8_t opcode_extension = 1;
    const bool    is_64bit         = true;
    emit_shift_imm8(assembler, destination, immediate, opcode_extension, is_64bit, "ror");
}

void
bal_x86_emit_sar_r64_imm8(bal_x86_assembler_t     *assembler,
                          const bal_x86_register_t destination,
                          const uint8_t            immediate)
{
    const uint8_t opcode_extension = 7;
    const bool    is_64bit         = true;
    emit_shift_imm8(assembler, destination, immediate, opcode_extension, is_64bit, "sar");
}

void
bal_x86_emit_shl_r64_imm8(bal_x86_assembler_t     *assembler,
                          const bal_x86_register_t destination,
                          const uint8_t            immediate)
{
    const uint8_t opcode_extension = 4;
    const bool    is_64bit         = true;
    emit_shift_imm8(assembler, destination, immediate, opcode_extension, is_64bit, "shl");
}

void
bal_x86_emit_shr_r64_imm8(bal_x86_assembler_t     *assembler,
                          const bal_x86_register_t destination,
                          const uint8_t            immediate)
{
    const uint8_t opcode_extension = 5;
    const bool    is_64bit         = true;
    emit_shift_imm8(assembler, destination, immediate, opcode_extension, is_64bit, "shr");
}

void
bal_x86_emit_ror_r32_imm8(bal_x86_assembler_t     *assembler,
                          const bal_x86_register_t destination,
                          const uint8_t            immediate)
{
    const uint8_t opcode_extension = 1;
    const bool    is_64bit         = false;
    emit_shift_imm8(assembler, destination, immediate, opcode_extension, is_64bit, "ror");
}

void
bal_x86_emit_sar_r32_imm8(bal_x86_assembler_t     *assembler,
                          const bal_x86_register_t destination,
                          const uint8_t            immediate)
{
    const uint8_t opcode_extension = 7;
    const bool    is_64bit         = false;
    emit_shift_imm8(assembler, destination, immediate, opcode_extension, is_64bit, "sar");
}

void
bal_x86_emit_shl_r32_imm8(bal_x86_assembler_t     *assembler,
                          const bal_x86_register_t destination,
                          const uint8_t            immediate)
{
    const uint8_t opcode_extension = 4;
    const bool    is_64bit         = false;
    emit_shift_imm8(assembler, destination, immediate, opcode_extension, is_64bit, "shl");
}

void
bal_x86_emit_shr_r32_imm8(bal_x86_assembler_t     *assembler,
                          const bal_x86_register_t destination,
                          const uint8_t            immediate)
{
    const uint8_t opcode_extension = 5;
    const bool    is_64bit         = false;
    emit_shift_imm8(assembler, destination, immediate, opcode_extension, is_64bit, "shr");
}

void
bal_x86_emit_or_r32_r32(bal_x86_assembler_t     *assembler,
                        const bal_x86_register_t destination,
                        const bal_x86_register_t source)
{
    const uint8_t opcode   = 0x0B;
    const bool    is_64bit = false;
    emit_register_register(assembler, opcode, is_64bit, destination, source, "or");
}

void
bal_x86_emit_xor_r32_r32(bal_x86_assembler_t     *assembler,
                         const bal_x86_register_t destination,
                         const bal_x86_register_t source)
{
    const uint8_t opcode   = 0x33;
    const bool    is_64bit = false;
    emit_register_register(assembler, opcode, is_64bit, destination, source, "xor");
}

void
bal_x86_emit_test_r32_r32(bal_x86_assembler_t     *assembler,
                          const bal_x86_register_t destination,
                          const bal_x86_register_t source)
{
    const uint8_t opcode   = 0x85;
    const bool    is_64bit = false;
    emit_register_register(assembler, opcode, is_64bit, destination, source, "test");
}

void
bal_x86_emit_not_r64(bal_x86_assembler_t *assembler, const bal_x86_register_t destination)
{
    const uint8_t opcode_extension = 2;
    const bool    is_64bit         = true;
    emit_group3_register(assembler, opcode_extension, is_64bit, destination, "not");
}

void
bal_x86_emit_not_r32(bal_x86_assembler_t *assembler, const bal_x86_register_t destination)
{
    const uint8_t opcode_extension = 2;
    const bool    is_64bit         = false;
    emit_group3_register(assembler, opcode_extension, is_64bit, destination, "not");
}

void
bal_x86_emit_neg_r64(bal_x86_assembler_t *assembler, const bal_x86_register_t destination)
{
    const uint8_t opcode_extension = 3;
    const bool    is_64bit         = true;
    emit_group3_register(assembler, opcode_extension, is_64bit, destination, "neg");
}

void
bal_x86_emit_neg_r32(bal_x86_assembler_t *assembler, const bal_x86_register_t destination)
{
    const uint8_t opcode_extension = 3;
    const bool    is_64bit         = false;
    emit_group3_register(assembler, opcode_extension, is_64bit, destination, "neg");
}

void
bal_x86_emit_div_r64(bal_x86_assembler_t *assembler, const bal_x86_register_t source)
{
    const uint8_t opcode_extension = 6;
    const bool    is_64bit         = true;
    emit_group3_register(assembler, opcode_extension, is_64bit, source, "div");
}

void
bal_x86_emit_div_r32(bal_x86_assembler_t *assembler, const bal_x86_register_t source)
{
    const uint8_t opcode_extension = 6;
    const bool    is_64bit         = false;
    emit_group3_register(assembler, opcode_extension, is_64bit, source, "div");
}

void
bal_x86_emit_idiv_r64(bal_x86_assembler_t *assembler, const bal_x86_register_t source)
{
    const uint8_t opcode_extension = 7;
    const bool    is_64bit         = true;
    emit_group3_register(assembler, opcode_extension, is_64bit, source, "idiv");
}

void
bal_x86_emit_idiv_r32(bal_x86_assembler_t *assembler, const bal_x86_register_t source)
{
    const uint8_t opcode_extension = 7;
    const bool    is_64bit         = false;
    emit_group3_register(assembler, opcode_extension, is_64bit, source, "idiv");
}

void
bal_x86_emit_cqo(bal_x86_assembler_t *assembler)
{
    if (BAL_UNLIKELY(NULL == assembler))
    {
        return;
    }

    if (BAL_UNLIKELY(assembler->status != BAL_SUCCESS))
    {
        BAL_LOG_ERROR(&bal_thread_logger, "Aborting function: assembler status != BAL_SUCCESS");
        return;
    }

    const size_t instruction_size_bytes = 2;
    const bool   can_emit_status        = can_emit(assembler, instruction_size_bytes);

    if (BAL_UNLIKELY(false == can_emit_status))
    {
        return;
    }

    BAL_LOG_DEBUG(&bal_thread_logger, "[+0x%04zx] cqo", assembler->offset);
    const uint8_t w = 1U;
    const uint8_t r = 0U;
    const uint8_t b = 0U;
    emit_rex(assembler->buffer, &assembler->offset, w, r, b);
    emit8(assembler->buffer, &assembler->offset, 0x99);
}

void
bal_x86_emit_cdq(bal_x86_assembler_t *assembler)
{
    if (BAL_UNLIKELY(NULL == assembler))
    {
        return;
    }

    if (BAL_UNLIKELY(assembler->status != BAL_SUCCESS))
    {
        BAL_LOG_ERROR(&bal_thread_logger, "Aborting function: assembler status != BAL_SUCCESS");
        return;
    }

    const size_t instruction_size_bytes = 1;
    const bool   can_emit_status        = can_emit(assembler, instruction_size_bytes);

    if (BAL_UNLIKELY(false == can_emit_status))
    {
        return;
    }

    BAL_LOG_DEBUG(&bal_thread_logger, "[+0x%04zx] cdq", assembler->offset);
    emit8(assembler->buffer, &assembler->offset, 0x99);
}

void
bal_x86_emit_bsr_r64_r64(bal_x86_assembler_t     *assembler,
                         const bal_x86_register_t destination,
                         const bal_x86_register_t source)
{
    const uint8_t opcode   = 0xBD;
    const bool    is_64bit = true;
    emit_register_register_0f(assembler, opcode, is_64bit, destination, source, "bsr");
}

void
bal_x86_emit_bsr_r32_r32(bal_x86_assembler_t     *assembler,
                         const bal_x86_register_t destination,
                         const bal_x86_register_t source)
{
    const uint8_t opcode   = 0xBD;
    const bool    is_64bit = false;
    emit_register_register_0f(assembler, opcode, is_64bit, destination, source, "bsr");
}

void
bal_x86_emit_bswap_r64(bal_x86_assembler_t *assembler, const bal_x86_register_t destination)
{
    if (BAL_UNLIKELY(NULL == assembler))
    {
        return;
    }

    if (BAL_UNLIKELY(assembler->status != BAL_SUCCESS))
    {
        BAL_LOG_ERROR(&bal_thread_logger, "Aborting function: assembler status != BAL_SUCCESS");
        return;
    }

    if (BAL_UNLIKELY(false == is_valid_register(destination)))
    {
        BAL_LOG_ERROR(&bal_thread_logger, "Invalid destination register: %d", destination);
        assembler->status = BAL_ERROR_INVALID_ARGUMENT;
        return;
    }

    const size_t instruction_size_bytes = 3;
    const bool   can_emit_status        = can_emit(assembler, instruction_size_bytes);

    if (BAL_UNLIKELY(false == can_emit_status))
    {
        return;
    }

    BAL_LOG_DEBUG(&bal_thread_logger, "[+0x%04zx] bswap r%d", assembler->offset, destination);

    const uint8_t w          = 1U;
    const uint8_t r          = 0U;
    const uint8_t b          = (uint8_t)destination >> 3U;
    const size_t  old_offset = assembler->offset;
    emit_rex(assembler->buffer, &assembler->offset, w, r, b);
    emit8(assembler->buffer, &assembler->offset, 0x0F);
    emit8(assembler->buffer, &assembler->offset, (uint8_t)(0xC8U + ((uint32_t)destination & 7U)));

    const size_t bytes_emitted = assembler->offset - old_offset;
    BAL_ASSERT_MSG(bytes_emitted == instruction_size_bytes,
                   "Bytes emitted %d does not match instruction size %d",
                   bytes_emitted,
                   instruction_size_bytes);
}

void
bal_x86_emit_bswap_r32(bal_x86_assembler_t *assembler, const bal_x86_register_t destination)
{
    if (BAL_UNLIKELY(NULL == assembler))
    {
        return;
    }

    if (BAL_UNLIKELY(assembler->status != BAL_SUCCESS))
    {
        BAL_LOG_ERROR(&bal_thread_logger, "Aborting function: assembler status != BAL_SUCCESS");
        return;
    }

    if (BAL_UNLIKELY(false == is_valid_register(destination)))
    {
        BAL_LOG_ERROR(&bal_thread_logger, "Invalid destination register: %d", destination);
        assembler->status = BAL_ERROR_INVALID_ARGUMENT;
        return;
    }

    const size_t instruction_size_bytes = 3;
    const bool   can_emit_status        = can_emit(assembler, instruction_size_bytes);

    if (BAL_UNLIKELY(false == can_emit_status))
    {
        return;
    }

    BAL_LOG_DEBUG(&bal_thread_logger, "[+0x%04zx] bswap r%dd", assembler->offset, destination);

    const uint8_t w          = 0U;
    const uint8_t r          = 0U;
    const uint8_t b          = (uint8_t)destination >> 3U;
    const size_t  old_offset = assembler->offset;
    emit_rex(assembler->buffer, &assembler->offset, w, r, b);
    emit8(assembler->buffer, &assembler->offset, 0x0F);
    emit8(assembler->buffer, &assembler->offset, (uint8_t)(0xC8U + ((uint32_t)destination & 7U)));

    const size_t bytes_emitted = assembler->offset - old_offset;
    BAL_ASSERT_MSG(bytes_emitted == instruction_size_bytes,
                   "Bytes emitted %d does not match instruction size %d",
                   bytes_emitted,
                   instruction_size_bytes);
}

void
bal_x86_emit_mul_r64(bal_x86_assembler_t *assembler, const bal_x86_register_t source)
{
    const uint8_t opcode_extension = 4;
    const bool    is_64bit         = true;
    emit_group3_register(assembler, opcode_extension, is_64bit, source, "mul");
}

void
bal_x86_emit_imul_r64(bal_x86_assembler_t *assembler, const bal_x86_register_t source)
{
    const uint8_t opcode_extension = 5;
    const bool    is_64bit         = true;
    emit_group3_register(assembler, opcode_extension, is_64bit, source, "imul");
}

void
bal_x86_emit_mfence(bal_x86_assembler_t *assembler)
{
    if (BAL_UNLIKELY(NULL == assembler))
    {
        return;
    }

    if (BAL_UNLIKELY(assembler->status != BAL_SUCCESS))
    {
        BAL_LOG_ERROR(&bal_thread_logger, "Aborting function: assembler status != BAL_SUCCESS");
        return;
    }

    const size_t instruction_size_bytes = 3;
    const bool   can_emit_status        = can_emit(assembler, instruction_size_bytes);

    if (BAL_UNLIKELY(false == can_emit_status))
    {
        return;
    }

    BAL_LOG_DEBUG(&bal_thread_logger, "[+0x%04zx] mfence", assembler->offset);
    emit8(assembler->buffer, &assembler->offset, 0x0F);
    emit8(assembler->buffer, &assembler->offset, 0xAE);
    emit8(assembler->buffer, &assembler->offset, 0xF0);
}

void
bal_x86_emit_load_r64_mem64(bal_x86_assembler_t     *assembler,
                            const bal_x86_register_t destination,
                            const bal_x86_register_t base)
{
    const uint8_t opcode[2] = { 0x8B, 0x00 };
    emit_memory_access(assembler, 0, 1, opcode, 1, destination, base, "load_r64_mem64");
}

void
bal_x86_emit_load_r32_mem32(bal_x86_assembler_t     *assembler,
                            const bal_x86_register_t destination,
                            const bal_x86_register_t base)
{
    const uint8_t opcode[2] = { 0x8B, 0x00 };
    emit_memory_access(assembler, 0, 0, opcode, 1, destination, base, "load_r32_mem32");
}

void
bal_x86_emit_load_r32_mem16(bal_x86_assembler_t     *assembler,
                            const bal_x86_register_t destination,
                            const bal_x86_register_t base)
{
    const uint8_t opcode[2] = { 0x0F, 0xB7 };
    emit_memory_access(assembler, 0, 0, opcode, 2, destination, base, "load_r32_mem16");
}

void
bal_x86_emit_load_r32_mem8(bal_x86_assembler_t     *assembler,
                           const bal_x86_register_t destination,
                           const bal_x86_register_t base)
{
    const uint8_t opcode[2] = { 0x0F, 0xB6 };
    emit_memory_access(assembler, 0, 0, opcode, 2, destination, base, "load_r32_mem8");
}

void
bal_x86_emit_load_sign_r64_mem32(bal_x86_assembler_t     *assembler,
                                 const bal_x86_register_t destination,
                                 const bal_x86_register_t base)
{
    const uint8_t opcode[2] = { 0x63, 0x00 };
    emit_memory_access(assembler, 0, 1, opcode, 1, destination, base, "load_sign_r64_mem32");
}

void
bal_x86_emit_load_sign_r64_mem16(bal_x86_assembler_t     *assembler,
                                 const bal_x86_register_t destination,
                                 const bal_x86_register_t base)
{
    const uint8_t opcode[2] = { 0x0F, 0xBF };
    emit_memory_access(assembler, 0, 1, opcode, 2, destination, base, "load_sign_r64_mem16");
}

void
bal_x86_emit_load_sign_r64_mem8(bal_x86_assembler_t     *assembler,
                                const bal_x86_register_t destination,
                                const bal_x86_register_t base)
{
    const uint8_t opcode[2] = { 0x0F, 0xBE };
    emit_memory_access(assembler, 0, 1, opcode, 2, destination, base, "load_sign_r64_mem8");
}

void
bal_x86_emit_load_sign_r32_mem16(bal_x86_assembler_t     *assembler,
                                 const bal_x86_register_t destination,
                                 const bal_x86_register_t base)
{
    const uint8_t opcode[2] = { 0x0F, 0xBF };
    emit_memory_access(assembler, 0, 0, opcode, 2, destination, base, "load_sign_r32_mem16");
}

void
bal_x86_emit_load_sign_r32_mem8(bal_x86_assembler_t     *assembler,
                                const bal_x86_register_t destination,
                                const bal_x86_register_t base)
{
    const uint8_t opcode[2] = { 0x0F, 0xBE };
    emit_memory_access(assembler, 0, 0, opcode, 2, destination, base, "load_sign_r32_mem8");
}

void
bal_x86_emit_store_mem64_r64(bal_x86_assembler_t     *assembler,
                             const bal_x86_register_t base,
                             const bal_x86_register_t source)
{
    const uint8_t opcode[2] = { 0x89, 0x00 };
    emit_memory_access(assembler, 0, 1, opcode, 1, source, base, "store_mem64_r64");
}

void
bal_x86_emit_store_mem32_r32(bal_x86_assembler_t     *assembler,
                             const bal_x86_register_t base,
                             const bal_x86_register_t source)
{
    const uint8_t opcode[2] = { 0x89, 0x00 };
    emit_memory_access(assembler, 0, 0, opcode, 1, source, base, "store_mem32_r32");
}

void
bal_x86_emit_store_mem16_r16(bal_x86_assembler_t     *assembler,
                             const bal_x86_register_t base,
                             const bal_x86_register_t source)
{
    const uint8_t opcode[2] = { 0x89, 0x00 };
    emit_memory_access(assembler, 1, 0, opcode, 1, source, base, "store_mem16_r16");
}

void
bal_x86_emit_store_mem8_r8(bal_x86_assembler_t     *assembler,
                           const bal_x86_register_t base,
                           const bal_x86_register_t source)
{
    const uint8_t opcode[2] = { 0x88, 0x00 };
    emit_memory_access(assembler, 0, 0, opcode, 1, source, base, "store_mem8_r8");
}

void
bal_x86_emit_call_r64(bal_x86_assembler_t *assembler, const bal_x86_register_t target)
{
    if (BAL_UNLIKELY(NULL == assembler))
    {
        return;
    }

    if (BAL_UNLIKELY(assembler->status != BAL_SUCCESS))
    {
        BAL_LOG_ERROR(&bal_thread_logger, "Aborting function: assembler status != BAL_SUCCESS");
        return;
    }

    if (BAL_UNLIKELY(false == is_valid_register(target)))
    {
        BAL_LOG_ERROR(&bal_thread_logger, "Invalid call target register: %d", target);
        assembler->status = BAL_ERROR_INVALID_ARGUMENT;
        return;
    }

    const size_t instruction_size_bytes = 3;
    const bool   can_emit_status        = can_emit(assembler, instruction_size_bytes);

    if (BAL_UNLIKELY(false == can_emit_status))
    {
        return;
    }

    BAL_LOG_DEBUG(&bal_thread_logger, "[+0x%04zx] call r%d", assembler->offset, target);
    const uint8_t w         = 0U;
    const uint8_t r         = 0U;
    const uint8_t b         = (uint8_t)target >> 3U;
    const uint8_t extension = 2U;
    emit_rex(assembler->buffer, &assembler->offset, w, r, b);
    emit8(assembler->buffer, &assembler->offset, 0xFF);
    emit_modrm_register(assembler->buffer, &assembler->offset, (int)extension, target);
}

void
bal_x86_emit_store_r32_rbp_offset(bal_x86_assembler_t     *assembler,
                                  const bal_x86_register_t source,
                                  const int32_t            offset)
{
    if (BAL_UNLIKELY(NULL == assembler))
    {
        return;
    }

    if (BAL_UNLIKELY(assembler->status != BAL_SUCCESS))
    {
        BAL_LOG_ERROR(&bal_thread_logger, "Aborting function: assembler status != BAL_SUCCESS");
        return;
    }

    if (BAL_UNLIKELY(false == is_valid_register(source)))
    {
        BAL_LOG_ERROR(&bal_thread_logger, "Invalid source register: %d", source);
        assembler->status = BAL_ERROR_INVALID_ARGUMENT;
        return;
    }

    const size_t instruction_size_bytes = 7;
    const bool   can_emit_status        = can_emit(assembler, instruction_size_bytes);

    if (BAL_UNLIKELY(false == can_emit_status))
    {
        return;
    }

    BAL_LOG_DEBUG(&bal_thread_logger,
                  "[+0x%04zx] mov dword [rbp + 0x%X], r%dd",
                  assembler->offset,
                  offset,
                  source);
    const uint8_t w = 0U;
    const uint8_t r = (uint8_t)source >> 3U;
    const uint8_t b = 0U;
    emit_rex(assembler->buffer, &assembler->offset, w, r, b);
    emit8(assembler->buffer, &assembler->offset, 0x89);
    emit_modrm_memory_disp32_rbp(assembler->buffer, &assembler->offset, source);
    emit32(assembler->buffer, &assembler->offset, (uint32_t)offset);
}

void
bal_x86_emit_cmov_r64_r64(bal_x86_assembler_t      *assembler,
                          const bal_x86_condition_t condition,
                          const bal_x86_register_t  destination,
                          const bal_x86_register_t  source)
{
    const uint8_t opcode   = (uint8_t)(0x40U | ((uint32_t)condition & 0xFU));
    const bool    is_64bit = true;
    emit_register_register_0f(assembler, opcode, is_64bit, destination, source, "cmov");
}

void
bal_x86_emit_imul_r64_r64(bal_x86_assembler_t     *assembler,
                          const bal_x86_register_t destination,
                          const bal_x86_register_t source)
{
    const uint8_t opcode   = 0xAF;
    const bool    is_64bit = true;
    emit_register_register_0f(assembler, opcode, is_64bit, destination, source, "imul");
}

void
bal_x86_emit_imul_r32_r32(bal_x86_assembler_t     *assembler,
                          const bal_x86_register_t destination,
                          const bal_x86_register_t source)
{
    const uint8_t opcode   = 0xAF;
    const bool    is_64bit = false;
    emit_register_register_0f(assembler, opcode, is_64bit, destination, source, "imul");
}

void
bal_x86_emit_movsxd_r64_r32(bal_x86_assembler_t     *assembler,
                            const bal_x86_register_t destination,
                            const bal_x86_register_t source)
{
    const uint8_t opcode   = 0x63;
    const bool    is_64bit = true;
    emit_register_register(assembler, opcode, is_64bit, destination, source, "movsxd");
}

void
bal_x86_emit_mov_r32_r32(bal_x86_assembler_t     *assembler,
                         const bal_x86_register_t destination,
                         const bal_x86_register_t source)
{
    const uint8_t opcode   = 0x8B;
    const bool    is_64bit = false;
    emit_register_register(assembler, opcode, is_64bit, destination, source, "mov");
}

void
bal_x86_emit_shl_r64_cl(bal_x86_assembler_t *assembler, const bal_x86_register_t destination)
{
    const uint8_t opcode_extension = 4;
    const bool    is_64bit         = true;
    emit_shift_cl(assembler, destination, opcode_extension, is_64bit, "shl");
}

void
bal_x86_emit_shl_r32_cl(bal_x86_assembler_t *assembler, const bal_x86_register_t destination)
{
    const uint8_t opcode_extension = 4;
    const bool    is_64bit         = false;
    emit_shift_cl(assembler, destination, opcode_extension, is_64bit, "shl");
}

void
bal_x86_emit_shr_r64_cl(bal_x86_assembler_t *assembler, const bal_x86_register_t destination)
{
    const uint8_t opcode_extension = 5;
    const bool    is_64bit         = true;
    emit_shift_cl(assembler, destination, opcode_extension, is_64bit, "shr");
}

void
bal_x86_emit_shr_r32_cl(bal_x86_assembler_t *assembler, const bal_x86_register_t destination)
{
    const uint8_t opcode_extension = 5;
    const bool    is_64bit         = false;
    emit_shift_cl(assembler, destination, opcode_extension, is_64bit, "shr");
}

void
bal_x86_emit_sar_r64_cl(bal_x86_assembler_t *assembler, const bal_x86_register_t destination)
{
    const uint8_t opcode_extension = 7;
    const bool    is_64bit         = true;
    emit_shift_cl(assembler, destination, opcode_extension, is_64bit, "sar");
}

void
bal_x86_emit_sar_r32_cl(bal_x86_assembler_t *assembler, const bal_x86_register_t destination)
{
    const uint8_t opcode_extension = 7;
    const bool    is_64bit         = false;
    emit_shift_cl(assembler, destination, opcode_extension, is_64bit, "sar");
}

void
bal_x86_emit_ror_r64_cl(bal_x86_assembler_t *assembler, const bal_x86_register_t destination)
{
    const uint8_t opcode_extension = 1;
    const bool    is_64bit         = true;
    emit_shift_cl(assembler, destination, opcode_extension, is_64bit, "ror");
}

void
bal_x86_emit_ror_r32_cl(bal_x86_assembler_t *assembler, const bal_x86_register_t destination)
{
    const uint8_t opcode_extension = 1;
    const bool    is_64bit         = false;
    emit_shift_cl(assembler, destination, opcode_extension, is_64bit, "ror");
}

void
bal_x86_emit_adc_r64_r64(bal_x86_assembler_t     *assembler,
                         const bal_x86_register_t destination,
                         const bal_x86_register_t source)
{
    const uint8_t opcode   = 0x13;
    const bool    is_64bit = true;
    emit_register_register(assembler, opcode, is_64bit, destination, source, "adc");
}

void
bal_x86_emit_adc_r32_r32(bal_x86_assembler_t     *assembler,
                         const bal_x86_register_t destination,
                         const bal_x86_register_t source)
{
    const uint8_t opcode   = 0x13;
    const bool    is_64bit = false;
    emit_register_register(assembler, opcode, is_64bit, destination, source, "adc");
}

void
bal_x86_emit_sbb_r64_r64(bal_x86_assembler_t     *assembler,
                         const bal_x86_register_t destination,
                         const bal_x86_register_t source)
{
    const uint8_t opcode   = 0x1B;
    const bool    is_64bit = true;
    emit_register_register(assembler, opcode, is_64bit, destination, source, "sbb");
}

void
bal_x86_emit_sbb_r32_r32(bal_x86_assembler_t     *assembler,
                         const bal_x86_register_t destination,
                         const bal_x86_register_t source)
{
    const uint8_t opcode   = 0x1B;
    const bool    is_64bit = false;
    emit_register_register(assembler, opcode, is_64bit, destination, source, "sbb");
}

void
bal_x86_emit_bt_r64_imm8(bal_x86_assembler_t     *assembler,
                         const bal_x86_register_t destination,
                         const uint8_t            bit)
{
    if (BAL_UNLIKELY(NULL == assembler))
    {
        return;
    }

    if (BAL_UNLIKELY(assembler->status != BAL_SUCCESS))
    {
        BAL_LOG_ERROR(&bal_thread_logger, "Aborting function: assembler status != BAL_SUCCESS");
        return;
    }

    if (BAL_UNLIKELY(false == is_valid_register(destination)) || BAL_UNLIKELY(bit > 63U))
    {
        BAL_LOG_ERROR(
            &bal_thread_logger, "Invalid bt operands: r%d, %u", destination, (unsigned)bit);
        assembler->status = BAL_ERROR_INVALID_ARGUMENT;
        return;
    }

    const size_t instruction_size_bytes = 5;
    const bool   can_emit_status        = can_emit(assembler, instruction_size_bytes);

    if (BAL_UNLIKELY(false == can_emit_status))
    {
        return;
    }

    BAL_LOG_DEBUG(
        &bal_thread_logger, "[+0x%04zx] bt r%d, %u", assembler->offset, destination, (unsigned)bit);

    const uint8_t w          = 1U;
    const uint8_t r          = 0U;
    const uint8_t b          = (uint8_t)destination >> 3U;
    const uint8_t extension  = 4U;
    const size_t  old_offset = assembler->offset;
    emit_rex(assembler->buffer, &assembler->offset, w, r, b);
    emit8(assembler->buffer, &assembler->offset, 0x0F);
    emit8(assembler->buffer, &assembler->offset, 0xBA);
    emit_modrm_register(assembler->buffer, &assembler->offset, (int)extension, destination);
    emit8(assembler->buffer, &assembler->offset, bit);

    const size_t bytes_emitted = assembler->offset - old_offset;
    BAL_ASSERT_MSG(bytes_emitted == instruction_size_bytes,
                   "Bytes emitted %d does not match instruction size %d",
                   bytes_emitted,
                   instruction_size_bytes);
}

void
bal_x86_emit_cmc(bal_x86_assembler_t *assembler)
{
    if (BAL_UNLIKELY(NULL == assembler))
    {
        return;
    }

    if (BAL_UNLIKELY(assembler->status != BAL_SUCCESS))
    {
        BAL_LOG_ERROR(&bal_thread_logger, "Aborting function: assembler status != BAL_SUCCESS");
        return;
    }

    const size_t instruction_size_bytes = 1;
    const bool   can_emit_status        = can_emit(assembler, instruction_size_bytes);

    if (BAL_UNLIKELY(false == can_emit_status))
    {
        return;
    }

    BAL_LOG_DEBUG(&bal_thread_logger, "[+0x%04zx] cmc", assembler->offset);
    emit8(assembler->buffer, &assembler->offset, 0xF5);
}

void
bal_x86_emit_cmov_r32_r32(bal_x86_assembler_t      *assembler,
                          const bal_x86_condition_t condition,
                          const bal_x86_register_t  destination,
                          const bal_x86_register_t  source)
{
    const uint8_t opcode   = (uint8_t)(0x40U | ((uint32_t)condition & 0xFU));
    const bool    is_64bit = false;
    emit_register_register_0f(assembler, opcode, is_64bit, destination, source, "cmov");
}

void
bal_x86_emit_push_r64(bal_x86_assembler_t *assembler, const bal_x86_register_t reg)
{
    if (BAL_UNLIKELY(NULL == assembler))
    {
        return;
    }

    if (BAL_UNLIKELY(assembler->status != BAL_SUCCESS))
    {
        BAL_LOG_ERROR(&bal_thread_logger, "Assembler status != BAL_SUCCESS, aborting function");
        assembler->status = BAL_ERROR_INVALID_ARGUMENT;
        return;
    }

    const bool is_valid_register_result = is_valid_register(reg);

    if (BAL_UNLIKELY(false == is_valid_register_result))
    {
        BAL_LOG_ERROR(&bal_thread_logger, "Invalid register: %d", reg);
        assembler->status = BAL_ERROR_INVALID_ARGUMENT;
        return;
    }

    // Registers R8 to R15 need a REX prefix in front of the one-byte opcode.
    const size_t instruction_size_bytes = reg > 7 ? 2U : 1U;
    const bool   can_emit_status        = can_emit(assembler, instruction_size_bytes);

    if (BAL_UNLIKELY(false == can_emit_status))
    {
        return;
    }

    BAL_LOG_DEBUG(&bal_thread_logger, "[+0x%04zx] push r%d", assembler->offset, reg);
    const size_t old_offset = assembler->offset;

    if (reg > 7)
    {
        const uint8_t w = 0;
        const uint8_t r = 0;

        // WARNING: reg is verified by is_valid_register() to fall within the safe enum range
        // [0, 15].
        const uint8_t b = (uint8_t)reg >> 3;

        emit_rex(assembler->buffer, &assembler->offset, w, r, b);
    }

    const uint8_t opcode = 0x50;
    emit8(assembler->buffer, &assembler->offset, opcode + (reg & 7));
    const size_t bytes_emitted = assembler->offset - old_offset;
    BAL_ASSERT_MSG(bytes_emitted <= instruction_size_bytes,
                   "Bytes emitted %d does not match instruction size %d",
                   bytes_emitted,
                   instruction_size_bytes);
}

void
bal_x86_emit_pop_r64(bal_x86_assembler_t *assembler, const bal_x86_register_t reg)
{
    if (BAL_UNLIKELY(NULL == assembler))
    {
        return;
    }

    if (BAL_UNLIKELY(assembler->status != BAL_SUCCESS))
    {
        BAL_LOG_ERROR(&bal_thread_logger, "Assembler status != BAL_SUCCESS, aborting function");
        assembler->status = BAL_ERROR_INVALID_ARGUMENT;
        return;
    }

    const bool is_valid_register_result = is_valid_register(reg);

    if (BAL_UNLIKELY(false == is_valid_register_result))
    {
        BAL_LOG_ERROR(&bal_thread_logger, "Invalid register: %d", reg);
        assembler->status = BAL_ERROR_INVALID_ARGUMENT;
        return;
    }

    // Registers R8 to R15 need a REX prefix in front of the one-byte opcode.
    const size_t instruction_size_bytes = reg > 7 ? 2U : 1U;
    const bool   can_emit_status        = can_emit(assembler, instruction_size_bytes);

    if (BAL_UNLIKELY(false == can_emit_status))
    {
        return;
    }

    BAL_LOG_DEBUG(&bal_thread_logger, "[+0x%04zx] pop r%d", assembler->offset, reg);
    const size_t old_offset = assembler->offset;

    if (reg > 7)
    {
        const uint8_t w = 0;
        const uint8_t r = 0;

        // WARNING: reg is verified by is_valid_register() to fall within the safe enum range
        // [0, 15].
        const uint8_t b = (uint8_t)reg >> 3;

        emit_rex(assembler->buffer, &assembler->offset, w, r, b);
    }

    const uint8_t opcode = 0x58;
    emit8(assembler->buffer, &assembler->offset, opcode + (reg & 7));
    const size_t bytes_emitted = assembler->offset - old_offset;
    BAL_ASSERT_MSG(bytes_emitted == instruction_size_bytes,
                   "Bytes emitted %d does not match instruction size %d",
                   bytes_emitted,
                   instruction_size_bytes);
}

void
bal_x86_emit_setcc_mem8_rbp_offset(bal_x86_assembler_t      *assembler,
                                   const bal_x86_condition_t condition,
                                   const int32_t             offset)
{
    if (BAL_UNLIKELY(NULL == assembler))
    {
        return;
    }

    if (BAL_UNLIKELY(assembler->status != BAL_SUCCESS))
    {
        BAL_LOG_ERROR(&bal_thread_logger, "Assembler status != BAL_SUCCESS, aborting function");
        assembler->status = BAL_ERROR_INVALID_ARGUMENT;
        return;
    }

    const size_t instruction_size_bytes = 7;
    const bool   can_emit_status        = can_emit(assembler, instruction_size_bytes);

    if (BAL_UNLIKELY(false == can_emit_status))
    {
        return;
    }

    BAL_LOG_DEBUG(&bal_thread_logger,
                  "[+0x%04zx] set%s byte ptr [rbp + 0x%X]",
                  assembler->offset,
                  condition_code_to_string(condition),
                  offset);
    const size_t old_offset = assembler->offset;

    // SETcc is 0x0F 0x90+cc.
    emit8(assembler->buffer, &assembler->offset, 0x0F);

    // WARNING: Cast to uint8_t is safe because bal_x86_condition_t enum values fall within 0x0 to
    // 0xF.
    emit8(assembler->buffer, &assembler->offset, 0x90 + (uint8_t)condition);

    // Opcode extension is 0.
    emit_modrm_memory_disp32_rbp(assembler->buffer, &assembler->offset, BAL_X86_RAX);

    // WARNING: Bit pattern of int32_t offset is preserved.
    emit32(assembler->buffer, &assembler->offset, (uint32_t)offset);

    const size_t bytes_emitted = assembler->offset - old_offset;
    BAL_ASSERT_MSG(bytes_emitted == instruction_size_bytes,
                   "Bytes emitted %d does not match instruction size %d",
                   bytes_emitted,
                   instruction_size_bytes);
}

void
bal_x86_emit_jcc_rel32(bal_x86_assembler_t      *assembler,
                       const bal_x86_condition_t condition,
                       const int32_t             offset)
{
    if (BAL_UNLIKELY(NULL == assembler))
    {
        return;
    }

    if (BAL_UNLIKELY(assembler->status != BAL_SUCCESS))
    {
        BAL_LOG_ERROR(&bal_thread_logger, "Assembler status != BAL_SUCCESS, aborting function");
        assembler->status = BAL_ERROR_INVALID_ARGUMENT;
        return;
    }

    const size_t instruction_size_bytes = 6;
    const bool   can_emit_status        = can_emit(assembler, instruction_size_bytes);

    if (BAL_UNLIKELY(false == can_emit_status))
    {
        return;
    }

    BAL_LOG_DEBUG(&bal_thread_logger,
                  "[0x%04zx] j%s 0x%X",
                  assembler->offset,
                  condition_code_to_string(condition),
                  offset);
    const size_t old_offset = assembler->offset;
    emit8(assembler->buffer, &assembler->offset, 0x0F);

    // WARNING: Cast to uint8_t is safe because bal_x86_condition_t enum values fall within 0x0 to
    // 0xF.
    emit8(assembler->buffer, &assembler->offset, 0x80 + (uint8_t)condition);

    // WARNING: Bit pattern of int32_t offset is preserved in uint32_t.
    emit32(assembler->buffer, &assembler->offset, (uint32_t)offset);

    const size_t bytes_emitted = assembler->offset - old_offset;
    BAL_ASSERT_MSG(bytes_emitted == instruction_size_bytes,
                   "Bytes emitted %d does not match instruction size %d",
                   bytes_emitted,
                   instruction_size_bytes);
}

void
bal_x86_emit_add_r64_r64(bal_x86_assembler_t     *assembler,
                         const bal_x86_register_t destination,
                         const bal_x86_register_t source)
{

    if (BAL_UNLIKELY(NULL == assembler))
    {
        return;
    }

    if (BAL_UNLIKELY(assembler->status != BAL_SUCCESS))
    {
        BAL_LOG_ERROR(&bal_thread_logger, "Aborting function: assembler->status != BAL_SUCCESS");
        assembler->status = BAL_ERROR_INVALID_ARGUMENT;
        return;
    }

    bool is_valid_register_result = is_valid_register(source);

    if (BAL_UNLIKELY(false == is_valid_register_result))
    {
        BAL_LOG_ERROR(&bal_thread_logger, "Aborting function: invalid source register: %d", source);
        assembler->status = BAL_ERROR_INVALID_ARGUMENT;
        return;
    }

    is_valid_register_result = is_valid_register(destination);

    if (BAL_UNLIKELY(false == is_valid_register_result))
    {
        BAL_LOG_ERROR(
            &bal_thread_logger, "Aborting function: invalid destination register: %d", destination);
        assembler->status = BAL_ERROR_INVALID_ARGUMENT;
        return;
    }

    const size_t instruction_size_bytes = 3;
    const bool   can_emit_status        = can_emit(assembler, instruction_size_bytes);

    if (BAL_UNLIKELY(false == can_emit_status))
    {
        return;
    }

    BAL_LOG_DEBUG(
        &bal_thread_logger, "[+0x%04zx] add r%d, r%d", assembler->offset, destination, source);
    const uint8_t w = 1;

    // WARNING: Destination is verified by is_valid_register().
    const uint8_t r = (uint8_t)destination >> 3;

    // WARNING: Source is verified by is_valid_register().
    const uint8_t b = (uint8_t)source >> 3;

    const uint8_t opcode     = 0x03;
    const size_t  old_offset = assembler->offset;
    emit_rex(assembler->buffer, &assembler->offset, w, r, b);
    emit8(assembler->buffer, &assembler->offset, opcode);
    emit_modrm_register(assembler->buffer, &assembler->offset, destination, source);
    const size_t bytes_emitted = assembler->offset - old_offset;
    BAL_ASSERT_MSG(bytes_emitted == instruction_size_bytes,
                   "Bytes emitted %d does not match instruction size %d",
                   bytes_emitted,
                   instruction_size_bytes);
}

void
bal_x86_emit_add_r32_r32(bal_x86_assembler_t *assembler,
                         bal_x86_register_t   destination,
                         bal_x86_register_t   source)
{
    if (BAL_UNLIKELY(NULL == assembler))
    {
        BAL_LOG_ERROR(&bal_thread_logger, "Aborting function: assembler is NULL.");
        return;
    }

    if (BAL_UNLIKELY(assembler->status != BAL_SUCCESS))
    {
        BAL_LOG_ERROR(&bal_thread_logger, "Aborting function: assembler->status != BAL_SUCCESS.");
        assembler->status = BAL_ERROR_INVALID_ARGUMENT;
    }

    bool is_valid_register_result = is_valid_register(source);

    if (BAL_UNLIKELY(false == is_valid_register_result))
    {
        BAL_LOG_ERROR(
            &bal_thread_logger, "Aborting function: invalid source register: %d.", source);
        assembler->status = BAL_ERROR_INVALID_ARGUMENT;
        return;
    }

    is_valid_register_result = is_valid_register(destination);

    if (BAL_UNLIKELY(false == is_valid_register_result))
    {
        BAL_LOG_ERROR(
            &bal_thread_logger, "Aborting function: invalid destination register: %d", destination);
        assembler->status = BAL_ERROR_INVALID_ARGUMENT;
        return;
    }

    const size_t instruction_size_bytes = 3U;
    const bool   can_emit_status        = can_emit(assembler, instruction_size_bytes);

    if (BAL_UNLIKELY(false == can_emit_status))
    {
        return;
    }

    BAL_LOG_DEBUG(
        &bal_thread_logger, "[+0x%04zx] add r%dd, r%dd", assembler->offset, destination, source);

    const uint8_t w          = 0U;
    const uint8_t r          = (uint8_t)destination >> 3U;
    const uint8_t b          = (uint8_t)source >> 3U;
    const uint8_t opcode     = 0x03U;
    const size_t  old_offset = assembler->offset;
    emit_rex(assembler->buffer, &assembler->offset, w, r, b);
    emit8(assembler->buffer, &assembler->offset, opcode);
    emit_modrm_register(assembler->buffer, &assembler->offset, destination, source);
    const size_t bytes_emitted = assembler->offset - old_offset;
    BAL_ASSERT_MSG(bytes_emitted == instruction_size_bytes,
                   "Bytes emitted %d does not match instruction size %d",
                   bytes_emitted,
                   instruction_size_bytes);
}

void
bal_x86_emit_sub_r64_r64(bal_x86_assembler_t     *assembler,
                         const bal_x86_register_t destination,
                         const bal_x86_register_t source)
{

    if (BAL_UNLIKELY(NULL == assembler))
    {
        return;
    }

    if (BAL_UNLIKELY(assembler->status != BAL_SUCCESS))
    {
        BAL_LOG_ERROR(&bal_thread_logger, "Aborting function: assembler->status != BAL_SUCCESS");
        assembler->status = BAL_ERROR_INVALID_ARGUMENT;
        return;
    }

    bool is_valid_register_result = is_valid_register(source);

    if (BAL_UNLIKELY(false == is_valid_register_result))
    {
        BAL_LOG_ERROR(&bal_thread_logger, "Aborting function: invalid source register: %d", source);
        assembler->status = BAL_ERROR_INVALID_ARGUMENT;
        return;
    }

    is_valid_register_result = is_valid_register(destination);

    if (BAL_UNLIKELY(false == is_valid_register_result))
    {
        BAL_LOG_ERROR(
            &bal_thread_logger, "Aborting function: invalid destination register: %d", destination);
        assembler->status = BAL_ERROR_INVALID_ARGUMENT;
        return;
    }

    const size_t instruction_size_bytes = 3;
    const bool   can_emit_status        = can_emit(assembler, instruction_size_bytes);

    if (BAL_UNLIKELY(false == can_emit_status))
    {
        return;
    }

    BAL_LOG_DEBUG(
        &bal_thread_logger, "[+0x%04zx] sub r%d, r%d", assembler->offset, destination, source);
    const uint8_t w = 1;

    // WARNING: Destination is verified by is_valid_register().
    const uint8_t r = (uint8_t)destination >> 3;

    // WARNING: Source is verified by is_valid_register().
    const uint8_t b = (uint8_t)source >> 3;

    const uint8_t opcode     = 0x2B;
    const size_t  old_offset = assembler->offset;
    emit_rex(assembler->buffer, &assembler->offset, w, r, b);
    emit8(assembler->buffer, &assembler->offset, opcode);
    emit_modrm_register(assembler->buffer, &assembler->offset, destination, source);
    const size_t bytes_emitted = assembler->offset - old_offset;
    BAL_ASSERT_MSG(bytes_emitted == instruction_size_bytes,
                   "Bytes emitted %d does not match instruction size %d",
                   bytes_emitted,
                   instruction_size_bytes);
}

void
bal_x86_emit_sub_r32_r32(bal_x86_assembler_t *assembler,
                         bal_x86_register_t   destination,
                         bal_x86_register_t   source)
{
    if (BAL_UNLIKELY(NULL == assembler))
    {
        BAL_LOG_ERROR(&bal_thread_logger, "Aborting function: assembler is NULL.");
        return;
    }

    if (BAL_UNLIKELY(assembler->status != BAL_SUCCESS))
    {
        BAL_LOG_ERROR(&bal_thread_logger, "Aborting function: assembler->status != BAL_SUCCESS.");
        assembler->status = BAL_ERROR_INVALID_ARGUMENT;
        return;
    }

    bool is_valid_register_result = is_valid_register(source);

    if (BAL_UNLIKELY(false == is_valid_register_result))
    {
        BAL_LOG_ERROR(
            &bal_thread_logger, "Aborting function: invalid source register: %d.", source);
        assembler->status = BAL_ERROR_INVALID_ARGUMENT;
        return;
    }

    is_valid_register_result = is_valid_register(destination);

    if (BAL_UNLIKELY(false == is_valid_register_result))
    {
        BAL_LOG_ERROR(
            &bal_thread_logger, "Aborting function: invalid destination register: %d", destination);
        assembler->status = BAL_ERROR_INVALID_ARGUMENT;
        return;
    }

    const size_t instruction_size_bytes = 3U;
    const bool   can_emit_status        = can_emit(assembler, instruction_size_bytes);

    if (BAL_UNLIKELY(false == can_emit_status))
    {
        return;
    }

    BAL_LOG_DEBUG(
        &bal_thread_logger, "[+0x%04zx] sub r%d, r%d", assembler->offset, destination, source);

    const uint8_t w          = 0U;
    const uint8_t r          = (uint8_t)destination >> 3U;
    const uint8_t b          = (uint8_t)source >> 3U;
    const uint8_t opcode     = 0x2BU;
    const size_t  old_offset = assembler->offset;
    emit_rex(assembler->buffer, &assembler->offset, w, r, b);
    emit8(assembler->buffer, &assembler->offset, opcode);
    emit_modrm_register(assembler->buffer, &assembler->offset, destination, source);
    const size_t bytes_emitted = assembler->offset - old_offset;
    BAL_ASSERT_MSG(bytes_emitted == instruction_size_bytes,
                   "Bytes emitted %d does not match instruction size %d",
                   bytes_emitted,
                   instruction_size_bytes);
}

void
bal_x86_emit_add_r64_imm32(bal_x86_assembler_t     *assembler,
                           const bal_x86_register_t destination,
                           const int32_t            immediate)
{
    if (BAL_UNLIKELY(NULL == assembler))
    {
        return;
    }

    if (BAL_UNLIKELY(assembler->status != BAL_SUCCESS))
    {
        BAL_LOG_ERROR(&bal_thread_logger, "Aborting function: assembler->status != BAL_SUCCESS");
        assembler->status = BAL_ERROR_INVALID_ARGUMENT;
        return;
    }

    const bool is_valid_register_result = is_valid_register(destination);

    if (BAL_UNLIKELY(false == is_valid_register_result))
    {
        BAL_LOG_ERROR(
            &bal_thread_logger, "Aborting function: invalid destination register: %d", destination);
        assembler->status = BAL_ERROR_INVALID_ARGUMENT;
        return;
    }

    const bool fit_in_8_bits = immediate >= -128 && immediate < 127;

    // WARNING: Destination is verified by is_valid_register().
    const uint8_t b = (uint8_t)destination >> 3;

    const size_t instruction_size_bytes = true == fit_in_8_bits ? 4 : 7;
    const bool   can_emit_status        = can_emit(assembler, instruction_size_bytes);

    if (BAL_UNLIKELY(false == can_emit_status))
    {
        return;
    }

    BAL_LOG_DEBUG(
        &bal_thread_logger, "[+0x%04zx] add r%d, 0x%X", assembler->offset, destination, immediate);
    const size_t  old_offset = assembler->offset;
    const uint8_t w          = 1;
    const uint8_t r          = 0;
    emit_rex(assembler->buffer, &assembler->offset, w, r, b);
    const uint8_t opcode = true == fit_in_8_bits ? 0x83 : 0x81;
    emit8(assembler->buffer, &assembler->offset, opcode);
    emit_modrm_register(assembler->buffer, &assembler->offset, BAL_X86_RAX, destination);

    if (true == fit_in_8_bits)
    {
        // WARNING: Cast is safe if fit_in_8_bits is true.
        emit8(assembler->buffer, &assembler->offset, (uint8_t)immediate);
    }
    else
    {
        // WARNING: Bit pattern of int32_t is preserved.
        emit32(assembler->buffer, &assembler->offset, (uint32_t)immediate);
    }

    const size_t bytes_emitted = assembler->offset - old_offset;
    BAL_ASSERT_MSG(bytes_emitted == instruction_size_bytes,
                   "Bytes emitted %d does not match instruction size %d",
                   bytes_emitted,
                   instruction_size_bytes);
}

void
bal_x86_emit_add_r32_imm32(bal_x86_assembler_t *assembler,
                           bal_x86_register_t   destination,
                           int32_t              immediate)
{
    if (BAL_UNLIKELY(NULL == assembler))
    {
        BAL_LOG_ERROR(&bal_thread_logger, "Aborting function: assembler is NULL.");
        return;
    }

    if (BAL_UNLIKELY(assembler->status != BAL_SUCCESS))
    {
        BAL_LOG_ERROR(&bal_thread_logger, "Aborting function: assembler->status != BAL_SUCCESS.");
        assembler->status = BAL_ERROR_INVALID_ARGUMENT;
        return;
    }

    const bool is_valid_register_result = is_valid_register(destination);

    if (BAL_UNLIKELY(false == is_valid_register_result))
    {
        BAL_LOG_ERROR(&bal_thread_logger,
                      "Aborting function: invalid destination register: %d.",
                      destination);
        assembler->status = BAL_ERROR_INVALID_ARGUMENT;
        return;
    }

    const bool   fit_in_8_bits          = immediate >= -128 && immediate <= 127;
    const size_t instruction_size_bytes = true == fit_in_8_bits ? 4 : 7;
    const bool   can_emit_status        = can_emit(assembler, instruction_size_bytes);

    if (BAL_UNLIKELY(false == can_emit_status))
    {
        return;
    }

    BAL_LOG_DEBUG(
        &bal_thread_logger, "[+0x%04zx] add r%dd, 0x%X", assembler->offset, destination, immediate);

    const size_t  old_offset = assembler->offset;
    const uint8_t w          = 0U;
    const uint8_t r          = 0U;
    const uint8_t b          = (uint8_t)destination >> 3U;
    emit_rex(assembler->buffer, &assembler->offset, w, r, b);
    const uint8_t opcode = true == fit_in_8_bits ? 0x83 : 0x81;
    emit8(assembler->buffer, &assembler->offset, opcode);
    emit_modrm_register(assembler->buffer, &assembler->offset, BAL_X86_RAX, destination);

    if (true == fit_in_8_bits)
    {
        emit8(assembler->buffer, &assembler->offset, (uint8_t)immediate);
    }
    else
    {
        emit32(assembler->buffer, &assembler->offset, (uint32_t)immediate);
    }

    const size_t bytes_emitted = assembler->offset - old_offset;
    BAL_ASSERT_MSG(bytes_emitted == instruction_size_bytes,
                   "Bytes emitted %d does not match instruction size %d",
                   bytes_emitted,
                   instruction_size_bytes);
}

void
bal_x86_emit_test_r64_r64(bal_x86_assembler_t     *assembler,
                          const bal_x86_register_t destination,
                          const bal_x86_register_t source)
{
    if (BAL_UNLIKELY(NULL == assembler))
    {
        return;
    }
    if (BAL_UNLIKELY(assembler->status != BAL_SUCCESS))
    {
        BAL_LOG_ERROR(&bal_thread_logger, "Aborting function: assembler status != BAL_SUCCESS");
        return;
    }

    bool is_valid_register_result = is_valid_register(destination);

    if (BAL_UNLIKELY(false == is_valid_register_result))
    {
        BAL_LOG_ERROR(&bal_thread_logger, "Invalid destination register: %d", destination);
        assembler->status = BAL_ERROR_INVALID_ARGUMENT;
        return;
    }

    is_valid_register_result = is_valid_register(source);

    if (BAL_UNLIKELY(false == is_valid_register_result))
    {
        BAL_LOG_ERROR(&bal_thread_logger, "Invalid source register: %d", source);
        assembler->status = BAL_ERROR_INVALID_ARGUMENT;
        return;
    }

    const size_t instruction_size_bytes = 3;
    const bool   can_emit_status        = can_emit(assembler, instruction_size_bytes);

    if (BAL_UNLIKELY(false == can_emit_status))
    {
        return;
    }

    BAL_LOG_DEBUG(
        &bal_thread_logger, "[+0x%04zx] test r%d, r%d", assembler->offset, destination, source);

    const uint8_t w = 1;
    const uint8_t r = (uint8_t)source >> 3;
    const uint8_t b = (uint8_t)destination >> 3;

    const uint8_t opcode     = 0x85;
    const size_t  old_offset = assembler->offset;
    emit_rex(assembler->buffer, &assembler->offset, w, r, b);
    emit8(assembler->buffer, &assembler->offset, opcode);
    emit_modrm_register(assembler->buffer, &assembler->offset, source, destination);

    const size_t bytes_emitted = assembler->offset - old_offset;
    BAL_ASSERT_MSG(bytes_emitted == instruction_size_bytes,
                   "Bytes emitted %d does not match instruction size %d",
                   bytes_emitted,
                   instruction_size_bytes);
}

void
bal_x86_emit_xor_r64_r64(bal_x86_assembler_t *assembler,
                         bal_x86_register_t   destination,
                         bal_x86_register_t   source)
{
    if (BAL_UNLIKELY(NULL == assembler))
    {
        return;
    }

    if (BAL_UNLIKELY(assembler->status != BAL_SUCCESS))
    {
        BAL_LOG_ERROR(&bal_thread_logger, "Aborting function: assembler status != BAL_SUCCESS");
        assembler->status = BAL_ERROR_INVALID_ARGUMENT;
        return;
    }

    bool is_valid_register_result = is_valid_register(source);

    if (BAL_UNLIKELY(false == is_valid_register_result))
    {
        BAL_LOG_ERROR(
            &bal_thread_logger, "Aborting function: invalid destination register %d", source);
        assembler->status = BAL_ERROR_INVALID_ARGUMENT;
        return;
    }

    is_valid_register_result = is_valid_register(destination);

    if (BAL_UNLIKELY(false == is_valid_register_result))
    {
        BAL_LOG_ERROR(
            &bal_thread_logger, "Aborting function: invalid destination register %d", destination);
        assembler->status = BAL_ERROR_INVALID_ARGUMENT;
        return;
    }

    const size_t instruction_size_bytes = 3;
    const bool   can_emit_status        = can_emit(assembler, instruction_size_bytes);

    if (BAL_UNLIKELY(false == can_emit_status))
    {
        return;
    }

    BAL_LOG_DEBUG(
        &bal_thread_logger, "[+0x%04zx] xor r%d, r%d", assembler->offset, destination, source);

    const uint8_t w = 1;

    // WARNING: Destination is verified by is_valid_register() to fall within the safe enum range
    // [0, 15].
    const uint8_t r = (uint8_t)destination >> 3;

    // WARNING: Source is verified by is_valid_register() to fall within the safe enum range
    // [0, 15].
    const uint8_t b = (uint8_t)source >> 3;

    const uint8_t opcode     = 0x33;
    const size_t  old_offset = assembler->offset;
    emit_rex(assembler->buffer, &assembler->offset, w, r, b);
    emit8(assembler->buffer, &assembler->offset, opcode);
    emit_modrm_register(assembler->buffer, &assembler->offset, destination, source);
    const size_t bytes_emitted = assembler->offset - old_offset;
    BAL_ASSERT_MSG(bytes_emitted == instruction_size_bytes,
                   "Bytes emitted %d does not match instruction size %d",
                   bytes_emitted,
                   instruction_size_bytes);
}

void
bal_x86_emit_sub_r64_imm32(bal_x86_assembler_t     *assembler,
                           const bal_x86_register_t destination,
                           const int32_t            immediate)
{
    if (BAL_UNLIKELY(NULL == assembler))
    {
        return;
    }
    if (BAL_UNLIKELY(assembler->status != BAL_SUCCESS))
    {
        BAL_LOG_ERROR(&bal_thread_logger, "Aborting function: assembler status != BAL_SUCCESS");
        return;
    }

    bool is_valid_register_result = is_valid_register(destination);

    if (BAL_UNLIKELY(false == is_valid_register_result))
    {
        BAL_LOG_ERROR(&bal_thread_logger, "Invalid destination register: %d", destination);
        assembler->status = BAL_ERROR_INVALID_ARGUMENT;
        return;
    }

    const bool   fit_in_8_bits          = immediate >= -128 && immediate < 127;
    const size_t instruction_size_bytes = true == fit_in_8_bits ? 4 : 7;
    const bool   can_emit_status        = can_emit(assembler, instruction_size_bytes);

    if (BAL_UNLIKELY(false == can_emit_status))
    {
        return;
    }

    BAL_LOG_DEBUG(
        &bal_thread_logger, "[+0x%04zx] sub r%d, 0x%X", assembler->offset, destination, immediate);

    const size_t  old_offset = assembler->offset;
    const uint8_t b          = (uint8_t)destination >> 3;
    const uint8_t w          = 1;
    const uint8_t r          = 0;
    emit_rex(assembler->buffer, &assembler->offset, w, r, b);
    const uint8_t opcode = true == fit_in_8_bits ? 0x83 : 0x81;
    emit8(assembler->buffer, &assembler->offset, opcode);
    emit_modrm_register(assembler->buffer, &assembler->offset, BAL_X86_RBP, destination);

    if (true == fit_in_8_bits)
    {
        emit8(assembler->buffer, &assembler->offset, (uint8_t)immediate);
    }
    else
    {
        emit32(assembler->buffer, &assembler->offset, (uint32_t)immediate);
    }

    const size_t bytes_emitted = assembler->offset - old_offset;
    BAL_ASSERT_MSG(bytes_emitted == instruction_size_bytes,
                   "Bytes emitted %d does not match instruction size %d",
                   bytes_emitted,
                   instruction_size_bytes);
}

void
bal_x86_emit_sub_r32_imm32(bal_x86_assembler_t *assembler,
                           bal_x86_register_t   destination,
                           int32_t              immediate)
{
    if (BAL_UNLIKELY(NULL == assembler))
    {
        BAL_LOG_ERROR(&bal_thread_logger, "Aborting function: assembler is NULL.");
        return;
    }

    if (BAL_UNLIKELY(assembler->status != BAL_SUCCESS))
    {
        BAL_LOG_ERROR(&bal_thread_logger, "Aborting function: assembler->status != BAL_SUCCESS.");
        return;
    }

    const bool is_valid_register_result = is_valid_register(destination);

    if (BAL_UNLIKELY(false == is_valid_register_result))
    {
        BAL_LOG_ERROR(
            &bal_thread_logger, "Aborting function: Invalid destination register: %d", destination);
    }

    const bool   fit_in_8_bits          = immediate >= -128 && immediate <= 127;
    const size_t instruction_size_bytes = true == fit_in_8_bits ? 4 : 7;
    const bool   can_emit_status        = can_emit(assembler, instruction_size_bytes);

    if (BAL_UNLIKELY(false == can_emit_status))
    {
        return;
    }

    BAL_LOG_DEBUG(
        &bal_thread_logger, "[+0x%04zx] sub r%dd, 0x%X", assembler->offset, destination, immediate);

    const size_t  old_offset = assembler->offset;
    const uint8_t w          = 0U;
    const uint8_t r          = 0U;
    const uint8_t b          = (uint8_t)destination >> 3U;

    emit_rex(assembler->buffer, &assembler->offset, w, r, b);
    const uint8_t opcode = true == fit_in_8_bits ? 0x83 : 0x81;
    emit8(assembler->buffer, &assembler->offset, opcode);
    emit_modrm_register(assembler->buffer, &assembler->offset, BAL_X86_RBP, destination);

    if (true == fit_in_8_bits)
    {
        emit8(assembler->buffer, &assembler->offset, (uint8_t)immediate);
    }
    else
    {
        emit32(assembler->buffer, &assembler->offset, (uint32_t)immediate);
    }

    const size_t bytes_emitted = assembler->offset - old_offset;
    BAL_ASSERT_MSG(bytes_emitted == instruction_size_bytes,
                   "Bytes emitted %d does not match instruction size %d",
                   bytes_emitted,
                   instruction_size_bytes);
}

void
bal_x86_emit_cmp_mem8_rbp_offset_imm(bal_x86_assembler_t *assembler,
                                     const int32_t        offset,
                                     const int8_t         immediate)
{
    if (BAL_UNLIKELY(NULL == assembler))
    {
        return;
    }

    if (BAL_UNLIKELY(assembler->status != BAL_SUCCESS))
    {
        BAL_LOG_ERROR(&bal_thread_logger, "Aborting function: assembler->status != BAL_SUCCESS ");
        return;
    }

    const size_t instruction_size_bytes = 8;
    const bool   can_emit_status        = can_emit(assembler, instruction_size_bytes);

    if (BAL_UNLIKELY(false == can_emit_status))
    {
        return;
    }

    BAL_LOG_DEBUG(&bal_thread_logger,
                  "[+0x%04zx] cmp byte ptr [rbp + 0x%X], 0x%X",
                  assembler->offset,
                  offset,
                  immediate);

    const size_t  old_offset = assembler->offset;
    const uint8_t w          = 0;
    const uint8_t r          = 0;
    const uint8_t b          = 0;
    const uint8_t opcode     = 0x80;
    emit_rex(assembler->buffer, &assembler->offset, w, r, b);
    emit8(assembler->buffer, &assembler->offset, opcode);
    emit_modrm_memory_disp32_rbp(assembler->buffer, &assembler->offset, BAL_X86_RDI);
    emit32(assembler->buffer, &assembler->offset, (uint32_t)offset);
    emit8(assembler->buffer, &assembler->offset, (uint8_t)immediate);

    const size_t bytes_emitted = assembler->offset - old_offset;
    BAL_ASSERT_MSG(bytes_emitted == instruction_size_bytes,
                   "Bytes emitted %d does not match instruction size %d",
                   bytes_emitted,
                   instruction_size_bytes);
}

void
bal_x86_emit_ud2(bal_x86_assembler_t *BAL_RESTRICT assembler)
{
    if (BAL_UNLIKELY(NULL == assembler))
    {
        return;
    }

    if (BAL_UNLIKELY(assembler->status != BAL_SUCCESS))
    {
        BAL_LOG_ERROR(&bal_thread_logger, "Aborting function: assembler status != BAL_SUCCESS.");
        return;
    }

    const size_t instruction_size_bytes = 2;
    const bool   can_emit_status        = can_emit(assembler, instruction_size_bytes);

    if (BAL_UNLIKELY(false == can_emit_status))
    {
        return;
    }

    BAL_LOG_DEBUG(&bal_thread_logger, "[+0x%04zx] ud2", assembler->offset);

    const size_t old_offset = assembler->offset;
    emit8(assembler->buffer, &assembler->offset, 0x0F);
    emit8(assembler->buffer, &assembler->offset, 0x0B);

    const size_t bytes_emitted = assembler->offset - old_offset;
    BAL_ASSERT_MSG(bytes_emitted == instruction_size_bytes,
                   "Bytes emitted %d does not match instruction size %d.",
                   bytes_emitted,
                   instruction_size_bytes);
}

bool
is_valid_register(const int reg)
{
    return reg >= BAL_X86_RAX && reg <= BAL_X86_R15;
}

bool
can_emit(bal_x86_assembler_t *assembler, const size_t size)
{
    if (BAL_UNLIKELY(NULL == assembler))
    {
        return false;
    }

    if (BAL_UNLIKELY(assembler->status != BAL_SUCCESS))
    {
        BAL_LOG_ERROR(&bal_thread_logger, "Aborting function: Assembler status != BAL_SUCCESS");
        return false;
    }

    if (BAL_UNLIKELY(SIZE_MAX - assembler->offset < size))
    {
        BAL_LOG_ERROR(&bal_thread_logger,
                      "x86 Assembler Integer Overflow. Current offset: %zu, Requested size: %zu",
                      assembler->offset,
                      size);
        assembler->status = BAL_ERROR_INSTRUCTION_OVERFLOW;
        return false;
    }

    const size_t assembler_size = assembler->offset + size;

    if (BAL_UNLIKELY(assembler_size > assembler->capacity))
    {
        BAL_LOG_ERROR(&bal_thread_logger,
                      "x86 Assembler Overflow. Capacity %zu reached",
                      assembler->capacity);
        assembler->status = BAL_ERROR_INSTRUCTION_OVERFLOW;
        return false;
    }

    return true;
}

void
emit8(uint8_t *BAL_RESTRICT buffer, size_t *BAL_RESTRICT offset, uint8_t const value)
{
    if (BAL_UNLIKELY(NULL == buffer))
    {
        return;
    }

    buffer += *offset;
    *buffer = value;
    *offset += 1;
}

void
emit32(uint8_t *BAL_RESTRICT buffer, size_t *BAL_RESTRICT offset, const uint32_t value)
{
    if (BAL_UNLIKELY(NULL == buffer))
    {
        return;
    }

    buffer += *offset;
    memcpy(buffer, &value, sizeof(uint32_t));
    *offset += sizeof(uint32_t);
}

void
emit64(uint8_t *BAL_RESTRICT buffer, size_t *BAL_RESTRICT offset, uint64_t const value)
{
    if (BAL_UNLIKELY(NULL == buffer))
    {
        return;
    }

    buffer += *offset;
    memcpy(buffer, &value, sizeof(uint64_t));
    *offset += sizeof(uint64_t);
}

void
emit_rex(uint8_t *BAL_RESTRICT buffer,
         size_t *BAL_RESTRICT  offset,
         const uint8_t         w,
         const uint8_t         r,
         const uint8_t         b)
{
    const uint8_t safe_w = w & 1U;
    const uint8_t safe_r = r & 1U;
    const uint8_t safe_b = b & 1U;

    // WARNING: Casting safe_* to unsigned prevents undefined behavior during bitwise shifts.
    // WARNING: Cast to uint8_t is safe as the result of the combined bitwise-OR  fits within
    // uint8_t [0x40, 0x4F].
    const uint8_t rex = (uint8_t)(0x40U | (unsigned)safe_w << 3U | (unsigned)safe_r << 2U | safe_b);

    emit8(buffer, offset, rex);
}

void
emit_shift_imm8(bal_x86_assembler_t *BAL_RESTRICT assembler,
                const bal_x86_register_t          destination,
                const uint8_t                     immediate,
                const uint8_t                     opcode_extension,
                const bool                        is_64bit,
                const char *BAL_RESTRICT          mnemonic)
{
    if (BAL_UNLIKELY(NULL == assembler))
    {
        return;
    }

    if (BAL_UNLIKELY(assembler->status != BAL_SUCCESS))
    {
        BAL_LOG_ERROR(&bal_thread_logger, "Aborting function: assembler status != BAL_SUCCESS");
        return;
    }

    if (BAL_UNLIKELY(false == is_valid_register(destination)))
    {
        BAL_LOG_ERROR(&bal_thread_logger, "Invalid destination register: %d", destination);
        assembler->status = BAL_ERROR_INVALID_ARGUMENT;
        return;
    }

    const uint8_t max_immediate = true == is_64bit ? 63U : 31U;

    if (BAL_UNLIKELY(immediate > max_immediate))
    {
        BAL_LOG_ERROR(&bal_thread_logger,
                      "Invalid %s immediate: %u (must be 0-%u)",
                      mnemonic,
                      immediate,
                      max_immediate);
        assembler->status = BAL_ERROR_INVALID_ARGUMENT;
        return;
    }

    if (BAL_UNLIKELY(opcode_extension > 7U))
    {
        BAL_LOG_ERROR(&bal_thread_logger,
                      "Invalid %s opcode extension: %u (must be 0-7)",
                      mnemonic,
                      opcode_extension);
        assembler->status = BAL_ERROR_INVALID_ARGUMENT;
        return;
    }

    const size_t instruction_size_bytes = 4;
    const bool   can_emit_status        = can_emit(assembler, instruction_size_bytes);

    if (BAL_UNLIKELY(false == can_emit_status))
    {
        return;
    }

    BAL_LOG_DEBUG(&bal_thread_logger,
                  "[+0x%04zx] %s r%d%s, %u",
                  assembler->offset,
                  mnemonic,
                  destination,
                  true == is_64bit ? "" : "d",
                  immediate);

    const uint8_t w          = true == is_64bit ? 1U : 0U;
    const uint8_t r          = 0;
    const uint8_t b          = (uint8_t)destination >> 3;
    const size_t  old_offset = assembler->offset;
    emit_rex(assembler->buffer, &assembler->offset, w, r, b);
    emit8(assembler->buffer, &assembler->offset, 0xC1);
    emit_modrm_register(assembler->buffer, &assembler->offset, (int)opcode_extension, destination);
    emit8(assembler->buffer, &assembler->offset, immediate);

    const size_t bytes_emitted = assembler->offset - old_offset;
    BAL_ASSERT_MSG(bytes_emitted == instruction_size_bytes,
                   "Bytes emitted %d does not match instruction size %d",
                   bytes_emitted,
                   instruction_size_bytes);
}

void
emit_register_register(bal_x86_assembler_t *BAL_RESTRICT assembler,
                       const uint8_t                     opcode,
                       const bool                        is_64bit,
                       const bal_x86_register_t          destination,
                       const bal_x86_register_t          source,
                       const char *BAL_RESTRICT          mnemonic)
{
    if (BAL_UNLIKELY(NULL == assembler))
    {
        return;
    }

    if (BAL_UNLIKELY(assembler->status != BAL_SUCCESS))
    {
        BAL_LOG_ERROR(&bal_thread_logger, "Aborting function: assembler status != BAL_SUCCESS");
        return;
    }

    if (BAL_UNLIKELY(false == is_valid_register(destination)))
    {
        BAL_LOG_ERROR(&bal_thread_logger, "Invalid destination register: %d", destination);
        assembler->status = BAL_ERROR_INVALID_ARGUMENT;
        return;
    }

    if (BAL_UNLIKELY(false == is_valid_register(source)))
    {
        BAL_LOG_ERROR(&bal_thread_logger, "Invalid source register: %d", source);
        assembler->status = BAL_ERROR_INVALID_ARGUMENT;
        return;
    }

    const size_t instruction_size_bytes = 3;
    const bool   can_emit_status        = can_emit(assembler, instruction_size_bytes);

    if (BAL_UNLIKELY(false == can_emit_status))
    {
        return;
    }

    const char *BAL_RESTRICT width_suffix = true == is_64bit ? "" : "d";
    BAL_LOG_DEBUG(&bal_thread_logger,
                  "[+0x%04zx] %s r%d%s, r%d%s",
                  assembler->offset,
                  mnemonic,
                  destination,
                  width_suffix,
                  source,
                  width_suffix);

    const uint8_t w          = true == is_64bit ? 1U : 0U;
    const uint8_t r          = (uint8_t)destination >> 3U;
    const uint8_t b          = (uint8_t)source >> 3U;
    const size_t  old_offset = assembler->offset;
    emit_rex(assembler->buffer, &assembler->offset, w, r, b);
    emit8(assembler->buffer, &assembler->offset, opcode);
    emit_modrm_register(assembler->buffer, &assembler->offset, destination, source);

    const size_t bytes_emitted = assembler->offset - old_offset;
    BAL_ASSERT_MSG(bytes_emitted == instruction_size_bytes,
                   "Bytes emitted %d does not match instruction size %d",
                   bytes_emitted,
                   instruction_size_bytes);
}

void
emit_memory_access(bal_x86_assembler_t *BAL_RESTRICT assembler,
                   const uint8_t                     operand_size_prefix,
                   const uint8_t                     rex_w,
                   const uint8_t *BAL_RESTRICT       opcode,
                   const size_t                      opcode_size,
                   const bal_x86_register_t          reg,
                   const bal_x86_register_t          base,
                   const char *BAL_RESTRICT          mnemonic)
{
    if (BAL_UNLIKELY(NULL == assembler))
    {
        return;
    }

    if (BAL_UNLIKELY(assembler->status != BAL_SUCCESS))
    {
        BAL_LOG_ERROR(&bal_thread_logger, "Aborting function: assembler status != BAL_SUCCESS");
        return;
    }

    if (BAL_UNLIKELY(false == is_valid_register(reg))
        || BAL_UNLIKELY(false == is_valid_register(base)))
    {
        BAL_LOG_ERROR(&bal_thread_logger, "Invalid %s operands: r%d, [r%d]", mnemonic, reg, base);
        assembler->status = BAL_ERROR_INVALID_ARGUMENT;
        return;
    }

    // [RSP] and [R12] need a SIB byte, [RBP] and [R13] a zero displacement, because their ModR/M
    // r/m encodings with mod = 00 mean something else.
    const uint32_t base_low               = (uint32_t)base & 7U;
    const bool     needs_sib              = 4U == base_low;
    const bool     needs_displacement     = 5U == base_low;
    const size_t   instruction_size_bytes = (size_t)operand_size_prefix + 1U + opcode_size + 1U
                                          + (true == needs_sib ? 1U : 0U)
                                          + (true == needs_displacement ? 1U : 0U);
    const bool can_emit_status = can_emit(assembler, instruction_size_bytes);

    if (BAL_UNLIKELY(false == can_emit_status))
    {
        return;
    }

    BAL_LOG_DEBUG(
        &bal_thread_logger, "[+0x%04zx] %s r%d, [r%d]", assembler->offset, mnemonic, reg, base);

    const size_t old_offset = assembler->offset;

    if (operand_size_prefix != 0U)
    {
        emit8(assembler->buffer, &assembler->offset, 0x66);
    }

    // The REX prefix is always emitted, which also makes the byte registers SPL, BPL, SIL and DIL
    // reachable instead of AH, CH, DH and BH.
    emit_rex(assembler->buffer, &assembler->offset, rex_w, (uint8_t)reg >> 3U, (uint8_t)base >> 3U);

    for (size_t i = 0U; i < opcode_size; ++i)
    {
        emit8(assembler->buffer, &assembler->offset, opcode[i]);
    }

    const uint32_t mod   = true == needs_displacement ? 1U : 0U;
    const uint8_t  modrm = (uint8_t)((mod << 6U) | (((uint32_t)reg & 7U) << 3U) | base_low);
    emit8(assembler->buffer, &assembler->offset, modrm);

    if (true == needs_sib)
    {
        emit8(assembler->buffer, &assembler->offset, 0x24);
    }

    if (true == needs_displacement)
    {
        emit8(assembler->buffer, &assembler->offset, 0x00);
    }

    const size_t bytes_emitted = assembler->offset - old_offset;
    BAL_ASSERT_MSG(bytes_emitted == instruction_size_bytes,
                   "Bytes emitted %d does not match instruction size %d",
                   bytes_emitted,
                   instruction_size_bytes);
}

void
emit_shift_cl(bal_x86_assembler_t *BAL_RESTRICT assembler,
              const bal_x86_register_t          destination,
              const uint8_t                     opcode_extension,
              const bool                        is_64bit,
              const char *BAL_RESTRICT          mnemonic)
{
    if (BAL_UNLIKELY(NULL == assembler))
    {
        return;
    }

    if (BAL_UNLIKELY(assembler->status != BAL_SUCCESS))
    {
        BAL_LOG_ERROR(&bal_thread_logger, "Aborting function: assembler status != BAL_SUCCESS");
        return;
    }

    if (BAL_UNLIKELY(false == is_valid_register(destination)))
    {
        BAL_LOG_ERROR(&bal_thread_logger, "Invalid destination register: %d", destination);
        assembler->status = BAL_ERROR_INVALID_ARGUMENT;
        return;
    }

    const size_t instruction_size_bytes = 3;
    const bool   can_emit_status        = can_emit(assembler, instruction_size_bytes);

    if (BAL_UNLIKELY(false == can_emit_status))
    {
        return;
    }

    BAL_LOG_DEBUG(&bal_thread_logger,
                  "[+0x%04zx] %s r%d%s, cl",
                  assembler->offset,
                  mnemonic,
                  destination,
                  true == is_64bit ? "" : "d");

    const uint8_t w          = true == is_64bit ? 1U : 0U;
    const uint8_t r          = 0U;
    const uint8_t b          = (uint8_t)destination >> 3U;
    const size_t  old_offset = assembler->offset;
    emit_rex(assembler->buffer, &assembler->offset, w, r, b);
    emit8(assembler->buffer, &assembler->offset, 0xD3);
    emit_modrm_register(assembler->buffer, &assembler->offset, (int)opcode_extension, destination);

    const size_t bytes_emitted = assembler->offset - old_offset;
    BAL_ASSERT_MSG(bytes_emitted == instruction_size_bytes,
                   "Bytes emitted %d does not match instruction size %d",
                   bytes_emitted,
                   instruction_size_bytes);
}

void
emit_register_register_0f(bal_x86_assembler_t *BAL_RESTRICT assembler,
                          const uint8_t                     opcode,
                          const bool                        is_64bit,
                          const bal_x86_register_t          destination,
                          const bal_x86_register_t          source,
                          const char *BAL_RESTRICT          mnemonic)
{
    if (BAL_UNLIKELY(NULL == assembler))
    {
        return;
    }

    if (BAL_UNLIKELY(assembler->status != BAL_SUCCESS))
    {
        BAL_LOG_ERROR(&bal_thread_logger, "Aborting function: assembler status != BAL_SUCCESS");
        return;
    }

    if (BAL_UNLIKELY(false == is_valid_register(destination)))
    {
        BAL_LOG_ERROR(&bal_thread_logger, "Invalid destination register: %d", destination);
        assembler->status = BAL_ERROR_INVALID_ARGUMENT;
        return;
    }

    if (BAL_UNLIKELY(false == is_valid_register(source)))
    {
        BAL_LOG_ERROR(&bal_thread_logger, "Invalid source register: %d", source);
        assembler->status = BAL_ERROR_INVALID_ARGUMENT;
        return;
    }

    const size_t instruction_size_bytes = 4;
    const bool   can_emit_status        = can_emit(assembler, instruction_size_bytes);

    if (BAL_UNLIKELY(false == can_emit_status))
    {
        return;
    }

    const char *BAL_RESTRICT width_suffix = true == is_64bit ? "" : "d";
    BAL_LOG_DEBUG(&bal_thread_logger,
                  "[+0x%04zx] %s(0x%02x) r%d%s, r%d%s",
                  assembler->offset,
                  mnemonic,
                  opcode,
                  destination,
                  width_suffix,
                  source,
                  width_suffix);

    const uint8_t w          = true == is_64bit ? 1U : 0U;
    const uint8_t r          = (uint8_t)destination >> 3U;
    const uint8_t b          = (uint8_t)source >> 3U;
    const size_t  old_offset = assembler->offset;
    emit_rex(assembler->buffer, &assembler->offset, w, r, b);
    emit8(assembler->buffer, &assembler->offset, 0x0F);
    emit8(assembler->buffer, &assembler->offset, opcode);
    emit_modrm_register(assembler->buffer, &assembler->offset, destination, source);

    const size_t bytes_emitted = assembler->offset - old_offset;
    BAL_ASSERT_MSG(bytes_emitted == instruction_size_bytes,
                   "Bytes emitted %d does not match instruction size %d",
                   bytes_emitted,
                   instruction_size_bytes);
}

void
emit_group3_register(bal_x86_assembler_t *BAL_RESTRICT assembler,
                     const uint8_t                     opcode_extension,
                     const bool                        is_64bit,
                     const bal_x86_register_t          destination,
                     const char *BAL_RESTRICT          mnemonic)
{
    if (BAL_UNLIKELY(NULL == assembler))
    {
        return;
    }

    if (BAL_UNLIKELY(assembler->status != BAL_SUCCESS))
    {
        BAL_LOG_ERROR(&bal_thread_logger, "Aborting function: assembler status != BAL_SUCCESS");
        return;
    }

    if (BAL_UNLIKELY(false == is_valid_register(destination)))
    {
        BAL_LOG_ERROR(&bal_thread_logger, "Invalid destination register: %d", destination);
        assembler->status = BAL_ERROR_INVALID_ARGUMENT;
        return;
    }

    const size_t instruction_size_bytes = 3;
    const bool   can_emit_status        = can_emit(assembler, instruction_size_bytes);

    if (BAL_UNLIKELY(false == can_emit_status))
    {
        return;
    }

    BAL_LOG_DEBUG(&bal_thread_logger,
                  "[+0x%04zx] %s r%d%s",
                  assembler->offset,
                  mnemonic,
                  destination,
                  true == is_64bit ? "" : "d");

    const uint8_t w          = true == is_64bit ? 1U : 0U;
    const uint8_t r          = 0U;
    const uint8_t b          = (uint8_t)destination >> 3U;
    const size_t  old_offset = assembler->offset;
    emit_rex(assembler->buffer, &assembler->offset, w, r, b);
    emit8(assembler->buffer, &assembler->offset, 0xF7);
    emit_modrm_register(assembler->buffer, &assembler->offset, (int)opcode_extension, destination);

    const size_t bytes_emitted = assembler->offset - old_offset;
    BAL_ASSERT_MSG(bytes_emitted == instruction_size_bytes,
                   "Bytes emitted %d does not match instruction size %d",
                   bytes_emitted,
                   instruction_size_bytes);
}

void
emit_modrm_register(uint8_t *BAL_RESTRICT buffer, size_t *BAL_RESTRICT offset, int reg, int rm)
{
    // WARNING: Lossless conversion of reg (0 - 15) before applying mask.
    const uint8_t safe_reg = (uint8_t)reg & 7;

    // WARNING: Lossless conversion of rm (0 - 15) before applying mask.
    const uint8_t safe_rm = (uint8_t)rm & 7U;

    // WARNING: Cast safe_reg to unsigned prevents undefined behavior during bitwise shifts.
    // WARNING: Cast to uint8_t is safe as the resulting byte fits within the uint8_t range
    // [0xC0, 0xFF].
    const uint8_t modrm = (uint8_t)(0xC0U | (unsigned)safe_reg << 3U | safe_rm);

    emit8(buffer, offset, modrm);
}

void
emit_modrm_memory_disp32_rbp(uint8_t *BAL_RESTRICT buffer, size_t *BAL_RESTRICT offset, int reg)
{
    // WARNING: Lossless conversion of reg (0 - 15) before applying mask.
    const uint8_t safe_reg = (uint8_t)reg & 7;

    // WARNING: Cast safe_reg to unsigned prevents undefined behavior during bitwise shifts.
    // WARNING: Cast to uint8_t is safe as the resulting byte fits within the uint8_t range
    // [0xC0, 0xFF].
    uint8_t const modrm = (uint8_t)(0x80U | (unsigned)safe_reg << 3U | 0x05U);

    emit8(buffer, offset, modrm);
}

/// Checks the common preconditions of the SSE and VEX emitters and reserves `size` bytes.
static bool
can_emit_vector(bal_x86_assembler_t *assembler, const size_t size, const uint32_t registers)
{
    if (BAL_UNLIKELY(NULL == assembler))
    {
        return false;
    }

    if (BAL_UNLIKELY(assembler->status != BAL_SUCCESS))
    {
        BAL_LOG_ERROR(&bal_thread_logger, "Aborting function: assembler status != BAL_SUCCESS");
        return false;
    }

    if (BAL_UNLIKELY(registers > 15U))
    {
        BAL_LOG_ERROR(&bal_thread_logger, "Invalid SSE register operand: %u", registers);
        assembler->status = BAL_ERROR_INVALID_ARGUMENT;
        return false;
    }

    return can_emit(assembler, size);
}

void
bal_x86_emit_sse_register(bal_x86_assembler_t *BAL_RESTRICT assembler,
                          const uint8_t                     prefix,
                          const bool                        is_64bit,
                          const uint8_t *BAL_RESTRICT       opcode,
                          const size_t                      opcode_size,
                          const uint8_t                     reg,
                          const uint8_t                     rm,
                          const bool                        has_immediate,
                          const uint8_t                     immediate,
                          const char *BAL_RESTRICT          mnemonic)
{
    const bool   needs_rex = true == is_64bit || reg > 7U || rm > 7U;
    const size_t size = (prefix != 0U ? 1U : 0U) + (true == needs_rex ? 1U : 0U) + opcode_size + 1U
                        + (true == has_immediate ? 1U : 0U);

    if (false == can_emit_vector(assembler, size, reg > rm ? reg : rm))
    {
        return;
    }

    BAL_LOG_DEBUG(&bal_thread_logger, "[+0x%04zx] %s %u, %u", assembler->offset, mnemonic, reg, rm);

    if (prefix != 0U)
    {
        emit8(assembler->buffer, &assembler->offset, prefix);
    }

    if (true == needs_rex)
    {
        emit_rex(assembler->buffer,
                 &assembler->offset,
                 true == is_64bit ? 1U : 0U,
                 (uint8_t)(reg >> 3U),
                 (uint8_t)(rm >> 3U));
    }

    for (size_t i = 0U; i < opcode_size; ++i)
    {
        emit8(assembler->buffer, &assembler->offset, opcode[i]);
    }

    emit_modrm_register(assembler->buffer, &assembler->offset, reg & 7U, rm & 7U);

    if (true == has_immediate)
    {
        emit8(assembler->buffer, &assembler->offset, immediate);
    }
}

void
bal_x86_emit_sse_rbp_offset(bal_x86_assembler_t *BAL_RESTRICT assembler,
                            const uint8_t                     prefix,
                            const bool                        is_64bit,
                            const uint8_t *BAL_RESTRICT       opcode,
                            const size_t                      opcode_size,
                            const uint8_t                     reg,
                            const int32_t                     offset,
                            const char *BAL_RESTRICT          mnemonic)
{
    const bool   needs_rex = true == is_64bit || reg > 7U;
    const size_t size
        = (prefix != 0U ? 1U : 0U) + (true == needs_rex ? 1U : 0U) + opcode_size + 1U + 4U;

    if (false == can_emit_vector(assembler, size, reg))
    {
        return;
    }

    BAL_LOG_DEBUG(&bal_thread_logger,
                  "[+0x%04zx] %s %u, [rbp + 0x%X]",
                  assembler->offset,
                  mnemonic,
                  reg,
                  (uint32_t)offset);

    if (prefix != 0U)
    {
        emit8(assembler->buffer, &assembler->offset, prefix);
    }

    if (true == needs_rex)
    {
        emit_rex(assembler->buffer,
                 &assembler->offset,
                 true == is_64bit ? 1U : 0U,
                 (uint8_t)(reg >> 3U),
                 0U);
    }

    for (size_t i = 0U; i < opcode_size; ++i)
    {
        emit8(assembler->buffer, &assembler->offset, opcode[i]);
    }

    emit_modrm_memory_disp32_rbp(assembler->buffer, &assembler->offset, reg & 7U);
    emit32(assembler->buffer, &assembler->offset, (uint32_t)offset);
}

void
bal_x86_emit_sse_base_offset(bal_x86_assembler_t *BAL_RESTRICT assembler,
                             const uint8_t                     prefix,
                             const bool                        is_64bit,
                             const uint8_t *BAL_RESTRICT       opcode,
                             const size_t                      opcode_size,
                             const uint8_t                     reg,
                             const bal_x86_register_t          base,
                             const int32_t                     offset,
                             const char *BAL_RESTRICT          mnemonic)
{
    const uint8_t base_number = (uint8_t)base;
    const bool    needs_rex   = true == is_64bit || reg > 7U || base_number > 7U;
    const bool    needs_sib   = 4U == (base_number & 7U); // RSP and R12 need a SIB byte.
    const size_t  size = (prefix != 0U ? 1U : 0U) + (true == needs_rex ? 1U : 0U) + opcode_size + 1U
                        + (true == needs_sib ? 1U : 0U) + 4U;

    if (false == can_emit_vector(assembler, size, reg > base_number ? reg : base_number))
    {
        return;
    }

    BAL_LOG_DEBUG(&bal_thread_logger,
                  "[+0x%04zx] %s %u, [r%u + 0x%X]",
                  assembler->offset,
                  mnemonic,
                  reg,
                  base_number,
                  (uint32_t)offset);

    if (prefix != 0U)
    {
        emit8(assembler->buffer, &assembler->offset, prefix);
    }

    if (true == needs_rex)
    {
        emit_rex(assembler->buffer,
                 &assembler->offset,
                 true == is_64bit ? 1U : 0U,
                 (uint8_t)(reg >> 3U),
                 (uint8_t)(base_number >> 3U));
    }

    for (size_t i = 0U; i < opcode_size; ++i)
    {
        emit8(assembler->buffer, &assembler->offset, opcode[i]);
    }

    // mod = 10 (disp32), reg, rm = base.
    emit8(assembler->buffer,
          &assembler->offset,
          (uint8_t)(0x80U | (unsigned)(reg & 7U) << 3U | (base_number & 7U)));

    if (true == needs_sib)
    {
        emit8(assembler->buffer, &assembler->offset, 0x24);
    }

    emit32(assembler->buffer, &assembler->offset, (uint32_t)offset);
}

void
bal_x86_emit_vex_0f38_register(bal_x86_assembler_t *BAL_RESTRICT assembler,
                               const bool                        is_64bit,
                               const uint8_t                     opcode,
                               const uint8_t                     reg,
                               const uint8_t                     vvvv,
                               const uint8_t                     rm,
                               const char *BAL_RESTRICT          mnemonic)
{
    const uint8_t largest = reg > rm ? (reg > vvvv ? reg : vvvv) : (rm > vvvv ? rm : vvvv);

    if (false == can_emit_vector(assembler, 5U, largest))
    {
        return;
    }

    BAL_LOG_DEBUG(&bal_thread_logger,
                  "[+0x%04zx] %s xmm%u, xmm%u, xmm%u",
                  assembler->offset,
                  mnemonic,
                  reg,
                  vvvv,
                  rm);

    // C4 [R X B mmmmm = 00010] [W vvvv L pp = 01]: R, X, B and vvvv are stored inverted.
    const uint32_t byte1 = ((~(uint32_t)reg >> 3U & 1U) << 7U) | (1U << 6U)
                           | ((~(uint32_t)rm >> 3U & 1U) << 5U) | 0x02U;
    const uint32_t byte2
        = ((true == is_64bit ? 1U : 0U) << 7U) | ((~(uint32_t)vvvv & 0xFU) << 3U) | 0x01U;
    emit8(assembler->buffer, &assembler->offset, 0xC4);
    emit8(assembler->buffer, &assembler->offset, (uint8_t)byte1);
    emit8(assembler->buffer, &assembler->offset, (uint8_t)byte2);
    emit8(assembler->buffer, &assembler->offset, opcode);
    emit_modrm_register(assembler->buffer, &assembler->offset, reg & 7U, rm & 7U);
}

void
bal_x86_emit_vex_0f3a_register_immediate(bal_x86_assembler_t *BAL_RESTRICT assembler,
                                         const bool                        is_64bit,
                                         const uint8_t                     opcode,
                                         const uint8_t                     reg,
                                         const uint8_t                     vvvv,
                                         const uint8_t                     rm,
                                         const uint8_t                     immediate,
                                         const char *BAL_RESTRICT          mnemonic)
{
    const uint8_t largest = reg > rm ? (reg > vvvv ? reg : vvvv) : (rm > vvvv ? rm : vvvv);

    if (false == can_emit_vector(assembler, 6U, largest))
    {
        return;
    }

    BAL_LOG_DEBUG(&bal_thread_logger,
                  "[+0x%04zx] %s xmm%u, xmm%u, %u",
                  assembler->offset,
                  mnemonic,
                  rm,
                  reg,
                  immediate);

    // C4 [R X B mmmmm = 00011] [W vvvv L pp = 01]: R, X, B and vvvv are stored inverted.
    const uint32_t byte1 = ((~(uint32_t)reg >> 3U & 1U) << 7U) | (1U << 6U)
                           | ((~(uint32_t)rm >> 3U & 1U) << 5U) | 0x03U;
    const uint32_t byte2
        = ((true == is_64bit ? 1U : 0U) << 7U) | ((~(uint32_t)vvvv & 0xFU) << 3U) | 0x01U;
    emit8(assembler->buffer, &assembler->offset, 0xC4);
    emit8(assembler->buffer, &assembler->offset, (uint8_t)byte1);
    emit8(assembler->buffer, &assembler->offset, (uint8_t)byte2);
    emit8(assembler->buffer, &assembler->offset, opcode);
    emit_modrm_register(assembler->buffer, &assembler->offset, reg & 7U, rm & 7U);
    emit8(assembler->buffer, &assembler->offset, immediate);
}

void
bal_x86_emit_vex(bal_x86_assembler_t *BAL_RESTRICT assembler,
                 const uint8_t                     map,
                 const uint8_t                     prefix,
                 const bool                        is_256,
                 const bool                        is_64bit,
                 const uint8_t                     opcode,
                 const uint8_t                     reg,
                 const uint8_t                     vvvv,
                 const uint8_t                     rm,
                 const bool                        has_immediate,
                 const uint8_t                     immediate,
                 const char *BAL_RESTRICT          mnemonic)
{
    const uint8_t largest = reg > rm ? (reg > vvvv ? reg : vvvv) : (rm > vvvv ? rm : vvvv);

    if (false == can_emit_vector(assembler, true == has_immediate ? 6U : 5U, largest))
    {
        return;
    }

    BAL_LOG_DEBUG(
        &bal_thread_logger, "[+0x%04zx] %s %u, %u, %u", assembler->offset, mnemonic, reg, vvvv, rm);

    // C4 [R X B mmmmm] [W vvvv L pp]: R, X, B and vvvv are stored inverted.
    const uint32_t byte1 = ((~(uint32_t)reg >> 3U & 1U) << 7U) | (1U << 6U)
                           | ((~(uint32_t)rm >> 3U & 1U) << 5U) | (map & 0x1FU);
    const uint32_t byte2 = ((true == is_64bit ? 1U : 0U) << 7U) | ((~(uint32_t)vvvv & 0xFU) << 3U)
                           | ((true == is_256 ? 1U : 0U) << 2U) | (prefix & 3U);
    emit8(assembler->buffer, &assembler->offset, 0xC4);
    emit8(assembler->buffer, &assembler->offset, (uint8_t)byte1);
    emit8(assembler->buffer, &assembler->offset, (uint8_t)byte2);
    emit8(assembler->buffer, &assembler->offset, opcode);
    emit_modrm_register(assembler->buffer, &assembler->offset, reg & 7U, rm & 7U);

    if (true == has_immediate)
    {
        emit8(assembler->buffer, &assembler->offset, immediate);
    }
}

void
bal_x86_emit_vzeroupper(bal_x86_assembler_t *BAL_RESTRICT assembler)
{
    if (false == can_emit_vector(assembler, 3U, 0U))
    {
        return;
    }

    BAL_LOG_DEBUG(&bal_thread_logger, "[+0x%04zx] vzeroupper", assembler->offset);
    emit8(assembler->buffer, &assembler->offset, 0xC5);
    emit8(assembler->buffer, &assembler->offset, 0xF8);
    emit8(assembler->buffer, &assembler->offset, 0x77);
}

/*** end of file ***/