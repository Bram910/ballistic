#include "backend/x86/bal_x86_tier1_compiler.h"
#include "backend/bal_cpu.h"
#include "bal_decoder.h"
#include "bal_engine.h"
#include "bal_engine_flags.h"
#include "bal_platform.h"

#include <string.h>

// Set which registers holds the first argument passed to the JIT block based on the OS calling
// convention.
#if BAL_PLATFORM_WINDOWS

#define BAL_X86_ABI_ARG1 BAL_X86_RCX

#else

#define BAL_X86_ABI_ARG1 BAL_X86_RDI

#endif

// Scratch registers for the Greedy Allocator. We avoid RDI and RSI since they are often used for
// the argument passing.
//
// Note that R11 was intentionally excluded from this. See [`BAL_X86_REGISTER_DISCARD_RESULT`].
static const bal_x86_register_t SCRATCH_REGISTERS[]
    = { BAL_X86_RAX, BAL_X86_RCX, BAL_X86_RDX, BAL_X86_R8, BAL_X86_R9,
        BAL_X86_R10, BAL_X86_R12, BAL_X86_R14, BAL_X86_R15 };

#define SCRATCH_REGISTERS_SIZE (sizeof(SCRATCH_REGISTERS) / sizeof(SCRATCH_REGISTERS[0]))

/// Dedicated throwaway register for instructions that update flags but discards the result.
#define BAL_X86_REGISTER_DISCARD_RESULT BAL_X86_R11

/// Holds zero whenever an instruction reads the zero register (WZR/XZR). R13 is saved by the block
/// prologue but is not part of the scratch pool, so it is never mapped to a guest register.
#define BAL_X86_REGISTER_ZERO BAL_X86_R13

static void reset_register_allocator(bal_tier1_compiler_t *compiler);

static bal_x86_register_t allocate_x86_register(bal_tier1_compiler_t *compiler,
                                                uint8_t               arm_register,
                                                bool                  skip_load_instruction);

static void evict_x86_register(bal_tier1_compiler_t *compiler, bal_x86_register_t x86_register);

static bal_x86_register_t read_zero_or_register(bal_tier1_compiler_t *compiler,
                                                uint8_t               arm_register);

static void flush_dirty_registers(bal_tier1_compiler_t *compiler);

static void terminate_block(bal_tier1_compiler_t *compiler,
                            bal_guest_address_t   next_pc,
                            bool                  is_pc_dynamic,
                            size_t                arm_instruction_count,
                            uint32_t              engine_flags);

static void terminate_block_conditional(bal_tier1_compiler_t *compiler,
                                        bal_x86_condition_t   condition,
                                        bal_guest_address_t   fallthrough_pc,
                                        bal_guest_address_t   taken_pc);

static void emit_epilogue(bal_x86_assembler_t *assembler);

static uint32_t extract_operand_value(uint32_t instruction, const bal_decoder_operand_t *operand);

static void translate_add_sub_imm(bal_tier1_compiler_t                     *compiler,
                                  const bal_decoder_instruction_metadata_t *metadata,
                                  uint32_t                                  instruction,
                                  bool                                      is_sub);

static bal_x86_register_t read_shifted_register(bal_tier1_compiler_t *compiler,
                                                uint8_t               arm_register,
                                                uint32_t              shift_type,
                                                uint32_t              shift_amount,
                                                bool                  is_32bit,
                                                bool                  is_inverted);

static void emit_binary_operation(bal_tier1_compiler_t  *compiler,
                                  bal_x86_macro_opcode_t opcode,
                                  bool                   is_commutative,
                                  bal_x86_register_t     x86_result,
                                  bool                   is_rn_zero,
                                  bal_x86_register_t     x86_rn,
                                  bal_x86_register_t     x86_operand2);

static void capture_nzcv(bal_tier1_compiler_t *compiler, bal_x86_condition_t carry_condition);

static bool decode_bit_masks(
    uint32_t n, uint32_t imms, uint32_t immr, bool is_32bit, uint64_t *mask);

static void translate_add_sub_shifted(bal_tier1_compiler_t *compiler, uint32_t instruction);

static void translate_logical(bal_tier1_compiler_t *compiler, uint32_t instruction);

static void translate_b_cond(bal_tier1_compiler_t *compiler,
                             uint32_t              instruction,
                             bal_guest_address_t   guest_address);

static void emit_compound_condition(bal_tier1_compiler_t *compiler, uint32_t condition);

static void translate_cbz_cbnz(bal_tier1_compiler_t                     *compiler,
                               const bal_decoder_instruction_metadata_t *metadata,
                               uint32_t                                  instruction,
                               bal_guest_address_t                       guest_address,
                               bool                                      is_cbnz);

static void translate_jump(const bal_tier1_compiler_t               *compiler,
                           const bal_decoder_instruction_metadata_t *metadata,
                           const uint32_t                            instruction,
                           const bal_guest_address_t                 guest_address,
                           bal_guest_address_t                      *target_pc);

static void translate_mov_immediate(bal_tier1_compiler_t                     *compiler,
                                    const bal_decoder_instruction_metadata_t *metadata,
                                    uint32_t                                  instruction);

bal_error_t
bal_tier1_compiler_init(bal_tier1_compiler_t         *compiler,
                        const bal_executable_buffer_t executable_buffer,
                        const size_t                  buffer_size,
                        bal_jit_debug_context_t      *debug_context)
{
    if (BAL_UNLIKELY(NULL == compiler))
    {
        BAL_LOG_ERROR(&bal_thread_logger, "Aborting function: compiler is NULL");
        return BAL_ERROR_INVALID_ARGUMENT;
    }

    if (BAL_UNLIKELY(NULL == executable_buffer.rw_pointer)
        || BAL_UNLIKELY(NULL == executable_buffer.rx_pointer))
    {
        BAL_LOG_ERROR(&bal_thread_logger, "Aborting function: executable_buffer is NULL");
        compiler->status = BAL_ERROR_INVALID_ARGUMENT;
        return compiler->status;
    }

    if (BAL_UNLIKELY(0 == buffer_size))
    {
        BAL_LOG_ERROR(&bal_thread_logger, "Aborting function: buffer_size is 0");
        compiler->status = BAL_ERROR_INVALID_ARGUMENT;
        return compiler->status;
    }

    compiler->debug_context = debug_context;
    bal_error_t error
        = bal_x86_assembler_init(&compiler->assembler, executable_buffer, buffer_size);

    if (BAL_UNLIKELY(error != BAL_SUCCESS))
    {
        return error;
    }

    error = bal_sliding_window_init(&compiler->window, &compiler->assembler);

    if (BAL_UNLIKELY(error != BAL_SUCCESS))
    {
        return error;
    }

    return BAL_SUCCESS;
}

void
bal_tier1_compiler_reset(bal_tier1_compiler_t *compiler)
{
    if (BAL_UNLIKELY(NULL == compiler))
    {
        return;
    }

    compiler->status = BAL_SUCCESS;
    reset_register_allocator(compiler);
    bal_sliding_window_reset(&compiler->window);
    bal_x86_assembler_reset(&compiler->assembler);
}

void *
bal_tier1_compiler_translate(bal_tier1_compiler_t         *compiler,
                             const bal_memory_interface_t *memory_interface,
                             bal_guest_address_t           guest_address,
                             const size_t                  max_instructions,
                             const uint32_t                engine_flags)
{
    if (BAL_UNLIKELY(NULL == compiler))
    {
        return NULL;
    }

    if (BAL_UNLIKELY(NULL == memory_interface))
    {
        BAL_LOG_ERROR(&bal_thread_logger, "Aborting function: memory_interface is NULL");
        compiler->status = BAL_ERROR_INVALID_ARGUMENT;
        return NULL;
    }

    if (BAL_UNLIKELY(NULL == memory_interface->translate))
    {
        BAL_LOG_ERROR(&bal_thread_logger, "Aborting function: memory_interface->translate is NULL");
        compiler->status = BAL_ERROR_INVALID_ARGUMENT;
        return NULL;
    }

    if (BAL_UNLIKELY(0 == max_instructions))
    {
        BAL_LOG_INFO(&bal_thread_logger, "Aborting function: max Instructions is 0");
        compiler->status = BAL_ERROR_INVALID_ARGUMENT;
        return NULL;
    }

    BAL_LOG_INFO(&bal_thread_logger, "Starting Tier 1 JIT Compilation");
    BAL_LOG_INFO(&bal_thread_logger,
                 "GVA: 0x%016llX | Max Instructions: %zu",
                 (unsigned long long)guest_address,
                 max_instructions);
    reset_register_allocator(compiler);
    void *host_address = compiler->assembler.rx_buffer + compiler->assembler.offset;

    // Debug tracking.
    const bal_guest_address_t guest_address_base           = guest_address;
    const size_t              start_x86_block_offset       = compiler->assembler.offset;
    size_t                    start_x86_instruction_offset = 0;
    bal_jit_instruction_map_t mappings[2048];
    uint32_t                  mapping_count = 0;
    const bool                debug_enabled = engine_flags & BAL_ENGINE_FLAG_JIT_DEBUG
                               && compiler->debug_context != NULL
                               && compiler->debug_context->entries != NULL;

    // Setup block.
    //
    bal_x86_emit_push_r64(&compiler->assembler, BAL_X86_RBP);
    bal_x86_emit_push_r64(&compiler->assembler, BAL_X86_R12);
    bal_x86_emit_push_r64(&compiler->assembler, BAL_X86_R13);
    bal_x86_emit_push_r64(&compiler->assembler, BAL_X86_R14);
    bal_x86_emit_push_r64(&compiler->assembler, BAL_X86_R15);
    bal_x86_emit_mov_r64_r64(&compiler->assembler, BAL_X86_RBP, BAL_X86_ABI_ARG1);

    bool   is_block_terminated   = false;
    bool   is_pc_dynamic         = false;
    size_t arm_instruction_count = 0;

    // Set by unconditional branches, whose target replaces the address following the block.
    bool                has_branch_target = false;
    bal_guest_address_t branch_target_pc  = 0;

    while (false == is_block_terminated)
    {
        size_t          max_readable_instructions_bytes = 0;
        const uint32_t *host_address_base = (const uint32_t *)memory_interface->translate(
            (void *)memory_interface, guest_address, &max_readable_instructions_bytes);

        if (BAL_UNLIKELY(NULL == host_address_base))
        {
            BAL_LOG_ERROR(&bal_thread_logger,
                          "Aborting function: memory translation fault at GVA 0x%016llX",
                          (unsigned long long)guest_address);
            compiler->status = BAL_ERROR_MEMORY_FAULT;
            break;
        }

        const uint32_t *BAL_RESTRICT host_address_cursor = host_address_base;
        size_t max_readable_instructions = max_readable_instructions_bytes / sizeof(uint32_t);

        // Has the remaining instructions crossed a memory page boundary? This should not happen on
        // ARM64 since instructions are strictly 4 byte aligned. So throw an error if this
        // happens.
        //
        if (BAL_UNLIKELY(0 == max_readable_instructions))
        {
            BAL_LOG_ERROR(&bal_thread_logger,
                          "Aborting function: insufficient memory at GVA 0x%016llX. Need 4 bytes "
                          "for an instruction, but only %zu bytes are readable",
                          (unsigned long long)guest_address,
                          max_readable_instructions_bytes);
            compiler->status = BAL_ERROR_PC_ALIGNMENT;
            break;
        }

        if (max_readable_instructions > max_instructions)
        {
            max_readable_instructions = max_instructions;
        }

        if (engine_flags & BAL_ENGINE_FLAG_STRICT_ALIGNMENT)
        {
            if (guest_address % 4 != 0)
            {
                BAL_LOG_ERROR(&bal_thread_logger,
                              "Aborting function: strict alignment fault at GVA 0x%016llX",
                              (unsigned long)guest_address);
                compiler->status = BAL_ERROR_PC_ALIGNMENT;
                break;
            }
        }

        for (size_t i = 0; i < max_readable_instructions; ++i)
        {
            const uint32_t                            instruction = *host_address_cursor;
            const bal_decoder_instruction_metadata_t *metadata    = bal_decode_arm64(instruction);

            if (BAL_UNLIKELY(NULL == metadata))
            {
                BAL_LOG_ERROR(&bal_thread_logger,
                              "Aborting function: failed to decode instruction %08X at GVA 0x%llX",
                              instruction,
                              (unsigned long long)guest_address);
                is_block_terminated = false;
                break;
            }

            if (BAL_OPERAND_TYPE_REGISTER_128 == metadata->operands[0].type)
            {
                BAL_LOG_DEBUG(&bal_thread_logger,
                              "Aborting function: Unsupported SIMD instruction: %s",
                              metadata->name);
                is_block_terminated = true;
                compiler->status    = BAL_ERROR_UNKNOWN_INSTRUCTION;
                break;
            }

            BAL_LOG_TRACE(&bal_thread_logger,
                          "[0X%016llX] %08x : %s",
                          (unsigned long long)guest_address,
                          instruction,
                          metadata->name);

            if (engine_flags & BAL_ENGINE_FLAG_TRAP_SVC)
            {
                if (BAL_UNLIKELY(0 == strncmp(metadata->name, "SVC", 3)))
                {
                    BAL_LOG_INFO(&bal_thread_logger,
                                 "Trapping on SVC instruction at 0x%016llX",
                                 (unsigned long long)guest_address);
                    is_block_terminated = true;
                    break;
                }
            }

            if (engine_flags & BAL_ENGINE_FLAG_TRAP_SMC_HVC)
            {
                if (BAL_UNLIKELY((0 == strncmp(metadata->name, "SMC", 3)
                                  || (strncmp(metadata->name, "HVC", 3) == 0))))
                {
                    BAL_LOG_INFO(&bal_thread_logger,
                                 "Trapping on SMC/HVC instruction at 0x%016llX",
                                 (unsigned long long)guest_address);
                    is_block_terminated = true;
                    break;
                }
            }

            if (engine_flags & BAL_ENGINE_FLAG_WFI_YIELDS_HOST)
            {
                if (BAL_UNLIKELY(0 == strncmp(metadata->name, "WFI", 3)))
                {
                    BAL_LOG_INFO(&bal_thread_logger,
                                 "WFI instruction encountered at 0x%016llX, yielding to host",
                                 (unsigned long long)guest_address);
                    is_block_terminated = true;
                    break;
                }
            }

            start_x86_instruction_offset = compiler->assembler.offset;

            switch (metadata->ir_opcode)
            {
                case OPCODE_MOV:;
                    const char variant = metadata->name[3];

                    if (BAL_LIKELY(variant == 'N') || BAL_LIKELY(variant == 'K')
                        || BAL_LIKELY(variant == 'Z'))
                    {
                        translate_mov_immediate(compiler, metadata, instruction);
                        break;
                    }

                    BAL_LOG_ERROR(&bal_thread_logger,
                                  "Aborting function: Invalid MOV instruction detected at %08x : "
                                  "%s",
                                  instruction,
                                  metadata->name);
                    is_block_terminated = true;
                    compiler->status    = BAL_ERROR_UNKNOWN_INSTRUCTION;
                    break;
                case OPCODE_ADD:
                case OPCODE_SUB:
                case OPCODE_CMP:;
                    // The decoder table also lists the alias encodings (CMP, CMN, NEG, ...)
                    // with fewer operands, so the encoding class is taken from the
                    // instruction bits rather than from the operand layout.
                    //
                    // ADD/SUB (shifted register): bits 28:24 = 01011 and bit 21 = 0.
                    if (0x0BU == ((instruction >> 24U) & 0x1FU)
                        && 0U == ((instruction >> 21U) & 1U))
                    {
                        translate_add_sub_shifted(compiler, instruction);
                        break;
                    }

                    // ADD/SUB (immediate): bits 28:23 = 100010.
                    if (0x22U == ((instruction >> 23U) & 0x3FU))
                    {
                        const bool is_sub = (OPCODE_SUB == metadata->ir_opcode)
                                            || (OPCODE_CMP == metadata->ir_opcode);
                        translate_add_sub_imm(compiler, metadata, instruction, is_sub);
                        break;
                    }

                    BAL_LOG_ERROR(&bal_thread_logger,
                                  "Aborting function: Tier 1 unsupported opcode variant detected "
                                  "at %08x : %s",
                                  instruction,
                                  metadata->name);
                    is_block_terminated = true;
                    compiler->status    = BAL_ERROR_UNKNOWN_INSTRUCTION;
                    break;
                case OPCODE_JUMP:
                    translate_jump(
                        compiler, metadata, instruction, guest_address, &branch_target_pc);
                    has_branch_target   = true;
                    is_block_terminated = true;
                    break;
                case OPCODE_BRANCH_ZERO:
                case OPCODE_BRANCH_NOT_ZERO:;
                    translate_cbz_cbnz(compiler,
                                       metadata,
                                       instruction,
                                       guest_address,
                                       OPCODE_BRANCH_NOT_ZERO == metadata->ir_opcode);
                    is_block_terminated = true;
                    break;
                case OPCODE_BRANCH_CONDITIONAL:;
                    translate_b_cond(compiler, instruction, guest_address);
                    is_block_terminated = true;
                    break;
                case OPCODE_RETURN:
                    BAL_LOG_DEBUG(&bal_thread_logger, "Block terminated by RET");
                    const uint8_t rn
                        = (uint8_t)extract_operand_value(instruction, &metadata->operands[0]);
                    const bool               skip_load_instruction = false;
                    const bal_x86_register_t x86_rn
                        = allocate_x86_register(compiler, rn, skip_load_instruction);

                    const bal_x86_macro_t store_pc_macro
                        = { .opcode              = BAL_X86_MACRO_STORE,
                            .source              = x86_rn,
                            .immediate_or_offset = offsetof(bal_cpu_t, pc) };
                    bal_sliding_window_push(&compiler->window, store_pc_macro);

                    is_pc_dynamic       = true;
                    is_block_terminated = true;
                    break;
                case OPCODE_TRAP:
                    BAL_LOG_INFO(&bal_thread_logger,
                                 "Block terminated by TRAP/Unknown instruction: %s: ",
                                 metadata->name);
                    const bal_x86_macro_t ud2_macro = {
                        .opcode = BAL_X86_MACRO_UD2,
                    };

                    // Sync guest state before the CPU faults.
                    bal_sliding_window_flush_all(&compiler->window);

                    bal_sliding_window_push(&compiler->window, ud2_macro);
                    compiler->status    = BAL_ERROR_UNKNOWN_INSTRUCTION;
                    is_block_terminated = true;
                    break;
                case OPCODE_AND:
                case OPCODE_OR:
                case OPCODE_OR_NOT:
                case OPCODE_XOR:
                    // Logical (shifted register): bits 28:24 = 01010.
                    // Logical (immediate): bits 28:23 = 100100.
                    if (0x0AU == ((instruction >> 24U) & 0x1FU)
                        || 0x24U == ((instruction >> 23U) & 0x3FU))
                    {
                        translate_logical(compiler, instruction);
                        break;
                    }

                    BAL_LOG_ERROR(&bal_thread_logger,
                                  "Aborting function: Tier 1 unsupported logical instruction "
                                  "at %08x : %s",
                                  instruction,
                                  metadata->name);
                    compiler->status    = BAL_ERROR_UNKNOWN_INSTRUCTION;
                    is_block_terminated = true;
                    break;
                default:
                    BAL_LOG_ERROR(&bal_thread_logger,
                                  " Aborting function: Tier 1 Unsupported Opcode: %08x : %s ",
                                  instruction,
                                  metadata->name);
                    compiler->status    = BAL_ERROR_UNKNOWN_INSTRUCTION;
                    is_block_terminated = true;
                    break;
            }

            if (BAL_UNLIKELY(compiler->status != BAL_SUCCESS
                             || compiler->assembler.status != BAL_SUCCESS))
            {
                if (BAL_SUCCESS == compiler->status)
                {
                    compiler->status = compiler->assembler.status;
                }

                BAL_LOG_ERROR(
                    &bal_thread_logger, "Status failure during translation: %d", compiler->status);
                is_block_terminated = true;
                break;
            }

            if (true == debug_enabled && mapping_count < 2048)
            {
                mappings[mapping_count].x86_offset
                    = (uint32_t)(start_x86_instruction_offset - start_x86_block_offset);
                mappings[mapping_count].guest_pc_offset
                    = (uint32_t)(guest_address - guest_address_base);
                ++mapping_count;
            }

            ++arm_instruction_count;
            guest_address += 4;
            ++host_address_cursor;

            if (i + 1 == max_instructions)
            {
                BAL_LOG_WARN(&bal_thread_logger,
                             "Max instructions limit reached, terminating block");
                is_block_terminated = true;
                break;
            }

            if (true == is_block_terminated)
            {
                break;
            }
        }
    }

    const bal_guest_address_t next_pc
        = true == has_branch_target ? branch_target_pc : guest_address;
    terminate_block(compiler, next_pc, is_pc_dynamic, arm_instruction_count, engine_flags);

    if (BAL_UNLIKELY(compiler->status != BAL_SUCCESS || compiler->assembler.status != BAL_SUCCESS))
    {
        const bal_error_t status
            = compiler->status == BAL_SUCCESS ? compiler->assembler.status : compiler->status;
        BAL_LOG_ERROR(&bal_thread_logger,
                      "Aborting function: block assembly was truncated due to error: %s",
                      bal_error_to_string(status));
        return NULL;
    }

    if (true == debug_enabled && mapping_count > 0)
    {
        void *BAL_RESTRICT rx_start = compiler->assembler.rx_buffer + start_x86_block_offset;
        const uint32_t rx_size = (uint32_t)(compiler->assembler.offset - start_x86_block_offset);
        bal_jit_debug_add_block(compiler->debug_context,
                                rx_start,
                                rx_size,
                                guest_address_base,
                                mappings,
                                mapping_count);
    }

    BAL_LOG_INFO(&bal_thread_logger,
                 "Tier 1 compiled block ends at GVA 0x%016llX",
                 (unsigned long long)guest_address);
    return host_address;
}

static void
reset_register_allocator(bal_tier1_compiler_t *compiler)
{
    if (BAL_UNLIKELY(NULL == compiler))
    {
        return;
    }

    BAL_LOG_TRACE(&bal_thread_logger, "Resetting Register Allocator");
    compiler->is_dirty        = 0U;
    compiler->eviction_cursor = 0U;
    (void)memset(compiler->arm_to_x86, -1, sizeof(compiler->arm_to_x86));
    (void)memset(compiler->x86_to_arm, -1, sizeof(compiler->x86_to_arm));
}

static bal_x86_register_t
allocate_x86_register(bal_tier1_compiler_t *BAL_RESTRICT compiler,
                      uint8_t                            arm_register,
                      const bool                         skip_load_instruction)
{
    if (BAL_UNLIKELY(NULL == compiler))
    {
        BAL_LOG_ERROR(&bal_thread_logger, "Aborting function: compiler is NULL");
        return BAL_X86_INVALID;
    }

    if (BAL_UNLIKELY(compiler->status != BAL_SUCCESS))
    {
        BAL_LOG_ERROR(&bal_thread_logger, "Aborting function, compiler->status != BAL_SUCCESS.");
        return BAL_X86_INVALID;
    }

    if (BAL_UNLIKELY(compiler->assembler.status != BAL_SUCCESS))
    {
        BAL_LOG_ERROR(&bal_thread_logger, "Aborting function: assembler status != BAL_SUCCESS.");
        return BAL_X86_INVALID;
    }

    if (BAL_UNLIKELY(arm_register >= 32U))
    {
        BAL_LOG_ERROR(&bal_thread_logger,
                      "Aborting function: ARM register index %u is out of "
                      "range (0-31).",
                      arm_register);
        compiler->status = BAL_ERROR_INVALID_ARGUMENT;
        return BAL_X86_INVALID;
    }

    if (compiler->arm_to_x86[arm_register] != -1)
    {
        const int8_t cached_x86_register = compiler->arm_to_x86[arm_register];

        if (BAL_UNLIKELY(cached_x86_register < 0 || cached_x86_register >= (int8_t)BAL_X86_R15))
        {
            BAL_LOG_ERROR(&bal_thread_logger,
                          "RegAlloc: corrupted mapping ARM X%u -> x86 r%d. "
                          "Valid range is 0-r%d. Resetting X%u.",
                          arm_register,
                          cached_x86_register,
                          (int)BAL_X86_R15,
                          arm_register);
            compiler->arm_to_x86[arm_register] = -1;
            compiler->status                   = BAL_ERROR_REGISTER_ALLOCATOR_CORRUPTED;
            return BAL_X86_INVALID;
        }

        if (BAL_UNLIKELY(compiler->x86_to_arm[cached_x86_register] != (int8_t)arm_register))
        {
            BAL_LOG_ERROR(&bal_thread_logger,
                          "RegAlloc: reverse x86 to ARM mapping inconsistency ARM X%u -> x86 r%d,"
                          " resetting both X%u and r%d",
                          arm_register,
                          cached_x86_register);
            compiler->arm_to_x86[arm_register]        = -1;
            compiler->x86_to_arm[cached_x86_register] = -1;
            compiler->status                          = BAL_ERROR_REGISTER_ALLOCATOR_CORRUPTED;
            return BAL_X86_INVALID;
        }

        BAL_LOG_DEBUG(&bal_thread_logger,
                      "RegAlloc: Hit - ARM X%u is already in x86 r%d",
                      arm_register,
                      cached_x86_register);
        return (bal_x86_register_t)cached_x86_register;
    }

    bal_x86_register_t                     free_register           = BAL_X86_INVALID;
    int8_t *BAL_RESTRICT                   x86_to_arm_cursor       = compiler->x86_to_arm;
    const bal_x86_register_t *BAL_RESTRICT scratch_register_cursor = SCRATCH_REGISTERS;

    for (size_t i = 0U; i < SCRATCH_REGISTERS_SIZE; ++i)
    {
        if (BAL_UNLIKELY(*scratch_register_cursor < 0
                         || *scratch_register_cursor > (int8_t)BAL_X86_R15))
        {
            BAL_LOG_ERROR(&bal_thread_logger,
                          "RegAlloc: scratch pool contains invalid entry %d "
                          "at index %zu",
                          *scratch_register_cursor,
                          i);
            compiler->status = BAL_ERROR_REGISTER_ALLOCATOR_CORRUPTED;
            return BAL_X86_INVALID;
        }

        if (-1 == x86_to_arm_cursor[*scratch_register_cursor])
        {
            free_register = *scratch_register_cursor;
            break;
        }

        ++scratch_register_cursor;
    }

    // Evict a clean register.
    if (BAL_X86_INVALID == free_register)
    {
        for (size_t i = 0U; i < SCRATCH_REGISTERS_SIZE; ++i)
        {
            const int8_t arm_victim_register = x86_to_arm_cursor[i];

            if (arm_victim_register != -1 && !(compiler->is_dirty & (1U << arm_victim_register)))
            {
                free_register = (bal_x86_register_t)i;
            }
        }

        // All registers are dirty, force an eviction.
        if (BAL_X86_INVALID == free_register)
        {
            const uint8_t slot_index = compiler->eviction_cursor % SCRATCH_REGISTERS_SIZE;
            free_register            = SCRATCH_REGISTERS[slot_index];
            ++compiler->eviction_cursor;
        }

        evict_x86_register(compiler, free_register);
    }

    compiler->arm_to_x86[arm_register]  = (int8_t)free_register;
    compiler->x86_to_arm[free_register] = (int8_t)arm_register;

    if (false == skip_load_instruction)
    {
        BAL_LOG_DEBUG(&bal_thread_logger,
                      "RegAlloc: Miss - mapped ARM X%u to x86 r%d",
                      arm_register,
                      free_register);
        const uint64_t        offset              = offsetof(bal_cpu_t, x[arm_register]);
        const bal_x86_macro_t load_register_macro = {
            .opcode              = BAL_X86_MACRO_LOAD,
            .destination         = free_register,
            .immediate_or_offset = offset,
        };

        bal_sliding_window_push(&compiler->window, load_register_macro);
        BAL_LOG_DEBUG(&bal_thread_logger, "RegAlloc: emitted LOAD macro for ARM X%u", arm_register);
    }
    else
    {
        BAL_LOG_DEBUG(&bal_thread_logger,
                      "RegAlloc: skipped LOAD macro for ARM X%u (write only)",
                      arm_register);
    }
    return free_register;
}

static void
evict_x86_register(bal_tier1_compiler_t *compiler, const bal_x86_register_t x86_register)
{
    const int8_t arm_victim = compiler->x86_to_arm[x86_register];

    if (BAL_UNLIKELY(-1 == arm_victim))
    {
        return;
    }

    const uint32_t dirty_bit = 1U << arm_victim;

    if (compiler->is_dirty & dirty_bit)
    {
        const uint64_t        offset      = offsetof(bal_cpu_t, x[arm_victim]);
        const bal_x86_macro_t store_macro = {
            .opcode              = BAL_X86_MACRO_STORE,
            .source              = x86_register,
            .immediate_or_offset = offset,
        };
        bal_sliding_window_push(&compiler->window, store_macro);
        compiler->is_dirty &= ~dirty_bit;
    }

    compiler->arm_to_x86[arm_victim]   = -1;
    compiler->x86_to_arm[x86_register] = -1;
    BAL_LOG_DEBUG(
        &bal_thread_logger, "RegAlloc: evicted ARM X%d from x86 r%d", arm_victim, x86_register);
}

/// Returns an x86 register holding the value of a source operand where register 31 encodes the zero
/// register rather than the stack pointer.
static bal_x86_register_t
read_zero_or_register(bal_tier1_compiler_t *compiler, const uint8_t arm_register)
{
    if (31U == arm_register)
    {
        const bal_x86_macro_t zero_macro = {
            .opcode              = BAL_X86_MACRO_MOV_REGISTER_IMMEDIATE,
            .destination         = BAL_X86_REGISTER_ZERO,
            .immediate_or_offset = 0,
        };
        bal_sliding_window_push(&compiler->window, zero_macro);
        return BAL_X86_REGISTER_ZERO;
    }

    const bool skip_load_instruction = false;
    return allocate_x86_register(compiler, arm_register, skip_load_instruction);
}

void
flush_dirty_registers(bal_tier1_compiler_t *compiler)
{
    if (NULL == compiler)
    {
        return;
    }

    const bal_logger_t *BAL_RESTRICT logger            = &bal_thread_logger;
    const uint32_t                   dirty_mask        = compiler->is_dirty;
    const int8_t *BAL_RESTRICT       arm_to_x86_cursor = compiler->arm_to_x86;

    for (uint8_t arm_register = 0; arm_register < 32; ++arm_register)
    {
        if (dirty_mask & (1U << arm_register))
        {
            const bal_x86_register_t x86_register = (bal_x86_register_t)*arm_to_x86_cursor;
            const uint64_t           offset       = offsetof(bal_cpu_t, x[arm_register]);
            const bal_x86_macro_t    store_macro  = {
                    .opcode              = BAL_X86_MACRO_STORE,
                    .source              = x86_register,
                    .immediate_or_offset = offset,
            };
            BAL_LOG_DEBUG(
                &bal_thread_logger, "Flush: dirty ARM X%u to x86 r%d", arm_register, x86_register);
            (void)logger;
            bal_sliding_window_push(&compiler->window, store_macro);
        }

        ++arm_to_x86_cursor;
    }
}

void
terminate_block(bal_tier1_compiler_t     *compiler,
                const bal_guest_address_t next_pc,
                const bool                is_pc_dynamic,
                const size_t              arm_instruction_count,
                const uint32_t            engine_flags)
{
    if (BAL_UNLIKELY(NULL == compiler))
    {
        return;
    }

    BAL_LOG_DEBUG(&bal_thread_logger, "Terminating basic block");

    if (engine_flags & BAL_ENGINE_FLAG_INSTRUCTION_COUNTING)
    {
        const bal_x86_macro_t count_macro = {
            .opcode              = BAL_X86_MACRO_ADD_CPU_ICOUNT,
            .destination         = BAL_X86_RBP,
            .immediate_or_offset = arm_instruction_count,
        };
        bal_sliding_window_push(&compiler->window, count_macro);
    }

    flush_dirty_registers(compiler);
    BAL_LOG_TRACE(&bal_thread_logger, "Flushing sliding window");
    bal_sliding_window_flush_all(&compiler->window);

    if (false == is_pc_dynamic)
    {
        BAL_LOG_TRACE(&bal_thread_logger, "Updating Guest PC");
        bal_x86_emit_mov_r64_imm64(&compiler->assembler, BAL_X86_RAX, next_pc);
        bal_x86_emit_store_r64_rbp_offset(
            &compiler->assembler, BAL_X86_RAX, offsetof(bal_cpu_t, pc));
    }

    BAL_LOG_TRACE(&bal_thread_logger, "Restoring callee-saved registers and emitting RET");
    emit_epilogue(&compiler->assembler);
}

void
terminate_block_conditional(bal_tier1_compiler_t     *compiler,
                            const bal_x86_condition_t condition,
                            const bal_guest_address_t fallthrough_pc,
                            const bal_guest_address_t taken_pc)
{
    if (BAL_UNLIKELY(NULL == compiler))
    {
        return;
    }

    bal_sliding_window_flush_all(&compiler->window);
    size_t         dirty_count = 0;
    const uint32_t dirty_mask  = compiler->is_dirty;

    for (uint8_t arm_register = 0; arm_register < 32; ++arm_register)
    {
        if (dirty_mask & (1U << arm_register))
        {
            ++dirty_count;
        }
    }

    // STORE macro: 7 bytes (REX + Opcode + ModRM + Disp32)
    // MOV RAX, IMM64: 5 bytes if <= 32-bit, 10 bytes if > 32-bit
    // MOV [RBP+PC], RAX: 7 bytes (REX + Opcode + ModRM + Disp32)
    // POP R15-R12: 8 bytes (4 registers x 2 bytes [REX.B + Opcode]
    // POP RBP: 1 byte
    // RET: 1 byte
    const size_t mov_fallthrough_size = (fallthrough_pc <= 0xFFFFFFFFULL) ? 5 : 10;
    const size_t epilogue_size        = (dirty_count * 7) + mov_fallthrough_size + 7 + 8 + 1 + 1;

    bal_x86_assembler_t *BAL_RESTRICT assembler   = &compiler->assembler;
    const int32_t                     jump_offset = (int32_t)epilogue_size;
    bal_x86_emit_jcc_rel32(assembler, condition, jump_offset);

    flush_dirty_registers(compiler);
    bal_sliding_window_flush_all(&compiler->window);

    // Emit fallthrough epilogue.
    bal_x86_emit_mov_r64_imm64(&compiler->assembler, BAL_X86_RAX, fallthrough_pc);
    bal_x86_emit_store_r64_rbp_offset(&compiler->assembler, BAL_X86_RAX, offsetof(bal_cpu_t, pc));
    emit_epilogue(&compiler->assembler);
    flush_dirty_registers(compiler);
    bal_sliding_window_flush_all(&compiler->window);

    // Emit taken epilogue.
    bal_x86_emit_mov_r64_imm64(&compiler->assembler, BAL_X86_RAX, taken_pc);
    bal_x86_emit_store_r64_rbp_offset(&compiler->assembler, BAL_X86_RAX, offsetof(bal_cpu_t, pc));
    emit_epilogue(&compiler->assembler);
}

/// Restores the callee-saved registers pushed by the block prologue and returns to the host.
static void
emit_epilogue(bal_x86_assembler_t *assembler)
{
    bal_x86_emit_pop_r64(assembler, BAL_X86_R15);
    bal_x86_emit_pop_r64(assembler, BAL_X86_R14);
    bal_x86_emit_pop_r64(assembler, BAL_X86_R13);
    bal_x86_emit_pop_r64(assembler, BAL_X86_R12);
    bal_x86_emit_pop_r64(assembler, BAL_X86_RBP);
    bal_x86_emit_ret(assembler);
}

BAL_HOT static uint32_t
extract_operand_value(const uint32_t instruction, const bal_decoder_operand_t *operand)
{
    if (BAL_OPERAND_TYPE_NONE == operand->type)
    {
        return 0;
    }

    const uint32_t mask = (1U << operand->bit_width) - 1;
    const uint32_t bits = instruction >> operand->bit_position & mask;
    return bits;
}

void
translate_add_sub_imm(bal_tier1_compiler_t *BAL_RESTRICT                     compiler,
                      const bal_decoder_instruction_metadata_t *BAL_RESTRICT metadata,
                      const uint32_t                                         instruction,
                      const bool                                             is_sub)
{
    const bal_decoder_operand_t *BAL_RESTRICT operand_cursor = metadata->operands;
    const uint8_t  rd           = (uint8_t)extract_operand_value(instruction, &operand_cursor[0]);
    const uint8_t  rn           = (uint8_t)extract_operand_value(instruction, &operand_cursor[1]);
    const uint32_t imm12        = extract_operand_value(instruction, &operand_cursor[2]);
    const uint32_t shift        = extract_operand_value(instruction, &operand_cursor[3]);
    const uint64_t shift_amount = 1 == shift ? 12 : 0;
    uint64_t       value        = (uint64_t)imm12 << shift_amount;

    // Does this instruction set flags?
    const bool is_setting_flags = (OPCODE_CMP == metadata->ir_opcode) || ('S' == metadata->name[3])
                                  || ('N' == metadata->name[2]);
    const bool               is_discarded_result = is_setting_flags && (31 == rd);
    const bool               skip_load_rn        = false;
    const bal_x86_register_t x86_rn = allocate_x86_register(compiler, rn, skip_load_rn);
    bal_x86_register_t       x86_rd;

    if (true == is_discarded_result)
    {
        x86_rd                          = BAL_X86_REGISTER_DISCARD_RESULT;
        const bal_x86_macro_t mov_macro = {
            .opcode      = BAL_X86_MACRO_MOV_REGISTER_REGISTER,
            .destination = x86_rd,
            .source      = x86_rn,
        };
        bal_sliding_window_push(&compiler->window, mov_macro);
    }
    else
    {
        const bool skip_load_rd = true;
        x86_rd                  = allocate_x86_register(compiler, rd, skip_load_rd);

        if (x86_rd != x86_rn)
        {
            const bal_x86_macro_t mov_macro = {
                .opcode      = BAL_X86_MACRO_MOV_REGISTER_REGISTER,
                .destination = x86_rd,
                .source      = x86_rn,
            };
            bal_sliding_window_push(&compiler->window, mov_macro);
        }
    }
    const bool             is_32bit = BAL_OPERAND_TYPE_REGISTER_32 == operand_cursor[0].type;
    bal_x86_macro_opcode_t arithmetic_opcode;

    if (true == is_sub)
    {
        arithmetic_opcode = true == is_32bit ? BAL_X86_MACRO_SUB_REGISTER_IMMEDIATE_32
                                             : BAL_X86_MACRO_SUB_REGISTER_IMMEDIATE;
    }
    else
    {
        arithmetic_opcode = true == is_32bit ? BAL_X86_MACRO_ADD_REGISTER_IMMEDIATE_32
                                             : BAL_X86_MACRO_ADD_REGISTER_IMMEDIATE;
    }

    const bal_x86_macro_t arithmetic_macro = {
        .opcode              = arithmetic_opcode,
        .destination         = x86_rd,
        .immediate_or_offset = value,
    };
    bal_sliding_window_push(&compiler->window, arithmetic_macro);

    if (true == is_setting_flags)
    {
        // ARM carry flag is inverted compared to x86's carry during subtraction.
        capture_nzcv(compiler, true == is_sub ? BAL_X86_COND_AE : BAL_X86_COND_B);
    }

    if (false == is_discarded_result)
    {
        compiler->is_dirty |= 1U << rd;
    }
}

void
translate_b_cond(bal_tier1_compiler_t *BAL_RESTRICT compiler,
                 const uint32_t                     instruction,
                 const bal_guest_address_t          guest_address)
{
    const uint32_t            cond           = instruction & 0xF;
    const uint32_t            imm19          = (instruction >> 5) & 0x7FFFF;
    const int32_t             signed_imm     = (int32_t)(imm19 ^ 0x40000) - 0x40000;
    const int64_t             offset         = (int64_t)signed_imm * 4;
    const bal_guest_address_t taken_pc       = guest_address + (uint64_t)offset;
    const bal_guest_address_t fallthrough_pc = guest_address + 4;

    bal_x86_assembler_t *BAL_RESTRICT assembler   = &compiler->assembler;
    int32_t                           flag_offset = 0;

    // If true, branch if flag == 0. If false, branch if flag != 0.
    bool test_for_zero = false;

    switch (cond)
    {
        case 0x0:
            flag_offset   = offsetof(bal_cpu_t, flag_z);
            test_for_zero = false;
            break; // EQ (Z=1)
        case 0x1:
            flag_offset   = offsetof(bal_cpu_t, flag_z);
            test_for_zero = true;
            break; // NE (Z=0)
        case 0x2:
            flag_offset   = offsetof(bal_cpu_t, flag_c);
            test_for_zero = false;
            break; // CS/HS (C=1)
        case 0x3:
            flag_offset   = offsetof(bal_cpu_t, flag_c);
            test_for_zero = true;
            break; // CC/LO (C=0)
        case 0x4:
            flag_offset   = offsetof(bal_cpu_t, flag_n);
            test_for_zero = false;
            break; // MI (N=1)
        case 0x5:
            flag_offset   = offsetof(bal_cpu_t, flag_n);
            test_for_zero = true;
            break; // PL (N=0)
        case 0x6:
            flag_offset   = offsetof(bal_cpu_t, flag_v);
            test_for_zero = false;
            break; // VS (V=1)
        case 0x7:
            flag_offset   = offsetof(bal_cpu_t, flag_v);
            test_for_zero = true;
            break; // VC (V=0)
        case 0x8:  // HI (C=1 and Z=0)
        case 0x9:  // LS (C=0 or Z=1)
        case 0xA:  // GE (N=V)
        case 0xB:  // LT (N!=V)
        case 0xC:  // GT (Z=0 and N=V)
        case 0xD:  // LE (Z=1 or N!=V)
            emit_compound_condition(compiler, cond);

            // The odd conditions are the negations of the even ones, which leave ZF set when they
            // hold.
            terminate_block_conditional(compiler,
                                        0U == (cond & 1U) ? BAL_X86_COND_E : BAL_X86_COND_NE,
                                        fallthrough_pc,
                                        taken_pc);
            return;
        case 0xF: // NV, which behaves exactly like AL in A64.
        case 0xE: // AL (always)
            flush_dirty_registers(compiler);
            bal_sliding_window_flush_all(&compiler->window);
            bal_x86_emit_mov_r64_imm64(assembler, BAL_X86_RAX, taken_pc);
            bal_x86_emit_store_r64_rbp_offset(assembler, BAL_X86_RAX, offsetof(bal_cpu_t, pc));
            emit_epilogue(assembler);
            return;
        default:
            BAL_LOG_ERROR(&bal_thread_logger, "Unsupported B.cond condition: %u", cond);
            compiler->status = BAL_ERROR_UNKNOWN_INSTRUCTION;
            return;
    }

    bal_sliding_window_flush_all(&compiler->window);
    bal_x86_emit_cmp_mem8_rbp_offset_imm(assembler, flag_offset, 0);
    const bal_x86_condition_t x86_cond = test_for_zero ? BAL_X86_COND_E : BAL_X86_COND_NE;
    terminate_block_conditional(compiler, x86_cond, fallthrough_pc, taken_pc);
}

/// Evaluates HI, GE or GT (or their negations LS, LT and LE) from the guest flags and leaves the
/// x86 ZF set when the even condition holds.
///
/// One 64-bit load from flag_c places C, Z, N and V in bytes 0 to 3, since they are consecutive in
/// bal_cpu_t. The condition is then reduced to bit 0 of BAL_X86_REGISTER_ZERO, which is zero
/// exactly when the even condition holds:
///
/// - HI: NOT(C) OR Z
/// - GE: N XOR V
/// - GT: (N XOR V) OR Z
static void
emit_compound_condition(bal_tier1_compiler_t *BAL_RESTRICT compiler, const uint32_t condition)
{
    const bal_x86_register_t flags   = BAL_X86_REGISTER_ZERO;
    const bal_x86_register_t scratch = BAL_X86_REGISTER_DISCARD_RESULT;

    const bal_x86_macro_t load_flags_macro = {
        .opcode              = BAL_X86_MACRO_LOAD,
        .destination         = flags,
        .immediate_or_offset = offsetof(bal_cpu_t, flag_c),
    };
    bal_sliding_window_push(&compiler->window, load_flags_macro);

    const bal_x86_macro_t copy_macro = {
        .opcode      = BAL_X86_MACRO_MOV_REGISTER_REGISTER,
        .destination = scratch,
        .source      = flags,
    };
    bal_sliding_window_push(&compiler->window, copy_macro);

    if (0x8U == (condition & ~1U))
    {
        // scratch = Z, flags = NOT(C), flags |= scratch.
        const bal_x86_macro_t shift_z_macro = {
            .opcode              = BAL_X86_MACRO_SHR_REGISTER_IMMEDIATE,
            .destination         = scratch,
            .immediate_or_offset = 8U,
        };
        bal_sliding_window_push(&compiler->window, shift_z_macro);

        const bal_x86_macro_t not_c_macro = {
            .opcode      = BAL_X86_MACRO_NOT_REGISTER,
            .destination = flags,
        };
        bal_sliding_window_push(&compiler->window, not_c_macro);

        const bal_x86_macro_t or_macro = {
            .opcode      = BAL_X86_MACRO_OR_REGISTER_REGISTER,
            .destination = flags,
            .source      = scratch,
        };
        bal_sliding_window_push(&compiler->window, or_macro);
    }
    else
    {
        // flags = N, scratch = V, flags ^= scratch.
        const bal_x86_macro_t shift_n_macro = {
            .opcode              = BAL_X86_MACRO_SHR_REGISTER_IMMEDIATE,
            .destination         = flags,
            .immediate_or_offset = 16U,
        };
        bal_sliding_window_push(&compiler->window, shift_n_macro);

        const bal_x86_macro_t shift_v_macro = {
            .opcode              = BAL_X86_MACRO_SHR_REGISTER_IMMEDIATE,
            .destination         = scratch,
            .immediate_or_offset = 24U,
        };
        bal_sliding_window_push(&compiler->window, shift_v_macro);

        const bal_x86_macro_t xor_macro = {
            .opcode      = BAL_X86_MACRO_XOR_REGISTER_REGISTER,
            .destination = flags,
            .source      = scratch,
        };
        bal_sliding_window_push(&compiler->window, xor_macro);

        if (0xCU == (condition & ~1U))
        {
            // Z sits in byte 0 of a load from flag_z.
            const bal_x86_macro_t load_z_macro = {
                .opcode              = BAL_X86_MACRO_LOAD,
                .destination         = scratch,
                .immediate_or_offset = offsetof(bal_cpu_t, flag_z),
            };
            bal_sliding_window_push(&compiler->window, load_z_macro);

            const bal_x86_macro_t or_macro = {
                .opcode      = BAL_X86_MACRO_OR_REGISTER_REGISTER,
                .destination = flags,
                .source      = scratch,
            };
            bal_sliding_window_push(&compiler->window, or_macro);
        }
    }

    // Keep bit 0 only. The AND sets ZF from the result, but the TEST makes the dependency explicit
    // since the AND immediate macro is expanded through a scratch register.
    const bal_x86_macro_t mask_macro = {
        .opcode              = BAL_X86_MACRO_AND_REGISTER_IMMEDIATE,
        .destination         = flags,
        .immediate_or_offset = 1U,
    };
    bal_sliding_window_push(&compiler->window, mask_macro);

    const bal_x86_macro_t test_macro = {
        .opcode      = BAL_X86_MACRO_TEST_REGISTER_REGISTER,
        .destination = flags,
        .source      = flags,
    };
    bal_sliding_window_push(&compiler->window, test_macro);
}

void
translate_cbz_cbnz(bal_tier1_compiler_t *BAL_RESTRICT                     compiler,
                   const bal_decoder_instruction_metadata_t *BAL_RESTRICT metadata,
                   const uint32_t                                         instruction,
                   const bal_guest_address_t                              guest_address,
                   const bool                                             is_cbnz)
{
    const uint8_t  rn        = (uint8_t)extract_operand_value(instruction, &metadata->operands[0]);
    const uint32_t imm19     = extract_operand_value(instruction, &metadata->operands[1]);
    const uint32_t bit_width = metadata->operands[1].bit_width;
    const uint32_t shift     = 32U - bit_width;

    const int32_t             signed_immediate = (int32_t)(imm19 << shift) >> shift;
    const int64_t             offset           = (int64_t)signed_immediate * 4;
    const bal_guest_address_t taken_pc         = guest_address + (uint64_t)offset;
    const bal_guest_address_t fallthrough_pc   = guest_address + 4;
    const bal_x86_register_t  x86_rn           = read_zero_or_register(compiler, rn);

    // CBZ Wt and CBNZ Wt only look at the low 32 bits of the register.
    const bool            is_32bit   = 0U == (instruction >> 31U);
    const bal_x86_macro_t test_macro = {
        .opcode      = true == is_32bit ? BAL_X86_MACRO_TEST_REGISTER_REGISTER_32
                                        : BAL_X86_MACRO_TEST_REGISTER_REGISTER,
        .destination = x86_rn,
        .source      = x86_rn,
    };
    bal_sliding_window_push(&compiler->window, test_macro);
    terminate_block_conditional(
        compiler, true == is_cbnz ? BAL_X86_COND_NE : BAL_X86_COND_E, fallthrough_pc, taken_pc);
}

void
translate_jump(const bal_tier1_compiler_t *BAL_RESTRICT               compiler,
               const bal_decoder_instruction_metadata_t *BAL_RESTRICT metadata,
               const uint32_t                                         instruction,
               const bal_guest_address_t                              guest_address,
               bal_guest_address_t *BAL_RESTRICT                      target_pc)
{
    const uint32_t imm26     = extract_operand_value(instruction, &metadata->operands[0]);
    const uint32_t bit_width = metadata->operands[0].bit_width;
    const uint32_t shift     = 32U - bit_width;

    // WARNING: Shift guarantees sign extension within 32-bits.
    const int32_t signed_immediate = (int32_t)(imm26 << shift) >> shift;

    // WARNING: ARM64 branch offsets are encoded as 4 bytes.
    const int64_t offset = (int64_t)signed_immediate * 4;

    // WARNING: Cast offset to uint64_t handles negative offsets safely via two's complement
    // wrapping.
    *target_pc = guest_address + (uint64_t)offset;

    BAL_LOG_DEBUG(
        &bal_thread_logger, "Translated JUMP to 0x%016llX", (unsigned long long)*target_pc);

    // WARNING: Prevents unsued local variable compiler warning.
    (void)compiler;
}

void
translate_mov_immediate(bal_tier1_compiler_t                     *compiler,
                        const bal_decoder_instruction_metadata_t *metadata,
                        const uint32_t                            instruction)
{
    const uint8_t rd = (uint8_t)extract_operand_value(instruction, &metadata->operands[0]);

    // MOVZ, MOVN and MOVK treat register 31 as the zero register, so the write is discarded.
    if (31 == rd)
    {
        return;
    }

    const uint64_t imm16   = extract_operand_value(instruction, &metadata->operands[1]);
    const uint64_t hw      = extract_operand_value(instruction, &metadata->operands[2]);
    const uint64_t shift   = hw * 16ULL;
    const uint64_t mask    = metadata->operands[0].type == BAL_OPERAND_TYPE_REGISTER_32
                                 ? 0xFFFFFFFFULL
                                 : 0xFFFFFFFFFFFFFFFFULL;
    const char     variant = metadata->name[3];

    if ('Z' == variant)
    {
        const uint64_t           value                 = imm16 << shift & mask;
        const bool               skip_load_instruction = true;
        const bal_x86_register_t x86_rd
            = allocate_x86_register(compiler, rd, skip_load_instruction);

        const bal_x86_macro_t mov_macro = {
            .opcode              = BAL_X86_MACRO_MOV_REGISTER_IMMEDIATE,
            .destination         = x86_rd,
            .immediate_or_offset = value,
        };
        bal_sliding_window_push(&compiler->window, mov_macro);
        compiler->is_dirty |= 1U << rd;
    }
    else if ('N' == variant)
    {
        const uint64_t           value                 = ~(imm16 << shift) & mask;
        const bool               skip_load_instruction = true;
        const bal_x86_register_t x86_rd
            = allocate_x86_register(compiler, rd, skip_load_instruction);
        const bal_x86_macro_t mov_macro = {
            .opcode              = BAL_X86_MACRO_MOV_REGISTER_IMMEDIATE,
            .destination         = x86_rd,
            .immediate_or_offset = value,
        };
        bal_sliding_window_push(&compiler->window, mov_macro);
        compiler->is_dirty |= 1U << rd;
    }
    else
    {
        // MOVK: Keep existing bits, overwrite 16 bits.
        //
        const bool               skip_load_instruction = false;
        const bal_x86_register_t x86_rd
            = allocate_x86_register(compiler, rd, skip_load_instruction);
        const uint64_t        clear_mask   = ~(0xFFFFULL << shift) & mask;
        const uint64_t        insert_value = imm16 << shift & mask;
        const bal_x86_macro_t and_macro    = {
               .opcode              = BAL_X86_MACRO_AND_REGISTER_IMMEDIATE,
               .destination         = x86_rd,
               .immediate_or_offset = clear_mask,
        };
        const bal_x86_macro_t or_macro = {
            .opcode              = BAL_X86_MACRO_OR_REGISTER_IMMEDIATE,
            .destination         = x86_rd,
            .immediate_or_offset = insert_value,
        };
        bal_sliding_window_push(&compiler->window, and_macro);
        bal_sliding_window_push(&compiler->window, or_macro);
    }

    compiler->is_dirty |= 1U << rd;
}

void
translate_add_sub_shifted(bal_tier1_compiler_t *BAL_RESTRICT compiler, const uint32_t instruction)
{
    const uint8_t  rd               = (uint8_t)(instruction & 0x1FU);
    const uint8_t  rn               = (uint8_t)((instruction >> 5U) & 0x1FU);
    const uint32_t shift_amount     = (instruction >> 10U) & 0x3FU;
    const uint8_t  rm               = (uint8_t)((instruction >> 16U) & 0x1FU);
    const uint32_t shift_type       = (instruction >> 22U) & 0x3U;
    const bool     is_setting_flags = 1U == ((instruction >> 29U) & 1U);
    const bool     is_sub           = 1U == ((instruction >> 30U) & 1U);
    const bool     is_32bit         = 0U == (instruction >> 31U);

    // ROR is reserved for ADD/SUB, as are 32-bit shifts of 32 or more.
    if (BAL_UNLIKELY(3U == shift_type) || BAL_UNLIKELY(true == is_32bit && shift_amount > 31U))
    {
        BAL_LOG_ERROR(&bal_thread_logger,
                      "Aborting function: reserved ADD/SUB (shifted register) encoding %08x",
                      instruction);
        compiler->status = BAL_ERROR_UNKNOWN_INSTRUCTION;
        return;
    }

    // Register 31 is the zero register for every operand of this form. Without flags to set,
    // writing it has no architectural effect.
    const bool is_discarded_result = 31U == rd;

    if (true == is_discarded_result && false == is_setting_flags)
    {
        return;
    }

    const bool               is_rn_zero  = 31U == rn;
    const bool               skip_load   = false;
    const bal_x86_register_t x86_rn      = true == is_rn_zero
                                               ? BAL_X86_REGISTER_ZERO
                                               : allocate_x86_register(compiler, rn, skip_load);
    const bool               is_inverted = false;
    const bal_x86_register_t x86_operand2
        = read_shifted_register(compiler, rm, shift_type, shift_amount, is_32bit, is_inverted);
    bal_x86_register_t x86_result = BAL_X86_REGISTER_DISCARD_RESULT;

    if (false == is_discarded_result)
    {
        const bool skip_load_rd = true;
        x86_result              = allocate_x86_register(compiler, rd, skip_load_rd);
    }

    bal_x86_macro_opcode_t opcode;

    if (true == is_sub)
    {
        opcode = true == is_32bit ? BAL_X86_MACRO_SUB_REGISTER_REGISTER_32
                                  : BAL_X86_MACRO_SUB_REGISTER_REGISTER;
    }
    else
    {
        opcode = true == is_32bit ? BAL_X86_MACRO_ADD_REGISTER_REGISTER_32
                                  : BAL_X86_MACRO_ADD_REGISTER_REGISTER;
    }

    const bool is_commutative = false == is_sub;
    emit_binary_operation(
        compiler, opcode, is_commutative, x86_result, is_rn_zero, x86_rn, x86_operand2);

    if (true == is_setting_flags)
    {
        // ARM carry flag is inverted compared to x86's carry during subtraction.
        capture_nzcv(compiler, true == is_sub ? BAL_X86_COND_AE : BAL_X86_COND_B);
    }

    if (false == is_discarded_result)
    {
        compiler->is_dirty |= 1U << rd;
    }
}

void
translate_logical(bal_tier1_compiler_t *BAL_RESTRICT compiler, const uint32_t instruction)
{
    const uint8_t  rd               = (uint8_t)(instruction & 0x1FU);
    const uint8_t  rn               = (uint8_t)((instruction >> 5U) & 0x1FU);
    const uint32_t opc              = (instruction >> 29U) & 0x3U;
    const bool     is_32bit         = 0U == (instruction >> 31U);
    const bool     is_immediate     = 0x24U == ((instruction >> 23U) & 0x3FU);
    const bool     is_setting_flags = 3U == opc;

    // Rd is the stack pointer for AND/ORR/EOR (immediate) and the zero register otherwise.
    const bool is_rd_stack_pointer = true == is_immediate && false == is_setting_flags;
    const bool is_discarded_result = 31U == rd && false == is_rd_stack_pointer;

    if (true == is_discarded_result && false == is_setting_flags)
    {
        return;
    }

    uint64_t bit_mask = 0U;

    if (true == is_immediate)
    {
        const uint32_t imms           = (instruction >> 10U) & 0x3FU;
        const uint32_t immr           = (instruction >> 16U) & 0x3FU;
        const uint32_t n              = (instruction >> 22U) & 1U;
        const bool     is_valid_masks = decode_bit_masks(n, imms, immr, is_32bit, &bit_mask);

        if (BAL_UNLIKELY(false == is_valid_masks))
        {
            BAL_LOG_ERROR(&bal_thread_logger,
                          "Aborting function: reserved logical immediate encoding %08x",
                          instruction);
            compiler->status = BAL_ERROR_UNKNOWN_INSTRUCTION;
            return;
        }
    }
    else if (BAL_UNLIKELY(true == is_32bit && ((instruction >> 10U) & 0x3FU) > 31U))
    {
        BAL_LOG_ERROR(&bal_thread_logger,
                      "Aborting function: reserved logical shifted register encoding %08x",
                      instruction);
        compiler->status = BAL_ERROR_UNKNOWN_INSTRUCTION;
        return;
    }

    const bool               is_rn_zero   = 31U == rn;
    const bool               skip_load    = false;
    const bal_x86_register_t x86_rn       = true == is_rn_zero
                                                ? BAL_X86_REGISTER_ZERO
                                                : allocate_x86_register(compiler, rn, skip_load);
    bal_x86_register_t       x86_operand2 = BAL_X86_REGISTER_ZERO;

    if (true == is_immediate)
    {
        const bal_x86_macro_t mask_macro = {
            .opcode              = BAL_X86_MACRO_MOV_REGISTER_IMMEDIATE,
            .destination         = BAL_X86_REGISTER_ZERO,
            .immediate_or_offset = bit_mask,
        };
        bal_sliding_window_push(&compiler->window, mask_macro);
    }
    else
    {
        const uint32_t shift_amount = (instruction >> 10U) & 0x3FU;
        const uint8_t  rm           = (uint8_t)((instruction >> 16U) & 0x1FU);
        const bool     is_inverted  = 1U == ((instruction >> 21U) & 1U);
        const uint32_t shift_type   = (instruction >> 22U) & 0x3U;
        x86_operand2
            = read_shifted_register(compiler, rm, shift_type, shift_amount, is_32bit, is_inverted);
    }

    bal_x86_register_t x86_result = BAL_X86_REGISTER_DISCARD_RESULT;

    if (false == is_discarded_result)
    {
        const bool skip_load_rd = true;
        x86_result              = allocate_x86_register(compiler, rd, skip_load_rd);
    }

    bal_x86_macro_opcode_t opcode;

    if (1U == opc)
    {
        opcode = true == is_32bit ? BAL_X86_MACRO_OR_REGISTER_REGISTER_32
                                  : BAL_X86_MACRO_OR_REGISTER_REGISTER;
    }
    else if (2U == opc)
    {
        opcode = true == is_32bit ? BAL_X86_MACRO_XOR_REGISTER_REGISTER_32
                                  : BAL_X86_MACRO_XOR_REGISTER_REGISTER;
    }
    else
    {
        opcode = true == is_32bit ? BAL_X86_MACRO_AND_REGISTER_REGISTER_32
                                  : BAL_X86_MACRO_AND_REGISTER_REGISTER;
    }

    const bool is_commutative = true;
    emit_binary_operation(
        compiler, opcode, is_commutative, x86_result, is_rn_zero, x86_rn, x86_operand2);

    if (true == is_setting_flags)
    {
        // x86 AND clears CF and OF, which matches the C = 0 and V = 0 that ANDS and BICS set.
        capture_nzcv(compiler, BAL_X86_COND_B);
    }

    if (false == is_discarded_result)
    {
        compiler->is_dirty |= 1U << rd;
    }
}

/// Returns an x86 register holding the second operand of a shifted register instruction: the
/// guest register shifted by an immediate and, for BIC/ORN/EON/BICS, inverted. Register 31 reads
/// as zero. When the operand needs a shift, an inversion or zero, it is built in
/// BAL_X86_REGISTER_ZERO so the guest register keeps its value.
static bal_x86_register_t
read_shifted_register(bal_tier1_compiler_t *BAL_RESTRICT compiler,
                      const uint8_t                      arm_register,
                      const uint32_t                     shift_type,
                      const uint32_t                     shift_amount,
                      const bool                         is_32bit,
                      const bool                         is_inverted)
{
    if (31U == arm_register)
    {
        // Any shift of zero is zero, so only the inversion matters.
        const uint64_t        all_ones   = true == is_32bit ? 0xFFFFFFFFULL : 0xFFFFFFFFFFFFFFFFULL;
        const bal_x86_macro_t zero_macro = {
            .opcode              = BAL_X86_MACRO_MOV_REGISTER_IMMEDIATE,
            .destination         = BAL_X86_REGISTER_ZERO,
            .immediate_or_offset = true == is_inverted ? all_ones : 0U,
        };
        bal_sliding_window_push(&compiler->window, zero_macro);
        return BAL_X86_REGISTER_ZERO;
    }

    const bool               skip_load_instruction = false;
    const bal_x86_register_t x86_register
        = allocate_x86_register(compiler, arm_register, skip_load_instruction);

    if (0U == shift_amount && false == is_inverted)
    {
        return x86_register;
    }

    const bal_x86_macro_t copy_macro = {
        .opcode      = BAL_X86_MACRO_MOV_REGISTER_REGISTER,
        .destination = BAL_X86_REGISTER_ZERO,
        .source      = x86_register,
    };
    bal_sliding_window_push(&compiler->window, copy_macro);

    if (shift_amount != 0U)
    {
        static const bal_x86_macro_opcode_t SHIFT_OPCODES_64[] = {
            BAL_X86_MACRO_SHL_REGISTER_IMMEDIATE,
            BAL_X86_MACRO_SHR_REGISTER_IMMEDIATE,
            BAL_X86_MACRO_SAR_REGISTER_IMMEDIATE,
            BAL_X86_MACRO_ROR_REGISTER_IMMEDIATE,
        };
        static const bal_x86_macro_opcode_t SHIFT_OPCODES_32[] = {
            BAL_X86_MACRO_SHL_REGISTER_IMMEDIATE_32,
            BAL_X86_MACRO_SHR_REGISTER_IMMEDIATE_32,
            BAL_X86_MACRO_SAR_REGISTER_IMMEDIATE_32,
            BAL_X86_MACRO_ROR_REGISTER_IMMEDIATE_32,
        };
        const bal_x86_macro_t shift_macro = {
            .opcode              = true == is_32bit ? SHIFT_OPCODES_32[shift_type & 3U]
                                                    : SHIFT_OPCODES_64[shift_type & 3U],
            .destination         = BAL_X86_REGISTER_ZERO,
            .immediate_or_offset = shift_amount,
        };
        bal_sliding_window_push(&compiler->window, shift_macro);
    }

    if (true == is_inverted)
    {
        const bal_x86_macro_t not_macro = {
            .opcode = true == is_32bit ? BAL_X86_MACRO_NOT_REGISTER_32 : BAL_X86_MACRO_NOT_REGISTER,
            .destination = BAL_X86_REGISTER_ZERO,
        };
        bal_sliding_window_push(&compiler->window, not_macro);
    }

    return BAL_X86_REGISTER_ZERO;
}

/// Emits `x86_result = rn <opcode> operand2` for a two operand x86 instruction, where rn reads as
/// zero when `is_rn_zero` is set.
///
/// x86_result may alias x86_rn or x86_operand2. When it aliases the second operand of a
/// non-commutative operation, the operand is first copied to BAL_X86_REGISTER_ZERO so it is not
/// overwritten by rn before it is read.
static void
emit_binary_operation(bal_tier1_compiler_t *BAL_RESTRICT compiler,
                      const bal_x86_macro_opcode_t       opcode,
                      const bool                         is_commutative,
                      const bal_x86_register_t           x86_result,
                      const bool                         is_rn_zero,
                      const bal_x86_register_t           x86_rn,
                      bal_x86_register_t                 x86_operand2)
{
    if (x86_result == x86_operand2)
    {
        if (true == is_commutative && false == is_rn_zero)
        {
            const bal_x86_macro_t operation_macro = {
                .opcode      = opcode,
                .destination = x86_result,
                .source      = x86_rn,
            };
            bal_sliding_window_push(&compiler->window, operation_macro);
            return;
        }

        const bal_x86_macro_t copy_macro = {
            .opcode      = BAL_X86_MACRO_MOV_REGISTER_REGISTER,
            .destination = BAL_X86_REGISTER_ZERO,
            .source      = x86_operand2,
        };
        bal_sliding_window_push(&compiler->window, copy_macro);
        x86_operand2 = BAL_X86_REGISTER_ZERO;
    }

    if (true == is_rn_zero)
    {
        const bal_x86_macro_t zero_macro = {
            .opcode              = BAL_X86_MACRO_MOV_REGISTER_IMMEDIATE,
            .destination         = x86_result,
            .immediate_or_offset = 0U,
        };
        bal_sliding_window_push(&compiler->window, zero_macro);
    }
    else
    {
        const bal_x86_macro_t copy_macro = {
            .opcode      = BAL_X86_MACRO_MOV_REGISTER_REGISTER,
            .destination = x86_result,
            .source      = x86_rn,
        };
        bal_sliding_window_push(&compiler->window, copy_macro);
    }

    const bal_x86_macro_t operation_macro = {
        .opcode      = opcode,
        .destination = x86_result,
        .source      = x86_operand2,
    };
    bal_sliding_window_push(&compiler->window, operation_macro);
}

/// Stores the x86 flags left by the previous instruction into the guest NZCV fields.
/// `carry_condition` selects the x86 condition that holds the guest C flag.
static void
capture_nzcv(bal_tier1_compiler_t *BAL_RESTRICT compiler, const bal_x86_condition_t carry_condition)
{
    const bal_x86_macro_t set_n_macro = {
        .opcode              = BAL_X86_MACRO_SETCC,
        .condition           = BAL_X86_COND_S,
        .immediate_or_offset = offsetof(bal_cpu_t, flag_n),
    };
    bal_sliding_window_push(&compiler->window, set_n_macro);

    const bal_x86_macro_t set_z_macro = {
        .opcode              = BAL_X86_MACRO_SETCC,
        .condition           = BAL_X86_COND_E,
        .immediate_or_offset = offsetof(bal_cpu_t, flag_z),
    };
    bal_sliding_window_push(&compiler->window, set_z_macro);

    const bal_x86_macro_t set_v_macro = {
        .opcode              = BAL_X86_MACRO_SETCC,
        .condition           = BAL_X86_COND_O,
        .immediate_or_offset = offsetof(bal_cpu_t, flag_v),
    };
    bal_sliding_window_push(&compiler->window, set_v_macro);

    const bal_x86_macro_t set_c_macro = {
        .opcode              = BAL_X86_MACRO_SETCC,
        .condition           = carry_condition,
        .immediate_or_offset = offsetof(bal_cpu_t, flag_c),
    };
    bal_sliding_window_push(&compiler->window, set_c_macro);
}

/// Expands the N:immr:imms bitmask immediate of the A64 logical instructions, following the
/// DecodeBitMasks() pseudocode of the Arm Architecture Reference Manual. Returns false for the
/// reserved encodings.
static bool
decode_bit_masks(const uint32_t         n,
                 const uint32_t         imms,
                 const uint32_t         immr,
                 const bool             is_32bit,
                 uint64_t *BAL_RESTRICT mask)
{
    // The element size is given by the highest set bit of N:NOT(imms).
    const uint32_t combined = (n << 6U) | (~imms & 0x3FU);
    uint32_t       length   = 6U;

    while (length > 0U && 0U == (combined & (1U << length)))
    {
        --length;
    }

    if (0U == length || 0U == (combined & (1U << length)))
    {
        return false;
    }

    const uint32_t element_size = 1U << length;

    if (true == is_32bit && element_size > 32U)
    {
        return false;
    }

    const uint32_t levels = element_size - 1U;
    const uint32_t s      = imms & levels;
    const uint32_t r      = immr & levels;

    // A run of all ones is reserved.
    if (s == levels)
    {
        return false;
    }

    const uint64_t element_mask
        = 64U == element_size ? 0xFFFFFFFFFFFFFFFFULL : (1ULL << element_size) - 1ULL;

    // s + 1 is at most 63 here because s != levels.
    uint64_t element = (1ULL << (s + 1U)) - 1ULL;

    if (r != 0U)
    {
        element = ((element >> r) | (element << (element_size - r))) & element_mask;
    }

    uint64_t result = 0U;

    for (uint32_t position = 0U; position < 64U; position += element_size)
    {
        result |= element << position;
    }

    *mask = true == is_32bit ? result & 0xFFFFFFFFULL : result;
    return true;
}
