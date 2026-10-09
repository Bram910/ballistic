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

static void translate_add_sub_extended(bal_tier1_compiler_t *compiler, uint32_t instruction);

static void translate_add_sub_carry(bal_tier1_compiler_t *compiler, uint32_t instruction);

static void translate_logical(bal_tier1_compiler_t *compiler, uint32_t instruction);

static void push_shift(bal_tier1_compiler_t  *compiler,
                       bal_x86_macro_opcode_t opcode_64,
                       bal_x86_macro_opcode_t opcode_32,
                       bool                   is_32bit,
                       bal_x86_register_t     x86_register,
                       uint32_t               amount);

static void translate_bitfield(bal_tier1_compiler_t *compiler, uint32_t instruction);

static void translate_extract(bal_tier1_compiler_t *compiler, uint32_t instruction);

static bal_x86_condition_t emit_condition_value(bal_tier1_compiler_t *compiler, uint32_t condition);

static void translate_conditional_select(bal_tier1_compiler_t *compiler, uint32_t instruction);

static void translate_multiply_add(bal_tier1_compiler_t *compiler, uint32_t instruction);

static void translate_b_cond(bal_tier1_compiler_t *compiler,
                             uint32_t              instruction,
                             bal_guest_address_t   guest_address);

static void emit_compound_condition(bal_tier1_compiler_t *compiler, uint32_t condition);

static void translate_branch_register(bal_tier1_compiler_t *compiler,
                                      uint32_t              instruction,
                                      bal_guest_address_t   guest_address);

static void translate_test_bit_branch(bal_tier1_compiler_t *compiler,
                                      uint32_t              instruction,
                                      bal_guest_address_t   guest_address);

static void write_link_register(bal_tier1_compiler_t *compiler, bal_guest_address_t return_address);

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
            compiler->pinned_registers   = 0U;

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

                    // ADC/SBC: bits 28:21 = 11010000 and bits 15:10 = 000000.
                    if (0xD0U == ((instruction >> 21U) & 0xFFU)
                        && 0U == ((instruction >> 10U) & 0x3FU))
                    {
                        translate_add_sub_carry(compiler, instruction);
                        break;
                    }

                    // ADD/SUB (extended register): bits 28:24 = 01011 and bit 21 = 1.
                    if (0x0BU == ((instruction >> 24U) & 0x1FU)
                        && 1U == ((instruction >> 21U) & 1U))
                    {
                        translate_add_sub_extended(compiler, instruction);
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
                    // BR Xn.
                    if (0xD61F0000U == (instruction & 0xFFFFFC1FU))
                    {
                        translate_branch_register(compiler, instruction, guest_address);
                        is_pc_dynamic = true;
                    }
                    else
                    {
                        translate_jump(
                            compiler, metadata, instruction, guest_address, &branch_target_pc);
                        has_branch_target = true;
                    }

                    is_block_terminated = true;
                    break;
                case OPCODE_CALL_HOST:
                    // BLR Xn.
                    if (0xD63F0000U == (instruction & 0xFFFFFC1FU))
                    {
                        translate_branch_register(compiler, instruction, guest_address);
                        is_pc_dynamic = true;
                    }
                    else
                    {
                        // BL: branch like B and leave the return address in X30.
                        translate_jump(
                            compiler, metadata, instruction, guest_address, &branch_target_pc);
                        write_link_register(compiler, guest_address + 4U);
                        has_branch_target = true;
                    }

                    is_block_terminated = true;
                    break;
                case OPCODE_TEST_BIT_ZERO:
                    translate_test_bit_branch(compiler, instruction, guest_address);
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
                    translate_branch_register(compiler, instruction, guest_address);
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
                case OPCODE_MUL:
                    // Data-processing (3 source): bits 28:24 = 11011 and op54 = 00.
                    if (0x1BU == ((instruction >> 24U) & 0x1FU)
                        && 0U == ((instruction >> 29U) & 0x3U))
                    {
                        translate_multiply_add(compiler, instruction);
                        break;
                    }

                    BAL_LOG_ERROR(&bal_thread_logger,
                                  "Aborting function: Tier 1 unsupported multiply instruction "
                                  "at %08x : %s",
                                  instruction,
                                  metadata->name);
                    compiler->status    = BAL_ERROR_UNKNOWN_INSTRUCTION;
                    is_block_terminated = true;
                    break;
                case OPCODE_CONDITIONAL_SELECT:
                    // Conditional select: bits 28:21 = 11010100.
                    if (0xD4U == ((instruction >> 21U) & 0xFFU))
                    {
                        translate_conditional_select(compiler, instruction);
                        break;
                    }

                    BAL_LOG_ERROR(&bal_thread_logger,
                                  "Aborting function: Tier 1 unsupported conditional instruction "
                                  "at %08x : %s",
                                  instruction,
                                  metadata->name);
                    compiler->status    = BAL_ERROR_UNKNOWN_INSTRUCTION;
                    is_block_terminated = true;
                    break;
                case OPCODE_SHIFT:
                    // Bitfield (SBFM, BFM, UBFM): bits 28:23 = 100110.
                    if (0x26U == ((instruction >> 23U) & 0x3FU))
                    {
                        translate_bitfield(compiler, instruction);
                        break;
                    }

                    // Extract (EXTR): bits 28:23 = 100111.
                    if (0x27U == ((instruction >> 23U) & 0x3FU))
                    {
                        translate_extract(compiler, instruction);
                        break;
                    }

                    BAL_LOG_ERROR(&bal_thread_logger,
                                  "Aborting function: Tier 1 unsupported shift instruction "
                                  "at %08x : %s",
                                  instruction,
                                  metadata->name);
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

        if (BAL_UNLIKELY(cached_x86_register < 0 || cached_x86_register > (int8_t)BAL_X86_R15))
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
        compiler->pinned_registers |= (uint16_t)(1U << (uint32_t)cached_x86_register);
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

    // Evict a clean register that the current instruction is not using.
    if (BAL_X86_INVALID == free_register)
    {
        for (size_t i = 0U; i < SCRATCH_REGISTERS_SIZE; ++i)
        {
            const bal_x86_register_t candidate           = SCRATCH_REGISTERS[i];
            const int8_t             arm_victim_register = x86_to_arm_cursor[candidate];
            const bool is_pinned = 0U != (compiler->pinned_registers & (1U << candidate));

            if (arm_victim_register != -1 && !(compiler->is_dirty & (1U << arm_victim_register))
                && false == is_pinned)
            {
                free_register = candidate;
                break;
            }
        }

        // All unpinned registers are dirty, force an eviction.
        for (size_t i = 0U; i < SCRATCH_REGISTERS_SIZE && BAL_X86_INVALID == free_register; ++i)
        {
            const uint8_t slot_index           = compiler->eviction_cursor % SCRATCH_REGISTERS_SIZE;
            const bal_x86_register_t candidate = SCRATCH_REGISTERS[slot_index];
            ++compiler->eviction_cursor;

            if (0U == (compiler->pinned_registers & (1U << candidate)))
            {
                free_register = candidate;
            }
        }

        evict_x86_register(compiler, free_register);
    }

    compiler->arm_to_x86[arm_register]  = (int8_t)free_register;
    compiler->x86_to_arm[free_register] = (int8_t)arm_register;
    compiler->pinned_registers |= (uint16_t)(1U << (uint32_t)free_register);

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

/// Translates BR, BLR and RET, which branch to the address held in Xn. BLR also leaves the return
/// address in X30. The target is stored to the guest PC before X30 is written, so BLR X30 branches
/// to the old value of X30.
static void
translate_branch_register(bal_tier1_compiler_t *BAL_RESTRICT compiler,
                          const uint32_t                     instruction,
                          const bal_guest_address_t          guest_address)
{
    const uint8_t            rn      = (uint8_t)((instruction >> 5U) & 0x1FU);
    const bool               is_link = 1U == ((instruction >> 21U) & 0x3U);
    const bal_x86_register_t x86_rn  = read_zero_or_register(compiler, rn);

    const bal_x86_macro_t store_pc_macro = {
        .opcode              = BAL_X86_MACRO_STORE,
        .source              = x86_rn,
        .immediate_or_offset = offsetof(bal_cpu_t, pc),
    };
    bal_sliding_window_push(&compiler->window, store_pc_macro);

    if (true == is_link)
    {
        write_link_register(compiler, guest_address + 4U);
    }
}

/// Translates TBZ and TBNZ, which branch when bit b5:b40 of Rt is zero or non-zero.
static void
translate_test_bit_branch(bal_tier1_compiler_t *BAL_RESTRICT compiler,
                          const uint32_t                     instruction,
                          const bal_guest_address_t          guest_address)
{
    const uint8_t  rt         = (uint8_t)(instruction & 0x1FU);
    const uint32_t imm14      = (instruction >> 5U) & 0x3FFFU;
    const uint32_t bit_number = ((instruction >> 31U) << 5U) | ((instruction >> 19U) & 0x1FU);
    const bool     is_tbnz    = 1U == ((instruction >> 24U) & 1U);

    // WARNING: Shifting the 14-bit field to the top and back sign-extends it.
    const int32_t             signed_immediate = (int32_t)(imm14 << 18U) >> 18U;
    const bal_guest_address_t taken_pc = guest_address + (uint64_t)((int64_t)signed_immediate * 4);
    const bal_guest_address_t fallthrough_pc = guest_address + 4U;

    // Isolate the bit in BAL_X86_REGISTER_ZERO so the guest register keeps its value.
    const bal_x86_register_t x86_rt     = read_zero_or_register(compiler, rt);
    const bal_x86_macro_t    copy_macro = {
           .opcode      = BAL_X86_MACRO_MOV_REGISTER_REGISTER,
           .destination = BAL_X86_REGISTER_ZERO,
           .source      = x86_rt,
    };
    bal_sliding_window_push(&compiler->window, copy_macro);

    if (bit_number != 0U)
    {
        const bal_x86_macro_t shift_macro = {
            .opcode              = BAL_X86_MACRO_SHR_REGISTER_IMMEDIATE,
            .destination         = BAL_X86_REGISTER_ZERO,
            .immediate_or_offset = bit_number,
        };
        bal_sliding_window_push(&compiler->window, shift_macro);
    }

    const bal_x86_macro_t mask_macro = {
        .opcode              = BAL_X86_MACRO_AND_REGISTER_IMMEDIATE,
        .destination         = BAL_X86_REGISTER_ZERO,
        .immediate_or_offset = 1U,
    };
    bal_sliding_window_push(&compiler->window, mask_macro);

    const bal_x86_macro_t test_macro = {
        .opcode      = BAL_X86_MACRO_TEST_REGISTER_REGISTER,
        .destination = BAL_X86_REGISTER_ZERO,
        .source      = BAL_X86_REGISTER_ZERO,
    };
    bal_sliding_window_push(&compiler->window, test_macro);

    terminate_block_conditional(
        compiler, true == is_tbnz ? BAL_X86_COND_NE : BAL_X86_COND_E, fallthrough_pc, taken_pc);
}

/// Writes the return address of BL and BLR to X30.
static void
write_link_register(bal_tier1_compiler_t *BAL_RESTRICT compiler,
                    const bal_guest_address_t          return_address)
{
    const uint8_t            link_register = 30U;
    const bool               skip_load     = true;
    const bal_x86_register_t x86_link_register
        = allocate_x86_register(compiler, link_register, skip_load);

    const bal_x86_macro_t mov_macro = {
        .opcode              = BAL_X86_MACRO_MOV_REGISTER_IMMEDIATE,
        .destination         = x86_link_register,
        .immediate_or_offset = return_address,
    };
    bal_sliding_window_push(&compiler->window, mov_macro);
    compiler->is_dirty |= 1U << link_register;
}

/// Pushes `x86_register <<= amount` (or >>=, depending on the opcodes) in the 64-bit or 32-bit
/// form. A zero amount emits nothing.
static void
push_shift(bal_tier1_compiler_t *BAL_RESTRICT compiler,
           const bal_x86_macro_opcode_t       opcode_64,
           const bal_x86_macro_opcode_t       opcode_32,
           const bool                         is_32bit,
           const bal_x86_register_t           x86_register,
           const uint32_t                     amount)
{
    if (0U == amount)
    {
        return;
    }

    const bal_x86_macro_t shift_macro = {
        .opcode              = true == is_32bit ? opcode_32 : opcode_64,
        .destination         = x86_register,
        .immediate_or_offset = amount,
    };
    bal_sliding_window_push(&compiler->window, shift_macro);
}

/// Translates SBFM, BFM and UBFM, and with them the LSL, LSR and ASR (immediate), SXTB/SXTH/SXTW,
/// UXTB/UXTH, SBFX/UBFX, SBFIZ/UBFIZ, BFI and BFXIL aliases.
///
/// The field is built in BAL_X86_REGISTER_ZERO with two shifts: a left shift that moves bit imms
/// of Rn to the top of the register, and a right shift (arithmetic for SBFM) that moves the field
/// to its destination. When imms >= immr the field is bits imms:immr of Rn extracted to bit 0,
/// otherwise it is bits imms:0 of Rn inserted at bit datasize - immr. SBFM and UBFM write this to
/// Rd; BFM merges it into Rd under the field mask.
void
translate_bitfield(bal_tier1_compiler_t *BAL_RESTRICT compiler, const uint32_t instruction)
{
    const uint8_t  rd       = (uint8_t)(instruction & 0x1FU);
    const uint8_t  rn       = (uint8_t)((instruction >> 5U) & 0x1FU);
    const uint32_t imms     = (instruction >> 10U) & 0x3FU;
    const uint32_t immr     = (instruction >> 16U) & 0x3FU;
    const uint32_t n        = (instruction >> 22U) & 1U;
    const uint32_t opc      = (instruction >> 29U) & 0x3U;
    const uint32_t sf       = instruction >> 31U;
    const bool     is_32bit = 0U == sf;
    const uint32_t width    = true == is_32bit ? 32U : 64U;

    if (BAL_UNLIKELY(3U == opc) || BAL_UNLIKELY(n != sf)
        || BAL_UNLIKELY(true == is_32bit && (imms > 31U || immr > 31U)))
    {
        BAL_LOG_ERROR(
            &bal_thread_logger, "Aborting function: reserved bitfield encoding %08x", instruction);
        compiler->status = BAL_ERROR_UNKNOWN_INSTRUCTION;
        return;
    }

    // Register 31 is the zero register for both operands.
    if (31U == rd)
    {
        return;
    }

    const bool               is_signed = 0U == opc;
    const bal_x86_register_t field     = BAL_X86_REGISTER_ZERO;

    if (31U == rn)
    {
        const bal_x86_macro_t zero_macro = {
            .opcode              = BAL_X86_MACRO_MOV_REGISTER_IMMEDIATE,
            .destination         = field,
            .immediate_or_offset = 0U,
        };
        bal_sliding_window_push(&compiler->window, zero_macro);
    }
    else
    {
        const bool               skip_load  = false;
        const bal_x86_register_t x86_rn     = allocate_x86_register(compiler, rn, skip_load);
        const bal_x86_macro_t    copy_macro = {
               .opcode      = BAL_X86_MACRO_MOV_REGISTER_REGISTER,
               .destination = field,
               .source      = x86_rn,
        };
        bal_sliding_window_push(&compiler->window, copy_macro);
    }

    const uint32_t left_shift  = width - 1U - imms;
    const uint32_t right_shift = imms >= immr ? left_shift + immr : immr - imms - 1U;
    push_shift(compiler,
               BAL_X86_MACRO_SHL_REGISTER_IMMEDIATE,
               BAL_X86_MACRO_SHL_REGISTER_IMMEDIATE_32,
               is_32bit,
               field,
               left_shift);

    if (true == is_signed)
    {
        push_shift(compiler,
                   BAL_X86_MACRO_SAR_REGISTER_IMMEDIATE,
                   BAL_X86_MACRO_SAR_REGISTER_IMMEDIATE_32,
                   is_32bit,
                   field,
                   right_shift);
    }
    else
    {
        push_shift(compiler,
                   BAL_X86_MACRO_SHR_REGISTER_IMMEDIATE,
                   BAL_X86_MACRO_SHR_REGISTER_IMMEDIATE_32,
                   is_32bit,
                   field,
                   right_shift);
    }

    // A 32-bit field with no shift still holds the upper half of Xn. Clear it the way a write to a
    // W register would.
    if (true == is_32bit && 0U == left_shift && 0U == right_shift)
    {
        const bal_x86_macro_t truncate_macro = {
            .opcode      = BAL_X86_MACRO_AND_REGISTER_REGISTER_32,
            .destination = field,
            .source      = field,
        };
        bal_sliding_window_push(&compiler->window, truncate_macro);
    }

    if (1U == opc)
    {
        // BFM keeps the bits of Rd outside the field.
        const uint32_t field_width    = imms >= immr ? imms - immr + 1U : imms + 1U;
        const uint32_t field_position = imms >= immr ? 0U : width - immr;
        const uint64_t field_ones
            = 64U == field_width ? 0xFFFFFFFFFFFFFFFFULL : (1ULL << field_width) - 1ULL;
        const uint64_t width_mask       = true == is_32bit ? 0xFFFFFFFFULL : 0xFFFFFFFFFFFFFFFFULL;
        const uint64_t keep_mask        = ~(field_ones << field_position) & width_mask;
        const bool     skip_load_rd     = false;
        const bal_x86_register_t x86_rd = allocate_x86_register(compiler, rd, skip_load_rd);

        const bal_x86_macro_t clear_macro = {
            .opcode              = BAL_X86_MACRO_AND_REGISTER_IMMEDIATE,
            .destination         = x86_rd,
            .immediate_or_offset = keep_mask,
        };
        bal_sliding_window_push(&compiler->window, clear_macro);

        const bal_x86_macro_t insert_macro = {
            .opcode      = BAL_X86_MACRO_OR_REGISTER_REGISTER,
            .destination = x86_rd,
            .source      = field,
        };
        bal_sliding_window_push(&compiler->window, insert_macro);
    }
    else
    {
        const bool               skip_load_rd = true;
        const bal_x86_register_t x86_rd       = allocate_x86_register(compiler, rd, skip_load_rd);
        const bal_x86_macro_t    move_macro   = {
                 .opcode      = BAL_X86_MACRO_MOV_REGISTER_REGISTER,
                 .destination = x86_rd,
                 .source      = field,
        };
        bal_sliding_window_push(&compiler->window, move_macro);
    }

    compiler->is_dirty |= 1U << rd;
}

/// Translates EXTR and its ROR (immediate) alias: Rd is bits lsb + datasize - 1:lsb of the
/// concatenation Rn:Rm. The low part is built in BAL_X86_REGISTER_ZERO and the high part in the
/// discard register before they are combined.
void
translate_extract(bal_tier1_compiler_t *BAL_RESTRICT compiler, const uint32_t instruction)
{
    const uint8_t  rd       = (uint8_t)(instruction & 0x1FU);
    const uint8_t  rn       = (uint8_t)((instruction >> 5U) & 0x1FU);
    const uint32_t lsb      = (instruction >> 10U) & 0x3FU;
    const uint8_t  rm       = (uint8_t)((instruction >> 16U) & 0x1FU);
    const uint32_t n        = (instruction >> 22U) & 1U;
    const uint32_t sf       = instruction >> 31U;
    const bool     is_32bit = 0U == sf;
    const uint32_t width    = true == is_32bit ? 32U : 64U;

    // Bits 30:29 and 21 must be zero; N must match sf.
    if (BAL_UNLIKELY(0U != ((instruction >> 29U) & 0x3U))
        || BAL_UNLIKELY(0U != ((instruction >> 21U) & 1U)) || BAL_UNLIKELY(n != sf)
        || BAL_UNLIKELY(lsb >= width))
    {
        BAL_LOG_ERROR(
            &bal_thread_logger, "Aborting function: reserved EXTR encoding %08x", instruction);
        compiler->status = BAL_ERROR_UNKNOWN_INSTRUCTION;
        return;
    }

    if (31U == rd)
    {
        return;
    }

    const bal_x86_register_t low       = BAL_X86_REGISTER_ZERO;
    const bal_x86_register_t high      = BAL_X86_REGISTER_DISCARD_RESULT;
    const bool               skip_load = false;

    // Copy both sources before any of them can be overwritten. Register 31 reads as zero.
    if (31U == rm)
    {
        const bal_x86_macro_t zero_macro = {
            .opcode              = BAL_X86_MACRO_MOV_REGISTER_IMMEDIATE,
            .destination         = low,
            .immediate_or_offset = 0U,
        };
        bal_sliding_window_push(&compiler->window, zero_macro);
    }
    else
    {
        const bal_x86_register_t x86_rm     = allocate_x86_register(compiler, rm, skip_load);
        const bal_x86_macro_t    copy_macro = {
               .opcode      = BAL_X86_MACRO_MOV_REGISTER_REGISTER,
               .destination = low,
               .source      = x86_rm,
        };
        bal_sliding_window_push(&compiler->window, copy_macro);
    }

    if (lsb != 0U)
    {
        if (31U == rn)
        {
            const bal_x86_macro_t zero_macro = {
                .opcode              = BAL_X86_MACRO_MOV_REGISTER_IMMEDIATE,
                .destination         = high,
                .immediate_or_offset = 0U,
            };
            bal_sliding_window_push(&compiler->window, zero_macro);
        }
        else
        {
            const bal_x86_register_t x86_rn     = allocate_x86_register(compiler, rn, skip_load);
            const bal_x86_macro_t    copy_macro = {
                   .opcode      = BAL_X86_MACRO_MOV_REGISTER_REGISTER,
                   .destination = high,
                   .source      = x86_rn,
            };
            bal_sliding_window_push(&compiler->window, copy_macro);
        }

        push_shift(compiler,
                   BAL_X86_MACRO_SHR_REGISTER_IMMEDIATE,
                   BAL_X86_MACRO_SHR_REGISTER_IMMEDIATE_32,
                   is_32bit,
                   low,
                   lsb);
        push_shift(compiler,
                   BAL_X86_MACRO_SHL_REGISTER_IMMEDIATE,
                   BAL_X86_MACRO_SHL_REGISTER_IMMEDIATE_32,
                   is_32bit,
                   high,
                   width - lsb);

        const bal_x86_macro_t combine_macro = {
            .opcode      = BAL_X86_MACRO_OR_REGISTER_REGISTER,
            .destination = low,
            .source      = high,
        };
        bal_sliding_window_push(&compiler->window, combine_macro);
    }
    else if (true == is_32bit)
    {
        const bal_x86_macro_t truncate_macro = {
            .opcode      = BAL_X86_MACRO_AND_REGISTER_REGISTER_32,
            .destination = low,
            .source      = low,
        };
        bal_sliding_window_push(&compiler->window, truncate_macro);
    }

    const bool               skip_load_rd = true;
    const bal_x86_register_t x86_rd       = allocate_x86_register(compiler, rd, skip_load_rd);
    const bal_x86_macro_t    move_macro   = {
             .opcode      = BAL_X86_MACRO_MOV_REGISTER_REGISTER,
             .destination = x86_rd,
             .source      = low,
    };
    bal_sliding_window_push(&compiler->window, move_macro);
    compiler->is_dirty |= 1U << rd;
}

/// Evaluates an A64 condition (0 to 13) from the guest flags into BAL_X86_REGISTER_DISCARD_RESULT
/// and returns the x86 condition that holds after `test r11, r11` when the A64 condition does not.
///
/// The conditions on a single flag load that flag and shift bit 0 to the top, so the register is
/// non-zero exactly when the flag is set. The compound conditions reuse emit_compound_condition(),
/// whose result is zero exactly when HI, GE or GT holds. Odd conditions negate the even ones.
static bal_x86_condition_t
emit_condition_value(bal_tier1_compiler_t *BAL_RESTRICT compiler, const uint32_t condition)
{
    const bal_x86_register_t value  = BAL_X86_REGISTER_DISCARD_RESULT;
    const bool               is_odd = 1U == (condition & 1U);
    bool                     is_zero_when_even_holds;

    if (condition < 8U)
    {
        static const size_t FLAG_OFFSETS[] = {
            offsetof(bal_cpu_t, flag_z),
            offsetof(bal_cpu_t, flag_c),
            offsetof(bal_cpu_t, flag_n),
            offsetof(bal_cpu_t, flag_v),
        };
        const bal_x86_macro_t load_macro = {
            .opcode              = BAL_X86_MACRO_LOAD,
            .destination         = value,
            .immediate_or_offset = FLAG_OFFSETS[(condition >> 1U) & 3U],
        };
        bal_sliding_window_push(&compiler->window, load_macro);

        const bal_x86_macro_t isolate_macro = {
            .opcode              = BAL_X86_MACRO_SHL_REGISTER_IMMEDIATE,
            .destination         = value,
            .immediate_or_offset = 63U,
        };
        bal_sliding_window_push(&compiler->window, isolate_macro);
        is_zero_when_even_holds = false;
    }
    else
    {
        emit_compound_condition(compiler, condition);
        const bal_x86_macro_t copy_macro = {
            .opcode      = BAL_X86_MACRO_MOV_REGISTER_REGISTER,
            .destination = value,
            .source      = BAL_X86_REGISTER_ZERO,
        };
        bal_sliding_window_push(&compiler->window, copy_macro);
        is_zero_when_even_holds = true;
    }

    // The A64 condition fails when the even condition fails (for even conditions) or holds (for
    // odd ones).
    const bool is_zero_when_failing = is_zero_when_even_holds == is_odd;
    return true == is_zero_when_failing ? BAL_X86_COND_E : BAL_X86_COND_NE;
}

/// Translates CSEL, CSINC, CSINV and CSNEG, and with them CSET, CSETM, CINC, CINV and CNEG.
///
/// Rd receives Rn when the condition holds and Rm, Rm + 1, NOT(Rm) or -Rm otherwise. The condition
/// is evaluated first since building the alternative can change the x86 flags. The alternative is
/// built in BAL_X86_REGISTER_ZERO, Rn is copied to Rd, and a `cmov` on the failing condition
/// replaces it.
void
translate_conditional_select(bal_tier1_compiler_t *BAL_RESTRICT compiler,
                             const uint32_t                     instruction)
{
    const uint8_t  rd        = (uint8_t)(instruction & 0x1FU);
    const uint8_t  rn        = (uint8_t)((instruction >> 5U) & 0x1FU);
    const uint32_t op2       = (instruction >> 10U) & 0x3U;
    const uint32_t condition = (instruction >> 12U) & 0xFU;
    const uint8_t  rm        = (uint8_t)((instruction >> 16U) & 0x1FU);
    const uint32_t s_bit     = (instruction >> 29U) & 1U;
    const uint32_t op        = (instruction >> 30U) & 1U;
    const bool     is_32bit  = 0U == (instruction >> 31U);

    if (BAL_UNLIKELY(s_bit != 0U) || BAL_UNLIKELY(op2 > 1U))
    {
        BAL_LOG_ERROR(&bal_thread_logger,
                      "Aborting function: reserved conditional select encoding %08x",
                      instruction);
        compiler->status = BAL_ERROR_UNKNOWN_INSTRUCTION;
        return;
    }

    // Every register operand treats 31 as the zero register.
    if (31U == rd)
    {
        return;
    }

    const bool          is_always      = condition >= 14U;
    bal_x86_condition_t fail_condition = BAL_X86_COND_E;

    if (false == is_always)
    {
        fail_condition = emit_condition_value(compiler, condition);

        const bal_x86_register_t alternative = BAL_X86_REGISTER_ZERO;

        if (31U == rm)
        {
            const bal_x86_macro_t zero_macro = {
                .opcode              = BAL_X86_MACRO_MOV_REGISTER_IMMEDIATE,
                .destination         = alternative,
                .immediate_or_offset = 0U,
            };
            bal_sliding_window_push(&compiler->window, zero_macro);
        }
        else
        {
            const bool               skip_load_rm = false;
            const bal_x86_register_t x86_rm     = allocate_x86_register(compiler, rm, skip_load_rm);
            const bal_x86_macro_t    copy_macro = {
                   .opcode      = BAL_X86_MACRO_MOV_REGISTER_REGISTER,
                   .destination = alternative,
                   .source      = x86_rm,
            };
            bal_sliding_window_push(&compiler->window, copy_macro);
        }

        if (0U == op && 1U == op2)
        {
            const bal_x86_macro_t increment_macro = {
                .opcode              = true == is_32bit ? BAL_X86_MACRO_ADD_REGISTER_IMMEDIATE_32
                                                        : BAL_X86_MACRO_ADD_REGISTER_IMMEDIATE,
                .destination         = alternative,
                .immediate_or_offset = 1U,
            };
            bal_sliding_window_push(&compiler->window, increment_macro);
        }
        else if (1U == op)
        {
            bal_x86_macro_opcode_t opcode;

            if (0U == op2)
            {
                opcode
                    = true == is_32bit ? BAL_X86_MACRO_NOT_REGISTER_32 : BAL_X86_MACRO_NOT_REGISTER;
            }
            else
            {
                opcode
                    = true == is_32bit ? BAL_X86_MACRO_NEG_REGISTER_32 : BAL_X86_MACRO_NEG_REGISTER;
            }

            const bal_x86_macro_t invert_macro = {
                .opcode      = opcode,
                .destination = alternative,
            };
            bal_sliding_window_push(&compiler->window, invert_macro);
        }
    }

    bal_x86_register_t x86_rn = BAL_X86_REGISTER_ZERO;

    if (rn != 31U)
    {
        const bool skip_load_rn = false;
        x86_rn                  = allocate_x86_register(compiler, rn, skip_load_rn);
    }

    const bool               skip_load_rd = true;
    const bal_x86_register_t x86_rd       = allocate_x86_register(compiler, rd, skip_load_rd);

    if (31U == rn)
    {
        const bal_x86_macro_t zero_macro = {
            .opcode              = BAL_X86_MACRO_MOV_REGISTER_IMMEDIATE,
            .destination         = x86_rd,
            .immediate_or_offset = 0U,
        };
        bal_sliding_window_push(&compiler->window, zero_macro);
    }
    else
    {
        const bal_x86_macro_t copy_macro = {
            .opcode      = BAL_X86_MACRO_MOV_REGISTER_REGISTER,
            .destination = x86_rd,
            .source      = x86_rn,
        };
        bal_sliding_window_push(&compiler->window, copy_macro);
    }

    if (true == is_always)
    {
        if (true == is_32bit)
        {
            const bal_x86_macro_t truncate_macro = {
                .opcode      = BAL_X86_MACRO_AND_REGISTER_REGISTER_32,
                .destination = x86_rd,
                .source      = x86_rd,
            };
            bal_sliding_window_push(&compiler->window, truncate_macro);
        }
    }
    else
    {
        const bal_x86_macro_t test_macro = {
            .opcode      = BAL_X86_MACRO_TEST_REGISTER_REGISTER,
            .destination = BAL_X86_REGISTER_DISCARD_RESULT,
            .source      = BAL_X86_REGISTER_DISCARD_RESULT,
        };
        bal_sliding_window_push(&compiler->window, test_macro);

        const bal_x86_macro_t select_macro = {
            .opcode      = true == is_32bit ? BAL_X86_MACRO_CMOV_REGISTER_REGISTER_32
                                            : BAL_X86_MACRO_CMOV_REGISTER_REGISTER,
            .destination = x86_rd,
            .source      = BAL_X86_REGISTER_ZERO,
            .condition   = fail_condition,
        };
        bal_sliding_window_push(&compiler->window, select_macro);
    }

    compiler->is_dirty |= 1U << rd;
}

/// Translates MADD and MSUB (and so MUL and MNEG), and the widening SMADDL, SMSUBL, UMADDL and
/// UMSUBL (and so SMULL, SMNEGL, UMULL and UMNEGL): Rd = Ra +/- Rn * Rm.
///
/// The product is built in BAL_X86_REGISTER_ZERO. The widening forms first sign- or zero-extend
/// Wn into it and Wm into the discard register. The accumulation then goes through
/// emit_binary_operation(), which handles Ra being the zero register and Rd aliasing Ra.
void
translate_multiply_add(bal_tier1_compiler_t *BAL_RESTRICT compiler, const uint32_t instruction)
{
    const uint8_t  rd             = (uint8_t)(instruction & 0x1FU);
    const uint8_t  rn             = (uint8_t)((instruction >> 5U) & 0x1FU);
    const uint8_t  ra             = (uint8_t)((instruction >> 10U) & 0x1FU);
    const bool     is_subtraction = 1U == ((instruction >> 15U) & 1U);
    const uint8_t  rm             = (uint8_t)((instruction >> 16U) & 0x1FU);
    const uint32_t op31           = (instruction >> 21U) & 0x7U;
    const bool     is_64bit       = 1U == (instruction >> 31U);

    // op31 = 000 is MADD/MSUB, 001 SMADDL/SMSUBL and 101 UMADDL/UMSUBL. The widening forms only
    // exist with sf = 1. SMULH and UMULH (010 and 110) are not supported yet.
    const bool is_widening = 1U == op31 || 5U == op31;

    if (BAL_UNLIKELY(op31 != 0U && false == is_widening)
        || BAL_UNLIKELY(true == is_widening && false == is_64bit))
    {
        BAL_LOG_ERROR(&bal_thread_logger,
                      "Aborting function: unsupported multiply encoding %08x",
                      instruction);
        compiler->status = BAL_ERROR_UNKNOWN_INSTRUCTION;
        return;
    }

    // Every register operand treats 31 as the zero register.
    if (31U == rd)
    {
        return;
    }

    const bool               is_32bit  = false == is_64bit;
    const bal_x86_register_t product   = BAL_X86_REGISTER_ZERO;
    const bool               skip_load = false;

    if (31U == rn || 31U == rm)
    {
        const bal_x86_macro_t zero_macro = {
            .opcode              = BAL_X86_MACRO_MOV_REGISTER_IMMEDIATE,
            .destination         = product,
            .immediate_or_offset = 0U,
        };
        bal_sliding_window_push(&compiler->window, zero_macro);
    }
    else
    {
        const bal_x86_register_t x86_rn = allocate_x86_register(compiler, rn, skip_load);
        const bal_x86_register_t x86_rm = allocate_x86_register(compiler, rm, skip_load);

        if (true == is_widening)
        {
            const bal_x86_macro_opcode_t extend_opcode
                = 1U == op31 ? BAL_X86_MACRO_MOVSXD_REGISTER_REGISTER
                             : BAL_X86_MACRO_MOV_REGISTER_REGISTER_32;
            const bal_x86_macro_t extend_rn_macro = {
                .opcode      = extend_opcode,
                .destination = product,
                .source      = x86_rn,
            };
            bal_sliding_window_push(&compiler->window, extend_rn_macro);

            const bal_x86_macro_t extend_rm_macro = {
                .opcode      = extend_opcode,
                .destination = BAL_X86_REGISTER_DISCARD_RESULT,
                .source      = x86_rm,
            };
            bal_sliding_window_push(&compiler->window, extend_rm_macro);

            const bal_x86_macro_t multiply_macro = {
                .opcode      = BAL_X86_MACRO_IMUL_REGISTER_REGISTER,
                .destination = product,
                .source      = BAL_X86_REGISTER_DISCARD_RESULT,
            };
            bal_sliding_window_push(&compiler->window, multiply_macro);
        }
        else
        {
            const bal_x86_macro_t copy_macro = {
                .opcode      = BAL_X86_MACRO_MOV_REGISTER_REGISTER,
                .destination = product,
                .source      = x86_rn,
            };
            bal_sliding_window_push(&compiler->window, copy_macro);

            const bal_x86_macro_t multiply_macro = {
                .opcode      = true == is_32bit ? BAL_X86_MACRO_IMUL_REGISTER_REGISTER_32
                                                : BAL_X86_MACRO_IMUL_REGISTER_REGISTER,
                .destination = product,
                .source      = x86_rm,
            };
            bal_sliding_window_push(&compiler->window, multiply_macro);
        }
    }

    const bool               is_ra_zero   = 31U == ra;
    const bal_x86_register_t x86_ra       = true == is_ra_zero
                                                ? BAL_X86_REGISTER_ZERO
                                                : allocate_x86_register(compiler, ra, skip_load);
    const bool               skip_load_rd = true;
    const bal_x86_register_t x86_rd       = allocate_x86_register(compiler, rd, skip_load_rd);

    bal_x86_macro_opcode_t opcode;

    if (true == is_subtraction)
    {
        opcode = true == is_32bit ? BAL_X86_MACRO_SUB_REGISTER_REGISTER_32
                                  : BAL_X86_MACRO_SUB_REGISTER_REGISTER;
    }
    else
    {
        opcode = true == is_32bit ? BAL_X86_MACRO_ADD_REGISTER_REGISTER_32
                                  : BAL_X86_MACRO_ADD_REGISTER_REGISTER;
    }

    const bool is_commutative = false == is_subtraction;
    emit_binary_operation(compiler, opcode, is_commutative, x86_rd, is_ra_zero, x86_ra, product);
    compiler->is_dirty |= 1U << rd;
}

/// Translates ADD, ADDS, SUB and SUBS (extended register), and so CMP and CMN (extended register):
/// Rd = Rn +/- (extend(Rm) << imm3).
///
/// Rn, and Rd for the forms that do not set flags, are the stack pointer when they encode register
/// 31, while Rm is the zero register. The extended operand is built in BAL_X86_REGISTER_ZERO by
/// shifting the source field to the top of the register and back down (arithmetically for the
/// signed extends), stopping imm3 bits short so the final left shift comes for free.
void
translate_add_sub_extended(bal_tier1_compiler_t *BAL_RESTRICT compiler, const uint32_t instruction)
{
    const uint8_t  rd               = (uint8_t)(instruction & 0x1FU);
    const uint8_t  rn               = (uint8_t)((instruction >> 5U) & 0x1FU);
    const uint32_t shift_amount     = (instruction >> 10U) & 0x7U;
    const uint32_t option           = (instruction >> 13U) & 0x7U;
    const uint8_t  rm               = (uint8_t)((instruction >> 16U) & 0x1FU);
    const uint32_t opt              = (instruction >> 22U) & 0x3U;
    const bool     is_setting_flags = 1U == ((instruction >> 29U) & 1U);
    const bool     is_sub           = 1U == ((instruction >> 30U) & 1U);
    const bool     is_32bit         = 0U == (instruction >> 31U);

    if (BAL_UNLIKELY(opt != 0U) || BAL_UNLIKELY(shift_amount > 4U))
    {
        BAL_LOG_ERROR(&bal_thread_logger,
                      "Aborting function: reserved ADD/SUB (extended register) encoding %08x",
                      instruction);
        compiler->status = BAL_ERROR_UNKNOWN_INSTRUCTION;
        return;
    }

    const bool               skip_load = false;
    const bal_x86_register_t x86_rn    = allocate_x86_register(compiler, rn, skip_load);
    const bal_x86_register_t operand2  = BAL_X86_REGISTER_ZERO;

    if (31U == rm)
    {
        const bal_x86_macro_t zero_macro = {
            .opcode              = BAL_X86_MACRO_MOV_REGISTER_IMMEDIATE,
            .destination         = operand2,
            .immediate_or_offset = 0U,
        };
        bal_sliding_window_push(&compiler->window, zero_macro);
    }
    else
    {
        const bal_x86_register_t x86_rm     = allocate_x86_register(compiler, rm, skip_load);
        const bal_x86_macro_t    copy_macro = {
               .opcode      = BAL_X86_MACRO_MOV_REGISTER_REGISTER,
               .destination = operand2,
               .source      = x86_rm,
        };
        bal_sliding_window_push(&compiler->window, copy_macro);

        // option<1:0> selects a byte, halfword, word or doubleword source; option<2> signedness.
        const uint32_t source_bits = 8U << (option & 0x3U);
        const bool     is_signed   = 4U == (option & 0x4U);

        if (source_bits < 64U)
        {
            const uint32_t top = 64U - source_bits;
            push_shift(compiler,
                       BAL_X86_MACRO_SHL_REGISTER_IMMEDIATE,
                       BAL_X86_MACRO_SHL_REGISTER_IMMEDIATE,
                       false,
                       operand2,
                       top);

            if (true == is_signed)
            {
                push_shift(compiler,
                           BAL_X86_MACRO_SAR_REGISTER_IMMEDIATE,
                           BAL_X86_MACRO_SAR_REGISTER_IMMEDIATE,
                           false,
                           operand2,
                           top - shift_amount);
            }
            else
            {
                push_shift(compiler,
                           BAL_X86_MACRO_SHR_REGISTER_IMMEDIATE,
                           BAL_X86_MACRO_SHR_REGISTER_IMMEDIATE,
                           false,
                           operand2,
                           top - shift_amount);
            }
        }
        else
        {
            push_shift(compiler,
                       BAL_X86_MACRO_SHL_REGISTER_IMMEDIATE,
                       BAL_X86_MACRO_SHL_REGISTER_IMMEDIATE,
                       false,
                       operand2,
                       shift_amount);
        }
    }

    // Rd is the stack pointer unless the instruction sets flags, where register 31 discards the
    // result (CMP and CMN).
    const bool         is_discarded_result = true == is_setting_flags && 31U == rd;
    bal_x86_register_t x86_result          = BAL_X86_REGISTER_DISCARD_RESULT;

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

    const bool is_rn_zero     = false;
    const bool is_commutative = false == is_sub;
    emit_binary_operation(
        compiler, opcode, is_commutative, x86_result, is_rn_zero, x86_rn, operand2);

    if (true == is_setting_flags)
    {
        capture_nzcv(compiler, true == is_sub ? BAL_X86_COND_AE : BAL_X86_COND_B);
    }

    if (false == is_discarded_result)
    {
        compiler->is_dirty |= 1U << rd;
    }
}

/// Translates ADC, ADCS, SBC and SBCS, and so NGC and NGCS: Rd = Rn + Rm + C, or Rn - Rm - NOT(C).
///
/// The guest carry is moved into the x86 carry first, with `bt` on the flag byte, and inverted for
/// the subtractions since x86 SBB subtracts the borrow. The operands are then set up with moves,
/// which leave the x86 flags alone, so ADC or SBB consumes the guest carry.
void
translate_add_sub_carry(bal_tier1_compiler_t *BAL_RESTRICT compiler, const uint32_t instruction)
{
    const uint8_t rd               = (uint8_t)(instruction & 0x1FU);
    const uint8_t rn               = (uint8_t)((instruction >> 5U) & 0x1FU);
    const uint8_t rm               = (uint8_t)((instruction >> 16U) & 0x1FU);
    const bool    is_setting_flags = 1U == ((instruction >> 29U) & 1U);
    const bool    is_sub           = 1U == ((instruction >> 30U) & 1U);
    const bool    is_32bit         = 0U == (instruction >> 31U);

    // Every register operand treats 31 as the zero register.
    const bool is_discarded_result = 31U == rd;

    if (true == is_discarded_result && false == is_setting_flags)
    {
        return;
    }

    const bal_x86_register_t operand2 = BAL_X86_REGISTER_ZERO;

    const bal_x86_macro_t load_carry_macro = {
        .opcode              = BAL_X86_MACRO_LOAD,
        .destination         = operand2,
        .immediate_or_offset = offsetof(bal_cpu_t, flag_c),
    };
    bal_sliding_window_push(&compiler->window, load_carry_macro);

    const bal_x86_macro_t carry_macro = {
        .opcode              = BAL_X86_MACRO_BT_REGISTER_IMMEDIATE,
        .destination         = operand2,
        .immediate_or_offset = 0U,
    };
    bal_sliding_window_push(&compiler->window, carry_macro);

    if (true == is_sub)
    {
        const bal_x86_macro_t invert_macro = {
            .opcode = BAL_X86_MACRO_CMC,
        };
        bal_sliding_window_push(&compiler->window, invert_macro);
    }

    // Only moves from here to the ADC/SBB so the carry survives. Rm is copied first so Rd may
    // alias it.
    const bool skip_load = false;

    if (31U == rm)
    {
        const bal_x86_macro_t zero_macro = {
            .opcode              = BAL_X86_MACRO_MOV_REGISTER_IMMEDIATE,
            .destination         = operand2,
            .immediate_or_offset = 0U,
        };
        bal_sliding_window_push(&compiler->window, zero_macro);
    }
    else
    {
        const bal_x86_register_t x86_rm     = allocate_x86_register(compiler, rm, skip_load);
        const bal_x86_macro_t    copy_macro = {
               .opcode      = BAL_X86_MACRO_MOV_REGISTER_REGISTER,
               .destination = operand2,
               .source      = x86_rm,
        };
        bal_sliding_window_push(&compiler->window, copy_macro);
    }

    const bool               is_rn_zero = 31U == rn;
    const bal_x86_register_t x86_rn     = true == is_rn_zero
                                              ? BAL_X86_REGISTER_ZERO
                                              : allocate_x86_register(compiler, rn, skip_load);
    bal_x86_register_t       x86_result = BAL_X86_REGISTER_DISCARD_RESULT;

    if (false == is_discarded_result)
    {
        const bool skip_load_rd = true;
        x86_result              = allocate_x86_register(compiler, rd, skip_load_rd);
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

    bal_x86_macro_opcode_t opcode;

    if (true == is_sub)
    {
        opcode = true == is_32bit ? BAL_X86_MACRO_SBB_REGISTER_REGISTER_32
                                  : BAL_X86_MACRO_SBB_REGISTER_REGISTER;
    }
    else
    {
        opcode = true == is_32bit ? BAL_X86_MACRO_ADC_REGISTER_REGISTER_32
                                  : BAL_X86_MACRO_ADC_REGISTER_REGISTER;
    }

    const bal_x86_macro_t operation_macro = {
        .opcode      = opcode,
        .destination = x86_result,
        .source      = operand2,
    };
    bal_sliding_window_push(&compiler->window, operation_macro);

    if (true == is_setting_flags)
    {
        capture_nzcv(compiler, true == is_sub ? BAL_X86_COND_AE : BAL_X86_COND_B);
    }

    if (false == is_discarded_result)
    {
        compiler->is_dirty |= 1U << rd;
    }
}
