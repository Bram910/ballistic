#include "backend/x86/bal_x86_tier1_compiler.h"
#include "backend/bal_atomic.h"
#include "backend/bal_cpu.h"
#include "backend/bal_float.h"
#include "backend/bal_pointer_authentication.h"
#include "backend/bal_simd.h"
#include "backend/bal_system.h"
#include "bal_decoder.h"
#include "bal_engine.h"
#include "bal_engine_flags.h"
#include "bal_platform.h"

#include <string.h>

// Set which registers holds the first argument passed to the JIT block based on the OS calling
// convention.
#if BAL_PLATFORM_WINDOWS

#define BAL_X86_ABI_ARG1 BAL_X86_RCX
#define BAL_X86_ABI_ARG2 BAL_X86_RDX
#define BAL_X86_ABI_ARG3 BAL_X86_R8

// The Windows x64 calling convention makes the caller reserve 32 bytes of shadow space.
#define BAL_X86_ABI_SHADOW_SPACE 32U

#else

#define BAL_X86_ABI_ARG1         BAL_X86_RDI
#define BAL_X86_ABI_ARG2         BAL_X86_RSI
#define BAL_X86_ABI_ARG3         BAL_X86_RDX
#define BAL_X86_ABI_SHADOW_SPACE 0U

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

static void emit_block_exit(bal_tier1_compiler_t *compiler);

static size_t emit_forward_jump(bal_x86_assembler_t *assembler,
                                bool                 is_conditional,
                                bal_x86_condition_t  condition);

static void patch_forward_jump(bal_x86_assembler_t *assembler, size_t jump, bool is_conditional);

static void emit_fold_mxcsr_flags(bal_x86_assembler_t *assembler);

static void emit_load_guest_mxcsr(bal_x86_assembler_t *assembler);

static bool emit_fast_float(bal_tier1_compiler_t *compiler, uint32_t instruction);

static uint32_t extract_operand_value(uint32_t instruction, const bal_decoder_operand_t *operand);

static void translate_add_sub_imm(bal_tier1_compiler_t *compiler, uint32_t instruction);

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

static void translate_variable_shift(bal_tier1_compiler_t *compiler, uint32_t instruction);

static void translate_divide(bal_tier1_compiler_t *compiler, uint32_t instruction);

static void translate_conditional_compare(bal_tier1_compiler_t *compiler, uint32_t instruction);

static void translate_multiply_high(bal_tier1_compiler_t *compiler, uint32_t instruction);

static void translate_system_register(bal_tier1_compiler_t *compiler, uint32_t instruction);

static void translate_load_store(bal_tier1_compiler_t         *compiler,
                                 const bal_memory_interface_t *memory_interface,
                                 uint32_t                      instruction,
                                 bal_guest_address_t           guest_address);

static void translate_atomic(bal_tier1_compiler_t         *compiler,
                             const bal_memory_interface_t *memory_interface,
                             uint32_t                      instruction,
                             bal_guest_address_t           guest_address);

static void translate_float(bal_tier1_compiler_t *compiler, uint32_t instruction);

static void translate_integer_helper(bal_tier1_compiler_t *compiler, uint32_t instruction);
static bool translate_pointer_authentication(bal_tier1_compiler_t         *compiler,
                                             const bal_memory_interface_t *memory_interface,
                                             uint32_t                      instruction,
                                             bal_guest_address_t           guest_address);

static void emit_register_helper_call(bal_tier1_compiler_t *compiler,
                                      uint32_t              instruction,
                                      uint64_t              function);

static void emit_guest_exception(bal_tier1_compiler_t *compiler,
                                 bal_exception_t       exception,
                                 uint64_t              immediate);

static bool translate_exception(bal_tier1_compiler_t *compiler,
                                uint32_t              instruction,
                                bal_guest_address_t   guest_address);

static void translate_system(bal_tier1_compiler_t         *compiler,
                             const bal_memory_interface_t *memory_interface,
                             uint32_t                      instruction,
                             bal_guest_address_t           guest_address);

static void translate_data_processing_1_source(bal_tier1_compiler_t *compiler,
                                               uint32_t              instruction);

static void emit_swap_bit_groups(bal_tier1_compiler_t *compiler, uint32_t shift, uint64_t mask);

static void push_macro(bal_tier1_compiler_t  *compiler,
                       bal_x86_macro_opcode_t opcode,
                       bal_x86_register_t     destination,
                       bal_x86_register_t     source);

static void translate_pc_relative(bal_tier1_compiler_t *compiler,
                                  uint32_t              instruction,
                                  bal_guest_address_t   guest_address);

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
    compiler->has_fma       = 0 != __builtin_cpu_supports("fma");
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
    compiler->is_guest_mxcsr_active = false;
    void *host_address              = compiler->assembler.rx_buffer + compiler->assembler.offset;

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

        // A64 instructions are always 4-byte aligned; a misaligned PC, reached through BR, BLR or
        // RET, raises a PC alignment fault. Fetching from it would also be a misaligned read.
        if (guest_address % 4 != 0)
        {
            BAL_LOG_ERROR(&bal_thread_logger,
                          "Aborting function: PC alignment fault at GVA 0x%016llX",
                          (unsigned long long)guest_address);
            compiler->status = BAL_ERROR_PC_ALIGNMENT;
            break;
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
                compiler->status    = BAL_ERROR_UNKNOWN_INSTRUCTION;
                is_block_terminated = true;
                break;
            }

            // The SIMD&FP helpers and the load/store translator check the encodings they
            // accept themselves; other translators do not know 128-bit operands.
            const bool is_checked_by_translator = OPCODE_FLOAT == metadata->ir_opcode
                                                  || OPCODE_LOAD == metadata->ir_opcode
                                                  || OPCODE_STORE == metadata->ir_opcode;

            if (BAL_OPERAND_TYPE_REGISTER_128 == metadata->operands[0].type
                && false == is_checked_by_translator)
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
                // WFI decodes as the generic HINT entry, so match its encoding instead.
                if (BAL_UNLIKELY(0xD503207FU == instruction))
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
                    // ADR and ADRP: bits 28:24 = 10000.
                    if (0x10U == ((instruction >> 24U) & 0x1FU))
                    {
                        translate_pc_relative(compiler, instruction, guest_address);
                        break;
                    }

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

                    // CCMN/CCMP: bits 28:21 = 11010010.
                    if (0xD2U == ((instruction >> 21U) & 0xFFU))
                    {
                        translate_conditional_compare(compiler, instruction);
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
                        translate_add_sub_imm(compiler, instruction);
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
                case OPCODE_NOP:
                    // DSB with the nXS qualifier needs FEAT_XS, which the guest CPU does not
                    // implement, so it is undefined.
                    if (0xD503323FU == (instruction & 0xFFFFF3FFU))
                    {
                        emit_guest_exception(compiler, BAL_EXCEPTION_UNDEFINED_INSTRUCTION, 0U);
                        has_branch_target   = true;
                        branch_target_pc    = guest_address;
                        is_block_terminated = true;
                        break;
                    }

                    // DMB and DSB order memory accesses for other observers. x86 keeps loads and
                    // stores in order except for a load passing an earlier store, so a full fence
                    // is always sufficient. ISB, SB, CLREX and the hints need nothing here.
                    if (0xD503309FU == (instruction & 0xFFFFF0DFU))
                    {
                        const bal_x86_macro_t fence_macro = {
                            .opcode = BAL_X86_MACRO_MFENCE,
                        };
                        bal_sliding_window_push(&compiler->window, fence_macro);
                    }
                    break;
                case OPCODE_LOAD:
                case OPCODE_STORE:
                    translate_load_store(compiler, memory_interface, instruction, guest_address);
                    break;
                case OPCODE_ATOMIC:
                    translate_atomic(compiler, memory_interface, instruction, guest_address);
                    break;
                case OPCODE_FLOAT:
                    translate_float(compiler, instruction);
                    break;
                case OPCODE_INTEGER_HELPER:
                    translate_integer_helper(compiler, instruction);
                    break;
                case OPCODE_POINTER_AUTHENTICATION:
                    // The authenticated branches leave their target in the guest PC.
                    if (true
                        == translate_pointer_authentication(
                            compiler, memory_interface, instruction, guest_address))
                    {
                        is_pc_dynamic       = true;
                        is_block_terminated = true;
                    }
                    break;
                case OPCODE_EXCEPTION:
                    // SVC resumes after itself; BRK and the undefined instructions report their
                    // own address.
                    if (true == translate_exception(compiler, instruction, guest_address))
                    {
                        has_branch_target = true;
                        branch_target_pc  = guest_address;
                    }

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
                case OPCODE_DIV:
                    // UDIV and SDIV: data-processing (2 source) with bits 30:21 = 0011010110
                    // and opcode = 00001x.
                    if (0xD6U == ((instruction >> 21U) & 0x3FFU)
                        && 1U == ((instruction >> 11U) & 0x1FU))
                    {
                        translate_divide(compiler, instruction);
                        break;
                    }

                    BAL_LOG_ERROR(&bal_thread_logger,
                                  "Aborting function: Tier 1 unsupported divide instruction "
                                  "at %08x : %s",
                                  instruction,
                                  metadata->name);
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
                case OPCODE_SYSTEM_REGISTER:
                    // MRS and MSR (register): bits 31:22 = 1101010100 and bit 20 = 1.
                    if (0xD5100000U == (instruction & 0xFFD00000U))
                    {
                        translate_system_register(compiler, instruction);
                        break;
                    }

                    // SYS: bits 31:19 = 1101010100001.
                    if (0xD5080000U == (instruction & 0xFFF80000U))
                    {
                        translate_system(compiler, memory_interface, instruction, guest_address);
                        break;
                    }

                    BAL_LOG_ERROR(&bal_thread_logger,
                                  "Aborting function: Tier 1 unsupported system instruction at "
                                  "%08x : %s",
                                  instruction,
                                  metadata->name);
                    compiler->status    = BAL_ERROR_UNKNOWN_INSTRUCTION;
                    is_block_terminated = true;
                    break;
                case OPCODE_REVERSE:
                case OPCODE_COUNT_LEADING:
                    // Data-processing (1 source): bits 30:21 = 1011010110 and opcode2 = 00000.
                    if (0x2D6U == ((instruction >> 21U) & 0x3FFU)
                        && 0U == ((instruction >> 16U) & 0x1FU))
                    {
                        translate_data_processing_1_source(compiler, instruction);
                        break;
                    }

                    BAL_LOG_ERROR(&bal_thread_logger,
                                  "Aborting function: Tier 1 unsupported instruction at %08x : %s",
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

                    // LSLV, LSRV, ASRV and RORV: data-processing (2 source) with bits 30:21 =
                    // 0011010110 and opcode = 0010xx.
                    if (0xD6U == ((instruction >> 21U) & 0x3FFU)
                        && 0x2U == ((instruction >> 12U) & 0xFU))
                    {
                        translate_variable_shift(compiler, instruction);
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
    emit_block_exit(compiler);
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

    // The jump over the fallthrough exit is patched once the exit's size is known.
    bal_x86_assembler_t *BAL_RESTRICT assembler = &compiler->assembler;
    const size_t                      jump      = emit_forward_jump(assembler, true, condition);

    flush_dirty_registers(compiler);
    bal_sliding_window_flush_all(&compiler->window);

    // Emit fallthrough epilogue.
    bal_x86_emit_mov_r64_imm64(&compiler->assembler, BAL_X86_RAX, fallthrough_pc);
    bal_x86_emit_store_r64_rbp_offset(&compiler->assembler, BAL_X86_RAX, offsetof(bal_cpu_t, pc));
    emit_block_exit(compiler);
    patch_forward_jump(assembler, jump, true);
    flush_dirty_registers(compiler);
    bal_sliding_window_flush_all(&compiler->window);

    // Emit taken epilogue.
    bal_x86_emit_mov_r64_imm64(&compiler->assembler, BAL_X86_RAX, taken_pc);
    bal_x86_emit_store_r64_rbp_offset(&compiler->assembler, BAL_X86_RAX, offsetof(bal_cpu_t, pc));
    emit_block_exit(compiler);
}

/// Restores the callee-saved registers pushed by the block prologue and returns to the host.
/// Emits a jump with a zero 32-bit displacement, a JCC on `condition` when `is_conditional`, and
/// returns its offset for patch_forward_jump().
static size_t
emit_forward_jump(bal_x86_assembler_t *BAL_RESTRICT assembler,
                  const bool                        is_conditional,
                  const bal_x86_condition_t         condition)
{
    const size_t offset = assembler->offset;

    if (true == is_conditional)
    {
        bal_x86_emit_jcc_rel32(assembler, condition, 0);
    }
    else
    {
        bal_x86_emit_jmp_rel32(assembler, 0);
    }

    return offset;
}

/// Points the jump emitted at `jump` by emit_forward_jump() to the current offset.
static void
patch_forward_jump(bal_x86_assembler_t *BAL_RESTRICT assembler,
                   const size_t                      jump,
                   const bool                        is_conditional)
{
    if (assembler->status != BAL_SUCCESS)
    {
        return;
    }

    // JCC rel32 is 0F 8x and four bytes; JMP rel32 is E9 and four bytes.
    const size_t   field    = jump + (true == is_conditional ? 2U : 1U);
    const uint32_t distance = (uint32_t)(assembler->offset - (field + 4U));
    uint8_t       *bytes    = assembler->buffer + field;
    bytes[0]                = (uint8_t)distance;
    bytes[1]                = (uint8_t)(distance >> 8U);
    bytes[2]                = (uint8_t)(distance >> 16U);
    bytes[3]                = (uint8_t)(distance >> 24U);
}

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

/// Translates ADD, ADDS, SUB and SUBS (immediate): sf op S 100010 sh imm12 Rn Rd. The fields come
/// from the encoding, since the decoder's alias entries (CMP, CMN, MOV to or from SP) list fewer
/// operands.
void
translate_add_sub_imm(bal_tier1_compiler_t *BAL_RESTRICT compiler, const uint32_t instruction)
{
    const uint8_t            rd                  = (uint8_t)(instruction & 0x1FU);
    const uint8_t            rn                  = (uint8_t)((instruction >> 5U) & 0x1FU);
    const uint32_t           imm12               = (instruction >> 10U) & 0xFFFU;
    const uint64_t           shift_amount        = 1U == ((instruction >> 22U) & 1U) ? 12U : 0U;
    uint64_t                 value               = (uint64_t)imm12 << shift_amount;
    const bool               is_sub              = 1U == ((instruction >> 30U) & 1U);
    const bool               is_setting_flags    = 1U == ((instruction >> 29U) & 1U);
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
    const bool             is_32bit = 0U == (instruction >> 31U);
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
            emit_block_exit(compiler);
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
    uint64_t   bit_mask            = 0U;

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

    // Only once the encoding is known to be allocated: a write to the zero register that sets no
    // flags does nothing.
    if (true == is_discarded_result && false == is_setting_flags)
    {
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

    // op31 = 000 is MADD/MSUB, 001 SMADDL/SMSUBL, 101 UMADDL/UMSUBL, 010 SMULH and 110 UMULH. All
    // but MADD/MSUB only exist with sf = 1.
    const bool is_widening = 1U == op31 || 5U == op31;

    if ((2U == op31 || 6U == op31) && true == is_64bit && false == is_subtraction)
    {
        translate_multiply_high(compiler, instruction);
        return;
    }

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

/// Translates LSLV, LSRV, ASRV and RORV, and so their LSL, LSR, ASR and ROR (register) aliases.
///
/// x86 only takes a variable shift count in CL, and RCX is part of the scratch pool, so RCX is
/// parked in the discard register while CL holds the count and restored afterwards. The value is
/// shifted in BAL_X86_REGISTER_ZERO. x86 masks the count modulo the operand size like A64 does.
void
translate_variable_shift(bal_tier1_compiler_t *BAL_RESTRICT compiler, const uint32_t instruction)
{
    const uint8_t  rd         = (uint8_t)(instruction & 0x1FU);
    const uint8_t  rn         = (uint8_t)((instruction >> 5U) & 0x1FU);
    const uint32_t shift_type = (instruction >> 10U) & 0x3U;
    const uint8_t  rm         = (uint8_t)((instruction >> 16U) & 0x1FU);
    const bool     is_32bit   = 0U == (instruction >> 31U);

    // Every register operand treats 31 as the zero register.
    if (31U == rd)
    {
        return;
    }

    const bal_x86_register_t value     = BAL_X86_REGISTER_ZERO;
    const bool               skip_load = false;

    if (31U == rn)
    {
        const bal_x86_macro_t zero_macro = {
            .opcode              = BAL_X86_MACRO_MOV_REGISTER_IMMEDIATE,
            .destination         = value,
            .immediate_or_offset = 0U,
        };
        bal_sliding_window_push(&compiler->window, zero_macro);
    }
    else
    {
        const bal_x86_register_t x86_rn     = allocate_x86_register(compiler, rn, skip_load);
        const bal_x86_macro_t    copy_macro = {
               .opcode      = BAL_X86_MACRO_MOV_REGISTER_REGISTER,
               .destination = value,
               .source      = x86_rn,
        };
        bal_sliding_window_push(&compiler->window, copy_macro);
    }

    if (31U == rm)
    {
        // A zero count leaves Rn unchanged, apart from truncating it to 32 bits.
        if (true == is_32bit)
        {
            const bal_x86_macro_t truncate_macro = {
                .opcode      = BAL_X86_MACRO_AND_REGISTER_REGISTER_32,
                .destination = value,
                .source      = value,
            };
            bal_sliding_window_push(&compiler->window, truncate_macro);
        }
    }
    else
    {
        static const bal_x86_macro_opcode_t SHIFT_OPCODES_64[] = {
            BAL_X86_MACRO_SHL_REGISTER_CL,
            BAL_X86_MACRO_SHR_REGISTER_CL,
            BAL_X86_MACRO_SAR_REGISTER_CL,
            BAL_X86_MACRO_ROR_REGISTER_CL,
        };
        static const bal_x86_macro_opcode_t SHIFT_OPCODES_32[] = {
            BAL_X86_MACRO_SHL_REGISTER_CL_32,
            BAL_X86_MACRO_SHR_REGISTER_CL_32,
            BAL_X86_MACRO_SAR_REGISTER_CL_32,
            BAL_X86_MACRO_ROR_REGISTER_CL_32,
        };
        const bal_x86_register_t x86_rm    = allocate_x86_register(compiler, rm, skip_load);
        const bal_x86_register_t saved_rcx = BAL_X86_REGISTER_DISCARD_RESULT;

        const bal_x86_macro_t save_macro = {
            .opcode      = BAL_X86_MACRO_MOV_REGISTER_REGISTER,
            .destination = saved_rcx,
            .source      = BAL_X86_RCX,
        };
        bal_sliding_window_push(&compiler->window, save_macro);

        const bal_x86_macro_t count_macro = {
            .opcode      = BAL_X86_MACRO_MOV_REGISTER_REGISTER,
            .destination = BAL_X86_RCX,
            .source      = x86_rm,
        };
        bal_sliding_window_push(&compiler->window, count_macro);

        const bal_x86_macro_t shift_macro = {
            .opcode
            = true == is_32bit ? SHIFT_OPCODES_32[shift_type] : SHIFT_OPCODES_64[shift_type],
            .destination = value,
        };
        bal_sliding_window_push(&compiler->window, shift_macro);

        const bal_x86_macro_t restore_macro = {
            .opcode      = BAL_X86_MACRO_MOV_REGISTER_REGISTER,
            .destination = BAL_X86_RCX,
            .source      = saved_rcx,
        };
        bal_sliding_window_push(&compiler->window, restore_macro);
    }

    const bool               skip_load_rd = true;
    const bal_x86_register_t x86_rd       = allocate_x86_register(compiler, rd, skip_load_rd);
    const bal_x86_macro_t    move_macro   = {
             .opcode      = BAL_X86_MACRO_MOV_REGISTER_REGISTER,
             .destination = x86_rd,
             .source      = value,
    };
    bal_sliding_window_push(&compiler->window, move_macro);
    compiler->is_dirty |= 1U << rd;
}

/// Translates ADR and ADRP. Both compute an address relative to the instruction, which is known at
/// translation time, so Rd receives a constant: PC + imm for ADR, and the 4 KiB page of PC plus
/// imm * 4096 for ADRP.
void
translate_pc_relative(bal_tier1_compiler_t *BAL_RESTRICT compiler,
                      const uint32_t                     instruction,
                      const bal_guest_address_t          guest_address)
{
    const uint8_t rd = (uint8_t)(instruction & 0x1FU);

    // Rd is the zero register when it encodes 31.
    if (31U == rd)
    {
        return;
    }

    const uint32_t immhi   = (instruction >> 5U) & 0x7FFFFU;
    const uint32_t immlo   = (instruction >> 29U) & 0x3U;
    const bool     is_page = 1U == (instruction >> 31U);

    // WARNING: Shifting the 21-bit immediate to the top and back sign-extends it.
    const int32_t signed_immediate = (int32_t)(((immhi << 2U) | immlo) << 11U) >> 11U;
    uint64_t      address;

    if (true == is_page)
    {
        address = (guest_address & ~0xFFFULL) + (uint64_t)((int64_t)signed_immediate * 4096);
    }
    else
    {
        address = guest_address + (uint64_t)(int64_t)signed_immediate;
    }

    const bool               skip_load = true;
    const bal_x86_register_t x86_rd    = allocate_x86_register(compiler, rd, skip_load);
    const bal_x86_macro_t    mov_macro = {
           .opcode              = BAL_X86_MACRO_MOV_REGISTER_IMMEDIATE,
           .destination         = x86_rd,
           .immediate_or_offset = address,
    };
    bal_sliding_window_push(&compiler->window, mov_macro);
    compiler->is_dirty |= 1U << rd;
}

/// Pushes a register to register macro. A shorthand for the long instruction sequences below.
static void
push_macro(bal_tier1_compiler_t *BAL_RESTRICT compiler,
           const bal_x86_macro_opcode_t       opcode,
           const bal_x86_register_t           destination,
           const bal_x86_register_t           source)
{
    const bal_x86_macro_t macro = {
        .opcode      = opcode,
        .destination = destination,
        .source      = source,
    };
    bal_sliding_window_push(&compiler->window, macro);
}

/// Translates UDIV and SDIV.
///
/// x86 divides RDX:RAX and raises #DE on a zero divisor and on INT_MIN / -1, where A64 returns 0
/// and INT_MIN. Both cases are steered to a harmless division before it runs:
///
/// - a zero divisor becomes 0 / 1, which is 0,
/// - for SDIV, a divisor of -1 becomes -dividend / 1, which is -x for every x and wraps INT_MIN to
///   itself like A64.
///
/// RAX and RDX belong to the scratch pool and may hold guest registers, so they are saved on the
/// stack around the division. The dividend and divisor are copied out first so the operands may
/// live in RAX or RDX.
void
translate_divide(bal_tier1_compiler_t *BAL_RESTRICT compiler, const uint32_t instruction)
{
    const uint8_t rd        = (uint8_t)(instruction & 0x1FU);
    const uint8_t rn        = (uint8_t)((instruction >> 5U) & 0x1FU);
    const bool    is_signed = 1U == ((instruction >> 10U) & 1U);
    const uint8_t rm        = (uint8_t)((instruction >> 16U) & 0x1FU);
    const bool    is_32bit  = 0U == (instruction >> 31U);

    // Every register operand treats 31 as the zero register.
    if (31U == rd)
    {
        return;
    }

    const bal_x86_register_t dividend        = BAL_X86_REGISTER_ZERO;
    const bal_x86_register_t divisor         = BAL_X86_REGISTER_DISCARD_RESULT;
    const bool               skip_load       = false;
    const uint8_t            operands[2]     = { rn, rm };
    const bal_x86_register_t destinations[2] = { dividend, divisor };

    for (size_t i = 0U; i < 2U; ++i)
    {
        if (31U == operands[i])
        {
            const bal_x86_macro_t zero_macro = {
                .opcode              = BAL_X86_MACRO_MOV_REGISTER_IMMEDIATE,
                .destination         = destinations[i],
                .immediate_or_offset = 0U,
            };
            bal_sliding_window_push(&compiler->window, zero_macro);
        }
        else
        {
            const bal_x86_register_t x86_operand
                = allocate_x86_register(compiler, operands[i], skip_load);
            push_macro(compiler, BAL_X86_MACRO_MOV_REGISTER_REGISTER, destinations[i], x86_operand);
        }
    }

    push_macro(compiler, BAL_X86_MACRO_PUSH_REGISTER, BAL_X86_INVALID, BAL_X86_RAX);
    push_macro(compiler, BAL_X86_MACRO_PUSH_REGISTER, BAL_X86_INVALID, BAL_X86_RDX);
    push_macro(compiler, BAL_X86_MACRO_MOV_REGISTER_REGISTER, BAL_X86_RAX, dividend);

    const bal_x86_macro_t zero_macro = {
        .opcode              = BAL_X86_MACRO_MOV_REGISTER_IMMEDIATE,
        .destination         = dividend,
        .immediate_or_offset = 0U,
    };
    const bal_x86_macro_t one_macro = {
        .opcode              = BAL_X86_MACRO_MOV_REGISTER_IMMEDIATE,
        .destination         = dividend,
        .immediate_or_offset = 1U,
    };

    // A zero divisor: divide 0 by 1. Moves leave ZF from the TEST intact.
    bal_sliding_window_push(&compiler->window, zero_macro);
    push_macro(compiler,
               true == is_32bit ? BAL_X86_MACRO_TEST_REGISTER_REGISTER_32
                                : BAL_X86_MACRO_TEST_REGISTER_REGISTER,
               divisor,
               divisor);

    const bal_x86_macro_t select_zero_dividend_macro = {
        .opcode      = BAL_X86_MACRO_CMOV_REGISTER_REGISTER,
        .destination = BAL_X86_RAX,
        .source      = dividend,
        .condition   = BAL_X86_COND_E,
    };
    bal_sliding_window_push(&compiler->window, select_zero_dividend_macro);
    bal_sliding_window_push(&compiler->window, one_macro);

    const bal_x86_macro_t select_one_divisor_macro = {
        .opcode      = BAL_X86_MACRO_CMOV_REGISTER_REGISTER,
        .destination = divisor,
        .source      = dividend,
        .condition   = BAL_X86_COND_E,
    };
    bal_sliding_window_push(&compiler->window, select_one_divisor_macro);

    if (true == is_signed)
    {
        // A divisor of -1: divide -dividend by 1. RDX is free until the sign extension.
        push_macro(compiler, BAL_X86_MACRO_MOV_REGISTER_REGISTER, dividend, BAL_X86_RAX);
        push_macro(compiler,
                   true == is_32bit ? BAL_X86_MACRO_NEG_REGISTER_32 : BAL_X86_MACRO_NEG_REGISTER,
                   dividend,
                   BAL_X86_INVALID);
        push_macro(compiler, BAL_X86_MACRO_MOV_REGISTER_REGISTER, BAL_X86_RDX, divisor);

        const bal_x86_macro_t increment_macro = {
            .opcode              = true == is_32bit ? BAL_X86_MACRO_ADD_REGISTER_IMMEDIATE_32
                                                    : BAL_X86_MACRO_ADD_REGISTER_IMMEDIATE,
            .destination         = BAL_X86_RDX,
            .immediate_or_offset = 1U,
        };
        bal_sliding_window_push(&compiler->window, increment_macro);

        const bal_x86_macro_t select_negated_dividend_macro = {
            .opcode      = BAL_X86_MACRO_CMOV_REGISTER_REGISTER,
            .destination = BAL_X86_RAX,
            .source      = dividend,
            .condition   = BAL_X86_COND_E,
        };
        bal_sliding_window_push(&compiler->window, select_negated_dividend_macro);
        bal_sliding_window_push(&compiler->window, one_macro);
        bal_sliding_window_push(&compiler->window, select_one_divisor_macro);

        push_macro(compiler,
                   true == is_32bit ? BAL_X86_MACRO_CDQ : BAL_X86_MACRO_CQO,
                   BAL_X86_INVALID,
                   BAL_X86_INVALID);
        push_macro(compiler,
                   true == is_32bit ? BAL_X86_MACRO_IDIV_REGISTER_32 : BAL_X86_MACRO_IDIV_REGISTER,
                   BAL_X86_INVALID,
                   divisor);
    }
    else
    {
        const bal_x86_macro_t clear_rdx_macro = {
            .opcode              = BAL_X86_MACRO_MOV_REGISTER_IMMEDIATE,
            .destination         = BAL_X86_RDX,
            .immediate_or_offset = 0U,
        };
        bal_sliding_window_push(&compiler->window, clear_rdx_macro);
        push_macro(compiler,
                   true == is_32bit ? BAL_X86_MACRO_DIV_REGISTER_32 : BAL_X86_MACRO_DIV_REGISTER,
                   BAL_X86_INVALID,
                   divisor);
    }

    // The 32-bit forms write EAX, which zero-extends into RAX.
    push_macro(compiler, BAL_X86_MACRO_MOV_REGISTER_REGISTER, dividend, BAL_X86_RAX);
    push_macro(compiler, BAL_X86_MACRO_POP_REGISTER, BAL_X86_RDX, BAL_X86_INVALID);
    push_macro(compiler, BAL_X86_MACRO_POP_REGISTER, BAL_X86_RAX, BAL_X86_INVALID);

    const bool               skip_load_rd = true;
    const bal_x86_register_t x86_rd       = allocate_x86_register(compiler, rd, skip_load_rd);
    push_macro(compiler, BAL_X86_MACRO_MOV_REGISTER_REGISTER, x86_rd, dividend);
    compiler->is_dirty |= 1U << rd;
}

/// Translates CCMN and CCMP in their register and immediate forms: when the condition holds, NZCV
/// is set by comparing Rn with the second operand (adding it for CCMN), otherwise NZCV is set to
/// the immediate nzcv field.
///
/// The condition is evaluated on the old flags first. The comparison then runs in
/// BAL_X86_REGISTER_ZERO and its flags are stored as usual. Finally the four flag bytes are loaded
/// as one word, together with the exception field after them, and replaced by the immediate flags
/// with a `cmov` when the condition failed. AL and NV always compare.
void
translate_conditional_compare(bal_tier1_compiler_t *BAL_RESTRICT compiler,
                              const uint32_t                     instruction)
{
    const uint32_t nzcv         = instruction & 0xFU;
    const uint32_t o3           = (instruction >> 4U) & 1U;
    const uint8_t  rn           = (uint8_t)((instruction >> 5U) & 0x1FU);
    const uint32_t o2           = (instruction >> 10U) & 1U;
    const bool     is_immediate = 1U == ((instruction >> 11U) & 1U);
    const uint32_t condition    = (instruction >> 12U) & 0xFU;
    const uint32_t operand2     = (instruction >> 16U) & 0x1FU;
    const uint32_t s_bit        = (instruction >> 29U) & 1U;
    const bool     is_sub       = 1U == ((instruction >> 30U) & 1U);
    const bool     is_32bit     = 0U == (instruction >> 31U);

    if (BAL_UNLIKELY(s_bit != 1U) || BAL_UNLIKELY(o2 != 0U) || BAL_UNLIKELY(o3 != 0U))
    {
        BAL_LOG_ERROR(&bal_thread_logger,
                      "Aborting function: reserved conditional compare encoding %08x",
                      instruction);
        compiler->status = BAL_ERROR_UNKNOWN_INSTRUCTION;
        return;
    }

    const bool          is_always      = condition >= 14U;
    bal_x86_condition_t fail_condition = BAL_X86_COND_E;

    if (false == is_always)
    {
        fail_condition = emit_condition_value(compiler, condition);
    }

    // Compare in the zero register. Register 31 is the zero register for Rn and Rm.
    const bal_x86_register_t comparison = BAL_X86_REGISTER_ZERO;
    const bool               skip_load  = false;

    if (31U == rn)
    {
        const bal_x86_macro_t zero_macro = {
            .opcode              = BAL_X86_MACRO_MOV_REGISTER_IMMEDIATE,
            .destination         = comparison,
            .immediate_or_offset = 0U,
        };
        bal_sliding_window_push(&compiler->window, zero_macro);
    }
    else
    {
        const bal_x86_register_t x86_rn = allocate_x86_register(compiler, rn, skip_load);
        push_macro(compiler, BAL_X86_MACRO_MOV_REGISTER_REGISTER, comparison, x86_rn);
    }

    if (true == is_immediate || 31U == operand2)
    {
        const uint64_t        immediate     = true == is_immediate ? operand2 : 0U;
        const bal_x86_macro_t compare_macro = {
            .opcode = true == is_sub ? (true == is_32bit ? BAL_X86_MACRO_SUB_REGISTER_IMMEDIATE_32
                                                         : BAL_X86_MACRO_SUB_REGISTER_IMMEDIATE)
                                     : (true == is_32bit ? BAL_X86_MACRO_ADD_REGISTER_IMMEDIATE_32
                                                         : BAL_X86_MACRO_ADD_REGISTER_IMMEDIATE),
            .destination         = comparison,
            .immediate_or_offset = immediate,
        };
        bal_sliding_window_push(&compiler->window, compare_macro);
    }
    else
    {
        const bal_x86_register_t x86_rm
            = allocate_x86_register(compiler, (uint8_t)operand2, skip_load);
        const bal_x86_macro_opcode_t opcode
            = true == is_sub ? (true == is_32bit ? BAL_X86_MACRO_SUB_REGISTER_REGISTER_32
                                                 : BAL_X86_MACRO_SUB_REGISTER_REGISTER)
                             : (true == is_32bit ? BAL_X86_MACRO_ADD_REGISTER_REGISTER_32
                                                 : BAL_X86_MACRO_ADD_REGISTER_REGISTER);
        push_macro(compiler, opcode, comparison, x86_rm);
    }

    capture_nzcv(compiler, true == is_sub ? BAL_X86_COND_AE : BAL_X86_COND_B);

    if (true == is_always)
    {
        return;
    }

    // flag_c, flag_z, flag_n and flag_v are consecutive bytes followed by the 32-bit exception
    // field, which is zero while a block runs, so one 64-bit load and store covers exactly them.
    const uint64_t immediate_flags
        = (uint64_t)((nzcv >> 1U) & 1U) | ((uint64_t)((nzcv >> 2U) & 1U) << 8U)
          | ((uint64_t)((nzcv >> 3U) & 1U) << 16U) | ((uint64_t)(nzcv & 1U) << 24U);
    const bal_x86_register_t condition_value = BAL_X86_REGISTER_DISCARD_RESULT;

    const bal_x86_macro_t load_flags_macro = {
        .opcode              = BAL_X86_MACRO_LOAD,
        .destination         = comparison,
        .immediate_or_offset = offsetof(bal_cpu_t, flag_c),
    };
    bal_sliding_window_push(&compiler->window, load_flags_macro);

    // The condition value is only needed for its zero flag, so the register is reused for the
    // immediate flags right after the TEST. Moves leave the flags alone.
    push_macro(compiler, BAL_X86_MACRO_TEST_REGISTER_REGISTER, condition_value, condition_value);

    const bal_x86_macro_t immediate_flags_macro = {
        .opcode              = BAL_X86_MACRO_MOV_REGISTER_IMMEDIATE,
        .destination         = condition_value,
        .immediate_or_offset = immediate_flags,
    };
    bal_sliding_window_push(&compiler->window, immediate_flags_macro);

    const bal_x86_macro_t select_macro = {
        .opcode      = BAL_X86_MACRO_CMOV_REGISTER_REGISTER,
        .destination = comparison,
        .source      = condition_value,
        .condition   = fail_condition,
    };
    bal_sliding_window_push(&compiler->window, select_macro);

    const bal_x86_macro_t store_flags_macro = {
        .opcode              = BAL_X86_MACRO_STORE,
        .source              = comparison,
        .immediate_or_offset = offsetof(bal_cpu_t, flag_c),
    };
    bal_sliding_window_push(&compiler->window, store_flags_macro);
}

/// Translates SMULH and UMULH, the upper 64 bits of the 128-bit product of Xn and Xm.
///
/// The one-operand x86 multiply leaves that half in RDX, so RAX and RDX are saved on the stack
/// around it, as for the division. The operands are copied out first so they may live in RAX or
/// RDX.
void
translate_multiply_high(bal_tier1_compiler_t *BAL_RESTRICT compiler, const uint32_t instruction)
{
    const uint8_t rd        = (uint8_t)(instruction & 0x1FU);
    const uint8_t rn        = (uint8_t)((instruction >> 5U) & 0x1FU);
    const uint8_t rm        = (uint8_t)((instruction >> 16U) & 0x1FU);
    const bool    is_signed = 0U == ((instruction >> 23U) & 1U);

    // Every register operand treats 31 as the zero register.
    if (31U == rd)
    {
        return;
    }

    const bal_x86_register_t multiplicand    = BAL_X86_REGISTER_ZERO;
    const bal_x86_register_t multiplier      = BAL_X86_REGISTER_DISCARD_RESULT;
    const uint8_t            operands[2]     = { rn, rm };
    const bal_x86_register_t destinations[2] = { multiplicand, multiplier };
    const bool               skip_load       = false;

    for (size_t i = 0U; i < 2U; ++i)
    {
        if (31U == operands[i])
        {
            const bal_x86_macro_t zero_macro = {
                .opcode              = BAL_X86_MACRO_MOV_REGISTER_IMMEDIATE,
                .destination         = destinations[i],
                .immediate_or_offset = 0U,
            };
            bal_sliding_window_push(&compiler->window, zero_macro);
        }
        else
        {
            const bal_x86_register_t x86_operand
                = allocate_x86_register(compiler, operands[i], skip_load);
            push_macro(compiler, BAL_X86_MACRO_MOV_REGISTER_REGISTER, destinations[i], x86_operand);
        }
    }

    push_macro(compiler, BAL_X86_MACRO_PUSH_REGISTER, BAL_X86_INVALID, BAL_X86_RAX);
    push_macro(compiler, BAL_X86_MACRO_PUSH_REGISTER, BAL_X86_INVALID, BAL_X86_RDX);
    push_macro(compiler, BAL_X86_MACRO_MOV_REGISTER_REGISTER, BAL_X86_RAX, multiplicand);
    push_macro(compiler,
               true == is_signed ? BAL_X86_MACRO_IMUL_WIDENING : BAL_X86_MACRO_MUL_WIDENING,
               BAL_X86_INVALID,
               multiplier);
    push_macro(compiler, BAL_X86_MACRO_MOV_REGISTER_REGISTER, multiplicand, BAL_X86_RDX);
    push_macro(compiler, BAL_X86_MACRO_POP_REGISTER, BAL_X86_RDX, BAL_X86_INVALID);
    push_macro(compiler, BAL_X86_MACRO_POP_REGISTER, BAL_X86_RAX, BAL_X86_INVALID);

    const bool               skip_load_rd = true;
    const bal_x86_register_t x86_rd       = allocate_x86_register(compiler, rd, skip_load_rd);
    push_macro(compiler, BAL_X86_MACRO_MOV_REGISTER_REGISTER, x86_rd, multiplicand);
    compiler->is_dirty |= 1U << rd;
}

/// Swaps adjacent groups of `shift` bits of BAL_X86_REGISTER_ZERO selected by `mask`:
/// value = ((value >> shift) & mask) | ((value & mask) << shift). RAX is saved on the stack and
/// used as the second temporary, since the AND immediate macro needs the discard register.
static void
emit_swap_bit_groups(bal_tier1_compiler_t *BAL_RESTRICT compiler,
                     const uint32_t                     shift,
                     const uint64_t                     mask)
{
    const bal_x86_register_t value = BAL_X86_REGISTER_ZERO;
    push_macro(compiler, BAL_X86_MACRO_PUSH_REGISTER, BAL_X86_INVALID, BAL_X86_RAX);
    push_macro(compiler, BAL_X86_MACRO_MOV_REGISTER_REGISTER, BAL_X86_RAX, value);
    push_shift(compiler,
               BAL_X86_MACRO_SHR_REGISTER_IMMEDIATE,
               BAL_X86_MACRO_SHR_REGISTER_IMMEDIATE,
               false,
               BAL_X86_RAX,
               shift);

    const bal_x86_macro_t mask_high_macro = {
        .opcode              = BAL_X86_MACRO_AND_REGISTER_IMMEDIATE,
        .destination         = BAL_X86_RAX,
        .immediate_or_offset = mask,
    };
    bal_sliding_window_push(&compiler->window, mask_high_macro);

    const bal_x86_macro_t mask_low_macro = {
        .opcode              = BAL_X86_MACRO_AND_REGISTER_IMMEDIATE,
        .destination         = value,
        .immediate_or_offset = mask,
    };
    bal_sliding_window_push(&compiler->window, mask_low_macro);
    push_shift(compiler,
               BAL_X86_MACRO_SHL_REGISTER_IMMEDIATE,
               BAL_X86_MACRO_SHL_REGISTER_IMMEDIATE,
               false,
               value,
               shift);
    push_macro(compiler, BAL_X86_MACRO_OR_REGISTER_REGISTER, value, BAL_X86_RAX);
    push_macro(compiler, BAL_X86_MACRO_POP_REGISTER, BAL_X86_RAX, BAL_X86_INVALID);
}

/// Translates the data-processing (1 source) instructions RBIT, REV16, REV32, REV, CLZ and CLS.
/// The value is transformed in BAL_X86_REGISTER_ZERO and then moved to Rd.
///
/// - REV and REV32 are byte swaps, REV32 followed by a 32-bit rotate that puts the words back in
///   order. REV16 swaps the bytes of each halfword and RBIT swaps nibbles, bit pairs and bits
///   after a byte swap, with emit_swap_bit_groups().
/// - CLZ uses BSR, whose result is converted with index XOR (datasize - 1). For a zero input BSR
///   sets ZF and the index is replaced first with a value that the XOR turns into datasize.
/// - CLS is CLZ(x XOR (x >> 1)) - 1 with an arithmetic shift: bit i of x XOR (x >> 1) is set where
///   bits i and i + 1 of x differ.
void
translate_data_processing_1_source(bal_tier1_compiler_t *BAL_RESTRICT compiler,
                                   const uint32_t                     instruction)
{
    const uint8_t  rd       = (uint8_t)(instruction & 0x1FU);
    const uint8_t  rn       = (uint8_t)((instruction >> 5U) & 0x1FU);
    const uint32_t opcode   = (instruction >> 10U) & 0x3FU;
    const bool     is_32bit = 0U == (instruction >> 31U);
    const uint32_t width    = true == is_32bit ? 32U : 64U;

    // opcode 000011 is REV for 64-bit registers only; above CLS are newer extensions.
    if (BAL_UNLIKELY(opcode > 5U) || BAL_UNLIKELY(3U == opcode && true == is_32bit))
    {
        BAL_LOG_ERROR(&bal_thread_logger,
                      "Aborting function: unsupported data-processing (1 source) encoding %08x",
                      instruction);
        compiler->status = BAL_ERROR_UNKNOWN_INSTRUCTION;
        return;
    }

    // Both register operands treat 31 as the zero register.
    if (31U == rd)
    {
        return;
    }

    const bal_x86_register_t value = BAL_X86_REGISTER_ZERO;

    if (31U == rn)
    {
        const bal_x86_macro_t zero_macro = {
            .opcode              = BAL_X86_MACRO_MOV_REGISTER_IMMEDIATE,
            .destination         = value,
            .immediate_or_offset = 0U,
        };
        bal_sliding_window_push(&compiler->window, zero_macro);
    }
    else
    {
        const bool               skip_load = false;
        const bal_x86_register_t x86_rn    = allocate_x86_register(compiler, rn, skip_load);
        push_macro(compiler, BAL_X86_MACRO_MOV_REGISTER_REGISTER, value, x86_rn);
    }

    switch (opcode)
    {
        case 0U: // RBIT
            push_macro(compiler,
                       true == is_32bit ? BAL_X86_MACRO_BSWAP_REGISTER_32
                                        : BAL_X86_MACRO_BSWAP_REGISTER,
                       value,
                       BAL_X86_INVALID);
            emit_swap_bit_groups(
                compiler, 4U, (true == is_32bit ? 0x0F0F0F0FULL : 0x0F0F0F0F0F0F0F0FULL));
            emit_swap_bit_groups(
                compiler, 2U, (true == is_32bit ? 0x33333333ULL : 0x3333333333333333ULL));
            emit_swap_bit_groups(
                compiler, 1U, (true == is_32bit ? 0x55555555ULL : 0x5555555555555555ULL));
            break;
        case 1U: // REV16
            if (true == is_32bit)
            {
                push_macro(compiler, BAL_X86_MACRO_MOV_REGISTER_REGISTER_32, value, value);
            }

            emit_swap_bit_groups(
                compiler, 8U, (true == is_32bit ? 0x00FF00FFULL : 0x00FF00FF00FF00FFULL));
            break;
        case 2U: // REV (32-bit) or REV32 (64-bit)
            push_macro(compiler,
                       true == is_32bit ? BAL_X86_MACRO_BSWAP_REGISTER_32
                                        : BAL_X86_MACRO_BSWAP_REGISTER,
                       value,
                       BAL_X86_INVALID);

            if (false == is_32bit)
            {
                push_shift(compiler,
                           BAL_X86_MACRO_ROR_REGISTER_IMMEDIATE,
                           BAL_X86_MACRO_ROR_REGISTER_IMMEDIATE,
                           false,
                           value,
                           32U);
            }
            break;
        case 3U: // REV (64-bit)
            push_macro(compiler, BAL_X86_MACRO_BSWAP_REGISTER, value, BAL_X86_INVALID);
            break;
        default: // CLZ and CLS
        {
            const bal_x86_register_t scratch = BAL_X86_REGISTER_DISCARD_RESULT;

            if (5U == opcode)
            {
                push_macro(compiler, BAL_X86_MACRO_MOV_REGISTER_REGISTER, scratch, value);
                push_shift(compiler,
                           BAL_X86_MACRO_SAR_REGISTER_IMMEDIATE,
                           BAL_X86_MACRO_SAR_REGISTER_IMMEDIATE_32,
                           is_32bit,
                           scratch,
                           1U);
                push_macro(compiler,
                           true == is_32bit ? BAL_X86_MACRO_XOR_REGISTER_REGISTER_32
                                            : BAL_X86_MACRO_XOR_REGISTER_REGISTER,
                           value,
                           scratch);
            }

            push_macro(compiler,
                       true == is_32bit ? BAL_X86_MACRO_BSR_REGISTER_REGISTER_32
                                        : BAL_X86_MACRO_BSR_REGISTER_REGISTER,
                       value,
                       value);

            // Moves leave ZF from BSR intact.
            const bal_x86_macro_t zero_input_macro = {
                .opcode              = BAL_X86_MACRO_MOV_REGISTER_IMMEDIATE,
                .destination         = scratch,
                .immediate_or_offset = width ^ (width - 1U),
            };
            bal_sliding_window_push(&compiler->window, zero_input_macro);

            const bal_x86_macro_t select_macro = {
                .opcode      = BAL_X86_MACRO_CMOV_REGISTER_REGISTER,
                .destination = value,
                .source      = scratch,
                .condition   = BAL_X86_COND_E,
            };
            bal_sliding_window_push(&compiler->window, select_macro);

            const bal_x86_macro_t highest_index_macro = {
                .opcode              = BAL_X86_MACRO_MOV_REGISTER_IMMEDIATE,
                .destination         = scratch,
                .immediate_or_offset = width - 1U,
            };
            bal_sliding_window_push(&compiler->window, highest_index_macro);
            push_macro(compiler, BAL_X86_MACRO_XOR_REGISTER_REGISTER, value, scratch);

            if (5U == opcode)
            {
                const bal_x86_macro_t decrement_macro = {
                    .opcode              = BAL_X86_MACRO_SUB_REGISTER_IMMEDIATE,
                    .destination         = value,
                    .immediate_or_offset = 1U,
                };
                bal_sliding_window_push(&compiler->window, decrement_macro);
            }
            break;
        }
    }

    const bool               skip_load_rd = true;
    const bal_x86_register_t x86_rd       = allocate_x86_register(compiler, rd, skip_load_rd);
    push_macro(compiler, BAL_X86_MACRO_MOV_REGISTER_REGISTER, x86_rd, value);
    compiler->is_dirty |= 1U << rd;
}

/// Encodes the o0:op1:CRn:CRm:op2 fields of a system register as they appear in bits 19:5 of MRS
/// and MSR.
#define SYSTEM_REGISTER(op0, op1, crn, crm, op2) \
    ((((op0) - 2U) << 14U) | ((op1) << 11U) | ((crn) << 7U) | ((crm) << 3U) | (op2))

#define SYSTEM_REGISTER_NZCV        SYSTEM_REGISTER(3U, 3U, 4U, 2U, 0U)
#define SYSTEM_REGISTER_FPCR        SYSTEM_REGISTER(3U, 3U, 4U, 4U, 0U)
#define SYSTEM_REGISTER_FPSR        SYSTEM_REGISTER(3U, 3U, 4U, 4U, 1U)
#define SYSTEM_REGISTER_TPIDR_EL0   SYSTEM_REGISTER(3U, 3U, 13U, 0U, 2U)
#define SYSTEM_REGISTER_TPIDRRO_EL0 SYSTEM_REGISTER(3U, 3U, 13U, 0U, 3U)
#define SYSTEM_REGISTER_CTR_EL0     SYSTEM_REGISTER(3U, 3U, 0U, 0U, 1U)
#define SYSTEM_REGISTER_DCZID_EL0   SYSTEM_REGISTER(3U, 3U, 0U, 0U, 7U)
#define SYSTEM_REGISTER_CNTFRQ_EL0  SYSTEM_REGISTER(3U, 3U, 14U, 0U, 0U)
#define SYSTEM_REGISTER_CNTPCT_EL0  SYSTEM_REGISTER(3U, 3U, 14U, 0U, 1U)
#define SYSTEM_REGISTER_CNTVCT_EL0  SYSTEM_REGISTER(3U, 3U, 14U, 0U, 2U)

/// The writable bits of FPCR (AHP, DN, FZ, RMode, Stride, FZ16, Len) and FPSR (N, Z, C, V, QC and
/// the cumulative exception flags), as implemented by Unicorn's reference model. FZ16 is writable
/// because the guest CPU implements half-precision arithmetic.
#define FPCR_WRITABLE_BITS 0x07FF0000ULL
#define FPSR_WRITABLE_BITS 0xF800009FULL

/// Cache type and DC ZVA block size of a Cortex-A57: 64-byte cache lines, a 64-byte DC ZVA block.
#define CTR_EL0_VALUE   0x8444C004ULL
#define DCZID_EL0_VALUE 0x4ULL

/// Translates MRS and MSR for the system registers an EL0 program uses: NZCV, FPCR, FPSR,
/// TPIDR_EL0, TPIDRRO_EL0 (read only), the CTR_EL0 and DCZID_EL0 identification registers, and the
/// generic timer's CNTFRQ_EL0, CNTPCT_EL0 and CNTVCT_EL0 (read only), which a runtime helper reads
/// from the host's counter. Other registers are rejected as unknown instructions.
///
/// NZCV is converted between the A64 bit layout (N, Z, C and V in bits 31:28) and the four flag
/// bytes of bal_cpu_t. Each bit is isolated by shifting it to bit 63 and back down to bit 0, which
/// clears every other bit, and then shifted to its place.
void
translate_system_register(bal_tier1_compiler_t *BAL_RESTRICT compiler, const uint32_t instruction)
{
    const uint8_t  rt              = (uint8_t)(instruction & 0x1FU);
    const uint32_t system_register = (instruction >> 5U) & 0x7FFFU;
    const bool     is_read         = 1U == ((instruction >> 21U) & 1U);

    // A64 bit of each flag, and its byte in the word at flag_c.
    static const uint32_t FLAG_BITS[4]      = { 29U, 30U, 31U, 28U };
    static const uint32_t FLAG_BYTE_BITS[4] = { 0U, 8U, 16U, 24U };

    if (true == is_read
        && (SYSTEM_REGISTER_CNTFRQ_EL0 == system_register
            || SYSTEM_REGISTER_CNTPCT_EL0 == system_register
            || SYSTEM_REGISTER_CNTVCT_EL0 == system_register))
    {
        emit_register_helper_call(
            compiler, instruction, (uint64_t)(uintptr_t)&bal_execute_counter_read);
        return;
    }

    // FPSR includes the flags that inline floating point has raised into MXCSR in this block.
    if (SYSTEM_REGISTER_FPSR == system_register && true == compiler->is_guest_mxcsr_active)
    {
        bal_sliding_window_flush_all(&compiler->window);
        emit_fold_mxcsr_flags(&compiler->assembler);
        emit_load_guest_mxcsr(&compiler->assembler);
    }

    const bal_x86_register_t value       = BAL_X86_REGISTER_ZERO;
    const bal_x86_register_t scratch     = BAL_X86_REGISTER_DISCARD_RESULT;
    size_t                   offset      = 0U;
    uint64_t                 mask        = 0xFFFFFFFFFFFFFFFFULL;
    uint64_t                 constant    = 0U;
    bool                     is_constant = false;

    switch (system_register)
    {
        case SYSTEM_REGISTER_NZCV:
            break;
        case SYSTEM_REGISTER_FPCR:
            offset = offsetof(bal_cpu_t, fpcr);
            mask   = FPCR_WRITABLE_BITS;
            break;
        case SYSTEM_REGISTER_FPSR:
            offset = offsetof(bal_cpu_t, fpsr);
            mask   = FPSR_WRITABLE_BITS;
            break;
        case SYSTEM_REGISTER_TPIDR_EL0:
            offset = offsetof(bal_cpu_t, tpidr_el0);
            break;
        case SYSTEM_REGISTER_TPIDRRO_EL0:
            offset = offsetof(bal_cpu_t, tpidrro_el0);
            break;
        case SYSTEM_REGISTER_CTR_EL0:
            constant    = CTR_EL0_VALUE;
            is_constant = true;
            break;
        case SYSTEM_REGISTER_DCZID_EL0:
            constant    = DCZID_EL0_VALUE;
            is_constant = true;
            break;
        default:
            offset = SIZE_MAX;
            break;
    }

    // Unknown registers, writes to read-only ones, and EL0-inaccessible writes.
    const bool is_read_only = SYSTEM_REGISTER_TPIDRRO_EL0 == system_register || true == is_constant;

    if (BAL_UNLIKELY(SIZE_MAX == offset) || BAL_UNLIKELY(false == is_read && true == is_read_only))
    {
        BAL_LOG_ERROR(&bal_thread_logger,
                      "Aborting function: unsupported system register access %08x",
                      instruction);
        compiler->status = BAL_ERROR_UNKNOWN_INSTRUCTION;
        return;
    }

    if (true == is_read)
    {
        // Register 31 is the zero register, so a read into it has no effect.
        if (31U == rt)
        {
            return;
        }

        if (true == is_constant)
        {
            const bal_x86_macro_t constant_macro = {
                .opcode              = BAL_X86_MACRO_MOV_REGISTER_IMMEDIATE,
                .destination         = value,
                .immediate_or_offset = constant,
            };
            bal_sliding_window_push(&compiler->window, constant_macro);
        }
        else if (SYSTEM_REGISTER_NZCV == system_register)
        {
            const bal_x86_macro_t zero_macro = {
                .opcode              = BAL_X86_MACRO_MOV_REGISTER_IMMEDIATE,
                .destination         = value,
                .immediate_or_offset = 0U,
            };
            bal_sliding_window_push(&compiler->window, zero_macro);

            for (size_t i = 0U; i < 4U; ++i)
            {
                const bal_x86_macro_t load_macro = {
                    .opcode              = BAL_X86_MACRO_LOAD,
                    .destination         = scratch,
                    .immediate_or_offset = offsetof(bal_cpu_t, flag_c) + i,
                };
                bal_sliding_window_push(&compiler->window, load_macro);
                push_shift(compiler,
                           BAL_X86_MACRO_SHL_REGISTER_IMMEDIATE,
                           BAL_X86_MACRO_SHL_REGISTER_IMMEDIATE,
                           false,
                           scratch,
                           63U);
                push_shift(compiler,
                           BAL_X86_MACRO_SHR_REGISTER_IMMEDIATE,
                           BAL_X86_MACRO_SHR_REGISTER_IMMEDIATE,
                           false,
                           scratch,
                           63U - FLAG_BITS[i]);
                push_macro(compiler, BAL_X86_MACRO_OR_REGISTER_REGISTER, value, scratch);
            }
        }
        else
        {
            const bal_x86_macro_t load_macro = {
                .opcode              = BAL_X86_MACRO_LOAD,
                .destination         = value,
                .immediate_or_offset = offset,
            };
            bal_sliding_window_push(&compiler->window, load_macro);
        }

        const bool               skip_load = true;
        const bal_x86_register_t x86_rt    = allocate_x86_register(compiler, rt, skip_load);
        push_macro(compiler, BAL_X86_MACRO_MOV_REGISTER_REGISTER, x86_rt, value);
        compiler->is_dirty |= 1U << rt;
        return;
    }

    // MSR: copy Xt, which reads as zero for register 31.
    if (31U == rt)
    {
        const bal_x86_macro_t zero_macro = {
            .opcode              = BAL_X86_MACRO_MOV_REGISTER_IMMEDIATE,
            .destination         = value,
            .immediate_or_offset = 0U,
        };
        bal_sliding_window_push(&compiler->window, zero_macro);
    }
    else
    {
        const bool               skip_load = false;
        const bal_x86_register_t x86_rt    = allocate_x86_register(compiler, rt, skip_load);
        push_macro(compiler, BAL_X86_MACRO_MOV_REGISTER_REGISTER, value, x86_rt);
    }

    if (SYSTEM_REGISTER_NZCV == system_register)
    {
        // Build the flag word in the discard register from the copy of Xt in the zero register.
        // flag_c to flag_v are followed by the exception field, which is zero while a block runs,
        // so a 64-bit store of the flags with zero above them covers exactly them.
        const bal_x86_register_t flags      = scratch;
        const bal_x86_macro_t    zero_macro = {
               .opcode              = BAL_X86_MACRO_MOV_REGISTER_IMMEDIATE,
               .destination         = flags,
               .immediate_or_offset = 0U,
        };
        bal_sliding_window_push(&compiler->window, zero_macro);

        for (size_t i = 0U; i < 4U; ++i)
        {
            push_macro(compiler, BAL_X86_MACRO_PUSH_REGISTER, BAL_X86_INVALID, value);
            push_shift(compiler,
                       BAL_X86_MACRO_SHL_REGISTER_IMMEDIATE,
                       BAL_X86_MACRO_SHL_REGISTER_IMMEDIATE,
                       false,
                       value,
                       63U - FLAG_BITS[i]);
            push_shift(compiler,
                       BAL_X86_MACRO_SHR_REGISTER_IMMEDIATE,
                       BAL_X86_MACRO_SHR_REGISTER_IMMEDIATE,
                       false,
                       value,
                       63U);
            push_shift(compiler,
                       BAL_X86_MACRO_SHL_REGISTER_IMMEDIATE,
                       BAL_X86_MACRO_SHL_REGISTER_IMMEDIATE,
                       false,
                       value,
                       FLAG_BYTE_BITS[i]);
            push_macro(compiler, BAL_X86_MACRO_OR_REGISTER_REGISTER, flags, value);
            push_macro(compiler, BAL_X86_MACRO_POP_REGISTER, value, BAL_X86_INVALID);
        }

        const bal_x86_macro_t store_macro = {
            .opcode              = BAL_X86_MACRO_STORE,
            .source              = flags,
            .immediate_or_offset = offsetof(bal_cpu_t, flag_c),
        };
        bal_sliding_window_push(&compiler->window, store_macro);
        return;
    }

    if (mask != 0xFFFFFFFFFFFFFFFFULL)
    {
        const bal_x86_macro_t mask_macro = {
            .opcode              = BAL_X86_MACRO_AND_REGISTER_IMMEDIATE,
            .destination         = value,
            .immediate_or_offset = mask,
        };
        bal_sliding_window_push(&compiler->window, mask_macro);
    }

    const bal_x86_macro_t store_macro = {
        .opcode              = BAL_X86_MACRO_STORE,
        .source              = value,
        .immediate_or_offset = offset,
    };
    bal_sliding_window_push(&compiler->window, store_macro);
}

/// Called from translated code to turn a guest address into a host pointer. Returns the host
/// address of `size` contiguous bytes at `guest_address`, or NULL when the memory interface does
/// not map all of them.
static const uint8_t *
translate_data_address(const bal_memory_interface_t *memory_interface,
                       const bal_guest_address_t     guest_address,
                       const uint64_t                size)
{
    size_t         available_bytes = 0U;
    const uint8_t *host_address
        = memory_interface->translate((void *)memory_interface, guest_address, &available_bytes);

    if (NULL == host_address || available_bytes < size)
    {
        return NULL;
    }

    return host_address;
}

/// Calls the C function at `function`, whose arguments are already in the ABI argument
/// registers. The return value is left in RAX.
///
/// The call may clobber every caller-saved register, so callers write dirty guest registers back
/// and empty the register cache first. R13 and RBP are callee-saved and survive the call.
static void
emit_runtime_call(bal_tier1_compiler_t *BAL_RESTRICT compiler, const uint64_t function)
{
    // The helpers use the host floating-point environment themselves: the flags the block has
    // raised so far go to FPSR first, and the guest environment is reloaded after the call.
    if (true == compiler->is_guest_mxcsr_active)
    {
        bal_sliding_window_flush_all(&compiler->window);
        emit_fold_mxcsr_flags(&compiler->assembler);
        emit_load_guest_mxcsr(&compiler->assembler);
    }

    if (BAL_X86_ABI_SHADOW_SPACE != 0U)
    {
        const bal_x86_macro_t reserve_macro = {
            .opcode              = BAL_X86_MACRO_SUB_REGISTER_IMMEDIATE,
            .destination         = BAL_X86_RSP,
            .immediate_or_offset = BAL_X86_ABI_SHADOW_SPACE,
        };
        bal_sliding_window_push(&compiler->window, reserve_macro);
    }

    const bal_x86_macro_t function_macro = {
        .opcode              = BAL_X86_MACRO_MOV_REGISTER_IMMEDIATE,
        .destination         = BAL_X86_RAX,
        .immediate_or_offset = function,
    };
    bal_sliding_window_push(&compiler->window, function_macro);
    push_macro(compiler, BAL_X86_MACRO_CALL_REGISTER, BAL_X86_RAX, BAL_X86_INVALID);

    if (BAL_X86_ABI_SHADOW_SPACE != 0U)
    {
        const bal_x86_macro_t release_macro = {
            .opcode              = BAL_X86_MACRO_ADD_REGISTER_IMMEDIATE,
            .destination         = BAL_X86_RSP,
            .immediate_or_offset = BAL_X86_ABI_SHADOW_SPACE,
        };
        bal_sliding_window_push(&compiler->window, release_macro);
    }

    if (true == compiler->is_guest_mxcsr_active)
    {
        bal_sliding_window_flush_all(&compiler->window);
        emit_load_guest_mxcsr(&compiler->assembler);
    }
}

/// Stops the block with a data abort when `result` is zero: the guest PC is set to this
/// instruction, the exception field to BAL_EXCEPTION_DATA_ABORT, and the block returns. The jump
/// over that exit is patched once its size is known.
static void
emit_data_abort_if_zero(bal_tier1_compiler_t *BAL_RESTRICT compiler,
                        const bal_x86_register_t           result,
                        const bal_guest_address_t          guest_address)
{
    push_macro(compiler, BAL_X86_MACRO_TEST_REGISTER_REGISTER, result, result);
    bal_sliding_window_flush_all(&compiler->window);

    bal_x86_assembler_t *BAL_RESTRICT assembler   = &compiler->assembler;
    const size_t                      jump_offset = assembler->offset;
    bal_x86_emit_jcc_rel32(assembler, BAL_X86_COND_NE, 0);
    const size_t exit_offset = assembler->offset;

    bal_x86_emit_mov_r64_imm64(assembler, BAL_X86_RAX, guest_address);
    bal_x86_emit_store_r64_rbp_offset(assembler, BAL_X86_RAX, offsetof(bal_cpu_t, pc));
    bal_x86_emit_mov_r64_imm64(assembler, BAL_X86_RAX, BAL_EXCEPTION_DATA_ABORT);
    bal_x86_emit_store_r32_rbp_offset(assembler, BAL_X86_RAX, offsetof(bal_cpu_t, exception));
    emit_block_exit(compiler);

    if (BAL_SUCCESS == assembler->status)
    {
        // The rel32 field is the last four bytes of the 6-byte jcc.
        const uint32_t        exit_size = (uint32_t)(assembler->offset - exit_offset);
        uint8_t *BAL_RESTRICT relative  = assembler->buffer + jump_offset + 2U;
        relative[0]                     = (uint8_t)exit_size;
        relative[1]                     = (uint8_t)(exit_size >> 8U);
        relative[2]                     = (uint8_t)(exit_size >> 16U);
        relative[3]                     = (uint8_t)(exit_size >> 24U);
    }
}

/// Calls a runtime helper `function(cpu, instruction)` that executes one instruction on the guest
/// registers, which are flushed to the cpu structure first.
static void
emit_register_helper_call(bal_tier1_compiler_t *BAL_RESTRICT compiler,
                          const uint32_t                     instruction,
                          const uint64_t                     function)
{
    flush_dirty_registers(compiler);
    reset_register_allocator(compiler);

    push_macro(compiler, BAL_X86_MACRO_MOV_REGISTER_REGISTER, BAL_X86_ABI_ARG1, BAL_X86_RBP);

    const bal_x86_macro_t instruction_macro = {
        .opcode              = BAL_X86_MACRO_MOV_REGISTER_IMMEDIATE,
        .destination         = BAL_X86_ABI_ARG2,
        .immediate_or_offset = instruction,
    };
    bal_sliding_window_push(&compiler->window, instruction_macro);

    emit_runtime_call(compiler, function);
}

/// Calls a runtime helper `function(cpu, memory_interface, instruction)` that executes one
/// memory instruction on the guest state, and raises a data abort when it returns zero.
///
/// The guest registers are flushed to the cpu structure first, since the helper reads and writes
/// them there.
static void
emit_memory_helper_call(bal_tier1_compiler_t *BAL_RESTRICT         compiler,
                        const bal_memory_interface_t *BAL_RESTRICT memory_interface,
                        const uint32_t                             instruction,
                        const bal_guest_address_t                  guest_address,
                        const uint64_t                             function)
{
    flush_dirty_registers(compiler);
    reset_register_allocator(compiler);

    push_macro(compiler, BAL_X86_MACRO_MOV_REGISTER_REGISTER, BAL_X86_ABI_ARG1, BAL_X86_RBP);

    const bal_x86_macro_t interface_macro = {
        .opcode              = BAL_X86_MACRO_MOV_REGISTER_IMMEDIATE,
        .destination         = BAL_X86_ABI_ARG2,
        .immediate_or_offset = (uint64_t)(uintptr_t)memory_interface,
    };
    bal_sliding_window_push(&compiler->window, interface_macro);

    const bal_x86_macro_t instruction_macro = {
        .opcode              = BAL_X86_MACRO_MOV_REGISTER_IMMEDIATE,
        .destination         = BAL_X86_ABI_ARG3,
        .immediate_or_offset = instruction,
    };
    bal_sliding_window_push(&compiler->window, instruction_macro);

    emit_runtime_call(compiler, function);
    emit_data_abort_if_zero(compiler, BAL_X86_RAX, guest_address);
}

/// Turns the guest address in BAL_X86_REGISTER_ZERO into a host pointer in the same register,
/// by calling translate_data_address() for `size` bytes, and raises a data abort when the address
/// is not mapped.
///
/// Dirty guest registers are written back and the register cache is emptied first, since the
/// call may clobber every caller-saved register.
static void
emit_translate_data_address(bal_tier1_compiler_t *BAL_RESTRICT         compiler,
                            const bal_memory_interface_t *BAL_RESTRICT memory_interface,
                            const uint64_t                             size,
                            const bal_guest_address_t                  guest_address)
{
    const bal_x86_register_t address = BAL_X86_REGISTER_ZERO;

    flush_dirty_registers(compiler);
    reset_register_allocator(compiler);

    push_macro(compiler, BAL_X86_MACRO_MOV_REGISTER_REGISTER, BAL_X86_ABI_ARG2, address);

    const bal_x86_macro_t interface_macro = {
        .opcode              = BAL_X86_MACRO_MOV_REGISTER_IMMEDIATE,
        .destination         = BAL_X86_ABI_ARG1,
        .immediate_or_offset = (uint64_t)(uintptr_t)memory_interface,
    };
    bal_sliding_window_push(&compiler->window, interface_macro);

    const bal_x86_macro_t size_macro = {
        .opcode              = BAL_X86_MACRO_MOV_REGISTER_IMMEDIATE,
        .destination         = BAL_X86_ABI_ARG3,
        .immediate_or_offset = size,
    };
    bal_sliding_window_push(&compiler->window, size_macro);

    emit_runtime_call(compiler, (uint64_t)(uintptr_t)&translate_data_address);
    push_macro(compiler, BAL_X86_MACRO_MOV_REGISTER_REGISTER, address, BAL_X86_RAX);
    emit_data_abort_if_zero(compiler, address, guest_address);
}

/// Puts the address formed by a base register (the stack pointer for register 31) plus a signed
/// offset into BAL_X86_REGISTER_ZERO.
static void
emit_base_plus_offset(bal_tier1_compiler_t *BAL_RESTRICT compiler,
                      const uint8_t                      rn,
                      const int64_t                      offset)
{
    const bool               skip_load = false;
    const bal_x86_register_t x86_rn    = allocate_x86_register(compiler, rn, skip_load);
    push_macro(compiler, BAL_X86_MACRO_MOV_REGISTER_REGISTER, BAL_X86_REGISTER_ZERO, x86_rn);

    if (offset != 0)
    {
        const bal_x86_macro_t add_macro = {
            .opcode              = BAL_X86_MACRO_ADD_REGISTER_IMMEDIATE,
            .destination         = BAL_X86_REGISTER_ZERO,
            .immediate_or_offset = (uint64_t)offset,
        };
        bal_sliding_window_push(&compiler->window, add_macro);
    }
}

/// Moves BAL_X86_REGISTER_ZERO by `bytes`, a two's complement value, between the registers of a
/// pair.
static void
emit_advance_address(bal_tier1_compiler_t *BAL_RESTRICT compiler, const uint64_t bytes)
{
    const bal_x86_macro_t add_macro = {
        .opcode              = BAL_X86_MACRO_ADD_REGISTER_IMMEDIATE,
        .destination         = BAL_X86_REGISTER_ZERO,
        .immediate_or_offset = bytes,
    };
    bal_sliding_window_push(&compiler->window, add_macro);
}

/// Loads or stores one register through the host pointer in BAL_X86_REGISTER_ZERO.
///
/// `size_log2` is the access size (0 to 3) and, for loads, `sign_extend` selects LDRSB/LDRSH/LDRSW:
/// 0 zero-extends, 32 sign-extends into a W register, 64 into an X register. Register 31 is the
/// zero register: loads into it are dropped and stores of it write zero.
static void
emit_register_access(bal_tier1_compiler_t *BAL_RESTRICT compiler,
                     const uint8_t                      rt,
                     const uint32_t                     size_log2,
                     const bool                         is_load,
                     const uint32_t                     sign_extend)
{
    const bal_x86_register_t pointer = BAL_X86_REGISTER_ZERO;

    if (true == is_load)
    {
        if (31U == rt)
        {
            return;
        }

        static const bal_x86_macro_opcode_t ZERO_EXTEND[4] = {
            BAL_X86_MACRO_LOAD_MEMORY_R32_MEM8,
            BAL_X86_MACRO_LOAD_MEMORY_R32_MEM16,
            BAL_X86_MACRO_LOAD_MEMORY_R32_MEM32,
            BAL_X86_MACRO_LOAD_MEMORY_R64_MEM64,
        };
        static const bal_x86_macro_opcode_t SIGN_EXTEND_32[4] = {
            BAL_X86_MACRO_LOAD_MEMORY_SIGN_R32_MEM8,
            BAL_X86_MACRO_LOAD_MEMORY_SIGN_R32_MEM16,
            BAL_X86_MACRO_LOAD_MEMORY_R32_MEM32,
            BAL_X86_MACRO_LOAD_MEMORY_R64_MEM64,
        };
        static const bal_x86_macro_opcode_t SIGN_EXTEND_64[4] = {
            BAL_X86_MACRO_LOAD_MEMORY_SIGN_R64_MEM8,
            BAL_X86_MACRO_LOAD_MEMORY_SIGN_R64_MEM16,
            BAL_X86_MACRO_LOAD_MEMORY_SIGN_R64_MEM32,
            BAL_X86_MACRO_LOAD_MEMORY_R64_MEM64,
        };
        bal_x86_macro_opcode_t opcode = ZERO_EXTEND[size_log2];

        if (32U == sign_extend)
        {
            opcode = SIGN_EXTEND_32[size_log2];
        }
        else if (64U == sign_extend)
        {
            opcode = SIGN_EXTEND_64[size_log2];
        }

        const bool               skip_load = true;
        const bal_x86_register_t x86_rt    = allocate_x86_register(compiler, rt, skip_load);
        push_macro(compiler, opcode, x86_rt, pointer);
        compiler->is_dirty |= 1U << rt;
        return;
    }

    static const bal_x86_macro_opcode_t STORES[4] = {
        BAL_X86_MACRO_STORE_MEMORY_MEM8_R8,
        BAL_X86_MACRO_STORE_MEMORY_MEM16_R16,
        BAL_X86_MACRO_STORE_MEMORY_MEM32_R32,
        BAL_X86_MACRO_STORE_MEMORY_MEM64_R64,
    };
    bal_x86_register_t x86_rt = BAL_X86_REGISTER_DISCARD_RESULT;

    if (31U == rt)
    {
        const bal_x86_macro_t zero_macro = {
            .opcode              = BAL_X86_MACRO_MOV_REGISTER_IMMEDIATE,
            .destination         = x86_rt,
            .immediate_or_offset = 0U,
        };
        bal_sliding_window_push(&compiler->window, zero_macro);
    }
    else
    {
        const bool skip_load = false;
        x86_rt               = allocate_x86_register(compiler, rt, skip_load);
    }

    push_macro(compiler, STORES[size_log2], pointer, x86_rt);
}

/// Loads or stores one SIMD&FP register through the host pointer in BAL_X86_REGISTER_ZERO.
///
/// `size_log2` is the access size, 0 (B) to 4 (Q). The registers live in the cpu structure, so
/// the data moves through BAL_X86_REGISTER_DISCARD_RESULT. A load clears the bits above the
/// access, as every scalar write to a SIMD&FP register does.
static void
emit_vector_access(bal_tier1_compiler_t *BAL_RESTRICT compiler,
                   const uint8_t                      vt,
                   const uint32_t                     size_log2,
                   const bool                         is_load)
{
    static const bal_x86_macro_opcode_t LOADS[4] = {
        BAL_X86_MACRO_LOAD_MEMORY_R32_MEM8,
        BAL_X86_MACRO_LOAD_MEMORY_R32_MEM16,
        BAL_X86_MACRO_LOAD_MEMORY_R32_MEM32,
        BAL_X86_MACRO_LOAD_MEMORY_R64_MEM64,
    };
    static const bal_x86_macro_opcode_t STORES[4] = {
        BAL_X86_MACRO_STORE_MEMORY_MEM8_R8,
        BAL_X86_MACRO_STORE_MEMORY_MEM16_R16,
        BAL_X86_MACRO_STORE_MEMORY_MEM32_R32,
        BAL_X86_MACRO_STORE_MEMORY_MEM64_R64,
    };
    const bal_x86_register_t pointer     = BAL_X86_REGISTER_ZERO;
    const bal_x86_register_t data        = BAL_X86_REGISTER_DISCARD_RESULT;
    const uint32_t           low_size    = size_log2 > 3U ? 3U : size_log2;
    const uint64_t           low_offset  = offsetof(bal_cpu_t, v[vt][0]);
    const uint64_t           high_offset = offsetof(bal_cpu_t, v[vt][1]);

    if (true == is_load)
    {
        push_macro(compiler, LOADS[low_size], data, pointer);
        const bal_x86_macro_t store_low = {
            .opcode              = BAL_X86_MACRO_STORE,
            .source              = data,
            .immediate_or_offset = low_offset,
        };
        bal_sliding_window_push(&compiler->window, store_low);

        if (4U == size_log2)
        {
            emit_advance_address(compiler, 8U);
            push_macro(compiler, BAL_X86_MACRO_LOAD_MEMORY_R64_MEM64, data, pointer);
            emit_advance_address(compiler, (uint64_t)-8);
        }
        else
        {
            const bal_x86_macro_t zero_macro = {
                .opcode              = BAL_X86_MACRO_MOV_REGISTER_IMMEDIATE,
                .destination         = data,
                .immediate_or_offset = 0U,
            };
            bal_sliding_window_push(&compiler->window, zero_macro);
        }

        const bal_x86_macro_t store_high = {
            .opcode              = BAL_X86_MACRO_STORE,
            .source              = data,
            .immediate_or_offset = high_offset,
        };
        bal_sliding_window_push(&compiler->window, store_high);
        return;
    }

    const bal_x86_macro_t load_low = {
        .opcode              = BAL_X86_MACRO_LOAD,
        .destination         = data,
        .immediate_or_offset = low_offset,
    };
    bal_sliding_window_push(&compiler->window, load_low);
    push_macro(compiler, STORES[low_size], pointer, data);

    if (4U == size_log2)
    {
        const bal_x86_macro_t load_high = {
            .opcode              = BAL_X86_MACRO_LOAD,
            .destination         = data,
            .immediate_or_offset = high_offset,
        };
        bal_sliding_window_push(&compiler->window, load_high);
        emit_advance_address(compiler, 8U);
        push_macro(compiler, BAL_X86_MACRO_STORE_MEMORY_MEM64_R64, pointer, data);
        emit_advance_address(compiler, (uint64_t)-8);
    }
}

/// Loads or stores a general purpose register, or a SIMD&FP register when `is_vector` is set.
static void
emit_access(bal_tier1_compiler_t *BAL_RESTRICT compiler,
            const bool                         is_vector,
            const uint8_t                      rt,
            const uint32_t                     size_log2,
            const bool                         is_load,
            const uint32_t                     sign_extend)
{
    if (true == is_vector)
    {
        emit_vector_access(compiler, rt, size_log2, is_load);
        return;
    }

    emit_register_access(compiler, rt, size_log2, is_load, sign_extend);
}

/// Computes the written-back base (base + offset, with register 31 as the stack pointer) of the
/// pre- and post-indexed forms and saves it on the stack. Called after the address is known to be
/// mapped and before the access, so the access sees the original base and a faulting access
/// leaves it unchanged.
static void
emit_writeback_begin(bal_tier1_compiler_t *BAL_RESTRICT compiler,
                     const uint8_t                      rn,
                     const int64_t                      offset)
{
    const bal_x86_register_t updated_base = BAL_X86_REGISTER_DISCARD_RESULT;
    const bool               skip_load    = false;
    const bal_x86_register_t x86_rn       = allocate_x86_register(compiler, rn, skip_load);
    push_macro(compiler, BAL_X86_MACRO_MOV_REGISTER_REGISTER, updated_base, x86_rn);

    const bal_x86_macro_t add_macro = {
        .opcode              = BAL_X86_MACRO_ADD_REGISTER_IMMEDIATE,
        .destination         = updated_base,
        .immediate_or_offset = (uint64_t)offset,
    };
    bal_sliding_window_push(&compiler->window, add_macro);
    push_macro(compiler, BAL_X86_MACRO_PUSH_REGISTER, BAL_X86_INVALID, updated_base);
}

/// Writes the base saved by emit_writeback_begin() back after the access. When the access loaded
/// into the base register, which is CONSTRAINED UNPREDICTABLE, the written-back address wins as in
/// Unicorn's reference model.
static void
emit_writeback_end(bal_tier1_compiler_t *BAL_RESTRICT compiler, const uint8_t rn)
{
    const bal_x86_register_t updated_base = BAL_X86_REGISTER_DISCARD_RESULT;
    push_macro(compiler, BAL_X86_MACRO_POP_REGISTER, updated_base, BAL_X86_INVALID);

    const bool               skip_load = true;
    const bal_x86_register_t x86_rn    = allocate_x86_register(compiler, rn, skip_load);
    push_macro(compiler, BAL_X86_MACRO_MOV_REGISTER_REGISTER, x86_rn, updated_base);
    compiler->is_dirty |= 1U << rn;
}

/// Rejects a load or store encoding tier 1 does not translate.
static void
reject_load_store(bal_tier1_compiler_t *BAL_RESTRICT compiler, const uint32_t instruction)
{
    BAL_LOG_ERROR(
        &bal_thread_logger, "Aborting function: unsupported load/store encoding %08x", instruction);
    compiler->status = BAL_ERROR_UNKNOWN_INSTRUCTION;
}

/// Translates the general purpose loads and stores: LDR/STR and their byte, halfword and
/// sign-extending forms with an unsigned offset, an unscaled offset (LDUR/STUR and the unprivileged
/// LDTR/STTR, which behave the same at EL0), pre- and post-indexing, a register offset, and a
/// PC-relative literal; LDP/STP/LDPSW and the non-temporal LDNP/STNP; and PRFM, which does
/// nothing.
///
/// The address is formed in BAL_X86_REGISTER_ZERO, translated to a host pointer with
/// emit_translate_data_address(), which ends the block with a data abort when it is not mapped,
/// and the access goes through the pointer. Writeback happens after the access, so a faulting
/// access leaves the base register unchanged. SIMD and floating-point accesses are not supported
/// yet.
void
translate_load_store(bal_tier1_compiler_t *BAL_RESTRICT         compiler,
                     const bal_memory_interface_t *BAL_RESTRICT memory_interface,
                     const uint32_t                             instruction,
                     const bal_guest_address_t                  guest_address)
{
    const uint8_t rt = (uint8_t)(instruction & 0x1FU);
    const uint8_t rn = (uint8_t)((instruction >> 5U) & 0x1FU);

    // Advanced SIMD structures: 0 Q 0011 0x ..., LD1-LD4, ST1-ST4 and LD1R-LD4R. The runtime
    // helper moves them element by element.
    if (0x0C000000U == (instruction & 0xBE000000U))
    {
        if (BAL_UNLIKELY(false == bal_is_simd_memory_supported(instruction)))
        {
            reject_load_store(compiler, instruction);
            return;
        }

        emit_memory_helper_call(compiler,
                                memory_interface,
                                instruction,
                                guest_address,
                                (uint64_t)(uintptr_t)&bal_execute_simd_memory);
        return;
    }

    // Load register (literal): opc 011 V 00 imm19 Rt.
    if (0x18U == ((instruction >> 24U) & 0x3BU))
    {
        const uint32_t opc       = instruction >> 30U;
        const bool     is_vector = 1U == ((instruction >> 26U) & 1U);

        // For SIMD&FP registers opc 00/01/10 select S/D/Q and 11 is unallocated.
        if (true == is_vector && BAL_UNLIKELY(3U == opc))
        {
            reject_load_store(compiler, instruction);
            return;
        }

        if (3U == opc)
        {
            return; // PRFM (literal)
        }

        const uint32_t imm19 = (instruction >> 5U) & 0x7FFFFU;

        // WARNING: Shifting the 19-bit field to the top and back sign-extends it.
        const int32_t  offset  = (int32_t)(imm19 << 13U) >> 13U;
        const uint64_t address = guest_address + (uint64_t)((int64_t)offset * 4);
        const uint32_t size_log2
            = true == is_vector ? 2U + opc : (0U == opc ? 2U : (1U == opc ? 3U : 2U));
        const bal_x86_macro_t address_macro = {
            .opcode              = BAL_X86_MACRO_MOV_REGISTER_IMMEDIATE,
            .destination         = BAL_X86_REGISTER_ZERO,
            .immediate_or_offset = address,
        };
        bal_sliding_window_push(&compiler->window, address_macro);
        emit_translate_data_address(compiler, memory_interface, 1ULL << size_log2, guest_address);
        emit_access(
            compiler, is_vector, rt, size_log2, true, 2U == opc && false == is_vector ? 64U : 0U);
        return;
    }

    // Load/store pair: opc 101 V type(2) L imm7 Rt2 Rn Rt.
    if (0x5U == ((instruction >> 27U) & 0x7U))
    {
        const uint32_t opc       = instruction >> 30U;
        const uint32_t type      = (instruction >> 23U) & 0x3U;
        const bool     is_load   = 1U == ((instruction >> 22U) & 1U);
        const uint32_t imm7      = (instruction >> 15U) & 0x7FU;
        const uint8_t  rt2       = (uint8_t)((instruction >> 10U) & 0x1FU);
        const bool     is_vector = 1U == ((instruction >> 26U) & 1U);

        // opc 11 is unallocated. For general purpose registers opc 01 is LDPSW for loads but STGP
        // for stores; for SIMD&FP registers opc 00/01/10 select S/D/Q.
        if (BAL_UNLIKELY(3U == opc)
            || BAL_UNLIKELY(1U == opc && false == is_load && false == is_vector))
        {
            reject_load_store(compiler, instruction);
            return;
        }

        const uint32_t size_log2   = true == is_vector ? 2U + opc : (2U == opc ? 3U : 2U);
        const int32_t  scaled      = ((int32_t)(imm7 << 25U) >> 25U) * (1 << size_log2);
        const bool     is_post     = 1U == type;
        const bool     is_pre      = 3U == type;
        const uint32_t sign_extend = 1U == opc && false == is_vector ? 64U : 0U;

        emit_base_plus_offset(compiler, rn, true == is_post ? 0 : scaled);
        emit_translate_data_address(compiler, memory_interface, 2ULL << size_log2, guest_address);

        if (true == is_pre || true == is_post)
        {
            emit_writeback_begin(compiler, rn, scaled);
        }

        // A load with Rt == Rt2 is CONSTRAINED UNPREDICTABLE. Unicorn's reference model leaves
        // the value from the lower address in a general purpose register but the one from the
        // higher address in a SIMD&FP register, so the access order follows it.
        if (true == is_vector)
        {
            emit_access(compiler, is_vector, rt, size_log2, is_load, sign_extend);
            emit_advance_address(compiler, 1ULL << size_log2);
            emit_access(compiler, is_vector, rt2, size_log2, is_load, sign_extend);
        }
        else
        {
            emit_advance_address(compiler, 1ULL << size_log2);
            emit_access(compiler, is_vector, rt2, size_log2, is_load, sign_extend);
            emit_advance_address(compiler, (uint64_t)(-(int64_t)(1ULL << size_log2)));
            emit_access(compiler, is_vector, rt, size_log2, is_load, sign_extend);
        }

        if (true == is_pre || true == is_post)
        {
            emit_writeback_end(compiler, rn);
        }

        return;
    }

    // Load/store register: size 111 V 0x opc ...
    if (BAL_UNLIKELY(0x7U != ((instruction >> 27U) & 0x7U)))
    {
        reject_load_store(compiler, instruction);
        return;
    }

    const bool     is_vector = 1U == ((instruction >> 26U) & 1U);
    const uint32_t opc       = (instruction >> 22U) & 0x3U;
    uint32_t       size_log2 = instruction >> 30U;
    bool           is_load   = opc != 0U;

    // SIMD&FP registers: opc bit 0 is the direction, and opc bit 1 with size 00 selects Q. The
    // other opc 1x encodings are unallocated.
    if (true == is_vector)
    {
        if (opc >= 2U)
        {
            if (BAL_UNLIKELY(size_log2 != 0U))
            {
                reject_load_store(compiler, instruction);
                return;
            }

            size_log2 = 4U;
        }

        is_load = 1U == (opc & 1U);
    }

    // The register offset form is mode 10 with a W or X index (option x1x). The pointer
    // authentication loads and the atomics share the rest of that space.
    const bool is_unsigned_offset = 1U == ((instruction >> 24U) & 1U);
    const bool is_register_offset
        = false == is_unsigned_offset && 1U == ((instruction >> 21U) & 1U);
    const uint32_t mode   = (instruction >> 10U) & 0x3U;
    const uint32_t option = (instruction >> 13U) & 0x7U;

    if (true == is_register_offset && (mode != 2U || 0U == (option & 0x2U)))
    {
        reject_load_store(compiler, instruction);
        return;
    }

    // opc 10/11 sign-extend to X/W, except for 64-bit accesses, where opc 10 is PRFM, and 32-bit
    // ones, where only opc 10 (LDRSW) exists.
    uint32_t sign_extend = 0U;

    if (opc >= 2U && false == is_vector)
    {
        if (3U == size_log2)
        {
            if (2U == opc)
            {
                return; // PRFM
            }

            reject_load_store(compiler, instruction);
            return;
        }

        if (2U == size_log2 && 3U == opc)
        {
            reject_load_store(compiler, instruction);
            return;
        }

        sign_extend = 2U == opc ? 64U : 32U;
    }

    if (true == is_unsigned_offset)
    {
        // Unsigned offset: imm12 scaled by the access size.
        const int64_t offset = (int64_t)(((instruction >> 10U) & 0xFFFU) << size_log2);
        emit_base_plus_offset(compiler, rn, offset);
        emit_translate_data_address(compiler, memory_interface, 1ULL << size_log2, guest_address);
        emit_access(compiler, is_vector, rt, size_log2, is_load, sign_extend);
        return;
    }

    if (true == is_register_offset)
    {
        // Register offset: Rn + extend(Rm) << (S ? size : 0).
        const uint8_t            rm        = (uint8_t)((instruction >> 16U) & 0x1FU);
        const uint32_t           amount    = 1U == ((instruction >> 12U) & 1U) ? size_log2 : 0U;
        const bool               skip_load = false;
        const bal_x86_register_t index     = BAL_X86_REGISTER_DISCARD_RESULT;

        if (31U == rm)
        {
            const bal_x86_macro_t zero_macro = {
                .opcode              = BAL_X86_MACRO_MOV_REGISTER_IMMEDIATE,
                .destination         = index,
                .immediate_or_offset = 0U,
            };
            bal_sliding_window_push(&compiler->window, zero_macro);
        }
        else
        {
            const bal_x86_register_t x86_rm = allocate_x86_register(compiler, rm, skip_load);
            push_macro(compiler, BAL_X86_MACRO_MOV_REGISTER_REGISTER, index, x86_rm);
        }

        // option 010 UXTW and 110 SXTW take a W register; 011 LSL and 111 SXTX an X register.
        if (0U == (option & 0x1U))
        {
            push_shift(compiler,
                       BAL_X86_MACRO_SHL_REGISTER_IMMEDIATE,
                       BAL_X86_MACRO_SHL_REGISTER_IMMEDIATE,
                       false,
                       index,
                       32U);
            push_shift(compiler,
                       4U == (option & 0x4U) ? BAL_X86_MACRO_SAR_REGISTER_IMMEDIATE
                                             : BAL_X86_MACRO_SHR_REGISTER_IMMEDIATE,
                       BAL_X86_MACRO_SHR_REGISTER_IMMEDIATE,
                       false,
                       index,
                       32U - amount);
        }
        else
        {
            push_shift(compiler,
                       BAL_X86_MACRO_SHL_REGISTER_IMMEDIATE,
                       BAL_X86_MACRO_SHL_REGISTER_IMMEDIATE,
                       false,
                       index,
                       amount);
        }

        emit_base_plus_offset(compiler, rn, 0);
        push_macro(compiler, BAL_X86_MACRO_ADD_REGISTER_REGISTER, BAL_X86_REGISTER_ZERO, index);
        emit_translate_data_address(compiler, memory_interface, 1ULL << size_log2, guest_address);
        emit_access(compiler, is_vector, rt, size_log2, is_load, sign_extend);
        return;
    }

    // imm9 forms: 00 unscaled (LDUR/STUR), 01 post-indexed, 10 unprivileged, 11 pre-indexed.
    const uint32_t imm9    = (instruction >> 12U) & 0x1FFU;
    const int64_t  offset  = (int64_t)((int32_t)(imm9 << 23U) >> 23U);
    const bool     is_post = 1U == mode;
    const bool     is_pre  = 3U == mode;

    emit_base_plus_offset(compiler, rn, true == is_post ? 0 : offset);
    emit_translate_data_address(compiler, memory_interface, 1ULL << size_log2, guest_address);

    if (true == is_pre || true == is_post)
    {
        emit_writeback_begin(compiler, rn, offset);
    }

    emit_access(compiler, is_vector, rt, size_log2, is_load, sign_extend);

    if (true == is_pre || true == is_post)
    {
        emit_writeback_end(compiler, rn);
    }
}

/// Translates the synchronisation instructions: exclusives, load-acquire/store-release (including
/// the RCpc forms), compare-and-swap, the LSE atomic memory operations, and CLREX.
///
/// They are rare enough in guest code that tier 1 does not inline them. Each one flushes the guest
/// registers and calls bal_execute_atomic(), which performs the access as one host atomic
/// operation and updates the registers in the cpu structure, and the block raises a data abort
/// when the helper reports an unmapped or misaligned access.
static void
translate_atomic(bal_tier1_compiler_t *BAL_RESTRICT         compiler,
                 const bal_memory_interface_t *BAL_RESTRICT memory_interface,
                 const uint32_t                             instruction,
                 const bal_guest_address_t                  guest_address)
{
    if (BAL_UNLIKELY(false == bal_is_atomic_supported(instruction)))
    {
        BAL_LOG_ERROR(
            &bal_thread_logger, "Aborting function: unsupported atomic encoding %08x", instruction);
        compiler->status = BAL_ERROR_UNKNOWN_INSTRUCTION;
        return;
    }

    emit_memory_helper_call(compiler,
                            memory_interface,
                            instruction,
                            guest_address,
                            (uint64_t)(uintptr_t)&bal_execute_atomic);
}

/// Translates the SIMD&FP data processing instructions: the scalar floating-point ones through
/// bal_execute_float() and the Advanced SIMD ones through bal_execute_simd().
///
/// Tier 1 calls the helper for each one with the guest registers flushed, so every instruction
/// gets the exact AArch64 behaviour: NaN propagation, FPCR.DN, FPCR.FZ, the FPCR rounding mode,
/// the FPSR flags and saturation. Inline fast paths for the common cases can sit in front of the
/// helpers later.
static void
translate_float(bal_tier1_compiler_t *BAL_RESTRICT compiler, const uint32_t instruction)
{
    if (true == emit_fast_float(compiler, instruction))
    {
        return;
    }

    uint64_t helper = (uint64_t)(uintptr_t)&bal_execute_float;

    if (false == bal_is_float_supported(instruction))
    {
        if (BAL_UNLIKELY(false == bal_is_simd_supported(instruction)))
        {
            BAL_LOG_ERROR(&bal_thread_logger,
                          "Aborting function: unsupported SIMD&FP encoding %08x",
                          instruction);
            compiler->status = BAL_ERROR_UNKNOWN_INSTRUCTION;
            return;
        }

        helper = (uint64_t)(uintptr_t)&bal_execute_simd;
    }

    emit_register_helper_call(compiler, instruction, helper);
}

/// Translates CRC32, CRC32C and the flag manipulation instructions CFINV, RMIF, SETF8 and SETF16
/// through bal_execute_integer_helper(). They are rare enough not to need inline code.
static void
translate_integer_helper(bal_tier1_compiler_t *BAL_RESTRICT compiler, const uint32_t instruction)
{
    if (BAL_UNLIKELY(false == bal_is_integer_helper_supported(instruction)))
    {
        BAL_LOG_ERROR(
            &bal_thread_logger, "Aborting function: unsupported encoding %08x", instruction);
        compiler->status = BAL_ERROR_UNKNOWN_INSTRUCTION;
        return;
    }

    emit_register_helper_call(
        compiler, instruction, (uint64_t)(uintptr_t)&bal_execute_integer_helper);
}

/// Translates pointer authentication: LDRAA and LDRAB through bal_execute_authenticated_load(),
/// which raises a data abort for an address that failed authentication, and everything else
/// through bal_execute_pointer_authentication(). Returns true for the authenticated branches, whose
/// helper stores the target to the guest PC.
static bool
translate_pointer_authentication(bal_tier1_compiler_t *BAL_RESTRICT         compiler,
                                 const bal_memory_interface_t *BAL_RESTRICT memory_interface,
                                 const uint32_t                             instruction,
                                 const bal_guest_address_t                  guest_address)
{
    if (true == bal_is_authenticated_load(instruction))
    {
        emit_memory_helper_call(compiler,
                                memory_interface,
                                instruction,
                                guest_address,
                                (uint64_t)(uintptr_t)&bal_execute_authenticated_load);
        return false;
    }

    if (BAL_UNLIKELY(false == bal_is_pointer_authentication_supported(instruction)))
    {
        BAL_LOG_ERROR(&bal_thread_logger,
                      "Aborting function: unsupported pointer authentication encoding %08x",
                      instruction);
        compiler->status = BAL_ERROR_UNKNOWN_INSTRUCTION;
        return false;
    }

    emit_register_helper_call(
        compiler, instruction, (uint64_t)(uintptr_t)&bal_execute_pointer_authentication);

    if (false == bal_is_authenticated_branch(instruction))
    {
        return false;
    }

    // BLRAA, BLRAB and their Z forms link once the helper has read the target, so BLRAA X30
    // branches to the old X30.
    if (1U == ((instruction >> 21U) & 0x3U))
    {
        write_link_register(compiler, guest_address + 4U);
    }

    return true;
}

/// Translates the exception-generating instructions. The block ends with bal_cpu_t::exception
/// and exception_immediate set for the host, which bal_engine_run_thread() reports as
/// BAL_ERROR_GUEST_EXCEPTION.
///
/// SVC raises BAL_EXCEPTION_SUPERVISOR_CALL and leaves the PC after it, where the host resumes
/// once it has handled the call. BRK raises BAL_EXCEPTION_BREAKPOINT, and UDF, HLT, HVC and SMC,
/// which are undefined at EL0, raise BAL_EXCEPTION_UNDEFINED_INSTRUCTION; those leave the PC on
/// the instruction. Returns true when the PC stays on the instruction.
static bool
translate_exception(bal_tier1_compiler_t *BAL_RESTRICT compiler,
                    const uint32_t                     instruction,
                    const bal_guest_address_t          guest_address)
{
    (void)guest_address;
    uint64_t        immediate      = (instruction >> 5U) & 0xFFFFU;
    bal_exception_t exception      = BAL_EXCEPTION_UNDEFINED_INSTRUCTION;
    bool            is_pc_retained = true;

    // Exception generation: 11010100 opc imm16 op2 LL, with op2 = 000.
    switch (instruction & 0xFFE0001FU)
    {
        case 0xD4000001U: // SVC
            exception      = BAL_EXCEPTION_SUPERVISOR_CALL;
            is_pc_retained = false;
            break;
        case 0xD4200000U: // BRK
            exception = BAL_EXCEPTION_BREAKPOINT;
            break;
        case 0xD4000002U: // HVC
        case 0xD4000003U: // SMC
        case 0xD4400000U: // HLT
            break;
        default:
            if (0U != (instruction & 0xFFFF0000U))
            {
                BAL_LOG_ERROR(&bal_thread_logger,
                              "Aborting function: unsupported exception generating encoding %08x",
                              instruction);
                compiler->status = BAL_ERROR_UNKNOWN_INSTRUCTION;
                return false;
            }

            immediate = instruction & 0xFFFFU; // UDF
            break;
    }

    emit_guest_exception(compiler, exception, immediate);
    return is_pc_retained;
}

/// Sets bal_cpu_t::exception and exception_immediate for the host. The caller ends the block,
/// and terminate_block() then writes the PC and returns.
static void
emit_guest_exception(bal_tier1_compiler_t *BAL_RESTRICT compiler,
                     const bal_exception_t              exception,
                     const uint64_t                     immediate)
{
    // RAX may cache a guest register, so the cache is written back and emptied before RAX is
    // used; otherwise terminate_block() would write the exception kind into that register.
    flush_dirty_registers(compiler);
    reset_register_allocator(compiler);
    bal_sliding_window_flush_all(&compiler->window);
    bal_x86_assembler_t *BAL_RESTRICT assembler = &compiler->assembler;
    bal_x86_emit_mov_r64_imm64(assembler, BAL_X86_RAX, immediate);
    bal_x86_emit_store_r64_rbp_offset(
        assembler, BAL_X86_RAX, offsetof(bal_cpu_t, exception_immediate));
    bal_x86_emit_mov_r64_imm64(assembler, BAL_X86_RAX, (uint64_t)exception);
    bal_x86_emit_store_r32_rbp_offset(assembler, BAL_X86_RAX, offsetof(bal_cpu_t, exception));
}

/// Translates the SYS instructions EL0 can execute: DC ZVA zeroes a 64-byte block through
/// bal_execute_cache_zero(), and the cache maintenance operations DC CVAC, CVAU, CVAP, CVADP,
/// CIVAC and IC IVAU do nothing, since translated code never caches guest memory.
static void
translate_system(bal_tier1_compiler_t *BAL_RESTRICT         compiler,
                 const bal_memory_interface_t *BAL_RESTRICT memory_interface,
                 const uint32_t                             instruction,
                 const bal_guest_address_t                  guest_address)
{
    // SYS #op1, Cn, Cm, #op2: op1 = 011 for the EL0 operations, Cn = C7.
    const uint32_t operation = (instruction >> 5U) & 0x3FFFU; // op1:CRn:CRm:op2

    switch (operation)
    {
        case (3U << 11U) | (7U << 7U) | (4U << 3U) | 1U: // DC ZVA
            emit_memory_helper_call(compiler,
                                    memory_interface,
                                    instruction,
                                    guest_address,
                                    (uint64_t)(uintptr_t)&bal_execute_cache_zero);
            return;
        case (3U << 11U) | (7U << 7U) | (10U << 3U) | 1U: // DC CVAC
        case (3U << 11U) | (7U << 7U) | (11U << 3U) | 1U: // DC CVAU
        case (3U << 11U) | (7U << 7U) | (12U << 3U) | 1U: // DC CVAP
        case (3U << 11U) | (7U << 7U) | (13U << 3U) | 1U: // DC CVADP
        case (3U << 11U) | (7U << 7U) | (14U << 3U) | 1U: // DC CIVAC
        case (3U << 11U) | (7U << 7U) | (5U << 3U) | 1U:  // IC IVAU
            return;
        default:
            BAL_LOG_ERROR(&bal_thread_logger,
                          "Aborting function: unsupported SYS operation %08x",
                          instruction);
            compiler->status = BAL_ERROR_UNKNOWN_INSTRUCTION;
            return;
    }
}

// -----------------------------------------------------------------------------
// Inline floating point and SIMD
// -----------------------------------------------------------------------------

/// The MXCSR inline floating point runs under: every exception masked, round to nearest, no
/// flush to zero or denormals are zero, and no flag raised. Guest FPCR = 0 matches it.
#define GUEST_MXCSR 0x1F80U

#define XMM0 0U
#define XMM1 1U
#define XMM2 2U
#define XMM3 3U

/// The scratch registers of the inline sequences, which the register cache never holds.
#define SCRATCH_A BAL_X86_REGISTER_DISCARD_RESULT
#define SCRATCH_B BAL_X86_REGISTER_ZERO

/// Emits `[prefix] [REX.W] 0F opcode` with two register operands and no immediate.
static void
emit_sse(bal_x86_assembler_t *BAL_RESTRICT assembler,
         const uint8_t                     prefix,
         const bool                        is_64bit,
         const uint8_t                     opcode,
         const uint8_t                     reg,
         const uint8_t                     rm,
         const char                       *mnemonic)
{
    const uint8_t bytes[2] = { 0x0F, opcode };
    bal_x86_emit_sse_register(assembler, prefix, is_64bit, bytes, 2U, reg, rm, false, 0U, mnemonic);
}

/// Emits `[prefix] 0F opcode` with two register operands and an 8-bit immediate.
static void
emit_sse_immediate(bal_x86_assembler_t *BAL_RESTRICT assembler,
                   const uint8_t                     prefix,
                   const uint8_t                     opcode,
                   const uint8_t                     reg,
                   const uint8_t                     rm,
                   const uint8_t                     immediate,
                   const char                       *mnemonic)
{
    const uint8_t bytes[2] = { 0x0F, opcode };
    bal_x86_emit_sse_register(
        assembler, prefix, false, bytes, 2U, reg, rm, true, immediate, mnemonic);
}

/// Emits `[prefix] [REX.W] 0F opcode` with a register and the memory operand [rbp + offset].
static void
emit_sse_memory(bal_x86_assembler_t *BAL_RESTRICT assembler,
                const uint8_t                     prefix,
                const bool                        is_64bit,
                const uint8_t                     opcode,
                const uint8_t                     reg,
                const size_t                      offset,
                const char                       *mnemonic)
{
    const uint8_t bytes[2] = { 0x0F, opcode };
    bal_x86_emit_sse_rbp_offset(
        assembler, prefix, is_64bit, bytes, 2U, reg, (int32_t)offset, mnemonic);
}

static void
emit_ldmxcsr(bal_x86_assembler_t *BAL_RESTRICT assembler, const size_t offset)
{
    emit_sse_memory(assembler, 0U, false, 0xAE, 2U, offset, "ldmxcsr");
}

static void
emit_stmxcsr(bal_x86_assembler_t *BAL_RESTRICT assembler, const size_t offset)
{
    emit_sse_memory(assembler, 0U, false, 0xAE, 3U, offset, "stmxcsr");
}

/// Loads GUEST_MXCSR, which also clears the MXCSR flags.
static void
emit_load_guest_mxcsr(bal_x86_assembler_t *BAL_RESTRICT assembler)
{
    bal_x86_emit_mov_r64_imm64(assembler, SCRATCH_A, GUEST_MXCSR);
    bal_x86_emit_store_r32_rbp_offset(assembler, SCRATCH_A, offsetof(bal_cpu_t, jit_mxcsr));
    emit_ldmxcsr(assembler, offsetof(bal_cpu_t, jit_mxcsr));
}

/// ORs the MXCSR exception flags into FPSR: IE, ZE, OE, UE and PE in bits 0 and 2-5 become IOC,
/// DZC, OFC, UFC and IXC in bits 0-4. DE has no FPSR counterpart while FPCR.FZ is clear.
static void
emit_fold_mxcsr_flags(bal_x86_assembler_t *BAL_RESTRICT assembler)
{
    const bal_x86_register_t flags   = SCRATCH_A;
    const bal_x86_register_t invalid = SCRATCH_B;
    emit_stmxcsr(assembler, offsetof(bal_cpu_t, jit_mxcsr));
    bal_x86_emit_load_r64_rbp_offset(assembler, flags, offsetof(bal_cpu_t, jit_mxcsr));
    bal_x86_emit_mov_r64_r64(assembler, invalid, flags);
    bal_x86_emit_shl_r64_imm8(assembler, invalid, 63U);
    bal_x86_emit_shr_r64_imm8(assembler, invalid, 63U);
    bal_x86_emit_shl_r64_imm8(assembler, flags, 58U);
    bal_x86_emit_shr_r64_imm8(assembler, flags, 60U);
    bal_x86_emit_shl_r64_imm8(assembler, flags, 1U);
    bal_x86_emit_or_r64_r64(assembler, flags, invalid);
    bal_x86_emit_load_r64_rbp_offset(assembler, invalid, offsetof(bal_cpu_t, fpsr));
    bal_x86_emit_or_r64_r64(assembler, invalid, flags);
    bal_x86_emit_store_r64_rbp_offset(assembler, invalid, offsetof(bal_cpu_t, fpsr));
}

/// Switches MXCSR to GUEST_MXCSR the first time a block runs floating point inline, saving the
/// host's value for emit_block_exit().
static void
emit_enter_guest_mxcsr(bal_tier1_compiler_t *BAL_RESTRICT compiler)
{
    if (true == compiler->is_guest_mxcsr_active)
    {
        return;
    }

    emit_stmxcsr(&compiler->assembler, offsetof(bal_cpu_t, jit_host_mxcsr));
    emit_load_guest_mxcsr(&compiler->assembler);
    compiler->is_guest_mxcsr_active = true;
}

/// Leaves the block: when it ran floating point inline, the flags go to FPSR and the host MXCSR
/// comes back first. The sliding window must be flushed.
static void
emit_block_exit(bal_tier1_compiler_t *BAL_RESTRICT compiler)
{
    if (true == compiler->is_guest_mxcsr_active)
    {
        emit_fold_mxcsr_flags(&compiler->assembler);
        emit_ldmxcsr(&compiler->assembler, offsetof(bal_cpu_t, jit_host_mxcsr));
    }

    emit_epilogue(&compiler->assembler);
}

/// Starts an inline sequence with a slow path: the guest registers go back to the cpu structure,
/// since both paths must agree on the register cache, and code is then emitted directly.
static void
begin_inline_sequence(bal_tier1_compiler_t *BAL_RESTRICT compiler)
{
    flush_dirty_registers(compiler);
    reset_register_allocator(compiler);
    bal_sliding_window_flush_all(&compiler->window);
}

/// Starts an inline floating-point sequence: begin_inline_sequence(), the guest MXCSR, and a jump
/// to the slow path unless FPCR is zero, the only mode inline floating point implements.
static size_t
begin_float_sequence(bal_tier1_compiler_t *BAL_RESTRICT compiler)
{
    bal_x86_assembler_t *BAL_RESTRICT assembler = &compiler->assembler;
    begin_inline_sequence(compiler);
    emit_enter_guest_mxcsr(compiler);
    bal_x86_emit_load_r64_rbp_offset(assembler, SCRATCH_A, offsetof(bal_cpu_t, fpcr));
    bal_x86_emit_test_r32_r32(assembler, SCRATCH_A, SCRATCH_A);
    return emit_forward_jump(assembler, true, BAL_X86_COND_NE);
}

/// Loads SIMD&FP register `index` into `xmm`: all 128 bits, or the low 64 with the rest cleared.
static void
emit_load_vector(bal_x86_assembler_t *BAL_RESTRICT assembler,
                 const uint8_t                     xmm,
                 const uint32_t                    index,
                 const bool                        is_quad)
{
    if (true == is_quad)
    {
        emit_sse_memory(
            assembler, 0U, false, 0x10, xmm, offsetof(bal_cpu_t, v[index][0]), "movups");
    }
    else
    {
        emit_sse_memory(
            assembler, 0xF3, false, 0x7E, xmm, offsetof(bal_cpu_t, v[index][0]), "movq");
    }
}

/// Loads the scalar in the low 32 or 64 bits of SIMD&FP register `index` into `xmm`, clearing the
/// other lanes.
static void
emit_load_scalar(bal_x86_assembler_t *BAL_RESTRICT assembler,
                 const uint8_t                     xmm,
                 const uint32_t                    index,
                 const bool                        is_double)
{
    emit_sse_memory(assembler,
                    true == is_double ? 0xF2 : 0xF3,
                    false,
                    0x10,
                    xmm,
                    offsetof(bal_cpu_t, v[index][0]),
                    "movs");
}

/// Clears bits 127:64 of SIMD&FP register `index`.
static void
emit_clear_upper_half(bal_x86_assembler_t *BAL_RESTRICT assembler, const uint32_t index)
{
    bal_x86_emit_xor_r32_r32(assembler, SCRATCH_B, SCRATCH_B);
    bal_x86_emit_store_r64_rbp_offset(
        assembler, SCRATCH_B, (int32_t)offsetof(bal_cpu_t, v[index][1]));
}

/// Stores `xmm` to SIMD&FP register `index`: all 128 bits, or the low 64 with the rest cleared.
static void
emit_store_vector(bal_x86_assembler_t *BAL_RESTRICT assembler,
                  const uint8_t                     xmm,
                  const uint32_t                    index,
                  const bool                        is_quad)
{
    if (true == is_quad)
    {
        emit_sse_memory(
            assembler, 0U, false, 0x11, xmm, offsetof(bal_cpu_t, v[index][0]), "movups");
        return;
    }

    emit_sse_memory(assembler, 0x66, false, 0xD6, xmm, offsetof(bal_cpu_t, v[index][0]), "movq");
    emit_clear_upper_half(assembler, index);
}

/// Stores the scalar result in SCRATCH_A, as emit_scalar_checks() leaves it, to SIMD&FP register
/// `index`, clearing the rest of the register.
static void
emit_store_scalar_result(bal_x86_assembler_t *BAL_RESTRICT assembler, const uint32_t index)
{
    bal_x86_emit_store_r64_rbp_offset(
        assembler, SCRATCH_A, (int32_t)offsetof(bal_cpu_t, v[index][0]));
    emit_clear_upper_half(assembler, index);
}

/// Checks the scalar result in xmm0, adding the jumps to the slow path to `jumps`: a NaN result,
/// whose NaN rules differ, and a result whose magnitude is the smallest normal number, which
/// AArch64 may count as an underflow because it detects tininess before rounding. The result's
/// bits are left zero-extended in SCRATCH_A. Returns the number of jumps added.
static uint32_t
emit_scalar_checks(bal_tier1_compiler_t *BAL_RESTRICT compiler, const bool is_double, size_t *jumps)
{
    bal_x86_assembler_t *BAL_RESTRICT assembler = &compiler->assembler;

    // A NaN result is unordered with itself.
    emit_sse(assembler, true == is_double ? 0x66 : 0U, false, 0x2E, XMM0, XMM0, "ucomis");
    jumps[0] = emit_forward_jump(assembler, true, BAL_X86_COND_P);

    // The magnitude shifted left by one is compared with the smallest normal number's.
    emit_sse(assembler, 0x66, is_double, 0x7E, XMM0, (uint8_t)SCRATCH_A, "movd");
    bal_x86_emit_mov_r64_r64(assembler, SCRATCH_B, SCRATCH_A);

    if (true == is_double)
    {
        bal_x86_emit_shl_r64_imm8(assembler, SCRATCH_B, 1U);
        bal_x86_emit_mov_r64_imm64(assembler, BAL_X86_RAX, 0x0020000000000000ULL);
    }
    else
    {
        bal_x86_emit_shl_r32_imm8(assembler, SCRATCH_B, 1U);
        bal_x86_emit_mov_r64_imm64(assembler, BAL_X86_RAX, 0x01000000ULL);
    }

    push_macro(compiler, BAL_X86_MACRO_SUB_REGISTER_REGISTER, SCRATCH_B, BAL_X86_RAX);
    bal_sliding_window_flush_all(&compiler->window);
    jumps[1] = emit_forward_jump(assembler, true, BAL_X86_COND_E);
    return 2U;
}

/// Checks every lane of the vector result in xmm0 as emit_scalar_checks() does the scalar one,
/// adding the jumps to the slow path to `jumps`. Uses xmm2 and xmm3. Returns the number of jumps.
static uint32_t
emit_vector_checks(bal_x86_assembler_t *BAL_RESTRICT assembler, const bool is_double, size_t *jumps)
{
    const uint8_t packed = true == is_double ? 0x66 : 0U;
    const uint8_t mask   = (uint8_t)SCRATCH_A;

    // CMPUNORD sets a lane to all ones when it holds a NaN; MOVMSK gathers the sign bits.
    emit_sse(assembler, 0U, false, 0x28, XMM3, XMM0, "movaps");
    emit_sse_immediate(assembler, packed, 0xC2, XMM3, XMM3, 3U, "cmpunordp");
    emit_sse(assembler, packed, false, 0x50, mask, XMM3, "movmskp");
    bal_x86_emit_test_r32_r32(assembler, SCRATCH_A, SCRATCH_A);
    jumps[0] = emit_forward_jump(assembler, true, BAL_X86_COND_NE);

    // Each magnitude shifted left by one is compared with the smallest normal number's.
    emit_sse(assembler, 0U, false, 0x28, XMM3, XMM0, "movaps");

    if (true == is_double)
    {
        emit_sse_immediate(assembler, 0x66, 0x73, 6U, XMM3, 1U, "psllq");
        bal_x86_emit_mov_r64_imm64(assembler, SCRATCH_A, 0x0020000000000000ULL);
        emit_sse(assembler, 0x66, true, 0x6E, XMM2, mask, "movq");
        emit_sse(assembler, 0x66, false, 0x6C, XMM2, XMM2, "punpcklqdq");
        emit_sse(assembler, 0x66, false, 0x76, XMM3, XMM2, "pcmpeqd");

        // A 64-bit lane matches when both of its 32-bit halves do.
        emit_sse_immediate(assembler, 0x66, 0x70, XMM2, XMM3, 0xB1, "pshufd");
        emit_sse(assembler, 0x66, false, 0xDB, XMM3, XMM2, "pand");
        emit_sse(assembler, 0x66, false, 0x50, mask, XMM3, "movmskpd");
    }
    else
    {
        emit_sse_immediate(assembler, 0x66, 0x72, 6U, XMM3, 1U, "pslld");
        bal_x86_emit_mov_r64_imm64(assembler, SCRATCH_A, 0x01000000ULL);
        emit_sse(assembler, 0x66, false, 0x6E, XMM2, mask, "movd");
        emit_sse_immediate(assembler, 0x66, 0x70, XMM2, XMM2, 0U, "pshufd");
        emit_sse(assembler, 0x66, false, 0x76, XMM3, XMM2, "pcmpeqd");
        emit_sse(assembler, 0U, false, 0x50, mask, XMM3, "movmskps");
    }

    bal_x86_emit_test_r32_r32(assembler, SCRATCH_A, SCRATCH_A);
    jumps[1] = emit_forward_jump(assembler, true, BAL_X86_COND_NE);
    return 2U;
}

/// Ends an inline sequence with the out-of-line slow path: the jumps in `slow_jumps` lead to a
/// call of `helper`, which executes the instruction with the exact AArch64 rules from the
/// unchanged registers. The flags the fast attempt raised are a subset of the helper's.
static void
emit_slow_path(bal_tier1_compiler_t *BAL_RESTRICT compiler,
               const uint32_t                     instruction,
               const uint64_t                     helper,
               const size_t                      *slow_jumps,
               const uint32_t                     slow_jump_count)
{
    bal_x86_assembler_t *BAL_RESTRICT assembler = &compiler->assembler;
    const size_t                      done = emit_forward_jump(assembler, false, BAL_X86_COND_O);

    for (uint32_t i = 0U; i < slow_jump_count; ++i)
    {
        patch_forward_jump(assembler, slow_jumps[i], true);
    }

    emit_register_helper_call(compiler, instruction, helper);
    bal_sliding_window_flush_all(&compiler->window);
    patch_forward_jump(assembler, done, false);
}

/// FADD, FSUB, FMUL and FDIV (scalar) in single and double precision.
///
/// The fast path computes with SSE under GUEST_MXCSR, where the result and flags equal the
/// AArch64 ones except in the cases emit_scalar_checks() sends to the slow path.
static bool
emit_fast_scalar_arithmetic(bal_tier1_compiler_t *BAL_RESTRICT compiler, const uint32_t instruction)
{
    // M 0 S 11110 ftype 1 Rm opcode 10 Rn Rd with opcode 0000-0011.
    const uint32_t ftype  = (instruction >> 22U) & 0x3U;
    const uint32_t opcode = (instruction >> 12U) & 0xFU;

    if (0x1E200800U != (instruction & 0xFF200C00U) || ftype > 1U || opcode > 3U)
    {
        return false;
    }

    static const uint8_t OPERATIONS[4] = { 0x59, 0x5E, 0x58, 0x5C }; // MUL, DIV, ADD, SUB
    const bool           is_double     = 1U == ftype;
    const uint32_t       rd            = instruction & 0x1FU;
    const uint32_t       rn            = (instruction >> 5U) & 0x1FU;
    const uint32_t       rm            = (instruction >> 16U) & 0x1FU;
    size_t               slow_jumps[3];

    slow_jumps[0] = begin_float_sequence(compiler);
    emit_load_scalar(&compiler->assembler, XMM0, rn, is_double);
    emit_sse_memory(&compiler->assembler,
                    true == is_double ? 0xF2 : 0xF3,
                    false,
                    OPERATIONS[opcode],
                    XMM0,
                    offsetof(bal_cpu_t, v[rm][0]),
                    "arithmetic");
    const uint32_t count = 1U + emit_scalar_checks(compiler, is_double, &slow_jumps[1]);
    emit_store_scalar_result(&compiler->assembler, rd);
    emit_slow_path(
        compiler, instruction, (uint64_t)(uintptr_t)&bal_execute_float, slow_jumps, count);
    return true;
}

/// FMADD, FMSUB, FNMADD and FNMSUB in single and double precision with FMA3, which rounds once as
/// the AArch64 instructions do: VFMADD231, VFNMADD231, VFNMSUB231 and VFMSUB231 give a + n * m,
/// a - n * m, -a - n * m and -a + n * m.
static bool
emit_fast_scalar_fused(bal_tier1_compiler_t *BAL_RESTRICT compiler, const uint32_t instruction)
{
    // M 0 S 11111 ftype o1 Rm o0 Ra Rn Rd.
    const uint32_t ftype = (instruction >> 22U) & 0x3U;

    if (0x1F000000U != (instruction & 0xFF000000U) || ftype > 1U || false == compiler->has_fma)
    {
        return false;
    }

    static const uint8_t OPERATIONS[4] = { 0xB9, 0xBD, 0xBF, 0xBB };
    const uint32_t       variant
        = (((instruction >> 21U) & 1U) << 1U) | ((instruction >> 15U) & 1U); // o1:o0
    const bool                        is_double = 1U == ftype;
    const uint32_t                    rd        = instruction & 0x1FU;
    const uint32_t                    rn        = (instruction >> 5U) & 0x1FU;
    const uint32_t                    ra        = (instruction >> 10U) & 0x1FU;
    const uint32_t                    rm        = (instruction >> 16U) & 0x1FU;
    bal_x86_assembler_t *BAL_RESTRICT assembler = &compiler->assembler;
    size_t                            slow_jumps[3];

    slow_jumps[0] = begin_float_sequence(compiler);
    emit_load_scalar(assembler, XMM0, ra, is_double);
    emit_load_scalar(assembler, XMM1, rn, is_double);
    emit_load_scalar(assembler, XMM2, rm, is_double);
    bal_x86_emit_vex_0f38_register(
        assembler, is_double, OPERATIONS[variant], XMM0, XMM1, XMM2, "vfmadd231s");
    const uint32_t count = 1U + emit_scalar_checks(compiler, is_double, &slow_jumps[1]);
    emit_store_scalar_result(assembler, rd);
    emit_slow_path(
        compiler, instruction, (uint64_t)(uintptr_t)&bal_execute_float, slow_jumps, count);
    return true;
}

/// FADD, FSUB, FMUL, FDIV, and with FMA3 FMLA and FMLS (vector), on 2S, 4S and 2D, with the slow
/// path of emit_vector_checks() for the whole instruction.
static bool
emit_fast_vector_float(bal_tier1_compiler_t *BAL_RESTRICT compiler, const uint32_t instruction)
{
    // 0 Q U 01110 a sz 1 Rm opcode 1 Rn Rd with opcode 11xxx.
    if (0x0E200400U != (instruction & 0x9F200400U))
    {
        return false;
    }

    const bool     is_quad   = 1U == ((instruction >> 30U) & 1U);
    const bool     is_double = 1U == ((instruction >> 22U) & 1U);
    const uint32_t key = (((instruction >> 29U) & 1U) << 6U) | (((instruction >> 23U) & 1U) << 5U)
                         | ((instruction >> 11U) & 0x1FU);
    uint8_t operation = 0U;
    bool    is_fused  = false;

    switch (key)
    {
        case 0x1AU: // FADD
            operation = 0x58;
            break;
        case 0x3AU: // FSUB
            operation = 0x5C;
            break;
        case 0x5BU: // FMUL
            operation = 0x59;
            break;
        case 0x5FU: // FDIV; the cleared upper lanes of a 64-bit operation would divide 0 by 0.
            if (false == is_quad)
            {
                return false;
            }
            operation = 0x5E;
            break;
        case 0x19U: // FMLA: VFMADD231
            operation = 0xB8;
            is_fused  = true;
            break;
        case 0x39U: // FMLS: VFNMADD231, d - n * m with a single rounding.
            operation = 0xBC;
            is_fused  = true;
            break;
        default:
            return false;
    }

    if ((true == is_double && false == is_quad) || (true == is_fused && false == compiler->has_fma))
    {
        return false;
    }

    const uint32_t                    rd        = instruction & 0x1FU;
    const uint32_t                    rn        = (instruction >> 5U) & 0x1FU;
    const uint32_t                    rm        = (instruction >> 16U) & 0x1FU;
    bal_x86_assembler_t *BAL_RESTRICT assembler = &compiler->assembler;
    size_t                            slow_jumps[3];

    slow_jumps[0] = begin_float_sequence(compiler);
    emit_load_vector(assembler, XMM1, rn, is_quad);
    emit_load_vector(assembler, XMM2, rm, is_quad);

    if (true == is_fused)
    {
        emit_load_vector(assembler, XMM0, rd, is_quad);
        bal_x86_emit_vex_0f38_register(
            assembler, is_double, operation, XMM0, XMM1, XMM2, "vfmadd231p");
    }
    else
    {
        emit_sse(assembler, 0U, false, 0x28, XMM0, XMM1, "movaps");
        emit_sse(
            assembler, true == is_double ? 0x66 : 0U, false, operation, XMM0, XMM2, "arithmetic");
    }

    const uint32_t count = 1U + emit_vector_checks(assembler, is_double, &slow_jumps[1]);
    emit_store_vector(assembler, XMM0, rd, is_quad);
    emit_slow_path(
        compiler, instruction, (uint64_t)(uintptr_t)&bal_execute_simd, slow_jumps, count);
    return true;
}

/// Copies lane `lane` of `xmm` to all of its lanes.
static void
emit_broadcast_lane(bal_x86_assembler_t *BAL_RESTRICT assembler,
                    const uint8_t                     xmm,
                    const bool                        is_double,
                    const uint32_t                    lane)
{
    if (true == is_double)
    {
        emit_sse(assembler, 0x66, false, 0 == lane ? 0x14 : 0x15, xmm, xmm, "unpcklpd/unpckhpd");
        return;
    }

    emit_sse_immediate(assembler, 0U, 0xC6, xmm, xmm, (uint8_t)(lane * 0x55U), "shufps");
}

/// FMUL, and with FMA3 FMLA and FMLS (by element), in vector and scalar form: the element is
/// broadcast and the operation is then the vector or scalar one, with the same slow path.
static bool
emit_fast_by_element(bal_tier1_compiler_t *BAL_RESTRICT compiler, const uint32_t instruction)
{
    // Vector 0 Q U 01111 1 sz L M Rm opcode H 0 Rn Rd; scalar 01 U 11111 1 sz L M Rm opcode H 0.
    const bool     is_vector = 0x0F000000U == (instruction & 0x9F000400U);
    const bool     is_scalar = 0x5F000000U == (instruction & 0xDF000400U);
    const uint32_t opcode    = (instruction >> 12U) & 0xFU;

    if ((false == is_vector && false == is_scalar) || 0U == ((instruction >> 23U) & 1U)
        || 1U == ((instruction >> 29U) & 1U)
        || (opcode != 0x1U && opcode != 0x5U && opcode != 0x9U))
    {
        return false;
    }

    const bool     is_quad   = 1U == ((instruction >> 30U) & 1U);
    const bool     is_double = 1U == ((instruction >> 22U) & 1U);
    const bool     is_fused  = opcode != 0x9U;
    const uint32_t l         = (instruction >> 21U) & 1U;
    const uint32_t h         = (instruction >> 11U) & 1U;
    const uint32_t lane      = true == is_double ? h : (h << 1U) | l;

    if ((true == is_double && (1U == l || (true == is_vector && false == is_quad)))
        || (true == is_fused && false == compiler->has_fma))
    {
        return false;
    }

    const uint32_t rd = instruction & 0x1FU;
    const uint32_t rn = (instruction >> 5U) & 0x1FU;
    const uint32_t rm = (instruction >> 16U) & 0x1FU; // M:Rm for the floating-point forms
    bal_x86_assembler_t *BAL_RESTRICT assembler = &compiler->assembler;
    size_t                            slow_jumps[3];

    slow_jumps[0] = begin_float_sequence(compiler);
    emit_load_vector(assembler, XMM2, rm, true);
    emit_broadcast_lane(assembler, XMM2, is_double, lane);
    const uint8_t packed = true == is_double ? 0x66 : 0U;
    const uint8_t scalar = true == is_double ? 0xF2 : 0xF3;

    if (true == is_scalar)
    {
        emit_load_scalar(assembler, XMM1, rn, is_double);

        if (true == is_fused)
        {
            emit_load_scalar(assembler, XMM0, rd, is_double);
            bal_x86_emit_vex_0f38_register(
                assembler, is_double, 0x5U == opcode ? 0xBD : 0xB9, XMM0, XMM1, XMM2, "vfmadd231s");
        }
        else
        {
            emit_sse(assembler, 0U, false, 0x28, XMM0, XMM1, "movaps");
            emit_sse(assembler, scalar, false, 0x59, XMM0, XMM2, "muls");
        }

        const uint32_t count = 1U + emit_scalar_checks(compiler, is_double, &slow_jumps[1]);
        emit_store_scalar_result(assembler, rd);
        emit_slow_path(
            compiler, instruction, (uint64_t)(uintptr_t)&bal_execute_simd, slow_jumps, count);
        return true;
    }

    // A 64-bit operation keeps its upper lanes zero, so they raise no flags.
    if (false == is_quad)
    {
        emit_sse(assembler, 0xF3, false, 0x7E, XMM2, XMM2, "movq");
    }

    emit_load_vector(assembler, XMM1, rn, is_quad);

    if (true == is_fused)
    {
        emit_load_vector(assembler, XMM0, rd, is_quad);
        bal_x86_emit_vex_0f38_register(
            assembler, is_double, 0x5U == opcode ? 0xBC : 0xB8, XMM0, XMM1, XMM2, "vfmadd231p");
    }
    else
    {
        emit_sse(assembler, 0U, false, 0x28, XMM0, XMM1, "movaps");
        emit_sse(assembler, packed, false, 0x59, XMM0, XMM2, "mulp");
    }

    const uint32_t count = 1U + emit_vector_checks(assembler, is_double, &slow_jumps[1]);
    emit_store_vector(assembler, XMM0, rd, is_quad);
    emit_slow_path(
        compiler, instruction, (uint64_t)(uintptr_t)&bal_execute_simd, slow_jumps, count);
    return true;
}

/// FCMP and FCMPE in single and double precision, against a register or zero. UCOMISS raises
/// Invalid Operation for signalling NaNs only and COMISS for every NaN, exactly as FCMP and FCMPE
/// raise IOC, so only a non-zero FPCR needs the slow path. The x86 flags map to NZCV as AE -> C,
/// E -> Z, B -> N and P -> V, except that an unordered result sets ZF and CF on x86 but only C and
/// V on AArch64.
static bool
emit_fast_compare(bal_tier1_compiler_t *BAL_RESTRICT compiler, const uint32_t instruction)
{
    // M 0 S 11110 ftype 1 Rm 00 1000 Rn opcode2 with opcode2 = E 0 0 0 0 or E 1 0 0 0.
    const uint32_t ftype = (instruction >> 22U) & 0x3U;

    if (0x1E202000U != (instruction & 0xFF20FC07U) || ftype > 1U)
    {
        return false;
    }

    const bool                        is_double    = 1U == ftype;
    const bool                        is_zero_form = 1U == ((instruction >> 3U) & 1U);
    const bool                        is_signaling = 1U == ((instruction >> 4U) & 1U);
    const uint32_t                    rn           = (instruction >> 5U) & 0x1FU;
    const uint32_t                    rm           = (instruction >> 16U) & 0x1FU;
    bal_x86_assembler_t *BAL_RESTRICT assembler    = &compiler->assembler;
    size_t                            slow_jumps[1];

    slow_jumps[0] = begin_float_sequence(compiler);
    emit_load_scalar(assembler, XMM0, rn, is_double);

    if (true == is_zero_form)
    {
        emit_sse(assembler, 0U, false, 0x57, XMM1, XMM1, "xorps");
    }
    else
    {
        emit_load_scalar(assembler, XMM1, rm, is_double);
    }

    emit_sse(assembler,
             true == is_double ? 0x66 : 0U,
             false,
             true == is_signaling ? 0x2F : 0x2E,
             XMM0,
             XMM1,
             "comis");
    bal_x86_emit_setcc_mem8_rbp_offset(assembler, BAL_X86_COND_AE, offsetof(bal_cpu_t, flag_c));
    bal_x86_emit_setcc_mem8_rbp_offset(assembler, BAL_X86_COND_E, offsetof(bal_cpu_t, flag_z));
    bal_x86_emit_setcc_mem8_rbp_offset(assembler, BAL_X86_COND_B, offsetof(bal_cpu_t, flag_n));
    bal_x86_emit_setcc_mem8_rbp_offset(assembler, BAL_X86_COND_P, offsetof(bal_cpu_t, flag_v));

    // Unordered: N = 0, Z = 0, C = 1, V = 1, written as the four flag bytes from flag_c.
    const size_t ordered = emit_forward_jump(assembler, true, BAL_X86_COND_NP);
    bal_x86_emit_mov_r64_imm64(assembler, SCRATCH_A, 0x01000001ULL);
    bal_x86_emit_store_r32_rbp_offset(assembler, SCRATCH_A, offsetof(bal_cpu_t, flag_c));
    patch_forward_jump(assembler, ordered, true);

    emit_slow_path(compiler, instruction, (uint64_t)(uintptr_t)&bal_execute_float, slow_jumps, 1U);
    return true;
}

/// SCVTF (scalar, integer) from a W or X register to single or double precision: CVTSI2SS rounds
/// to nearest under GUEST_MXCSR and raises Inexact as SCVTF raises IXC, and the result is never a
/// NaN or tiny.
static bool
emit_fast_convert_from_integer(bal_tier1_compiler_t *BAL_RESTRICT compiler,
                               const uint32_t                     instruction)
{
    // sf 0 0 11110 ftype 1 00 010 000000 Rn Rd.
    const uint32_t ftype = (instruction >> 22U) & 0x3U;

    if (0x1E220000U != (instruction & 0x7F3FFC00U) || ftype > 1U)
    {
        return false;
    }

    const bool                        is_double = 1U == ftype;
    const bool                        is_64bit  = 1U == (instruction >> 31U);
    const uint32_t                    rd        = instruction & 0x1FU;
    const uint32_t                    rn        = (instruction >> 5U) & 0x1FU;
    bal_x86_assembler_t *BAL_RESTRICT assembler = &compiler->assembler;
    size_t                            slow_jumps[1];

    slow_jumps[0] = begin_float_sequence(compiler);

    if (31U == rn)
    {
        bal_x86_emit_xor_r32_r32(assembler, SCRATCH_A, SCRATCH_A);
    }
    else
    {
        bal_x86_emit_load_r64_rbp_offset(assembler, SCRATCH_A, (int32_t)offsetof(bal_cpu_t, x[rn]));
    }

    // CVTSI2SS merges into the destination, so xmm0 is cleared first.
    emit_sse(assembler, 0U, false, 0x57, XMM0, XMM0, "xorps");
    emit_sse(assembler,
             true == is_double ? 0xF2 : 0xF3,
             is_64bit,
             0x2A,
             XMM0,
             (uint8_t)SCRATCH_A,
             "cvtsi2s");
    emit_sse(assembler, 0x66, is_double, 0x7E, XMM0, (uint8_t)SCRATCH_A, "movd");
    emit_store_scalar_result(assembler, rd);
    emit_slow_path(compiler, instruction, (uint64_t)(uintptr_t)&bal_execute_float, slow_jumps, 1U);
    return true;
}

/// FCVTZS (scalar, integer) from single or double precision to a W or X register. CVTTSS2SI
/// truncates exactly as FCVTZS does and raises Inexact as it raises IXC; out of range values and
/// NaNs give the integer indefinite value, where AArch64 saturates, so that value takes the slow
/// path. The most negative integer, a valid result, takes it too and gets the same answer.
static bool
emit_fast_convert_to_integer(bal_tier1_compiler_t *BAL_RESTRICT compiler,
                             const uint32_t                     instruction)
{
    // sf 0 0 11110 ftype 1 11 000 000000 Rn Rd.
    const uint32_t ftype = (instruction >> 22U) & 0x3U;

    if (0x1E380000U != (instruction & 0x7F3FFC00U) || ftype > 1U)
    {
        return false;
    }

    const bool                        is_double = 1U == ftype;
    const bool                        is_64bit  = 1U == (instruction >> 31U);
    const uint32_t                    rd        = instruction & 0x1FU;
    const uint32_t                    rn        = (instruction >> 5U) & 0x1FU;
    bal_x86_assembler_t *BAL_RESTRICT assembler = &compiler->assembler;
    size_t                            slow_jumps[2];

    slow_jumps[0] = begin_float_sequence(compiler);
    emit_sse_memory(assembler,
                    true == is_double ? 0xF2 : 0xF3,
                    is_64bit,
                    0x2C,
                    (uint8_t)SCRATCH_A,
                    offsetof(bal_cpu_t, v[rn][0]),
                    "cvtts2si");
    bal_x86_emit_mov_r64_imm64(
        assembler, BAL_X86_RAX, true == is_64bit ? 0x8000000000000000ULL : 0x80000000ULL);
    bal_x86_emit_mov_r64_r64(assembler, SCRATCH_B, SCRATCH_A);
    push_macro(compiler, BAL_X86_MACRO_SUB_REGISTER_REGISTER, SCRATCH_B, BAL_X86_RAX);
    bal_sliding_window_flush_all(&compiler->window);
    slow_jumps[1] = emit_forward_jump(assembler, true, BAL_X86_COND_E);

    // A W result is zero-extended by the 32-bit conversion.
    if (rd != 31U)
    {
        bal_x86_emit_store_r64_rbp_offset(
            assembler, SCRATCH_A, (int32_t)offsetof(bal_cpu_t, x[rd]));
    }

    emit_slow_path(compiler, instruction, (uint64_t)(uintptr_t)&bal_execute_float, slow_jumps, 2U);
    return true;
}

/// Pushes a LOAD of the 64-bit cpu structure field at `offset` into `destination`.
static void
push_load(bal_tier1_compiler_t *BAL_RESTRICT compiler,
          const bal_x86_register_t           destination,
          const size_t                       offset)
{
    const bal_x86_macro_t macro = {
        .opcode              = BAL_X86_MACRO_LOAD,
        .destination         = destination,
        .immediate_or_offset = offset,
    };
    bal_sliding_window_push(&compiler->window, macro);
}

/// Pushes a STORE of `source` to the 64-bit cpu structure field at `offset`.
static void
push_store(bal_tier1_compiler_t *BAL_RESTRICT compiler,
           const bal_x86_register_t           source,
           const size_t                       offset)
{
    const bal_x86_macro_t macro = {
        .opcode              = BAL_X86_MACRO_STORE,
        .source              = source,
        .immediate_or_offset = offset,
    };
    bal_sliding_window_push(&compiler->window, macro);
}

/// Pushes a scalar write of SCRATCH_A to SIMD&FP register `index`, clearing bits 127:64.
static void
push_scalar_write(bal_tier1_compiler_t *BAL_RESTRICT compiler, const uint32_t index)
{
    push_store(compiler, SCRATCH_A, offsetof(bal_cpu_t, v[index][0]));
    const bal_x86_macro_t zero_macro = {
        .opcode              = BAL_X86_MACRO_MOV_REGISTER_IMMEDIATE,
        .destination         = SCRATCH_A,
        .immediate_or_offset = 0U,
    };
    bal_sliding_window_push(&compiler->window, zero_macro);
    push_store(compiler, SCRATCH_A, offsetof(bal_cpu_t, v[index][1]));
}

/// FMOV (register), FABS and FNEG (scalar), and FMOV between general purpose and SIMD&FP
/// registers, in single and double precision. They move bits without arithmetic, so they raise
/// no exceptions and need no slow path; the general purpose side uses the register cache.
static bool
emit_fast_move(bal_tier1_compiler_t *BAL_RESTRICT compiler, const uint32_t instruction)
{
    const uint32_t rd = instruction & 0x1FU;
    const uint32_t rn = (instruction >> 5U) & 0x1FU;

    // FMOV, FABS, FNEG: M 0 S 11110 ftype 1 0000 opc 10000 Rn Rd with opc 00-10.
    const uint32_t ftype = (instruction >> 22U) & 0x3U;

    if (0x1E204000U == (instruction & 0xFF3E7C00U) && ftype <= 1U
        && ((instruction >> 15U) & 0x3U) <= 2U)
    {
        const bool     is_double = 1U == ftype;
        const uint32_t operation = (instruction >> 15U) & 0x3U; // FMOV, FABS, FNEG
        push_load(compiler, SCRATCH_A, offsetof(bal_cpu_t, v[rn][0]));

        if (false == is_double)
        {
            push_macro(compiler, BAL_X86_MACRO_MOV_REGISTER_REGISTER_32, SCRATCH_A, SCRATCH_A);
        }

        if (1U == operation)
        {
            // Clear the sign bit by shifting it out and back.
            const bal_x86_macro_opcode_t left  = true == is_double
                                                     ? BAL_X86_MACRO_SHL_REGISTER_IMMEDIATE
                                                     : BAL_X86_MACRO_SHL_REGISTER_IMMEDIATE_32;
            const bal_x86_macro_opcode_t right = true == is_double
                                                     ? BAL_X86_MACRO_SHR_REGISTER_IMMEDIATE
                                                     : BAL_X86_MACRO_SHR_REGISTER_IMMEDIATE_32;
            const bal_x86_macro_t        shift_left
                = { .opcode = left, .destination = SCRATCH_A, .immediate_or_offset = 1U };
            const bal_x86_macro_t shift_right
                = { .opcode = right, .destination = SCRATCH_A, .immediate_or_offset = 1U };
            bal_sliding_window_push(&compiler->window, shift_left);
            bal_sliding_window_push(&compiler->window, shift_right);
        }
        else if (2U == operation)
        {
            // Adding the sign bit flips it and leaves the other bits alone.
            const bal_x86_macro_t sign_macro = {
                .opcode              = BAL_X86_MACRO_MOV_REGISTER_IMMEDIATE,
                .destination         = SCRATCH_B,
                .immediate_or_offset = true == is_double ? 0x8000000000000000ULL : 0x80000000ULL,
            };
            bal_sliding_window_push(&compiler->window, sign_macro);
            push_macro(compiler,
                       true == is_double ? BAL_X86_MACRO_ADD_REGISTER_REGISTER
                                         : BAL_X86_MACRO_ADD_REGISTER_REGISTER_32,
                       SCRATCH_A,
                       SCRATCH_B);
        }

        push_scalar_write(compiler, rd);
        return true;
    }

    // FMOV (general): W to and from S, X to and from D, and X to and from the top half of Q.
    switch (instruction & 0xFFFFFC00U)
    {
        case 0x1E260000U: // FMOV Wd, Sn
        case 0x9E660000U: // FMOV Xd, Dn
        case 0x9EAE0000U: // FMOV Xd, Vn.D[1]
        {
            if (31U == rd)
            {
                return true;
            }

            const bool               skip_load = true;
            const bal_x86_register_t x86_rd
                = allocate_x86_register(compiler, (uint8_t)rd, skip_load);
            push_load(
                compiler,
                x86_rd,
                offsetof(bal_cpu_t, v[rn][0x9EAE0000U == (instruction & 0xFFFFFC00U) ? 1 : 0]));

            if (0x1E260000U == (instruction & 0xFFFFFC00U))
            {
                push_macro(compiler, BAL_X86_MACRO_MOV_REGISTER_REGISTER_32, x86_rd, x86_rd);
            }

            compiler->is_dirty |= 1U << rd;
            return true;
        }
        case 0x1E270000U: // FMOV Sd, Wn
        case 0x9E670000U: // FMOV Dd, Xn
        case 0x9EAF0000U: // FMOV Vd.D[1], Xn
        {
            if (31U == rn)
            {
                const bal_x86_macro_t zero_macro = {
                    .opcode              = BAL_X86_MACRO_MOV_REGISTER_IMMEDIATE,
                    .destination         = SCRATCH_A,
                    .immediate_or_offset = 0U,
                };
                bal_sliding_window_push(&compiler->window, zero_macro);
            }
            else
            {
                const bool               skip_load = false;
                const bal_x86_register_t x86_rn
                    = allocate_x86_register(compiler, (uint8_t)rn, skip_load);
                push_macro(compiler,
                           0x1E270000U == (instruction & 0xFFFFFC00U)
                               ? BAL_X86_MACRO_MOV_REGISTER_REGISTER_32
                               : BAL_X86_MACRO_MOV_REGISTER_REGISTER,
                           SCRATCH_A,
                           x86_rn);
            }

            if (0x9EAF0000U == (instruction & 0xFFFFFC00U))
            {
                push_store(compiler, SCRATCH_A, offsetof(bal_cpu_t, v[rd][1]));
            }
            else
            {
                push_scalar_write(compiler, rd);
            }

            return true;
        }
        default:
            return false;
    }
}

/// FCSEL in single and double precision: a selection of bits on the guest flags, evaluated as
/// CSEL evaluates them.
static bool
emit_fast_select(bal_tier1_compiler_t *BAL_RESTRICT compiler, const uint32_t instruction)
{
    // M 0 S 11110 ftype 1 Rm cond 11 Rn Rd.
    const uint32_t ftype = (instruction >> 22U) & 0x3U;

    if (0x1E200C00U != (instruction & 0xFF200C00U) || ftype > 1U)
    {
        return false;
    }

    const uint32_t condition = (instruction >> 12U) & 0xFU;
    const uint32_t rd        = instruction & 0x1FU;
    const uint32_t rn        = (instruction >> 5U) & 0x1FU;
    const uint32_t rm        = (instruction >> 16U) & 0x1FU;
    push_load(compiler, SCRATCH_A, offsetof(bal_cpu_t, v[rn][0]));

    // AL and NV always select Rn.
    if (condition < 14U)
    {
        const bal_x86_condition_t fail_condition = emit_condition_value(compiler, condition);
        push_macro(compiler, BAL_X86_MACRO_TEST_REGISTER_REGISTER, SCRATCH_A, SCRATCH_A);

        // Loads leave the flags alone.
        push_load(compiler, SCRATCH_A, offsetof(bal_cpu_t, v[rn][0]));
        push_load(compiler, SCRATCH_B, offsetof(bal_cpu_t, v[rm][0]));
        const bal_x86_macro_t select_macro = {
            .opcode      = BAL_X86_MACRO_CMOV_REGISTER_REGISTER,
            .destination = SCRATCH_A,
            .source      = SCRATCH_B,
            .condition   = fail_condition,
        };
        bal_sliding_window_push(&compiler->window, select_macro);
    }

    if (0U == ftype)
    {
        push_macro(compiler, BAL_X86_MACRO_MOV_REGISTER_REGISTER_32, SCRATCH_A, SCRATCH_A);
    }

    push_scalar_write(compiler, rd);
    return true;
}

/// DUP (element) on 32-bit and 64-bit elements: PSHUFD copies the lane to every lane.
static bool
emit_fast_duplicate(bal_tier1_compiler_t *BAL_RESTRICT compiler, const uint32_t instruction)
{
    // 0 Q 0 01110000 imm5 0 0000 1 Rn Rd.
    if (0x0E000400U != (instruction & 0xBFE0FC00U))
    {
        return false;
    }

    const uint32_t imm5    = (instruction >> 16U) & 0x1FU;
    const bool     is_quad = 1U == ((instruction >> 30U) & 1U);
    uint8_t        pattern = 0U;

    if (0x4U == (imm5 & 0x7U)) // 32-bit lanes
    {
        pattern = (uint8_t)((imm5 >> 3U) * 0x55U);
    }
    else if (0x8U == (imm5 & 0xFU) && true == is_quad) // 64-bit lanes
    {
        pattern = 0U == (imm5 >> 4U) ? 0x44 : 0xEE;
    }
    else
    {
        return false;
    }

    const uint32_t                    rd        = instruction & 0x1FU;
    const uint32_t                    rn        = (instruction >> 5U) & 0x1FU;
    bal_x86_assembler_t *BAL_RESTRICT assembler = &compiler->assembler;

    bal_sliding_window_flush_all(&compiler->window);
    emit_load_vector(assembler, XMM0, rn, true);
    emit_sse_immediate(assembler, 0x66, 0x70, XMM0, XMM0, pattern, "pshufd");
    emit_store_vector(assembler, XMM0, rd, is_quad);
    return true;
}

/// ADD, SUB, AND, BIC, ORR and EOR (vector), which SSE2 computes exactly; MOV is ORR with Rn = Rm.
static bool
emit_fast_vector_integer(bal_tier1_compiler_t *BAL_RESTRICT compiler, const uint32_t instruction)
{
    // 0 Q U 01110 size 1 Rm opcode 1 Rn Rd.
    if (0x0E200400U != (instruction & 0x9F200400U))
    {
        return false;
    }

    const bool     is_quad   = 1U == ((instruction >> 30U) & 1U);
    const uint32_t u         = (instruction >> 29U) & 1U;
    const uint32_t size      = (instruction >> 22U) & 0x3U;
    const uint32_t opcode    = (instruction >> 11U) & 0x1FU;
    uint8_t        operation = 0U;
    bool           is_bic    = false;

    if (0x10U == opcode)
    {
        static const uint8_t ADDS[4] = { 0xFC, 0xFD, 0xFE, 0xD4 };
        static const uint8_t SUBS[4] = { 0xF8, 0xF9, 0xFA, 0xFB };

        if (3U == size && false == is_quad)
        {
            return false;
        }

        operation = 0U == u ? ADDS[size] : SUBS[size];
    }
    else if (0x03U == opcode)
    {
        switch ((u << 2U) | size)
        {
            case 0U:
                operation = 0xDB; // AND: PAND
                break;
            case 1U:
                operation = 0xDF; // BIC: PANDN computes ~m & n
                is_bic    = true;
                break;
            case 2U:
                operation = 0xEB; // ORR: POR
                break;
            case 4U:
                operation = 0xEF; // EOR: PXOR
                break;
            default:
                return false;
        }
    }
    else
    {
        return false;
    }

    const uint32_t                    rd        = instruction & 0x1FU;
    const uint32_t                    rn        = (instruction >> 5U) & 0x1FU;
    const uint32_t                    rm        = (instruction >> 16U) & 0x1FU;
    bal_x86_assembler_t *BAL_RESTRICT assembler = &compiler->assembler;

    // Only the cpu structure and xmm registers are involved, so the register cache stays.
    bal_sliding_window_flush_all(&compiler->window);
    emit_load_vector(assembler, XMM0, rn, is_quad);
    emit_load_vector(assembler, XMM1, rm, is_quad);

    if (true == is_bic)
    {
        emit_sse(assembler, 0x66, false, operation, XMM1, XMM0, "pandn");
        emit_store_vector(assembler, XMM1, rd, is_quad);
        return true;
    }

    emit_sse(assembler, 0x66, false, operation, XMM0, XMM1, "padd/psub/logical");
    emit_store_vector(assembler, XMM0, rd, is_quad);
    return true;
}

/// Emits an inline translation of the SIMD&FP data processing instruction when tier 1 has one.
/// Returns false to leave it to the runtime helpers.
static bool
emit_fast_float(bal_tier1_compiler_t *BAL_RESTRICT compiler, const uint32_t instruction)
{
    return emit_fast_scalar_arithmetic(compiler, instruction)
           || emit_fast_scalar_fused(compiler, instruction)
           || emit_fast_vector_float(compiler, instruction)
           || emit_fast_by_element(compiler, instruction)
           || emit_fast_compare(compiler, instruction)
           || emit_fast_convert_from_integer(compiler, instruction)
           || emit_fast_convert_to_integer(compiler, instruction)
           || emit_fast_move(compiler, instruction) || emit_fast_select(compiler, instruction)
           || emit_fast_duplicate(compiler, instruction)
           || emit_fast_vector_integer(compiler, instruction);
}
