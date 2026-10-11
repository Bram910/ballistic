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

#include <cpuid.h>
#include <string.h>

// Set which registers holds the first argument passed to the JIT block based on the OS calling
// convention.
#if BAL_PLATFORM_WINDOWS

#define BAL_X86_ABI_ARG1 BAL_X86_RCX
#define BAL_X86_ABI_ARG2 BAL_X86_RDX
#define BAL_X86_ABI_ARG3 BAL_X86_R8
#define BAL_X86_ABI_ARG4 BAL_X86_R9

// The Windows x64 calling convention makes the caller reserve 32 bytes of shadow space.
#define BAL_X86_ABI_SHADOW_SPACE 32U

#else

#define BAL_X86_ABI_ARG1         BAL_X86_RDI
#define BAL_X86_ABI_ARG2         BAL_X86_RSI
#define BAL_X86_ABI_ARG3         BAL_X86_RDX
#define BAL_X86_ABI_ARG4         BAL_X86_RCX
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

static void emit_exit_to(bal_tier1_compiler_t *compiler, bal_guest_address_t next_pc);

static void emit_indirect_exit(bal_tier1_compiler_t *compiler);

static size_t emit_forward_jump(bal_x86_assembler_t *assembler,
                                bool                 is_conditional,
                                bal_x86_condition_t  condition);

static void patch_forward_jump(bal_x86_assembler_t *assembler, size_t jump, bool is_conditional);

static void emit_fold_mxcsr_flags(bal_x86_assembler_t *assembler);

static void emit_load_guest_mxcsr(bal_x86_assembler_t *assembler);

static bool emit_fast_float(bal_tier1_compiler_t *compiler, uint32_t instruction);

static uint64_t float_helper(uint32_t instruction);

static void emit_load_half(bal_x86_assembler_t *assembler, uint8_t xmm, uint32_t index);

static void emit_halves_to_singles(bal_x86_assembler_t *assembler,
                                   uint8_t              destination,
                                   uint8_t              source);

static void emit_singles_to_halves(bal_x86_assembler_t *assembler,
                                   uint8_t              destination,
                                   uint8_t              source);

static size_t emit_half_checks(bal_x86_assembler_t *assembler);

static void emit_saturation_if_any(bal_x86_assembler_t *assembler, uint8_t xmm);

static void emit_load_element(bal_x86_assembler_t *assembler,
                              uint8_t              xmm,
                              uint32_t             index,
                              uint32_t             size);

static void emit_store_element(bal_x86_assembler_t *assembler,
                               uint8_t              xmm,
                               uint32_t             index,
                               uint32_t             size);

static bool emit_fast_vector_interleaved(bal_tier1_compiler_t         *compiler,
                                         const bal_memory_interface_t *memory_interface,
                                         uint32_t                      instruction,
                                         bal_guest_address_t           guest_address);

static bool emit_fast_vector_single(bal_tier1_compiler_t         *compiler,
                                    const bal_memory_interface_t *memory_interface,
                                    uint32_t                      instruction,
                                    bal_guest_address_t           guest_address);

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
static void emit_memory_thunk(bal_tier1_compiler_t *compiler);
static void translate_pstate(bal_tier1_compiler_t *compiler, uint32_t instruction);
static void emit_store_cpu_word(bal_tier1_compiler_t *compiler, int32_t offset, uint32_t value);
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

/// Whether the host has the BAL_TIER1_FEATURE_* bit `feature`.
static bool
has_feature(const bal_tier1_compiler_t *BAL_RESTRICT compiler, const uint32_t feature)
{
    return 0U != (compiler->features & feature);
}

/// The BAL_TIER1_FEATURE_* bits of the host. __builtin_cpu_supports() cannot test F16C (CPUID.1:ECX
/// bit 29) or the SHA extensions (CPUID.7.0:EBX bit 29), so CPUID does.
static uint8_t
host_features(void)
{
    unsigned int eax      = 0U;
    unsigned int ebx      = 0U;
    unsigned int ecx      = 0U;
    unsigned int edx      = 0U;
    const bool   has_f16c = 0 != __get_cpuid(1U, &eax, &ebx, &ecx, &edx) && 0U != (ecx & bit_F16C);
    const bool   has_sha
        = 0 != __get_cpuid_count(7U, 0U, &eax, &ebx, &ecx, &edx) && 0U != (ebx & bit_SHA);
    uint32_t features = 0U;

    if (0 != __builtin_cpu_supports("fma") && true == has_f16c)
    {
        features |= BAL_TIER1_FEATURE_FMA;
    }

    if (0 != __builtin_cpu_supports("sse4.1"))
    {
        features |= BAL_TIER1_FEATURE_SSE41;
    }

    if (0 != __builtin_cpu_supports("sse4.2"))
    {
        features |= BAL_TIER1_FEATURE_SSE42;
    }

    if (0 != __builtin_cpu_supports("aes") && 0 != __builtin_cpu_supports("pclmul"))
    {
        features |= BAL_TIER1_FEATURE_AES;
    }

    if (0 != __builtin_cpu_supports("avx2"))
    {
        features |= BAL_TIER1_FEATURE_AVX2;
    }

    if (true == has_sha && 0U != (features & BAL_TIER1_FEATURE_SSE41))
    {
        features |= BAL_TIER1_FEATURE_SHA;
    }

    return (uint8_t)features;
}

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
    compiler->features      = host_features();
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

    emit_memory_thunk(compiler);
    return compiler->assembler.status;
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
    emit_memory_thunk(compiler);
}

/// Translates up to `max_instructions` instructions at `guest_address` into one block. When an
/// instruction cannot be translated, returns NULL with BAL_ERROR_UNKNOWN_INSTRUCTION, the number
/// of instructions before it in `translated_count` and its encoding in `failed_instruction`.
static void *
translate_block(bal_tier1_compiler_t         *compiler,
                const bal_memory_interface_t *memory_interface,
                bal_guest_address_t           guest_address,
                const size_t                  max_instructions,
                const uint32_t                engine_flags,
                size_t                       *translated_count,
                uint32_t                     *failed_instruction)
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

    if (BAL_UNLIKELY(compiler->assembler.offset - start_x86_block_offset
                     != BAL_TIER1_BLOCK_PROLOGUE_SIZE))
    {
        BAL_LOG_ERROR(&bal_thread_logger, "Aborting function: unexpected block prologue size");
        compiler->status = BAL_ERROR_UNKNOWN_INSTRUCTION;
        return NULL;
    }

    compiler->is_linkable
        = 0U
          == (engine_flags & (BAL_ENGINE_FLAG_SINGLE_STEP | BAL_ENGINE_FLAG_DISABLE_BLOCK_CACHE));

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
            *failed_instruction                                   = instruction;

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

                    // ISB ends the block and tells the engine to discard its translations if an
                    // IC IVAU said the guest has written code; see bal_cpu_t::instruction_barrier.
                    if (0xD50330DFU == (instruction & 0xFFFFF0FFU))
                    {
                        emit_store_cpu_word(
                            compiler, (int32_t)offsetof(bal_cpu_t, instruction_barrier), 1U);
                        compiler->is_linkable = false;
                        is_block_terminated   = true;
                        break;
                    }

                    // DMB and DSB order memory accesses for other observers. x86 keeps loads and
                    // stores in order except for a load passing an earlier store, so a full fence
                    // is always sufficient. SB, CLREX and the hints need nothing here.
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

                    // MSR (immediate): bits 31:19 = 1101010100000, CRn = 0100 and Rt = 11111.
                    if (0xD500401FU == (instruction & 0xFFF8F01FU))
                    {
                        translate_pstate(compiler, instruction);
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

            if (arm_instruction_count == max_instructions)
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
        *translated_count = arm_instruction_count;
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

/// Emits a block that raises BAL_EXCEPTION_UNSUPPORTED_INSTRUCTION for `instruction` at
/// `guest_address`, leaving the PC on it.
static void *
translate_unsupported(bal_tier1_compiler_t     *compiler,
                      const bal_guest_address_t guest_address,
                      const uint32_t            instruction,
                      const uint32_t            engine_flags)
{
    reset_register_allocator(compiler);
    void *host_address = compiler->assembler.rx_buffer + compiler->assembler.offset;

    bal_x86_emit_push_r64(&compiler->assembler, BAL_X86_RBP);
    bal_x86_emit_push_r64(&compiler->assembler, BAL_X86_R12);
    bal_x86_emit_push_r64(&compiler->assembler, BAL_X86_R13);
    bal_x86_emit_push_r64(&compiler->assembler, BAL_X86_R14);
    bal_x86_emit_push_r64(&compiler->assembler, BAL_X86_R15);
    bal_x86_emit_mov_r64_r64(&compiler->assembler, BAL_X86_RBP, BAL_X86_ABI_ARG1);

    emit_guest_exception(compiler, BAL_EXCEPTION_UNSUPPORTED_INSTRUCTION, instruction);
    terminate_block(compiler, guest_address, false, 1U, engine_flags);

    if (BAL_UNLIKELY(compiler->status != BAL_SUCCESS || compiler->assembler.status != BAL_SUCCESS))
    {
        return NULL;
    }

    return host_address;
}

void *
bal_tier1_compiler_translate(bal_tier1_compiler_t         *compiler,
                             const bal_memory_interface_t *memory_interface,
                             const bal_guest_address_t     guest_address,
                             const size_t                  max_instructions,
                             const uint32_t                engine_flags)
{
    if (BAL_UNLIKELY(NULL == compiler))
    {
        return NULL;
    }

    const size_t start_offset       = compiler->assembler.offset;
    size_t       translated_count   = 0;
    uint32_t     failed_instruction = 0;
    void        *block              = translate_block(compiler,
                                  memory_interface,
                                  guest_address,
                                  max_instructions,
                                  engine_flags,
                                  &translated_count,
                                  &failed_instruction);

    // An instruction tier 1 cannot translate raises an exception only when it is reached, and an
    // instruction fetch fault only when execution gets to the unmapped address: the partial block
    // is dropped and the block ends before the instruction. A block starting with an
    // untranslatable instruction raises the exception at once.
    const bool is_unknown = BAL_ERROR_UNKNOWN_INSTRUCTION == compiler->status;
    const bool is_late_fetch_fault
        = (BAL_ERROR_MEMORY_FAULT == compiler->status || BAL_ERROR_PC_ALIGNMENT == compiler->status)
          && translated_count > 0U;

    if (BAL_LIKELY(NULL != block) || compiler->assembler.status != BAL_SUCCESS
        || (false == is_unknown && false == is_late_fetch_fault))
    {
        return block;
    }

    compiler->assembler.offset = start_offset;
    compiler->status           = BAL_SUCCESS;
    bal_sliding_window_reset(&compiler->window);

    if (translated_count > 0U)
    {
        return translate_block(compiler,
                               memory_interface,
                               guest_address,
                               translated_count,
                               engine_flags,
                               &translated_count,
                               &failed_instruction);
    }

    BAL_LOG_INFO(&bal_thread_logger,
                 "Unsupported instruction %08x at GVA 0x%016llX raises an exception",
                 failed_instruction,
                 (unsigned long long)guest_address);
    return translate_unsupported(compiler, guest_address, failed_instruction, engine_flags);
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
        emit_exit_to(compiler, next_pc);
        return;
    }

    emit_indirect_exit(compiler);
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

    emit_exit_to(compiler, fallthrough_pc);
    patch_forward_jump(assembler, jump, true);
    flush_dirty_registers(compiler);
    bal_sliding_window_flush_all(&compiler->window);
    emit_exit_to(compiler, taken_pc);
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
            emit_exit_to(compiler, taken_pc);
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
#define SYSTEM_REGISTER_DIT         SYSTEM_REGISTER(3U, 3U, 4U, 2U, 5U)
#define SYSTEM_REGISTER_SSBS        SYSTEM_REGISTER(3U, 3U, 4U, 2U, 6U)

#define PSTATE_DIT  (1ULL << 24U)
#define PSTATE_SSBS (1ULL << 12U)

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

    // FPSR includes the flags inline floating point has raised into MXCSR, in this block or the
    // ones before it. They go to FPSR first, and MXCSR is cleared so that a write replaces them.
    if (SYSTEM_REGISTER_FPSR == system_register)
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
        case SYSTEM_REGISTER_DIT:
            offset = offsetof(bal_cpu_t, dit);
            mask   = PSTATE_DIT;
            break;
        case SYSTEM_REGISTER_SSBS:
            offset = offsetof(bal_cpu_t, ssbs);
            mask   = PSTATE_SSBS;
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

/// Translates MSR (immediate) to DIT and SSBS, which set the field from bit 0 of CRm. The other
/// PSTATE fields are EL1 ones, or belong to extensions the guest CPU lacks.
static void
translate_pstate(bal_tier1_compiler_t *BAL_RESTRICT compiler, const uint32_t instruction)
{
    const uint32_t op1    = (instruction >> 16U) & 0x7U;
    const uint32_t op2    = (instruction >> 5U) & 0x7U;
    const uint64_t is_on  = (instruction >> 8U) & 1U;
    size_t         offset = 0U;
    uint64_t       bit    = 0U;

    if (3U == op1 && 2U == op2)
    {
        offset = offsetof(bal_cpu_t, dit);
        bit    = PSTATE_DIT;
    }
    else if (3U == op1 && 1U == op2)
    {
        offset = offsetof(bal_cpu_t, ssbs);
        bit    = PSTATE_SSBS;
    }
    else
    {
        BAL_LOG_ERROR(
            &bal_thread_logger, "Aborting function: unsupported PSTATE field %08x", instruction);
        compiler->status = BAL_ERROR_UNKNOWN_INSTRUCTION;
        return;
    }

    const bal_x86_macro_t value_macro = {
        .opcode              = BAL_X86_MACRO_MOV_REGISTER_IMMEDIATE,
        .destination         = BAL_X86_REGISTER_ZERO,
        .immediate_or_offset = 0U != is_on ? bit : 0U,
    };
    bal_sliding_window_push(&compiler->window, value_macro);

    const bal_x86_macro_t store_macro = {
        .opcode              = BAL_X86_MACRO_STORE,
        .source              = BAL_X86_REGISTER_ZERO,
        .immediate_or_offset = offset,
    };
    bal_sliding_window_push(&compiler->window, store_macro);
}

/// Called from translated code to turn a guest address into a host pointer. Returns the host
/// address of `size` contiguous bytes at `guest_address`, or NULL when the memory interface does
/// not map all of them.
static const uint8_t *
translate_data_address(const bal_cpu_t          *cpu,
                       const bal_guest_address_t guest_address,
                       const uint64_t            size)
{
    const bal_memory_interface_t *memory_interface
        = (const bal_memory_interface_t *)(uintptr_t)cpu->jit_memory_interface;
    size_t         available_bytes = 0U;
    const uint8_t *host_address
        = memory_interface->translate((void *)memory_interface, guest_address, &available_bytes);

    if (NULL == host_address || available_bytes < size)
    {
        return NULL;
    }

    // Cache the page when the memory interface maps all of it contiguously, so the next accesses
    // to it stay in translated code.
    const bal_guest_address_t page_base
        = guest_address & ~(bal_guest_address_t)(BAL_TIER1_GUEST_PAGE_SIZE - 1U);
    size_t         page_bytes = 0U;
    const uint8_t *page_host
        = memory_interface->translate((void *)memory_interface, page_base, &page_bytes);

    if (page_host != NULL && page_bytes >= BAL_TIER1_GUEST_PAGE_SIZE
        && page_host + (guest_address - page_base) == host_address)
    {
        bal_tier1_tlb_entry_t *tlb   = (bal_tier1_tlb_entry_t *)(uintptr_t)cpu->jit_memory_tlb;
        const uint64_t         page  = guest_address >> 12U;
        bal_tier1_tlb_entry_t *entry = &tlb[page & (BAL_TIER1_TLB_ENTRIES - 1U)];
        entry->page                  = page;
        entry->host_offset           = (uint64_t)(uintptr_t)page_host - page_base;
    }

    return host_address;
}

uint64_t
bal_tier1_fpsr_flags_from_mxcsr(const uint32_t mxcsr)
{
    return (uint64_t)((mxcsr & 1U) | (((mxcsr >> 2U) & 0xFU) << 1U));
}

void
bal_tier1_tlb_clear(bal_tier1_tlb_entry_t *tlb)
{
    for (uint32_t i = 0U; i < BAL_TIER1_TLB_ENTRIES; ++i)
    {
        tlb[i].page        = BAL_TIER1_TLB_EMPTY_PAGE;
        tlb[i].host_offset = 0U;
    }
}

/// Calls the C function at `function`, whose arguments are already in the ABI argument
/// registers. The return value is left in RAX.
///
/// The call may clobber every caller-saved register, so callers write dirty guest registers back
/// and empty the register cache first. R13 and RBP are callee-saved and survive the call.
static void
emit_runtime_call(bal_tier1_compiler_t *BAL_RESTRICT compiler, const uint64_t function)
{
    // Translated code runs under BAL_TIER1_GUEST_MXCSR, with the flags inline floating point
    // raises left in MXCSR. The floating-point helpers clear and test the flags themselves, so
    // the pending ones go to FPSR first, and whatever the helper leaves behind is cleared after
    // the call. The other helpers do no floating point.
    const bool is_float_helper = (uint64_t)(uintptr_t)&bal_execute_float == function
                                 || (uint64_t)(uintptr_t)&bal_execute_simd == function;

    if (true == is_float_helper)
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

    if (true == is_float_helper)
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

/// Emits the TLB miss handler every block shares, at the start of the code buffer. It takes the
/// guest address in BAL_X86_REGISTER_ZERO and the access size in BAL_X86_REGISTER_DISCARD_RESULT,
/// calls translate_data_address(), and returns the host address, or zero, in
/// BAL_X86_REGISTER_ZERO with every guest register preserved. translate_data_address() does no
/// floating point, so MXCSR can stay as it is.
static void
emit_memory_thunk(bal_tier1_compiler_t *BAL_RESTRICT compiler)
{
    static const bal_x86_register_t CALLER_SAVED[]
        = { BAL_X86_RAX, BAL_X86_RCX, BAL_X86_RDX, BAL_X86_R8, BAL_X86_R9, BAL_X86_R10 };
    const size_t                      count     = sizeof(CALLER_SAVED) / sizeof(CALLER_SAVED[0]);
    bal_x86_assembler_t *BAL_RESTRICT assembler = &compiler->assembler;

    for (size_t i = 0U; i < count; ++i)
    {
        bal_x86_emit_push_r64(assembler, CALLER_SAVED[i]);
    }

    // The return address and six registers leave RSP 8 bytes off the 16-byte alignment calls
    // need.
    bal_x86_emit_sub_r64_imm32(assembler, BAL_X86_RSP, 8 + (int32_t)BAL_X86_ABI_SHADOW_SPACE);
    bal_x86_emit_mov_r64_r64(assembler, BAL_X86_ABI_ARG3, BAL_X86_REGISTER_DISCARD_RESULT);
    bal_x86_emit_mov_r64_r64(assembler, BAL_X86_ABI_ARG2, BAL_X86_REGISTER_ZERO);
    bal_x86_emit_mov_r64_r64(assembler, BAL_X86_ABI_ARG1, BAL_X86_RBP);
    bal_x86_emit_mov_r64_imm64(
        assembler, BAL_X86_RAX, (uint64_t)(uintptr_t)&translate_data_address);
    bal_x86_emit_call_r64(assembler, BAL_X86_RAX);
    bal_x86_emit_add_r64_imm32(assembler, BAL_X86_RSP, 8 + (int32_t)BAL_X86_ABI_SHADOW_SPACE);
    bal_x86_emit_mov_r64_r64(assembler, BAL_X86_REGISTER_ZERO, BAL_X86_RAX);

    for (size_t i = count; i > 0U; --i)
    {
        bal_x86_emit_pop_r64(assembler, CALLER_SAVED[i - 1U]);
    }

    bal_x86_emit_ret(assembler);
}

/// Turns the guest address in BAL_X86_REGISTER_ZERO into a host pointer in the same register for
/// an access of `size` bytes, and raises a data abort when the address is not mapped.
///
/// The page is looked up in the software TLB (bal_cpu_t::jit_memory_tlb) first; an access within
/// a cached page needs no call. Otherwise translate_data_address() asks the memory interface and
/// caches the page. The register cache is the same on both paths, so the guest registers stay in
/// their host registers across the access.
static void
emit_translate_data_address(bal_tier1_compiler_t *BAL_RESTRICT         compiler,
                            const bal_memory_interface_t *BAL_RESTRICT memory_interface,
                            const uint64_t                             size,
                            const bal_guest_address_t                  guest_address)
{
    const bal_x86_register_t          address   = BAL_X86_REGISTER_ZERO;
    const bal_x86_register_t          scratch   = BAL_X86_REGISTER_DISCARD_RESULT;
    bal_x86_assembler_t *BAL_RESTRICT assembler = &compiler->assembler;
    (void)memory_interface;

    // The guest registers stay in their host registers on both paths: the fast path saves the
    // two it borrows, and the slow path the caller-saved ones the call may clobber.
    bal_sliding_window_flush_all(&compiler->window);
    bal_x86_emit_push_r64(assembler, BAL_X86_RAX);
    bal_x86_emit_push_r64(assembler, BAL_X86_RCX);

    // RAX = page number, RCX = its TLB entry.
    bal_x86_emit_mov_r64_r64(assembler, BAL_X86_RAX, address);
    bal_x86_emit_shr_r64_imm8(assembler, BAL_X86_RAX, 12U);
    bal_x86_emit_mov_r64_r64(assembler, BAL_X86_RCX, BAL_X86_RAX);
    bal_x86_emit_shl_r64_imm8(assembler, BAL_X86_RCX, 56U);
    bal_x86_emit_shr_r64_imm8(assembler, BAL_X86_RCX, 52U);
    bal_x86_emit_load_r64_rbp_offset(assembler, scratch, offsetof(bal_cpu_t, jit_memory_tlb));
    bal_x86_emit_add_r64_r64(assembler, BAL_X86_RCX, scratch);

    // The entry must hold this page, and the access must not cross into the next one.
    bal_x86_emit_load_r64_mem64(assembler, scratch, BAL_X86_RCX);
    bal_x86_emit_xor_r64_r64(assembler, scratch, BAL_X86_RAX);
    const size_t is_other_page = emit_forward_jump(assembler, true, BAL_X86_COND_NE);
    bal_x86_emit_mov_r64_r64(assembler, scratch, address);
    bal_x86_emit_shl_r64_imm8(assembler, scratch, 52U);
    bal_x86_emit_shr_r64_imm8(assembler, scratch, 52U);
    bal_x86_emit_add_r64_imm32(assembler, scratch, (int32_t)size - 1);
    bal_x86_emit_shr_r64_imm8(assembler, scratch, 12U);
    const size_t is_crossing = emit_forward_jump(assembler, true, BAL_X86_COND_NE);

    bal_x86_emit_add_r64_imm32(assembler, BAL_X86_RCX, 8);
    bal_x86_emit_load_r64_mem64(assembler, scratch, BAL_X86_RCX);
    bal_x86_emit_add_r64_r64(assembler, address, scratch);
    bal_x86_emit_pop_r64(assembler, BAL_X86_RCX);
    bal_x86_emit_pop_r64(assembler, BAL_X86_RAX);
    const size_t is_translated = emit_forward_jump(assembler, false, BAL_X86_COND_E);

    // Slow path: translate_data_address() through the memory interface, which also fills the
    // TLB entry.
    patch_forward_jump(assembler, is_other_page, true);
    patch_forward_jump(assembler, is_crossing, true);
    bal_x86_emit_pop_r64(assembler, BAL_X86_RCX);
    bal_x86_emit_pop_r64(assembler, BAL_X86_RAX);
    bal_x86_emit_mov_r64_imm64(assembler, scratch, size);
    bal_x86_emit_call_rel32(assembler, (int32_t)(0 - (int64_t)(assembler->offset + 5U)));

    // An unmapped address ends the block with a data abort, after the guest registers are
    // written back.
    bal_x86_emit_test_r64_r64(assembler, address, address);
    const size_t is_mapped = emit_forward_jump(assembler, true, BAL_X86_COND_NE);
    flush_dirty_registers(compiler);
    bal_sliding_window_flush_all(&compiler->window);
    bal_x86_emit_mov_r64_imm64(assembler, BAL_X86_RAX, guest_address);
    bal_x86_emit_store_r64_rbp_offset(assembler, BAL_X86_RAX, offsetof(bal_cpu_t, pc));
    bal_x86_emit_mov_r64_imm64(assembler, BAL_X86_RAX, BAL_EXCEPTION_DATA_ABORT);
    bal_x86_emit_store_r32_rbp_offset(assembler, BAL_X86_RAX, offsetof(bal_cpu_t, exception));
    emit_block_exit(compiler);

    patch_forward_jump(assembler, is_mapped, true);
    patch_forward_jump(assembler, is_translated, false);
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

/// Ends a post-indexed structure load or store: the base register advances by the transfer size
/// `total` (Rm = 31) or by Xm. The access has left the base alone, so a fault leaves it unchanged.
static void
emit_structure_writeback(bal_tier1_compiler_t *BAL_RESTRICT compiler,
                         const uint8_t                      rn,
                         const uint8_t                      rm,
                         const uint64_t                     total)
{
    const bool               skip_load = false;
    const bal_x86_register_t x86_rn    = allocate_x86_register(compiler, rn, skip_load);

    if (31U == rm)
    {
        const bal_x86_macro_t add_macro = {
            .opcode              = BAL_X86_MACRO_ADD_REGISTER_IMMEDIATE,
            .destination         = x86_rn,
            .immediate_or_offset = total,
        };
        bal_sliding_window_push(&compiler->window, add_macro);
    }
    else
    {
        const bal_x86_register_t x86_rm = allocate_x86_register(compiler, rm, skip_load);
        push_macro(compiler, BAL_X86_MACRO_ADD_REGISTER_REGISTER, x86_rn, x86_rm);
    }

    compiler->is_dirty |= 1U << rn;
}

/// LD1 and ST1 (multiple structures) with one to four registers, without an offset or
/// post-indexed. They copy whole registers to or from consecutive memory, so after the address
/// translation they are plain 64-bit moves. Returns false for the other structure forms.
static bool
emit_fast_vector_structure(bal_tier1_compiler_t *BAL_RESTRICT         compiler,
                           const bal_memory_interface_t *BAL_RESTRICT memory_interface,
                           const uint32_t                             instruction,
                           const bal_guest_address_t                  guest_address)
{
    // 0 Q 0011000 L 000000 opcode size Rn Rt, or post-indexed 0 Q 0011001 L 0 Rm opcode size Rn Rt.
    const bool is_plain = 0x0C000000U == (instruction & 0xBFBF0000U);
    const bool is_post  = 0x0C800000U == (instruction & 0xBFA00000U);

    if (false == is_plain && false == is_post)
    {
        return false;
    }

    uint32_t count = 0U;

    switch ((instruction >> 12U) & 0xFU)
    {
        case 0x7U:
            count = 1U;
            break;
        case 0xAU:
            count = 2U;
            break;
        case 0x6U:
            count = 3U;
            break;
        case 0x2U:
            count = 4U;
            break;
        default:
            return false; // LD2 to LD4 interleave the elements.
    }

    const bool               is_quad = 1U == ((instruction >> 30U) & 1U);
    const bool               is_load = 1U == ((instruction >> 22U) & 1U);
    const uint8_t            rt      = (uint8_t)(instruction & 0x1FU);
    const uint8_t            rn      = (uint8_t)((instruction >> 5U) & 0x1FU);
    const uint8_t            rm      = (uint8_t)((instruction >> 16U) & 0x1FU);
    const uint32_t           halves  = true == is_quad ? 2U : 1U;
    const uint64_t           total   = (uint64_t)count * halves * 8U;
    const bal_x86_register_t pointer = BAL_X86_REGISTER_ZERO;
    const bal_x86_register_t data    = BAL_X86_REGISTER_DISCARD_RESULT;

    emit_base_plus_offset(compiler, rn, 0);
    emit_translate_data_address(compiler, memory_interface, total, guest_address);

    bal_x86_assembler_t *BAL_RESTRICT assembler = &compiler->assembler;
    bal_sliding_window_flush_all(&compiler->window);

    for (uint32_t r = 0U; r < count; ++r)
    {
        const uint32_t index = (rt + r) % 32U;

        for (uint32_t half = 0U; half < halves; ++half)
        {
            const int32_t offset = (int32_t)offsetof(bal_cpu_t, v[index][half]);

            if (true == is_load)
            {
                bal_x86_emit_load_r64_mem64(assembler, data, pointer);
                bal_x86_emit_store_r64_rbp_offset(assembler, data, offset);
            }
            else
            {
                bal_x86_emit_load_r64_rbp_offset(assembler, data, offset);
                bal_x86_emit_store_mem64_r64(assembler, pointer, data);
            }

            bal_x86_emit_add_r64_imm32(assembler, pointer, 8);
        }

        // A 64-bit load clears the upper half of the register.
        if (true == is_load && false == is_quad)
        {
            bal_x86_emit_xor_r32_r32(assembler, data, data);
            bal_x86_emit_store_r64_rbp_offset(
                assembler, data, (int32_t)offsetof(bal_cpu_t, v[index][1]));
        }
    }

    if (true == is_post)
    {
        emit_structure_writeback(compiler, rn, rm, total);
    }

    return true;
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

        if (true
                == emit_fast_vector_structure(
                    compiler, memory_interface, instruction, guest_address)
            || true
                   == emit_fast_vector_interleaved(
                       compiler, memory_interface, instruction, guest_address)
            || true
                   == emit_fast_vector_single(
                       compiler, memory_interface, instruction, guest_address))
        {
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

    // The hint-space forms other than XPACLRI do nothing while their key is disabled, which is
    // the common case, so they test the enable bit and skip the call. The test happens at run time
    // because the host may enable the keys after the code is translated.
    const bool is_keyed_hint
        = 0xD503201FU == (instruction & 0xFFFFF01FU) && instruction != 0xD50320FFU;

    if (true == is_keyed_hint)
    {
        flush_dirty_registers(compiler);
        reset_register_allocator(compiler);
        bal_sliding_window_flush_all(&compiler->window);

        // op2 bit 1 picks the B key, enabled by SCTLR_EL1.EnIB (bit 30) rather than EnIA (31).
        bal_x86_assembler_t *BAL_RESTRICT assembler  = &compiler->assembler;
        const uint8_t                     enable_bit = 0U != ((instruction >> 6U) & 1U) ? 30U : 31U;
        bal_x86_emit_load_r64_rbp_offset(
            assembler, BAL_X86_REGISTER_DISCARD_RESULT, (int32_t)offsetof(bal_cpu_t, sctlr_el1));
        bal_x86_emit_bt_r64_imm8(assembler, BAL_X86_REGISTER_DISCARD_RESULT, enable_bit);
        const size_t skip = emit_forward_jump(assembler, true, BAL_X86_COND_AE);

        emit_register_helper_call(
            compiler, instruction, (uint64_t)(uintptr_t)&bal_execute_pointer_authentication);
        bal_sliding_window_flush_all(&compiler->window);
        patch_forward_jump(assembler, skip, true);
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
    // The block must return to the engine, which reports the exception.
    compiler->is_linkable = false;

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

/// Stores `value` to the 32-bit field of the cpu structure at `offset`.
static void
emit_store_cpu_word(bal_tier1_compiler_t *BAL_RESTRICT compiler,
                    const int32_t                      offset,
                    const uint32_t                     value)
{
    bal_sliding_window_flush_all(&compiler->window);
    bal_x86_emit_mov_r64_imm64(&compiler->assembler, BAL_X86_REGISTER_DISCARD_RESULT, value);
    bal_x86_emit_store_r32_rbp_offset(
        &compiler->assembler, BAL_X86_REGISTER_DISCARD_RESULT, offset);
}

/// Translates the SYS instructions EL0 can execute: DC ZVA zeroes a 64-byte block through
/// bal_execute_cache_zero(), IC IVAU records in bal_cpu_t::instruction_cache_invalidated that the
/// guest has written code, and the data cache maintenance operations DC CVAC, CVAU, CVAP, CVADP
/// and CIVAC do nothing, since translated code never caches guest memory.
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
            return;
        case (3U << 11U) | (7U << 7U) | (5U << 3U) | 1U: // IC IVAU
            emit_store_cpu_word(
                compiler, (int32_t)offsetof(bal_cpu_t, instruction_cache_invalidated), 1U);
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

/// The scratch SSE registers beyond xmm3, caller-saved in both host calling conventions.
#define XMM4 4U
#define XMM5 5U

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

/// Leaves the block for the guest address `next_pc`.
///
/// A linkable exit first checks bal_cpu_t::jit_exit_requested, then has a jump the engine can
/// point at the block for `next_pc`, skipping its prologue. Until then the jump falls through
/// to the ordinary exit, which records the jump's address in bal_cpu_t::jit_link_site, stores
/// the PC and returns to the engine.
static void
emit_exit_to(bal_tier1_compiler_t *BAL_RESTRICT compiler, const bal_guest_address_t next_pc)
{
    bal_x86_assembler_t *BAL_RESTRICT assembler = &compiler->assembler;

    if (true == compiler->is_linkable)
    {
        bal_x86_emit_cmp_mem8_rbp_offset_imm(
            assembler, (int32_t)offsetof(bal_cpu_t, jit_exit_requested), 0);
        const size_t is_exit_requested = emit_forward_jump(assembler, true, BAL_X86_COND_NE);
        const size_t link_site         = assembler->offset;
        bal_x86_emit_jmp_rel32(assembler, 0);
        patch_forward_jump(assembler, is_exit_requested, true);
        bal_x86_emit_mov_r64_imm64(
            assembler, BAL_X86_RAX, (uint64_t)(uintptr_t)(assembler->rx_buffer + link_site));
        bal_x86_emit_store_r64_rbp_offset(
            assembler, BAL_X86_RAX, offsetof(bal_cpu_t, jit_link_site));
    }

    bal_x86_emit_mov_r64_imm64(assembler, BAL_X86_RAX, next_pc);
    bal_x86_emit_store_r64_rbp_offset(assembler, BAL_X86_RAX, offsetof(bal_cpu_t, pc));
    emit_epilogue(assembler);
}

/// Leaves the block for the address in the guest PC, which an indirect branch has stored.
///
/// A linkable exit looks the PC up in the engine's table of translated blocks and jumps
/// straight into the block it finds; on a miss, or when the host has asked for the engine back,
/// it returns to the engine. Every guest register is in the cpu structure by now, so RAX, RCX
/// and RDX are free.
static void
emit_indirect_exit(bal_tier1_compiler_t *BAL_RESTRICT compiler)
{
    bal_x86_assembler_t *BAL_RESTRICT assembler = &compiler->assembler;

    if (true == compiler->is_linkable)
    {
        bal_x86_emit_cmp_mem8_rbp_offset_imm(
            assembler, (int32_t)offsetof(bal_cpu_t, jit_exit_requested), 0);
        const size_t is_exit_requested = emit_forward_jump(assembler, true, BAL_X86_COND_NE);

        // RCX = table + ((pc >> 2) & (entries - 1)) * 16.
        bal_x86_emit_load_r64_rbp_offset(assembler, BAL_X86_RAX, offsetof(bal_cpu_t, pc));
        bal_x86_emit_mov_r64_r64(assembler, BAL_X86_RCX, BAL_X86_RAX);
        bal_x86_emit_shr_r64_imm8(assembler, BAL_X86_RCX, 2U);
        bal_x86_emit_mov_r64_imm64(assembler, BAL_X86_RDX, BAL_TIER1_LOOKUP_ENTRIES - 1U);
        bal_x86_emit_and_r64_r64(assembler, BAL_X86_RCX, BAL_X86_RDX);
        bal_x86_emit_shl_r64_imm8(assembler, BAL_X86_RCX, 4U);
        bal_x86_emit_load_r64_rbp_offset(
            assembler, BAL_X86_RDX, offsetof(bal_cpu_t, jit_lookup_table));
        bal_x86_emit_add_r64_r64(assembler, BAL_X86_RCX, BAL_X86_RDX);

        // The entry must hold this PC and a block.
        bal_x86_emit_load_r64_mem64(assembler, BAL_X86_RDX, BAL_X86_RCX);
        bal_x86_emit_xor_r64_r64(assembler, BAL_X86_RDX, BAL_X86_RAX);
        const size_t is_other_pc = emit_forward_jump(assembler, true, BAL_X86_COND_NE);
        bal_x86_emit_add_r64_imm32(assembler, BAL_X86_RCX, 8);
        bal_x86_emit_load_r64_mem64(assembler, BAL_X86_RDX, BAL_X86_RCX);
        bal_x86_emit_test_r64_r64(assembler, BAL_X86_RDX, BAL_X86_RDX);
        const size_t is_empty = emit_forward_jump(assembler, true, BAL_X86_COND_E);
        bal_x86_emit_jmp_r64(assembler, BAL_X86_RDX);

        patch_forward_jump(assembler, is_exit_requested, true);
        patch_forward_jump(assembler, is_other_pc, true);
        patch_forward_jump(assembler, is_empty, true);
    }

    emit_epilogue(assembler);
}

static void
emit_block_exit(bal_tier1_compiler_t *BAL_RESTRICT compiler)
{
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

/// Stores `xmm` to SIMD&FP register `index`: all 128 bits, or the low 64 with the rest cleared,
/// which clears the upper half of `xmm` too. One 128-bit store lets a later 128-bit load of the
/// register take its value from the store buffer, which two 64-bit stores would not.
static void
emit_store_vector(bal_x86_assembler_t *BAL_RESTRICT assembler,
                  const uint8_t                     xmm,
                  const uint32_t                    index,
                  const bool                        is_quad)
{
    if (false == is_quad)
    {
        emit_sse(assembler, 0xF3, false, 0x7E, xmm, xmm, "movq");
    }

    emit_sse_memory(assembler, 0U, false, 0x11, xmm, offsetof(bal_cpu_t, v[index][0]), "movups");
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

    if (0x1F000000U != (instruction & 0xFF000000U) || ftype > 1U
        || false == has_feature(compiler, BAL_TIER1_FEATURE_FMA))
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

    if ((true == is_double && false == is_quad)
        || (true == is_fused && false == has_feature(compiler, BAL_TIER1_FEATURE_FMA)))
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
        || (true == is_fused && false == has_feature(compiler, BAL_TIER1_FEATURE_FMA)))
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

/// Sets NZCV as FCMP, or with `is_signaling` FCMPE, does for the scalars in xmm0 and xmm1, in
/// single or double precision.
static void
emit_compare_xmm_flags(bal_x86_assembler_t *BAL_RESTRICT assembler,
                       const bool                        is_double,
                       const bool                        is_signaling)
{
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
}

/// Sets NZCV as FCMP, or with `is_signaling` FCMPE, does for the scalars in SIMD&FP registers
/// `rn` and `rm`, or `rn` and zero with `is_zero_form`, in single or double precision.
static void
emit_compare_flags(bal_x86_assembler_t *BAL_RESTRICT assembler,
                   const bool                        is_double,
                   const bool                        is_signaling,
                   const uint32_t                    rn,
                   const uint32_t                    rm,
                   const bool                        is_zero_form)
{
    emit_load_scalar(assembler, XMM0, rn, is_double);

    if (true == is_zero_form)
    {
        emit_sse(assembler, 0U, false, 0x57, XMM1, XMM1, "xorps");
    }
    else
    {
        emit_load_scalar(assembler, XMM1, rm, is_double);
    }

    emit_compare_xmm_flags(assembler, is_double, is_signaling);
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
    emit_compare_flags(assembler, is_double, is_signaling, rn, rm, is_zero_form);

    emit_slow_path(compiler, instruction, (uint64_t)(uintptr_t)&bal_execute_float, slow_jumps, 1U);
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

/// Pushes the clearing of all but the low `bits` bits of `reg`, with a shift out and back.
static void
push_keep_low_bits(bal_tier1_compiler_t *BAL_RESTRICT compiler,
                   const bal_x86_register_t           reg,
                   const uint32_t                     bits)
{
    const bal_x86_macro_t shift_left = {
        .opcode              = BAL_X86_MACRO_SHL_REGISTER_IMMEDIATE,
        .destination         = reg,
        .immediate_or_offset = 64U - bits,
    };
    const bal_x86_macro_t shift_right = {
        .opcode              = BAL_X86_MACRO_SHR_REGISTER_IMMEDIATE,
        .destination         = reg,
        .immediate_or_offset = 64U - bits,
    };
    bal_sliding_window_push(&compiler->window, shift_left);
    bal_sliding_window_push(&compiler->window, shift_right);
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

    // In half precision: the low 16 bits, with the sign cleared or flipped.
    if (0x1EE04000U == (instruction & 0xFFFE7C00U) && ((instruction >> 15U) & 0x3U) <= 2U)
    {
        const uint32_t operation = (instruction >> 15U) & 0x3U;
        push_load(compiler, SCRATCH_A, offsetof(bal_cpu_t, v[rn][0]));

        if (2U == operation)
        {
            const bal_x86_macro_t sign_macro = {
                .opcode              = BAL_X86_MACRO_MOV_REGISTER_IMMEDIATE,
                .destination         = SCRATCH_B,
                .immediate_or_offset = 0x8000U,
            };
            bal_sliding_window_push(&compiler->window, sign_macro);
            push_macro(compiler, BAL_X86_MACRO_XOR_REGISTER_REGISTER, SCRATCH_A, SCRATCH_B);
        }

        push_keep_low_bits(compiler, SCRATCH_A, 1U == operation ? 15U : 16U);
        push_scalar_write(compiler, rd);
        return true;
    }

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
    // W and X to and from H move the low 16 bits.
    switch (instruction & 0xFFFFFC00U)
    {
        case 0x1EE60000U: // FMOV Wd, Hn
        case 0x9EE60000U: // FMOV Xd, Hn
        {
            if (31U == rd)
            {
                return true;
            }

            const bool               skip_load = true;
            const bal_x86_register_t x86_rd
                = allocate_x86_register(compiler, (uint8_t)rd, skip_load);
            push_load(compiler, x86_rd, offsetof(bal_cpu_t, v[rn][0]));
            push_keep_low_bits(compiler, x86_rd, 16U);
            compiler->is_dirty |= 1U << rd;
            return true;
        }
        case 0x1EE70000U: // FMOV Hd, Wn
        case 0x9EE70000U: // FMOV Hd, Xn
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
                push_macro(compiler, BAL_X86_MACRO_MOV_REGISTER_REGISTER, SCRATCH_A, x86_rn);
                push_keep_low_bits(compiler, SCRATCH_A, 16U);
            }

            push_scalar_write(compiler, rd);
            return true;
        }
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

/// FCSEL in half, single and double precision: a selection of bits on the guest flags, evaluated
/// as CSEL evaluates them.
static bool
emit_fast_select(bal_tier1_compiler_t *BAL_RESTRICT compiler, const uint32_t instruction)
{
    // M 0 S 11110 ftype 1 Rm cond 11 Rn Rd.
    const uint32_t ftype = (instruction >> 22U) & 0x3U;

    if (0x1E200C00U != (instruction & 0xFF200C00U) || 2U == ftype)
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
    else if (3U == ftype)
    {
        push_keep_low_bits(compiler, SCRATCH_A, 16U);
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

// -----------------------------------------------------------------------------
// More inline Advanced SIMD
// -----------------------------------------------------------------------------

/// Emits `66 0F 38 opcode` (SSSE3 and SSE4.1) with two register operands.
static void
emit_sse38(bal_x86_assembler_t *BAL_RESTRICT assembler,
           const uint8_t                     opcode,
           const uint8_t                     reg,
           const uint8_t                     rm,
           const char                       *mnemonic)
{
    const uint8_t bytes[3] = { 0x0F, 0x38, opcode };
    bal_x86_emit_sse_register(assembler, 0x66, false, bytes, 3U, reg, rm, false, 0U, mnemonic);
}

/// Emits `66 0F 3A opcode ib` (SSSE3 and SSE4.1) with two register operands.
static void
emit_sse3a(bal_x86_assembler_t *BAL_RESTRICT assembler,
           const uint8_t                     opcode,
           const uint8_t                     reg,
           const uint8_t                     rm,
           const uint8_t                     immediate,
           const char                       *mnemonic)
{
    const uint8_t bytes[3] = { 0x0F, 0x3A, opcode };
    bal_x86_emit_sse_register(
        assembler, 0x66, false, bytes, 3U, reg, rm, true, immediate, mnemonic);
}

/// Sets every bit of `xmm`.
static void
emit_all_ones(bal_x86_assembler_t *BAL_RESTRICT assembler, const uint8_t xmm)
{
    emit_sse(assembler, 0x66, false, 0x76, xmm, xmm, "pcmpeqd");
}

/// Clears `xmm`.
static void
emit_zero_vector(bal_x86_assembler_t *BAL_RESTRICT assembler, const uint8_t xmm)
{
    emit_sse(assembler, 0x66, false, 0xEF, xmm, xmm, "pxor");
}

/// Fills both 64-bit halves of `xmm` with `pattern`.
static void
emit_broadcast_pattern(bal_x86_assembler_t *BAL_RESTRICT assembler,
                       const uint8_t                     xmm,
                       const uint64_t                    pattern)
{
    bal_x86_emit_mov_r64_imm64(assembler, SCRATCH_A, pattern);
    emit_sse(assembler, 0x66, true, 0x6E, xmm, (uint8_t)SCRATCH_A, "movq");
    emit_sse(assembler, 0x66, false, 0x6C, xmm, xmm, "punpcklqdq");
}

/// The sign bit of every `8 << size`-bit lane, as a 64-bit pattern.
static uint64_t
lane_sign_pattern(const uint32_t size)
{
    static const uint64_t PATTERNS[4] = {
        0x8080808080808080ULL, 0x8000800080008000ULL, 0x8000000080000000ULL, 0x8000000000000000ULL
    };
    return PATTERNS[size];
}

/// Emits PCMPEQB/W/D/Q `a`, `b`: a lane becomes all ones where they are equal. Quadwords need
/// SSE4.1, which the caller checks.
static void
emit_lanes_equal(bal_x86_assembler_t *BAL_RESTRICT assembler,
                 const uint32_t                    size,
                 const uint8_t                     a,
                 const uint8_t                     b)
{
    static const uint8_t OPCODES[3] = { 0x74, 0x75, 0x76 };

    if (3U == size)
    {
        emit_sse38(assembler, 0x29, a, b, "pcmpeqq");
        return;
    }

    emit_sse(assembler, 0x66, false, OPCODES[size], a, b, "pcmpeq");
}

/// Emits PCMPGTB/W/D/Q `a`, `b`: a lane becomes all ones where `a` is greater, signed.
/// Doublewords need SSE4.2, which the caller checks.
static void
emit_lanes_greater(bal_x86_assembler_t *BAL_RESTRICT assembler,
                   const uint32_t                    size,
                   const uint8_t                     a,
                   const uint8_t                     b)
{
    static const uint8_t OPCODES[3] = { 0x64, 0x65, 0x66 };

    if (3U == size)
    {
        emit_sse38(assembler, 0x37, a, b, "pcmpgtq");
        return;
    }

    emit_sse(assembler, 0x66, false, OPCODES[size], a, b, "pcmpgt");
}

/// Inverts `xmm`, using `scratch`.
static void
emit_not_vector(bal_x86_assembler_t *BAL_RESTRICT assembler,
                const uint8_t                     xmm,
                const uint8_t                     scratch)
{
    emit_all_ones(assembler, scratch);
    emit_sse(assembler, 0x66, false, 0xEF, xmm, scratch, "pxor");
}

/// Advanced SIMD three same, integer: ADD, SUB, the logical operations (AND, BIC, ORR, ORN, EOR,
/// BSL, BIT, BIF), the comparisons (CMEQ, CMTST, CMGT, CMHI, CMGE, CMHS), SMAX, SMIN, UMAX, UMIN
/// and MUL, and the scalar ADD, SUB and comparisons, which work on doublewords. None of them
/// touches FPSR, so they need no slow path.
static bool
emit_fast_vector_integer(bal_tier1_compiler_t *BAL_RESTRICT compiler, const uint32_t instruction)
{
    // 0 Q U 01110 size 1 Rm opcode 1 Rn Rd, or the scalar 01 U 11110 size 1 Rm opcode 1 Rn Rd.
    const bool is_scalar = 0x5E200400U == (instruction & 0xDF200400U);

    if (0x0E200400U != (instruction & 0x9F200400U) && false == is_scalar)
    {
        return false;
    }

    const bool     is_quad = false == is_scalar && 1U == ((instruction >> 30U) & 1U);
    const uint32_t u       = (instruction >> 29U) & 1U;
    const uint32_t size    = (instruction >> 22U) & 0x3U;
    const uint32_t opcode  = (instruction >> 11U) & 0x1FU;
    const bool     sse41   = has_feature(compiler, BAL_TIER1_FEATURE_SSE41);

    if (true == is_scalar
        && (size != 3U
            || (opcode != 0x10U && opcode != 0x11U && opcode != 0x06U && opcode != 0x07U)))
    {
        return false;
    }

    // SMAX/SMIN on halfwords and UMAX/UMIN on bytes are SSE2; the rest need SSE4.1. Each entry is
    // the opcode, with 0x3800 marking the 0F 38 map.
    static const uint16_t MAXIMA[2][3] = { { 0x383C, 0x00EE, 0x383D }, { 0x00DE, 0x383E, 0x383F } };
    static const uint16_t MINIMA[2][3] = { { 0x3838, 0x00EA, 0x3839 }, { 0x00DA, 0x383A, 0x383B } };

    switch (opcode)
    {
        case 0x10U: // ADD, SUB
        case 0x03U: // logical
            if (0x10U == opcode && 3U == size && false == is_quad && false == is_scalar)
            {
                return false;
            }
            break;
        case 0x11U: // CMTST, CMEQ
            if (3U == size && ((false == is_quad && false == is_scalar) || false == sse41))
            {
                return false;
            }
            break;
        case 0x06U: // CMGT, CMHI
        case 0x07U: // CMGE, CMHS
            if (3U == size
                && ((false == is_quad && false == is_scalar)
                    || false == has_feature(compiler, BAL_TIER1_FEATURE_SSE42)))
            {
                return false;
            }
            break;
        case 0x0CU: // SMAX, UMAX
        case 0x0DU: // SMIN, UMIN
        {
            if (3U == size)
            {
                return false;
            }

            const uint16_t entry = 0x0CU == opcode ? MAXIMA[u][size] : MINIMA[u][size];

            if (0U != (entry & 0xFF00U) && false == sse41)
            {
                return false;
            }
            break;
        }
        case 0x13U: // MUL (PMUL with U = 1 is left to the helper)
            if (1U == u || 0U == size || 3U == size || (2U == size && false == sse41))
            {
                return false;
            }
            break;
        default:
            return false;
    }

    const uint32_t                    rd        = instruction & 0x1FU;
    const uint32_t                    rn        = (instruction >> 5U) & 0x1FU;
    const uint32_t                    rm        = (instruction >> 16U) & 0x1FU;
    bal_x86_assembler_t *BAL_RESTRICT assembler = &compiler->assembler;
    uint8_t                           result    = XMM0;

    // Only the cpu structure and xmm registers are involved, so the register cache stays.
    bal_sliding_window_flush_all(&compiler->window);
    emit_load_vector(assembler, XMM0, rn, is_quad);
    emit_load_vector(assembler, XMM1, rm, is_quad);

    switch (opcode)
    {
        case 0x10U: {
            static const uint8_t ADDS[4] = { 0xFC, 0xFD, 0xFE, 0xD4 };
            static const uint8_t SUBS[4] = { 0xF8, 0xF9, 0xFA, 0xFB };
            emit_sse(
                assembler, 0x66, false, 0U == u ? ADDS[size] : SUBS[size], XMM0, XMM1, "padd/psub");
            break;
        }
        case 0x03U:
            switch ((u << 2U) | size)
            {
                case 0U: // AND
                    emit_sse(assembler, 0x66, false, 0xDB, XMM0, XMM1, "pand");
                    break;
                case 1U: // BIC: PANDN computes ~m & n
                    emit_sse(assembler, 0x66, false, 0xDF, XMM1, XMM0, "pandn");
                    result = XMM1;
                    break;
                case 2U: // ORR
                    emit_sse(assembler, 0x66, false, 0xEB, XMM0, XMM1, "por");
                    break;
                case 3U: // ORN: n | ~m
                    emit_not_vector(assembler, XMM1, XMM2);
                    emit_sse(assembler, 0x66, false, 0xEB, XMM0, XMM1, "por");
                    break;
                case 4U: // EOR
                    emit_sse(assembler, 0x66, false, 0xEF, XMM0, XMM1, "pxor");
                    break;
                case 5U: // BSL: ((n ^ m) & d) ^ m
                    emit_load_vector(assembler, XMM2, rd, is_quad);
                    emit_sse(assembler, 0x66, false, 0xEF, XMM0, XMM1, "pxor");
                    emit_sse(assembler, 0x66, false, 0xDB, XMM0, XMM2, "pand");
                    emit_sse(assembler, 0x66, false, 0xEF, XMM0, XMM1, "pxor");
                    break;
                case 6U: // BIT: ((n ^ d) & m) ^ d
                    emit_load_vector(assembler, XMM2, rd, is_quad);
                    emit_sse(assembler, 0x66, false, 0xEF, XMM0, XMM2, "pxor");
                    emit_sse(assembler, 0x66, false, 0xDB, XMM0, XMM1, "pand");
                    emit_sse(assembler, 0x66, false, 0xEF, XMM0, XMM2, "pxor");
                    break;
                default: // BIF: ((n ^ d) & ~m) ^ d
                    emit_load_vector(assembler, XMM2, rd, is_quad);
                    emit_sse(assembler, 0x66, false, 0xEF, XMM0, XMM2, "pxor");
                    emit_sse(assembler, 0x66, false, 0xDF, XMM1, XMM0, "pandn");
                    emit_sse(assembler, 0x66, false, 0xEF, XMM1, XMM2, "pxor");
                    result = XMM1;
                    break;
            }
            break;
        case 0x11U:
            if (1U == u) // CMEQ
            {
                emit_lanes_equal(assembler, size, XMM0, XMM1);
                break;
            }

            // CMTST: all ones where n & m is not zero.
            emit_sse(assembler, 0x66, false, 0xDB, XMM0, XMM1, "pand");
            emit_zero_vector(assembler, XMM1);
            emit_lanes_equal(assembler, size, XMM0, XMM1);
            emit_not_vector(assembler, XMM0, XMM1);
            break;
        case 0x06U: // CMGT n > m; CMHI compares with the sign bits flipped.
        case 0x07U: // CMGE n >= m is NOT (m > n); CMHS likewise.
            if (1U == u)
            {
                emit_broadcast_pattern(assembler, XMM2, lane_sign_pattern(size));
                emit_sse(assembler, 0x66, false, 0xEF, XMM0, XMM2, "pxor");
                emit_sse(assembler, 0x66, false, 0xEF, XMM1, XMM2, "pxor");
            }

            if (0x06U == opcode)
            {
                emit_lanes_greater(assembler, size, XMM0, XMM1);
            }
            else
            {
                emit_lanes_greater(assembler, size, XMM1, XMM0);
                emit_not_vector(assembler, XMM1, XMM2);
                result = XMM1;
            }
            break;
        case 0x0CU:
        case 0x0DU: {
            const uint16_t entry = 0x0CU == opcode ? MAXIMA[u][size] : MINIMA[u][size];

            if (0U != (entry & 0xFF00U))
            {
                emit_sse38(assembler, (uint8_t)entry, XMM0, XMM1, "pmax/pmin");
            }
            else
            {
                emit_sse(assembler, 0x66, false, (uint8_t)entry, XMM0, XMM1, "pmax/pmin");
            }
            break;
        }
        default: // MUL
            if (1U == size)
            {
                emit_sse(assembler, 0x66, false, 0xD5, XMM0, XMM1, "pmullw");
            }
            else
            {
                emit_sse38(assembler, 0x40, XMM0, XMM1, "pmulld");
            }
            break;
    }

    emit_store_vector(assembler, result, rd, is_quad);
    return true;
}

/// Advanced SIMD two-register miscellaneous without floating-point arithmetic: NOT, NEG, ABS, the
/// comparisons with zero (CMEQ, CMGT, CMGE, CMLT, CMLE), REV64 and REV32 on halfwords and words,
/// XTN, and FABS and FNEG, which only change sign bits; and the scalar NEG, ABS and comparisons
/// with zero, which work on doublewords.
static bool
emit_fast_vector_misc(bal_tier1_compiler_t *BAL_RESTRICT compiler, const uint32_t instruction)
{
    // 0 Q U 01110 size 10000 opcode 10 Rn Rd, or the scalar 01 U 11110 size 10000 opcode 10 Rn Rd.
    const bool is_scalar = 0x5E200800U == (instruction & 0xDF3E0C00U);

    if (0x0E200800U != (instruction & 0x9F3E0C00U) && false == is_scalar)
    {
        return false;
    }

    const bool     is_quad = false == is_scalar && 1U == ((instruction >> 30U) & 1U);
    const uint32_t u       = (instruction >> 29U) & 1U;
    const uint32_t size    = (instruction >> 22U) & 0x3U;
    const uint32_t opcode  = (instruction >> 12U) & 0x1FU;
    const uint32_t key     = (u << 5U) | opcode;
    const bool     sse41   = has_feature(compiler, BAL_TIER1_FEATURE_SSE41);
    const bool     is_wide = 3U == size;

    // Doubleword lanes exist in full vectors and scalars.
    const bool has_wide_lanes = true == is_quad || true == is_scalar;

    if (true == is_scalar
        && (false == is_wide
            || (key != 0x2BU && key != 0x0BU && key != 0x09U && key != 0x08U && key != 0x28U
                && key != 0x0AU && key != 0x29U)))
    {
        return false;
    }

    switch (key)
    {
        case 0x25U: // NOT
            if (size != 0U)
            {
                return false;
            }
            break;
        case 0x2BU: // NEG
            if (true == is_wide && false == has_wide_lanes)
            {
                return false;
            }
            break;
        case 0x0BU: // ABS
            if (false == sse41
                || (true == is_wide
                    && (false == has_wide_lanes
                        || false == has_feature(compiler, BAL_TIER1_FEATURE_SSE42))))
            {
                return false;
            }
            break;
        case 0x09U: // CMEQ #0
            if (true == is_wide && (false == has_wide_lanes || false == sse41))
            {
                return false;
            }
            break;
        case 0x08U: // CMGT #0
        case 0x28U: // CMGE #0
        case 0x0AU: // CMLT #0
        case 0x29U: // CMLE #0
            if (true == is_wide
                && (false == has_wide_lanes
                    || false == has_feature(compiler, BAL_TIER1_FEATURE_SSE42)))
            {
                return false;
            }
            break;
        case 0x00U: // REV64
            if (size != 1U && size != 2U)
            {
                return false;
            }
            break;
        case 0x20U: // REV32
            if (size != 1U)
            {
                return false;
            }
            break;
        case 0x12U: // XTN (not XTN2)
            if (true == is_quad || true == is_wide || (1U == size && false == sse41))
            {
                return false;
            }
            break;
        case 0x0FU: // FABS
        case 0x2FU: // FNEG
            if (size < 2U || (true == is_wide && false == is_quad))
            {
                return false;
            }
            break;
        default:
            return false;
    }

    const uint32_t                    rd        = instruction & 0x1FU;
    const uint32_t                    rn        = (instruction >> 5U) & 0x1FU;
    bal_x86_assembler_t *BAL_RESTRICT assembler = &compiler->assembler;
    uint8_t                           result    = XMM0;
    bal_sliding_window_flush_all(&compiler->window);

    // XTN narrows a whole 128-bit register into a 64-bit one.
    emit_load_vector(assembler, XMM0, rn, true == is_quad || 0x12U == key);

    switch (key)
    {
        case 0x25U:
            emit_not_vector(assembler, XMM0, XMM1);
            break;
        case 0x2BU: {
            static const uint8_t SUBS[4] = { 0xF8, 0xF9, 0xFA, 0xFB };
            emit_zero_vector(assembler, XMM1);
            emit_sse(assembler, 0x66, false, SUBS[size], XMM1, XMM0, "psub");
            result = XMM1;
            break;
        }
        case 0x0BU:
            if (true == is_wide)
            {
                // (x ^ m) - m, with m all ones in the lanes below zero.
                emit_zero_vector(assembler, XMM1);
                emit_lanes_greater(assembler, 3U, XMM1, XMM0);
                emit_sse(assembler, 0x66, false, 0xEF, XMM0, XMM1, "pxor");
                emit_sse(assembler, 0x66, false, 0xFB, XMM0, XMM1, "psubq");
                break;
            }

            emit_sse38(assembler, (uint8_t)(0x1CU + size), XMM0, XMM0, "pabs");
            break;
        case 0x09U: // x == 0
            emit_zero_vector(assembler, XMM1);
            emit_lanes_equal(assembler, size, XMM0, XMM1);
            break;
        case 0x08U: // x > 0
            emit_zero_vector(assembler, XMM1);
            emit_lanes_greater(assembler, size, XMM0, XMM1);
            break;
        case 0x0AU: // 0 > x
            emit_zero_vector(assembler, XMM1);
            emit_lanes_greater(assembler, size, XMM1, XMM0);
            result = XMM1;
            break;
        case 0x28U: // x >= 0 is NOT (0 > x)
            emit_zero_vector(assembler, XMM1);
            emit_lanes_greater(assembler, size, XMM1, XMM0);
            emit_not_vector(assembler, XMM1, XMM2);
            result = XMM1;
            break;
        case 0x29U: // x <= 0 is NOT (x > 0)
            emit_zero_vector(assembler, XMM1);
            emit_lanes_greater(assembler, size, XMM0, XMM1);
            emit_not_vector(assembler, XMM0, XMM2);
            break;
        case 0x00U:
            if (2U == size)
            {
                emit_sse_immediate(assembler, 0x66, 0x70, XMM0, XMM0, 0xB1, "pshufd");
            }
            else
            {
                emit_sse_immediate(assembler, 0xF2, 0x70, XMM0, XMM0, 0x1B, "pshuflw");
                emit_sse_immediate(assembler, 0xF3, 0x70, XMM0, XMM0, 0x1B, "pshufhw");
            }
            break;
        case 0x20U:
            emit_sse_immediate(assembler, 0xF2, 0x70, XMM0, XMM0, 0xB1, "pshuflw");
            emit_sse_immediate(assembler, 0xF3, 0x70, XMM0, XMM0, 0xB1, "pshufhw");
            break;
        case 0x12U:
            if (2U == size)
            {
                emit_sse_immediate(assembler, 0x66, 0x70, XMM0, XMM0, 0x08, "pshufd");
                break;
            }

            // Keep the low half of each lane, then pack without saturating anything.
            emit_all_ones(assembler, XMM1);

            if (0U == size)
            {
                emit_sse_immediate(assembler, 0x66, 0x71, 2U, XMM1, 8U, "psrlw");
                emit_sse(assembler, 0x66, false, 0xDB, XMM0, XMM1, "pand");
                emit_sse(assembler, 0x66, false, 0x67, XMM0, XMM0, "packuswb");
            }
            else
            {
                emit_sse_immediate(assembler, 0x66, 0x72, 2U, XMM1, 16U, "psrld");
                emit_sse(assembler, 0x66, false, 0xDB, XMM0, XMM1, "pand");
                emit_sse38(assembler, 0x2B, XMM0, XMM0, "packusdw");
            }
            break;
        default: // FABS clears and FNEG flips the sign bits.
        {
            const uint8_t shift_opcode = true == is_wide ? 0x73 : 0x72;
            emit_all_ones(assembler, XMM1);

            if (0x0FU == key)
            {
                emit_sse_immediate(assembler, 0x66, shift_opcode, 2U, XMM1, 1U, "psrl");
                emit_sse(assembler, 0x66, false, 0xDB, XMM0, XMM1, "pand");
            }
            else
            {
                emit_sse_immediate(
                    assembler, 0x66, shift_opcode, 6U, XMM1, true == is_wide ? 63U : 31U, "psll");
                emit_sse(assembler, 0x66, false, 0xEF, XMM0, XMM1, "pxor");
            }
            break;
        }
    }

    emit_store_vector(assembler, result, rd, is_quad);
    return true;
}

/// Advanced SIMD two-register miscellaneous floating point, under FPCR = 0 like the other fast
/// paths: FRINTN, FRINTM, FRINTP, FRINTZ, FRINTX and FRINTI (ROUNDPS/ROUNDPD), FCVTZS and SCVTF on
/// words, the comparisons with zero, and FSQRT. Their flags match the AArch64 ones under
/// GUEST_MXCSR; a conversion out of range and a NaN square root go to the slow path.
static bool
emit_fast_float_misc(bal_tier1_compiler_t *BAL_RESTRICT compiler, const uint32_t instruction)
{
    // 0 Q U 01110 a sz 10000 opcode 10 Rn Rd.
    if (0x0E200800U != (instruction & 0x9F3E0C00U))
    {
        return false;
    }

    const bool     is_quad   = 1U == ((instruction >> 30U) & 1U);
    const bool     is_double = 1U == ((instruction >> 22U) & 1U);
    const uint32_t key = (((instruction >> 29U) & 1U) << 6U) | (((instruction >> 23U) & 1U) << 5U)
                         | ((instruction >> 12U) & 0x1FU);
    uint8_t rounding = 0xFFU;

    switch (key)
    {
        case 0x18U: // FRINTN
            rounding = 0x08;
            break;
        case 0x19U: // FRINTM
            rounding = 0x09;
            break;
        case 0x38U: // FRINTP
            rounding = 0x0A;
            break;
        case 0x39U: // FRINTZ
            rounding = 0x0B;
            break;
        case 0x59U: // FRINTX: the current mode, raising Inexact
            rounding = 0x04;
            break;
        case 0x79U: // FRINTI: the current mode
            rounding = 0x0C;
            break;
        case 0x3BU: // FCVTZS
        case 0x1DU: // SCVTF
            if (true == is_double)
            {
                return false;
            }
            break;
        case 0x2CU: // FCMGT #0
        case 0x2DU: // FCMEQ #0
        case 0x2EU: // FCMLT #0
        case 0x6CU: // FCMGE #0
        case 0x6DU: // FCMLE #0
        case 0x7FU: // FSQRT
            break;
        default:
            return false;
    }

    if ((true == is_double && false == is_quad)
        || (rounding != 0xFFU && false == has_feature(compiler, BAL_TIER1_FEATURE_SSE41)))
    {
        return false;
    }

    const uint32_t                    rd        = instruction & 0x1FU;
    const uint32_t                    rn        = (instruction >> 5U) & 0x1FU;
    const uint8_t                     packed    = true == is_double ? 0x66 : 0U;
    bal_x86_assembler_t *BAL_RESTRICT assembler = &compiler->assembler;
    uint8_t                           result    = XMM0;
    size_t                            slow_jumps[3];
    uint32_t                          count = 1U;

    slow_jumps[0] = begin_float_sequence(compiler);
    emit_load_vector(assembler, XMM0, rn, is_quad);

    if (rounding != 0xFFU)
    {
        emit_sse3a(assembler, true == is_double ? 0x09 : 0x08, XMM0, XMM0, rounding, "roundp");
    }
    else if (0x3BU == key)
    {
        // CVTTPS2DQ gives 0x80000000 for NaNs and values out of range, which AArch64 saturates.
        emit_sse(assembler, 0xF3, false, 0x5B, XMM0, XMM0, "cvttps2dq");
        emit_all_ones(assembler, XMM1);
        emit_sse_immediate(assembler, 0x66, 0x72, 6U, XMM1, 31U, "pslld");
        emit_sse(assembler, 0x66, false, 0x76, XMM1, XMM0, "pcmpeqd");
        emit_sse(assembler, 0U, false, 0x50, (uint8_t)SCRATCH_A, XMM1, "movmskps");
        bal_x86_emit_test_r32_r32(assembler, SCRATCH_A, SCRATCH_A);
        slow_jumps[count++] = emit_forward_jump(assembler, true, BAL_X86_COND_NE);
    }
    else if (0x1DU == key)
    {
        emit_sse(assembler, 0U, false, 0x5B, XMM0, XMM0, "cvtdq2ps");
    }
    else if (0x7FU == key)
    {
        emit_sse(assembler, packed, false, 0x51, XMM0, XMM0, "sqrtp");
        count += emit_vector_checks(assembler, is_double, &slow_jumps[1]);
    }
    else
    {
        // CMPP with EQ_OQ signals Invalid only for signalling NaNs, like FCMEQ; LT_OS and LE_OS
        // signal for every NaN, like FCMGT, FCMGE, FCMLT and FCMLE.
        emit_zero_vector(assembler, XMM1);

        switch (key)
        {
            case 0x2CU: // 0 < x
                emit_sse_immediate(assembler, packed, 0xC2, XMM1, XMM0, 1U, "cmpltp");
                result = XMM1;
                break;
            case 0x2DU: // x == 0
                emit_sse_immediate(assembler, packed, 0xC2, XMM0, XMM1, 0U, "cmpeqp");
                break;
            case 0x2EU: // x < 0
                emit_sse_immediate(assembler, packed, 0xC2, XMM0, XMM1, 1U, "cmpltp");
                break;
            case 0x6CU: // 0 <= x
                emit_sse_immediate(assembler, packed, 0xC2, XMM1, XMM0, 2U, "cmplep");
                result = XMM1;
                break;
            default: // x <= 0
                emit_sse_immediate(assembler, packed, 0xC2, XMM0, XMM1, 2U, "cmplep");
                break;
        }
    }

    emit_store_vector(assembler, result, rd, is_quad);
    emit_slow_path(
        compiler, instruction, (uint64_t)(uintptr_t)&bal_execute_simd, slow_jumps, count);
    return true;
}

/// FCMEQ, FCMGE and FCMGT (vector), under FPCR = 0. CMPP signals Invalid as these do (see
/// emit_fast_float_misc()) and the results are masks, so only the FPCR check can fail.
static bool
emit_fast_vector_float_compare(bal_tier1_compiler_t *BAL_RESTRICT compiler,
                               const uint32_t                     instruction)
{
    // 0 Q U 01110 a sz 1 Rm 11100 1 Rn Rd.
    if (0x0E20E400U != (instruction & 0x9F20FC00U))
    {
        return false;
    }

    const bool     is_quad   = 1U == ((instruction >> 30U) & 1U);
    const bool     is_double = 1U == ((instruction >> 22U) & 1U);
    const uint32_t key       = (((instruction >> 29U) & 1U) << 1U) | ((instruction >> 23U) & 1U);

    // FCMEQ (U = 0, a = 0), FCMGE (U = 1, a = 0), FCMGT (U = 1, a = 1).
    if (1U == key || (true == is_double && false == is_quad))
    {
        return false;
    }

    const uint32_t                    rd        = instruction & 0x1FU;
    const uint32_t                    rn        = (instruction >> 5U) & 0x1FU;
    const uint32_t                    rm        = (instruction >> 16U) & 0x1FU;
    const uint8_t                     packed    = true == is_double ? 0x66 : 0U;
    bal_x86_assembler_t *BAL_RESTRICT assembler = &compiler->assembler;
    size_t                            slow_jumps[1];

    slow_jumps[0] = begin_float_sequence(compiler);
    emit_load_vector(assembler, XMM0, rn, is_quad);
    emit_load_vector(assembler, XMM1, rm, is_quad);

    if (0U == key)
    {
        emit_sse_immediate(assembler, packed, 0xC2, XMM0, XMM1, 0U, "cmpeqp");
        emit_store_vector(assembler, XMM0, rd, is_quad);
    }
    else
    {
        // n >= m is m <= n, and n > m is m < n.
        emit_sse_immediate(assembler, packed, 0xC2, XMM1, XMM0, 2U == key ? 2U : 1U, "cmpp");
        emit_store_vector(assembler, XMM1, rd, is_quad);
    }

    emit_slow_path(compiler, instruction, (uint64_t)(uintptr_t)&bal_execute_simd, slow_jumps, 1U);
    return true;
}

/// Advanced SIMD permute: ZIP1 and ZIP2 for every element size, and UZP1, UZP2, TRN1 and TRN2 on
/// words and doublewords.
static bool
emit_fast_permute(bal_tier1_compiler_t *BAL_RESTRICT compiler, const uint32_t instruction)
{
    // 0 Q 0 01110 size 0 Rm 0 opcode 10 Rn Rd.
    if (0x0E000800U != (instruction & 0xBF208C00U))
    {
        return false;
    }

    const bool     is_quad = 1U == ((instruction >> 30U) & 1U);
    const uint32_t size    = (instruction >> 22U) & 0x3U;
    const uint32_t opcode  = (instruction >> 12U) & 0x7U;
    const bool     is_zip  = 0x3U == (opcode & 0x3U);
    const bool     is_two  = 0U != (opcode & 0x4U);

    if (0U == (opcode & 0x3U) || (3U == size && false == is_quad) || (false == is_zip && size < 2U))
    {
        return false;
    }

    static const uint8_t              LOW[4]    = { 0x60, 0x61, 0x62, 0x6C };
    static const uint8_t              HIGH[4]   = { 0x68, 0x69, 0x6A, 0x6D };
    const uint32_t                    rd        = instruction & 0x1FU;
    const uint32_t                    rn        = (instruction >> 5U) & 0x1FU;
    const uint32_t                    rm        = (instruction >> 16U) & 0x1FU;
    bal_x86_assembler_t *BAL_RESTRICT assembler = &compiler->assembler;
    bal_sliding_window_flush_all(&compiler->window);
    emit_load_vector(assembler, XMM0, rn, is_quad);
    emit_load_vector(assembler, XMM1, rm, is_quad);

    // On doublewords all six are interleavings; on 64-bit vectors of words UZP and TRN are too.
    if (true == is_zip || 3U == size || false == is_quad)
    {
        if (true == is_quad)
        {
            emit_sse(assembler,
                     0x66,
                     false,
                     true == is_two ? HIGH[size] : LOW[size],
                     XMM0,
                     XMM1,
                     "punpck");
        }
        else
        {
            // The interleaved low halves; the "2" forms take its upper 64 bits.
            emit_sse(assembler, 0x66, false, LOW[size], XMM0, XMM1, "punpckl");

            if (true == is_two)
            {
                emit_sse_immediate(assembler, 0x66, 0x73, 3U, XMM0, 8U, "psrldq");
            }
        }
    }
    else
    {
        // UZP picks the even (or odd) words of n then m; TRN then interleaves them.
        emit_sse_immediate(assembler, 0U, 0xC6, XMM0, XMM1, true == is_two ? 0xDD : 0x88, "shufps");

        if (2U == (opcode & 0x3U))
        {
            emit_sse_immediate(assembler, 0x66, 0x70, XMM0, XMM0, 0xD8, "pshufd");
        }
    }

    emit_store_vector(assembler, XMM0, rd, is_quad);
    return true;
}

/// EXT: the bytes of m:n from byte imm4 on.
static bool
emit_fast_extract(bal_tier1_compiler_t *BAL_RESTRICT compiler, const uint32_t instruction)
{
    // 0 Q 101110 00 0 Rm 0 imm4 0 Rn Rd.
    if (0x2E000000U != (instruction & 0xBFE08400U))
    {
        return false;
    }

    const bool     is_quad = 1U == ((instruction >> 30U) & 1U);
    const uint32_t imm4    = (instruction >> 11U) & 0xFU;

    if ((false == is_quad && imm4 >= 8U)
        || (true == is_quad && false == has_feature(compiler, BAL_TIER1_FEATURE_SSE41)))
    {
        return false;
    }

    const uint32_t                    rd        = instruction & 0x1FU;
    const uint32_t                    rn        = (instruction >> 5U) & 0x1FU;
    const uint32_t                    rm        = (instruction >> 16U) & 0x1FU;
    bal_x86_assembler_t *BAL_RESTRICT assembler = &compiler->assembler;
    bal_sliding_window_flush_all(&compiler->window);
    emit_load_vector(assembler, XMM0, rn, is_quad);
    emit_load_vector(assembler, XMM1, rm, is_quad);

    if (true == is_quad)
    {
        emit_sse3a(assembler, 0x0F, XMM1, XMM0, (uint8_t)imm4, "palignr");
        emit_store_vector(assembler, XMM1, rd, true);
        return true;
    }

    emit_sse(assembler, 0x66, false, 0x6C, XMM0, XMM1, "punpcklqdq");

    if (imm4 != 0U)
    {
        emit_sse_immediate(assembler, 0x66, 0x73, 3U, XMM0, (uint8_t)imm4, "psrldq");
    }

    emit_store_vector(assembler, XMM0, rd, false);
    return true;
}

/// TBL with a one-register table: PSHUFB, with every index above 15 made to select zero.
static bool
emit_fast_table(bal_tier1_compiler_t *BAL_RESTRICT compiler, const uint32_t instruction)
{
    // 0 Q 001110 000 Rm 0 00 0 00 Rn Rd.
    if (0x0E000000U != (instruction & 0xBFE0FC00U)
        || false == has_feature(compiler, BAL_TIER1_FEATURE_SSE41))
    {
        return false;
    }

    const bool                        is_quad   = 1U == ((instruction >> 30U) & 1U);
    const uint32_t                    rd        = instruction & 0x1FU;
    const uint32_t                    rn        = (instruction >> 5U) & 0x1FU;
    const uint32_t                    rm        = (instruction >> 16U) & 0x1FU;
    bal_x86_assembler_t *BAL_RESTRICT assembler = &compiler->assembler;
    bal_sliding_window_flush_all(&compiler->window);
    emit_load_vector(assembler, XMM0, rn, true);
    emit_load_vector(assembler, XMM1, rm, is_quad);

    // PSHUFB zeroes a byte whose index has bit 7 set, which covers 128-255; 16-127 get it set.
    emit_broadcast_pattern(assembler, XMM2, 0x0F0F0F0F0F0F0F0FULL);
    emit_sse(assembler, 0U, false, 0x28, XMM3, XMM1, "movaps");
    emit_sse(assembler, 0x66, false, 0x64, XMM3, XMM2, "pcmpgtb");
    emit_sse(assembler, 0x66, false, 0xEB, XMM1, XMM3, "por");
    emit_sse38(assembler, 0x00, XMM0, XMM1, "pshufb");
    emit_store_vector(assembler, XMM0, rd, is_quad);
    return true;
}

/// Advanced SIMD modified immediate: MOVI, MVNI, FMOV, ORR and BIC. The interpreter evaluates
/// the immediate at translation time, so the code only stores, ORs or ANDs constants.
static bool
emit_fast_modified_immediate(bal_tier1_compiler_t *BAL_RESTRICT compiler,
                             const uint32_t                     instruction)
{
    // 0 Q op 0111100000 a b c cmode o2 1 d e f g h Rd.
    if (0x0F000400U != (instruction & 0x9FF80400U) || false == bal_is_simd_supported(instruction))
    {
        return false;
    }

    const bool     is_quad = 1U == ((instruction >> 30U) & 1U);
    const uint32_t op      = (instruction >> 29U) & 1U;
    const uint32_t cmode   = (instruction >> 12U) & 0xFU;
    const uint32_t rd      = instruction & 0x1FU;

    // ORR and BIC (immediate): cmode 0xx1 or 10x1. The rest set the register outright.
    const bool is_logical = 1U == (cmode & 1U) && cmode < 0xCU;

    // Evaluate on a zeroed register (MOVI, MVNI, FMOV and ORR give the immediate) or, for BIC, on
    // one of all ones (giving the mask to AND with).
    bal_cpu_t scratch;
    (void)memset(&scratch, 0, sizeof(scratch));

    if (true == is_logical && 1U == op)
    {
        scratch.v[rd][0] = ~0ULL;
        scratch.v[rd][1] = ~0ULL;
    }

    (void)bal_execute_simd(&scratch, instruction);
    bal_x86_assembler_t *BAL_RESTRICT assembler = &compiler->assembler;
    bal_sliding_window_flush_all(&compiler->window);

    for (uint32_t half = 0U; half < 2U; ++half)
    {
        const int32_t  offset = (int32_t)offsetof(bal_cpu_t, v[rd][half]);
        const uint64_t value  = scratch.v[rd][half];

        if (false == is_logical || (1U == half && false == is_quad))
        {
            // A 64-bit operation also clears the upper half.
            bal_x86_emit_mov_r64_imm64(
                assembler, SCRATCH_A, 1U == half && false == is_quad ? 0U : value);
            bal_x86_emit_store_r64_rbp_offset(assembler, SCRATCH_A, offset);
            continue;
        }

        bal_x86_emit_load_r64_rbp_offset(assembler, SCRATCH_A, offset);
        bal_x86_emit_mov_r64_imm64(assembler, SCRATCH_B, value);

        if (0U == op)
        {
            bal_x86_emit_or_r64_r64(assembler, SCRATCH_A, SCRATCH_B);
        }
        else
        {
            bal_x86_emit_and_r64_r64(assembler, SCRATCH_A, SCRATCH_B);
        }

        bal_x86_emit_store_r64_rbp_offset(assembler, SCRATCH_A, offset);
    }

    return true;
}

/// Points SCRATCH_B at byte `byte` of SIMD&FP register `index` in the cpu structure.
static void
emit_vector_element_address(bal_x86_assembler_t *BAL_RESTRICT assembler,
                            const uint32_t                    index,
                            const uint32_t                    byte)
{
    bal_x86_emit_mov_r64_r64(assembler, SCRATCH_B, BAL_X86_RBP);
    bal_x86_emit_add_r64_imm32(
        assembler, SCRATCH_B, (int32_t)(offsetof(bal_cpu_t, v[index][0]) + byte));
}

/// Moves between general purpose registers and vector elements: UMOV, SMOV, INS (general and
/// element) and DUP (general).
static bool
emit_fast_element_move(bal_tier1_compiler_t *BAL_RESTRICT compiler, const uint32_t instruction)
{
    // 0 Q op 01110000 imm5 0 imm4 1 Rn Rd.
    if (0x0E000400U != (instruction & 0x9FE08400U))
    {
        return false;
    }

    const bool     is_quad = 1U == ((instruction >> 30U) & 1U);
    const uint32_t op      = (instruction >> 29U) & 1U;
    const uint32_t imm5    = (instruction >> 16U) & 0x1FU;
    const uint32_t imm4    = (instruction >> 11U) & 0xFU;
    const uint32_t rd      = instruction & 0x1FU;
    const uint32_t rn      = (instruction >> 5U) & 0x1FU;

    if (0U == (imm5 & 0xFU))
    {
        return false;
    }

    const uint32_t size  = (uint32_t)__builtin_ctz(imm5);
    const uint32_t index = imm5 >> (size + 1U);
    const uint32_t bytes = 1U << size;
    enum
    {
        MOVE_UMOV,
        MOVE_SMOV,
        MOVE_INS_GENERAL,
        MOVE_INS_ELEMENT,
        MOVE_DUP_GENERAL,
    } kind
        = MOVE_UMOV;

    if (1U == op)
    {
        if (false == is_quad)
        {
            return false;
        }

        kind = MOVE_INS_ELEMENT;
    }
    else
    {
        switch (imm4)
        {
            case 0x7U: // UMOV: W from bytes to words, X from doublewords
                if ((true == is_quad) != (3U == size))
                {
                    return false;
                }
                kind = MOVE_UMOV;
                break;
            case 0x5U: // SMOV: W from bytes and halfwords, X from bytes to words
                if (size > (true == is_quad ? 2U : 1U))
                {
                    return false;
                }
                kind = MOVE_SMOV;
                break;
            case 0x3U: // INS (general)
                if (false == is_quad)
                {
                    return false;
                }
                kind = MOVE_INS_GENERAL;
                break;
            case 0x1U: // DUP (general)
                if (3U == size && false == is_quad)
                {
                    return false;
                }
                kind = MOVE_DUP_GENERAL;
                break;
            default:
                return false;
        }
    }

    bal_x86_assembler_t *BAL_RESTRICT assembler = &compiler->assembler;

    if (MOVE_UMOV == kind || MOVE_SMOV == kind)
    {
        if (31U == rd)
        {
            return true;
        }

        const bool               skip_load = true;
        const bal_x86_register_t x86_rd = allocate_x86_register(compiler, (uint8_t)rd, skip_load);
        bal_sliding_window_flush_all(&compiler->window);
        emit_vector_element_address(assembler, rn, index * bytes);

        if (MOVE_UMOV == kind)
        {
            switch (size)
            {
                case 0U:
                    bal_x86_emit_load_r32_mem8(assembler, x86_rd, SCRATCH_B);
                    break;
                case 1U:
                    bal_x86_emit_load_r32_mem16(assembler, x86_rd, SCRATCH_B);
                    break;
                case 2U:
                    bal_x86_emit_load_r32_mem32(assembler, x86_rd, SCRATCH_B);
                    break;
                default:
                    bal_x86_emit_load_r64_mem64(assembler, x86_rd, SCRATCH_B);
                    break;
            }
        }
        else if (true == is_quad)
        {
            switch (size)
            {
                case 0U:
                    bal_x86_emit_load_sign_r64_mem8(assembler, x86_rd, SCRATCH_B);
                    break;
                case 1U:
                    bal_x86_emit_load_sign_r64_mem16(assembler, x86_rd, SCRATCH_B);
                    break;
                default:
                    bal_x86_emit_load_sign_r64_mem32(assembler, x86_rd, SCRATCH_B);
                    break;
            }
        }
        else if (0U == size)
        {
            bal_x86_emit_load_sign_r32_mem8(assembler, x86_rd, SCRATCH_B);
        }
        else
        {
            bal_x86_emit_load_sign_r32_mem16(assembler, x86_rd, SCRATCH_B);
        }

        compiler->is_dirty |= 1U << rd;
        return true;
    }

    if (MOVE_INS_ELEMENT == kind)
    {
        const uint32_t source = (imm4 >> size) * bytes;
        bal_sliding_window_flush_all(&compiler->window);
        emit_vector_element_address(assembler, rn, source);

        switch (size)
        {
            case 0U:
                bal_x86_emit_load_r32_mem8(assembler, SCRATCH_A, SCRATCH_B);
                break;
            case 1U:
                bal_x86_emit_load_r32_mem16(assembler, SCRATCH_A, SCRATCH_B);
                break;
            case 2U:
                bal_x86_emit_load_r32_mem32(assembler, SCRATCH_A, SCRATCH_B);
                break;
            default:
                bal_x86_emit_load_r64_mem64(assembler, SCRATCH_A, SCRATCH_B);
                break;
        }

        emit_vector_element_address(assembler, rd, index * bytes);
    }
    else
    {
        // INS and DUP (general) take Wn or Xn, which reads as zero for register 31.
        bal_x86_register_t source = SCRATCH_A;

        if (31U == rn)
        {
            bal_sliding_window_flush_all(&compiler->window);
            bal_x86_emit_xor_r32_r32(assembler, SCRATCH_A, SCRATCH_A);
        }
        else
        {
            const bool skip_load = false;
            source               = allocate_x86_register(compiler, (uint8_t)rn, skip_load);
            bal_sliding_window_flush_all(&compiler->window);
        }

        if (MOVE_DUP_GENERAL == kind)
        {
            emit_sse(assembler, 0x66, true, 0x6E, XMM0, (uint8_t)source, "movq");

            switch (size)
            {
                case 0U:
                    emit_sse(assembler, 0x66, false, 0x60, XMM0, XMM0, "punpcklbw");
                    emit_sse_immediate(assembler, 0xF2, 0x70, XMM0, XMM0, 0U, "pshuflw");
                    emit_sse(assembler, 0x66, false, 0x6C, XMM0, XMM0, "punpcklqdq");
                    break;
                case 1U:
                    emit_sse_immediate(assembler, 0xF2, 0x70, XMM0, XMM0, 0U, "pshuflw");
                    emit_sse(assembler, 0x66, false, 0x6C, XMM0, XMM0, "punpcklqdq");
                    break;
                case 2U:
                    emit_sse_immediate(assembler, 0x66, 0x70, XMM0, XMM0, 0U, "pshufd");
                    break;
                default:
                    emit_sse(assembler, 0x66, false, 0x6C, XMM0, XMM0, "punpcklqdq");
                    break;
            }

            emit_store_vector(assembler, XMM0, rd, is_quad);
            return true;
        }

        if (source != SCRATCH_A)
        {
            bal_x86_emit_mov_r64_r64(assembler, SCRATCH_A, source);
        }

        emit_vector_element_address(assembler, rd, index * bytes);
    }

    switch (size)
    {
        case 0U:
            bal_x86_emit_store_mem8_r8(assembler, SCRATCH_B, SCRATCH_A);
            break;
        case 1U:
            bal_x86_emit_store_mem16_r16(assembler, SCRATCH_B, SCRATCH_A);
            break;
        case 2U:
            bal_x86_emit_store_mem32_r32(assembler, SCRATCH_B, SCRATCH_A);
            break;
        default:
            bal_x86_emit_store_mem64_r64(assembler, SCRATCH_B, SCRATCH_A);
            break;
    }

    return true;
}

/// FCVT between single and double precision (scalar). Widening is exact; narrowing rounds as
/// AArch64 does under FPCR = 0, and emit_scalar_checks() sends NaNs and results at the smallest
/// normal magnitude, where the underflow flag can differ, to the slow path.
static bool
emit_fast_convert_precision(bal_tier1_compiler_t *BAL_RESTRICT compiler, const uint32_t instruction)
{
    // FCVT Dd, Sn is 0x1E22C000 and FCVT Sd, Dn 0x1E624000.
    const uint32_t form = instruction & 0xFFFFFC00U;

    if (form != 0x1E22C000U && form != 0x1E624000U)
    {
        return false;
    }

    const bool     is_widening = 0x1E22C000U == form;
    const uint32_t rd          = instruction & 0x1FU;
    const uint32_t rn          = (instruction >> 5U) & 0x1FU;
    size_t         slow_jumps[3];

    slow_jumps[0] = begin_float_sequence(compiler);
    emit_load_scalar(&compiler->assembler, XMM0, rn, false == is_widening);
    emit_sse(&compiler->assembler,
             true == is_widening ? 0xF3 : 0xF2,
             false,
             0x5A,
             XMM0,
             XMM0,
             "cvtss2sd/cvtsd2ss");
    const uint32_t count = 1U + emit_scalar_checks(compiler, is_widening, &slow_jumps[1]);
    emit_store_scalar_result(&compiler->assembler, rd);
    emit_slow_path(
        compiler, instruction, (uint64_t)(uintptr_t)&bal_execute_float, slow_jumps, count);
    return true;
}

/// Computes the maximum or (with `is_minimum`) minimum of xmm0 and xmm1 into xmm0, lane by lane
/// with the packed `prefix` or in lane 0 with a scalar one (`packed` is the matching MOVMSK
/// prefix), as FMAX and FMIN do when no operand is a NaN. CMPUNORD checks that first, quietly, so
/// a quiet NaN raises no flag, and returns the jump to the slow path taken when one is. MAXP and
/// MINP return the second operand when the two are equal, which only matters for zeros of
/// different signs: there the AND (maximum) or OR (minimum) of the two gives AArch64's +0 or -0.
/// Uses xmm2 and xmm3.
static size_t
emit_float_extreme(bal_x86_assembler_t *BAL_RESTRICT assembler,
                   const uint8_t                     prefix,
                   const uint8_t                     packed,
                   const bool                        is_scalar,
                   const bool                        is_minimum)
{
    // Any NaN operand goes to the slow path.
    emit_sse(assembler, 0U, false, 0x28, XMM2, XMM0, "movaps");
    emit_sse_immediate(assembler, prefix, 0xC2, XMM2, XMM1, 3U, "cmpunord");
    emit_sse(assembler, packed, false, 0x50, (uint8_t)SCRATCH_A, XMM2, "movmskp");

    if (true == is_scalar)
    {
        bal_x86_emit_shl_r32_imm8(assembler, SCRATCH_A, 31U);
    }

    bal_x86_emit_test_r32_r32(assembler, SCRATCH_A, SCRATCH_A);
    const size_t jump = emit_forward_jump(assembler, true, BAL_X86_COND_NE);

    // XMM2 = max or min, XMM3 = the lanes where the operands are equal, XMM0 = their AND or OR.
    emit_sse(assembler, 0U, false, 0x28, XMM2, XMM0, "movaps");
    emit_sse(assembler, prefix, false, true == is_minimum ? 0x5D : 0x5F, XMM2, XMM1, "max/min");
    emit_sse(assembler, 0U, false, 0x28, XMM3, XMM0, "movaps");
    emit_sse_immediate(assembler, prefix, 0xC2, XMM3, XMM1, 0U, "cmpeq");
    emit_sse(assembler, 0U, false, true == is_minimum ? 0x56 : 0x54, XMM0, XMM1, "orps/andps");
    emit_sse(assembler, 0U, false, 0x54, XMM0, XMM3, "andps");
    emit_sse(assembler, 0U, false, 0x55, XMM3, XMM2, "andnps");
    emit_sse(assembler, 0U, false, 0x56, XMM0, XMM3, "orps");
    return jump;
}

/// FMAX, FMIN, FMAXNM and FMINNM, scalar and vector, when no operand is a NaN: then the NM forms
/// equal the others; emit_float_extreme() computes them.
static bool
emit_fast_float_minmax(bal_tier1_compiler_t *BAL_RESTRICT compiler, const uint32_t instruction)
{
    bool is_scalar  = false;
    bool is_double  = false;
    bool is_minimum = false;
    bool is_quad    = true;

    if (0x1E200800U == (instruction & 0xFF200C00U))
    {
        // M 0 S 11110 ftype 1 Rm opcode 10 Rn Rd, opcode 01xx.
        const uint32_t ftype  = (instruction >> 22U) & 0x3U;
        const uint32_t opcode = (instruction >> 12U) & 0xFU;

        if (ftype > 1U || 0x4U != (opcode & 0xCU))
        {
            return false;
        }

        is_scalar  = true;
        is_double  = 1U == ftype;
        is_minimum = 1U == (opcode & 1U);
    }
    else if (0x0E20F400U == (instruction & 0xBF20FC00U)
             || 0x0E20C400U == (instruction & 0xBF20FC00U))
    {
        // 0 Q 0 01110 a sz 1 Rm 11110 1 (FMAX, FMIN) or 11000 1 (FMAXNM, FMINNM) Rn Rd.
        is_double  = 1U == ((instruction >> 22U) & 1U);
        is_minimum = 1U == ((instruction >> 23U) & 1U);
        is_quad    = 1U == ((instruction >> 30U) & 1U);

        if (true == is_double && false == is_quad)
        {
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
    const uint8_t                     scalar    = true == is_double ? 0xF2 : 0xF3;
    const uint8_t                     packed    = true == is_double ? 0x66 : 0U;
    const uint8_t                     prefix    = true == is_scalar ? scalar : packed;
    bal_x86_assembler_t *BAL_RESTRICT assembler = &compiler->assembler;
    size_t                            slow_jumps[2];

    slow_jumps[0] = begin_float_sequence(compiler);

    if (true == is_scalar)
    {
        emit_load_scalar(assembler, XMM0, rn, is_double);
        emit_load_scalar(assembler, XMM1, rm, is_double);
    }
    else
    {
        emit_load_vector(assembler, XMM0, rn, is_quad);
        emit_load_vector(assembler, XMM1, rm, is_quad);
    }

    slow_jumps[1] = emit_float_extreme(assembler, prefix, packed, is_scalar, is_minimum);

    if (true == is_scalar)
    {
        // The upper lanes of a scalar operation hold the AND or OR of zeros, so a 64-bit move
        // gives the zero-extended result.
        emit_sse(assembler, 0x66, true, 0x7E, XMM0, (uint8_t)SCRATCH_A, "movq");

        if (false == is_double)
        {
            bal_x86_emit_mov_r32_r32(assembler, SCRATCH_A, SCRATCH_A);
        }

        emit_store_scalar_result(assembler, rd);
    }
    else
    {
        emit_store_vector(assembler, XMM0, rd, is_quad);
    }

    emit_slow_path(compiler,
                   instruction,
                   true == is_scalar ? (uint64_t)(uintptr_t)&bal_execute_float
                                     : (uint64_t)(uintptr_t)&bal_execute_simd,
                   slow_jumps,
                   2U);
    return true;
}

// -----------------------------------------------------------------------------
// Inline conversions and rounding
// -----------------------------------------------------------------------------

/// The ROUNDSS immediates for the AArch64 rounding modes nearest even, toward plus infinity,
/// toward minus infinity and toward zero, with the precision exception suppressed.
static const uint8_t ROUND_SUPPRESSED[4] = { 0x08, 0x0A, 0x09, 0x0B };

/// Emits a jump to the slow path, and returns it, for a scalar in SIMD&FP register `index` whose
/// biased exponent is `exponent` or more, which includes infinities and NaNs. Uses SCRATCH_A and
/// RAX.
static size_t
emit_magnitude_check(bal_tier1_compiler_t *BAL_RESTRICT compiler,
                     const uint32_t                     index,
                     const bool                         is_double,
                     const uint64_t                     exponent)
{
    bal_x86_assembler_t *BAL_RESTRICT assembler = &compiler->assembler;
    bal_x86_emit_load_r64_rbp_offset(
        assembler, SCRATCH_A, (int32_t)offsetof(bal_cpu_t, v[index][0]));

    // Shifting out the sign (and, for single precision, the 32 bits above the value) leaves the
    // exponent at the top.
    bal_x86_emit_shl_r64_imm8(assembler, SCRATCH_A, true == is_double ? 1U : 33U);
    bal_x86_emit_mov_r64_imm64(assembler, BAL_X86_RAX, exponent << (true == is_double ? 53U : 56U));
    bal_x86_emit_sub_r64_r64(assembler, SCRATCH_A, BAL_X86_RAX);
    return emit_forward_jump(assembler, true, BAL_X86_COND_AE);
}

/// Returns a jump to the slow path for a scalar in `xmm` whose biased exponent is `exponent` or
/// more, as emit_magnitude_check() does for a SIMD&FP register. Uses SCRATCH_A and RAX.
static size_t
emit_register_magnitude_check(bal_x86_assembler_t *BAL_RESTRICT assembler,
                              const uint8_t                     xmm,
                              const bool                        is_double,
                              const uint64_t                    exponent)
{
    emit_sse(assembler, 0x66, is_double, 0x7E, xmm, (uint8_t)SCRATCH_A, "movd/movq");
    bal_x86_emit_shl_r64_imm8(assembler, SCRATCH_A, true == is_double ? 1U : 33U);
    bal_x86_emit_mov_r64_imm64(assembler, BAL_X86_RAX, exponent << (true == is_double ? 53U : 56U));
    bal_x86_emit_sub_r64_r64(assembler, SCRATCH_A, BAL_X86_RAX);
    return emit_forward_jump(assembler, true, BAL_X86_COND_AE);
}

/// Loads `bits` into the low 64 bits of `xmm`, clearing the rest.
static void
emit_load_constant(bal_x86_assembler_t *BAL_RESTRICT assembler,
                   const uint8_t                     xmm,
                   const uint64_t                    bits)
{
    bal_x86_emit_mov_r64_imm64(assembler, SCRATCH_A, bits);
    emit_sse(assembler, 0x66, true, 0x6E, xmm, (uint8_t)SCRATCH_A, "movq");
}

/// Rounds the scalar in `source` to an integer into `destination` with ROUNDSS or ROUNDSD and
/// `immediate`. ROUNDSS keeps the upper lanes of its destination, so a copy of the source first
/// keeps the result from depending on the destination's last writer.
static void
emit_round_scalar(bal_x86_assembler_t *BAL_RESTRICT assembler,
                  const bool                        is_double,
                  const uint8_t                     destination,
                  const uint8_t                     source,
                  const uint8_t                     immediate)
{
    emit_sse(assembler, 0U, false, 0x28, destination, source, "movaps");
    emit_sse3a(
        assembler, true == is_double ? 0x0B : 0x0A, destination, destination, immediate, "rounds");
}

/// Rounds the scalar in xmm0 to an integer with ties away from zero into xmm1, using xmm2 and
/// xmm3: t = trunc(x), plus copysign(1, x) when |x - t| >= 0.5. Every step is exact for finite
/// values and raises nothing; an infinity or NaN raises Invalid in the subtraction.
static void
emit_round_ties_away(bal_x86_assembler_t *BAL_RESTRICT assembler, const bool is_double)
{
    const uint8_t scalar   = true == is_double ? 0xF2 : 0xF3;
    const uint8_t shift_op = true == is_double ? 0x73 : 0x72;

    emit_round_scalar(assembler, is_double, XMM1, XMM0, 0x0B);
    emit_sse(assembler, 0U, false, 0x28, XMM2, XMM0, "movaps");
    emit_sse(assembler, scalar, false, 0x5C, XMM2, XMM1, "subs");
    emit_all_ones(assembler, XMM3);
    emit_sse_immediate(assembler, 0x66, shift_op, 2U, XMM3, 1U, "psrl");
    emit_sse(assembler, 0U, false, 0x54, XMM2, XMM3, "andps");
    emit_load_constant(assembler, XMM3, true == is_double ? 0x3FE0000000000000ULL : 0x3F000000ULL);
    emit_sse_immediate(assembler, scalar, 0xC2, XMM2, XMM3, 5U, "cmpnlts");
    emit_load_constant(assembler, XMM3, true == is_double ? 0x3FF0000000000000ULL : 0x3F800000ULL);
    emit_sse(assembler, 0U, false, 0x54, XMM2, XMM3, "andps");
    emit_all_ones(assembler, XMM3);
    emit_sse_immediate(assembler, 0x66, shift_op, 6U, XMM3, true == is_double ? 63U : 31U, "psll");
    emit_sse(assembler, 0U, false, 0x54, XMM3, XMM0, "andps");
    emit_sse(assembler, 0U, false, 0x56, XMM2, XMM3, "orps");
    emit_sse(assembler, scalar, false, 0x58, XMM1, XMM2, "adds");
}

/// Rounds the lanes of `x` to integers with ties away from zero into `destination`, as
/// emit_round_ties_away() does a scalar, using `scratch` and `other`. Infinities and NaNs raise
/// Invalid in the subtraction.
static void
emit_round_ties_away_packed(bal_x86_assembler_t *BAL_RESTRICT assembler,
                            const bool                        is_double,
                            const uint8_t                     destination,
                            const uint8_t                     x,
                            const uint8_t                     scratch,
                            const uint8_t                     other)
{
    const uint8_t packed   = true == is_double ? 0x66 : 0U;
    const uint8_t shift_op = true == is_double ? 0x73 : 0x72;
    emit_sse3a(assembler, true == is_double ? 0x09 : 0x08, destination, x, 0x0B, "roundp");
    emit_sse(assembler, 0U, false, 0x28, scratch, x, "movaps");
    emit_sse(assembler, packed, false, 0x5C, scratch, destination, "subp");
    emit_all_ones(assembler, other);
    emit_sse_immediate(assembler, 0x66, shift_op, 2U, other, 1U, "psrl");
    emit_sse(assembler, 0U, false, 0x54, scratch, other, "andps");
    emit_broadcast_pattern(
        assembler, other, true == is_double ? 0x3FE0000000000000ULL : 0x3F0000003F000000ULL);
    emit_sse_immediate(assembler, packed, 0xC2, scratch, other, 5U, "cmpnltp");
    emit_broadcast_pattern(
        assembler, other, true == is_double ? 0x3FF0000000000000ULL : 0x3F8000003F800000ULL);
    emit_sse(assembler, 0U, false, 0x54, scratch, other, "andps");
    emit_all_ones(assembler, other);
    emit_sse_immediate(assembler, 0x66, shift_op, 6U, other, true == is_double ? 63U : 31U, "psll");
    emit_sse(assembler, 0U, false, 0x54, other, x, "andps");
    emit_sse(assembler, 0U, false, 0x56, scratch, other, "orps");
    emit_sse(assembler, packed, false, 0x58, destination, scratch, "addp");
}

/// FCVTNS, FCVTNU, FCVTPS, FCVTPU, FCVTMS, FCVTMU, FCVTZS, FCVTZU, FCVTAS and FCVTAU (scalar,
/// integer) and FCVTZS and FCVTZU (scalar, fixed-point), from single or double precision to a W
/// or X register.
///
/// FCVTZS converts with CVTTSS2SI, whose Inexact matches IXC and whose integer indefinite value,
/// given for NaNs and values out of range with Invalid alone, takes the slow path, which
/// saturates. The other forms round to an integer first, with ROUNDSS (precision exception
/// suppressed) or emit_round_ties_away(), and convert exactly; once the integer is known to be
/// in range (for the unsigned forms: not negative, and below 2^32 for a W register), a second
/// ROUNDSS raises Inexact when the value was not an integer. A saturating conversion so raises
/// Invalid alone, as AArch64 does. The fixed-point forms scale by 2^fbits in double precision,
/// exactly, first.
static bool
emit_fast_float_to_integer(bal_tier1_compiler_t *BAL_RESTRICT compiler, const uint32_t instruction)
{
    // sf 0 0 11110 ftype 1 rmode opcode 000000 Rn Rd, or sf 0 0 11110 ftype 0 11 00U scale Rn Rd.
    const uint32_t ftype    = (instruction >> 22U) & 0x3U;
    const uint32_t rmode    = (instruction >> 19U) & 0x3U;
    const uint32_t opcode   = (instruction >> 16U) & 0x7U;
    const bool     is_64bit = 1U == (instruction >> 31U);
    uint32_t       fbits    = 0U;
    uint32_t       mode     = rmode; // Nearest, plus infinity, minus infinity, zero, ties away.

    if (0x1E200000U == (instruction & 0x7F20FC00U))
    {
        if (opcode > 5U || 2U == (opcode & 6U) || (opcode >= 4U && rmode != 0U))
        {
            return false;
        }

        mode = opcode >= 4U ? 4U : rmode;
    }
    else if (0x1E180000U == (instruction & 0x7F3E0000U))
    {
        const uint32_t scale = (instruction >> 10U) & 0x3FU;

        if (false == is_64bit && scale < 32U)
        {
            return false;
        }

        fbits = 64U - scale;
    }
    else
    {
        return false;
    }

    if (2U == ftype || (3U == ftype && false == has_feature(compiler, BAL_TIER1_FEATURE_FMA)))
    {
        return false;
    }

    const bool                        is_half     = 3U == ftype;
    const bool                        is_unsigned = 1U == (opcode & 1U);
    const bool                        is_direct   = 3U == mode && false == is_unsigned;
    const uint32_t                    rd          = instruction & 0x1FU;
    const uint32_t                    rn          = (instruction >> 5U) & 0x1FU;
    bal_x86_assembler_t *BAL_RESTRICT assembler   = &compiler->assembler;
    bool                              in_double   = 1U == ftype;
    uint8_t                           source      = XMM0;
    size_t                            slow_jumps[3];
    uint32_t                          count = 1U;

    slow_jumps[0] = begin_float_sequence(compiler);

    // An unsigned X of 2^63 or more is in range but would make the signed conversion raise
    // Invalid, and magnitudes of 2^64 and more saturate; checking first also keeps the scaling
    // from overflowing.
    const uint64_t bias = true == in_double ? 1023U : 127U;

    if (true == is_half)
    {
        // A half is below 2^16 and converts to single precision exactly.
        emit_load_half(assembler, XMM2, rn);
        emit_halves_to_singles(assembler, XMM0, XMM2);
    }
    else
    {
        if (true == is_unsigned && true == is_64bit)
        {
            slow_jumps[count++] = emit_magnitude_check(compiler, rn, in_double, bias + 63U - fbits);
        }
        else if (fbits != 0U && true == in_double)
        {
            slow_jumps[count++] = emit_magnitude_check(compiler, rn, true, 1023U + 64U);
        }

        emit_load_scalar(assembler, XMM0, rn, in_double);
    }

    if (fbits != 0U)
    {
        if (false == in_double)
        {
            emit_sse(assembler, 0xF3, false, 0x5A, XMM0, XMM0, "cvtss2sd");
            in_double = true;
        }

        emit_load_constant(assembler, XMM1, (uint64_t)(1023U + fbits) << 52U);
        emit_sse(assembler, 0xF2, false, 0x59, XMM0, XMM1, "mulsd");

        // A scaled half can reach 2^63, where an unsigned X is in range.
        if (true == is_half && true == is_unsigned && true == is_64bit)
        {
            slow_jumps[count++] = emit_register_magnitude_check(assembler, XMM0, true, 1023U + 63U);
        }
    }

    if (false == is_direct)
    {
        if (4U == mode)
        {
            emit_round_ties_away(assembler, in_double);
        }
        else
        {
            emit_round_scalar(assembler, in_double, XMM1, XMM0, ROUND_SUPPRESSED[mode]);
        }

        source = XMM1;
    }

    // An unsigned W converts to 64 bits, so that values up to 2^32 - 1 fit.
    emit_sse(assembler,
             true == in_double ? 0xF2 : 0xF3,
             true == is_64bit || true == is_unsigned,
             0x2C,
             (uint8_t)SCRATCH_A,
             source,
             "cvtts2si");

    if (false == is_unsigned)
    {
        bal_x86_emit_mov_r64_imm64(
            assembler, BAL_X86_RAX, true == is_64bit ? 0x8000000000000000ULL : 0x80000000ULL);
        bal_x86_emit_mov_r64_r64(assembler, SCRATCH_B, SCRATCH_A);
        bal_x86_emit_sub_r64_r64(assembler, SCRATCH_B, BAL_X86_RAX);
        slow_jumps[count++] = emit_forward_jump(assembler, true, BAL_X86_COND_E);
    }
    else if (true == is_64bit)
    {
        bal_x86_emit_test_r64_r64(assembler, SCRATCH_A, SCRATCH_A);
        slow_jumps[count++] = emit_forward_jump(assembler, true, BAL_X86_COND_S);
    }
    else
    {
        bal_x86_emit_mov_r64_r64(assembler, SCRATCH_B, SCRATCH_A);
        bal_x86_emit_shr_r64_imm8(assembler, SCRATCH_B, 32U);
        slow_jumps[count++] = emit_forward_jump(assembler, true, BAL_X86_COND_NE);
    }

    if (false == is_direct)
    {
        // Truncation without suppressing the precision exception raises Inexact exactly when the
        // value is not an integer.
        emit_round_scalar(assembler, in_double, XMM2, XMM0, 0x03);
    }

    if (rd != 31U)
    {
        bal_x86_emit_store_r64_rbp_offset(
            assembler, SCRATCH_A, (int32_t)offsetof(bal_cpu_t, x[rd]));
    }

    emit_slow_path(
        compiler, instruction, (uint64_t)(uintptr_t)&bal_execute_float, slow_jumps, count);
    return true;
}

/// SCVTF and UCVTF (scalar, integer and fixed-point) from a W or X register to single or double
/// precision. CVTSI2SS rounds to nearest under GUEST_MXCSR and raises Inexact as they raise IXC;
/// an unsigned W converts as a zero-extended X, and an unsigned X of 2^63 or more takes the slow
/// path. The fixed-point forms then scale by 2^-fbits, which is exact because the converted
/// integer is zero or at least 1 in magnitude.
static bool
emit_fast_integer_to_float(bal_tier1_compiler_t *BAL_RESTRICT compiler, const uint32_t instruction)
{
    // sf 0 0 11110 ftype 1 00 01U 000000 Rn Rd, or sf 0 0 11110 ftype 0 00 01U scale Rn Rd.
    const uint32_t ftype    = (instruction >> 22U) & 0x3U;
    const bool     is_64bit = 1U == (instruction >> 31U);
    uint32_t       fbits    = 0U;

    if (0x1E220000U == (instruction & 0x7F3EFC00U))
    {
        // Integer.
    }
    else if (0x1E020000U == (instruction & 0x7F3E0000U))
    {
        const uint32_t scale = (instruction >> 10U) & 0x3FU;

        if (false == is_64bit && scale < 32U)
        {
            return false;
        }

        fbits = 64U - scale;
    }
    else
    {
        return false;
    }

    if (2U == ftype || (3U == ftype && false == has_feature(compiler, BAL_TIER1_FEATURE_FMA)))
    {
        return false;
    }

    const bool                        is_half     = 3U == ftype;
    const bool                        is_double   = 1U == ftype;
    const bool                        is_unsigned = 1U == ((instruction >> 16U) & 1U);
    const uint32_t                    rd          = instruction & 0x1FU;
    const uint32_t                    rn          = (instruction >> 5U) & 0x1FU;
    bal_x86_assembler_t *BAL_RESTRICT assembler   = &compiler->assembler;
    size_t                            slow_jumps[3];
    uint32_t                          count = 1U;

    slow_jumps[0] = begin_float_sequence(compiler);

    if (31U == rn)
    {
        bal_x86_emit_xor_r32_r32(assembler, SCRATCH_A, SCRATCH_A);
    }
    else
    {
        bal_x86_emit_load_r64_rbp_offset(assembler, SCRATCH_A, (int32_t)offsetof(bal_cpu_t, x[rn]));
    }

    if (true == is_half)
    {
        // Integers below 2^24 in magnitude convert to single precision exactly, so that only the
        // conversion to half precision rounds; the others take the slow path.
        if (false == is_unsigned && false == is_64bit)
        {
            bal_x86_emit_movsxd_r64_r32(assembler, SCRATCH_A, SCRATCH_A);
        }
        else if (false == is_64bit)
        {
            bal_x86_emit_mov_r32_r32(assembler, SCRATCH_A, SCRATCH_A);
        }

        bal_x86_emit_mov_r64_r64(assembler, SCRATCH_B, SCRATCH_A);

        if (false == is_unsigned)
        {
            bal_x86_emit_mov_r64_imm64(assembler, BAL_X86_RAX, 1ULL << 24U);
            bal_x86_emit_add_r64_r64(assembler, SCRATCH_B, BAL_X86_RAX);
        }

        bal_x86_emit_shr_r64_imm8(assembler, SCRATCH_B, true == is_unsigned ? 24U : 25U);
        slow_jumps[count++] = emit_forward_jump(assembler, true, BAL_X86_COND_NE);
    }
    else if (true == is_unsigned)
    {
        if (true == is_64bit)
        {
            bal_x86_emit_test_r64_r64(assembler, SCRATCH_A, SCRATCH_A);
            slow_jumps[count++] = emit_forward_jump(assembler, true, BAL_X86_COND_S);
        }
        else
        {
            bal_x86_emit_mov_r32_r32(assembler, SCRATCH_A, SCRATCH_A);
        }
    }

    // CVTSI2SS merges into the destination, so xmm0 is cleared first.
    emit_sse(assembler, 0U, false, 0x57, XMM0, XMM0, "xorps");
    emit_sse(assembler,
             true == is_double ? 0xF2 : 0xF3,
             true == is_64bit || true == is_unsigned || true == is_half,
             0x2A,
             XMM0,
             (uint8_t)SCRATCH_A,
             "cvtsi2s");

    if (fbits != 0U)
    {
        emit_load_constant(assembler,
                           XMM1,
                           true == is_double ? (uint64_t)(1023U - fbits) << 52U
                                             : (uint64_t)(127U - fbits) << 23U);
        emit_sse(assembler, true == is_double ? 0xF2 : 0xF3, false, 0x59, XMM0, XMM1, "muls");
    }

    if (true == is_half)
    {
        emit_singles_to_halves(assembler, XMM0, XMM0);
        slow_jumps[count++] = emit_half_checks(assembler);
        emit_sse(assembler, 0x66, false, 0x7E, XMM0, (uint8_t)SCRATCH_A, "movd");
        bal_x86_emit_shl_r32_imm8(assembler, SCRATCH_A, 16U);
        bal_x86_emit_shr_r32_imm8(assembler, SCRATCH_A, 16U);
        emit_store_scalar_result(assembler, rd);
        emit_slow_path(
            compiler, instruction, (uint64_t)(uintptr_t)&bal_execute_float, slow_jumps, count);
        return true;
    }

    emit_sse(assembler, 0x66, is_double, 0x7E, XMM0, (uint8_t)SCRATCH_A, "movd");
    emit_store_scalar_result(assembler, rd);
    emit_slow_path(
        compiler, instruction, (uint64_t)(uintptr_t)&bal_execute_float, slow_jumps, count);
    return true;
}

/// FRINTN, FRINTP, FRINTM, FRINTZ, FRINTA, FRINTX and FRINTI (scalar) in single and double
/// precision. ROUNDSS rounds as they do, raising Inexact only for FRINTX and quieting NaNs as
/// AArch64 does under FPCR = 0. FRINTA, which rounds ties away from zero, uses
/// emit_round_ties_away() on magnitudes below 2^23 (2^52); larger values are already integers
/// and, with infinities and NaNs, take the slow path.
static bool
emit_fast_round(bal_tier1_compiler_t *BAL_RESTRICT compiler, const uint32_t instruction)
{
    // 0 0 0 11110 ftype 1 001 rmode 10000 Rn Rd.
    const uint32_t ftype = (instruction >> 22U) & 0x3U;
    const uint32_t rmode = (instruction >> 15U) & 0x7U;

    if (0x1E244000U != (instruction & 0xFF3C7C00U) || 2U == ftype || 5U == rmode
        || (3U == ftype && false == has_feature(compiler, BAL_TIER1_FEATURE_FMA)))
    {
        return false;
    }

    // FRINTN, FRINTP, FRINTM, FRINTZ, FRINTA, -, FRINTX (the current mode, raising Inexact) and
    // FRINTI (the current mode).
    static const uint8_t IMMEDIATES[8] = { 0x08, 0x0A, 0x09, 0x0B, 0U, 0U, 0x04, 0x0C };

    const bool                        is_double = 1U == ftype;
    const uint32_t                    rd        = instruction & 0x1FU;
    const uint32_t                    rn        = (instruction >> 5U) & 0x1FU;
    bal_x86_assembler_t *BAL_RESTRICT assembler = &compiler->assembler;
    size_t                            slow_jumps[2];
    uint32_t                          count = 1U;

    slow_jumps[0] = begin_float_sequence(compiler);

    if (3U == ftype)
    {
        // Half precision rounds its exact single precision value, whose integers it holds;
        // infinities and NaNs take the slow path.
        emit_load_half(assembler, XMM2, rn);
        emit_halves_to_singles(assembler, XMM0, XMM2);
        slow_jumps[count++] = emit_register_magnitude_check(assembler, XMM0, false, 255U);

        if (4U == rmode)
        {
            emit_round_ties_away(assembler, false);
        }
        else
        {
            emit_round_scalar(assembler, false, XMM1, XMM0, IMMEDIATES[rmode]);
        }

        emit_singles_to_halves(assembler, XMM1, XMM1);
        emit_sse(assembler, 0x66, false, 0x7E, XMM1, (uint8_t)SCRATCH_A, "movd");
        bal_x86_emit_shl_r32_imm8(assembler, SCRATCH_A, 16U);
        bal_x86_emit_shr_r32_imm8(assembler, SCRATCH_A, 16U);
    }
    else if (4U == rmode)
    {
        slow_jumps[count++]
            = emit_magnitude_check(compiler, rn, is_double, true == is_double ? 1075U : 150U);
        emit_load_scalar(assembler, XMM0, rn, is_double);
        emit_round_ties_away(assembler, is_double);
    }
    else
    {
        emit_load_scalar(assembler, XMM0, rn, is_double);
        emit_round_scalar(assembler, is_double, XMM1, XMM0, IMMEDIATES[rmode]);
    }

    if (ftype != 3U)
    {
        emit_sse(assembler, 0x66, is_double, 0x7E, XMM1, (uint8_t)SCRATCH_A, "movd");
    }

    emit_store_scalar_result(assembler, rd);
    emit_slow_path(
        compiler, instruction, (uint64_t)(uintptr_t)&bal_execute_float, slow_jumps, count);
    return true;
}

/// FCVTNS, FCVTMS, FCVTPS, FCVTZS, FCVTNU, FCVTMU, FCVTPU and FCVTZU (vector) on words, which
/// round like emit_fast_float_to_integer() lane by lane, and UCVTF (vector) on words, which
/// converts the upper and lower 16 bits of each lane exactly and rounds once in adding them; and
/// the same with SCVTF and FCVTZS on the single precision scalar, as lane 0.
static bool
emit_fast_vector_convert(bal_tier1_compiler_t *BAL_RESTRICT compiler, const uint32_t instruction)
{
    // 0 Q U 01110 a 0 10000 opcode 10 Rn Rd, or the scalar 01 U 11110 a 0 10000 opcode 10 Rn Rd.
    const bool is_scalar = 0x5E200800U == (instruction & 0xDF7E0C00U);

    if ((0x0E200800U != (instruction & 0x9F7E0C00U) && false == is_scalar)
        || false == has_feature(compiler, BAL_TIER1_FEATURE_SSE41))
    {
        return false;
    }

    const uint32_t key = (((instruction >> 29U) & 1U) << 6U) | (((instruction >> 23U) & 1U) << 5U)
                         | ((instruction >> 12U) & 0x1FU);
    uint32_t mode        = 0U;
    bool     is_unsigned = false;

    switch (key)
    {
        case 0x5AU: // FCVTNU
            is_unsigned = true;
            mode        = 0U;
            break;
        case 0x1AU: // FCVTNS
            mode = 0U;
            break;
        case 0x7AU: // FCVTPU
            is_unsigned = true;
            mode        = 1U;
            break;
        case 0x3AU: // FCVTPS
            mode = 1U;
            break;
        case 0x5BU: // FCVTMU
            is_unsigned = true;
            mode        = 2U;
            break;
        case 0x1BU: // FCVTMS
            mode = 2U;
            break;
        case 0x7BU: // FCVTZU
            is_unsigned = true;
            mode        = 3U;
            break;
        case 0x5DU: // UCVTF
            mode = 4U;
            break;
        case 0x5CU: // FCVTAU
            is_unsigned = true;
            mode        = 6U;
            break;
        case 0x1CU: // FCVTAS
            mode = 6U;
            break;
        case 0x3BU: // FCVTZS (the vector form is emit_fast_float_misc())
            mode = 3U;
            break;
        case 0x1DU: // SCVTF (likewise)
            mode = 5U;
            break;
        default:
            return false;
    }

    const bool     is_quad = false == is_scalar && 1U == ((instruction >> 30U) & 1U);
    const uint32_t rd      = instruction & 0x1FU;
    const uint32_t rn      = (instruction >> 5U) & 0x1FU;
    bal_x86_assembler_t *BAL_RESTRICT assembler = &compiler->assembler;
    size_t                            slow_jumps[3];
    uint32_t                          count = 1U;

    slow_jumps[0] = begin_float_sequence(compiler);

    if (true == is_scalar)
    {
        emit_load_scalar(assembler, XMM0, rn, false);
    }
    else
    {
        emit_load_vector(assembler, XMM0, rn, is_quad);
    }

    if (true == is_unsigned)
    {
        // Lanes of 2^31 and more (and NaNs) would make the signed conversion raise Invalid,
        // although some are in range: their exponents, above 157, take the slow path first.
        emit_sse(assembler, 0U, false, 0x28, XMM2, XMM0, "movaps");
        emit_sse_immediate(assembler, 0x66, 0x72, 6U, XMM2, 1U, "pslld");
        emit_sse_immediate(assembler, 0x66, 0x72, 2U, XMM2, 24U, "psrld");
        emit_broadcast_pattern(assembler, XMM3, 0x0000009D0000009DULL);
        emit_sse(assembler, 0x66, false, 0x66, XMM2, XMM3, "pcmpgtd");
        emit_sse(assembler, 0U, false, 0x50, (uint8_t)SCRATCH_A, XMM2, "movmskps");
        bal_x86_emit_test_r32_r32(assembler, SCRATCH_A, SCRATCH_A);
        slow_jumps[count++] = emit_forward_jump(assembler, true, BAL_X86_COND_NE);
    }

    if (4U == mode)
    {
        // (float)(x >> 16) * 65536 + (float)(x & 0xFFFF): only the addition rounds.
        emit_sse(assembler, 0U, false, 0x28, XMM1, XMM0, "movaps");
        emit_sse_immediate(assembler, 0x66, 0x72, 2U, XMM1, 16U, "psrld");
        emit_sse(assembler, 0U, false, 0x5B, XMM1, XMM1, "cvtdq2ps");
        bal_x86_emit_mov_r64_imm64(assembler, SCRATCH_A, 0x47800000ULL);
        emit_sse(assembler, 0x66, false, 0x6E, XMM2, (uint8_t)SCRATCH_A, "movd");
        emit_sse_immediate(assembler, 0x66, 0x70, XMM2, XMM2, 0U, "pshufd");
        emit_sse(assembler, 0U, false, 0x59, XMM1, XMM2, "mulps");
        emit_sse_immediate(assembler, 0x66, 0x72, 6U, XMM0, 16U, "pslld");
        emit_sse_immediate(assembler, 0x66, 0x72, 2U, XMM0, 16U, "psrld");
        emit_sse(assembler, 0U, false, 0x5B, XMM0, XMM0, "cvtdq2ps");
        emit_sse(assembler, 0U, false, 0x58, XMM1, XMM0, "addps");
    }
    else if (5U == mode)
    {
        emit_sse(assembler, 0U, false, 0x5B, XMM1, XMM0, "cvtdq2ps");
    }
    else
    {
        if (6U == mode)
        {
            emit_round_ties_away_packed(assembler, false, XMM1, XMM0, XMM2, XMM3);
        }
        else
        {
            emit_sse3a(assembler, 0x08, XMM1, XMM0, ROUND_SUPPRESSED[mode], "roundps");
        }

        emit_sse(assembler, 0xF3, false, 0x5B, XMM1, XMM1, "cvttps2dq");

        // The indefinite value 0x80000000 marks NaNs and values out of range; the unsigned forms
        // also send every negative integer (and so each lane of 2^31 or more) to the slow path.
        if (true == is_unsigned)
        {
            emit_sse(assembler, 0U, false, 0x50, (uint8_t)SCRATCH_A, XMM1, "movmskps");
        }
        else
        {
            emit_all_ones(assembler, XMM2);
            emit_sse_immediate(assembler, 0x66, 0x72, 6U, XMM2, 31U, "pslld");
            emit_sse(assembler, 0x66, false, 0x76, XMM2, XMM1, "pcmpeqd");
            emit_sse(assembler, 0U, false, 0x50, (uint8_t)SCRATCH_A, XMM2, "movmskps");
        }

        bal_x86_emit_test_r32_r32(assembler, SCRATCH_A, SCRATCH_A);
        slow_jumps[count++] = emit_forward_jump(assembler, true, BAL_X86_COND_NE);
        emit_sse3a(assembler, 0x08, XMM2, XMM0, 0x03, "roundps");
    }

    if (true == is_scalar)
    {
        emit_sse(assembler, 0x66, false, 0x7E, XMM1, (uint8_t)SCRATCH_A, "movd");
        emit_store_scalar_result(assembler, rd);
    }
    else
    {
        emit_store_vector(assembler, XMM1, rd, is_quad);
    }

    emit_slow_path(compiler, instruction, float_helper(instruction), slow_jumps, count);
    return true;
}

// -----------------------------------------------------------------------------
// Inline Advanced SIMD integer arithmetic
// -----------------------------------------------------------------------------

/// PADDB, PADDW, PADDD and PADDQ, and the PSUB opcodes, by lane size.
static const uint8_t LANE_ADD[4] = { 0xFC, 0xFD, 0xFE, 0xD4 };
static const uint8_t LANE_SUB[4] = { 0xF8, 0xF9, 0xFA, 0xFB };

/// The value 1 in every `8 << size`-bit lane, as a 64-bit pattern.
static uint64_t
lane_one_pattern(const uint32_t size)
{
    static const uint64_t PATTERNS[4]
        = { 0x0101010101010101ULL, 0x0001000100010001ULL, 0x0000000100000001ULL, 1ULL };
    return PATTERNS[size];
}

/// Loads the 128-bit constant `high`:`low` into `xmm` (SSE4.1).
static void
emit_load_vector_constant(bal_x86_assembler_t *BAL_RESTRICT assembler,
                          const uint8_t                     xmm,
                          const uint64_t                    low,
                          const uint64_t                    high)
{
    if (low == high)
    {
        emit_broadcast_pattern(assembler, xmm, low);
        return;
    }

    static const uint8_t PINSRQ[3] = { 0x0F, 0x3A, 0x22 };
    emit_load_constant(assembler, xmm, low);
    bal_x86_emit_mov_r64_imm64(assembler, SCRATCH_A, high);
    bal_x86_emit_sse_register(
        assembler, 0x66, true, PINSRQ, 3U, xmm, (uint8_t)SCRATCH_A, true, 1U, "pinsrq");
}

/// Emits PMAXSB/W/D or PMAXUB/W/D (or the PMIN forms) `a`, `b` on `8 << size`-bit lanes, signed
/// unless `u` is 1. Some need SSE4.1, which the callers check.
static void
emit_lane_extreme(bal_x86_assembler_t *BAL_RESTRICT assembler,
                  const bool                        is_minimum,
                  const uint32_t                    u,
                  const uint32_t                    size,
                  const uint8_t                     a,
                  const uint8_t                     b)
{
    // Each entry is the opcode, with 0x3800 marking the 0F 38 map.
    static const uint16_t MAXIMA[2][3] = { { 0x383C, 0x00EE, 0x383D }, { 0x00DE, 0x383E, 0x383F } };
    static const uint16_t MINIMA[2][3] = { { 0x3838, 0x00EA, 0x3839 }, { 0x00DA, 0x383A, 0x383B } };
    const uint16_t        entry        = true == is_minimum ? MINIMA[u][size] : MAXIMA[u][size];

    if (0U != (entry & 0xFF00U))
    {
        emit_sse38(assembler, (uint8_t)entry, a, b, "pmax/pmin");
        return;
    }

    emit_sse(assembler, 0x66, false, (uint8_t)entry, a, b, "pmax/pmin");
}

/// Sets FPSR.QC.
static void
emit_set_saturation(bal_x86_assembler_t *BAL_RESTRICT assembler)
{
    bal_x86_emit_load_r64_rbp_offset(assembler, SCRATCH_B, (int32_t)offsetof(bal_cpu_t, fpsr));
    bal_x86_emit_mov_r64_imm64(assembler, SCRATCH_A, 1ULL << 27U);
    bal_x86_emit_or_r64_r64(assembler, SCRATCH_B, SCRATCH_A);
    bal_x86_emit_store_r64_rbp_offset(assembler, SCRATCH_B, (int32_t)offsetof(bal_cpu_t, fpsr));
}

/// Sets FPSR.QC unless `a` and `b` are equal, clobbering `a`.
static void
emit_saturation_unless_equal(bal_x86_assembler_t *BAL_RESTRICT assembler,
                             const uint8_t                     a,
                             const uint8_t                     b)
{
    emit_sse(assembler, 0x66, false, 0x74, a, b, "pcmpeqb");
    emit_sse(assembler, 0x66, false, 0xD7, (uint8_t)SCRATCH_A, a, "pmovmskb");
    bal_x86_emit_sub_r32_imm32(assembler, SCRATCH_A, 0xFFFF);
    const size_t equal = emit_forward_jump(assembler, true, BAL_X86_COND_E);
    emit_set_saturation(assembler);
    patch_forward_jump(assembler, equal, true);
}

/// Sign- or zero-extends the `8 << size`-bit lanes in the lower (or, for `is_upper`, the upper)
/// half of `xmm` to twice their width (SSE4.1).
static void
emit_widen(bal_x86_assembler_t *BAL_RESTRICT assembler,
           const uint8_t                     xmm,
           const uint32_t                    size,
           const bool                        is_unsigned,
           const bool                        is_upper)
{
    static const uint8_t ZERO_EXTEND[3] = { 0x30, 0x33, 0x35 };
    static const uint8_t SIGN_EXTEND[3] = { 0x20, 0x23, 0x25 };

    if (true == is_upper)
    {
        emit_sse_immediate(assembler, 0x66, 0x73, 3U, xmm, 8U, "psrldq");
    }

    emit_sse38(assembler,
               true == is_unsigned ? ZERO_EXTEND[size] : SIGN_EXTEND[size],
               xmm,
               xmm,
               "pmovzx/pmovsx");
}

/// Splits the `8 << size`-bit elements of xmm0 followed by those of xmm1 (64 or 128 bits each)
/// into the even ones, in order, in xmm0 and the odd ones in xmm1, as the pairwise operations
/// pair them. Uses xmm2 and xmm3 (SSE4.1). Only the low 64 bits are meaningful for 64-bit
/// vectors.
static void
emit_deinterleave(bal_x86_assembler_t *BAL_RESTRICT assembler,
                  const uint32_t                    size,
                  const bool                        is_quad)
{
    // PSHUFB masks gathering the even bytes or halfwords into the low half and the odd ones into
    // the high half.
    static const uint64_t LOW[2]  = { 0x0E0C0A0806040200ULL, 0x0D0C090805040100ULL };
    static const uint64_t HIGH[2] = { 0x0F0D0B0907050301ULL, 0x0F0E0B0A07060302ULL };

    if (false == is_quad)
    {
        emit_sse(assembler, 0x66, false, 0x6C, XMM0, XMM1, "punpcklqdq");

        if (2U == size)
        {
            emit_sse_immediate(assembler, 0x66, 0x70, XMM1, XMM0, 0x0D, "pshufd");
            emit_sse_immediate(assembler, 0x66, 0x70, XMM0, XMM0, 0x08, "pshufd");
            return;
        }

        emit_load_vector_constant(assembler, XMM2, LOW[size], HIGH[size]);
        emit_sse38(assembler, 0x00, XMM0, XMM2, "pshufb");
        emit_sse(assembler, 0U, false, 0x28, XMM1, XMM0, "movaps");
        emit_sse_immediate(assembler, 0x66, 0x73, 3U, XMM1, 8U, "psrldq");
        return;
    }

    if (3U == size)
    {
        emit_sse(assembler, 0U, false, 0x28, XMM2, XMM0, "movaps");
        emit_sse(assembler, 0x66, false, 0x6C, XMM0, XMM1, "punpcklqdq");
        emit_sse(assembler, 0x66, false, 0x6D, XMM2, XMM1, "punpckhqdq");
        emit_sse(assembler, 0U, false, 0x28, XMM1, XMM2, "movaps");
        return;
    }

    if (2U == size)
    {
        emit_sse(assembler, 0U, false, 0x28, XMM2, XMM0, "movaps");
        emit_sse_immediate(assembler, 0U, 0xC6, XMM0, XMM1, 0x88, "shufps");
        emit_sse_immediate(assembler, 0U, 0xC6, XMM2, XMM1, 0xDD, "shufps");
        emit_sse(assembler, 0U, false, 0x28, XMM1, XMM2, "movaps");
        return;
    }

    emit_load_vector_constant(assembler, XMM2, LOW[size], HIGH[size]);
    emit_sse38(assembler, 0x00, XMM0, XMM2, "pshufb");
    emit_sse38(assembler, 0x00, XMM1, XMM2, "pshufb");
    emit_sse(assembler, 0U, false, 0x28, XMM3, XMM0, "movaps");
    emit_sse(assembler, 0x66, false, 0x6C, XMM0, XMM1, "punpcklqdq");
    emit_sse(assembler, 0x66, false, 0x6D, XMM3, XMM1, "punpckhqdq");
    emit_sse(assembler, 0U, false, 0x28, XMM1, XMM3, "movaps");
}

/// More of Advanced SIMD three same, integer: SQADD, UQADD, SQSUB and UQSUB on bytes and
/// halfwords, SHADD, UHADD, SRHADD and URHADD, SABD, UABD, SABA and UABA, MLA and MLS, and the
/// pairwise ADDP, SMAXP, UMAXP, SMINP and UMINP.
static bool
emit_fast_vector_integer_more(bal_tier1_compiler_t *BAL_RESTRICT compiler,
                              const uint32_t                     instruction)
{
    // 0 Q U 01110 size 1 Rm opcode 1 Rn Rd.
    if (0x0E200400U != (instruction & 0x9F200400U)
        || false == has_feature(compiler, BAL_TIER1_FEATURE_SSE41))
    {
        return false;
    }

    const bool     is_quad = 1U == ((instruction >> 30U) & 1U);
    const uint32_t u       = (instruction >> 29U) & 1U;
    const uint32_t size    = (instruction >> 22U) & 0x3U;
    const uint32_t opcode  = (instruction >> 11U) & 0x1FU;

    switch (opcode)
    {
        case 0x01U: // SQADD, UQADD
        case 0x05U: // SQSUB, UQSUB
            if (size > 1U)
            {
                return false;
            }
            break;
        case 0x00U: // SHADD, UHADD
        case 0x02U: // SRHADD, URHADD
        case 0x0EU: // SABD, UABD
        case 0x0FU: // SABA, UABA
            if (3U == size)
            {
                return false;
            }
            break;
        case 0x12U: // MLA, MLS
            if (0U == size || 3U == size)
            {
                return false;
            }
            break;
        case 0x14U: // SMAXP, UMAXP
        case 0x15U: // SMINP, UMINP
            if (3U == size)
            {
                return false;
            }
            break;
        case 0x17U: // ADDP
            if (1U == u || (3U == size && false == is_quad))
            {
                return false;
            }
            break;
        default:
            return false;
    }

    const uint32_t                    rd        = instruction & 0x1FU;
    const uint32_t                    rn        = (instruction >> 5U) & 0x1FU;
    const uint32_t                    rm        = (instruction >> 16U) & 0x1FU;
    bal_x86_assembler_t *BAL_RESTRICT assembler = &compiler->assembler;
    uint8_t                           result    = XMM0;

    bal_sliding_window_flush_all(&compiler->window);
    emit_load_vector(assembler, XMM0, rn, is_quad);
    emit_load_vector(assembler, XMM1, rm, is_quad);

    switch (opcode)
    {
        case 0x01U:
        case 0x05U: {
            // PADDSB, PADDUSB, PSUBSB, PSUBUSB and their word forms saturate as these do; a
            // lane saturated where the wrapping result differs.
            static const uint8_t SATURATING[2][2][2]
                = { { { 0xEC, 0xED }, { 0xDC, 0xDD } }, { { 0xE8, 0xE9 }, { 0xD8, 0xD9 } } };
            const uint32_t is_subtract = 0x05U == opcode ? 1U : 0U;
            emit_sse(assembler, 0U, false, 0x28, XMM2, XMM0, "movaps");
            emit_sse(assembler, 0x66, false, SATURATING[is_subtract][u][size], XMM0, XMM1, "padds");
            emit_sse(assembler,
                     0x66,
                     false,
                     1U == is_subtract ? LANE_SUB[size] : LANE_ADD[size],
                     XMM2,
                     XMM1,
                     "padd/psub");
            emit_store_vector(assembler, XMM0, rd, is_quad);
            emit_saturation_unless_equal(assembler, XMM2, XMM0);
            return true;
        }
        case 0x00U:
        case 0x02U:
            // Signed lanes are biased to unsigned ones, which keeps their order and averages.
            if (0U == u)
            {
                emit_broadcast_pattern(assembler, XMM3, lane_sign_pattern(size));
                emit_sse(assembler, 0x66, false, 0xEF, XMM0, XMM3, "pxor");
                emit_sse(assembler, 0x66, false, 0xEF, XMM1, XMM3, "pxor");
            }

            if (size < 2U)
            {
                // PAVGB and PAVGW round up; the truncating average is one less where the sum is
                // odd.
                emit_sse(assembler, 0U, false, 0x28, XMM2, XMM0, "movaps");
                emit_sse(assembler, 0x66, false, 0U == size ? 0xE0 : 0xE3, XMM2, XMM1, "pavg");

                if (0x00U == opcode)
                {
                    emit_sse(assembler, 0x66, false, 0xEF, XMM0, XMM1, "pxor");
                    emit_broadcast_pattern(assembler, XMM1, lane_one_pattern(size));
                    emit_sse(assembler, 0x66, false, 0xDB, XMM0, XMM1, "pand");
                    emit_sse(assembler, 0x66, false, LANE_SUB[size], XMM2, XMM0, "psub");
                }

                result = XMM2;
            }
            else
            {
                // (n & m) + ((n ^ m) >> 1), or rounding up (n | m) - ((n ^ m) >> 1).
                emit_sse(assembler, 0U, false, 0x28, XMM2, XMM0, "movaps");
                emit_sse(assembler, 0x66, false, 0xEF, XMM2, XMM1, "pxor");
                emit_sse_immediate(assembler, 0x66, 0x72, 2U, XMM2, 1U, "psrld");

                if (0x00U == opcode)
                {
                    emit_sse(assembler, 0x66, false, 0xDB, XMM0, XMM1, "pand");
                    emit_sse(assembler, 0x66, false, 0xFE, XMM0, XMM2, "paddd");
                }
                else
                {
                    emit_sse(assembler, 0x66, false, 0xEB, XMM0, XMM1, "por");
                    emit_sse(assembler, 0x66, false, 0xFA, XMM0, XMM2, "psubd");
                }
            }

            if (0U == u)
            {
                emit_sse(assembler, 0x66, false, 0xEF, result, XMM3, "pxor");
            }
            break;
        case 0x0EU:
        case 0x0FU:
            // |n - m| is the larger minus the smaller, which wraps to the right unsigned value.
            emit_sse(assembler, 0U, false, 0x28, XMM2, XMM0, "movaps");
            emit_lane_extreme(assembler, false, u, size, XMM0, XMM1);
            emit_lane_extreme(assembler, true, u, size, XMM2, XMM1);
            emit_sse(assembler, 0x66, false, LANE_SUB[size], XMM0, XMM2, "psub");

            if (0x0FU == opcode)
            {
                emit_load_vector(assembler, XMM1, rd, is_quad);
                emit_sse(assembler, 0x66, false, LANE_ADD[size], XMM0, XMM1, "padd");
            }
            break;
        case 0x12U:
            if (1U == size)
            {
                emit_sse(assembler, 0x66, false, 0xD5, XMM0, XMM1, "pmullw");
            }
            else
            {
                emit_sse38(assembler, 0x40, XMM0, XMM1, "pmulld");
            }

            emit_load_vector(assembler, XMM1, rd, is_quad);
            emit_sse(assembler,
                     0x66,
                     false,
                     0U == u ? LANE_ADD[size] : LANE_SUB[size],
                     XMM1,
                     XMM0,
                     "padd/psub");
            result = XMM1;
            break;
        default:
            emit_deinterleave(assembler, size, is_quad);

            if (0x17U == opcode)
            {
                emit_sse(assembler, 0x66, false, LANE_ADD[size], XMM0, XMM1, "padd");
            }
            else
            {
                emit_lane_extreme(assembler, 0x15U == opcode, u, size, XMM0, XMM1);
            }
            break;
    }

    emit_store_vector(assembler, result, rd, is_quad);
    return true;
}

/// ADDP (scalar): the sum of the two doublewords.
static bool
emit_fast_add_pair(bal_tier1_compiler_t *BAL_RESTRICT compiler, const uint32_t instruction)
{
    // 01 0 11110 11 11000 11011 10 Rn Rd.
    if (0x5EF1B800U != (instruction & 0xFFFFFC00U))
    {
        return false;
    }

    const uint32_t                    rd        = instruction & 0x1FU;
    const uint32_t                    rn        = (instruction >> 5U) & 0x1FU;
    bal_x86_assembler_t *BAL_RESTRICT assembler = &compiler->assembler;
    bal_sliding_window_flush_all(&compiler->window);
    emit_load_vector(assembler, XMM0, rn, true);
    emit_sse_immediate(assembler, 0x66, 0x70, XMM1, XMM0, 0xEE, "pshufd");
    emit_sse(assembler, 0x66, false, 0xD4, XMM0, XMM1, "paddq");
    emit_store_vector(assembler, XMM0, rd, false);
    return true;
}

/// Advanced SIMD three different, widening: SADDL, UADDL, SADDW, UADDW, SSUBL, USUBL, SSUBW,
/// USUBW, SABAL, UABAL, SABDL, UABDL, SMLAL, UMLAL, SMLSL, UMLSL, SMULL and UMULL, with the "2"
/// forms, which take the upper halves of the narrow operands. Extended with PMOVSX or PMOVZX,
/// the operands give exact results in the wide lanes; PMULDQ and PMULUDQ multiply words into
/// doublewords. The absolute differences fit the narrow lanes as unsigned values and are
/// zero-extended.
static bool
emit_fast_widening(bal_tier1_compiler_t *BAL_RESTRICT compiler, const uint32_t instruction)
{
    // 0 Q U 01110 size 1 Rm opcode 00 Rn Rd.
    const uint32_t size   = (instruction >> 22U) & 0x3U;
    const uint32_t opcode = (instruction >> 12U) & 0xFU;

    if (0x0E200000U != (instruction & 0x9F200C00U) || 3U == size
        || false == has_feature(compiler, BAL_TIER1_FEATURE_SSE41))
    {
        return false;
    }

    switch (opcode)
    {
        case 0x0U: // SADDL, UADDL
        case 0x1U: // SADDW, UADDW
        case 0x2U: // SSUBL, USUBL
        case 0x3U: // SSUBW, USUBW
        case 0x5U: // SABAL, UABAL
        case 0x7U: // SABDL, UABDL
        case 0x8U: // SMLAL, UMLAL
        case 0xAU: // SMLSL, UMLSL
        case 0xCU: // SMULL, UMULL
            break;
        default:
            return false;
    }

    const bool                        is_upper    = 1U == ((instruction >> 30U) & 1U);
    const bool                        is_unsigned = 1U == ((instruction >> 29U) & 1U);
    const uint32_t                    rd          = instruction & 0x1FU;
    const uint32_t                    rn          = (instruction >> 5U) & 0x1FU;
    const uint32_t                    rm          = (instruction >> 16U) & 0x1FU;
    const uint32_t                    wide        = size + 1U;
    bal_x86_assembler_t *BAL_RESTRICT assembler   = &compiler->assembler;
    uint8_t                           result      = XMM0;

    bal_sliding_window_flush_all(&compiler->window);
    emit_load_vector(assembler, XMM0, rn, true);
    emit_load_vector(assembler, XMM1, rm, true);

    if (0x5U == opcode || 0x7U == opcode)
    {
        if (true == is_upper)
        {
            emit_sse_immediate(assembler, 0x66, 0x73, 3U, XMM0, 8U, "psrldq");
            emit_sse_immediate(assembler, 0x66, 0x73, 3U, XMM1, 8U, "psrldq");
        }

        const uint32_t u = true == is_unsigned ? 1U : 0U;
        emit_sse(assembler, 0U, false, 0x28, XMM2, XMM0, "movaps");
        emit_lane_extreme(assembler, false, u, size, XMM0, XMM1);
        emit_lane_extreme(assembler, true, u, size, XMM2, XMM1);
        emit_sse(assembler, 0x66, false, LANE_SUB[size], XMM0, XMM2, "psub");
        emit_widen(assembler, XMM0, size, true, false);

        if (0x5U == opcode)
        {
            emit_load_vector(assembler, XMM1, rd, true);
            emit_sse(assembler, 0x66, false, LANE_ADD[wide], XMM0, XMM1, "padd");
        }

        emit_store_vector(assembler, XMM0, rd, true);
        return true;
    }

    // The W forms take a wide first operand.
    if (opcode != 0x1U && opcode != 0x3U)
    {
        emit_widen(assembler, XMM0, size, is_unsigned, is_upper);
    }

    emit_widen(assembler, XMM1, size, is_unsigned, is_upper);

    switch (opcode)
    {
        case 0x0U:
        case 0x1U:
            emit_sse(assembler, 0x66, false, LANE_ADD[wide], XMM0, XMM1, "padd");
            break;
        case 0x2U:
        case 0x3U:
            emit_sse(assembler, 0x66, false, LANE_SUB[wide], XMM0, XMM1, "psub");
            break;
        default:
            if (0U == size)
            {
                emit_sse(assembler, 0x66, false, 0xD5, XMM0, XMM1, "pmullw");
            }
            else if (1U == size)
            {
                emit_sse38(assembler, 0x40, XMM0, XMM1, "pmulld");
            }
            else if (true == is_unsigned)
            {
                emit_sse(assembler, 0x66, false, 0xF4, XMM0, XMM1, "pmuludq");
            }
            else
            {
                emit_sse38(assembler, 0x28, XMM0, XMM1, "pmuldq");
            }

            if (opcode != 0xCU)
            {
                emit_load_vector(assembler, XMM1, rd, true);
                emit_sse(assembler,
                         0x66,
                         false,
                         0x8U == opcode ? LANE_ADD[wide] : LANE_SUB[wide],
                         XMM1,
                         XMM0,
                         "padd/psub");
                result = XMM1;
            }
            break;
    }

    emit_store_vector(assembler, result, rd, true);
    return true;
}

/// Copies element `index` of the `8 << size`-bit lanes of `xmm` (halfwords or words) to all of
/// its lanes.
static void
emit_broadcast_element(bal_x86_assembler_t *BAL_RESTRICT assembler,
                       const uint8_t                     xmm,
                       const uint32_t                    size,
                       const uint32_t                    index)
{
    if (2U == size)
    {
        emit_sse_immediate(assembler, 0x66, 0x70, xmm, xmm, (uint8_t)(index * 0x55U), "pshufd");
        return;
    }

    if (index < 4U)
    {
        emit_sse_immediate(assembler, 0xF2, 0x70, xmm, xmm, (uint8_t)(index * 0x55U), "pshuflw");
        emit_sse(assembler, 0x66, false, 0x6C, xmm, xmm, "punpcklqdq");
        return;
    }

    emit_sse_immediate(assembler, 0xF3, 0x70, xmm, xmm, (uint8_t)((index - 4U) * 0x55U), "pshufhw");
    emit_sse(assembler, 0x66, false, 0x6D, xmm, xmm, "punpckhqdq");
}

/// MUL, MLA and MLS (by element), and SMULL, UMULL, SMLAL, UMLAL, SMLSL and UMLSL (by element)
/// with their "2" forms, on halfwords and words: the element is broadcast and the operation is
/// then the vector one.
static bool
emit_fast_integer_by_element(bal_tier1_compiler_t *BAL_RESTRICT compiler,
                             const uint32_t                     instruction)
{
    // 0 Q U 01111 size L M Rm opcode H 0 Rn Rd.
    const uint32_t size   = (instruction >> 22U) & 0x3U;
    const uint32_t opcode = (instruction >> 12U) & 0xFU;
    const uint32_t u      = (instruction >> 29U) & 1U;

    if (0x0F000000U != (instruction & 0x9F000400U) || 0U == size || 3U == size
        || false == has_feature(compiler, BAL_TIER1_FEATURE_SSE41))
    {
        return false;
    }

    bool is_long = false;

    switch (opcode)
    {
        case 0x8U: // MUL
            if (1U == u)
            {
                return false;
            }
            break;
        case 0x0U: // MLA
        case 0x4U: // MLS
            if (0U == u)
            {
                return false;
            }
            break;
        case 0xAU: // SMULL, UMULL
        case 0x2U: // SMLAL, UMLAL
        case 0x6U: // SMLSL, UMLSL
            is_long = true;
            break;
        default:
            return false;
    }

    const bool     is_quad = 1U == ((instruction >> 30U) & 1U);
    const uint32_t h       = (instruction >> 11U) & 1U;
    const uint32_t l       = (instruction >> 21U) & 1U;
    const uint32_t m       = (instruction >> 20U) & 1U;
    const uint32_t index   = 1U == size ? (h << 2U) | (l << 1U) | m : (h << 1U) | l;
    const uint32_t rm   = 1U == size ? (instruction >> 16U) & 0xFU : (instruction >> 16U) & 0x1FU;
    const uint32_t rd   = instruction & 0x1FU;
    const uint32_t rn   = (instruction >> 5U) & 0x1FU;
    const uint32_t wide = size + 1U;
    bal_x86_assembler_t *BAL_RESTRICT assembler = &compiler->assembler;

    bal_sliding_window_flush_all(&compiler->window);
    emit_load_vector(assembler, XMM0, rn, true == is_quad || true == is_long);
    emit_load_vector(assembler, XMM1, rm, true);
    emit_broadcast_element(assembler, XMM1, size, index);

    if (false == is_long)
    {
        if (1U == size)
        {
            emit_sse(assembler, 0x66, false, 0xD5, XMM0, XMM1, "pmullw");
        }
        else
        {
            emit_sse38(assembler, 0x40, XMM0, XMM1, "pmulld");
        }

        if (0x8U == opcode)
        {
            emit_store_vector(assembler, XMM0, rd, is_quad);
            return true;
        }

        emit_load_vector(assembler, XMM1, rd, is_quad);
        emit_sse(assembler,
                 0x66,
                 false,
                 0x0U == opcode ? LANE_ADD[size] : LANE_SUB[size],
                 XMM1,
                 XMM0,
                 "padd/psub");
        emit_store_vector(assembler, XMM1, rd, is_quad);
        return true;
    }

    const bool is_unsigned = 1U == u;
    emit_widen(assembler, XMM0, size, is_unsigned, is_quad);
    emit_widen(assembler, XMM1, size, is_unsigned, false);

    if (1U == size)
    {
        emit_sse38(assembler, 0x40, XMM0, XMM1, "pmulld");
    }
    else if (true == is_unsigned)
    {
        emit_sse(assembler, 0x66, false, 0xF4, XMM0, XMM1, "pmuludq");
    }
    else
    {
        emit_sse38(assembler, 0x28, XMM0, XMM1, "pmuldq");
    }

    if (0xAU == opcode)
    {
        emit_store_vector(assembler, XMM0, rd, true);
        return true;
    }

    emit_load_vector(assembler, XMM1, rd, true);
    emit_sse(assembler,
             0x66,
             false,
             0x2U == opcode ? LANE_ADD[wide] : LANE_SUB[wide],
             XMM1,
             XMM0,
             "padd/psub");
    emit_store_vector(assembler, XMM1, rd, true);
    return true;
}

/// Repeats the `8 << size`-bit `value` across 64 bits.
static uint64_t
replicate_lane(const uint64_t value, const uint32_t size)
{
    const uint32_t esize = 8U << size;
    const uint64_t lane  = 64U == esize ? value : value & ((1ULL << esize) - 1U);
    uint64_t       bits  = 0U;

    for (uint32_t i = 0U; i < 64U; i += esize)
    {
        bits |= lane << i;
    }

    return bits;
}

/// The PSLL, PSRL and PSRA (by immediate) opcodes by lane size: 0F 71 for words, 72 for
/// doublewords and 73 for quadwords, with /6, /2 and /4.
static const uint8_t LANE_SHIFT[4] = { 0U, 0x71, 0x72, 0x73 };

/// Shifts the `8 << size`-bit lanes of `xmm` left by `amount`, below the lane size. Bytes shift
/// as words and lose the bits that crossed into the next byte; that mask uses `scratch`.
static void
emit_lanes_shift_left(bal_x86_assembler_t *BAL_RESTRICT assembler,
                      const uint8_t                     xmm,
                      const uint32_t                    size,
                      const uint32_t                    amount,
                      const uint8_t                     scratch)
{
    if (0U == amount)
    {
        return;
    }

    if (0U == size)
    {
        emit_sse_immediate(assembler, 0x66, 0x71, 6U, xmm, (uint8_t)amount, "psllw");
        emit_broadcast_pattern(assembler, scratch, replicate_lane((0xFFU << amount) & 0xFFU, 0U));
        emit_sse(assembler, 0x66, false, 0xDB, xmm, scratch, "pand");
        return;
    }

    emit_sse_immediate(assembler, 0x66, LANE_SHIFT[size], 6U, xmm, (uint8_t)amount, "psll");
}

/// Shifts the lanes of `xmm` right logically by `amount`, up to the lane size.
static void
emit_lanes_shift_right(bal_x86_assembler_t *BAL_RESTRICT assembler,
                       const uint8_t                     xmm,
                       const uint32_t                    size,
                       const uint32_t                    amount,
                       const uint8_t                     scratch)
{
    if (0U == amount)
    {
        return;
    }

    if (0U == size)
    {
        if (amount >= 8U)
        {
            emit_zero_vector(assembler, xmm);
            return;
        }

        emit_sse_immediate(assembler, 0x66, 0x71, 2U, xmm, (uint8_t)amount, "psrlw");
        emit_broadcast_pattern(assembler, scratch, replicate_lane(0xFFU >> amount, 0U));
        emit_sse(assembler, 0x66, false, 0xDB, xmm, scratch, "pand");
        return;
    }

    emit_sse_immediate(assembler, 0x66, LANE_SHIFT[size], 2U, xmm, (uint8_t)amount, "psrl");
}

/// Shifts the lanes of `xmm` right arithmetically by `amount`, up to the lane size. Bytes and
/// doublewords, which SSE cannot shift so, shift logically and then sign-extend from the moved
/// sign bit m: (x ^ m) - m.
static void
emit_lanes_shift_arithmetic(bal_x86_assembler_t *BAL_RESTRICT assembler,
                            const uint8_t                     xmm,
                            const uint32_t                    size,
                            const uint32_t                    amount,
                            const uint8_t                     scratch)
{
    // A shift by the lane size fills the lane with the sign as one less does.
    const uint32_t esize   = 8U << size;
    const uint32_t clamped = amount < esize ? amount : esize - 1U;

    if (0U == clamped)
    {
        return;
    }

    if (1U == size || 2U == size)
    {
        emit_sse_immediate(assembler, 0x66, LANE_SHIFT[size], 4U, xmm, (uint8_t)clamped, "psra");
        return;
    }

    emit_lanes_shift_right(assembler, xmm, size, clamped, scratch);
    emit_broadcast_pattern(
        assembler, scratch, replicate_lane(1ULL << (esize - 1U - clamped), size));
    emit_sse(assembler, 0x66, false, 0xEF, xmm, scratch, "pxor");
    emit_sse(assembler, 0x66, false, LANE_SUB[size], xmm, scratch, "psub");
}

/// SSHR or USHR of the lanes of xmm0 by `amount` (1 to the lane size), and with `is_rounding`
/// SRSHR or URSHR, which add bit `amount` - 1 of the value. Uses xmm1 and xmm2.
static void
emit_shift_right_lanes(bal_x86_assembler_t *BAL_RESTRICT assembler,
                       const uint32_t                    size,
                       const uint32_t                    amount,
                       const bool                        is_signed,
                       const bool                        is_rounding)
{
    if (true == is_rounding)
    {
        emit_sse(assembler, 0U, false, 0x28, XMM1, XMM0, "movaps");
        emit_lanes_shift_right(assembler, XMM1, size, amount - 1U, XMM2);
        emit_broadcast_pattern(assembler, XMM2, lane_one_pattern(size));
        emit_sse(assembler, 0x66, false, 0xDB, XMM1, XMM2, "pand");
    }

    if (true == is_signed)
    {
        emit_lanes_shift_arithmetic(assembler, XMM0, size, amount, XMM2);
    }
    else
    {
        emit_lanes_shift_right(assembler, XMM0, size, amount, XMM2);
    }

    if (true == is_rounding)
    {
        emit_sse(assembler, 0x66, false, LANE_ADD[size], XMM0, XMM1, "padd");
    }
}

/// Keeps the low half of each `16 << narrow_size`-bit lane of `xmm`, packed into its low 64 bits
/// (SSE4.1). Uses `scratch`.
static void
emit_narrow(bal_x86_assembler_t *BAL_RESTRICT assembler,
            const uint8_t                     xmm,
            const uint32_t                    narrow_size,
            const uint8_t                     scratch)
{
    if (2U == narrow_size)
    {
        emit_sse_immediate(assembler, 0x66, 0x70, xmm, xmm, 0x08, "pshufd");
        return;
    }

    // Clearing the upper halves keeps the unsigned saturation of the packs from changing a lane.
    emit_all_ones(assembler, scratch);

    if (0U == narrow_size)
    {
        emit_sse_immediate(assembler, 0x66, 0x71, 2U, scratch, 8U, "psrlw");
        emit_sse(assembler, 0x66, false, 0xDB, xmm, scratch, "pand");
        emit_sse(assembler, 0x66, false, 0x67, xmm, xmm, "packuswb");
        return;
    }

    emit_sse_immediate(assembler, 0x66, 0x72, 2U, scratch, 16U, "psrld");
    emit_sse(assembler, 0x66, false, 0xDB, xmm, scratch, "pand");
    emit_sse38(assembler, 0x2B, xmm, xmm, "packusdw");
}

/// Stores the narrowed lanes in the low half of xmm0 to SIMD&FP register `rd`: as its low half
/// with the rest cleared, or for the "2" forms as its upper half, keeping the lower. Uses xmm3.
static void
emit_store_narrow(bal_x86_assembler_t *BAL_RESTRICT assembler,
                  const uint32_t                    rd,
                  const bool                        is_upper)
{
    if (false == is_upper)
    {
        emit_store_vector(assembler, XMM0, rd, false);
        return;
    }

    emit_load_vector(assembler, XMM3, rd, false);
    emit_sse(assembler, 0x66, false, 0x6C, XMM3, XMM0, "punpcklqdq");
    emit_store_vector(assembler, XMM3, rd, true);
}

/// Advanced SIMD shift by immediate, vector and scalar (doublewords): SSHR, USHR, SSRA, USRA,
/// SRSHR, URSHR, SRSRA, URSRA, SHL, SLI, SRI, SHRN, RSHRN, SSHLL and USHLL (with SXTL and UXTL).
/// None of them touches FPSR.
static bool
emit_fast_shift_immediate(bal_tier1_compiler_t *BAL_RESTRICT compiler, const uint32_t instruction)
{
    // 0 Q U 011110 immh immb opcode 1 Rn Rd, or the scalar 01 U 111110 immh immb opcode 1 Rn Rd;
    // immh is not zero.
    const uint32_t immh      = (instruction >> 19U) & 0xFU;
    const bool     is_scalar = 0x5F000400U == (instruction & 0xDF800400U);

    if ((0x0F000400U != (instruction & 0x9F800400U) && false == is_scalar) || 0U == immh
        || false == has_feature(compiler, BAL_TIER1_FEATURE_SSE41))
    {
        return false;
    }

    const bool     is_quad = false == is_scalar && 1U == ((instruction >> 30U) & 1U);
    const uint32_t u       = (instruction >> 29U) & 1U;
    const uint32_t opcode  = (instruction >> 11U) & 0x1FU;
    const uint32_t immhb   = (instruction >> 16U) & 0x7FU;
    const uint32_t size
        = 0U != (immh & 8U) ? 3U : (0U != (immh & 4U) ? 2U : (0U != (immh & 2U) ? 1U : 0U));
    const uint32_t esize = 8U << size;

    switch (opcode)
    {
        case 0x00U: // SSHR, USHR
        case 0x02U: // SSRA, USRA
        case 0x04U: // SRSHR, URSHR
        case 0x06U: // SRSRA, URSRA
        case 0x0AU: // SHL, SLI
            break;
        case 0x08U: // SRI
            if (0U == u)
            {
                return false;
            }
            break;
        case 0x10U: // SHRN
        case 0x11U: // RSHRN
        case 0x14U: // SSHLL, USHLL
            if (3U == size || true == is_scalar || (opcode != 0x14U && 1U == u))
            {
                return false;
            }
            break;
        default:
            return false;
    }

    // The scalar forms here exist on doublewords only, and doublewords need a full vector.
    if ((true == is_scalar && size != 3U) || (3U == size && false == is_quad && false == is_scalar))
    {
        return false;
    }

    const uint32_t                    rd        = instruction & 0x1FU;
    const uint32_t                    rn        = (instruction >> 5U) & 0x1FU;
    const uint32_t                    right     = 2U * esize - immhb;
    const uint32_t                    left      = immhb - esize;
    const uint64_t                    ones      = 64U == esize ? ~0ULL : (1ULL << esize) - 1U;
    bal_x86_assembler_t *BAL_RESTRICT assembler = &compiler->assembler;

    bal_sliding_window_flush_all(&compiler->window);

    switch (opcode)
    {
        case 0x00U:
        case 0x02U:
        case 0x04U:
        case 0x06U:
            emit_load_vector(assembler, XMM0, rn, is_quad);
            emit_shift_right_lanes(assembler, size, right, 0U == u, 0U != (opcode & 0x4U));

            if (0U != (opcode & 0x2U))
            {
                emit_load_vector(assembler, XMM1, rd, is_quad);
                emit_sse(assembler, 0x66, false, LANE_ADD[size], XMM0, XMM1, "padd");
            }

            emit_store_vector(assembler, XMM0, rd, is_quad);
            return true;
        case 0x08U:
        case 0x0AU: {
            // SRI and SLI keep the bits of Rd that the shifted value does not reach.
            const bool     is_right = 0x08U == opcode;
            const uint64_t reached
                = true == is_right ? (right >= esize ? 0U : ones >> right) : (ones << left) & ones;
            emit_load_vector(assembler, XMM0, rn, is_quad);

            if (true == is_right)
            {
                emit_lanes_shift_right(assembler, XMM0, size, right, XMM2);
            }
            else
            {
                emit_lanes_shift_left(assembler, XMM0, size, left, XMM2);
            }

            if (1U == u)
            {
                emit_load_vector(assembler, XMM1, rd, is_quad);
                emit_broadcast_pattern(assembler, XMM2, replicate_lane(reached, size));
                emit_sse(assembler, 0x66, false, 0xDF, XMM2, XMM1, "pandn");
                emit_sse(assembler, 0x66, false, 0xEB, XMM0, XMM2, "por");
            }

            emit_store_vector(assembler, XMM0, rd, is_quad);
            return true;
        }
        case 0x10U:
        case 0x11U:
            // The shift works on the wide lanes, whose low halves are kept.
            emit_load_vector(assembler, XMM0, rn, true);
            emit_shift_right_lanes(assembler, size + 1U, right, false, 0x11U == opcode);
            emit_narrow(assembler, XMM0, size, XMM1);
            emit_store_narrow(assembler, rd, is_quad);
            return true;
        default:
            emit_load_vector(assembler, XMM0, rn, true);
            emit_widen(assembler, XMM0, size, 1U == u, is_quad);
            emit_lanes_shift_left(assembler, XMM0, size + 1U, left, XMM2);
            emit_store_vector(assembler, XMM0, rd, true);
            return true;
    }
}

/// Narrows the doublewords of xmm0 to words in its low 64 bits with saturation as SQXTN (`key`
/// 0x14), UQXTN (0x34) or SQXTUN (0x32) does, clamping each to the word range (with the sign
/// bits flipped for UQXTN's unsigned comparison), and setting FPSR.QC when a lane saturates.
/// Uses xmm1 to xmm5 (SSE4.2).
static void
emit_doubleword_saturating_narrow(bal_x86_assembler_t *BAL_RESTRICT assembler, const uint32_t key)
{
    const bool     is_unsigned_source = 0x34U == key;
    const uint64_t highest            = 0x14U == key ? 0x7FFFFFFFULL : 0xFFFFFFFFULL;
    const uint64_t lowest             = 0x14U == key ? 0xFFFFFFFF80000000ULL : 0U;
    const uint64_t bias               = true == is_unsigned_source ? 1ULL << 63U : 0U;

    // XMM1 = the lanes above the range, XMM4 = those below it.
    emit_sse(assembler, 0U, false, 0x28, XMM1, XMM0, "movaps");
    emit_broadcast_pattern(assembler, XMM2, highest ^ bias);

    if (true == is_unsigned_source)
    {
        emit_broadcast_pattern(assembler, XMM3, bias);
        emit_sse(assembler, 0x66, false, 0xEF, XMM1, XMM3, "pxor");
    }

    emit_lanes_greater(assembler, 3U, XMM1, XMM2);
    emit_zero_vector(assembler, XMM4);

    if (false == is_unsigned_source)
    {
        emit_broadcast_pattern(assembler, XMM4, lowest);
        emit_lanes_greater(assembler, 3U, XMM4, XMM0);
    }

    emit_sse(assembler, 0U, false, 0x28, XMM5, XMM1, "movaps");
    emit_sse(assembler, 0x66, false, 0xEB, XMM5, XMM4, "por");
    emit_saturation_if_any(assembler, XMM5);
    emit_broadcast_pattern(assembler, XMM2, highest);
    emit_sse(assembler, 0x66, false, 0xDB, XMM2, XMM1, "pand");
    emit_broadcast_pattern(assembler, XMM3, lowest);
    emit_sse(assembler, 0x66, false, 0xDB, XMM3, XMM4, "pand");
    emit_sse(assembler, 0x66, false, 0xDF, XMM5, XMM0, "pandn");
    emit_sse(assembler, 0x66, false, 0xEB, XMM5, XMM2, "por");
    emit_sse(assembler, 0x66, false, 0xEB, XMM5, XMM3, "por");
    emit_sse_immediate(assembler, 0x66, 0x70, XMM0, XMM5, 0x08, "pshufd");
}

/// Narrows the `16 << size`-bit lanes of xmm0 into its low 64 bits with saturation as SQXTN (`key`
/// 0x14), UQXTN (0x34) or SQXTUN (0x32) does, setting FPSR.QC when a lane saturates. Uses xmm1 and
/// xmm2, and for doublewords (`size` 2) emit_doubleword_saturating_narrow().
static void
emit_saturating_narrow(bal_x86_assembler_t *BAL_RESTRICT assembler,
                       const uint32_t                    size,
                       const uint32_t                    key)
{
    // The packs saturate as these do: PACKSS for SQXTN, PACKUS (signed to unsigned) for
    // SQXTUN, and for UQXTN PACKUS after PMINU clamps each unsigned lane. A lane
    // saturated where extending the result back does not give the source.
    if (2U == size)
    {
        emit_doubleword_saturating_narrow(assembler, key);
        return;
    }

    const bool is_unsigned_result = key != 0x14U;
    emit_sse(assembler, 0U, false, 0x28, XMM1, XMM0, "movaps");

    if (0x34U == key)
    {
        emit_broadcast_pattern(
            assembler, XMM2, 0U == size ? 0x00FF00FF00FF00FFULL : 0x0000FFFF0000FFFFULL);
        emit_sse38(assembler, 0U == size ? 0x3A : 0x3B, XMM0, XMM2, "pminuw/pminud");
    }

    if (0x14U == key)
    {
        emit_sse(assembler, 0x66, false, 0U == size ? 0x63 : 0x6B, XMM0, XMM0, "packss");
    }
    else if (0U == size)
    {
        emit_sse(assembler, 0x66, false, 0x67, XMM0, XMM0, "packuswb");
    }
    else
    {
        emit_sse38(assembler, 0x2B, XMM0, XMM0, "packusdw");
    }

    emit_sse(assembler, 0U, false, 0x28, XMM2, XMM0, "movaps");
    emit_widen(assembler, XMM2, size, is_unsigned_result, false);
    emit_saturation_unless_equal(assembler, XMM2, XMM1);
}

/// More of Advanced SIMD two-register miscellaneous: XTN2, CNT, RBIT, SHLL and SHLL2, SADDLP,
/// UADDLP, SADALP and UADALP, and SQXTN, UQXTN and SQXTUN with their "2" forms, which set FPSR.QC
/// when a lane saturates.
static bool
emit_fast_vector_misc_more(bal_tier1_compiler_t *BAL_RESTRICT compiler, const uint32_t instruction)
{
    // 0 Q U 01110 size 10000 opcode 10 Rn Rd.
    if (0x0E200800U != (instruction & 0x9F3E0C00U)
        || false == has_feature(compiler, BAL_TIER1_FEATURE_SSE41))
    {
        return false;
    }

    const bool     is_quad = 1U == ((instruction >> 30U) & 1U);
    const uint32_t u       = (instruction >> 29U) & 1U;
    const uint32_t size    = (instruction >> 22U) & 0x3U;
    const uint32_t key     = (u << 5U) | ((instruction >> 12U) & 0x1FU);

    switch (key)
    {
        case 0x12U: // XTN2 (XTN is emit_fast_vector_misc())
            if (false == is_quad || 3U == size)
            {
                return false;
            }
            break;
        case 0x05U: // CNT
            if (size != 0U)
            {
                return false;
            }
            break;
        case 0x25U: // RBIT (NOT is size 0)
            if (size != 1U)
            {
                return false;
            }
            break;
        case 0x33U: // SHLL
        case 0x02U: // SADDLP
        case 0x22U: // UADDLP
        case 0x06U: // SADALP
        case 0x26U: // UADALP
            if (3U == size)
            {
                return false;
            }
            break;
        case 0x14U: // SQXTN
        case 0x34U: // UQXTN
        case 0x32U: // SQXTUN
            if (size > 2U
                || (2U == size && false == has_feature(compiler, BAL_TIER1_FEATURE_SSE42)))
            {
                return false;
            }
            break;
        default:
            return false;
    }

    const uint32_t                    rd        = instruction & 0x1FU;
    const uint32_t                    rn        = (instruction >> 5U) & 0x1FU;
    bal_x86_assembler_t *BAL_RESTRICT assembler = &compiler->assembler;
    bal_sliding_window_flush_all(&compiler->window);

    switch (key)
    {
        case 0x12U:
            emit_load_vector(assembler, XMM0, rn, true);
            emit_narrow(assembler, XMM0, size, XMM1);
            emit_store_narrow(assembler, rd, true);
            return true;
        case 0x05U:
        case 0x25U: {
            // Lookups of each nibble with PSHUFB: its bit count, or its bits reversed.
            const bool is_count = 0x05U == key;
            emit_load_vector(assembler, XMM0, rn, is_quad);
            emit_sse(assembler, 0U, false, 0x28, XMM1, XMM0, "movaps");
            emit_sse_immediate(assembler, 0x66, 0x71, 2U, XMM1, 4U, "psrlw");
            emit_broadcast_pattern(assembler, XMM2, 0x0F0F0F0F0F0F0F0FULL);
            emit_sse(assembler, 0x66, false, 0xDB, XMM0, XMM2, "pand");
            emit_sse(assembler, 0x66, false, 0xDB, XMM1, XMM2, "pand");
            emit_load_vector_constant(
                assembler,
                XMM3,
                true == is_count ? 0x0302020102010100ULL : 0x0E060A020C040800ULL,
                true == is_count ? 0x0403030203020201ULL : 0x0F070B030D050901ULL);
            emit_sse(assembler, 0U, false, 0x28, XMM2, XMM3, "movaps");
            emit_sse38(assembler, 0x00, XMM3, XMM0, "pshufb");
            emit_sse38(assembler, 0x00, XMM2, XMM1, "pshufb");

            if (true == is_count)
            {
                emit_sse(assembler, 0x66, false, 0xFC, XMM3, XMM2, "paddb");
            }
            else
            {
                // The reversed low nibble becomes the high one; no bit crosses into the next
                // byte because the lookups are below 16.
                emit_sse_immediate(assembler, 0x66, 0x71, 6U, XMM3, 4U, "psllw");
                emit_sse(assembler, 0x66, false, 0xEB, XMM3, XMM2, "por");
            }

            emit_store_vector(assembler, XMM3, rd, is_quad);
            return true;
        }
        case 0x33U:
            emit_load_vector(assembler, XMM0, rn, true);
            emit_widen(assembler, XMM0, size, true, is_quad);
            emit_lanes_shift_left(assembler, XMM0, size + 1U, 8U << size, XMM1);
            emit_store_vector(assembler, XMM0, rd, true);
            return true;
        case 0x02U:
        case 0x22U:
        case 0x06U:
        case 0x26U:
            emit_load_vector(assembler, XMM0, rn, is_quad);

            if (0U == size)
            {
                // PMADDUBSW multiplies unsigned bytes of its first operand by signed bytes of the
                // second and adds the pairs; with ones the sums cannot saturate.
                emit_broadcast_pattern(assembler, XMM1, 0x0101010101010101ULL);

                if (1U == u)
                {
                    emit_sse38(assembler, 0x04, XMM0, XMM1, "pmaddubsw");
                }
                else
                {
                    emit_sse38(assembler, 0x04, XMM1, XMM0, "pmaddubsw");
                    emit_sse(assembler, 0U, false, 0x28, XMM0, XMM1, "movaps");
                }
            }
            else if (1U == size)
            {
                // PMADDWD adds signed pairs; unsigned ones are biased by -32768 each and
                // corrected by 65536.
                emit_broadcast_pattern(assembler, XMM1, 0x0001000100010001ULL);

                if (1U == u)
                {
                    emit_broadcast_pattern(assembler, XMM2, 0x8000800080008000ULL);
                    emit_sse(assembler, 0x66, false, 0xEF, XMM0, XMM2, "pxor");
                }

                emit_sse(assembler, 0x66, false, 0xF5, XMM0, XMM1, "pmaddwd");

                if (1U == u)
                {
                    emit_broadcast_pattern(assembler, XMM2, 0x0001000000010000ULL);
                    emit_sse(assembler, 0x66, false, 0xFE, XMM0, XMM2, "paddd");
                }
            }
            else
            {
                // Words 0 and 2 and words 1 and 3, extended to doublewords and added.
                emit_sse_immediate(assembler, 0x66, 0x70, XMM0, XMM0, 0xD8, "pshufd");
                emit_sse(assembler, 0U, false, 0x28, XMM1, XMM0, "movaps");
                emit_widen(assembler, XMM0, 2U, 1U == u, false);
                emit_widen(assembler, XMM1, 2U, 1U == u, true);
                emit_sse(assembler, 0x66, false, 0xD4, XMM0, XMM1, "paddq");
            }

            if (0x06U == (key & 0x1FU))
            {
                emit_load_vector(assembler, XMM1, rd, is_quad);
                emit_sse(assembler, 0x66, false, LANE_ADD[size + 1U], XMM0, XMM1, "padd");
            }

            emit_store_vector(assembler, XMM0, rd, is_quad);
            return true;
        default:
            emit_load_vector(assembler, XMM0, rn, true);
            emit_saturating_narrow(assembler, size, key);
            emit_store_narrow(assembler, rd, is_quad);
            return true;
    }
}

/// Advanced SIMD across lanes: ADDV, SMAXV, SMINV, UMAXV and UMINV, which fold the upper half of
/// the vector onto the lower until one lane is left, and SADDLV and UADDLV, which PSADBW (bytes),
/// PMADDWD (halfwords) or widening (words) sum.
static bool
emit_fast_across_lanes(bal_tier1_compiler_t *BAL_RESTRICT compiler, const uint32_t instruction)
{
    // 0 Q U 01110 size 11000 opcode 10 Rn Rd.
    if (0x0E300800U != (instruction & 0x9F3E0C00U)
        || false == has_feature(compiler, BAL_TIER1_FEATURE_SSE41))
    {
        return false;
    }

    const bool     is_quad = 1U == ((instruction >> 30U) & 1U);
    const uint32_t u       = (instruction >> 29U) & 1U;
    const uint32_t size    = (instruction >> 22U) & 0x3U;
    const uint32_t opcode  = (instruction >> 12U) & 0x1FU;
    const bool     is_add  = 0x1BU == opcode && 0U == u;
    const bool     is_max  = 0x0AU == opcode;
    const bool     is_min  = 0x1AU == opcode;
    const bool     is_long = 0x03U == opcode;

    if ((false == is_add && false == is_max && false == is_min && false == is_long) || 3U == size
        || (2U == size && false == is_quad))
    {
        return false;
    }

    const uint32_t                    rd        = instruction & 0x1FU;
    const uint32_t                    rn        = (instruction >> 5U) & 0x1FU;
    const uint32_t                    lanes     = (true == is_quad ? 16U : 8U) >> size;
    const uint32_t                    result    = true == is_long ? size + 1U : size;
    bal_x86_assembler_t *BAL_RESTRICT assembler = &compiler->assembler;

    bal_sliding_window_flush_all(&compiler->window);
    emit_load_vector(assembler, XMM0, rn, is_quad);

    if (false == is_long)
    {
        for (uint32_t half = (true == is_quad ? 16U : 8U) / 2U; half >= (1U << size); half /= 2U)
        {
            emit_sse(assembler, 0U, false, 0x28, XMM1, XMM0, "movaps");
            emit_sse_immediate(assembler, 0x66, 0x73, 3U, XMM1, (uint8_t)half, "psrldq");

            if (true == is_add)
            {
                emit_sse(assembler, 0x66, false, LANE_ADD[size], XMM0, XMM1, "padd");
            }
            else
            {
                emit_lane_extreme(assembler, is_min, u, size, XMM0, XMM1);
            }
        }
    }
    else if (0U == size)
    {
        // Signed bytes are biased by 128 each, taken off the total.
        if (0U == u)
        {
            emit_broadcast_pattern(assembler, XMM1, 0x8080808080808080ULL);
            emit_sse(assembler, 0x66, false, 0xEF, XMM0, XMM1, "pxor");
        }

        // PSADBW sums each half; a 64-bit vector's upper half is not part of it.
        emit_zero_vector(assembler, XMM1);
        emit_sse(assembler, 0x66, false, 0xF6, XMM0, XMM1, "psadbw");

        if (true == is_quad)
        {
            emit_sse_immediate(assembler, 0x66, 0x70, XMM1, XMM0, 0xEE, "pshufd");
            emit_sse(assembler, 0x66, false, 0xD4, XMM0, XMM1, "paddq");
        }
    }
    else if (1U == size)
    {
        emit_broadcast_pattern(assembler, XMM1, 0x0001000100010001ULL);

        if (1U == u)
        {
            emit_broadcast_pattern(assembler, XMM2, 0x8000800080008000ULL);
            emit_sse(assembler, 0x66, false, 0xEF, XMM0, XMM2, "pxor");
        }

        emit_sse(assembler, 0x66, false, 0xF5, XMM0, XMM1, "pmaddwd");

        if (1U == u)
        {
            emit_broadcast_pattern(assembler, XMM2, 0x0001000000010000ULL);
            emit_sse(assembler, 0x66, false, 0xFE, XMM0, XMM2, "paddd");
        }

        for (uint32_t half = (true == is_quad ? 16U : 8U) / 2U; half >= 4U; half /= 2U)
        {
            emit_sse(assembler, 0U, false, 0x28, XMM1, XMM0, "movaps");
            emit_sse_immediate(assembler, 0x66, 0x73, 3U, XMM1, (uint8_t)half, "psrldq");
            emit_sse(assembler, 0x66, false, 0xFE, XMM0, XMM1, "paddd");
        }
    }
    else
    {
        emit_sse(assembler, 0U, false, 0x28, XMM1, XMM0, "movaps");
        emit_widen(assembler, XMM0, 2U, 1U == u, false);
        emit_widen(assembler, XMM1, 2U, 1U == u, true);
        emit_sse(assembler, 0x66, false, 0xD4, XMM0, XMM1, "paddq");
        emit_sse_immediate(assembler, 0x66, 0x70, XMM1, XMM0, 0xEE, "pshufd");
        emit_sse(assembler, 0x66, false, 0xD4, XMM0, XMM1, "paddq");
    }

    // The result is lane 0, zero-extended.
    emit_sse(assembler, 0x66, 3U == result, 0x7E, XMM0, (uint8_t)SCRATCH_A, "movd/movq");

    if (true == is_long && 0U == size && 0U == u)
    {
        bal_x86_emit_sub_r32_imm32(assembler, SCRATCH_A, (int32_t)(128U * lanes));
    }

    if (result < 2U)
    {
        const uint8_t shift = 0U == result ? 24U : 16U;
        bal_x86_emit_shl_r32_imm8(assembler, SCRATCH_A, shift);
        bal_x86_emit_shr_r32_imm8(assembler, SCRATCH_A, shift);
    }

    emit_store_scalar_result(assembler, rd);
    return true;
}

/// FADDP (vector and scalar) in single and double precision. HADDPS and HADDPD add the pairs in
/// the order FADDP does, each sum rounded once, and the results are checked like FADD's.
static bool
emit_fast_float_add_pair(bal_tier1_compiler_t *BAL_RESTRICT compiler, const uint32_t instruction)
{
    // Vector: 0 Q 1 01110 0 sz 1 Rm 11010 1 Rn Rd. Scalar: 01 1 11110 0 sz 11000 01101 10 Rn Rd.
    const bool is_scalar = 0x7E30D800U == (instruction & 0xFFBFFC00U);
    const bool is_double = 1U == ((instruction >> 22U) & 1U);
    const bool is_quad   = true == is_scalar || 1U == ((instruction >> 30U) & 1U);

    if ((0x2E20D400U != (instruction & 0xBFA0FC00U) && false == is_scalar)
        || (true == is_double && false == is_quad))
    {
        return false;
    }

    const uint32_t                    rd        = instruction & 0x1FU;
    const uint32_t                    rn        = (instruction >> 5U) & 0x1FU;
    const uint32_t                    rm        = (instruction >> 16U) & 0x1FU;
    const uint8_t                     prefix    = true == is_double ? 0x66 : 0xF2;
    bal_x86_assembler_t *BAL_RESTRICT assembler = &compiler->assembler;
    size_t                            slow_jumps[3];

    slow_jumps[0] = begin_float_sequence(compiler);

    if (true == is_scalar)
    {
        // The two elements of the 64-bit (single) or 128-bit (double) source.
        emit_load_vector(assembler, XMM0, rn, is_double);
        emit_sse(assembler, prefix, false, 0x7C, XMM0, XMM0, "haddp");
        const uint32_t count = 1U + emit_scalar_checks(compiler, is_double, &slow_jumps[1]);
        emit_store_scalar_result(assembler, rd);
        emit_slow_path(
            compiler, instruction, (uint64_t)(uintptr_t)&bal_execute_simd, slow_jumps, count);
        return true;
    }

    emit_load_vector(assembler, XMM0, rn, is_quad);
    emit_load_vector(assembler, XMM1, rm, is_quad);

    if (false == is_quad)
    {
        emit_sse(assembler, 0x66, false, 0x6C, XMM0, XMM1, "punpcklqdq");
        emit_sse(assembler, prefix, false, 0x7C, XMM0, XMM0, "haddps");
    }
    else
    {
        emit_sse(assembler, prefix, false, 0x7C, XMM0, XMM1, "haddp");
    }

    const uint32_t count = 1U + emit_vector_checks(assembler, is_double, &slow_jumps[1]);
    emit_store_vector(assembler, XMM0, rd, is_quad);
    emit_slow_path(
        compiler, instruction, (uint64_t)(uintptr_t)&bal_execute_simd, slow_jumps, count);
    return true;
}

/// FMAXP, FMINP, FMAXNMP and FMINNMP (vector and scalar) and FMAXV, FMINV, FMAXNMV and FMINNMV
/// (4S): the pairs are split into two vectors and combined with emit_float_extreme(). Without
/// NaNs, which take the slow path, the NM forms equal the others and the order of the
/// reductions does not matter.
static bool
emit_fast_float_extreme_pair(bal_tier1_compiler_t *BAL_RESTRICT compiler,
                             const uint32_t                     instruction)
{
    // Vector: 0 Q 1 01110 o1 sz 1 Rm 11110 1 (FMAXP, FMINP) or 11000 1 (FMAXNMP, FMINNMP) Rn Rd.
    // Scalar: 01 1 11110 o1 sz 11000 01111 10 or 01100 10 Rn Rd.
    // Across lanes: 0 1 1 01110 o1 0 11000 01111 10 or 01100 10 Rn Rd.
    const uint32_t vector_form = instruction & 0xBF20FC00U;
    const uint32_t other_form  = instruction & 0xFF3FFC00U;
    const bool     is_vector   = 0x2E20F400U == vector_form || 0x2E20C400U == vector_form;
    const bool     is_scalar   = 0x7E30F800U == other_form || 0x7E30C800U == other_form;
    const bool     is_across   = 0x6E30F800U == other_form || 0x6E30C800U == other_form;
    const bool     is_double   = 1U == ((instruction >> 22U) & 1U);
    const bool     is_minimum  = 1U == ((instruction >> 23U) & 1U);
    const bool     is_quad     = false == is_vector || 1U == ((instruction >> 30U) & 1U);

    if ((false == is_vector && false == is_scalar && false == is_across)
        || (true == is_double && (false == is_quad || true == is_across)))
    {
        return false;
    }

    const uint32_t                    rd        = instruction & 0x1FU;
    const uint32_t                    rn        = (instruction >> 5U) & 0x1FU;
    const uint32_t                    rm        = (instruction >> 16U) & 0x1FU;
    const uint8_t                     packed    = true == is_double ? 0x66 : 0U;
    const uint8_t                     scalar    = true == is_double ? 0xF2 : 0xF3;
    bal_x86_assembler_t *BAL_RESTRICT assembler = &compiler->assembler;
    size_t                            slow_jumps[3];
    uint32_t                          count = 1U;

    slow_jumps[0] = begin_float_sequence(compiler);

    if (true == is_vector)
    {
        emit_load_vector(assembler, XMM0, rn, is_quad);
        emit_load_vector(assembler, XMM1, rm, is_quad);

        if (true == is_double)
        {
            emit_sse(assembler, 0U, false, 0x28, XMM2, XMM0, "movaps");
            emit_sse(assembler, 0x66, false, 0x14, XMM0, XMM1, "unpcklpd");
            emit_sse(assembler, 0x66, false, 0x15, XMM2, XMM1, "unpckhpd");
            emit_sse(assembler, 0U, false, 0x28, XMM1, XMM2, "movaps");
        }
        else if (true == is_quad)
        {
            emit_sse(assembler, 0U, false, 0x28, XMM2, XMM0, "movaps");
            emit_sse_immediate(assembler, 0U, 0xC6, XMM0, XMM1, 0x88, "shufps");
            emit_sse_immediate(assembler, 0U, 0xC6, XMM2, XMM1, 0xDD, "shufps");
            emit_sse(assembler, 0U, false, 0x28, XMM1, XMM2, "movaps");
        }
        else
        {
            // n0 n1 m0 m1 becomes n0 m0 and n1 m1, the upper lanes repeating lane 0.
            emit_sse(assembler, 0x66, false, 0x6C, XMM0, XMM1, "punpcklqdq");
            emit_sse_immediate(assembler, 0x66, 0x70, XMM1, XMM0, 0x0D, "pshufd");
            emit_sse_immediate(assembler, 0x66, 0x70, XMM0, XMM0, 0x08, "pshufd");
        }

        slow_jumps[count++] = emit_float_extreme(assembler, packed, packed, false, is_minimum);
        emit_store_vector(assembler, XMM0, rd, is_quad);
        emit_slow_path(
            compiler, instruction, (uint64_t)(uintptr_t)&bal_execute_simd, slow_jumps, count);
        return true;
    }

    if (true == is_across)
    {
        // Lanes 0 and 1 against 2 and 3, then the two results.
        emit_load_vector(assembler, XMM0, rn, true);
        emit_sse_immediate(assembler, 0x66, 0x70, XMM1, XMM0, 0xEE, "pshufd");
        slow_jumps[count++] = emit_float_extreme(assembler, 0U, 0U, false, is_minimum);
        emit_sse_immediate(assembler, 0x66, 0x70, XMM1, XMM0, 0x55, "pshufd");
    }
    else
    {
        emit_load_vector(assembler, XMM0, rn, is_double);
        emit_sse_immediate(
            assembler, 0x66, 0x70, XMM1, XMM0, true == is_double ? 0xEE : 0x55, "pshufd");
    }

    slow_jumps[count++] = emit_float_extreme(assembler, scalar, packed, true, is_minimum);
    emit_sse(assembler, 0x66, is_double, 0x7E, XMM0, (uint8_t)SCRATCH_A, "movd/movq");
    emit_store_scalar_result(assembler, rd);
    emit_slow_path(
        compiler, instruction, (uint64_t)(uintptr_t)&bal_execute_simd, slow_jumps, count);
    return true;
}

// -----------------------------------------------------------------------------
// Inline Advanced SIMD structure loads and stores
// -----------------------------------------------------------------------------

/// Loads the 128 bits at [base + offset] into `xmm`, or with `is_partial` the 64 bits, clearing
/// the rest.
static void
emit_load_memory_vector(bal_x86_assembler_t *BAL_RESTRICT assembler,
                        const uint8_t                     xmm,
                        const bal_x86_register_t          base,
                        const int32_t                     offset,
                        const bool                        is_partial)
{
    static const uint8_t MOVUPS[2] = { 0x0F, 0x10 };
    static const uint8_t MOVQ[2]   = { 0x0F, 0x7E };
    bal_x86_emit_sse_base_offset(assembler,
                                 true == is_partial ? 0xF3 : 0U,
                                 false,
                                 true == is_partial ? MOVQ : MOVUPS,
                                 2U,
                                 xmm,
                                 base,
                                 offset,
                                 "movups/movq");
}

/// Stores `xmm` to [base + offset], all 128 bits or with `is_partial` the low 64.
static void
emit_store_memory_vector(bal_x86_assembler_t *BAL_RESTRICT assembler,
                         const uint8_t                     xmm,
                         const bal_x86_register_t          base,
                         const int32_t                     offset,
                         const bool                        is_partial)
{
    static const uint8_t MOVUPS[2] = { 0x0F, 0x11 };
    static const uint8_t MOVQ[2]   = { 0x0F, 0xD6 };
    bal_x86_emit_sse_base_offset(assembler,
                                 true == is_partial ? 0x66 : 0U,
                                 false,
                                 true == is_partial ? MOVQ : MOVUPS,
                                 2U,
                                 xmm,
                                 base,
                                 offset,
                                 "movups/movq");
}

/// Gathers bytes into xmm0: byte j is byte `map[j]` % 16 of source `map[j]` / 16, or zero where
/// `map[j]` is 0xFF. Source s is the 128 bits at [base + offsets[s]], or the 64 bits there when
/// bit s of `partial` is set. Each source used is shuffled into place with PSHUFB, unless it is
/// already in place, and ORed in. Uses xmm1 and xmm2 (SSSE3).
static void
emit_byte_gather(bal_x86_assembler_t *BAL_RESTRICT assembler,
                 const bal_x86_register_t          base,
                 const int32_t                    *offsets,
                 const uint32_t                    partial,
                 const uint32_t                    count,
                 const uint8_t                    *map)
{
    bool is_first = true;

    for (uint32_t source = 0U; source < count; ++source)
    {
        uint64_t halves[2]   = { 0U, 0U };
        bool     is_used     = false;
        bool     is_in_place = true;

        for (uint32_t byte = 0U; byte < 16U; ++byte)
        {
            uint64_t selector = 0x80U;

            if (map[byte] != 0xFFU && (uint32_t)map[byte] / 16U == source)
            {
                selector = map[byte] % 16U;
                is_used  = true;
            }

            is_in_place = is_in_place && selector == byte;
            halves[byte / 8U] |= selector << (8U * (byte % 8U));
        }

        if (false == is_used)
        {
            continue;
        }

        const uint8_t destination = true == is_first ? XMM0 : XMM2;
        emit_load_memory_vector(
            assembler, destination, base, offsets[source], 0U != ((partial >> source) & 1U));

        if (false == is_in_place)
        {
            emit_load_vector_constant(assembler, XMM1, halves[0], halves[1]);
            emit_sse38(assembler, 0x00, destination, XMM1, "pshufb");
        }

        if (false == is_first)
        {
            emit_sse(assembler, 0x66, false, 0xEB, XMM0, XMM2, "por");
        }

        is_first = false;
    }

    if (true == is_first)
    {
        emit_zero_vector(assembler, XMM0);
    }
}

/// LD2, LD3 and LD4 and ST2, ST3 and ST4 (multiple structures), without an offset or
/// post-indexed: element e of register r is memory element e * count + r. LD2 and ST2 on words
/// and doublewords use SHUFPS and the unpacks; the others gather each register (loads) or each
/// 16 bytes of memory (stores) with emit_byte_gather().
static bool
emit_fast_vector_interleaved(bal_tier1_compiler_t *BAL_RESTRICT         compiler,
                             const bal_memory_interface_t *BAL_RESTRICT memory_interface,
                             const uint32_t                             instruction,
                             const bal_guest_address_t                  guest_address)
{
    // 0 Q 0011000 L 000000 opcode size Rn Rt, or post-indexed 0 Q 0011001 L 0 Rm opcode size Rn Rt.
    const bool     is_plain = 0x0C000000U == (instruction & 0xBFBF0000U);
    const bool     is_post  = 0x0C800000U == (instruction & 0xBFA00000U);
    const bool     is_quad  = 1U == ((instruction >> 30U) & 1U);
    const uint32_t size     = (instruction >> 10U) & 0x3U;
    uint32_t       count    = 0U;

    switch ((instruction >> 12U) & 0xFU)
    {
        case 0x8U:
            count = 2U;
            break;
        case 0x4U:
            count = 3U;
            break;
        case 0x0U:
            count = 4U;
            break;
        default:
            return false;
    }

    if ((false == is_plain && false == is_post)
        || false == has_feature(compiler, BAL_TIER1_FEATURE_SSE41)
        || (3U == size && false == is_quad))
    {
        return false;
    }

    const bool     is_load      = 1U == ((instruction >> 22U) & 1U);
    const uint8_t  rt           = (uint8_t)(instruction & 0x1FU);
    const uint8_t  rn           = (uint8_t)((instruction >> 5U) & 0x1FU);
    const uint8_t  rm           = (uint8_t)((instruction >> 16U) & 0x1FU);
    const uint32_t element      = 1U << size;
    const uint32_t vector_bytes = true == is_quad ? 16U : 8U;
    const uint32_t total        = count * vector_bytes;
    const uint32_t chunks       = (total + 15U) / 16U;

    emit_base_plus_offset(compiler, rn, 0);
    emit_translate_data_address(compiler, memory_interface, total, guest_address);

    bal_x86_assembler_t *BAL_RESTRICT assembler = &compiler->assembler;
    const bal_x86_register_t          pointer   = BAL_X86_REGISTER_ZERO;
    bal_sliding_window_flush_all(&compiler->window);

    if (2U == count && true == is_quad && size >= 2U)
    {
        const uint32_t second = (rt + 1U) % 32U;

        if (true == is_load)
        {
            emit_load_memory_vector(assembler, XMM0, pointer, 0, false);
            emit_load_memory_vector(assembler, XMM1, pointer, 16, false);
            emit_sse(assembler, 0U, false, 0x28, XMM2, XMM0, "movaps");

            if (2U == size)
            {
                emit_sse_immediate(assembler, 0U, 0xC6, XMM0, XMM1, 0x88, "shufps");
                emit_sse_immediate(assembler, 0U, 0xC6, XMM2, XMM1, 0xDD, "shufps");
            }
            else
            {
                emit_sse(assembler, 0x66, false, 0x6C, XMM0, XMM1, "punpcklqdq");
                emit_sse(assembler, 0x66, false, 0x6D, XMM2, XMM1, "punpckhqdq");
            }

            emit_store_vector(assembler, XMM0, rt, true);
            emit_store_vector(assembler, XMM2, second, true);
        }
        else
        {
            emit_load_vector(assembler, XMM0, rt, true);
            emit_load_vector(assembler, XMM1, second, true);
            emit_sse(assembler, 0U, false, 0x28, XMM2, XMM0, "movaps");
            emit_sse(assembler, 0x66, false, 2U == size ? 0x62 : 0x6C, XMM0, XMM1, "punpckl");
            emit_sse(assembler, 0x66, false, 2U == size ? 0x6A : 0x6D, XMM2, XMM1, "punpckh");
            emit_store_memory_vector(assembler, XMM0, pointer, 0, false);
            emit_store_memory_vector(assembler, XMM2, pointer, 16, false);
        }
    }
    else if (true == is_load)
    {
        // Memory byte m lies in chunk m / 16 at m % 16, so the map entry is m itself.
        const int32_t  offsets[4] = { 0, 16, 32, 48 };
        const uint32_t partial    = 0U != (total % 16U) ? 1U << (chunks - 1U) : 0U;

        for (uint32_t r = 0U; r < count; ++r)
        {
            uint8_t map[16];

            for (uint32_t byte = 0U; byte < 16U; ++byte)
            {
                const uint32_t memory_element = ((byte / element) * count) + r;

                map[byte] = byte < vector_bytes
                                ? (uint8_t)((memory_element * element) + (byte % element))
                                : 0xFFU;
            }

            emit_byte_gather(assembler, pointer, offsets, partial, chunks, map);
            emit_store_vector(assembler, XMM0, (rt + r) % 32U, true);
        }
    }
    else
    {
        int32_t offsets[4];

        for (uint32_t r = 0U; r < count; ++r)
        {
            offsets[r] = (int32_t)offsetof(bal_cpu_t, v[(rt + r) % 32U][0]);
        }

        for (uint32_t chunk = 0U; chunk < chunks; ++chunk)
        {
            uint8_t map[16];

            for (uint32_t byte = 0U; byte < 16U; ++byte)
            {
                const uint32_t memory_byte    = (chunk * 16U) + byte;
                const uint32_t memory_element = memory_byte / element;
                const uint32_t r              = memory_element % count;
                const uint32_t e              = memory_element / count;

                map[byte] = memory_byte < total
                                ? (uint8_t)((r * 16U) + (e * element) + (memory_byte % element))
                                : 0xFFU;
            }

            emit_byte_gather(assembler, BAL_X86_RBP, offsets, 0U, count, map);
            emit_store_memory_vector(
                assembler, XMM0, pointer, (int32_t)(chunk * 16U), total - (chunk * 16U) < 16U);
        }
    }

    if (true == is_post)
    {
        emit_structure_writeback(compiler, rn, rm, total);
    }

    return true;
}

/// LD1R, and LD1 and ST1 (single structure) of one lane, without an offset or post-indexed.
/// A lane moves through BAL_X86_REGISTER_DISCARD_RESULT between the guest memory and the lane's
/// place in the cpu structure; LD1R broadcasts the element in an SSE register.
static bool
emit_fast_vector_single(bal_tier1_compiler_t *BAL_RESTRICT         compiler,
                        const bal_memory_interface_t *BAL_RESTRICT memory_interface,
                        const uint32_t                             instruction,
                        const bal_guest_address_t                  guest_address)
{
    // 0 Q 0011010 L R 00000 opcode S size Rn Rt, or post-indexed 0 Q 0011011 L R Rm opcode S size
    // Rn Rt. R and the low opcode bit select LD2 to LD4.
    const bool     is_plain = 0x0D000000U == (instruction & 0xBF9F0000U);
    const bool     is_post  = 0x0D800000U == (instruction & 0xBF800000U);
    const uint32_t q        = (instruction >> 30U) & 1U;
    const uint32_t s        = (instruction >> 12U) & 1U;
    const uint32_t size     = (instruction >> 10U) & 0x3U;
    const bool     is_load  = 1U == ((instruction >> 22U) & 1U);

    if ((false == is_plain && false == is_post) || 1U == ((instruction >> 21U) & 1U)
        || false == has_feature(compiler, BAL_TIER1_FEATURE_SSE41))
    {
        return false;
    }

    bool     is_replicate = false;
    uint32_t log2         = 0U;
    uint32_t index        = 0U;

    switch ((instruction >> 13U) & 0x7U)
    {
        case 0x0U:
            index = (q << 3U) | (s << 2U) | size;
            break;
        case 0x2U:
            if (1U == (size & 1U))
            {
                return false;
            }

            log2  = 1U;
            index = (q << 2U) | (s << 1U) | (size >> 1U);
            break;
        case 0x4U:
            if (0U == size)
            {
                log2  = 2U;
                index = (q << 1U) | s;
            }
            else if (1U == size && 0U == s)
            {
                log2  = 3U;
                index = q;
            }
            else
            {
                return false;
            }
            break;
        case 0x6U:
            if (false == is_load || 1U == s)
            {
                return false;
            }

            is_replicate = true;
            log2         = size;
            break;
        default:
            return false;
    }

    const uint8_t            rt      = (uint8_t)(instruction & 0x1FU);
    const uint8_t            rn      = (uint8_t)((instruction >> 5U) & 0x1FU);
    const uint8_t            rm      = (uint8_t)((instruction >> 16U) & 0x1FU);
    const uint64_t           bytes   = 1ULL << log2;
    const bal_x86_register_t pointer = BAL_X86_REGISTER_ZERO;
    const bal_x86_register_t data    = BAL_X86_REGISTER_DISCARD_RESULT;

    emit_base_plus_offset(compiler, rn, 0);
    emit_translate_data_address(compiler, memory_interface, bytes, guest_address);

    bal_x86_assembler_t *BAL_RESTRICT assembler = &compiler->assembler;
    const int32_t lane_offset = (int32_t)(offsetof(bal_cpu_t, v[rt][0]) + (index << log2));
    bal_sliding_window_flush_all(&compiler->window);

    if (true == is_load || true == is_replicate)
    {
        switch (log2)
        {
            case 0U:
                bal_x86_emit_load_r32_mem8(assembler, data, pointer);
                break;
            case 1U:
                bal_x86_emit_load_r32_mem16(assembler, data, pointer);
                break;
            case 2U:
                bal_x86_emit_load_r32_mem32(assembler, data, pointer);
                break;
            default:
                bal_x86_emit_load_r64_mem64(assembler, data, pointer);
                break;
        }
    }

    if (true == is_replicate)
    {
        emit_sse(assembler, 0x66, true, 0x6E, XMM0, (uint8_t)data, "movq");

        switch (log2)
        {
            case 0U:
                emit_zero_vector(assembler, XMM1);
                emit_sse38(assembler, 0x00, XMM0, XMM1, "pshufb");
                break;
            case 1U:
                emit_sse_immediate(assembler, 0xF2, 0x70, XMM0, XMM0, 0U, "pshuflw");
                emit_sse(assembler, 0x66, false, 0x6C, XMM0, XMM0, "punpcklqdq");
                break;
            case 2U:
                emit_sse_immediate(assembler, 0x66, 0x70, XMM0, XMM0, 0U, "pshufd");
                break;
            default:
                emit_sse(assembler, 0x66, false, 0x6C, XMM0, XMM0, "punpcklqdq");
                break;
        }

        emit_store_vector(assembler, XMM0, rt, 1U == q);
    }
    else if (true == is_load)
    {
        // The guest pointer is no longer needed, so it points at the lane.
        bal_x86_emit_mov_r64_r64(assembler, pointer, BAL_X86_RBP);
        bal_x86_emit_add_r64_imm32(assembler, pointer, lane_offset);

        switch (log2)
        {
            case 0U:
                bal_x86_emit_store_mem8_r8(assembler, pointer, data);
                break;
            case 1U:
                bal_x86_emit_store_mem16_r16(assembler, pointer, data);
                break;
            case 2U:
                bal_x86_emit_store_mem32_r32(assembler, pointer, data);
                break;
            default:
                bal_x86_emit_store_mem64_r64(assembler, pointer, data);
                break;
        }
    }
    else
    {
        bal_x86_emit_mov_r64_r64(assembler, data, BAL_X86_RBP);
        bal_x86_emit_add_r64_imm32(assembler, data, lane_offset);

        switch (log2)
        {
            case 0U:
                bal_x86_emit_load_r32_mem8(assembler, data, data);
                bal_x86_emit_store_mem8_r8(assembler, pointer, data);
                break;
            case 1U:
                bal_x86_emit_load_r32_mem16(assembler, data, data);
                bal_x86_emit_store_mem16_r16(assembler, pointer, data);
                break;
            case 2U:
                bal_x86_emit_load_r32_mem32(assembler, data, data);
                bal_x86_emit_store_mem32_r32(assembler, pointer, data);
                break;
            default:
                bal_x86_emit_load_r64_mem64(assembler, data, data);
                bal_x86_emit_store_mem64_r64(assembler, pointer, data);
                break;
        }
    }

    if (true == is_post)
    {
        emit_structure_writeback(compiler, rn, rm, bytes);
    }

    return true;
}

/// Advanced SIMD instructions that only move bytes, for every element size: ZIP, UZP and TRN,
/// REV16, REV32 and REV64, DUP (element, vector and scalar), INS (element) and EXT. The
/// interpreter runs the instruction at translation time on registers whose bytes name their own
/// register and position, twice with different names, which gives the source of every result
/// byte; emit_byte_gather() then collects them. The emitters before this one translate the
/// common sizes with fewer instructions.
static bool
emit_fast_byte_permute(bal_tier1_compiler_t *BAL_RESTRICT compiler, const uint32_t instruction)
{
    const bool is_candidate = 0x0E000800U == (instruction & 0xBF208C00U)     // ZIP, UZP, TRN
                              || 0x0E200800U == (instruction & 0x9F3FEC00U)  // REV64, REV32, REV16
                              || 0x0E000400U == (instruction & 0xBFE0FC00U)  // DUP (element)
                              || 0x5E000400U == (instruction & 0xFFE0FC00U)  // DUP (scalar)
                              || 0x6E000400U == (instruction & 0xFFE08400U)  // INS (element)
                              || 0x2E000000U == (instruction & 0xBFE08400U); // EXT

    if (false == is_candidate || false == has_feature(compiler, BAL_TIER1_FEATURE_SSE41)
        || false == bal_is_simd_supported(instruction))
    {
        return false;
    }

    // The sources: Rn, Rm, and Rd, which INS partly keeps. A register named twice is one source.
    const uint32_t rd           = instruction & 0x1FU;
    const uint32_t registers[3] = { (instruction >> 5U) & 0x1FU, (instruction >> 16U) & 0x1FU, rd };
    uint8_t        maps[2][16];

    for (uint32_t variant = 0U; variant < 2U; ++variant)
    {
        bal_cpu_t probe;
        (void)memset(&probe, 0, sizeof(probe));

        for (uint32_t source = 3U; source-- > 0U;)
        {
            uint8_t bytes[16];

            for (uint32_t byte = 0U; byte < 16U; ++byte)
            {
                const uint32_t name = (source * 16U) + byte;

                bytes[byte] = (uint8_t)(0U == variant ? 1U + name : 0xFFU - name);
            }

            (void)memcpy(probe.v[registers[source]], bytes, sizeof(bytes));
        }

        (void)bal_execute_simd(&probe, instruction);
        uint8_t result[16];
        (void)memcpy(result, probe.v[rd], sizeof(result));

        for (uint32_t byte = 0U; byte < 16U; ++byte)
        {
            const uint32_t value = result[byte];

            if (0U == value)
            {
                maps[variant][byte] = 0xFFU;
                continue;
            }

            const uint32_t name = 0U == variant ? value - 1U : 0xFFU - value;

            if (name >= 48U)
            {
                return false;
            }

            // The lowest source that is the same register supplies the byte.
            uint32_t source = name / 16U;

            for (uint32_t other = 0U; other < source; ++other)
            {
                if (registers[other] == registers[source])
                {
                    source = other;
                    break;
                }
            }

            maps[variant][byte] = (uint8_t)((source * 16U) + (name % 16U));
        }
    }

    if (0 != memcmp(maps[0], maps[1], sizeof(maps[0])))
    {
        return false;
    }

    const int32_t offsets[3] = {
        (int32_t)offsetof(bal_cpu_t, v[registers[0]][0]),
        (int32_t)offsetof(bal_cpu_t, v[registers[1]][0]),
        (int32_t)offsetof(bal_cpu_t, v[registers[2]][0]),
    };

    bal_x86_assembler_t *BAL_RESTRICT assembler = &compiler->assembler;
    bal_sliding_window_flush_all(&compiler->window);
    emit_byte_gather(assembler, BAL_X86_RBP, offsets, 0U, 3U, maps[0]);
    emit_store_vector(assembler, XMM0, rd, true);
    return true;
}

/// TBL with two to four table registers and TBX with one to four. For each table register t,
/// PSHUFB looks up index - 16t, which PADDUSB with 0x70 leaves with bit 7 set (selecting zero)
/// unless it is below 16, and the results are ORed. TBX keeps the bytes of Rd whose indices are
/// beyond the table, where PMINUB with the last index changes the index.
static bool
emit_fast_table_lookup(bal_tier1_compiler_t *BAL_RESTRICT compiler, const uint32_t instruction)
{
    // 0 Q 001110 000 Rm 0 len op 00 Rn Rd.
    const uint32_t length = ((instruction >> 13U) & 0x3U) + 1U;
    const bool     is_tbx = 1U == ((instruction >> 12U) & 1U);

    if (0x0E000000U != (instruction & 0xBFE08C00U)
        || false == has_feature(compiler, BAL_TIER1_FEATURE_SSE41)
        || (false == is_tbx && 1U == length))
    {
        return false;
    }

    const bool                        is_quad   = 1U == ((instruction >> 30U) & 1U);
    const uint32_t                    rd        = instruction & 0x1FU;
    const uint32_t                    rn        = (instruction >> 5U) & 0x1FU;
    const uint32_t                    rm        = (instruction >> 16U) & 0x1FU;
    bal_x86_assembler_t *BAL_RESTRICT assembler = &compiler->assembler;

    bal_sliding_window_flush_all(&compiler->window);
    emit_load_vector(assembler, XMM3, rm, is_quad);
    emit_zero_vector(assembler, XMM0);

    for (uint32_t t = 0U; t < length; ++t)
    {
        emit_sse(assembler, 0U, false, 0x28, XMM1, XMM3, "movaps");

        if (t != 0U)
        {
            emit_broadcast_pattern(assembler, XMM2, replicate_lane(16U * t, 0U));
            emit_sse(assembler, 0x66, false, 0xF8, XMM1, XMM2, "psubb");
        }

        emit_broadcast_pattern(assembler, XMM2, 0x7070707070707070ULL);
        emit_sse(assembler, 0x66, false, 0xDC, XMM1, XMM2, "paddusb");
        emit_load_vector(assembler, XMM2, (rn + t) % 32U, true);
        emit_sse38(assembler, 0x00, XMM2, XMM1, "pshufb");
        emit_sse(assembler, 0x66, false, 0xEB, XMM0, XMM2, "por");
    }

    if (true == is_tbx)
    {
        emit_sse(assembler, 0U, false, 0x28, XMM1, XMM3, "movaps");
        emit_broadcast_pattern(assembler, XMM2, replicate_lane((16U * length) - 1U, 0U));
        emit_sse(assembler, 0x66, false, 0xDA, XMM1, XMM2, "pminub");
        emit_sse(assembler, 0x66, false, 0x74, XMM1, XMM3, "pcmpeqb");
        emit_load_vector(assembler, XMM2, rd, is_quad);
        emit_sse(assembler, 0x66, false, 0xDF, XMM1, XMM2, "pandn");
        emit_sse(assembler, 0x66, false, 0xEB, XMM0, XMM1, "por");
    }

    emit_store_vector(assembler, XMM0, rd, is_quad);
    return true;
}

// -----------------------------------------------------------------------------
// Inline half precision (F16C)
// -----------------------------------------------------------------------------

/// VCVTPH2PS: the four halves in the low 64 bits of `source` to singles in `destination`, exactly.
static void
emit_halves_to_singles(bal_x86_assembler_t *BAL_RESTRICT assembler,
                       const uint8_t                     destination,
                       const uint8_t                     source)
{
    bal_x86_emit_vex_0f38_register(assembler, false, 0x13, destination, 0U, source, "vcvtph2ps");
}

/// VCVTPS2PH: the four singles of `source` to halves in the low 64 bits of `destination`,
/// clearing the rest, rounded as MXCSR says.
static void
emit_singles_to_halves(bal_x86_assembler_t *BAL_RESTRICT assembler,
                       const uint8_t                     destination,
                       const uint8_t                     source)
{
    bal_x86_emit_vex_0f3a_register_immediate(
        assembler, false, 0x1D, source, 0U, destination, 4U, "vcvtps2ph");
}

/// Returns a jump to the slow path for a half precision lane of xmm0 that is a NaN, whose NaN
/// rules differ, or whose magnitude is the smallest normal number, where AArch64, detecting
/// tininess before rounding, may raise Underflow when x86 does not. Uses xmm1 to xmm3.
static size_t
emit_half_checks(bal_x86_assembler_t *BAL_RESTRICT assembler)
{
    emit_sse(assembler, 0U, false, 0x28, XMM1, XMM0, "movaps");
    emit_broadcast_pattern(assembler, XMM2, 0x7FFF7FFF7FFF7FFFULL);
    emit_sse(assembler, 0x66, false, 0xDB, XMM1, XMM2, "pand");
    emit_sse(assembler, 0U, false, 0x28, XMM2, XMM1, "movaps");
    emit_broadcast_pattern(assembler, XMM3, 0x7C007C007C007C00ULL);
    emit_sse(assembler, 0x66, false, 0x65, XMM2, XMM3, "pcmpgtw");
    emit_broadcast_pattern(assembler, XMM3, 0x0400040004000400ULL);
    emit_sse(assembler, 0x66, false, 0x75, XMM1, XMM3, "pcmpeqw");
    emit_sse(assembler, 0x66, false, 0xEB, XMM1, XMM2, "por");
    emit_sse(assembler, 0x66, false, 0xD7, (uint8_t)SCRATCH_A, XMM1, "pmovmskb");
    bal_x86_emit_test_r32_r32(assembler, SCRATCH_A, SCRATCH_A);
    return emit_forward_jump(assembler, true, BAL_X86_COND_NE);
}

/// Returns a jump to the slow path when a lane of `xmm` is a NaN, quietly. Uses xmm2.
static size_t
emit_nan_check(bal_x86_assembler_t *BAL_RESTRICT assembler, const uint8_t xmm, const bool is_double)
{
    const uint8_t packed = true == is_double ? 0x66 : 0U;
    emit_sse(assembler, 0U, false, 0x28, XMM2, xmm, "movaps");
    emit_sse_immediate(assembler, packed, 0xC2, XMM2, XMM2, 3U, "cmpunordp");
    emit_sse(assembler, packed, false, 0x50, (uint8_t)SCRATCH_A, XMM2, "movmskp");
    bal_x86_emit_test_r32_r32(assembler, SCRATCH_A, SCRATCH_A);
    return emit_forward_jump(assembler, true, BAL_X86_COND_NE);
}

/// FADD, FSUB, FMUL, FDIV and FABD (vector) in half precision. The halves convert to single
/// precision exactly, the operation rounds there and VCVTPS2PH rounds again to half precision:
/// single precision has more than twice the bits plus two, so the result is the correctly
/// rounded one, and a single precision result of halves can neither overflow nor be tiny.
/// emit_half_checks() sends NaNs and the smallest normal magnitude to the slow path.
static bool
emit_fast_half_arithmetic(bal_tier1_compiler_t *BAL_RESTRICT compiler, const uint32_t instruction)
{
    // 0 Q U 01110 a 10 Rm 00 opcode 1 Rn Rd.
    if (0x0E400400U != (instruction & 0x9F60C400U)
        || false == has_feature(compiler, BAL_TIER1_FEATURE_FMA))
    {
        return false;
    }

    const uint32_t key = (((instruction >> 29U) & 1U) << 4U) | (((instruction >> 23U) & 1U) << 3U)
                         | ((instruction >> 11U) & 0x7U);
    uint8_t operation = 0U;

    switch (key)
    {
        case 0x02U: // FADD
            operation = 0x58;
            break;
        case 0x0AU: // FSUB
        case 0x1AU: // FABD
            operation = 0x5C;
            break;
        case 0x13U: // FMUL
            operation = 0x59;
            break;
        case 0x17U: // FDIV
            operation = 0x5E;
            break;
        default:
            return false;
    }

    const bool                        is_quad   = 1U == ((instruction >> 30U) & 1U);
    const bool                        is_abs    = 0x1AU == key;
    const uint32_t                    rd        = instruction & 0x1FU;
    const uint32_t                    rn        = (instruction >> 5U) & 0x1FU;
    const uint32_t                    rm        = (instruction >> 16U) & 0x1FU;
    bal_x86_assembler_t *BAL_RESTRICT assembler = &compiler->assembler;
    size_t                            slow_jumps[2];

    slow_jumps[0] = begin_float_sequence(compiler);
    emit_load_vector(assembler, XMM2, rn, is_quad);
    emit_load_vector(assembler, XMM3, rm, is_quad);
    emit_halves_to_singles(assembler, XMM0, XMM2);
    emit_halves_to_singles(assembler, XMM1, XMM3);
    emit_sse(assembler, 0U, false, operation, XMM0, XMM1, "op");

    if (true == is_abs)
    {
        emit_all_ones(assembler, XMM1);
        emit_sse_immediate(assembler, 0x66, 0x72, 2U, XMM1, 1U, "psrld");
        emit_sse(assembler, 0U, false, 0x54, XMM0, XMM1, "andps");
    }

    emit_singles_to_halves(assembler, XMM0, XMM0);

    if (true == is_quad)
    {
        emit_sse_immediate(assembler, 0x66, 0x73, 3U, XMM2, 8U, "psrldq");
        emit_sse_immediate(assembler, 0x66, 0x73, 3U, XMM3, 8U, "psrldq");
        emit_halves_to_singles(assembler, XMM1, XMM2);
        emit_halves_to_singles(assembler, XMM2, XMM3);
        emit_sse(assembler, 0U, false, operation, XMM1, XMM2, "op");

        if (true == is_abs)
        {
            emit_all_ones(assembler, XMM2);
            emit_sse_immediate(assembler, 0x66, 0x72, 2U, XMM2, 1U, "psrld");
            emit_sse(assembler, 0U, false, 0x54, XMM1, XMM2, "andps");
        }

        emit_singles_to_halves(assembler, XMM1, XMM1);
        emit_sse(assembler, 0x66, false, 0x6C, XMM0, XMM1, "punpcklqdq");
    }

    slow_jumps[1] = emit_half_checks(assembler);
    emit_store_vector(assembler, XMM0, rd, is_quad);
    emit_slow_path(compiler, instruction, (uint64_t)(uintptr_t)&bal_execute_simd, slow_jumps, 2U);
    return true;
}

/// FCVTL and FCVTN (vector), with their "2" forms, between half and single and between single and
/// double precision. Widening is exact and narrowing rounds once; NaNs take the slow path, as do
/// narrowed results at the smallest normal magnitude (see emit_half_checks()).
static bool
emit_fast_vector_precision(bal_tier1_compiler_t *BAL_RESTRICT compiler, const uint32_t instruction)
{
    // 0 Q 0 01110 0 sz 10000 1011 x 10 Rn Rd: FCVTL with x = 1, FCVTN with x = 0.
    const uint32_t form      = instruction & 0xBFBFFC00U;
    const bool     is_long   = 0x0E217800U == form;
    const bool     is_double = 1U == ((instruction >> 22U) & 1U);

    if ((false == is_long && form != 0x0E216800U)
        || false == has_feature(compiler, BAL_TIER1_FEATURE_FMA))
    {
        return false;
    }

    const bool                        is_upper  = 1U == ((instruction >> 30U) & 1U);
    const uint32_t                    rd        = instruction & 0x1FU;
    const uint32_t                    rn        = (instruction >> 5U) & 0x1FU;
    bal_x86_assembler_t *BAL_RESTRICT assembler = &compiler->assembler;
    size_t                            slow_jumps[4];
    uint32_t                          count = 1U;

    slow_jumps[0] = begin_float_sequence(compiler);
    emit_load_vector(assembler, XMM0, rn, true);

    if (true == is_long)
    {
        if (true == is_upper)
        {
            emit_sse_immediate(assembler, 0x66, 0x73, 3U, XMM0, 8U, "psrldq");
        }

        if (true == is_double)
        {
            emit_sse(assembler, 0U, false, 0x5A, XMM1, XMM0, "cvtps2pd");
        }
        else
        {
            emit_halves_to_singles(assembler, XMM1, XMM0);
        }

        slow_jumps[count++] = emit_nan_check(assembler, XMM1, is_double);
        emit_store_vector(assembler, XMM1, rd, true);
    }
    else
    {
        slow_jumps[count++] = emit_nan_check(assembler, XMM0, is_double);

        if (true == is_double)
        {
            emit_sse(assembler, 0x66, false, 0x5A, XMM0, XMM0, "cvtpd2ps");
            count += emit_vector_checks(assembler, false, &slow_jumps[count]);
        }
        else
        {
            emit_singles_to_halves(assembler, XMM0, XMM0);
            slow_jumps[count++] = emit_half_checks(assembler);
        }

        emit_store_narrow(assembler, rd, is_upper);
    }

    emit_slow_path(
        compiler, instruction, (uint64_t)(uintptr_t)&bal_execute_simd, slow_jumps, count);
    return true;
}

/// FCVT (scalar) from half to single or double precision, exact, and from single to half
/// precision, rounding once. NaNs take the slow path, as do half results at the smallest normal
/// magnitude.
static bool
emit_fast_half_convert(bal_tier1_compiler_t *BAL_RESTRICT compiler, const uint32_t instruction)
{
    // FCVT Sd, Hn is 0x1EE24000, FCVT Dd, Hn 0x1EE2C000 and FCVT Hd, Sn 0x1E23C000.
    const uint32_t form = instruction & 0xFFFFFC00U;

    if ((form != 0x1EE24000U && form != 0x1EE2C000U && form != 0x1E23C000U)
        || false == has_feature(compiler, BAL_TIER1_FEATURE_FMA))
    {
        return false;
    }

    const uint32_t                    rd        = instruction & 0x1FU;
    const uint32_t                    rn        = (instruction >> 5U) & 0x1FU;
    bal_x86_assembler_t *BAL_RESTRICT assembler = &compiler->assembler;
    size_t                            slow_jumps[2];

    slow_jumps[0] = begin_float_sequence(compiler);
    emit_load_scalar(assembler, XMM0, rn, false);

    if (0x1E23C000U == form)
    {
        slow_jumps[1] = emit_nan_check(assembler, XMM0, false);
        emit_singles_to_halves(assembler, XMM0, XMM0);
        const size_t check = emit_half_checks(assembler);
        emit_sse(assembler, 0x66, false, 0x7E, XMM0, (uint8_t)SCRATCH_A, "movd");
        bal_x86_emit_shl_r32_imm8(assembler, SCRATCH_A, 16U);
        bal_x86_emit_shr_r32_imm8(assembler, SCRATCH_A, 16U);
        emit_store_scalar_result(assembler, rd);
        const size_t jumps[3] = { slow_jumps[0], slow_jumps[1], check };
        emit_slow_path(compiler, instruction, (uint64_t)(uintptr_t)&bal_execute_float, jumps, 3U);
        return true;
    }

    // The bits above the half are cleared, so that only it is converted (into lane 0).
    const bool is_double = 0x1EE2C000U == form;
    emit_sse_immediate(assembler, 0x66, 0x72, 6U, XMM0, 16U, "pslld");
    emit_sse_immediate(assembler, 0x66, 0x72, 2U, XMM0, 16U, "psrld");
    emit_halves_to_singles(assembler, XMM1, XMM0);
    slow_jumps[1] = emit_nan_check(assembler, XMM1, false);

    if (true == is_double)
    {
        emit_sse(assembler, 0xF3, false, 0x5A, XMM1, XMM1, "cvtss2sd");
    }

    emit_sse(assembler, 0x66, is_double, 0x7E, XMM1, (uint8_t)SCRATCH_A, "movd/movq");
    emit_store_scalar_result(assembler, rd);
    emit_slow_path(compiler, instruction, (uint64_t)(uintptr_t)&bal_execute_float, slow_jumps, 2U);
    return true;
}

/// The FRECPE estimates (RecipEstimate() & 0xFF) by the top 8 fraction bits.
static const uint8_t RECIPROCAL_ESTIMATES[256] = {
    0xFF, 0xFD, 0xFB, 0xF9, 0xF7, 0xF5, 0xF3, 0xF1, 0xF0, 0xEE, 0xEC, 0xEA, 0xE8, 0xE6, 0xE5, 0xE3,
    0xE1, 0xDF, 0xDD, 0xDC, 0xDA, 0xD8, 0xD7, 0xD5, 0xD3, 0xD2, 0xD0, 0xCE, 0xCD, 0xCB, 0xC9, 0xC8,
    0xC6, 0xC5, 0xC3, 0xC2, 0xC0, 0xBF, 0xBD, 0xBC, 0xBA, 0xB9, 0xB7, 0xB6, 0xB4, 0xB3, 0xB1, 0xB0,
    0xAE, 0xAD, 0xAC, 0xAA, 0xA9, 0xA7, 0xA6, 0xA5, 0xA3, 0xA2, 0xA1, 0x9F, 0x9E, 0x9D, 0x9C, 0x9A,
    0x99, 0x98, 0x96, 0x95, 0x94, 0x93, 0x91, 0x90, 0x8F, 0x8E, 0x8D, 0x8B, 0x8A, 0x89, 0x88, 0x87,
    0x86, 0x84, 0x83, 0x82, 0x81, 0x80, 0x7F, 0x7E, 0x7C, 0x7B, 0x7A, 0x79, 0x78, 0x77, 0x76, 0x75,
    0x74, 0x73, 0x72, 0x71, 0x70, 0x6F, 0x6E, 0x6D, 0x6C, 0x6B, 0x6A, 0x69, 0x68, 0x67, 0x66, 0x65,
    0x64, 0x63, 0x62, 0x61, 0x60, 0x5F, 0x5E, 0x5D, 0x5C, 0x5B, 0x5A, 0x59, 0x58, 0x58, 0x57, 0x56,
    0x55, 0x54, 0x53, 0x52, 0x51, 0x51, 0x50, 0x4F, 0x4E, 0x4D, 0x4C, 0x4B, 0x4B, 0x4A, 0x49, 0x48,
    0x47, 0x46, 0x46, 0x45, 0x44, 0x43, 0x42, 0x42, 0x41, 0x40, 0x3F, 0x3F, 0x3E, 0x3D, 0x3C, 0x3B,
    0x3B, 0x3A, 0x39, 0x38, 0x38, 0x37, 0x36, 0x35, 0x35, 0x34, 0x33, 0x33, 0x32, 0x31, 0x30, 0x30,
    0x2F, 0x2E, 0x2E, 0x2D, 0x2C, 0x2C, 0x2B, 0x2A, 0x2A, 0x29, 0x28, 0x28, 0x27, 0x26, 0x26, 0x25,
    0x24, 0x24, 0x23, 0x22, 0x22, 0x21, 0x20, 0x20, 0x1F, 0x1E, 0x1E, 0x1D, 0x1D, 0x1C, 0x1B, 0x1B,
    0x1A, 0x1A, 0x19, 0x18, 0x18, 0x17, 0x17, 0x16, 0x15, 0x15, 0x14, 0x14, 0x13, 0x12, 0x12, 0x11,
    0x11, 0x10, 0x10, 0x0F, 0x0F, 0x0E, 0x0D, 0x0D, 0x0C, 0x0C, 0x0B, 0x0B, 0x0A, 0x0A, 0x09, 0x09,
    0x08, 0x07, 0x07, 0x06, 0x06, 0x05, 0x05, 0x04, 0x04, 0x03, 0x03, 0x02, 0x02, 0x01, 0x01, 0x00,
};

/// The FRSQRTE estimates (RecipSqrtEstimate() & 0xFF) by the low exponent bit and the top 8
/// fraction bits.
static const uint8_t RECIPROCAL_SQUARE_ROOT_ESTIMATES[512] = {
    0x69, 0x69, 0x68, 0x68, 0x67, 0x67, 0x65, 0x65, 0x64, 0x64, 0x63, 0x63, 0x61, 0x61, 0x60, 0x60,
    0x5F, 0x5F, 0x5D, 0x5D, 0x5C, 0x5C, 0x5B, 0x5B, 0x5A, 0x5A, 0x58, 0x58, 0x57, 0x57, 0x56, 0x56,
    0x55, 0x55, 0x54, 0x54, 0x52, 0x52, 0x51, 0x51, 0x50, 0x50, 0x4F, 0x4F, 0x4E, 0x4E, 0x4D, 0x4D,
    0x4C, 0x4C, 0x4B, 0x4B, 0x4A, 0x4A, 0x48, 0x48, 0x47, 0x47, 0x46, 0x46, 0x45, 0x45, 0x44, 0x44,
    0x43, 0x43, 0x42, 0x42, 0x41, 0x41, 0x40, 0x40, 0x3F, 0x3F, 0x3E, 0x3E, 0x3D, 0x3D, 0x3C, 0x3C,
    0x3C, 0x3C, 0x3B, 0x3B, 0x3A, 0x3A, 0x39, 0x39, 0x38, 0x38, 0x37, 0x37, 0x36, 0x36, 0x35, 0x35,
    0x34, 0x34, 0x33, 0x33, 0x33, 0x33, 0x32, 0x32, 0x31, 0x31, 0x30, 0x30, 0x2F, 0x2F, 0x2E, 0x2E,
    0x2E, 0x2E, 0x2D, 0x2D, 0x2C, 0x2C, 0x2B, 0x2B, 0x2A, 0x2A, 0x2A, 0x2A, 0x29, 0x29, 0x28, 0x28,
    0x27, 0x27, 0x26, 0x26, 0x26, 0x26, 0x25, 0x25, 0x24, 0x24, 0x23, 0x23, 0x23, 0x23, 0x22, 0x22,
    0x21, 0x21, 0x21, 0x21, 0x20, 0x20, 0x1F, 0x1F, 0x1E, 0x1E, 0x1E, 0x1E, 0x1D, 0x1D, 0x1C, 0x1C,
    0x1C, 0x1C, 0x1B, 0x1B, 0x1A, 0x1A, 0x1A, 0x1A, 0x19, 0x19, 0x18, 0x18, 0x18, 0x18, 0x17, 0x17,
    0x16, 0x16, 0x16, 0x16, 0x15, 0x15, 0x14, 0x14, 0x14, 0x14, 0x13, 0x13, 0x13, 0x13, 0x12, 0x12,
    0x11, 0x11, 0x11, 0x11, 0x10, 0x10, 0x10, 0x10, 0x0F, 0x0F, 0x0E, 0x0E, 0x0E, 0x0E, 0x0D, 0x0D,
    0x0D, 0x0D, 0x0C, 0x0C, 0x0B, 0x0B, 0x0B, 0x0B, 0x0A, 0x0A, 0x0A, 0x0A, 0x09, 0x09, 0x09, 0x09,
    0x08, 0x08, 0x08, 0x08, 0x07, 0x07, 0x06, 0x06, 0x06, 0x06, 0x05, 0x05, 0x05, 0x05, 0x04, 0x04,
    0x04, 0x04, 0x03, 0x03, 0x03, 0x03, 0x02, 0x02, 0x02, 0x02, 0x01, 0x01, 0x01, 0x01, 0x00, 0x00,
    0xFF, 0xFF, 0xFD, 0xFD, 0xFB, 0xFB, 0xF9, 0xF9, 0xF7, 0xF7, 0xF5, 0xF5, 0xF3, 0xF3, 0xF2, 0xF2,
    0xF0, 0xF0, 0xEE, 0xEE, 0xEC, 0xEC, 0xEA, 0xEA, 0xE9, 0xE9, 0xE7, 0xE7, 0xE5, 0xE5, 0xE4, 0xE4,
    0xE2, 0xE2, 0xE0, 0xE0, 0xDF, 0xDF, 0xDD, 0xDD, 0xDB, 0xDB, 0xDA, 0xDA, 0xD8, 0xD8, 0xD7, 0xD7,
    0xD5, 0xD5, 0xD4, 0xD4, 0xD2, 0xD2, 0xD1, 0xD1, 0xCF, 0xCF, 0xCE, 0xCE, 0xCC, 0xCC, 0xCB, 0xCB,
    0xC9, 0xC9, 0xC8, 0xC8, 0xC6, 0xC6, 0xC5, 0xC5, 0xC4, 0xC4, 0xC2, 0xC2, 0xC1, 0xC1, 0xC0, 0xC0,
    0xBE, 0xBE, 0xBD, 0xBD, 0xBC, 0xBC, 0xBA, 0xBA, 0xB9, 0xB9, 0xB8, 0xB8, 0xB7, 0xB7, 0xB5, 0xB5,
    0xB4, 0xB4, 0xB3, 0xB3, 0xB2, 0xB2, 0xB0, 0xB0, 0xAF, 0xAF, 0xAE, 0xAE, 0xAD, 0xAD, 0xAC, 0xAC,
    0xAA, 0xAA, 0xA9, 0xA9, 0xA8, 0xA8, 0xA7, 0xA7, 0xA6, 0xA6, 0xA5, 0xA5, 0xA4, 0xA4, 0xA3, 0xA3,
    0xA2, 0xA2, 0xA0, 0xA0, 0x9F, 0x9F, 0x9E, 0x9E, 0x9D, 0x9D, 0x9C, 0x9C, 0x9B, 0x9B, 0x9A, 0x9A,
    0x99, 0x99, 0x98, 0x98, 0x97, 0x97, 0x96, 0x96, 0x95, 0x95, 0x94, 0x94, 0x93, 0x93, 0x92, 0x92,
    0x91, 0x91, 0x90, 0x90, 0x8F, 0x8F, 0x8E, 0x8E, 0x8D, 0x8D, 0x8C, 0x8C, 0x8C, 0x8C, 0x8B, 0x8B,
    0x8A, 0x8A, 0x89, 0x89, 0x88, 0x88, 0x87, 0x87, 0x86, 0x86, 0x85, 0x85, 0x84, 0x84, 0x83, 0x83,
    0x83, 0x83, 0x82, 0x82, 0x81, 0x81, 0x80, 0x80, 0x7F, 0x7F, 0x7E, 0x7E, 0x7E, 0x7E, 0x7D, 0x7D,
    0x7C, 0x7C, 0x7B, 0x7B, 0x7A, 0x7A, 0x79, 0x79, 0x79, 0x79, 0x78, 0x78, 0x77, 0x77, 0x76, 0x76,
    0x76, 0x76, 0x75, 0x75, 0x74, 0x74, 0x73, 0x73, 0x72, 0x72, 0x72, 0x72, 0x71, 0x71, 0x70, 0x70,
    0x6F, 0x6F, 0x6F, 0x6F, 0x6E, 0x6E, 0x6D, 0x6D, 0x6D, 0x6D, 0x6C, 0x6C, 0x6B, 0x6B, 0x6A, 0x6A,
};

// -----------------------------------------------------------------------------
// Inline estimates, steps, compares and the rest
// -----------------------------------------------------------------------------

/// The runtime helper that executes `instruction`, as translate_float() chooses it.
static uint64_t
float_helper(const uint32_t instruction)
{
    return true == bal_is_float_supported(instruction) ? (uint64_t)(uintptr_t)&bal_execute_float
                                                       : (uint64_t)(uintptr_t)&bal_execute_simd;
}

/// Puts all ones in the lanes of `destination` where `source` is a zero or an infinity: shifted
/// left by one, dropping the sign, these are zero or the infinity's exponent alone. Uses xmm4 and
/// xmm5 (SSE4.1).
static void
emit_zero_or_infinity(bal_x86_assembler_t *BAL_RESTRICT assembler,
                      const uint8_t                     destination,
                      const uint8_t                     source,
                      const bool                        is_double)
{
    const uint32_t size = true == is_double ? 3U : 2U;
    emit_sse(assembler, 0U, false, 0x28, destination, source, "movaps");
    emit_sse_immediate(
        assembler, 0x66, true == is_double ? 0x73 : 0x72, 6U, destination, 1U, "psll");
    emit_sse(assembler, 0U, false, 0x28, XMM4, destination, "movaps");
    emit_zero_vector(assembler, XMM5);
    emit_lanes_equal(assembler, size, destination, XMM5);
    emit_broadcast_pattern(
        assembler, XMM5, true == is_double ? 0xFFE0000000000000ULL : 0xFF000000FF000000ULL);
    emit_lanes_equal(assembler, size, XMM4, XMM5);
    emit_sse(assembler, 0x66, false, 0xEB, destination, XMM4, "por");
}

/// FMULX, FRECPS and FRSQRTS (vector and scalar) in single and double precision. FMULX is FMUL
/// whose zero times infinity gives 2 with that sign; FRECPS is 2 - n * m and FRSQRTS
/// (3 - n * m) / 2, both fused and giving 2 or 1.5 for zero times infinity. Lanes where both
/// operands are zeros or infinities take the slow path, which keeps the host multiplication from
/// raising Invalid. FRSQRTS computes 1.5 - (n / 2) * m with one rounding, which needs n / 2 to be
/// exact: lanes whose n has the exponent 0 or 1 take the slow path too.
static bool
emit_fast_float_step(bal_tier1_compiler_t *BAL_RESTRICT compiler, const uint32_t instruction)
{
    // Vector 0 Q 0 01110 a sz 1 Rm opcode 1 Rn Rd, scalar 01 0 11110 a sz 1 Rm opcode 1 Rn Rd:
    // FMULX is a = 0 and opcode 11011, FRECPS a = 0 and FRSQRTS a = 1 with opcode 11111. The
    // scalar forms are the vector ones with bit 28 set and Q = 1.
    // FMULX by element: 0 Q 1 01111 1 sz L M Rm 1001 H 0 Rn Rd, or the scalar 01 1 11111 ...
    const bool is_element_vector = 0x2F809000U == (instruction & 0xBF80F400U);
    const bool is_element_scalar = 0x7F809000U == (instruction & 0xFF80F400U);
    const bool is_element        = true == is_element_vector || true == is_element_scalar;
    const bool is_scalar = 0x5E000000U == (instruction & 0xFF000000U) || true == is_element_scalar;
    const uint32_t vector
        = true == is_element ? 0x0E20DC00U : (instruction & 0xBFA0FC00U) & ~0x50000000U;
    const bool is_double = 1U == ((instruction >> 22U) & 1U);
    const bool is_quad   = false == is_scalar && 1U == ((instruction >> 30U) & 1U);

    // A double precision element has the index H alone.
    if ((false == is_scalar && false == is_element && 0x0E000000U != (instruction & 0xBF000000U))
        || (true == is_element && true == is_double && 1U == ((instruction >> 21U) & 1U))
        || (vector != 0x0E20DC00U && vector != 0x0E20FC00U && vector != 0x0EA0FC00U)
        || (true == is_double && false == is_quad && false == is_scalar)
        || false == has_feature(compiler, BAL_TIER1_FEATURE_FMA)
        || false == has_feature(compiler, BAL_TIER1_FEATURE_SSE41))
    {
        return false;
    }

    const bool                        is_multiply = 0x0E20DC00U == vector;
    const bool                        is_halving  = 0x0EA0FC00U == vector;
    const uint32_t                    rd          = instruction & 0x1FU;
    const uint32_t                    rn          = (instruction >> 5U) & 0x1FU;
    const uint32_t                    rm          = (instruction >> 16U) & 0x1FU;
    const uint8_t                     packed      = true == is_double ? 0x66 : 0U;
    const uint32_t                    lanes       = true == is_scalar   ? 1U
                                                    : true == is_double ? 2U
                                                    : true == is_quad   ? 4U
                                                                        : 2U;
    bal_x86_assembler_t *BAL_RESTRICT assembler   = &compiler->assembler;
    size_t                            slow_jumps[4];

    slow_jumps[0] = begin_float_sequence(compiler);

    if (true == is_scalar)
    {
        emit_load_scalar(assembler, XMM0, rn, is_double);
    }
    else
    {
        emit_load_vector(assembler, XMM0, rn, is_quad);
    }

    if (true == is_element)
    {
        // The element, broadcast to the lanes in use only, so that zero times infinity cannot
        // arise in the others.
        const uint32_t h     = (instruction >> 11U) & 1U;
        const uint32_t index = true == is_double ? h : (h << 1U) | ((instruction >> 21U) & 1U);
        emit_load_vector(assembler, XMM1, rm, true);
        emit_broadcast_lane(assembler, XMM1, is_double, index);

        if (true == is_scalar)
        {
            emit_load_vector_constant(
                assembler, XMM2, true == is_double ? ~0ULL : 0xFFFFFFFFULL, 0U);
            emit_sse(assembler, 0x66, false, 0xDB, XMM1, XMM2, "pand");
        }
        else if (false == is_quad)
        {
            emit_sse(assembler, 0xF3, false, 0x7E, XMM1, XMM1, "movq");
        }
    }
    else if (true == is_scalar)
    {
        emit_load_scalar(assembler, XMM1, rm, is_double);
    }
    else
    {
        emit_load_vector(assembler, XMM1, rm, is_quad);
    }

    emit_zero_or_infinity(assembler, XMM2, XMM0, is_double);
    emit_zero_or_infinity(assembler, XMM3, XMM1, is_double);
    emit_sse(assembler, 0x66, false, 0xDB, XMM2, XMM3, "pand");

    if (true == is_halving)
    {
        // The exponent minus 2 is negative for exponents 0 and 1.
        emit_sse(assembler, 0U, false, 0x28, XMM3, XMM0, "movaps");
        emit_sse_immediate(assembler, 0x66, true == is_double ? 0x73 : 0x72, 6U, XMM3, 1U, "psll");
        emit_sse_immediate(assembler,
                           0x66,
                           true == is_double ? 0x73 : 0x72,
                           2U,
                           XMM3,
                           true == is_double ? 53U : 24U,
                           "psrl");
        emit_broadcast_pattern(assembler, XMM4, true == is_double ? 2ULL : 0x0000000200000002ULL);
        emit_sse(assembler, 0x66, false, true == is_double ? 0xFB : 0xFA, XMM3, XMM4, "psub");
        emit_sse(assembler, 0x66, false, 0xEB, XMM2, XMM3, "por");
    }

    emit_sse(assembler, packed, false, 0x50, (uint8_t)SCRATCH_A, XMM2, "movmskp");
    bal_x86_emit_mov_r64_imm64(assembler, SCRATCH_B, (1ULL << lanes) - 1U);
    bal_x86_emit_and_r32_r32(assembler, SCRATCH_A, SCRATCH_B);
    slow_jumps[1] = emit_forward_jump(assembler, true, BAL_X86_COND_NE);

    if (true == is_multiply)
    {
        emit_sse(assembler, packed, false, 0x59, XMM0, XMM1, "mulp");
    }
    else
    {
        // VFNMADD231 computes addend - n * m.
        const uint64_t half = true == is_double ? 0x3FE0000000000000ULL : 0x3F0000003F000000ULL;
        const uint64_t addend
            = true == is_halving
                  ? (true == is_double ? 0x3FF8000000000000ULL : 0x3FC000003FC00000ULL)
                  : (true == is_double ? 0x4000000000000000ULL : 0x4000000040000000ULL);

        if (true == is_halving)
        {
            emit_broadcast_pattern(assembler, XMM2, half);
            emit_sse(assembler, packed, false, 0x59, XMM0, XMM2, "mulp");
        }

        emit_broadcast_pattern(assembler, XMM2, addend);
        bal_x86_emit_vex_0f38_register(assembler, is_double, 0xBC, XMM2, XMM0, XMM1, "vfnmadd231p");
        emit_sse(assembler, 0U, false, 0x28, XMM0, XMM2, "movaps");
    }

    const uint32_t count = 2U + emit_vector_checks(assembler, is_double, &slow_jumps[2]);

    if (true == is_scalar)
    {
        emit_sse(assembler, 0x66, is_double, 0x7E, XMM0, (uint8_t)SCRATCH_A, "movd/movq");
        emit_store_scalar_result(assembler, rd);
    }
    else
    {
        emit_store_vector(assembler, XMM0, rd, is_quad);
    }

    emit_slow_path(compiler, instruction, float_helper(instruction), slow_jumps, count);
    return true;
}

/// FACGE and FACGT (vector and scalar), and the scalar FCMEQ, FCMGE and FCMGT (register) and
/// FCMEQ, FCMGE, FCMGT, FCMLE and FCMLT (zero). The compares signal Invalid as these do (see
/// emit_fast_float_misc()); the absolute compares clear the sign bits first.
static bool
emit_fast_float_compare_more(bal_tier1_compiler_t *BAL_RESTRICT compiler,
                             const uint32_t                     instruction)
{
    // Register: 0 Q U 01110 E sz 1 Rm 1110 a 1 Rn Rd (vector, absolute only) or 01 U 11110 E sz 1
    // Rm 1110 a 1 Rn Rd. Zero: 01 U 11110 1 sz 10000 011 op 10 Rn Rd.
    const uint32_t three     = instruction & 0xFFA0FC00U;
    const uint32_t vector    = instruction & 0xBFA0FC00U;
    const uint32_t zero      = instruction & 0xFFBFFC00U;
    uint32_t       relation  = 0U; // 0 EQ, 1 GE, 2 GT, 3 LE, 4 LT, with 8 for absolute values.
    bool           is_zero   = false;
    bool           is_scalar = true;

    switch (three)
    {
        case 0x5E20E400U:
            relation = 0U;
            break;
        case 0x7E20E400U:
            relation = 1U;
            break;
        case 0x7EA0E400U:
            relation = 2U;
            break;
        case 0x7E20EC00U:
            relation = 9U;
            break;
        case 0x7EA0EC00U:
            relation = 10U;
            break;
        default:
            switch (zero)
            {
                case 0x5EA0D800U:
                    relation = 0U;
                    is_zero  = true;
                    break;
                case 0x7EA0C800U:
                    relation = 1U;
                    is_zero  = true;
                    break;
                case 0x5EA0C800U:
                    relation = 2U;
                    is_zero  = true;
                    break;
                case 0x7EA0D800U:
                    relation = 3U;
                    is_zero  = true;
                    break;
                case 0x5EA0E800U:
                    relation = 4U;
                    is_zero  = true;
                    break;
                default:
                    if (0x2E20EC00U == vector || 0x2EA0EC00U == vector)
                    {
                        relation  = 0x2EA0EC00U == vector ? 10U : 9U;
                        is_scalar = false;
                        break;
                    }

                    return false;
            }
            break;
    }

    const bool is_double = 1U == ((instruction >> 22U) & 1U);
    const bool is_quad   = false == is_scalar && 1U == ((instruction >> 30U) & 1U);

    if (true == is_double && false == is_quad && false == is_scalar)
    {
        return false;
    }

    const uint32_t                    rd        = instruction & 0x1FU;
    const uint32_t                    rn        = (instruction >> 5U) & 0x1FU;
    const uint32_t                    rm        = (instruction >> 16U) & 0x1FU;
    const uint8_t                     packed    = true == is_double ? 0x66 : 0U;
    bal_x86_assembler_t *BAL_RESTRICT assembler = &compiler->assembler;
    size_t                            slow_jumps[1];

    slow_jumps[0] = begin_float_sequence(compiler);

    if (true == is_scalar)
    {
        emit_load_scalar(assembler, XMM0, rn, is_double);
    }
    else
    {
        emit_load_vector(assembler, XMM0, rn, is_quad);
    }

    if (true == is_zero)
    {
        emit_zero_vector(assembler, XMM1);
    }
    else if (true == is_scalar)
    {
        emit_load_scalar(assembler, XMM1, rm, is_double);
    }
    else
    {
        emit_load_vector(assembler, XMM1, rm, is_quad);
    }

    if (relation >= 8U)
    {
        emit_all_ones(assembler, XMM2);
        emit_sse_immediate(assembler, 0x66, true == is_double ? 0x73 : 0x72, 2U, XMM2, 1U, "psrl");
        emit_sse(assembler, 0U, false, 0x54, XMM0, XMM2, "andps");
        emit_sse(assembler, 0U, false, 0x54, XMM1, XMM2, "andps");
    }

    // CMPP: EQ_OQ is quiet like FCMEQ; LT_OS and LE_OS signal like the others. n >= m is
    // m <= n and n > m is m < n.
    uint8_t result = XMM0;

    switch (relation & 7U)
    {
        case 0U:
            emit_sse_immediate(assembler, packed, 0xC2, XMM0, XMM1, 0U, "cmpeqp");
            break;
        case 1U:
            emit_sse_immediate(assembler, packed, 0xC2, XMM1, XMM0, 2U, "cmplep");
            result = XMM1;
            break;
        case 2U:
            emit_sse_immediate(assembler, packed, 0xC2, XMM1, XMM0, 1U, "cmpltp");
            result = XMM1;
            break;
        case 3U:
            emit_sse_immediate(assembler, packed, 0xC2, XMM0, XMM1, 2U, "cmplep");
            break;
        default:
            emit_sse_immediate(assembler, packed, 0xC2, XMM0, XMM1, 1U, "cmpltp");
            break;
    }

    if (true == is_scalar)
    {
        emit_sse(assembler, 0x66, is_double, 0x7E, result, (uint8_t)SCRATCH_A, "movd/movq");
        emit_store_scalar_result(assembler, rd);
    }
    else
    {
        emit_store_vector(assembler, result, rd, is_quad);
    }

    emit_slow_path(compiler, instruction, float_helper(instruction), slow_jumps, 1U);
    return true;
}

/// Computes the FRECPE or (`is_root`) FRSQRTE estimates of the first `lanes` single precision
/// lanes of xmm0 into xmm1, using xmm2 to xmm5, SCRATCH_A and SCRATCH_B, and returns the jump to
/// the slow path for a lane out of the tables' range: for single precision (or with `is_half`
/// the converted halves) the denormal inputs and results, infinities and NaNs, and for FRSQRTE
/// negative inputs.
static size_t
emit_estimate(bal_x86_assembler_t *BAL_RESTRICT assembler,
              const bool                        is_root,
              const uint32_t                    lanes,
              const bool                        is_half)
{
    // XMM2 = the exponents; XMM3 = the lanes out of range.
    emit_sse(assembler, 0U, false, 0x28, XMM2, XMM0, "movaps");
    emit_sse_immediate(assembler, 0x66, 0x72, 6U, XMM2, 1U, "pslld");
    emit_sse_immediate(assembler, 0x66, 0x72, 2U, XMM2, 24U, "psrld");

    if (true == is_root)
    {
        // Not above 0x007FFFFF (signed: negative, zero or denormal; for halves below 2^-14) or
        // above 0x7F7FFFFF.
        emit_broadcast_pattern(
            assembler, XMM3, true == is_half ? 0x3880000038800000ULL : 0x0080000000800000ULL);
        emit_sse(assembler, 0x66, false, 0x66, XMM3, XMM0, "pcmpgtd");
        emit_broadcast_pattern(assembler, XMM4, 0x7F7FFFFF7F7FFFFFULL);
        emit_sse(assembler, 0U, false, 0x28, XMM5, XMM0, "movaps");
        emit_sse(assembler, 0x66, false, 0x66, XMM5, XMM4, "pcmpgtd");
    }
    else
    {
        // An exponent below 1 or above 252 (for halves below 113 or above 140).
        emit_broadcast_pattern(
            assembler, XMM3, true == is_half ? 0x0000007100000071ULL : 0x0000000100000001ULL);
        emit_sse(assembler, 0x66, false, 0x66, XMM3, XMM2, "pcmpgtd");
        emit_broadcast_pattern(
            assembler, XMM4, true == is_half ? 0x0000008C0000008CULL : 0x000000FC000000FCULL);
        emit_sse(assembler, 0U, false, 0x28, XMM5, XMM2, "movaps");
        emit_sse(assembler, 0x66, false, 0x66, XMM5, XMM4, "pcmpgtd");
    }

    emit_sse(assembler, 0x66, false, 0xEB, XMM3, XMM5, "por");
    emit_sse(assembler, 0U, false, 0x50, (uint8_t)SCRATCH_A, XMM3, "movmskps");
    bal_x86_emit_mov_r64_imm64(assembler, SCRATCH_B, (1ULL << lanes) - 1U);
    bal_x86_emit_and_r32_r32(assembler, SCRATCH_A, SCRATCH_B);
    const size_t out_of_range = emit_forward_jump(assembler, true, BAL_X86_COND_NE);

    // Each lane's estimate goes into XMM1: index (x >> 15) & 0x1FF (FRSQRTE: the low exponent bit
    // and 8 fraction bits) or & 0xFF.
    static const uint8_t PEXTRD[3] = { 0x0F, 0x3A, 0x16 };
    static const uint8_t PINSRD[3] = { 0x0F, 0x3A, 0x22 };
    const uint64_t table = true == is_root ? (uint64_t)(uintptr_t)RECIPROCAL_SQUARE_ROOT_ESTIMATES
                                           : (uint64_t)(uintptr_t)RECIPROCAL_ESTIMATES;
    emit_zero_vector(assembler, XMM1);

    for (uint32_t lane = 0U; lane < lanes; ++lane)
    {
        bal_x86_emit_sse_register(assembler,
                                  0x66,
                                  false,
                                  PEXTRD,
                                  3U,
                                  XMM0,
                                  (uint8_t)SCRATCH_A,
                                  true,
                                  (uint8_t)lane,
                                  "pextrd");
        bal_x86_emit_shr_r32_imm8(assembler, SCRATCH_A, 15U);
        bal_x86_emit_mov_r64_imm64(assembler, SCRATCH_B, true == is_root ? 0x1FFU : 0xFFU);
        bal_x86_emit_and_r32_r32(assembler, SCRATCH_A, SCRATCH_B);
        bal_x86_emit_mov_r64_imm64(assembler, SCRATCH_B, table);
        bal_x86_emit_add_r64_r64(assembler, SCRATCH_A, SCRATCH_B);
        bal_x86_emit_load_r32_mem8(assembler, SCRATCH_A, SCRATCH_A);
        bal_x86_emit_sse_register(assembler,
                                  0x66,
                                  false,
                                  PINSRD,
                                  3U,
                                  XMM1,
                                  (uint8_t)SCRATCH_A,
                                  true,
                                  (uint8_t)lane,
                                  "pinsrd");
    }

    // FRECPE: sign | (253 - exponent) << 23 | estimate << 15; FRSQRTE: (380 - exponent) / 2 << 23
    // | estimate << 15.
    emit_sse_immediate(assembler, 0x66, 0x72, 6U, XMM1, 15U, "pslld");
    emit_broadcast_pattern(
        assembler, XMM3, true == is_root ? 0x0000017C0000017CULL : 0x000000FD000000FDULL);
    emit_sse(assembler, 0x66, false, 0xFA, XMM3, XMM2, "psubd");

    if (true == is_root)
    {
        emit_sse_immediate(assembler, 0x66, 0x72, 2U, XMM3, 1U, "psrld");
    }
    else
    {
        emit_all_ones(assembler, XMM2);
        emit_sse_immediate(assembler, 0x66, 0x72, 6U, XMM2, 31U, "pslld");
        emit_sse(assembler, 0x66, false, 0xDB, XMM0, XMM2, "pand");
        emit_sse(assembler, 0x66, false, 0xEB, XMM1, XMM0, "por");
    }

    emit_sse_immediate(assembler, 0x66, 0x72, 6U, XMM3, 23U, "pslld");
    emit_sse(assembler, 0x66, false, 0xEB, XMM1, XMM3, "por");
    return out_of_range;
}

/// FRECPE and FRSQRTE (vector and scalar) in single precision. Their 8-bit estimates come from
/// RECIPROCAL_ESTIMATES and RECIPROCAL_SQUARE_ROOT_ESTIMATES, looked up lane by lane through
/// general purpose registers; the exponents follow arithmetically. Lanes the tables do not cover
/// take the slow path: for FRECPE exponents 0 and 253 and above (denormal inputs or results,
/// infinities and NaNs), for FRSQRTE everything but positive normal numbers.
static bool
emit_fast_estimate(bal_tier1_compiler_t *BAL_RESTRICT compiler, const uint32_t instruction)
{
    // Vector 0 Q 1? 01110 1 0 10000 1110 1 10 Rn Rd (FRECPE U = 0, FRSQRTE U = 1), scalar
    // 01 U 11110 1 0 10000 11101 10 Rn Rd.
    const uint32_t vector    = instruction & 0x9FFFFC00U;
    const bool     is_scalar = 0x5EA1D800U == (instruction & 0xDFFFFC00U);

    if ((0x0EA1D800U != vector && false == is_scalar)
        || false == has_feature(compiler, BAL_TIER1_FEATURE_SSE41))
    {
        return false;
    }

    const bool     is_root = 1U == ((instruction >> 29U) & 1U);
    const bool     is_quad = false == is_scalar && 1U == ((instruction >> 30U) & 1U);
    const uint32_t lanes   = true == is_scalar ? 1U : (true == is_quad ? 4U : 2U);
    const uint32_t rd      = instruction & 0x1FU;
    const uint32_t rn      = (instruction >> 5U) & 0x1FU;
    bal_x86_assembler_t *BAL_RESTRICT assembler = &compiler->assembler;
    size_t                            slow_jumps[2];

    slow_jumps[0] = begin_float_sequence(compiler);

    if (true == is_scalar)
    {
        emit_load_scalar(assembler, XMM0, rn, false);
    }
    else
    {
        emit_load_vector(assembler, XMM0, rn, is_quad);
    }

    slow_jumps[1] = emit_estimate(assembler, is_root, lanes, false);

    if (true == is_scalar)
    {
        emit_sse(assembler, 0x66, false, 0x7E, XMM1, (uint8_t)SCRATCH_A, "movd");
        emit_store_scalar_result(assembler, rd);
    }
    else
    {
        emit_store_vector(assembler, XMM1, rd, is_quad);
    }

    emit_slow_path(compiler, instruction, float_helper(instruction), slow_jumps, 2U);
    return true;
}

/// Computes SQDMULH, or with `is_rounding` SQRDMULH, of xmm0 and xmm1 into xmm0 on halfwords
/// (`size` 1) or words, setting FPSR.QC when a lane saturates. Halfwords use PMULHRSW, or
/// PMULHW and PMULLW for the doubled high half; words multiply the even and odd lanes into
/// doublewords with PMULDQ. Only the minimum times itself overflows, giving the minimum, which
/// becomes the maximum. Uses xmm2 to xmm4 (SSE4.1).
static void
emit_doubling_multiply_high(bal_x86_assembler_t *BAL_RESTRICT assembler,
                            const uint32_t                    size,
                            const bool                        is_rounding)
{
    if (1U == size)
    {
        if (true == is_rounding)
        {
            emit_sse38(assembler, 0x0B, XMM0, XMM1, "pmulhrsw");
        }
        else
        {
            emit_sse(assembler, 0U, false, 0x28, XMM2, XMM0, "movaps");
            emit_sse(assembler, 0x66, false, 0xE5, XMM2, XMM1, "pmulhw");
            emit_sse(assembler, 0x66, false, 0xD5, XMM0, XMM1, "pmullw");
            emit_sse_immediate(assembler, 0x66, 0x71, 2U, XMM0, 15U, "psrlw");
            emit_sse_immediate(assembler, 0x66, 0x71, 6U, XMM2, 1U, "psllw");
            emit_sse(assembler, 0x66, false, 0xEB, XMM0, XMM2, "por");
        }
    }
    else
    {
        emit_sse(assembler, 0U, false, 0x28, XMM2, XMM0, "movaps");
        emit_sse(assembler, 0U, false, 0x28, XMM3, XMM1, "movaps");
        emit_sse38(assembler, 0x28, XMM2, XMM3, "pmuldq");
        emit_sse_immediate(assembler, 0x66, 0x73, 2U, XMM0, 32U, "psrlq");
        emit_sse_immediate(assembler, 0x66, 0x73, 2U, XMM1, 32U, "psrlq");
        emit_sse38(assembler, 0x28, XMM0, XMM1, "pmuldq");
        emit_sse_immediate(assembler, 0x66, 0x73, 6U, XMM2, 1U, "psllq");
        emit_sse_immediate(assembler, 0x66, 0x73, 6U, XMM0, 1U, "psllq");

        if (true == is_rounding)
        {
            emit_broadcast_pattern(assembler, XMM4, 0x80000000ULL);
            emit_sse(assembler, 0x66, false, 0xD4, XMM2, XMM4, "paddq");
            emit_sse(assembler, 0x66, false, 0xD4, XMM0, XMM4, "paddq");
        }

        // The high words of the even and odd products, in lane order.
        emit_sse_immediate(assembler, 0U, 0xC6, XMM2, XMM0, 0xDD, "shufps");
        emit_sse_immediate(assembler, 0x66, 0x70, XMM0, XMM2, 0xD8, "pshufd");
    }

    emit_broadcast_pattern(assembler, XMM2, lane_sign_pattern(size));
    emit_lanes_equal(assembler, size, XMM2, XMM0);
    emit_sse(assembler, 0x66, false, 0xEF, XMM0, XMM2, "pxor");
    emit_sse(assembler, 0x66, false, 0xD7, (uint8_t)SCRATCH_A, XMM2, "pmovmskb");
    bal_x86_emit_test_r32_r32(assembler, SCRATCH_A, SCRATCH_A);
    const size_t unsaturated = emit_forward_jump(assembler, true, BAL_X86_COND_E);
    emit_set_saturation(assembler);
    patch_forward_jump(assembler, unsaturated, true);
}

/// SQDMULH and SQRDMULH (vector and by element) on halfwords and words.
static bool
emit_fast_doubling_multiply(bal_tier1_compiler_t *BAL_RESTRICT compiler, const uint32_t instruction)
{
    // Vector 0 Q U 01110 size 1 Rm 10110 1 Rn Rd; by element 0 Q 0 01111 size L M Rm 110 R H 0 Rn
    // Rd.
    const uint32_t size       = (instruction >> 22U) & 0x3U;
    const bool     is_vector  = 0x0E20B400U == (instruction & 0x9F20FC00U);
    const bool     is_element = 0x0F00C000U == (instruction & 0xBF00E400U);

    if ((false == is_vector && false == is_element) || 0U == size || 3U == size
        || false == has_feature(compiler, BAL_TIER1_FEATURE_SSE41))
    {
        return false;
    }

    const bool is_quad = 1U == ((instruction >> 30U) & 1U);
    const bool is_rounding
        = true == is_vector ? 1U == ((instruction >> 29U) & 1U) : 1U == ((instruction >> 12U) & 1U);
    const uint32_t                    rd        = instruction & 0x1FU;
    const uint32_t                    rn        = (instruction >> 5U) & 0x1FU;
    bal_x86_assembler_t *BAL_RESTRICT assembler = &compiler->assembler;

    bal_sliding_window_flush_all(&compiler->window);
    emit_load_vector(assembler, XMM0, rn, is_quad);

    if (true == is_vector)
    {
        emit_load_vector(assembler, XMM1, (instruction >> 16U) & 0x1FU, is_quad);
    }
    else
    {
        const uint32_t h     = (instruction >> 11U) & 1U;
        const uint32_t l     = (instruction >> 21U) & 1U;
        const uint32_t m     = (instruction >> 20U) & 1U;
        const uint32_t index = 1U == size ? (h << 2U) | (l << 1U) | m : (h << 1U) | l;
        const uint32_t rm = 1U == size ? (instruction >> 16U) & 0xFU : (instruction >> 16U) & 0x1FU;
        emit_load_vector(assembler, XMM1, rm, true);
        emit_broadcast_element(assembler, XMM1, size, index);
    }

    emit_doubling_multiply_high(assembler, size, is_rounding);
    emit_store_vector(assembler, XMM0, rd, is_quad);
    return true;
}

/// SDOT and UDOT (vector and by element): the bytes are extended to halfwords, whose products
/// PMADDWD adds in pairs (exactly, unsigned bytes being positive halfwords), and PHADDD adds the
/// pairs of pairs.
static bool
emit_fast_dot_product(bal_tier1_compiler_t *BAL_RESTRICT compiler, const uint32_t instruction)
{
    // Vector 0 Q U 01110 10 0 Rm 1001 0 1 Rn Rd; by element 0 Q U 01111 10 L M Rm 1110 H 0 Rn Rd.
    const bool is_vector  = 0x0E809400U == (instruction & 0x9FE0FC00U);
    const bool is_element = 0x0F80E000U == (instruction & 0x9FC0F400U);

    if ((false == is_vector && false == is_element)
        || false == has_feature(compiler, BAL_TIER1_FEATURE_SSE41))
    {
        return false;
    }

    const bool                        is_quad     = 1U == ((instruction >> 30U) & 1U);
    const bool                        is_unsigned = 1U == ((instruction >> 29U) & 1U);
    const uint32_t                    rd          = instruction & 0x1FU;
    const uint32_t                    rn          = (instruction >> 5U) & 0x1FU;
    const uint32_t                    rm          = (instruction >> 16U) & 0x1FU;
    bal_x86_assembler_t *BAL_RESTRICT assembler   = &compiler->assembler;

    bal_sliding_window_flush_all(&compiler->window);
    emit_load_vector(assembler, XMM0, rn, is_quad);
    emit_load_vector(assembler, XMM1, rm, true == is_quad || true == is_element);

    if (true == is_element)
    {
        const uint32_t index = (((instruction >> 11U) & 1U) << 1U) | ((instruction >> 21U) & 1U);
        emit_sse_immediate(assembler, 0x66, 0x70, XMM1, XMM1, (uint8_t)(index * 0x55U), "pshufd");
    }

    // XMM2:XMM0 and XMM3:XMM1 hold the halfwords of the low and high eight bytes.
    emit_sse(assembler, 0U, false, 0x28, XMM2, XMM0, "movaps");
    emit_sse(assembler, 0U, false, 0x28, XMM3, XMM1, "movaps");
    emit_widen(assembler, XMM0, 0U, is_unsigned, false);
    emit_widen(assembler, XMM1, 0U, is_unsigned, false);
    emit_widen(assembler, XMM2, 0U, is_unsigned, true);
    emit_widen(assembler, XMM3, 0U, is_unsigned, true);
    emit_sse(assembler, 0x66, false, 0xF5, XMM0, XMM1, "pmaddwd");
    emit_sse(assembler, 0x66, false, 0xF5, XMM2, XMM3, "pmaddwd");
    emit_sse38(assembler, 0x02, XMM0, XMM2, "phaddd");
    emit_load_vector(assembler, XMM1, rd, is_quad);
    emit_sse(assembler, 0x66, false, 0xFE, XMM0, XMM1, "paddd");
    emit_store_vector(assembler, XMM0, rd, is_quad);
    return true;
}

/// SQSHRN, UQSHRN, SQRSHRN, UQRSHRN, SQSHRUN and SQRSHRUN (vector with the "2" forms, and
/// scalar): the wide lanes shift right (arithmetically for the signed sources), then narrow with
/// saturation as SQXTN, UQXTN or SQXTUN does.
static bool
emit_fast_saturating_shift_narrow(bal_tier1_compiler_t *BAL_RESTRICT compiler,
                                  const uint32_t                     instruction)
{
    // 0 Q U 011110 immh immb opcode 1 Rn Rd: opcode 1001 R (SQSHRN, UQSHRN) or, with U = 1,
    // 1000 R (SQSHRUN).
    const uint32_t immh   = (instruction >> 19U) & 0xFU;
    const uint32_t u      = (instruction >> 29U) & 1U;
    const uint32_t opcode = (instruction >> 11U) & 0x1FU;

    const bool is_scalar = 0x5F000400U == (instruction & 0xDF800400U);

    if ((0x0F000400U != (instruction & 0x9F800400U) && false == is_scalar) || 0U == immh
        || immh > 7U || false == has_feature(compiler, BAL_TIER1_FEATURE_SSE42))
    {
        return false;
    }

    const bool is_unsigned_source    = 1U == u && (0x12U == opcode || 0x13U == opcode);
    const bool is_signed_to_unsigned = 1U == u && (0x10U == opcode || 0x11U == opcode);

    if (opcode != 0x12U && opcode != 0x13U && false == is_signed_to_unsigned)
    {
        return false;
    }

    const uint32_t size    = 0U != (immh & 4U) ? 2U : (0U != (immh & 2U) ? 1U : 0U);
    const uint32_t esize   = 8U << size;
    const uint32_t right   = (2U * esize) - ((instruction >> 16U) & 0x7FU);
    const bool     is_quad = false == is_scalar && 1U == ((instruction >> 30U) & 1U);
    const uint32_t rd      = instruction & 0x1FU;
    const uint32_t rn      = (instruction >> 5U) & 0x1FU;
    bal_x86_assembler_t *BAL_RESTRICT assembler = &compiler->assembler;

    bal_sliding_window_flush_all(&compiler->window);

    if (true == is_scalar)
    {
        emit_load_element(assembler, XMM0, rn, size + 1U);
    }
    else
    {
        emit_load_vector(assembler, XMM0, rn, true);
    }

    emit_shift_right_lanes(
        assembler, size + 1U, right, false == is_unsigned_source, 1U == (opcode & 1U));
    emit_saturating_narrow(assembler,
                           size,
                           true == is_unsigned_source      ? 0x34U
                           : true == is_signed_to_unsigned ? 0x32U
                                                           : 0x14U);

    if (true == is_scalar)
    {
        emit_store_element(assembler, XMM0, rd, size);
    }
    else
    {
        emit_store_narrow(assembler, rd, is_quad);
    }

    return true;
}

/// FCCMP and FCCMPE: the comparison of FCMP and FCMPE when the condition holds, and otherwise the
/// immediate flags.
static bool
emit_fast_conditional_compare(bal_tier1_compiler_t *BAL_RESTRICT compiler,
                              const uint32_t                     instruction)
{
    // M 0 S 11110 ftype 1 Rm cond 01 Rn op nzcv.
    const uint32_t ftype = (instruction >> 22U) & 0x3U;

    if (0x1E200400U != (instruction & 0xFF200C00U) || 2U == ftype
        || (3U == ftype && false == has_feature(compiler, BAL_TIER1_FEATURE_FMA)))
    {
        return false;
    }

    const uint32_t                    condition = (instruction >> 12U) & 0xFU;
    const uint32_t                    nzcv      = instruction & 0xFU;
    bal_x86_assembler_t *BAL_RESTRICT assembler = &compiler->assembler;
    size_t                            slow_jumps[1];
    size_t                            failing = 0U;

    slow_jumps[0] = begin_float_sequence(compiler);

    if (condition < 14U)
    {
        const bal_x86_condition_t fail_condition = emit_condition_value(compiler, condition);
        push_macro(compiler, BAL_X86_MACRO_TEST_REGISTER_REGISTER, SCRATCH_A, SCRATCH_A);
        bal_sliding_window_flush_all(&compiler->window);
        failing = emit_forward_jump(assembler, true, fail_condition);
    }

    if (3U == ftype)
    {
        // Half precision compares its exact single precision values.
        emit_load_half(assembler, XMM2, (instruction >> 5U) & 0x1FU);
        emit_halves_to_singles(assembler, XMM0, XMM2);
        emit_load_half(assembler, XMM3, (instruction >> 16U) & 0x1FU);
        emit_halves_to_singles(assembler, XMM1, XMM3);
        emit_compare_xmm_flags(assembler, false, 1U == ((instruction >> 4U) & 1U));
    }
    else
    {
        emit_compare_flags(assembler,
                           1U == ftype,
                           1U == ((instruction >> 4U) & 1U),
                           (instruction >> 5U) & 0x1FU,
                           (instruction >> 16U) & 0x1FU,
                           false);
    }

    if (condition < 14U)
    {
        // The four flag bytes from flag_c are C, Z, N and V.
        const size_t done = emit_forward_jump(assembler, false, BAL_X86_COND_O);
        patch_forward_jump(assembler, failing, true);
        bal_x86_emit_mov_r64_imm64(assembler,
                                   SCRATCH_A,
                                   ((nzcv >> 1U) & 1U) | (((nzcv >> 2U) & 1U) << 8U)
                                       | (((nzcv >> 3U) & 1U) << 16U) | ((nzcv & 1U) << 24U));
        bal_x86_emit_store_r32_rbp_offset(assembler, SCRATCH_A, offsetof(bal_cpu_t, flag_c));
        patch_forward_jump(assembler, done, false);
    }

    emit_slow_path(compiler, instruction, float_helper(instruction), slow_jumps, 1U);
    return true;
}

/// AESE, AESD, AESMC and AESIMC with AES-NI, and PMULL and PMULL2 on doublewords with PCLMULQDQ.
/// AESE is AESENCLAST of Rd ^ Rn with a zero key and AESD AESDECLAST; MixColumns alone is
/// AESENC after AESDECLAST, whose inverse ShiftRows and SubBytes AESENC's undo.
static bool
emit_fast_crypto(bal_tier1_compiler_t *BAL_RESTRICT compiler, const uint32_t instruction)
{
    const uint32_t form     = instruction & 0xFFFFFC00U;
    const bool     is_pmull = 0x0EE0E000U == (instruction & 0xBFE0FC00U);

    if ((false == is_pmull && form != 0x4E284800U && form != 0x4E285800U && form != 0x4E286800U
         && form != 0x4E287800U)
        || false == has_feature(compiler, BAL_TIER1_FEATURE_AES))
    {
        return false;
    }

    const uint32_t                    rd        = instruction & 0x1FU;
    const uint32_t                    rn        = (instruction >> 5U) & 0x1FU;
    bal_x86_assembler_t *BAL_RESTRICT assembler = &compiler->assembler;

    bal_sliding_window_flush_all(&compiler->window);
    emit_load_vector(assembler, XMM0, rn, true);

    if (true == is_pmull)
    {
        const bool is_upper = 1U == ((instruction >> 30U) & 1U);
        emit_load_vector(assembler, XMM1, (instruction >> 16U) & 0x1FU, true);
        emit_sse3a(assembler, 0x44, XMM0, XMM1, true == is_upper ? 0x11 : 0x00, "pclmulqdq");
        emit_store_vector(assembler, XMM0, rd, true);
        return true;
    }

    emit_zero_vector(assembler, XMM1);

    switch (form)
    {
        case 0x4E284800U:
        case 0x4E285800U:
            emit_load_vector(assembler, XMM2, rd, true);
            emit_sse(assembler, 0x66, false, 0xEF, XMM0, XMM2, "pxor");
            emit_sse38(
                assembler, 0x4E284800U == form ? 0xDD : 0xDF, XMM0, XMM1, "aesenclast/aesdeclast");
            break;
        case 0x4E286800U:
            emit_sse38(assembler, 0xDF, XMM0, XMM1, "aesdeclast");
            emit_sse38(assembler, 0xDC, XMM0, XMM1, "aesenc");
            break;
        default:
            emit_sse38(assembler, 0xDB, XMM0, XMM0, "aesimc");
            break;
    }

    emit_store_vector(assembler, XMM0, rd, true);
    return true;
}

/// Loads the half precision scalar of SIMD&FP register `index` into `xmm`, clearing the rest.
static void
emit_load_half(bal_x86_assembler_t *BAL_RESTRICT assembler, const uint8_t xmm, const uint32_t index)
{
    emit_load_scalar(assembler, xmm, index, false);
    emit_sse_immediate(assembler, 0x66, 0x72, 6U, xmm, 16U, "pslld");
    emit_sse_immediate(assembler, 0x66, 0x72, 2U, xmm, 16U, "psrld");
}

/// FSQRT and FNMUL (scalar) in single and double precision, and FADD, FSUB, FMUL, FDIV, FNMUL and
/// FSQRT (scalar) in half precision, which compute in single precision and round as
/// emit_fast_half_arithmetic() describes. FNMUL negates the rounded product, as AArch64 does.
static bool
emit_fast_scalar_more(bal_tier1_compiler_t *BAL_RESTRICT compiler, const uint32_t instruction)
{
    // FSQRT: 0 0 0 11110 ftype 1 0000 11 10000 Rn Rd. Arithmetic: M 0 S 11110 ftype 1 Rm opcode 10
    // Rn Rd with FMUL, FDIV, FADD and FSUB (opcodes 0-3, half precision only here) and FNMUL (8).
    static const uint8_t OPERATIONS[4] = { 0x59, 0x5E, 0x58, 0x5C };
    const uint32_t       ftype         = (instruction >> 22U) & 0x3U;
    const uint32_t       opcode        = (instruction >> 12U) & 0xFU;
    const bool           is_half       = 3U == ftype;
    const bool           is_root       = 0x1E21C000U == (instruction & 0xFF3FFC00U);
    const bool           is_negated    = 0x1E208800U == (instruction & 0xFF20FC00U);
    const bool           is_arithmetic
        = 0x1E200800U == (instruction & 0xFF200C00U) && opcode <= 3U && is_half;

    if ((false == is_root && false == is_negated && false == is_arithmetic) || 2U == ftype
        || (true == is_half && false == has_feature(compiler, BAL_TIER1_FEATURE_FMA)))
    {
        return false;
    }

    const uint8_t                     operation = true == is_root      ? 0x51
                                                  : true == is_negated ? 0x59
                                                                       : OPERATIONS[opcode];
    const bool                        is_double = 1U == ftype;
    const uint32_t                    rd        = instruction & 0x1FU;
    const uint32_t                    rn        = (instruction >> 5U) & 0x1FU;
    const uint32_t                    rm        = (instruction >> 16U) & 0x1FU;
    bal_x86_assembler_t *BAL_RESTRICT assembler = &compiler->assembler;
    size_t                            slow_jumps[3];
    uint32_t                          count = 1U;

    slow_jumps[0] = begin_float_sequence(compiler);

    if (true == is_half)
    {
        emit_load_half(assembler, XMM2, rn);
        emit_halves_to_singles(assembler, XMM0, XMM2);

        if (false == is_root)
        {
            emit_load_half(assembler, XMM3, rm);
            emit_halves_to_singles(assembler, XMM1, XMM3);
        }

        emit_sse(assembler, 0xF3, false, operation, XMM0, true == is_root ? XMM0 : XMM1, "op");
        emit_singles_to_halves(assembler, XMM0, XMM0);
        slow_jumps[count++] = emit_half_checks(assembler);
        emit_sse(assembler, 0x66, false, 0x7E, XMM0, (uint8_t)SCRATCH_A, "movd");
        bal_x86_emit_shl_r32_imm8(assembler, SCRATCH_A, 16U);
        bal_x86_emit_shr_r32_imm8(assembler, SCRATCH_A, 16U);
    }
    else
    {
        const uint8_t prefix = true == is_double ? 0xF2 : 0xF3;
        emit_load_scalar(assembler, XMM0, rn, is_double);

        if (true == is_root)
        {
            emit_sse(assembler, prefix, false, 0x51, XMM0, XMM0, "sqrts");
        }
        else
        {
            emit_sse_memory(
                assembler, prefix, false, 0x59, XMM0, offsetof(bal_cpu_t, v[rm][0]), "muls");
        }

        count += emit_scalar_checks(compiler, is_double, &slow_jumps[1]);
    }

    if (true == is_negated)
    {
        bal_x86_emit_mov_r64_imm64(assembler,
                                   SCRATCH_B,
                                   true == is_half     ? 0x8000U
                                   : true == is_double ? 0x8000000000000000ULL
                                                       : 0x80000000U);
        bal_x86_emit_xor_r64_r64(assembler, SCRATCH_A, SCRATCH_B);
    }

    emit_store_scalar_result(assembler, rd);
    emit_slow_path(compiler, instruction, float_helper(instruction), slow_jumps, count);
    return true;
}

/// FCMP and FCMPE in half precision: the halves convert to single precision exactly, and the
/// comparison is the single precision one.
static bool
emit_fast_half_compare(bal_tier1_compiler_t *BAL_RESTRICT compiler, const uint32_t instruction)
{
    // 0 0 0 11110 11 1 Rm 00 1000 Rn opcode2.
    if (0x1EE02000U != (instruction & 0xFFE0FC07U)
        || false == has_feature(compiler, BAL_TIER1_FEATURE_FMA))
    {
        return false;
    }

    bal_x86_assembler_t *BAL_RESTRICT assembler = &compiler->assembler;
    size_t                            slow_jumps[1];

    slow_jumps[0] = begin_float_sequence(compiler);
    emit_load_half(assembler, XMM2, (instruction >> 5U) & 0x1FU);
    emit_halves_to_singles(assembler, XMM0, XMM2);

    if (1U == ((instruction >> 3U) & 1U))
    {
        emit_zero_vector(assembler, XMM1);
    }
    else
    {
        emit_load_half(assembler, XMM3, (instruction >> 16U) & 0x1FU);
        emit_halves_to_singles(assembler, XMM1, XMM3);
    }

    emit_compare_xmm_flags(assembler, false, 1U == ((instruction >> 4U) & 1U));
    emit_slow_path(compiler, instruction, float_helper(instruction), slow_jumps, 1U);
    return true;
}

/// FRINTA (vector) in single and double precision with emit_round_ties_away_packed(); infinities
/// and NaNs, which would raise Invalid there, take the slow path.
static bool
emit_fast_round_away(bal_tier1_compiler_t *BAL_RESTRICT compiler, const uint32_t instruction)
{
    // 0 Q 1 01110 0 sz 10000 11000 10 Rn Rd.
    const bool is_double = 1U == ((instruction >> 22U) & 1U);
    const bool is_quad   = 1U == ((instruction >> 30U) & 1U);

    if (0x2E218800U != (instruction & 0xBFBFFC00U) || (true == is_double && false == is_quad)
        || false == has_feature(compiler, BAL_TIER1_FEATURE_SSE41))
    {
        return false;
    }

    bal_x86_assembler_t *BAL_RESTRICT assembler = &compiler->assembler;
    size_t                            slow_jumps[2];

    slow_jumps[0] = begin_float_sequence(compiler);
    emit_load_vector(assembler, XMM0, (instruction >> 5U) & 0x1FU, is_quad);
    emit_broadcast_pattern(
        assembler, XMM1, true == is_double ? 0x7FF0000000000000ULL : 0x7F8000007F800000ULL);
    emit_sse(assembler, 0U, false, 0x28, XMM2, XMM0, "movaps");
    emit_sse(assembler, 0x66, false, 0xDB, XMM2, XMM1, "pand");
    emit_lanes_equal(assembler, true == is_double ? 3U : 2U, XMM2, XMM1);
    emit_sse(assembler, 0x66, false, 0xD7, (uint8_t)SCRATCH_A, XMM2, "pmovmskb");
    bal_x86_emit_test_r32_r32(assembler, SCRATCH_A, SCRATCH_A);
    slow_jumps[1] = emit_forward_jump(assembler, true, BAL_X86_COND_NE);
    emit_round_ties_away_packed(assembler, is_double, XMM1, XMM0, XMM2, XMM3);
    emit_store_vector(assembler, XMM1, instruction & 0x1FU, is_quad);
    emit_slow_path(compiler, instruction, (uint64_t)(uintptr_t)&bal_execute_simd, slow_jumps, 2U);
    return true;
}

/// Half precision vector operations computed lane by lane in single precision, four lanes at a
/// time: FSQRT, FRINTN, FRINTM, FRINTP, FRINTZ, FRINTA, FRINTX, FRINTI, SCVTF and UCVTF (from
/// halfwords), FCVT{N,P,M,Z,A}{S,U} (to halfwords), the comparisons with zero, and FCMEQ, FCMGE,
/// FCMGT, FACGE and FACGT; the conversions and comparisons also on the scalar, as lane 0.
/// Rounding results and conversions from integers round once to half precision, and
/// emit_half_checks() sends NaNs and the smallest normal magnitude to the slow path; comparisons
/// narrow their masks with PACKSSDW. The conversions to integers round as
/// emit_fast_vector_convert() does and narrow with saturation, a lane out of range taking the
/// slow path. FRINTA's infinities and NaNs take the slow path. FABS and FNEG only change sign
/// bits.
static bool
emit_fast_half_vector(bal_tier1_compiler_t *BAL_RESTRICT compiler, const uint32_t instruction)
{
    // Two registers: 0 Q U 01110 a 1111 00 opcode 10 Rn Rd, or the scalar 01 U 11110 a 1111 00
    // opcode 10 Rn Rd. Three: 0 Q U 01110 E 10 Rm 00 opcode 1 Rn Rd, or the scalar 01 U 11110 ...
    const uint32_t scalar_three    = instruction & 0xFFE0FC00U;
    const bool     is_scalar_two   = 0x5E780800U == (instruction & 0xDF7E0C00U);
    const bool     is_scalar_three = 0x5E402400U == scalar_three || 0x7E402400U == scalar_three
                                 || 0x7EC02400U == scalar_three || 0x7E402C00U == scalar_three
                                 || 0x7EC02C00U == scalar_three;
    const bool     is_scalar = true == is_scalar_two || true == is_scalar_three;
    const bool     is_two    = true == is_scalar_two || 0x0E780800U == (instruction & 0x9F7E0C00U);
    const uint32_t three
        = true == is_scalar_three ? scalar_three - 0x50000000U : instruction & 0xBFE0FC00U;
    const uint32_t key = (((instruction >> 29U) & 1U) << 6U) | (((instruction >> 23U) & 1U) << 5U)
                         | ((instruction >> 12U) & 0x1FU);
    uint8_t  rounding = 0xFFU;
    uint32_t relation = 0xFFU; // 0 EQ, 1 GE, 2 GT, 3 LE, 4 LT, with 8 for absolute values.
    uint32_t convert  = 0xFFU; // The rounding mode of a conversion to integers, 6 ties away.
    bool     is_zero  = false;
    bool     is_unsigned_convert = false;

    if (false == has_feature(compiler, BAL_TIER1_FEATURE_FMA)
        || false == has_feature(compiler, BAL_TIER1_FEATURE_SSE41))
    {
        return false;
    }

    if (true == is_two)
    {
        switch (key)
        {
            case 0x18U: // FRINTN
                rounding = 0x08;
                break;
            case 0x19U: // FRINTM
                rounding = 0x09;
                break;
            case 0x38U: // FRINTP
                rounding = 0x0A;
                break;
            case 0x39U: // FRINTZ
                rounding = 0x0B;
                break;
            case 0x59U: // FRINTX
                rounding = 0x04;
                break;
            case 0x79U: // FRINTI
                rounding = 0x0C;
                break;
            case 0x2DU: // FCMEQ #0
                relation = 0U;
                is_zero  = true;
                break;
            case 0x6CU: // FCMGE #0
                relation = 1U;
                is_zero  = true;
                break;
            case 0x2CU: // FCMGT #0
                relation = 2U;
                is_zero  = true;
                break;
            case 0x6DU: // FCMLE #0
                relation = 3U;
                is_zero  = true;
                break;
            case 0x2EU: // FCMLT #0
                relation = 4U;
                is_zero  = true;
                break;
            case 0x58U: // FRINTA
                rounding = 0xFEU;
                break;
            case 0x5AU: // FCVTNU
            case 0x7AU: // FCVTPU
            case 0x5BU: // FCVTMU
            case 0x7BU: // FCVTZU
            case 0x5CU: // FCVTAU
                is_unsigned_convert = true;
                convert             = 0x5CU == key ? 6U : ((key >> 5U) & 1U) | ((key & 1U) << 1U);
                break;
            case 0x1AU: // FCVTNS
            case 0x3AU: // FCVTPS
            case 0x1BU: // FCVTMS
            case 0x3BU: // FCVTZS
            case 0x1CU: // FCVTAS
                convert = 0x1CU == key ? 6U : ((key >> 5U) & 1U) | ((key & 1U) << 1U);
                break;
            case 0x1DU: // SCVTF
            case 0x5DU: // UCVTF
            case 0x7FU: // FSQRT
            case 0x2FU: // FABS
            case 0x6FU: // FNEG
                break;
            default:
                return false;
        }

        // The scalar forms here are the conversions and the comparisons with zero.
        if (true == is_scalar && rounding != 0xFFU)
        {
            return false;
        }

        if (true == is_scalar && (0x7FU == key || 0x2FU == key || 0x6FU == key))
        {
            return false;
        }
    }
    else
    {
        switch (three)
        {
            case 0x0E402400U:
                relation = 0U;
                break;
            case 0x2E402400U:
                relation = 1U;
                break;
            case 0x2EC02400U:
                relation = 2U;
                break;
            case 0x2E402C00U:
                relation = 9U;
                break;
            case 0x2EC02C00U:
                relation = 10U;
                break;
            default:
                return false;
        }
    }

    const bool     is_quad = false == is_scalar && 1U == ((instruction >> 30U) & 1U);
    const uint32_t rd      = instruction & 0x1FU;
    const uint32_t rn      = (instruction >> 5U) & 0x1FU;
    const uint32_t rm      = (instruction >> 16U) & 0x1FU;
    bal_x86_assembler_t *BAL_RESTRICT assembler = &compiler->assembler;

    if (true == is_two && (0x2FU == key || 0x6FU == key))
    {
        bal_sliding_window_flush_all(&compiler->window);
        emit_load_vector(assembler, XMM0, rn, is_quad);
        emit_broadcast_pattern(
            assembler, XMM1, 0x2FU == key ? 0x7FFF7FFF7FFF7FFFULL : 0x8000800080008000ULL);
        emit_sse(assembler, 0x66, false, 0x2FU == key ? 0xDB : 0xEF, XMM0, XMM1, "pand/pxor");
        emit_store_vector(assembler, XMM0, rd, is_quad);
        return true;
    }

    size_t   slow_jumps[6];
    uint32_t count = 1U;
    slow_jumps[0]  = begin_float_sequence(compiler);

    if (true == is_scalar)
    {
        emit_load_half(assembler, XMM4, rn);
    }
    else
    {
        emit_load_vector(assembler, XMM4, rn, is_quad);
    }

    if (relation != 0xFFU && false == is_zero)
    {
        if (true == is_scalar)
        {
            emit_load_half(assembler, XMM5, rm);
        }
        else
        {
            emit_load_vector(assembler, XMM5, rm, is_quad);
        }
    }

    // The low four lanes go to xmm0's low half, the high four to its high half.
    for (uint32_t part = 0U; part < (true == is_quad ? 2U : 1U); ++part)
    {
        const uint8_t result = 0U == part ? XMM0 : XMM1;

        if (1U == part)
        {
            emit_sse_immediate(assembler, 0x66, 0x73, 3U, XMM4, 8U, "psrldq");
            emit_sse_immediate(assembler, 0x66, 0x73, 3U, XMM5, 8U, "psrldq");
        }

        if (0x1DU == key || 0x5DU == key)
        {
            emit_sse(assembler, 0U, false, 0x28, result, XMM4, "movaps");
            emit_widen(assembler, result, 1U, 0x5DU == key, false);
            emit_sse(assembler, 0U, false, 0x5B, result, result, "cvtdq2ps");
        }
        else
        {
            emit_halves_to_singles(assembler, result, XMM4);
        }

        if (convert != 0xFFU)
        {
            if (6U == convert)
            {
                emit_round_ties_away_packed(assembler, false, XMM2, result, XMM3, XMM5);
            }
            else
            {
                emit_sse3a(assembler, 0x08, XMM2, result, ROUND_SUPPRESSED[convert], "roundps");
            }

            // In range where narrowing with saturation and widening back gives the integer, and
            // only then Inexact.
            emit_sse(assembler, 0xF3, false, 0x5B, XMM2, XMM2, "cvttps2dq");
            emit_sse(assembler, 0U, false, 0x28, XMM3, XMM2, "movaps");

            if (true == is_unsigned_convert)
            {
                emit_sse38(assembler, 0x2B, XMM3, XMM3, "packusdw");
            }
            else
            {
                emit_sse(assembler, 0x66, false, 0x6B, XMM3, XMM3, "packssdw");
            }

            emit_widen(assembler, XMM3, 1U, is_unsigned_convert, false);
            emit_lanes_equal(assembler, 2U, XMM3, XMM2);
            emit_sse(assembler, 0U, false, 0x50, (uint8_t)SCRATCH_A, XMM3, "movmskps");
            bal_x86_emit_sub_r32_imm32(assembler, SCRATCH_A, 0xF);
            slow_jumps[count++] = emit_forward_jump(assembler, true, BAL_X86_COND_NE);
            emit_sse3a(assembler, 0x08, XMM3, result, 0x03, "roundps");

            if (true == is_unsigned_convert)
            {
                emit_sse38(assembler, 0x2B, XMM2, XMM2, "packusdw");
            }
            else
            {
                emit_sse(assembler, 0x66, false, 0x6B, XMM2, XMM2, "packssdw");
            }

            emit_sse(assembler, 0U, false, 0x28, result, XMM2, "movaps");
            continue;
        }

        if (0xFEU == rounding)
        {
            emit_broadcast_pattern(assembler, XMM2, 0x7F8000007F800000ULL);
            emit_sse(assembler, 0U, false, 0x28, XMM3, result, "movaps");
            emit_sse(assembler, 0x66, false, 0xDB, XMM3, XMM2, "pand");
            emit_sse(assembler, 0x66, false, 0x76, XMM3, XMM2, "pcmpeqd");
            emit_sse(assembler, 0U, false, 0x50, (uint8_t)SCRATCH_A, XMM3, "movmskps");
            bal_x86_emit_test_r32_r32(assembler, SCRATCH_A, SCRATCH_A);
            slow_jumps[count++] = emit_forward_jump(assembler, true, BAL_X86_COND_NE);
            emit_round_ties_away_packed(assembler, false, XMM2, result, XMM3, XMM5);
            emit_sse(assembler, 0U, false, 0x28, result, XMM2, "movaps");
        }
        else if (rounding != 0xFFU)
        {
            emit_sse3a(assembler, 0x08, result, result, rounding, "roundps");
        }
        else if (0x7FU == key)
        {
            emit_sse(assembler, 0U, false, 0x51, result, result, "sqrtps");
        }
        else if (relation != 0xFFU)
        {
            if (true == is_zero)
            {
                emit_zero_vector(assembler, XMM3);
            }
            else
            {
                emit_halves_to_singles(assembler, XMM3, XMM5);
            }

            if (relation >= 8U)
            {
                emit_all_ones(assembler, XMM2);
                emit_sse_immediate(assembler, 0x66, 0x72, 2U, XMM2, 1U, "psrld");
                emit_sse(assembler, 0U, false, 0x54, result, XMM2, "andps");
                emit_sse(assembler, 0U, false, 0x54, XMM3, XMM2, "andps");
            }

            // n >= m is m <= n and n > m is m < n; the masks end in `result`.
            switch (relation & 7U)
            {
                case 0U:
                    emit_sse_immediate(assembler, 0U, 0xC2, result, XMM3, 0U, "cmpeqps");
                    break;
                case 1U:
                    emit_sse_immediate(assembler, 0U, 0xC2, XMM3, result, 2U, "cmpleps");
                    emit_sse(assembler, 0U, false, 0x28, result, XMM3, "movaps");
                    break;
                case 2U:
                    emit_sse_immediate(assembler, 0U, 0xC2, XMM3, result, 1U, "cmpltps");
                    emit_sse(assembler, 0U, false, 0x28, result, XMM3, "movaps");
                    break;
                case 3U:
                    emit_sse_immediate(assembler, 0U, 0xC2, result, XMM3, 2U, "cmpleps");
                    break;
                default:
                    emit_sse_immediate(assembler, 0U, 0xC2, result, XMM3, 1U, "cmpltps");
                    break;
            }

            emit_sse(assembler, 0x66, false, 0x6B, result, result, "packssdw");
            continue;
        }

        emit_singles_to_halves(assembler, result, result);
    }

    if (true == is_quad)
    {
        emit_sse(assembler, 0x66, false, 0x6C, XMM0, XMM1, "punpcklqdq");
    }

    if (0xFFU == relation && 0xFFU == convert)
    {
        slow_jumps[count++] = emit_half_checks(assembler);
    }

    if (true == is_scalar)
    {
        emit_store_element(assembler, XMM0, rd, 1U);
    }
    else
    {
        emit_store_vector(assembler, XMM0, rd, is_quad);
    }
    emit_slow_path(
        compiler, instruction, (uint64_t)(uintptr_t)&bal_execute_simd, slow_jumps, count);
    return true;
}

// -----------------------------------------------------------------------------
// Inline shifts by register (AVX2)
// -----------------------------------------------------------------------------

/// The kinds of saturation of emit_variable_shift().
typedef enum
{
    SATURATE_NONE,
    SATURATE_SIGNED,
    SATURATE_UNSIGNED,
    SATURATE_SIGNED_TO_UNSIGNED,
} saturation_t;

/// Emits the AVX2 VPSLLV, VPSRLV or VPSRAV (`opcode` 0x47, 0x45 or 0x46) `destination` = `value`
/// shifted by the lanes of `count`, on 64-bit lanes with `is_64bit`.
static void
emit_variable_lane_shift(bal_x86_assembler_t *BAL_RESTRICT assembler,
                         const uint8_t                     opcode,
                         const bool                        is_64bit,
                         const uint8_t                     destination,
                         const uint8_t                     value,
                         const uint8_t                     count)
{
    bal_x86_emit_vex_0f38_register(assembler, is_64bit, opcode, destination, value, count, "vpsv");
}

/// `destination` = `value` shifted right arithmetically by the lanes of `count`. 64-bit lanes,
/// which AVX2 cannot shift so, shift logically with the sign complemented in and out. Uses
/// `scratch` for those.
static void
emit_variable_arithmetic_shift(bal_x86_assembler_t *BAL_RESTRICT assembler,
                               const bool                        is_64bit,
                               const uint8_t                     destination,
                               const uint8_t                     value,
                               const uint8_t                     count,
                               const uint8_t                     scratch)
{
    if (false == is_64bit)
    {
        emit_variable_lane_shift(assembler, 0x46, false, destination, value, count);
        return;
    }

    emit_zero_vector(assembler, scratch);
    emit_lanes_greater(assembler, 3U, scratch, value);
    emit_sse(assembler, 0U, false, 0x28, destination, value, "movaps");
    emit_sse(assembler, 0x66, false, 0xEF, destination, scratch, "pxor");
    emit_variable_lane_shift(assembler, 0x45, true, destination, destination, count);
    emit_sse(assembler, 0x66, false, 0xEF, destination, scratch, "pxor");
}

/// Shifts the `8 << size`-bit elements of SIMD&FP register `rn` by the signed low bytes of the
/// elements of `rm` (or, when `rm` is above 31, by the constant `rm` - 32), into `rd`: left for
/// positive counts and right for negative ones, logically or (`is_signed`) arithmetically, with
/// `is_rounding` adding the last bit shifted out, and with `saturation` on left shifts that lose
/// bits, setting FPSR.QC. These are SSHL, USHL, SRSHL, URSHL, SQSHL, UQSHL, SQRSHL and UQRSHL,
/// and SQSHL, UQSHL and SQSHLU by immediate.
///
/// The elements are widened to 32-bit lanes (64-bit ones stay), four at a time, so that VPSLLV
/// and VPSRLV give zero for counts of the lane size and more, as the instructions do; each group
/// of four writes its own elements of `rd`, which the later groups do not read. A left shift
/// saturates where shifting it back does not give the element or the result leaves the element
/// size. With `is_scalar` only element 0 is kept.
static void
emit_variable_shift(bal_tier1_compiler_t *BAL_RESTRICT compiler,
                    const uint32_t                     rd,
                    const uint32_t                     rn,
                    const uint32_t                     rm,
                    const uint32_t                     size,
                    const bool                         is_quad,
                    const bool                         is_scalar,
                    const bool                         is_signed,
                    const bool                         is_rounding,
                    const saturation_t                 saturation)
{
    bal_x86_assembler_t *BAL_RESTRICT assembler = &compiler->assembler;
    const uint32_t                    esize     = 8U << size;
    const bool                        is_64bit  = 3U == size;
    const uint32_t                    width     = true == is_64bit ? 64U : 32U;
    const uint32_t       lanes  = true == is_scalar ? 1U : (true == is_quad ? 128U : 64U) / esize;
    const uint32_t       groups = true == is_64bit ? 1U : (lanes + 3U) / 4U;
    const bool           is_constant = rm >= 32U;
    const uint8_t        sub_op      = true == is_64bit ? 0xFB : 0xFA;
    const uint8_t        add_op      = true == is_64bit ? 0xD4 : 0xFE;
    const uint64_t       maximum     = SATURATE_SIGNED == saturation ? (1ULL << (esize - 1U)) - 1U
                                       : 64U == esize                ? ~0ULL
                                                                     : (1ULL << esize) - 1U;
    static const uint8_t SIGN_EXTEND[2] = { 0x21, 0x23 }; // PMOVSXBD, PMOVSXWD
    static const uint8_t ZERO_EXTEND[2] = { 0x31, 0x33 }; // PMOVZXBD, PMOVZXWD

    bal_sliding_window_flush_all(&compiler->window);

    for (uint32_t group = 0U; group < groups; ++group)
    {
        // XMM0 = the elements, extended to the lane width; XMM1 = the counts.
        // The unused lanes of a 64-bit vector or a scalar hold zero, which shifts without
        // saturating.
        const uint8_t skip = (uint8_t)(group * 4U * (esize / 8U));
        emit_load_vector(assembler, XMM0, rn, true == is_quad);

        if (skip != 0U)
        {
            emit_sse_immediate(assembler, 0x66, 0x73, 3U, XMM0, skip, "psrldq");
        }

        if (esize < 32U)
        {
            emit_sse38(assembler,
                       true == is_signed ? SIGN_EXTEND[size] : ZERO_EXTEND[size],
                       XMM0,
                       XMM0,
                       "pmovsx/pmovzx");
        }

        if (true == is_scalar)
        {
            emit_load_vector_constant(
                assembler, XMM2, true == is_64bit ? ~0ULL : 0xFFFFFFFFULL, 0U);
            emit_sse(assembler, 0x66, false, 0xDB, XMM0, XMM2, "pand");
        }

        if (true == is_constant)
        {
            emit_broadcast_pattern(
                assembler, XMM1, true == is_64bit ? rm - 32U : 0x0000000100000001ULL * (rm - 32U));
        }
        else
        {
            emit_load_vector(assembler, XMM1, rm, true);

            if (skip != 0U)
            {
                emit_sse_immediate(assembler, 0x66, 0x73, 3U, XMM1, skip, "psrldq");
            }

            if (esize < 32U)
            {
                emit_sse38(assembler, SIGN_EXTEND[size], XMM1, XMM1, "pmovsx");
            }

            if (true == is_64bit)
            {
                // The low byte, sign-extended: (b ^ 0x80) - 0x80.
                emit_broadcast_pattern(assembler, XMM2, 0xFFULL);
                emit_sse(assembler, 0x66, false, 0xDB, XMM1, XMM2, "pand");
                emit_broadcast_pattern(assembler, XMM2, 0x80ULL);
                emit_sse(assembler, 0x66, false, 0xEF, XMM1, XMM2, "pxor");
                emit_sse(assembler, 0x66, false, 0xFB, XMM1, XMM2, "psubq");
            }
            else if (esize != 8U)
            {
                emit_sse_immediate(assembler, 0x66, 0x72, 6U, XMM1, 24U, "pslld");
                emit_sse_immediate(assembler, 0x66, 0x72, 4U, XMM1, 24U, "psrad");
            }
        }

        // XMM3 = the left shift; a negative count, huge as an unsigned one, gives zero.
        emit_variable_lane_shift(assembler, 0x47, is_64bit, XMM3, XMM0, XMM1);

        if (saturation != SATURATE_NONE)
        {
            // XMM4 = the lanes to keep: where shifting back gives the element and the result
            // fits the element, or where the count is negative.
            if (SATURATE_SIGNED == saturation)
            {
                emit_variable_arithmetic_shift(assembler, is_64bit, XMM4, XMM3, XMM1, XMM5);
            }
            else
            {
                emit_variable_lane_shift(assembler, 0x45, is_64bit, XMM4, XMM3, XMM1);
            }

            emit_lanes_equal(assembler, true == is_64bit ? 3U : 2U, XMM4, XMM0);

            if (esize < width || saturation != SATURATE_SIGNED)
            {
                emit_sse(assembler, 0U, false, 0x28, XMM5, XMM3, "movaps");

                if (SATURATE_SIGNED == saturation)
                {
                    emit_sse_immediate(
                        assembler, 0x66, 0x72, 6U, XMM5, (uint8_t)(32U - esize), "pslld");
                    emit_sse_immediate(
                        assembler, 0x66, 0x72, 4U, XMM5, (uint8_t)(32U - esize), "psrad");
                    emit_lanes_equal(assembler, 2U, XMM5, XMM3);
                }
                else
                {
                    // Nothing above the element (and no sign for SQSHLU).
                    if (esize < width)
                    {
                        emit_sse_immediate(
                            assembler, 0x66, 0x72, 2U, XMM5, (uint8_t)esize, "psrld");
                    }
                    else
                    {
                        emit_zero_vector(assembler, XMM5);
                    }

                    emit_zero_vector(assembler, XMM2);
                    emit_lanes_equal(assembler, true == is_64bit ? 3U : 2U, XMM5, XMM2);

                    if (SATURATE_SIGNED_TO_UNSIGNED == saturation)
                    {
                        emit_zero_vector(assembler, XMM2);
                        emit_lanes_greater(assembler, true == is_64bit ? 3U : 2U, XMM2, XMM0);
                        emit_sse(assembler, 0x66, false, 0xDF, XMM2, XMM5, "pandn");
                        emit_sse(assembler, 0U, false, 0x28, XMM5, XMM2, "movaps");
                    }
                }

                emit_sse(assembler, 0x66, false, 0xDB, XMM4, XMM5, "pand");
            }

            emit_zero_vector(assembler, XMM2);
            emit_lanes_greater(assembler, true == is_64bit ? 3U : 2U, XMM2, XMM1);
            emit_sse(assembler, 0x66, false, 0xEB, XMM4, XMM2, "por");
            emit_sse(assembler, 0x66, false, 0xD7, (uint8_t)SCRATCH_A, XMM4, "pmovmskb");
            bal_x86_emit_sub_r32_imm32(assembler, SCRATCH_A, 0xFFFF);
            const size_t kept = emit_forward_jump(assembler, true, BAL_X86_COND_E);
            emit_set_saturation(assembler);
            patch_forward_jump(assembler, kept, true);

            // XMM5 = the saturated value: the maximum, its complement (the minimum) for negative
            // signed elements, or zero for SQSHLU's.
            emit_broadcast_pattern(
                assembler, XMM5, true == is_64bit ? maximum : 0x0000000100000001ULL * maximum);

            if (saturation != SATURATE_UNSIGNED)
            {
                emit_zero_vector(assembler, XMM2);
                emit_lanes_greater(assembler, true == is_64bit ? 3U : 2U, XMM2, XMM0);

                if (SATURATE_SIGNED == saturation)
                {
                    emit_sse(assembler, 0x66, false, 0xEF, XMM5, XMM2, "pxor");
                }
                else
                {
                    emit_sse(assembler, 0x66, false, 0xDF, XMM2, XMM5, "pandn");
                    emit_sse(assembler, 0U, false, 0x28, XMM5, XMM2, "movaps");
                }
            }

            emit_sse(assembler, 0x66, false, 0xDB, XMM3, XMM4, "pand");
            emit_sse(assembler, 0x66, false, 0xDF, XMM4, XMM5, "pandn");
            emit_sse(assembler, 0x66, false, 0xEB, XMM3, XMM4, "por");
        }

        // XMM4 = the right shift count; XMM5 = the right shift.
        emit_zero_vector(assembler, XMM4);
        emit_sse(assembler, 0x66, false, sub_op, XMM4, XMM1, "psub");

        if (true == is_signed && true == is_rounding)
        {
            // Shifting a signed element right by more than its size rounds to zero as shifting it
            // by its size does: the sign plus the rounding bit, its sign.
            emit_broadcast_pattern(
                assembler, XMM5, true == is_64bit ? esize : 0x0000000100000001ULL * esize);

            if (true == is_64bit)
            {
                emit_sse(assembler, 0U, false, 0x28, XMM2, XMM4, "movaps");
                emit_lanes_greater(assembler, 3U, XMM2, XMM5);
                emit_sse(assembler, 0x66, false, 0xDB, XMM5, XMM2, "pand");
                emit_sse(assembler, 0x66, false, 0xDF, XMM2, XMM4, "pandn");
                emit_sse(assembler, 0x66, false, 0xEB, XMM2, XMM5, "por");
                emit_sse(assembler, 0U, false, 0x28, XMM4, XMM2, "movaps");
            }
            else
            {
                emit_sse38(assembler, 0x39, XMM4, XMM5, "pminsd");
            }
        }

        if (true == is_signed)
        {
            emit_variable_arithmetic_shift(assembler, is_64bit, XMM5, XMM0, XMM4, XMM2);
        }
        else
        {
            emit_variable_lane_shift(assembler, 0x45, is_64bit, XMM5, XMM0, XMM4);
        }

        if (true == is_rounding)
        {
            // Bit count - 1 of the element, at bit 0.
            emit_all_ones(assembler, XMM2);
            emit_sse(assembler, 0x66, false, add_op, XMM2, XMM4, "padd");
            emit_variable_lane_shift(assembler, 0x45, is_64bit, XMM2, XMM0, XMM2);
            emit_sse_immediate(assembler,
                               0x66,
                               true == is_64bit ? 0x73 : 0x72,
                               6U,
                               XMM2,
                               (uint8_t)(width - 1U),
                               "psll");
            emit_sse_immediate(assembler,
                               0x66,
                               true == is_64bit ? 0x73 : 0x72,
                               2U,
                               XMM2,
                               (uint8_t)(width - 1U),
                               "psrl");
            emit_sse(assembler, 0x66, false, add_op, XMM5, XMM2, "padd");
        }

        // XMM2 = the right shift where the count is negative, the left one elsewhere.
        emit_zero_vector(assembler, XMM2);
        emit_lanes_greater(assembler, true == is_64bit ? 3U : 2U, XMM2, XMM1);
        emit_sse(assembler, 0x66, false, 0xDB, XMM5, XMM2, "pand");
        emit_sse(assembler, 0x66, false, 0xDF, XMM2, XMM3, "pandn");
        emit_sse(assembler, 0x66, false, 0xEB, XMM2, XMM5, "por");

        if (true == is_scalar)
        {
            emit_sse(assembler, 0x66, is_64bit, 0x7E, XMM2, (uint8_t)SCRATCH_A, "movd/movq");

            if (esize < 32U)
            {
                bal_x86_emit_shl_r32_imm8(assembler, SCRATCH_A, (uint8_t)(32U - esize));
                bal_x86_emit_shr_r32_imm8(assembler, SCRATCH_A, (uint8_t)(32U - esize));
            }

            emit_store_scalar_result(assembler, rd);
            return;
        }

        if (esize >= 32U)
        {
            emit_store_vector(assembler, XMM2, rd, is_quad);
            return;
        }

        // The low byte or halfword of each lane, to the group's place in Rd.
        emit_load_vector_constant(assembler,
                                  XMM3,
                                  8U == esize ? 0x808080800C080400ULL : 0x0D0C090805040100ULL,
                                  0x8080808080808080ULL);
        emit_sse38(assembler, 0x00, XMM2, XMM3, "pshufb");
        emit_sse_memory(assembler,
                        0x66,
                        false,
                        8U == esize ? 0x7E : 0xD6,
                        XMM2,
                        offsetof(bal_cpu_t, v[rd][0]) + skip,
                        "movd/movq");
    }

    if (false == is_quad)
    {
        emit_clear_upper_half(assembler, rd);
    }
}

/// SSHL, USHL, SRSHL, URSHL, SQSHL, UQSHL, SQRSHL and UQRSHL (vector and scalar), and SQSHL,
/// UQSHL and SQSHLU by immediate (vector and scalar), with emit_variable_shift().
static bool
emit_fast_register_shift(bal_tier1_compiler_t *BAL_RESTRICT compiler, const uint32_t instruction)
{
    // Register: 0 Q U 01110 size 1 Rm 010 S R 1 Rn Rd, or the scalar 01 U 11110 size 1 Rm 010 S R
    // 1 Rn Rd. Immediate: 0 Q U 011110 immh immb 011 op 0 1 Rn Rd (or 01 U 111110 for scalars) with
    // SQSHLU U = 1 op = 0 and SQSHL, UQSHL op = 1.
    const bool is_vector_register = 0x0E204400U == (instruction & 0x9F20E400U);
    const bool is_scalar_register = 0x5E204400U == (instruction & 0xDF20E400U);
    const bool is_vector_immediate
        = 0x0F006400U == (instruction & 0x9F80EC00U) && 0U != ((instruction >> 19U) & 0xFU);
    const bool is_scalar_immediate
        = 0x5F006400U == (instruction & 0xDF80EC00U) && 0U != ((instruction >> 19U) & 0xFU);

    if (false == has_feature(compiler, BAL_TIER1_FEATURE_AVX2)
        || false == has_feature(compiler, BAL_TIER1_FEATURE_SSE42))
    {
        return false;
    }

    const uint32_t u         = (instruction >> 29U) & 1U;
    const uint32_t rd        = instruction & 0x1FU;
    const uint32_t rn        = (instruction >> 5U) & 0x1FU;
    const bool     is_scalar = true == is_scalar_register || true == is_scalar_immediate;
    const bool     is_quad   = false == is_scalar && 1U == ((instruction >> 30U) & 1U);

    if (true == is_vector_register || true == is_scalar_register)
    {
        const uint32_t size          = (instruction >> 22U) & 0x3U;
        const bool     is_saturating = 1U == ((instruction >> 11U) & 1U);
        const bool     is_rounding   = 1U == ((instruction >> 12U) & 1U);

        // Doublewords need a full vector; the scalar shifts without saturation exist on them only.
        if ((3U == size && false == is_quad && false == is_scalar)
            || (true == is_scalar && false == is_saturating && size != 3U))
        {
            return false;
        }

        emit_variable_shift(compiler,
                            rd,
                            rn,
                            (instruction >> 16U) & 0x1FU,
                            size,
                            is_quad,
                            is_scalar,
                            0U == u,
                            is_rounding,
                            false == is_saturating ? SATURATE_NONE
                            : 0U == u              ? SATURATE_SIGNED
                                                   : SATURATE_UNSIGNED);
        return true;
    }

    if (false == is_vector_immediate && false == is_scalar_immediate)
    {
        return false;
    }

    const uint32_t immh   = (instruction >> 19U) & 0xFU;
    const bool     is_one = 1U == ((instruction >> 12U) & 1U);
    const uint32_t size
        = 0U != (immh & 8U) ? 3U : (0U != (immh & 4U) ? 2U : (0U != (immh & 2U) ? 1U : 0U));

    // SQSHLU is U = 1 with op = 0; op = 0 with U = 0 is unallocated.
    if ((false == is_one && 0U == u) || (3U == size && false == is_quad && false == is_scalar))
    {
        return false;
    }

    const uint32_t amount = ((instruction >> 16U) & 0x7FU) - (8U << size);
    emit_variable_shift(compiler,
                        rd,
                        rn,
                        32U + amount,
                        size,
                        is_quad,
                        is_scalar,
                        false == is_one || 0U == u,
                        false,
                        false == is_one ? SATURATE_SIGNED_TO_UNSIGNED
                        : 0U == u       ? SATURATE_SIGNED
                                        : SATURATE_UNSIGNED);
    return true;
}

// -----------------------------------------------------------------------------
// Inline counts, narrow-high arithmetic and saturating forms
// -----------------------------------------------------------------------------

/// Loads element 0 of SIMD&FP register `index`, `8 << size` bits, into `xmm`, clearing the rest.
/// Uses SCRATCH_A.
static void
emit_load_element(bal_x86_assembler_t *BAL_RESTRICT assembler,
                  const uint8_t                     xmm,
                  const uint32_t                    index,
                  const uint32_t                    size)
{
    emit_load_vector(assembler, xmm, index, false);

    if (size < 3U)
    {
        bal_x86_emit_mov_r64_imm64(assembler, SCRATCH_A, (1ULL << (8U << size)) - 1U);
        emit_sse(assembler, 0x66, true, 0x6E, XMM5, (uint8_t)SCRATCH_A, "movq");
        emit_sse(assembler, 0x66, false, 0xDB, xmm, XMM5, "pand");
    }
}

/// Stores element 0 of `xmm`, `8 << size` bits, to SIMD&FP register `index`, clearing the rest.
static void
emit_store_element(bal_x86_assembler_t *BAL_RESTRICT assembler,
                   const uint8_t                     xmm,
                   const uint32_t                    index,
                   const uint32_t                    size)
{
    emit_sse(assembler, 0x66, 3U == size, 0x7E, xmm, (uint8_t)SCRATCH_A, "movd/movq");

    if (size < 2U)
    {
        const uint8_t shift = (uint8_t)(32U - (8U << size));
        bal_x86_emit_shl_r32_imm8(assembler, SCRATCH_A, shift);
        bal_x86_emit_shr_r32_imm8(assembler, SCRATCH_A, shift);
    }

    emit_store_scalar_result(assembler, index);
}

/// Counts the leading zero bits of the `8 << size`-bit lanes (bytes, halfwords or words) of xmm0
/// into xmm0, using xmm1 to xmm4. Bytes look the count of each nibble up with PSHUFB. Words keep
/// only their top set bit, whose exponent as a float is its position; halfwords count as the top
/// halves of words, at most 16.
static void
emit_count_leading_zeros(bal_x86_assembler_t *BAL_RESTRICT assembler, const uint32_t size)
{
    if (0U == size)
    {
        emit_sse(assembler, 0U, false, 0x28, XMM1, XMM0, "movaps");
        emit_sse_immediate(assembler, 0x66, 0x71, 2U, XMM1, 4U, "psrlw");
        emit_broadcast_pattern(assembler, XMM2, 0x0F0F0F0F0F0F0F0FULL);
        emit_sse(assembler, 0x66, false, 0xDB, XMM0, XMM2, "pand");
        emit_sse(assembler, 0x66, false, 0xDB, XMM1, XMM2, "pand");
        emit_load_vector_constant(assembler, XMM3, 0x0101010102020304ULL, 0U);
        emit_sse(assembler, 0U, false, 0x28, XMM2, XMM3, "movaps");
        emit_sse38(assembler, 0x00, XMM2, XMM1, "pshufb");
        emit_sse38(assembler, 0x00, XMM3, XMM0, "pshufb");

        // The low nibble's count adds where the high nibble is zero.
        emit_zero_vector(assembler, XMM4);
        emit_sse(assembler, 0x66, false, 0x74, XMM1, XMM4, "pcmpeqb");
        emit_sse(assembler, 0x66, false, 0xDB, XMM3, XMM1, "pand");
        emit_sse(assembler, 0x66, false, 0xFC, XMM2, XMM3, "paddb");
        emit_sse(assembler, 0U, false, 0x28, XMM0, XMM2, "movaps");
        return;
    }

    const uint32_t parts = 2U == size ? 1U : 2U;

    for (uint32_t part = 0U; part < parts; ++part)
    {
        // XMM1 = the word to count: the word, or for halfwords the odd ones in place and the even
        // ones shifted up.
        const uint8_t result = 0U == part ? XMM3 : XMM4;
        emit_sse(assembler, 0U, false, 0x28, XMM1, XMM0, "movaps");

        if (1U == size)
        {
            if (0U == part)
            {
                emit_sse_immediate(assembler, 0x66, 0x72, 2U, XMM1, 16U, "psrld");
                emit_sse_immediate(assembler, 0x66, 0x72, 6U, XMM1, 16U, "pslld");
            }
            else
            {
                emit_sse_immediate(assembler, 0x66, 0x72, 6U, XMM1, 16U, "pslld");
            }
        }

        for (uint8_t shift = 1U; shift <= 16U; shift = (uint8_t)(shift * 2U))
        {
            emit_sse(assembler, 0U, false, 0x28, XMM2, XMM1, "movaps");
            emit_sse_immediate(assembler, 0x66, 0x72, 2U, XMM2, shift, "psrld");
            emit_sse(assembler, 0x66, false, 0xEB, XMM1, XMM2, "por");
        }

        emit_sse(assembler, 0U, false, 0x28, XMM2, XMM1, "movaps");
        emit_sse_immediate(assembler, 0x66, 0x72, 2U, XMM2, 1U, "psrld");
        emit_sse(assembler, 0x66, false, 0xDF, XMM2, XMM1, "pandn");
        emit_sse(assembler, 0U, false, 0x5B, XMM2, XMM2, "cvtdq2ps");
        emit_sse_immediate(assembler, 0x66, 0x72, 6U, XMM2, 1U, "pslld");
        emit_sse_immediate(assembler, 0x66, 0x72, 2U, XMM2, 24U, "psrld");
        emit_broadcast_pattern(assembler, result, 0x0000009E0000009EULL);
        emit_sse(assembler, 0x66, false, 0xFA, result, XMM2, "psubd");
        emit_broadcast_pattern(
            assembler, XMM2, 2U == size ? 0x0000002000000020ULL : 0x0000001000000010ULL);
        emit_sse38(assembler, 0x3B, result, XMM2, "pminud");
    }

    if (1U == size)
    {
        emit_sse_immediate(assembler, 0x66, 0x72, 6U, XMM3, 16U, "pslld");
        emit_sse(assembler, 0x66, false, 0xEB, XMM3, XMM4, "por");
    }

    emit_sse(assembler, 0U, false, 0x28, XMM0, XMM3, "movaps");
}

/// CLZ and CLS (vector) on bytes, halfwords and words. CLS counts the leading zeros of
/// x ^ (x >> 1), shifted arithmetically, less one.
static bool
emit_fast_count_leading(bal_tier1_compiler_t *BAL_RESTRICT compiler, const uint32_t instruction)
{
    // 0 Q U 01110 size 10000 00100 10 Rn Rd: CLS with U = 0, CLZ with U = 1.
    const uint32_t size = (instruction >> 22U) & 0x3U;

    if (0x0E204800U != (instruction & 0x9F3FFC00U) || 3U == size
        || false == has_feature(compiler, BAL_TIER1_FEATURE_SSE41))
    {
        return false;
    }

    const bool                        is_quad   = 1U == ((instruction >> 30U) & 1U);
    const bool                        is_cls    = 0U == ((instruction >> 29U) & 1U);
    bal_x86_assembler_t *BAL_RESTRICT assembler = &compiler->assembler;

    bal_sliding_window_flush_all(&compiler->window);
    emit_load_vector(assembler, XMM0, (instruction >> 5U) & 0x1FU, is_quad);

    if (true == is_cls)
    {
        emit_sse(assembler, 0U, false, 0x28, XMM1, XMM0, "movaps");
        emit_lanes_shift_arithmetic(assembler, XMM1, size, 1U, XMM2);
        emit_sse(assembler, 0x66, false, 0xEF, XMM0, XMM1, "pxor");
    }

    emit_count_leading_zeros(assembler, size);

    if (true == is_cls)
    {
        emit_all_ones(assembler, XMM1);
        emit_sse(assembler, 0x66, false, LANE_ADD[size], XMM0, XMM1, "padd");
    }

    emit_store_vector(assembler, XMM0, instruction & 0x1FU, is_quad);
    return true;
}

/// ADDHN, RADDHN, SUBHN and RSUBHN with their "2" forms: the high halves of the wide sums or
/// differences, rounded by adding half their weight first for the R forms.
static bool
emit_fast_narrow_high(bal_tier1_compiler_t *BAL_RESTRICT compiler, const uint32_t instruction)
{
    // 0 Q U 01110 size 1 Rm 01 o1 0 00 Rn Rd.
    const uint32_t size = (instruction >> 22U) & 0x3U;

    if (0x0E204000U != (instruction & 0x9F20DC00U) || 3U == size
        || false == has_feature(compiler, BAL_TIER1_FEATURE_SSE41))
    {
        return false;
    }

    const bool                        is_quad     = 1U == ((instruction >> 30U) & 1U);
    const bool                        is_rounding = 1U == ((instruction >> 29U) & 1U);
    const bool                        is_subtract = 1U == ((instruction >> 13U) & 1U);
    const uint32_t                    esize       = 8U << size;
    bal_x86_assembler_t *BAL_RESTRICT assembler   = &compiler->assembler;

    bal_sliding_window_flush_all(&compiler->window);
    emit_load_vector(assembler, XMM0, (instruction >> 5U) & 0x1FU, true);
    emit_load_vector(assembler, XMM1, (instruction >> 16U) & 0x1FU, true);
    emit_sse(assembler,
             0x66,
             false,
             true == is_subtract ? LANE_SUB[size + 1U] : LANE_ADD[size + 1U],
             XMM0,
             XMM1,
             "padd/psub");

    if (true == is_rounding)
    {
        emit_broadcast_pattern(assembler, XMM1, replicate_lane(1ULL << (esize - 1U), size + 1U));
        emit_sse(assembler, 0x66, false, LANE_ADD[size + 1U], XMM0, XMM1, "padd");
    }

    emit_lanes_shift_right(assembler, XMM0, size + 1U, esize, XMM1);
    emit_narrow(assembler, XMM0, size, XMM1);
    emit_store_narrow(assembler, instruction & 0x1FU, is_quad);
    return true;
}

/// SHSUB and UHSUB: (n - m) >> 1 in one more bit, which is ((n ^ m) >> 1) - (~n & m); signed
/// lanes are biased to unsigned ones, which keeps their differences.
static bool
emit_fast_halving_subtract(bal_tier1_compiler_t *BAL_RESTRICT compiler, const uint32_t instruction)
{
    // 0 Q U 01110 size 1 Rm 00100 1 Rn Rd.
    const uint32_t size = (instruction >> 22U) & 0x3U;

    if (0x0E202400U != (instruction & 0x9F20FC00U) || 3U == size)
    {
        return false;
    }

    const bool                        is_quad   = 1U == ((instruction >> 30U) & 1U);
    bal_x86_assembler_t *BAL_RESTRICT assembler = &compiler->assembler;

    bal_sliding_window_flush_all(&compiler->window);
    emit_load_vector(assembler, XMM0, (instruction >> 5U) & 0x1FU, is_quad);
    emit_load_vector(assembler, XMM1, (instruction >> 16U) & 0x1FU, is_quad);

    if (0U == ((instruction >> 29U) & 1U))
    {
        emit_broadcast_pattern(assembler, XMM2, lane_sign_pattern(size));
        emit_sse(assembler, 0x66, false, 0xEF, XMM0, XMM2, "pxor");
        emit_sse(assembler, 0x66, false, 0xEF, XMM1, XMM2, "pxor");
    }

    emit_sse(assembler, 0U, false, 0x28, XMM2, XMM0, "movaps");
    emit_sse(assembler, 0x66, false, 0xEF, XMM2, XMM1, "pxor");
    emit_lanes_shift_right(assembler, XMM2, size, 1U, XMM3);
    emit_sse(assembler, 0x66, false, 0xDF, XMM0, XMM1, "pandn");
    emit_sse(assembler, 0x66, false, LANE_SUB[size], XMM2, XMM0, "psub");
    emit_store_vector(assembler, XMM2, instruction & 0x1FU, is_quad);
    return true;
}

/// MUL, MLA and MLS (vector) on bytes: PMULLW multiplies the even bytes in the low bytes of
/// words, and the odd ones as the high bytes of words whose low bytes are zero.
static bool
emit_fast_byte_multiply(bal_tier1_compiler_t *BAL_RESTRICT compiler, const uint32_t instruction)
{
    // 0 Q U 01110 00 1 Rm 1001 o 1 Rn Rd: MUL is U = 0, o = 1; MLA U = 0, o = 0; MLS U = 1, o = 0.
    const uint32_t u      = (instruction >> 29U) & 1U;
    const uint32_t opcode = (instruction >> 11U) & 1U;

    if (0x0E209400U != (instruction & 0x9FE0F400U) || (1U == u && 1U == opcode))
    {
        return false;
    }

    const bool                        is_quad   = 1U == ((instruction >> 30U) & 1U);
    const uint32_t                    rd        = instruction & 0x1FU;
    bal_x86_assembler_t *BAL_RESTRICT assembler = &compiler->assembler;

    bal_sliding_window_flush_all(&compiler->window);
    emit_load_vector(assembler, XMM0, (instruction >> 5U) & 0x1FU, is_quad);
    emit_load_vector(assembler, XMM1, (instruction >> 16U) & 0x1FU, is_quad);
    emit_sse(assembler, 0U, false, 0x28, XMM2, XMM0, "movaps");
    emit_sse(assembler, 0x66, false, 0xD5, XMM2, XMM1, "pmullw");
    emit_sse_immediate(assembler, 0x66, 0x71, 2U, XMM0, 8U, "psrlw");
    emit_broadcast_pattern(assembler, XMM3, 0xFF00FF00FF00FF00ULL);
    emit_sse(assembler, 0x66, false, 0xDB, XMM1, XMM3, "pand");
    emit_sse(assembler, 0x66, false, 0xD5, XMM0, XMM1, "pmullw");
    emit_sse(assembler, 0x66, false, 0xDF, XMM3, XMM2, "pandn");
    emit_sse(assembler, 0x66, false, 0xEB, XMM0, XMM3, "por");

    if (0U == opcode)
    {
        emit_load_vector(assembler, XMM1, rd, is_quad);
        emit_sse(assembler, 0x66, false, 1U == u ? 0xF8 : 0xFC, XMM1, XMM0, "psubb/paddb");
        emit_store_vector(assembler, XMM1, rd, is_quad);
        return true;
    }

    emit_store_vector(assembler, XMM0, rd, is_quad);
    return true;
}

/// Sets FPSR.QC when a lane of the mask in `xmm` is set (all ones), using SCRATCH_A.
static void
emit_saturation_if_any(bal_x86_assembler_t *BAL_RESTRICT assembler, const uint8_t xmm)
{
    emit_sse(assembler, 0x66, false, 0xD7, (uint8_t)SCRATCH_A, xmm, "pmovmskb");
    bal_x86_emit_test_r32_r32(assembler, SCRATCH_A, SCRATCH_A);
    const size_t none = emit_forward_jump(assembler, true, BAL_X86_COND_E);
    emit_set_saturation(assembler);
    patch_forward_jump(assembler, none, true);
}

/// SQABS and SQNEG (vector and scalar): the wrapping absolute value or negation, where the
/// minimum, which wraps to itself, becomes the maximum and sets FPSR.QC.
static bool
emit_fast_saturating_unary(bal_tier1_compiler_t *BAL_RESTRICT compiler, const uint32_t instruction)
{
    // 0 Q U 01110 size 10000 00111 10 Rn Rd, or the scalar 01 U 11110 size 10000 00111 10 Rn Rd.
    const bool     is_scalar = 0x5E207800U == (instruction & 0xDF3FFC00U);
    const uint32_t size      = (instruction >> 22U) & 0x3U;
    const bool     is_quad   = false == is_scalar && 1U == ((instruction >> 30U) & 1U);

    if ((0x0E207800U != (instruction & 0x9F3FFC00U) && false == is_scalar)
        || (3U == size && false == is_quad && false == is_scalar)
        || false == has_feature(compiler, BAL_TIER1_FEATURE_SSE42))
    {
        return false;
    }

    const bool                        is_negate = 1U == ((instruction >> 29U) & 1U);
    const uint32_t                    rd        = instruction & 0x1FU;
    const uint32_t                    rn        = (instruction >> 5U) & 0x1FU;
    bal_x86_assembler_t *BAL_RESTRICT assembler = &compiler->assembler;

    bal_sliding_window_flush_all(&compiler->window);

    if (true == is_scalar)
    {
        emit_load_element(assembler, XMM0, rn, size);
    }
    else
    {
        emit_load_vector(assembler, XMM0, rn, is_quad);
    }

    if (true == is_negate)
    {
        emit_zero_vector(assembler, XMM1);
        emit_sse(assembler, 0x66, false, LANE_SUB[size], XMM1, XMM0, "psub");
    }
    else if (3U == size)
    {
        emit_zero_vector(assembler, XMM2);
        emit_lanes_greater(assembler, 3U, XMM2, XMM0);
        emit_sse(assembler, 0U, false, 0x28, XMM1, XMM0, "movaps");
        emit_sse(assembler, 0x66, false, 0xEF, XMM1, XMM2, "pxor");
        emit_sse(assembler, 0x66, false, 0xFB, XMM1, XMM2, "psubq");
    }
    else
    {
        emit_sse38(assembler, (uint8_t)(0x1CU + size), XMM1, XMM0, "pabs");
    }

    emit_broadcast_pattern(assembler, XMM2, lane_sign_pattern(size));
    emit_lanes_equal(assembler, size, XMM2, XMM0);
    emit_sse(assembler, 0x66, false, 0xEF, XMM1, XMM2, "pxor");
    emit_saturation_if_any(assembler, XMM2);

    if (true == is_scalar)
    {
        emit_store_element(assembler, XMM1, rd, size);
    }
    else
    {
        emit_store_vector(assembler, XMM1, rd, is_quad);
    }

    return true;
}

/// SQADD, UQADD, SQSUB and UQSUB on words and doublewords (vector) and on every size (scalar).
/// Bytes and halfwords use the saturating SSE forms; the wider lanes find the overflow from the
/// signs ((n ^ r) & (m ^ r) for a signed sum r, (n ^ m) & (n ^ r) for a difference) or an
/// unsigned comparison, and saturate to the bound it crossed, setting FPSR.QC.
static bool
emit_fast_saturating_arithmetic(bal_tier1_compiler_t *BAL_RESTRICT compiler,
                                const uint32_t                     instruction)
{
    // 0 Q U 01110 size 1 Rm 00 s 01 1 Rn Rd (SQADD and UQADD s = 0, SQSUB and UQSUB s = 1), or
    // the scalar 01 U 11110 size ...
    const bool     is_scalar = 0x5E200C00U == (instruction & 0xDF20DC00U);
    const uint32_t size      = (instruction >> 22U) & 0x3U;
    const bool     is_quad   = false == is_scalar && 1U == ((instruction >> 30U) & 1U);

    if ((0x0E200C00U != (instruction & 0x9F20DC00U) && false == is_scalar)
        || (false == is_scalar && size < 2U)
        || (3U == size && false == is_quad && false == is_scalar)
        || false == has_feature(compiler, BAL_TIER1_FEATURE_SSE42))
    {
        return false;
    }

    const bool                        is_unsigned = 1U == ((instruction >> 29U) & 1U);
    const bool                        is_subtract = 1U == ((instruction >> 13U) & 1U);
    const uint32_t                    rd          = instruction & 0x1FU;
    const uint32_t                    rn          = (instruction >> 5U) & 0x1FU;
    const uint32_t                    rm          = (instruction >> 16U) & 0x1FU;
    const uint32_t                    compare     = 3U == size ? 3U : 2U;
    bal_x86_assembler_t *BAL_RESTRICT assembler   = &compiler->assembler;

    bal_sliding_window_flush_all(&compiler->window);

    if (true == is_scalar)
    {
        emit_load_element(assembler, XMM0, rn, size);
        emit_load_element(assembler, XMM1, rm, size);
    }
    else
    {
        emit_load_vector(assembler, XMM0, rn, is_quad);
        emit_load_vector(assembler, XMM1, rm, is_quad);
    }

    if (size < 2U)
    {
        // As emit_fast_vector_integer_more() does for vectors.
        static const uint8_t SATURATING[2][2][2]
            = { { { 0xEC, 0xED }, { 0xDC, 0xDD } }, { { 0xE8, 0xE9 }, { 0xD8, 0xD9 } } };
        emit_sse(assembler, 0U, false, 0x28, XMM2, XMM0, "movaps");
        emit_sse(assembler,
                 0x66,
                 false,
                 SATURATING[true == is_subtract ? 1 : 0][true == is_unsigned ? 1 : 0][size],
                 XMM0,
                 XMM1,
                 "padds");
        emit_sse(assembler,
                 0x66,
                 false,
                 true == is_subtract ? LANE_SUB[size] : LANE_ADD[size],
                 XMM2,
                 XMM1,
                 "padd/psub");
        emit_store_element(assembler, XMM0, rd, size);
        emit_saturation_unless_equal(assembler, XMM2, XMM0);
        return true;
    }

    // XMM2 = the wrapping result; XMM3 = the lanes that overflowed; XMM4 = their bound.
    emit_sse(assembler, 0U, false, 0x28, XMM2, XMM0, "movaps");
    emit_sse(assembler,
             0x66,
             false,
             true == is_subtract ? LANE_SUB[size] : LANE_ADD[size],
             XMM2,
             XMM1,
             "padd/psub");

    if (false == is_unsigned)
    {
        emit_sse(assembler, 0U, false, 0x28, XMM3, XMM0, "movaps");
        emit_sse(assembler, 0x66, false, 0xEF, XMM3, XMM2, "pxor");

        if (true == is_subtract)
        {
            emit_sse(assembler, 0U, false, 0x28, XMM4, XMM0, "movaps");
            emit_sse(assembler, 0x66, false, 0xEF, XMM4, XMM1, "pxor");
        }
        else
        {
            emit_sse(assembler, 0U, false, 0x28, XMM4, XMM1, "movaps");
            emit_sse(assembler, 0x66, false, 0xEF, XMM4, XMM2, "pxor");
        }

        emit_sse(assembler, 0x66, false, 0xDB, XMM4, XMM3, "pand");
        emit_zero_vector(assembler, XMM3);
        emit_lanes_greater(assembler, compare, XMM3, XMM4);

        // The bound is the maximum for n >= 0 and its complement, the minimum, for n < 0.
        emit_zero_vector(assembler, XMM4);
        emit_lanes_greater(assembler, compare, XMM4, XMM0);
        emit_broadcast_pattern(
            assembler, XMM5, 3U == size ? 0x7FFFFFFFFFFFFFFFULL : 0x7FFFFFFF7FFFFFFFULL);
        emit_sse(assembler, 0x66, false, 0xEF, XMM4, XMM5, "pxor");
    }
    else
    {
        // Biased by the sign bit, the unsigned comparisons become signed ones: a sum below n
        // wrapped (to all ones), as does n below m in a difference (to zero).
        emit_broadcast_pattern(assembler, XMM5, lane_sign_pattern(size));
        emit_sse(assembler, 0U, false, 0x28, XMM3, XMM0, "movaps");
        emit_sse(assembler, 0x66, false, 0xEF, XMM3, XMM5, "pxor");

        if (true == is_subtract)
        {
            emit_sse(assembler, 0x66, false, 0xEF, XMM1, XMM5, "pxor");
            emit_lanes_greater(assembler, compare, XMM1, XMM3);
            emit_sse(assembler, 0U, false, 0x28, XMM3, XMM1, "movaps");
            emit_zero_vector(assembler, XMM4);
        }
        else
        {
            emit_sse(assembler, 0U, false, 0x28, XMM4, XMM2, "movaps");
            emit_sse(assembler, 0x66, false, 0xEF, XMM4, XMM5, "pxor");
            emit_lanes_greater(assembler, compare, XMM3, XMM4);
            emit_all_ones(assembler, XMM4);
        }
    }

    emit_sse(assembler, 0x66, false, 0xDB, XMM4, XMM3, "pand");
    emit_sse(assembler, 0x66, false, 0xDF, XMM3, XMM2, "pandn");
    emit_sse(assembler, 0x66, false, 0xEB, XMM4, XMM3, "por");

    if (true == is_scalar)
    {
        emit_store_element(assembler, XMM4, rd, size);
    }
    else
    {
        emit_store_vector(assembler, XMM4, rd, is_quad);
    }

    // A lane saturated where the result differs from the wrapping one.
    emit_sse(assembler, 0x66, false, 0x74, XMM2, XMM4, "pcmpeqb");
    emit_all_ones(assembler, XMM3);
    emit_sse(assembler, 0x66, false, 0xEF, XMM2, XMM3, "pxor");
    emit_saturation_if_any(assembler, XMM2);
    return true;
}

/// SUQADD and USQADD (vector) on words and (scalar) on bytes, halfwords and words: Rd plus Rn,
/// one signed and one unsigned, added in lanes of twice the size, half a vector at a time, and
/// narrowed with saturation as SQXTN (SUQADD's signed result) or SQXTUN (USQADD's unsigned one)
/// does.
static bool
emit_fast_mixed_saturating_add_more(bal_tier1_compiler_t *BAL_RESTRICT compiler,
                                    const uint32_t                     instruction)
{
    // 0 Q U 01110 size 10000 00011 10 Rn Rd, or the scalar 01 U 11110 size 10000 00011 10 Rn Rd.
    const uint32_t size      = (instruction >> 22U) & 0x3U;
    const bool     is_scalar = 0x5E203800U == (instruction & 0xDF3FFC00U);

    if ((0x0E203800U != (instruction & 0x9F3FFC00U) && false == is_scalar) || 3U == size
        || (false == is_scalar && size != 2U)
        || false == has_feature(compiler, BAL_TIER1_FEATURE_SSE42))
    {
        return false;
    }

    const bool     is_quad     = false == is_scalar && 1U == ((instruction >> 30U) & 1U);
    const bool     is_unsigned = 1U == ((instruction >> 29U) & 1U);
    const uint32_t rd          = instruction & 0x1FU;
    const uint32_t rn          = (instruction >> 5U) & 0x1FU;
    bal_x86_assembler_t *BAL_RESTRICT assembler = &compiler->assembler;

    // The register cache is emptied, so that RAX and RDX are free for the halves.
    begin_inline_sequence(compiler);

    for (uint32_t half = 0U; half < (true == is_quad ? 2U : 1U); ++half)
    {
        if (true == is_scalar)
        {
            emit_load_element(assembler, XMM0, rd, size);
            emit_load_element(assembler, XMM1, rn, size);
        }
        else
        {
            emit_load_vector(assembler, XMM0, rd, true);
            emit_load_vector(assembler, XMM1, rn, true);
        }

        emit_widen(assembler, XMM0, size, is_unsigned, 1U == half);
        emit_widen(assembler, XMM1, size, false == is_unsigned, 1U == half);
        emit_sse(assembler, 0x66, false, LANE_ADD[size + 1U], XMM0, XMM1, "padd");
        emit_saturating_narrow(assembler, size, true == is_unsigned ? 0x32U : 0x14U);
        emit_sse(assembler,
                 0x66,
                 true,
                 0x7E,
                 XMM0,
                 (uint8_t)(0U == half ? BAL_X86_RAX : BAL_X86_RDX),
                 "movq");
    }

    emit_sse(assembler, 0x66, true, 0x6E, XMM0, (uint8_t)BAL_X86_RAX, "movq");

    if (true == is_scalar)
    {
        emit_store_element(assembler, XMM0, rd, size);
        return true;
    }

    if (true == is_quad)
    {
        emit_sse(assembler, 0x66, true, 0x6E, XMM1, (uint8_t)BAL_X86_RDX, "movq");
        emit_sse(assembler, 0x66, false, 0x6C, XMM0, XMM1, "punpcklqdq");
    }

    emit_store_vector(assembler, XMM0, rd, is_quad);
    return true;
}

/// SUQADD and USQADD (vector) on bytes and halfwords: Rd plus Rn, one signed and one unsigned,
/// added in lanes of twice the size and narrowed with the signed (SUQADD) or unsigned (USQADD)
/// saturation of the packs; a lane saturated where widening the result back differs.
static bool
emit_fast_mixed_saturating_add(bal_tier1_compiler_t *BAL_RESTRICT compiler,
                               const uint32_t                     instruction)
{
    // 0 Q U 01110 size 10000 00011 10 Rn Rd.
    const uint32_t size = (instruction >> 22U) & 0x3U;

    if (0x0E203800U != (instruction & 0x9F3FFC00U) || size > 1U
        || false == has_feature(compiler, BAL_TIER1_FEATURE_SSE41))
    {
        return false;
    }

    const bool                        is_quad     = 1U == ((instruction >> 30U) & 1U);
    const bool                        is_unsigned = 1U == ((instruction >> 29U) & 1U);
    const uint32_t                    rd          = instruction & 0x1FU;
    const uint32_t                    rn          = (instruction >> 5U) & 0x1FU;
    bal_x86_assembler_t *BAL_RESTRICT assembler   = &compiler->assembler;

    bal_sliding_window_flush_all(&compiler->window);

    // XMM0 and XMM1 = the low and high halves' sums; the accumulator Rd is signed for SUQADD.
    for (uint32_t half = 0U; half < 2U; ++half)
    {
        const uint8_t sum = 0U == half ? XMM0 : XMM1;
        emit_load_vector(assembler, sum, rd, true);
        emit_load_vector(assembler, XMM2, rn, true);
        emit_widen(assembler, sum, size, is_unsigned, 1U == half);
        emit_widen(assembler, XMM2, size, false == is_unsigned, 1U == half);
        emit_sse(assembler, 0x66, false, LANE_ADD[size + 1U], sum, XMM2, "padd");
    }

    emit_sse(assembler, 0U, false, 0x28, XMM2, XMM0, "movaps");

    if (true == is_unsigned)
    {
        if (0U == size)
        {
            emit_sse(assembler, 0x66, false, 0x67, XMM2, XMM1, "packuswb");
        }
        else
        {
            emit_sse38(assembler, 0x2B, XMM2, XMM1, "packusdw");
        }
    }
    else
    {
        emit_sse(assembler, 0x66, false, 0U == size ? 0x63 : 0x6B, XMM2, XMM1, "packss");
    }

    // Widened back, the result equals the sums unless a lane saturated (only the vector's lanes
    // count).
    emit_sse(assembler, 0U, false, 0x28, XMM3, XMM2, "movaps");
    emit_sse(assembler, 0U, false, 0x28, XMM4, XMM2, "movaps");
    emit_widen(assembler, XMM3, size, is_unsigned, false);
    emit_widen(assembler, XMM4, size, is_unsigned, true);
    emit_lanes_equal(assembler, size + 1U, XMM3, XMM0);
    emit_lanes_equal(assembler, size + 1U, XMM4, XMM1);

    if (false == is_quad)
    {
        emit_all_ones(assembler, XMM4);
    }

    emit_sse(assembler, 0x66, false, 0xDB, XMM3, XMM4, "pand");
    emit_all_ones(assembler, XMM4);
    emit_sse(assembler, 0x66, false, 0xEF, XMM3, XMM4, "pxor");
    emit_store_vector(assembler, XMM2, rd, is_quad);
    emit_saturation_if_any(assembler, XMM3);
    return true;
}

/// SQXTN, UQXTN and SQXTUN (scalar) narrowing to bytes and halfwords, and SQDMULH and SQRDMULH
/// (scalar, register and by element) on halfwords and words: the vector forms on element 0.
static bool
emit_fast_saturating_scalar(bal_tier1_compiler_t *BAL_RESTRICT compiler, const uint32_t instruction)
{
    // Narrowing: 01 U 11110 size 10000 1 o 100 10 Rn Rd (SQXTN o = 1 U = 0, UQXTN o = 1 U = 1,
    // SQXTUN o = 0 U = 1). Doubling: 01 U 11110 size 1 Rm 10110 1 Rn Rd, or by element
    // 01 0 11111 size L M Rm 110 R H 0 Rn Rd.
    const uint32_t size       = (instruction >> 22U) & 0x3U;
    const uint32_t u          = (instruction >> 29U) & 1U;
    const uint32_t narrow_key = (u << 5U) | ((instruction >> 12U) & 0x1FU);
    const bool     is_narrow  = 0x5E200800U == (instruction & 0xDF3E0C00U)
                           && (0x14U == narrow_key || 0x34U == narrow_key || 0x32U == narrow_key);
    const bool is_register = 0x5E20B400U == (instruction & 0xDF20FC00U);
    const bool is_element  = 0x5F00C000U == (instruction & 0xFF00E400U);

    if ((false == is_narrow && false == is_register && false == is_element)
        || false == has_feature(compiler, BAL_TIER1_FEATURE_SSE41))
    {
        return false;
    }

    const uint32_t                    rd        = instruction & 0x1FU;
    const uint32_t                    rn        = (instruction >> 5U) & 0x1FU;
    bal_x86_assembler_t *BAL_RESTRICT assembler = &compiler->assembler;

    if (true == is_narrow)
    {
        if (size > 2U || (2U == size && false == has_feature(compiler, BAL_TIER1_FEATURE_SSE42)))
        {
            return false;
        }

        bal_sliding_window_flush_all(&compiler->window);
        emit_load_element(assembler, XMM0, rn, size + 1U);
        emit_saturating_narrow(assembler, size, narrow_key);
        emit_store_element(assembler, XMM0, rd, size);
        return true;
    }

    if (size != 1U && size != 2U)
    {
        return false;
    }

    bal_sliding_window_flush_all(&compiler->window);
    emit_load_element(assembler, XMM0, rn, size);

    if (true == is_register)
    {
        emit_load_element(assembler, XMM1, (instruction >> 16U) & 0x1FU, size);
    }
    else
    {
        const uint32_t h     = (instruction >> 11U) & 1U;
        const uint32_t l     = (instruction >> 21U) & 1U;
        const uint32_t m     = (instruction >> 20U) & 1U;
        const uint32_t index = 1U == size ? (h << 2U) | (l << 1U) | m : (h << 1U) | l;
        const uint32_t rm = 1U == size ? (instruction >> 16U) & 0xFU : (instruction >> 16U) & 0x1FU;
        emit_load_vector(assembler, XMM1, rm, true);
        emit_broadcast_element(assembler, XMM1, size, index);
    }

    emit_doubling_multiply_high(
        assembler, size, true == is_register ? 1U == u : 1U == ((instruction >> 12U) & 1U));
    emit_store_element(assembler, XMM0, rd, size);
    return true;
}

/// Emits a SHA extension instruction `0F map opcode` (no prefix) on two registers, with an 8-bit
/// immediate for the 0F 3A map.
static void
emit_sha(bal_x86_assembler_t *BAL_RESTRICT assembler,
         const uint8_t                     map,
         const uint8_t                     opcode,
         const uint8_t                     reg,
         const uint8_t                     rm,
         const uint8_t                     immediate,
         const char                       *mnemonic)
{
    const uint8_t bytes[3] = { 0x0F, map, opcode };
    bal_x86_emit_sse_register(
        assembler, 0U, false, bytes, 3U, reg, rm, 0x3A == map, immediate, mnemonic);
}

/// SHA1H, SHA1SU0 and SHA1SU1 with SSE, and with the SHA extensions SHA1C, SHA1P, SHA1M,
/// SHA256H, SHA256H2, SHA256SU0 and SHA256SU1.
///
/// SHA256SU0 is SHA256MSG1 and SHA256SU1 is SHA256MSG2 after adding Rd to the words
/// Rm<31:0>:Rn<127:32>. SHA256H and SHA256H2 run two SHA256RNDS2 on the state rearranged from
/// ABCD and EFGH to x86's ABEF and CDGH. SHA1C, SHA1P and SHA1M are SHA1RNDS4, which numbers the
/// words the other way round and adds E to the first and the round constant, which AArch64 code
/// adds to the words itself, to each.
static bool
emit_fast_sha(bal_tier1_compiler_t *BAL_RESTRICT compiler, const uint32_t instruction)
{
    // Three registers: 01011110 000 Rm 0 opcode 00 Rn Rd. Two: 01011110 0010100 00 opcode 10 Rn Rd.
    const uint32_t three = instruction & 0xFFE0FC00U;
    const uint32_t two   = instruction & 0xFFFFFC00U;
    const uint32_t rd    = instruction & 0x1FU;
    const uint32_t rn    = (instruction >> 5U) & 0x1FU;
    const uint32_t rm    = (instruction >> 16U) & 0x1FU;

    bal_x86_assembler_t *BAL_RESTRICT assembler = &compiler->assembler;

    switch (two)
    {
        case 0x5E280800U: // SHA1H: ROL(Sn, 30)
            bal_sliding_window_flush_all(&compiler->window);
            bal_x86_emit_load_r64_rbp_offset(
                assembler, SCRATCH_A, (int32_t)offsetof(bal_cpu_t, v[rn][0]));
            bal_x86_emit_ror_r32_imm8(assembler, SCRATCH_A, 2U);
            emit_store_scalar_result(assembler, rd);
            return true;
        case 0x5E281800U: // SHA1SU1
            bal_sliding_window_flush_all(&compiler->window);
            emit_load_vector(assembler, XMM0, rn, true);
            emit_sse_immediate(assembler, 0x66, 0x73, 3U, XMM0, 4U, "psrldq");
            emit_load_vector(assembler, XMM1, rd, true);
            emit_sse(assembler, 0x66, false, 0xEF, XMM0, XMM1, "pxor");

            // ROL by one, and lane 0 rotated by two into lane 3.
            emit_sse(assembler, 0U, false, 0x28, XMM1, XMM0, "movaps");
            emit_sse_immediate(assembler, 0x66, 0x72, 6U, XMM1, 1U, "pslld");
            emit_sse(assembler, 0U, false, 0x28, XMM2, XMM0, "movaps");
            emit_sse_immediate(assembler, 0x66, 0x72, 2U, XMM2, 31U, "psrld");
            emit_sse(assembler, 0x66, false, 0xEB, XMM1, XMM2, "por");
            emit_sse_immediate(assembler, 0x66, 0x73, 7U, XMM0, 12U, "pslldq");
            emit_sse(assembler, 0U, false, 0x28, XMM2, XMM0, "movaps");
            emit_sse_immediate(assembler, 0x66, 0x72, 6U, XMM0, 2U, "pslld");
            emit_sse_immediate(assembler, 0x66, 0x72, 2U, XMM2, 30U, "psrld");
            emit_sse(assembler, 0x66, false, 0xEB, XMM0, XMM2, "por");
            emit_sse(assembler, 0x66, false, 0xEF, XMM1, XMM0, "pxor");
            emit_store_vector(assembler, XMM1, rd, true);
            return true;
        case 0x5E282800U: // SHA256SU0
            if (false == has_feature(compiler, BAL_TIER1_FEATURE_SHA))
            {
                return false;
            }

            bal_sliding_window_flush_all(&compiler->window);
            emit_load_vector(assembler, XMM0, rd, true);
            emit_load_vector(assembler, XMM1, rn, true);
            emit_sha(assembler, 0x38, 0xCC, XMM0, XMM1, 0U, "sha256msg1");
            emit_store_vector(assembler, XMM0, rd, true);
            return true;
        default:
            break;
    }

    switch (three)
    {
        case 0x5E003000U: // SHA1SU0: Rn<63:0>:Rd<127:64> ^ Rd ^ Rm
            bal_sliding_window_flush_all(&compiler->window);
            emit_load_vector(assembler, XMM0, rd, true);
            emit_load_vector(assembler, XMM1, rn, true);
            emit_sse(assembler, 0U, false, 0x28, XMM2, XMM0, "movaps");
            emit_sse_immediate(assembler, 0U, 0xC6, XMM2, XMM1, 0x4E, "shufps");
            emit_sse(assembler, 0x66, false, 0xEF, XMM2, XMM0, "pxor");
            emit_load_vector(assembler, XMM1, rm, true);
            emit_sse(assembler, 0x66, false, 0xEF, XMM2, XMM1, "pxor");
            emit_store_vector(assembler, XMM2, rd, true);
            return true;
        case 0x5E000000U: // SHA1C
        case 0x5E001000U: // SHA1P
        case 0x5E002000U: // SHA1M
        case 0x5E004000U: // SHA256H
        case 0x5E005000U: // SHA256H2
        case 0x5E006000U: // SHA256SU1
            if (false == has_feature(compiler, BAL_TIER1_FEATURE_SHA))
            {
                return false;
            }
            break;
        default:
            return false;
    }

    bal_sliding_window_flush_all(&compiler->window);

    if (0x5E006000U == three)
    {
        // Rd + Rm<31:0>:Rn<127:32>, then SHA256MSG2 with Rm.
        emit_load_vector(assembler, XMM0, rm, true);
        emit_load_vector(assembler, XMM1, rn, true);
        emit_sse3a(assembler, 0x0F, XMM0, XMM1, 4U, "palignr");
        emit_load_vector(assembler, XMM1, rd, true);
        emit_sse(assembler, 0x66, false, 0xFE, XMM0, XMM1, "paddd");
        emit_load_vector(assembler, XMM1, rm, true);
        emit_sha(assembler, 0x38, 0xCD, XMM0, XMM1, 0U, "sha256msg2");
        emit_store_vector(assembler, XMM0, rd, true);
        return true;
    }

    if (three >= 0x5E004000U)
    {
        // ABCD and EFGH: Rd and Rn for SHA256H, Rn and Rd for SHA256H2. XMM1 = ABEF (lanes F E B
        // A), XMM2 = CDGH (H G D C), XMM0 = the words, which SHA256RNDS2 reads implicitly.
        const bool     is_first = 0x5E004000U == three;
        const uint32_t abcd     = true == is_first ? rd : rn;
        const uint32_t efgh     = true == is_first ? rn : rd;
        emit_load_vector(assembler, XMM1, efgh, true);
        emit_load_vector(assembler, XMM3, abcd, true);
        emit_sse(assembler, 0U, false, 0x28, XMM2, XMM1, "movaps");
        emit_sse(assembler, 0x66, false, 0x6C, XMM1, XMM3, "punpcklqdq");
        emit_sse(assembler, 0x66, false, 0x6D, XMM2, XMM3, "punpckhqdq");
        emit_sse_immediate(assembler, 0x66, 0x70, XMM1, XMM1, 0xB1, "pshufd");
        emit_sse_immediate(assembler, 0x66, 0x70, XMM2, XMM2, 0xB1, "pshufd");
        emit_load_vector(assembler, XMM0, rm, true);
        emit_sha(assembler, 0x38, 0xCB, XMM2, XMM1, 0U, "sha256rnds2");
        emit_sse_immediate(assembler, 0x66, 0x70, XMM0, XMM0, 0x0E, "pshufd");
        emit_sha(assembler, 0x38, 0xCB, XMM1, XMM2, 0U, "sha256rnds2");

        // XMM1 = ABEF and XMM2 = CDGH after four rounds, back to ABCD or EFGH.
        emit_sse(assembler, 0x66, false, true == is_first ? 0x6D : 0x6C, XMM1, XMM2, "punpckhqdq");
        emit_sse_immediate(assembler, 0x66, 0x70, XMM1, XMM1, 0xB1, "pshufd");
        emit_store_vector(assembler, XMM1, rd, true);
        return true;
    }

    // SHA1C, SHA1P and SHA1M: the round constants of rounds 0-19, 20-39 and 40-59.
    static const uint32_t CONSTANTS[3] = { 0x5A827999U, 0x6ED9EBA1U, 0x8F1BBCDCU };
    const uint32_t        function     = (three >> 12U) & 0x3U;
    emit_load_vector(assembler, XMM0, rd, true);
    emit_sse_immediate(assembler, 0x66, 0x70, XMM0, XMM0, 0x1B, "pshufd");
    emit_load_vector(assembler, XMM1, rm, true);
    emit_sse_immediate(assembler, 0x66, 0x70, XMM1, XMM1, 0x1B, "pshufd");
    bal_x86_emit_load_r64_rbp_offset(assembler, SCRATCH_A, (int32_t)offsetof(bal_cpu_t, v[rn][0]));
    emit_sse(assembler, 0x66, false, 0x6E, XMM2, (uint8_t)SCRATCH_A, "movd");
    emit_sse_immediate(assembler, 0x66, 0x73, 7U, XMM2, 12U, "pslldq");
    emit_sse(assembler, 0x66, false, 0xFE, XMM1, XMM2, "paddd");
    emit_broadcast_pattern(assembler, XMM2, 0x0000000100000001ULL * CONSTANTS[function]);
    emit_sse(assembler, 0x66, false, 0xFA, XMM1, XMM2, "psubd");
    emit_sha(assembler, 0x3A, 0xCC, XMM0, XMM1, (uint8_t)function, "sha1rnds4");
    emit_sse_immediate(assembler, 0x66, 0x70, XMM0, XMM0, 0x1B, "pshufd");
    emit_store_vector(assembler, XMM0, rd, true);
    return true;
}

// -----------------------------------------------------------------------------
// Inline fused half precision arithmetic (AVX)
// -----------------------------------------------------------------------------

/// Emits a VEX instruction from the 0F map: `opcode reg, vvvv, rm` with the implied prefix
/// `prefix` (0 none, 1 66), on YMM registers with `is_256`.
static void
emit_vex(bal_x86_assembler_t *BAL_RESTRICT assembler,
         const uint8_t                     prefix,
         const bool                        is_256,
         const uint8_t                     opcode,
         const uint8_t                     reg,
         const uint8_t                     vvvv,
         const uint8_t                     rm,
         const char                       *mnemonic)
{
    bal_x86_emit_vex(
        assembler, 1U, prefix, is_256, false, opcode, reg, vvvv, rm, false, 0U, mnemonic);
}

/// Emits VCMPPD `reg`, `vvvv`, `rm`, `predicate` on YMM registers.
static void
emit_vex_compare(bal_x86_assembler_t *BAL_RESTRICT assembler,
                 const uint8_t                     reg,
                 const uint8_t                     vvvv,
                 const uint8_t                     rm,
                 const uint8_t                     predicate)
{
    bal_x86_emit_vex(
        assembler, 1U, 1U, true, false, 0xC2, reg, vvvv, rm, true, predicate, "vcmppd");
}

/// Puts the all-ones masks of the four 64-bit lanes of YMM `source` into the 32-bit lanes of
/// `destination` (an XMM register), using xmm4.
static void
emit_narrow_masks(bal_x86_assembler_t *BAL_RESTRICT assembler,
                  const uint8_t                     destination,
                  const uint8_t                     source)
{
    bal_x86_emit_vex(
        assembler, 3U, 1U, true, false, 0x19, source, 0U, XMM4, true, 1U, "vextractf128");
    bal_x86_emit_vex(
        assembler, 1U, 0U, false, false, 0xC6, destination, source, XMM4, true, 0xDD, "vshufps");
}

/// Rounds c + a * b once to half precision for the four single precision lanes a, b and c in
/// xmm0, xmm1 and xmm2 (exact conversions of halves, or constants), into the four halves in the
/// low 64 bits of xmm0. Uses xmm0 to xmm5 as YMM registers and SCRATCH_A, and adds two jumps to
/// the slow path to `jumps`: for an infinite or NaN operand, whose rules (and Invalid from
/// infinity minus infinity in TwoSum) differ, and for a sum that is not exact in double precision.
///
/// The product of two halves is exact in double precision, and so is the sum unless its terms'
/// exponents are far apart: TwoSum gives the sum's error, and a nonzero one (or a NaN) takes the
/// slow path. The exact sum is then rounded to single precision to odd (towards zero, with the
/// last bit set if that was inexact), which keeps VCVTPS2PH's rounding to half precision the
/// correct one. The sum's sign of rounding is that of (back - sum) * sum, back being the nearest
/// single precision value converted back.
static void
emit_half_fused(bal_x86_assembler_t *BAL_RESTRICT assembler, size_t *jumps)
{
    // An exponent of all ones in any operand.
    emit_broadcast_pattern(assembler, XMM4, 0x7F8000007F800000ULL);
    emit_zero_vector(assembler, XMM5);

    for (uint8_t operand = XMM0; operand <= XMM2; ++operand)
    {
        emit_sse(assembler, 0U, false, 0x28, XMM3, operand, "movaps");
        emit_sse(assembler, 0x66, false, 0xDB, XMM3, XMM4, "pand");
        emit_sse(assembler, 0x66, false, 0x76, XMM3, XMM4, "pcmpeqd");
        emit_sse(assembler, 0x66, false, 0xEB, XMM5, XMM3, "por");
    }

    emit_sse(assembler, 0U, false, 0x50, (uint8_t)SCRATCH_A, XMM5, "movmskps");
    bal_x86_emit_test_r32_r32(assembler, SCRATCH_A, SCRATCH_A);
    jumps[0] = emit_forward_jump(assembler, true, BAL_X86_COND_NE);

    // YMM3 = a * b, YMM4 = c, YMM5 = s = a * b + c.
    emit_vex(assembler, 0U, true, 0x5A, XMM3, 0U, XMM0, "vcvtps2pd");
    emit_vex(assembler, 0U, true, 0x5A, XMM4, 0U, XMM1, "vcvtps2pd");
    emit_vex(assembler, 1U, true, 0x59, XMM3, XMM3, XMM4, "vmulpd");
    emit_vex(assembler, 0U, true, 0x5A, XMM4, 0U, XMM2, "vcvtps2pd");
    emit_vex(assembler, 1U, true, 0x58, XMM5, XMM3, XMM4, "vaddpd");

    // TwoSum: bb = s - p, error = (p - (s - bb)) + (c - bb).
    emit_vex(assembler, 1U, true, 0x5C, XMM0, XMM5, XMM3, "vsubpd");
    emit_vex(assembler, 1U, true, 0x5C, XMM1, XMM5, XMM0, "vsubpd");
    emit_vex(assembler, 1U, true, 0x5C, XMM1, XMM3, XMM1, "vsubpd");
    emit_vex(assembler, 1U, true, 0x5C, XMM0, XMM4, XMM0, "vsubpd");
    emit_vex(assembler, 1U, true, 0x58, XMM0, XMM1, XMM0, "vaddpd");
    emit_vex(assembler, 1U, true, 0x57, XMM1, XMM1, XMM1, "vxorpd");
    emit_vex_compare(assembler, XMM0, XMM0, XMM1, 4U); // NEQ_UQ
    emit_vex(assembler, 1U, true, 0x50, (uint8_t)SCRATCH_A, 0U, XMM0, "vmovmskpd");

    // XMM0 = s to nearest single precision; YMM2 = its distance back to s; YMM3 = the lanes
    // rounded away from zero.
    emit_vex(assembler, 1U, true, 0x5A, XMM0, 0U, XMM5, "vcvtpd2ps");
    emit_vex(assembler, 0U, true, 0x5A, XMM1, 0U, XMM0, "vcvtps2pd");
    emit_vex(assembler, 1U, true, 0x5C, XMM2, XMM1, XMM5, "vsubpd");
    emit_vex(assembler, 1U, true, 0x59, XMM3, XMM2, XMM5, "vmulpd");
    emit_vex(assembler, 1U, true, 0x57, XMM4, XMM4, XMM4, "vxorpd");
    emit_vex_compare(assembler, XMM3, XMM3, XMM4, 0x1E); // GT_OQ
    emit_vex_compare(assembler, XMM2, XMM2, XMM4, 0x0C); // NEQ_OQ
    emit_narrow_masks(assembler, XMM3, XMM3);
    emit_narrow_masks(assembler, XMM2, XMM2);
    bal_x86_emit_vzeroupper(assembler);

    // The decision waits for VZEROUPPER, so that the slow path runs without upper halves set.
    bal_x86_emit_test_r32_r32(assembler, SCRATCH_A, SCRATCH_A);
    jumps[1] = emit_forward_jump(assembler, true, BAL_X86_COND_NE);

    // Towards zero is one less in magnitude where rounding went away from it; then the last bit
    // is set where it was inexact.
    emit_sse(assembler, 0x66, false, 0xFE, XMM0, XMM3, "paddd");
    emit_sse_immediate(assembler, 0x66, 0x72, 2U, XMM2, 31U, "psrld");
    emit_sse(assembler, 0x66, false, 0xEB, XMM0, XMM2, "por");
    emit_singles_to_halves(assembler, XMM0, XMM0);
}

/// Converts four halves of the raw lanes in `source` (from half lane `first`) to single
/// precision in `destination`, flipping their signs first with `is_negated`. Uses xmm4.
static void
emit_half_operand(bal_x86_assembler_t *BAL_RESTRICT assembler,
                  const uint8_t                     destination,
                  const uint8_t                     source,
                  const uint32_t                    first,
                  const bool                        is_negated)
{
    if (first != 0U)
    {
        emit_sse_immediate(assembler, 0x66, 0x73, 3U, source, (uint8_t)(2U * first), "psrldq");
    }

    if (true == is_negated)
    {
        emit_broadcast_pattern(assembler, XMM4, 0x8000800080008000ULL);
        emit_sse(assembler, 0x66, false, 0xEF, source, XMM4, "pxor");
    }

    emit_halves_to_singles(assembler, destination, source);
}

/// FMLA and FMLS (vector, by element and scalar by element), FMADD, FMSUB, FNMADD and FNMSUB,
/// and FRECPS and FRSQRTS (vector and scalar), in half precision, through emit_half_fused().
/// FMLS and the subtracting forms negate an operand, which the fused rounding allows. FRECPS is
/// 2 - n * m and FRSQRTS 1.5 - (n / 2) * m (halving is exact); their zero times infinity, which
/// gives 2 or 1.5, takes the slow path with the other infinities.
static bool
emit_fast_half_fused(bal_tier1_compiler_t *BAL_RESTRICT compiler, const uint32_t instruction)
{
    // Vector: 0 Q 0 01110 a 10 Rm 00 001 1 Rn Rd (FMLA a = 0, FMLS a = 1) or 00 111 1 (FRECPS,
    // FRSQRTS). By element: 0 Q 0 01111 00 L M Rm 0 S 01 H 0 Rn Rd, scalar 01 0 11111 ... Scalar
    // steps: 01 0 11110 a 10 Rm 00 111 1. Multiply-add: 0 0 0 11111 11 o1 Rm o0 Ra Rn Rd.
    const uint32_t three   = instruction & 0xBF60FC00U;
    const uint32_t element = instruction & 0xBFC0B400U;
    const bool     is_add  = 0x0E400C00U == three;
    const bool     is_step = 0x0E403C00U == three || 0x5E403C00U == (instruction & 0xFF60FC00U);
    const bool     is_lane = 0x0F001000U == element || 0x5F001000U == (instruction & 0xFFC0B400U);
    const bool     is_madd = 0x1FC00000U == (instruction & 0xFFC00000U);

    if ((false == is_add && false == is_step && false == is_lane && false == is_madd)
        || false == has_feature(compiler, BAL_TIER1_FEATURE_FMA)
        || false == has_feature(compiler, BAL_TIER1_FEATURE_SSE41))
    {
        return false;
    }

    const bool is_scalar = true == is_madd || 0x5F001000U == (instruction & 0xFFC0B400U)
                           || 0x5E403C00U == (instruction & 0xFF60FC00U);
    const bool     is_quad = false == is_scalar && 1U == ((instruction >> 30U) & 1U);
    const uint32_t rd      = instruction & 0x1FU;
    const uint32_t rn      = (instruction >> 5U) & 0x1FU;
    const uint32_t rm      = (instruction >> 16U) & 0x1FU;
    const bool     is_root = true == is_step && 1U == ((instruction >> 23U) & 1U);

    // Whether n and the addend are negated, and the addend: Rd, Ra or a constant.
    bool     negate_n      = false;
    bool     negate_addend = false;
    uint32_t addend        = rd;

    if (true == is_add)
    {
        negate_n = 1U == ((instruction >> 23U) & 1U);
    }
    else if (true == is_lane)
    {
        negate_n = 1U == ((instruction >> 14U) & 1U);
    }
    else if (true == is_madd)
    {
        const uint32_t o1 = (instruction >> 21U) & 1U;
        const uint32_t o0 = (instruction >> 15U) & 1U;
        addend            = (instruction >> 10U) & 0x1FU;
        negate_n          = o0 != o1;
        negate_addend     = 1U == o1;
    }
    else
    {
        negate_n = true;
    }

    bal_x86_assembler_t *BAL_RESTRICT assembler = &compiler->assembler;
    size_t                            slow_jumps[6];
    uint32_t                          count = 1U;

    slow_jumps[0] = begin_float_sequence(compiler);

    // The register cache is empty now: RCX holds the element of a by element form, RAX and RDX
    // the two halves of the result.
    if (true == is_lane)
    {
        const uint32_t index = (((instruction >> 11U) & 1U) << 2U)
                               | (((instruction >> 21U) & 1U) << 1U) | ((instruction >> 20U) & 1U);
        bal_x86_emit_load_r64_rbp_offset(
            assembler, BAL_X86_RCX, (int32_t)offsetof(bal_cpu_t, v[rm & 0xFU][index / 4U]));
        bal_x86_emit_shr_r64_imm8(assembler, BAL_X86_RCX, (uint8_t)(16U * (index % 4U)));
    }

    const uint32_t groups = true == is_quad ? 2U : 1U;

    for (uint32_t group = 0U; group < groups; ++group)
    {
        // XMM0 = n, XMM1 = m, XMM2 = the addend, in single precision.
        if (true == is_scalar)
        {
            emit_load_element(assembler, XMM3, rn, 1U);
        }
        else
        {
            emit_load_vector(assembler, XMM3, rn, is_quad);
        }

        emit_half_operand(assembler, XMM0, XMM3, 4U * group, negate_n);

        if (true == is_root)
        {
            emit_broadcast_pattern(assembler, XMM3, 0x3F0000003F000000ULL);
            emit_sse(assembler, 0U, false, 0x59, XMM0, XMM3, "mulps");
        }

        if (true == is_lane)
        {
            emit_sse(assembler, 0x66, false, 0x6E, XMM3, (uint8_t)BAL_X86_RCX, "movd");
            emit_sse_immediate(assembler, 0xF2, 0x70, XMM3, XMM3, 0U, "pshuflw");
            emit_halves_to_singles(assembler, XMM1, XMM3);
        }
        else
        {
            if (true == is_scalar)
            {
                emit_load_element(assembler, XMM3, rm, 1U);
            }
            else
            {
                emit_load_vector(assembler, XMM3, rm, is_quad);
            }

            emit_half_operand(assembler, XMM1, XMM3, 4U * group, false);
        }

        if (true == is_step)
        {
            emit_broadcast_pattern(
                assembler, XMM2, true == is_root ? 0x3FC000003FC00000ULL : 0x4000000040000000ULL);
        }
        else
        {
            if (true == is_scalar)
            {
                emit_load_element(assembler, XMM3, addend, 1U);
            }
            else
            {
                emit_load_vector(assembler, XMM3, addend, is_quad);
            }

            emit_half_operand(assembler, XMM2, XMM3, 4U * group, negate_addend);
        }

        emit_half_fused(assembler, &slow_jumps[count]);
        count += 2U;
        emit_sse(assembler,
                 0x66,
                 true,
                 0x7E,
                 XMM0,
                 (uint8_t)(0U == group ? BAL_X86_RAX : BAL_X86_RDX),
                 "movq");
    }

    emit_sse(assembler, 0x66, true, 0x6E, XMM0, (uint8_t)BAL_X86_RAX, "movq");

    if (true == is_quad)
    {
        emit_sse(assembler, 0x66, true, 0x6E, XMM1, (uint8_t)BAL_X86_RDX, "movq");
        emit_sse(assembler, 0x66, false, 0x6C, XMM0, XMM1, "punpcklqdq");
    }

    slow_jumps[count++] = emit_half_checks(assembler);

    if (true == is_scalar)
    {
        emit_store_element(assembler, XMM0, rd, 1U);
    }
    else
    {
        emit_store_vector(assembler, XMM0, rd, is_quad);
    }

    emit_slow_path(compiler, instruction, float_helper(instruction), slow_jumps, count);
    return true;
}

// -----------------------------------------------------------------------------
// The rest of half precision
// -----------------------------------------------------------------------------

/// Loads the `half`-th group of four halves of SIMD&FP register `index` (all of a 64-bit vector,
/// or the halfword element 0 with `is_scalar`) into `raw`, converted to single precision into
/// `single`.
static void
emit_half_group(bal_x86_assembler_t *BAL_RESTRICT assembler,
                const uint8_t                     single,
                const uint8_t                     raw,
                const uint32_t                    index,
                const uint32_t                    group,
                const bool                        is_quad,
                const bool                        is_scalar)
{
    if (true == is_scalar)
    {
        emit_load_half(assembler, raw, index);
    }
    else
    {
        emit_load_vector(assembler, raw, index, is_quad);
    }

    if (group != 0U)
    {
        emit_sse_immediate(assembler, 0x66, 0x73, 3U, raw, 8U, "psrldq");
    }

    emit_halves_to_singles(assembler, single, raw);
}

/// Returns a jump to the slow path for a single precision lane of `xmm` that is infinite or a
/// NaN. Uses xmm2 and xmm3.
static size_t
emit_infinity_check(bal_x86_assembler_t *BAL_RESTRICT assembler, const uint8_t xmm)
{
    emit_broadcast_pattern(assembler, XMM2, 0x7F8000007F800000ULL);
    emit_sse(assembler, 0U, false, 0x28, XMM3, xmm, "movaps");
    emit_sse(assembler, 0x66, false, 0xDB, XMM3, XMM2, "pand");
    emit_sse(assembler, 0x66, false, 0x76, XMM3, XMM2, "pcmpeqd");
    emit_sse(assembler, 0U, false, 0x50, (uint8_t)SCRATCH_A, XMM3, "movmskps");
    bal_x86_emit_test_r32_r32(assembler, SCRATCH_A, SCRATCH_A);
    return emit_forward_jump(assembler, true, BAL_X86_COND_NE);
}

/// Ends a half precision sequence whose results are in RAX (and RDX for the high four lanes of a
/// full vector): checks them with emit_half_checks() when `is_checked`, stores them to `rd`, and
/// emits the slow path.
static void
emit_half_store(bal_tier1_compiler_t *BAL_RESTRICT compiler,
                const uint32_t                     instruction,
                const uint32_t                     rd,
                const bool                         is_quad,
                const bool                         is_scalar,
                const bool                         is_checked,
                size_t                            *slow_jumps,
                uint32_t                           count)
{
    bal_x86_assembler_t *BAL_RESTRICT assembler = &compiler->assembler;
    emit_sse(assembler, 0x66, true, 0x6E, XMM0, (uint8_t)BAL_X86_RAX, "movq");

    if (true == is_quad)
    {
        emit_sse(assembler, 0x66, true, 0x6E, XMM1, (uint8_t)BAL_X86_RDX, "movq");
        emit_sse(assembler, 0x66, false, 0x6C, XMM0, XMM1, "punpcklqdq");
    }

    if (true == is_checked)
    {
        slow_jumps[count++] = emit_half_checks(assembler);
    }

    if (true == is_scalar)
    {
        emit_store_element(assembler, XMM0, rd, 1U);
    }
    else
    {
        emit_store_vector(assembler, XMM0, rd, is_quad);
    }

    emit_slow_path(compiler, instruction, float_helper(instruction), slow_jumps, count);
}

/// Saves the four halves in the low 64 bits of xmm0, the results of `group`, in RAX or RDX.
static void
emit_half_result(bal_x86_assembler_t *BAL_RESTRICT assembler, const uint32_t group)
{
    emit_sse(assembler,
             0x66,
             true,
             0x7E,
             XMM0,
             (uint8_t)(0U == group ? BAL_X86_RAX : BAL_X86_RDX),
             "movq");
}

/// FMAX, FMIN, FMAXNM and FMINNM (vector and scalar), FMAXP, FMINP, FMAXNMP and FMINNMP (vector
/// and scalar), and FMAXV, FMINV, FMAXNMV and FMINNMV in half precision: the halves convert to
/// single precision exactly and emit_float_extreme() picks among them, so converting back is
/// exact. The pairwise forms split the elements first, as emit_deinterleave() does.
static bool
emit_fast_half_extreme(bal_tier1_compiler_t *BAL_RESTRICT compiler, const uint32_t instruction)
{
    // Vector: 0 Q U 01110 o1 10 Rm 00 opcode 1 Rn Rd with opcode 000 (NM) or 110, U = 1 pairwise.
    // Scalar: 0 0 0 11110 11 1 Rm 01 NM o1 10 Rn Rd. Scalar pairwise: 01 0 11110 o1 0 11000 opcode
    // 10 Rn Rd with opcode 01100 (NM) or 01111. Across: 0 Q 0 01110 o1 0 11000 opcode 10 Rn Rd.
    const uint32_t vector         = instruction & 0x9F60FC00U;
    const bool     is_vector      = 0x0E400400U == vector || 0x0E403400U == vector;
    const bool     is_scalar_two  = 0x1EE04800U == (instruction & 0xFFE0CC00U);
    const uint32_t other          = instruction & 0xFF7FFC00U;
    const bool     is_scalar_pair = 0x5E30C800U == other || 0x5E30F800U == other;
    const uint32_t across         = instruction & 0xBF7FFC00U;
    const bool     is_across      = 0x0E30C800U == across || 0x0E30F800U == across;

    if ((false == is_vector && false == is_scalar_two && false == is_scalar_pair
         && false == is_across)
        || false == has_feature(compiler, BAL_TIER1_FEATURE_FMA)
        || false == has_feature(compiler, BAL_TIER1_FEATURE_SSE41))
    {
        return false;
    }

    const bool     is_minimum = true == is_scalar_two ? 1U == ((instruction >> 12U) & 1U)
                                                      : 1U == ((instruction >> 23U) & 1U);
    const bool     is_pair    = true == is_vector && 1U == ((instruction >> 29U) & 1U);
    const bool     is_scalar  = true == is_scalar_two || true == is_scalar_pair;
    const bool     is_quad    = false == is_scalar && 1U == ((instruction >> 30U) & 1U);
    const uint32_t rd         = instruction & 0x1FU;
    const uint32_t rn         = (instruction >> 5U) & 0x1FU;
    const uint32_t rm         = (instruction >> 16U) & 0x1FU;
    bal_x86_assembler_t *BAL_RESTRICT assembler = &compiler->assembler;
    size_t                            slow_jumps[5];
    uint32_t                          count = 1U;

    slow_jumps[0] = begin_float_sequence(compiler);

    if (true == is_across || true == is_scalar_pair)
    {
        // The elements of Rn, reduced to one.
        if (true == is_scalar_pair)
        {
            emit_load_scalar(assembler, XMM4, rn, false);
            emit_halves_to_singles(assembler, XMM0, XMM4);
            emit_sse_immediate(assembler, 0x66, 0x70, XMM1, XMM0, 0x55, "pshufd");
            slow_jumps[count++] = emit_float_extreme(assembler, 0xF3, 0U, true, is_minimum);
        }
        else
        {
            emit_half_group(assembler, XMM0, XMM4, rn, 0U, is_quad, false);

            if (true == is_quad)
            {
                emit_half_group(assembler, XMM1, XMM4, rn, 1U, true, false);
                slow_jumps[count++] = emit_float_extreme(assembler, 0U, 0U, false, is_minimum);
            }

            emit_sse_immediate(assembler, 0x66, 0x70, XMM1, XMM0, 0xEE, "pshufd");
            slow_jumps[count++] = emit_float_extreme(assembler, 0U, 0U, false, is_minimum);
            emit_sse_immediate(assembler, 0x66, 0x70, XMM1, XMM0, 0x55, "pshufd");
            slow_jumps[count++] = emit_float_extreme(assembler, 0xF3, 0U, true, is_minimum);
        }

        // The scalar extreme leaves other bits in lanes 1 to 3, which must not raise flags.
        emit_load_vector_constant(assembler, XMM1, 0xFFFFFFFFULL, 0U);
        emit_sse(assembler, 0x66, false, 0xDB, XMM0, XMM1, "pand");
        emit_singles_to_halves(assembler, XMM0, XMM0);
        emit_store_element(assembler, XMM0, rd, 1U);
        emit_slow_path(compiler, instruction, float_helper(instruction), slow_jumps, count);
        return true;
    }

    // XMM4 and XMM5 = the raw operands: Rn and Rm, or their even and odd elements.
    if (true == is_pair)
    {
        emit_load_vector(assembler, XMM0, rn, is_quad);
        emit_load_vector(assembler, XMM1, rm, is_quad);
        emit_deinterleave(assembler, 1U, is_quad);
        emit_sse(assembler, 0U, false, 0x28, XMM4, XMM0, "movaps");
        emit_sse(assembler, 0U, false, 0x28, XMM5, XMM1, "movaps");
    }

    for (uint32_t group = 0U; group < (true == is_quad ? 2U : 1U); ++group)
    {
        if (true == is_pair)
        {
            emit_halves_to_singles(assembler, XMM0, XMM4);
            emit_halves_to_singles(assembler, XMM1, XMM5);
            emit_sse_immediate(assembler, 0x66, 0x73, 3U, XMM4, 8U, "psrldq");
            emit_sse_immediate(assembler, 0x66, 0x73, 3U, XMM5, 8U, "psrldq");
        }
        else
        {
            emit_half_group(assembler, XMM0, XMM4, rn, group, is_quad, is_scalar);
            emit_half_group(assembler, XMM1, XMM5, rm, group, is_quad, is_scalar);
        }

        slow_jumps[count++] = emit_float_extreme(assembler, 0U, 0U, false, is_minimum);
        emit_singles_to_halves(assembler, XMM0, XMM0);
        emit_half_result(assembler, group);
    }

    emit_half_store(compiler, instruction, rd, is_quad, is_scalar, false, slow_jumps, count);
    return true;
}

/// FADDP (vector and scalar), FABD (scalar), FMULX (vector and scalar), and FMUL and FMULX (by
/// element, vector and scalar) in half precision, computed in single precision and rounded once
/// more, which gives the correctly rounded sum or product (see emit_fast_half_arithmetic()).
/// FMULX's infinities and NaNs take the slow path, which covers its zero times infinity.
static bool
emit_fast_half_arithmetic_more(bal_tier1_compiler_t *BAL_RESTRICT compiler,
                               const uint32_t                     instruction)
{
    // FADDP: 0 Q 1 01110 0 10 Rm 00 010 1 Rn Rd, scalar 01 0 11110 0 0 11000 01101 10 Rn Rd. FABD:
    // 01 1 11110 1 10 Rm 00 010 1 Rn Rd. FMULX: 0 Q 0 01110 0 10 Rm 00 011 1 Rn Rd, scalar 01 0
    // 11110 ... By element: 0 Q U 01111 00 L M Rm 1001 H 0 Rn Rd (FMUL U = 0, FMULX U = 1),
    // scalar 01 U 11111 ...
    const bool is_pair            = 0x2E401400U == (instruction & 0xBFE0FC00U);
    const bool is_scalar_pair     = 0x5E30D800U == (instruction & 0xFFFFFC00U);
    const bool is_difference      = 0x7EC01400U == (instruction & 0xFFE0FC00U);
    const bool is_extended        = 0x0E401C00U == (instruction & 0xBFE0FC00U);
    const bool is_scalar_extended = 0x5E401C00U == (instruction & 0xFFE0FC00U);
    const bool is_element         = 0x0F009000U == (instruction & 0x9FC0F400U);
    const bool is_scalar_element  = 0x5F009000U == (instruction & 0xDFC0F400U);

    if ((false == is_pair && false == is_scalar_pair && false == is_difference
         && false == is_extended && false == is_scalar_extended && false == is_element
         && false == is_scalar_element)
        || false == has_feature(compiler, BAL_TIER1_FEATURE_FMA)
        || false == has_feature(compiler, BAL_TIER1_FEATURE_SSE41))
    {
        return false;
    }

    const bool is_scalar = true == is_scalar_pair || true == is_difference
                           || true == is_scalar_extended || true == is_scalar_element;
    const bool is_quad = false == is_scalar && 1U == ((instruction >> 30U) & 1U);
    const bool is_mulx = true == is_extended || true == is_scalar_extended
                         || ((true == is_element || true == is_scalar_element)
                             && 1U == ((instruction >> 29U) & 1U));
    const uint32_t                    rd        = instruction & 0x1FU;
    const uint32_t                    rn        = (instruction >> 5U) & 0x1FU;
    const uint32_t                    rm        = (instruction >> 16U) & 0x1FU;
    bal_x86_assembler_t *BAL_RESTRICT assembler = &compiler->assembler;
    size_t                            slow_jumps[6];
    uint32_t                          count = 1U;

    slow_jumps[0] = begin_float_sequence(compiler);

    if (true == is_scalar_pair)
    {
        emit_load_scalar(assembler, XMM4, rn, false);
        emit_halves_to_singles(assembler, XMM0, XMM4);
        emit_sse(assembler, 0xF2, false, 0x7C, XMM0, XMM0, "haddps");
        emit_singles_to_halves(assembler, XMM0, XMM0);
        emit_half_result(assembler, 0U);
        emit_half_store(compiler, instruction, rd, false, true, true, slow_jumps, count);
        return true;
    }

    // RCX = the element of a by element form, whose register Rd may be.
    if (true == is_element || true == is_scalar_element)
    {
        const uint32_t index = (((instruction >> 11U) & 1U) << 2U)
                               | (((instruction >> 21U) & 1U) << 1U) | ((instruction >> 20U) & 1U);
        bal_x86_emit_load_r64_rbp_offset(
            assembler, BAL_X86_RCX, (int32_t)offsetof(bal_cpu_t, v[rm & 0xFU][index / 4U]));
        bal_x86_emit_shr_r64_imm8(assembler, BAL_X86_RCX, (uint8_t)(16U * (index % 4U)));
    }

    if (true == is_pair)
    {
        emit_load_vector(assembler, XMM0, rn, is_quad);
        emit_load_vector(assembler, XMM1, rm, is_quad);
        emit_deinterleave(assembler, 1U, is_quad);
        emit_sse(assembler, 0U, false, 0x28, XMM4, XMM0, "movaps");
        emit_sse(assembler, 0U, false, 0x28, XMM5, XMM1, "movaps");
    }

    for (uint32_t group = 0U; group < (true == is_quad ? 2U : 1U); ++group)
    {
        if (true == is_pair)
        {
            emit_halves_to_singles(assembler, XMM0, XMM4);
            emit_halves_to_singles(assembler, XMM1, XMM5);
            emit_sse_immediate(assembler, 0x66, 0x73, 3U, XMM4, 8U, "psrldq");
            emit_sse_immediate(assembler, 0x66, 0x73, 3U, XMM5, 8U, "psrldq");
        }
        else
        {
            emit_half_group(assembler, XMM0, XMM4, rn, group, is_quad, is_scalar);

            if (true == is_element || true == is_scalar_element)
            {
                emit_sse(assembler, 0x66, false, 0x6E, XMM5, (uint8_t)BAL_X86_RCX, "movd");
                emit_sse_immediate(assembler, 0xF2, 0x70, XMM5, XMM5, 0U, "pshuflw");
                emit_halves_to_singles(assembler, XMM1, XMM5);

                if (true == is_scalar)
                {
                    // Only lane 0 is the scalar's; zero times infinity elsewhere must not count.
                    emit_load_vector_constant(assembler, XMM5, 0xFFFFFFFFULL, 0U);
                    emit_sse(assembler, 0x66, false, 0xDB, XMM1, XMM5, "pand");
                }
            }
            else
            {
                emit_half_group(assembler, XMM1, XMM5, rm, group, is_quad, is_scalar);
            }
        }

        if (true == is_mulx)
        {
            slow_jumps[count++] = emit_infinity_check(assembler, XMM0);
            slow_jumps[count++] = emit_infinity_check(assembler, XMM1);
        }

        if (true == is_pair)
        {
            emit_sse(assembler, 0U, false, 0x58, XMM0, XMM1, "addps");
        }
        else if (true == is_difference)
        {
            emit_sse(assembler, 0U, false, 0x5C, XMM0, XMM1, "subps");
            emit_all_ones(assembler, XMM1);
            emit_sse_immediate(assembler, 0x66, 0x72, 2U, XMM1, 1U, "psrld");
            emit_sse(assembler, 0U, false, 0x54, XMM0, XMM1, "andps");
        }
        else
        {
            emit_sse(assembler, 0U, false, 0x59, XMM0, XMM1, "mulps");
        }

        emit_singles_to_halves(assembler, XMM0, XMM0);
        emit_half_result(assembler, group);
    }

    emit_half_store(compiler, instruction, rd, is_quad, is_scalar, true, slow_jumps, count);
    return true;
}

/// FCVT from double to half precision: rounded to single precision to odd (towards zero with the
/// last bit set if inexact), as emit_half_fused() does, from which VCVTPS2PH rounds correctly.
/// Infinities, whose distance back would raise Invalid, and NaNs take the slow path.
static bool
emit_fast_double_to_half(bal_tier1_compiler_t *BAL_RESTRICT compiler, const uint32_t instruction)
{
    // FCVT Hd, Dn: 0x1E63C000.
    if (0x1E63C000U != (instruction & 0xFFFFFC00U)
        || false == has_feature(compiler, BAL_TIER1_FEATURE_FMA))
    {
        return false;
    }

    bal_x86_assembler_t *BAL_RESTRICT assembler = &compiler->assembler;
    size_t                            slow_jumps[3];

    slow_jumps[0] = begin_float_sequence(compiler);
    emit_load_scalar(assembler, XMM1, (instruction >> 5U) & 0x1FU, true);
    slow_jumps[1] = emit_register_magnitude_check(assembler, XMM1, true, 2047U);

    // XMM0 = d to nearest single precision; XMM2 = back - d; XMM3 = the rounding away from zero.
    emit_sse(assembler, 0U, false, 0x57, XMM0, XMM0, "xorps");
    emit_sse(assembler, 0xF2, false, 0x5A, XMM0, XMM1, "cvtsd2ss");
    emit_sse(assembler, 0U, false, 0x57, XMM2, XMM2, "xorps");
    emit_sse(assembler, 0xF3, false, 0x5A, XMM2, XMM0, "cvtss2sd");
    emit_sse(assembler, 0xF2, false, 0x5C, XMM2, XMM1, "subsd");
    emit_sse(assembler, 0U, false, 0x28, XMM3, XMM2, "movaps");
    emit_sse(assembler, 0xF2, false, 0x59, XMM3, XMM1, "mulsd");
    emit_zero_vector(assembler, XMM4);
    emit_sse_immediate(assembler, 0x66, 0xC2, XMM3, XMM4, 6U, "cmpnlepd");
    emit_sse_immediate(assembler, 0x66, 0xC2, XMM2, XMM4, 4U, "cmpneqpd");
    emit_sse_immediate(assembler, 0x66, 0x70, XMM3, XMM3, 0x08, "pshufd");
    emit_sse_immediate(assembler, 0x66, 0x70, XMM2, XMM2, 0x08, "pshufd");
    emit_sse(assembler, 0x66, false, 0xFE, XMM0, XMM3, "paddd");
    emit_sse_immediate(assembler, 0x66, 0x72, 2U, XMM2, 31U, "psrld");
    emit_sse(assembler, 0x66, false, 0xEB, XMM0, XMM2, "por");

    // Only lane 0 holds the value.
    emit_load_vector_constant(assembler, XMM1, 0xFFFFFFFFULL, 0U);
    emit_sse(assembler, 0x66, false, 0xDB, XMM0, XMM1, "pand");
    emit_singles_to_halves(assembler, XMM0, XMM0);
    slow_jumps[2] = emit_half_checks(assembler);
    emit_store_element(assembler, XMM0, instruction & 0x1FU, 1U);
    emit_slow_path(compiler, instruction, float_helper(instruction), slow_jumps, 3U);
    return true;
}

/// FRECPE and FRSQRTE (vector and scalar) in half precision: their estimates are those of the
/// exact single precision values (the same eight fraction bits index the same tables, and the
/// exponents map with the biases), converted back exactly, so emit_estimate() computes them with
/// the half precision ranges.
static bool
emit_fast_half_estimate(bal_tier1_compiler_t *BAL_RESTRICT compiler, const uint32_t instruction)
{
    // 0 Q U 01110 1 1111 00 11101 10 Rn Rd, or the scalar 01 U 11110 ...
    const bool is_scalar = 0x5EF9D800U == (instruction & 0xDFFFFC00U);

    if ((0x0EF9D800U != (instruction & 0x9FFFFC00U) && false == is_scalar)
        || false == has_feature(compiler, BAL_TIER1_FEATURE_FMA)
        || false == has_feature(compiler, BAL_TIER1_FEATURE_SSE41))
    {
        return false;
    }

    const bool     is_root = 1U == ((instruction >> 29U) & 1U);
    const bool     is_quad = false == is_scalar && 1U == ((instruction >> 30U) & 1U);
    const uint32_t rn      = (instruction >> 5U) & 0x1FU;
    bal_x86_assembler_t *BAL_RESTRICT assembler = &compiler->assembler;
    size_t                            slow_jumps[3];
    uint32_t                          count = 1U;

    slow_jumps[0] = begin_float_sequence(compiler);

    for (uint32_t group = 0U; group < (true == is_quad ? 2U : 1U); ++group)
    {
        emit_half_group(assembler, XMM0, XMM4, rn, group, is_quad, is_scalar);
        slow_jumps[count++] = emit_estimate(assembler, is_root, true == is_scalar ? 1U : 4U, true);

        // The unused lanes of a scalar get exponents of their own, which must not raise flags.
        if (true == is_scalar)
        {
            emit_load_vector_constant(assembler, XMM2, 0xFFFFFFFFULL, 0U);
            emit_sse(assembler, 0x66, false, 0xDB, XMM1, XMM2, "pand");
        }

        emit_singles_to_halves(assembler, XMM0, XMM1);
        emit_half_result(assembler, group);
    }

    emit_half_store(
        compiler, instruction, instruction & 0x1FU, is_quad, is_scalar, false, slow_jumps, count);
    return true;
}

/// The URSQRTE estimates (RecipSqrtEstimate() & 0xFF) by the operand's top nine bits, from 128.
static const uint8_t UNSIGNED_SQUARE_ROOT_ESTIMATES[512] = {
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0xFF, 0xFD, 0xFB, 0xF9, 0xF7, 0xF5, 0xF3, 0xF2, 0xF0, 0xEE, 0xEC, 0xEA, 0xE9, 0xE7, 0xE5, 0xE4,
    0xE2, 0xE0, 0xDF, 0xDD, 0xDB, 0xDA, 0xD8, 0xD7, 0xD5, 0xD4, 0xD2, 0xD1, 0xCF, 0xCE, 0xCC, 0xCB,
    0xC9, 0xC8, 0xC6, 0xC5, 0xC4, 0xC2, 0xC1, 0xC0, 0xBE, 0xBD, 0xBC, 0xBA, 0xB9, 0xB8, 0xB7, 0xB5,
    0xB4, 0xB3, 0xB2, 0xB0, 0xAF, 0xAE, 0xAD, 0xAC, 0xAA, 0xA9, 0xA8, 0xA7, 0xA6, 0xA5, 0xA4, 0xA3,
    0xA2, 0xA0, 0x9F, 0x9E, 0x9D, 0x9C, 0x9B, 0x9A, 0x99, 0x98, 0x97, 0x96, 0x95, 0x94, 0x93, 0x92,
    0x91, 0x90, 0x8F, 0x8E, 0x8D, 0x8C, 0x8C, 0x8B, 0x8A, 0x89, 0x88, 0x87, 0x86, 0x85, 0x84, 0x83,
    0x83, 0x82, 0x81, 0x80, 0x7F, 0x7E, 0x7E, 0x7D, 0x7C, 0x7B, 0x7A, 0x79, 0x79, 0x78, 0x77, 0x76,
    0x76, 0x75, 0x74, 0x73, 0x72, 0x72, 0x71, 0x70, 0x6F, 0x6F, 0x6E, 0x6D, 0x6D, 0x6C, 0x6B, 0x6A,
    0x69, 0x69, 0x68, 0x68, 0x67, 0x67, 0x65, 0x65, 0x64, 0x64, 0x63, 0x63, 0x61, 0x61, 0x60, 0x60,
    0x5F, 0x5F, 0x5D, 0x5D, 0x5C, 0x5C, 0x5B, 0x5B, 0x5A, 0x5A, 0x58, 0x58, 0x57, 0x57, 0x56, 0x56,
    0x55, 0x55, 0x54, 0x54, 0x52, 0x52, 0x51, 0x51, 0x50, 0x50, 0x4F, 0x4F, 0x4E, 0x4E, 0x4D, 0x4D,
    0x4C, 0x4C, 0x4B, 0x4B, 0x4A, 0x4A, 0x48, 0x48, 0x47, 0x47, 0x46, 0x46, 0x45, 0x45, 0x44, 0x44,
    0x43, 0x43, 0x42, 0x42, 0x41, 0x41, 0x40, 0x40, 0x3F, 0x3F, 0x3E, 0x3E, 0x3D, 0x3D, 0x3C, 0x3C,
    0x3C, 0x3C, 0x3B, 0x3B, 0x3A, 0x3A, 0x39, 0x39, 0x38, 0x38, 0x37, 0x37, 0x36, 0x36, 0x35, 0x35,
    0x34, 0x34, 0x33, 0x33, 0x33, 0x33, 0x32, 0x32, 0x31, 0x31, 0x30, 0x30, 0x2F, 0x2F, 0x2E, 0x2E,
    0x2E, 0x2E, 0x2D, 0x2D, 0x2C, 0x2C, 0x2B, 0x2B, 0x2A, 0x2A, 0x2A, 0x2A, 0x29, 0x29, 0x28, 0x28,
    0x27, 0x27, 0x26, 0x26, 0x26, 0x26, 0x25, 0x25, 0x24, 0x24, 0x23, 0x23, 0x23, 0x23, 0x22, 0x22,
    0x21, 0x21, 0x21, 0x21, 0x20, 0x20, 0x1F, 0x1F, 0x1E, 0x1E, 0x1E, 0x1E, 0x1D, 0x1D, 0x1C, 0x1C,
    0x1C, 0x1C, 0x1B, 0x1B, 0x1A, 0x1A, 0x1A, 0x1A, 0x19, 0x19, 0x18, 0x18, 0x18, 0x18, 0x17, 0x17,
    0x16, 0x16, 0x16, 0x16, 0x15, 0x15, 0x14, 0x14, 0x14, 0x14, 0x13, 0x13, 0x13, 0x13, 0x12, 0x12,
    0x11, 0x11, 0x11, 0x11, 0x10, 0x10, 0x10, 0x10, 0x0F, 0x0F, 0x0E, 0x0E, 0x0E, 0x0E, 0x0D, 0x0D,
    0x0D, 0x0D, 0x0C, 0x0C, 0x0B, 0x0B, 0x0B, 0x0B, 0x0A, 0x0A, 0x0A, 0x0A, 0x09, 0x09, 0x09, 0x09,
    0x08, 0x08, 0x08, 0x08, 0x07, 0x07, 0x06, 0x06, 0x06, 0x06, 0x05, 0x05, 0x05, 0x05, 0x04, 0x04,
    0x04, 0x04, 0x03, 0x03, 0x03, 0x03, 0x02, 0x02, 0x02, 0x02, 0x01, 0x01, 0x01, 0x01, 0x00, 0x00,
};

/// FABD (vector and scalar) in single and double precision: the difference, checked as FSUB's,
/// with its sign cleared.
static bool
emit_fast_float_difference(bal_tier1_compiler_t *BAL_RESTRICT compiler, const uint32_t instruction)
{
    // 0 Q 1 01110 1 sz 1 Rm 11010 1 Rn Rd, or the scalar 01 1 11110 1 sz 1 Rm 11010 1 Rn Rd.
    const bool is_scalar = 0x7EA0D400U == (instruction & 0xFFA0FC00U);
    const bool is_double = 1U == ((instruction >> 22U) & 1U);
    const bool is_quad   = false == is_scalar && 1U == ((instruction >> 30U) & 1U);

    if ((0x2EA0D400U != (instruction & 0xBFA0FC00U) && false == is_scalar)
        || (true == is_double && false == is_quad && false == is_scalar))
    {
        return false;
    }

    const uint32_t                    rd        = instruction & 0x1FU;
    const uint32_t                    rn        = (instruction >> 5U) & 0x1FU;
    const uint32_t                    rm        = (instruction >> 16U) & 0x1FU;
    bal_x86_assembler_t *BAL_RESTRICT assembler = &compiler->assembler;
    size_t                            slow_jumps[3];

    slow_jumps[0] = begin_float_sequence(compiler);

    if (true == is_scalar)
    {
        emit_load_scalar(assembler, XMM0, rn, is_double);
        emit_load_scalar(assembler, XMM1, rm, is_double);
    }
    else
    {
        emit_load_vector(assembler, XMM0, rn, is_quad);
        emit_load_vector(assembler, XMM1, rm, is_quad);
    }

    emit_sse(assembler, true == is_double ? 0x66 : 0U, false, 0x5C, XMM0, XMM1, "subp");
    const uint32_t count = 1U + emit_vector_checks(assembler, is_double, &slow_jumps[1]);
    emit_all_ones(assembler, XMM1);
    emit_sse_immediate(assembler, 0x66, true == is_double ? 0x73 : 0x72, 2U, XMM1, 1U, "psrl");
    emit_sse(assembler, 0U, false, 0x54, XMM0, XMM1, "andps");

    if (true == is_scalar)
    {
        emit_sse(assembler, 0x66, is_double, 0x7E, XMM0, (uint8_t)SCRATCH_A, "movd/movq");
        emit_store_scalar_result(assembler, rd);
    }
    else
    {
        emit_store_vector(assembler, XMM0, rd, is_quad);
    }

    emit_slow_path(compiler, instruction, float_helper(instruction), slow_jumps, count);
    return true;
}

/// FRECPX in half, single and double precision: the sign with the exponent inverted, or the
/// largest normal exponent for zeros and denormals, and a zero fraction. Infinities and NaNs
/// take the slow path.
static bool
emit_fast_reciprocal_exponent(bal_tier1_compiler_t *BAL_RESTRICT compiler,
                              const uint32_t                     instruction)
{
    // 01 0 11110 1 sz 10000 11111 10 Rn Rd, or with bits 22-17 111100 for half precision.
    const uint32_t form = instruction & 0xFFFFFC00U;

    if ((0x5EA1F800U != form && 0x5EE1F800U != form && 0x5EF9F800U != form)
        || false == has_feature(compiler, BAL_TIER1_FEATURE_SSE41))
    {
        return false;
    }

    // The precision's exponent mask, its lowest exponent bit, and its size.
    const uint32_t size = 0x5EF9F800U == form ? 1U : (0x5EE1F800U == form ? 3U : 2U);
    const uint32_t bits = 8U << size;
    const uint64_t exponent
        = 1U == size ? 0x7C00U : (3U == size ? 0x7FF0000000000000ULL : 0x7F800000U);
    const uint64_t lowest
        = 1U == size ? 0x0400U : (3U == size ? 0x0010000000000000ULL : 0x00800000U);
    const uint32_t                    lane = 3U == size ? 3U : 2U; // The comparison's lane size.
    bal_x86_assembler_t *BAL_RESTRICT assembler = &compiler->assembler;
    size_t                            slow_jumps[2];

    slow_jumps[0] = begin_float_sequence(compiler);
    emit_load_element(assembler, XMM0, (instruction >> 5U) & 0x1FU, size);

    // An exponent of all ones: shifted to the top without the sign, at least the mask there.
    emit_sse(assembler, 0x66, true, 0x7E, XMM0, (uint8_t)SCRATCH_A, "movq");
    bal_x86_emit_shl_r64_imm8(assembler, SCRATCH_A, (uint8_t)(65U - bits));
    bal_x86_emit_mov_r64_imm64(assembler, BAL_X86_RAX, exponent << (65U - bits));
    bal_x86_emit_sub_r64_r64(assembler, SCRATCH_A, BAL_X86_RAX);
    slow_jumps[1] = emit_forward_jump(assembler, true, BAL_X86_COND_AE);

    // XMM1 = the inverted exponent, less the lowest exponent bit where it was zero; XMM0 = the
    // sign.
    emit_load_constant(assembler, XMM3, exponent);
    emit_sse(assembler, 0U, false, 0x28, XMM1, XMM0, "movaps");
    emit_sse(assembler, 0x66, false, 0xDB, XMM1, XMM3, "pand");
    emit_zero_vector(assembler, XMM2);
    emit_lanes_equal(assembler, lane, XMM2, XMM1);
    emit_load_constant(assembler, XMM4, lowest);
    emit_sse(assembler, 0x66, false, 0xDB, XMM2, XMM4, "pand");
    emit_sse(assembler, 0x66, false, 0xEF, XMM1, XMM3, "pxor");
    emit_sse(assembler, 0x66, false, 3U == size ? 0xFB : 0xFA, XMM1, XMM2, "psub");
    emit_load_constant(assembler, XMM3, 1ULL << (bits - 1U));
    emit_sse(assembler, 0x66, false, 0xDB, XMM0, XMM3, "pand");
    emit_sse(assembler, 0x66, false, 0xEB, XMM0, XMM1, "por");
    emit_store_element(assembler, XMM0, instruction & 0x1FU, size);
    emit_slow_path(compiler, instruction, float_helper(instruction), slow_jumps, 2U);
    return true;
}

/// FCVTXN (vector, with the "2" form, and scalar): double to single precision rounded to odd,
/// that is towards zero with the last bit set when inexact, from CVTPD2PS's nearest value as
/// emit_fast_double_to_half() does. Only magnitudes in the normal single precision range run
/// inline; zeros, tiny and huge values, infinities and NaNs take the slow path.
static bool
emit_fast_convert_odd(bal_tier1_compiler_t *BAL_RESTRICT compiler, const uint32_t instruction)
{
    // 0 Q 1 01110 0 1 10000 10110 10 Rn Rd, or the scalar 01 1 11110 0 1 10000 10110 10 Rn Rd.
    const bool is_scalar = 0x7E616800U == (instruction & 0xFFFFFC00U);

    if ((0x2E616800U != (instruction & 0xBFFFFC00U) && false == is_scalar)
        || false == has_feature(compiler, BAL_TIER1_FEATURE_SSE42))
    {
        return false;
    }

    const uint32_t                    rd        = instruction & 0x1FU;
    const uint32_t                    rn        = (instruction >> 5U) & 0x1FU;
    bal_x86_assembler_t *BAL_RESTRICT assembler = &compiler->assembler;
    size_t                            slow_jumps[2];

    slow_jumps[0] = begin_float_sequence(compiler);
    emit_load_vector(assembler, XMM1, rn, false == is_scalar);

    // Exponents from 897 (2^-126) to 1150 (below 2^128): (e - 897) unsigned below 254.
    emit_sse(assembler, 0U, false, 0x28, XMM2, XMM1, "movaps");
    emit_sse_immediate(assembler, 0x66, 0x73, 6U, XMM2, 1U, "psllq");
    emit_sse_immediate(assembler, 0x66, 0x73, 2U, XMM2, 53U, "psrlq");
    emit_broadcast_pattern(assembler, XMM3, 897U);
    emit_sse(assembler, 0x66, false, 0xFB, XMM2, XMM3, "psubq");
    emit_zero_vector(assembler, XMM3);
    emit_lanes_greater(assembler, 3U, XMM3, XMM2);
    emit_broadcast_pattern(assembler, XMM4, 253U);
    emit_lanes_greater(assembler, 3U, XMM2, XMM4);
    emit_sse(assembler, 0x66, false, 0xEB, XMM2, XMM3, "por");
    emit_sse(assembler, 0x66, false, 0x50, (uint8_t)SCRATCH_A, XMM2, "movmskpd");

    if (true == is_scalar)
    {
        bal_x86_emit_shl_r32_imm8(assembler, SCRATCH_A, 31U);
    }

    bal_x86_emit_test_r32_r32(assembler, SCRATCH_A, SCRATCH_A);
    slow_jumps[1] = emit_forward_jump(assembler, true, BAL_X86_COND_NE);

    // XMM0 = the nearest singles; XMM2 = back - n; XMM3 = the roundings away from zero.
    emit_sse(assembler, 0x66, false, 0x5A, XMM0, XMM1, "cvtpd2ps");
    emit_sse(assembler, 0U, false, 0x5A, XMM2, XMM0, "cvtps2pd");
    emit_sse(assembler, 0x66, false, 0x5C, XMM2, XMM1, "subpd");
    emit_sse(assembler, 0U, false, 0x28, XMM3, XMM2, "movaps");
    emit_sse(assembler, 0x66, false, 0x59, XMM3, XMM1, "mulpd");
    emit_zero_vector(assembler, XMM4);
    emit_sse_immediate(assembler, 0x66, 0xC2, XMM3, XMM4, 6U, "cmpnlepd");
    emit_sse_immediate(assembler, 0x66, 0xC2, XMM2, XMM4, 4U, "cmpneqpd");
    emit_sse_immediate(assembler, 0x66, 0x70, XMM3, XMM3, 0x08, "pshufd");
    emit_sse_immediate(assembler, 0x66, 0x70, XMM2, XMM2, 0x08, "pshufd");
    emit_sse(assembler, 0x66, false, 0xFE, XMM0, XMM3, "paddd");
    emit_sse_immediate(assembler, 0x66, 0x72, 2U, XMM2, 31U, "psrld");
    emit_sse(assembler, 0x66, false, 0xEB, XMM0, XMM2, "por");

    if (true == is_scalar)
    {
        emit_sse(assembler, 0x66, false, 0x7E, XMM0, (uint8_t)SCRATCH_A, "movd");
        emit_store_scalar_result(assembler, rd);
    }
    else
    {
        emit_store_narrow(assembler, rd, 1U == ((instruction >> 30U) & 1U));
    }

    emit_slow_path(compiler, instruction, float_helper(instruction), slow_jumps, 2U);
    return true;
}

/// URECPE and URSQRTE (vector) on words: the 9-bit estimate of the top bits, from
/// RECIPROCAL_ESTIMATES or UNSIGNED_SQUARE_ROOT_ESTIMATES lane by lane, above 23 zero bits; all
/// ones for operands below 2^31 (URECPE) or 2^30 (URSQRTE). They touch no flags.
static bool
emit_fast_unsigned_estimate(bal_tier1_compiler_t *BAL_RESTRICT compiler, const uint32_t instruction)
{
    // 0 Q U 01110 1 0 10000 11100 10 Rn Rd.
    if (0x0EA1C800U != (instruction & 0x9FFFFC00U)
        || false == has_feature(compiler, BAL_TIER1_FEATURE_SSE41))
    {
        return false;
    }

    static const uint8_t              PEXTRD[3] = { 0x0F, 0x3A, 0x16 };
    static const uint8_t              PINSRD[3] = { 0x0F, 0x3A, 0x22 };
    const bool                        is_root   = 1U == ((instruction >> 29U) & 1U);
    const bool                        is_quad   = 1U == ((instruction >> 30U) & 1U);
    bal_x86_assembler_t *BAL_RESTRICT assembler = &compiler->assembler;

    bal_sliding_window_flush_all(&compiler->window);
    emit_load_vector(assembler, XMM0, (instruction >> 5U) & 0x1FU, is_quad);
    emit_zero_vector(assembler, XMM1);

    for (uint32_t lane = 0U; lane < (true == is_quad ? 4U : 2U); ++lane)
    {
        bal_x86_emit_sse_register(assembler,
                                  0x66,
                                  false,
                                  PEXTRD,
                                  3U,
                                  XMM0,
                                  (uint8_t)SCRATCH_A,
                                  true,
                                  (uint8_t)lane,
                                  "pextrd");
        bal_x86_emit_shr_r32_imm8(assembler, SCRATCH_A, 23U);

        if (false == is_root)
        {
            bal_x86_emit_mov_r64_imm64(assembler, SCRATCH_B, 0xFFU);
            bal_x86_emit_and_r32_r32(assembler, SCRATCH_A, SCRATCH_B);
        }

        bal_x86_emit_mov_r64_imm64(assembler,
                                   SCRATCH_B,
                                   true == is_root
                                       ? (uint64_t)(uintptr_t)UNSIGNED_SQUARE_ROOT_ESTIMATES
                                       : (uint64_t)(uintptr_t)RECIPROCAL_ESTIMATES);
        bal_x86_emit_add_r64_r64(assembler, SCRATCH_A, SCRATCH_B);
        bal_x86_emit_load_r32_mem8(assembler, SCRATCH_A, SCRATCH_A);
        bal_x86_emit_sse_register(assembler,
                                  0x66,
                                  false,
                                  PINSRD,
                                  3U,
                                  XMM1,
                                  (uint8_t)SCRATCH_A,
                                  true,
                                  (uint8_t)lane,
                                  "pinsrd");
    }

    // XMM1 = 1:estimate:zeros; XMM2 = the operands too small, which give all ones.
    emit_sse_immediate(assembler, 0x66, 0x72, 6U, XMM1, 23U, "pslld");
    emit_broadcast_pattern(assembler, XMM2, 0x8000000080000000ULL);
    emit_sse(assembler, 0x66, false, 0xEB, XMM1, XMM2, "por");

    if (true == is_root)
    {
        // Operands in [0, 2^30): below 2^30 signed and not negative.
        emit_broadcast_pattern(assembler, XMM2, 0x4000000040000000ULL);
        emit_sse(assembler, 0x66, false, 0x66, XMM2, XMM0, "pcmpgtd");
        emit_zero_vector(assembler, XMM3);
        emit_sse(assembler, 0x66, false, 0x66, XMM3, XMM0, "pcmpgtd");
        emit_sse(assembler, 0x66, false, 0xDF, XMM3, XMM2, "pandn");
        emit_sse(assembler, 0x66, false, 0xEB, XMM1, XMM3, "por");
    }
    else
    {
        // Operands with the top bit clear: not below zero signed.
        emit_zero_vector(assembler, XMM2);
        emit_sse(assembler, 0x66, false, 0x66, XMM2, XMM0, "pcmpgtd");
        emit_not_vector(assembler, XMM2, XMM3);
        emit_sse(assembler, 0x66, false, 0xEB, XMM1, XMM2, "por");
    }

    emit_store_vector(assembler, XMM1, instruction & 0x1FU, is_quad);
    return true;
}

/// PMUL (vector) on bytes and PMULL and PMULL2 from bytes to halfwords: carry-less products,
/// the XOR of the first operand shifted by each set bit of the second.
static bool
emit_fast_polynomial_multiply(bal_tier1_compiler_t *BAL_RESTRICT compiler,
                              const uint32_t                     instruction)
{
    // PMUL: 0 Q 1 01110 00 1 Rm 10011 1 Rn Rd. PMULL: 0 Q 0 01110 00 1 Rm 1110 00 Rn Rd.
    const bool is_pmul  = 0x2E209C00U == (instruction & 0xBFE0FC00U);
    const bool is_pmull = 0x0E20E000U == (instruction & 0xBFE0FC00U);

    if ((false == is_pmul && false == is_pmull)
        || false == has_feature(compiler, BAL_TIER1_FEATURE_SSE41))
    {
        return false;
    }

    const bool                        is_quad   = 1U == ((instruction >> 30U) & 1U);
    bal_x86_assembler_t *BAL_RESTRICT assembler = &compiler->assembler;

    bal_sliding_window_flush_all(&compiler->window);
    emit_load_vector(assembler, XMM0, (instruction >> 5U) & 0x1FU, true == is_quad || is_pmull);
    emit_load_vector(assembler, XMM1, (instruction >> 16U) & 0x1FU, true == is_quad || is_pmull);

    if (true == is_pmull)
    {
        emit_widen(assembler, XMM0, 0U, true, is_quad);
        emit_widen(assembler, XMM1, 0U, true, is_quad);
    }

    emit_zero_vector(assembler, XMM2);

    for (uint8_t bit = 0U; bit < 8U; ++bit)
    {
        // XMM3 = all ones where the bit of m is set; XMM4 = n shifted by the bit.
        emit_sse(assembler, 0U, false, 0x28, XMM3, XMM1, "movaps");

        if (true == is_pmull)
        {
            emit_sse_immediate(assembler, 0x66, 0x71, 6U, XMM3, (uint8_t)(15U - bit), "psllw");
            emit_sse_immediate(assembler, 0x66, 0x71, 4U, XMM3, 15U, "psraw");
        }
        else
        {
            if (bit != 7U)
            {
                emit_sse_immediate(assembler, 0x66, 0x71, 6U, XMM3, (uint8_t)(7U - bit), "psllw");
            }

            emit_zero_vector(assembler, XMM4);
            emit_sse(assembler, 0x66, false, 0x64, XMM4, XMM3, "pcmpgtb");
            emit_sse(assembler, 0U, false, 0x28, XMM3, XMM4, "movaps");
        }

        emit_sse(assembler, 0U, false, 0x28, XMM4, XMM0, "movaps");

        if (true == is_pmull)
        {
            emit_lanes_shift_left(assembler, XMM4, 1U, bit, XMM5);
        }
        else
        {
            emit_lanes_shift_left(assembler, XMM4, 0U, bit, XMM5);
        }

        emit_sse(assembler, 0x66, false, 0xDB, XMM4, XMM3, "pand");
        emit_sse(assembler, 0x66, false, 0xEF, XMM2, XMM4, "pxor");
    }

    emit_store_vector(assembler, XMM2, instruction & 0x1FU, true == is_quad || true == is_pmull);
    return true;
}

/// Converts the double in lane 0 of xmm0, scaled by 2^`fbits`, to a 64-bit integer in
/// `destination` as emit_fast_float_to_integer() does (`mode` 0-3 the rounding modes, 4 ties
/// away), adding the jumps to the slow path to `jumps` and returning their number. Uses xmm1 to
/// xmm3, SCRATCH_A, SCRATCH_B and RAX.
static uint32_t
emit_double_to_integer_lane(bal_x86_assembler_t *BAL_RESTRICT assembler,
                            const uint32_t                    mode,
                            const bool                        is_unsigned,
                            const uint32_t                    fbits,
                            const bal_x86_register_t          destination,
                            size_t                           *jumps)
{
    const bool is_direct = 3U == mode && false == is_unsigned;
    uint32_t   count     = 0U;

    // Unsigned values of 2^63 and more would make the signed conversion raise Invalid, and
    // magnitudes of 2^64 and more saturate (and would overflow the scaling).
    jumps[count++] = emit_register_magnitude_check(
        assembler, XMM0, true, true == is_unsigned ? 1023U + 63U - fbits : 1023U + 64U - fbits);

    if (fbits != 0U)
    {
        emit_load_constant(assembler, XMM1, (uint64_t)(1023U + fbits) << 52U);
        emit_sse(assembler, 0xF2, false, 0x59, XMM0, XMM1, "mulsd");
    }

    uint8_t source = XMM0;

    if (false == is_direct)
    {
        if (4U == mode)
        {
            emit_round_ties_away(assembler, true);
        }
        else
        {
            emit_round_scalar(assembler, true, XMM1, XMM0, ROUND_SUPPRESSED[mode]);
        }

        source = XMM1;
    }

    emit_sse(assembler, 0xF2, true, 0x2C, (uint8_t)destination, source, "cvttsd2si");

    if (true == is_unsigned)
    {
        bal_x86_emit_test_r64_r64(assembler, destination, destination);
        jumps[count++] = emit_forward_jump(assembler, true, BAL_X86_COND_S);
    }
    else
    {
        bal_x86_emit_mov_r64_imm64(assembler, BAL_X86_RAX, 0x8000000000000000ULL);
        bal_x86_emit_mov_r64_r64(assembler, SCRATCH_B, destination);
        bal_x86_emit_sub_r64_r64(assembler, SCRATCH_B, BAL_X86_RAX);
        jumps[count++] = emit_forward_jump(assembler, true, BAL_X86_COND_E);
    }

    if (false == is_direct)
    {
        emit_round_scalar(assembler, true, XMM2, XMM0, 0x03);
    }

    return count;
}

/// FCVT{N,P,M,Z,A}{S,U}, SCVTF and UCVTF (vector) on doublewords and on the double precision
/// scalar, lane by lane through general purpose registers as the scalar conversions to and from
/// X registers; and FCVTZS, FCVTZU, SCVTF and UCVTF (vector and scalar, fixed-point) on words,
/// which scale by a power of two exactly around the packed conversions, and on doublewords and
/// the double precision scalar, lane by lane.
static bool
emit_fast_convert_lanes(bal_tier1_compiler_t *BAL_RESTRICT compiler, const uint32_t instruction)
{
    // Two registers: 0 Q U 01110 a 1 10000 opcode 10 Rn Rd, or the scalar 01 U 11110 ... Fixed:
    // 0 Q U 011110 immh immb 111 oo 1 Rn Rd (oo = 00 SCVTF and UCVTF, 11 FCVTZS and FCVTZU) or
    // the scalar 01 U 111110 ...
    const bool     is_two_vector = 0x0E600800U == (instruction & 0x9F7E0C00U);
    const bool     is_two_scalar = 0x5E600800U == (instruction & 0xDF7E0C00U);
    const uint32_t fix_opcode    = (instruction >> 11U) & 0x3U; // 00 or 11 in 111xx.
    const bool     is_fix_opcode = 0U == fix_opcode || 3U == fix_opcode;
    const bool     is_fix_vector = 0x0F00E400U == (instruction & 0x9F80E400U) && is_fix_opcode;
    const bool     is_fix_scalar = 0x5F00E400U == (instruction & 0xDF80E400U) && is_fix_opcode;
    const bool     is_unsigned   = 1U == ((instruction >> 29U) & 1U);
    const uint32_t immh          = (instruction >> 19U) & 0xFU;
    const uint32_t rd            = instruction & 0x1FU;
    const uint32_t rn            = (instruction >> 5U) & 0x1FU;

    if (false == has_feature(compiler, BAL_TIER1_FEATURE_SSE41))
    {
        return false;
    }

    bool     is_scalar = false;
    bool     is_double = true;
    bool     is_to_int = true;
    uint32_t mode      = 3U;
    uint32_t fbits     = 0U;

    if (true == is_two_vector || true == is_two_scalar)
    {
        const uint32_t key = (((instruction >> 29U) & 1U) << 6U)
                             | (((instruction >> 23U) & 1U) << 5U) | ((instruction >> 12U) & 0x1FU);
        is_scalar = is_two_scalar;

        switch (key & 0x3FU)
        {
            case 0x1AU: // FCVTNS, FCVTNU
                mode = 0U;
                break;
            case 0x3AU: // FCVTPS, FCVTPU
                mode = 1U;
                break;
            case 0x1BU: // FCVTMS, FCVTMU
                mode = 2U;
                break;
            case 0x3BU: // FCVTZS, FCVTZU
                mode = 3U;
                break;
            case 0x1CU: // FCVTAS, FCVTAU
                mode = 4U;
                break;
            case 0x1DU: // SCVTF, UCVTF
                is_to_int = false;
                break;
            default:
                return false;
        }
    }
    else if (true == is_fix_vector || true == is_fix_scalar)
    {
        // immh 1xxx is doublewords and 01xx words; half precision stays with the helper.
        if (immh < 4U)
        {
            return false;
        }

        is_scalar = is_fix_scalar;
        is_double = 0U != (immh & 8U);
        is_to_int = 1U == ((instruction >> 11U) & 1U);
        fbits     = (true == is_double ? 128U : 64U) - ((instruction >> 16U) & 0x7FU);
    }
    else
    {
        return false;
    }

    const bool is_quad = false == is_scalar && 1U == ((instruction >> 30U) & 1U);

    if (true == is_double && false == is_quad && false == is_scalar)
    {
        return false;
    }

    bal_x86_assembler_t *BAL_RESTRICT assembler = &compiler->assembler;
    size_t                            slow_jumps[7];
    uint32_t                          count = 1U;

    slow_jumps[0] = begin_float_sequence(compiler);

    if (false == is_double)
    {
        // Words: XMM0 = the operands; the result goes to XMM1.
        if (true == is_scalar)
        {
            emit_load_scalar(assembler, XMM0, rn, false);
        }
        else
        {
            emit_load_vector(assembler, XMM0, rn, is_quad);
        }

        if (false == is_to_int)
        {
            if (true == is_unsigned)
            {
                // (float)(x >> 16) * 65536 + (float)(x & 0xFFFF), as UCVTF (vector) does.
                emit_sse(assembler, 0U, false, 0x28, XMM1, XMM0, "movaps");
                emit_sse_immediate(assembler, 0x66, 0x72, 2U, XMM1, 16U, "psrld");
                emit_sse(assembler, 0U, false, 0x5B, XMM1, XMM1, "cvtdq2ps");
                emit_broadcast_pattern(assembler, XMM2, 0x4780000047800000ULL);
                emit_sse(assembler, 0U, false, 0x59, XMM1, XMM2, "mulps");
                emit_sse_immediate(assembler, 0x66, 0x72, 6U, XMM0, 16U, "pslld");
                emit_sse_immediate(assembler, 0x66, 0x72, 2U, XMM0, 16U, "psrld");
                emit_sse(assembler, 0U, false, 0x5B, XMM0, XMM0, "cvtdq2ps");
                emit_sse(assembler, 0U, false, 0x58, XMM1, XMM0, "addps");
            }
            else
            {
                emit_sse(assembler, 0U, false, 0x5B, XMM1, XMM0, "cvtdq2ps");
            }

            // The integers are zero or at least 1, so scaling them is exact.
            emit_broadcast_pattern(
                assembler, XMM2, 0x0000000100000001ULL * ((127U - fbits) << 23U));
            emit_sse(assembler, 0U, false, 0x59, XMM1, XMM2, "mulps");
        }
        else
        {
            // Exponents of 159 (2^32) and more saturate, or for FCVTZU 158 and more may not fit
            // the signed conversion; checking them first keeps the scaling from overflowing.
            emit_sse(assembler, 0U, false, 0x28, XMM2, XMM0, "movaps");
            emit_sse_immediate(assembler, 0x66, 0x72, 6U, XMM2, 1U, "pslld");
            emit_sse_immediate(assembler, 0x66, 0x72, 2U, XMM2, 24U, "psrld");
            emit_broadcast_pattern(assembler,
                                   XMM3,
                                   0x0000000100000001ULL
                                       * ((true == is_unsigned ? 157U : 158U) - fbits));
            emit_sse(assembler, 0x66, false, 0x66, XMM2, XMM3, "pcmpgtd");
            emit_sse(assembler, 0U, false, 0x50, (uint8_t)SCRATCH_A, XMM2, "movmskps");
            bal_x86_emit_test_r32_r32(assembler, SCRATCH_A, SCRATCH_A);
            slow_jumps[count++] = emit_forward_jump(assembler, true, BAL_X86_COND_NE);
            emit_broadcast_pattern(
                assembler, XMM2, 0x0000000100000001ULL * ((127U + fbits) << 23U));
            emit_sse(assembler, 0U, false, 0x59, XMM0, XMM2, "mulps");

            // Truncate with the precision exception suppressed, convert exactly; an integer out
            // of range (the indefinite value, or a negative one for FCVTZU) takes the slow path,
            // and only then does Inexact come.
            emit_sse3a(assembler, 0x08, XMM1, XMM0, 0x0B, "roundps");
            emit_sse(assembler, 0xF3, false, 0x5B, XMM1, XMM1, "cvttps2dq");

            if (true == is_unsigned)
            {
                emit_sse(assembler, 0U, false, 0x50, (uint8_t)SCRATCH_A, XMM1, "movmskps");
            }
            else
            {
                emit_all_ones(assembler, XMM2);
                emit_sse_immediate(assembler, 0x66, 0x72, 6U, XMM2, 31U, "pslld");
                emit_sse(assembler, 0x66, false, 0x76, XMM2, XMM1, "pcmpeqd");
                emit_sse(assembler, 0U, false, 0x50, (uint8_t)SCRATCH_A, XMM2, "movmskps");
            }

            bal_x86_emit_test_r32_r32(assembler, SCRATCH_A, SCRATCH_A);
            slow_jumps[count++] = emit_forward_jump(assembler, true, BAL_X86_COND_NE);
            emit_sse3a(assembler, 0x08, XMM2, XMM0, 0x03, "roundps");
        }

        if (true == is_scalar)
        {
            emit_sse(assembler, 0x66, false, 0x7E, XMM1, (uint8_t)SCRATCH_A, "movd");
            emit_store_scalar_result(assembler, rd);
        }
        else
        {
            emit_store_vector(assembler, XMM1, rd, is_quad);
        }

        emit_slow_path(compiler, instruction, float_helper(instruction), slow_jumps, count);
        return true;
    }

    // Doublewords, lane by lane: the results go to RCX and RDX.
    const uint32_t lanes = true == is_scalar ? 1U : 2U;

    for (uint32_t lane = 0U; lane < lanes; ++lane)
    {
        const bal_x86_register_t result = 0U == lane ? BAL_X86_RCX : BAL_X86_RDX;
        bal_x86_emit_load_r64_rbp_offset(
            assembler, SCRATCH_A, (int32_t)offsetof(bal_cpu_t, v[rn][lane]));

        if (true == is_to_int)
        {
            emit_sse(assembler, 0x66, true, 0x6E, XMM0, (uint8_t)SCRATCH_A, "movq");
            count += emit_double_to_integer_lane(
                assembler, mode, is_unsigned, fbits, result, &slow_jumps[count]);
            continue;
        }

        if (true == is_unsigned)
        {
            bal_x86_emit_test_r64_r64(assembler, SCRATCH_A, SCRATCH_A);
            slow_jumps[count++] = emit_forward_jump(assembler, true, BAL_X86_COND_S);
        }

        // CVTSI2SD rounds once; scaling the result, zero or at least 1, is exact.
        emit_sse(assembler, 0U, false, 0x57, XMM0, XMM0, "xorps");
        emit_sse(assembler, 0xF2, true, 0x2A, XMM0, (uint8_t)SCRATCH_A, "cvtsi2sd");

        if (fbits != 0U)
        {
            emit_load_constant(assembler, XMM1, (uint64_t)(1023U - fbits) << 52U);
            emit_sse(assembler, 0xF2, false, 0x59, XMM0, XMM1, "mulsd");
        }

        emit_sse(assembler, 0x66, true, 0x7E, XMM0, (uint8_t)result, "movq");
    }

    emit_sse(assembler, 0x66, true, 0x6E, XMM0, (uint8_t)BAL_X86_RCX, "movq");

    if (true == is_scalar)
    {
        emit_store_vector(assembler, XMM0, rd, false);
    }
    else
    {
        emit_sse(assembler, 0x66, true, 0x6E, XMM1, (uint8_t)BAL_X86_RDX, "movq");
        emit_sse(assembler, 0x66, false, 0x6C, XMM0, XMM1, "punpcklqdq");
        emit_store_vector(assembler, XMM0, rd, true);
    }

    emit_slow_path(compiler, instruction, float_helper(instruction), slow_jumps, count);
    return true;
}

/// Sets xmm0 to xmm0 + xmm1, or xmm0 - xmm1 with `is_subtract`, with signed saturation on words
/// (`size` 2) or doublewords, setting FPSR.QC when a lane saturates. Uses xmm2 to xmm5.
static void
emit_signed_saturating_add(bal_x86_assembler_t *BAL_RESTRICT assembler,
                           const uint32_t                    size,
                           const bool                        is_subtract)
{
    // XMM2 = the wrapping result; XMM3 = the lanes that overflowed (from the signs, as
    // emit_fast_saturating_arithmetic() finds them); XMM4 = the bound each crossed.
    emit_sse(assembler, 0U, false, 0x28, XMM2, XMM0, "movaps");
    emit_sse(assembler,
             0x66,
             false,
             true == is_subtract ? LANE_SUB[size] : LANE_ADD[size],
             XMM2,
             XMM1,
             "padd/psub");
    emit_sse(assembler, 0U, false, 0x28, XMM3, XMM0, "movaps");
    emit_sse(assembler, 0x66, false, 0xEF, XMM3, XMM2, "pxor");
    emit_sse(assembler, 0U, false, 0x28, XMM4, true == is_subtract ? XMM0 : XMM1, "movaps");
    emit_sse(assembler, 0x66, false, 0xEF, XMM4, true == is_subtract ? XMM1 : XMM2, "pxor");
    emit_sse(assembler, 0x66, false, 0xDB, XMM4, XMM3, "pand");
    emit_zero_vector(assembler, XMM3);
    emit_lanes_greater(assembler, size, XMM3, XMM4);
    emit_saturation_if_any(assembler, XMM3);
    emit_zero_vector(assembler, XMM4);
    emit_lanes_greater(assembler, size, XMM4, XMM0);
    emit_broadcast_pattern(
        assembler, XMM5, 3U == size ? 0x7FFFFFFFFFFFFFFFULL : 0x7FFFFFFF7FFFFFFFULL);
    emit_sse(assembler, 0x66, false, 0xEF, XMM4, XMM5, "pxor");
    emit_sse(assembler, 0x66, false, 0xDB, XMM4, XMM3, "pand");
    emit_sse(assembler, 0x66, false, 0xDF, XMM3, XMM2, "pandn");
    emit_sse(assembler, 0x66, false, 0xEB, XMM4, XMM3, "por");
    emit_sse(assembler, 0U, false, 0x28, XMM0, XMM4, "movaps");
}

/// Loads the second operand of a by element (`is_element`) or register form into xmm1: element
/// `index` of `rm` broadcast, or `rm` itself (its element 0 with `is_scalar`).
static void
emit_load_second_operand(bal_x86_assembler_t *BAL_RESTRICT assembler,
                         const uint32_t                    instruction,
                         const uint32_t                    size,
                         const bool                        is_element,
                         const bool                        is_scalar)
{
    if (true == is_element)
    {
        const uint32_t h     = (instruction >> 11U) & 1U;
        const uint32_t l     = (instruction >> 21U) & 1U;
        const uint32_t m     = (instruction >> 20U) & 1U;
        const uint32_t index = 1U == size ? (h << 2U) | (l << 1U) | m : (h << 1U) | l;
        const uint32_t rm = 1U == size ? (instruction >> 16U) & 0xFU : (instruction >> 16U) & 0x1FU;
        emit_load_vector(assembler, XMM1, rm, true);
        emit_broadcast_element(assembler, XMM1, size, index);
        return;
    }

    if (true == is_scalar)
    {
        emit_load_element(assembler, XMM1, (instruction >> 16U) & 0x1FU, size);
        return;
    }

    emit_load_vector(assembler, XMM1, (instruction >> 16U) & 0x1FU, true);
}

/// SQDMULL, SQDMLAL and SQDMLSL (vector with the "2" forms, scalar, and by element) on halfwords
/// and words: the products, widened and doubled, overflow only for the minimum times itself,
/// which saturates; the accumulating forms then add to or subtract from Rd with saturation.
/// FPSR.QC is set when a lane saturates.
static bool
emit_fast_doubling_long(bal_tier1_compiler_t *BAL_RESTRICT compiler, const uint32_t instruction)
{
    // Three different: 0 Q 0 01110 size 1 Rm opcode 00 Rn Rd with opcode 1101 (SQDMULL), 1001
    // (SQDMLAL) or 1011 (SQDMLSL), or the scalar 01 0 11110 ... By element: 0 Q 0 01111 size L M
    // Rm opcode H 0 Rn Rd with 1011, 0011 or 0111, or the scalar 01 0 11111 ...
    const uint32_t size       = (instruction >> 22U) & 0x3U;
    const uint32_t three      = (instruction >> 12U) & 0xFU;
    const bool     is_vector  = 0x0E200000U == (instruction & 0xBF200C00U);
    const bool     is_scalar3 = 0x5E200000U == (instruction & 0xFF200C00U);
    const bool     is_element = 0x0F000000U == (instruction & 0xBF000400U);
    const bool     is_scalarE = 0x5F000000U == (instruction & 0xFF000400U);
    uint32_t       operation  = 0xFFU; // 0 multiply, 1 add, 2 subtract.

    if (true == is_vector || true == is_scalar3)
    {
        operation = 0xDU == three ? 0U : (0x9U == three ? 1U : (0xBU == three ? 2U : 0xFFU));
    }
    else if (true == is_element || true == is_scalarE)
    {
        operation = 0xBU == three ? 0U : (0x3U == three ? 1U : (0x7U == three ? 2U : 0xFFU));
    }

    if (0xFFU == operation || (size != 1U && size != 2U)
        || false == has_feature(compiler, BAL_TIER1_FEATURE_SSE42))
    {
        return false;
    }

    const bool     is_scalar = true == is_scalar3 || true == is_scalarE;
    const bool     is_upper  = false == is_scalar && 1U == ((instruction >> 30U) & 1U);
    const uint32_t rd        = instruction & 0x1FU;
    const uint32_t rn        = (instruction >> 5U) & 0x1FU;
    bal_x86_assembler_t *BAL_RESTRICT assembler = &compiler->assembler;

    bal_sliding_window_flush_all(&compiler->window);

    if (true == is_scalar)
    {
        emit_load_element(assembler, XMM0, rn, size);
    }
    else
    {
        emit_load_vector(assembler, XMM0, rn, true);
    }

    emit_load_second_operand(
        assembler, instruction, size, true == is_element || true == is_scalarE, is_scalar);

    if (true == is_upper)
    {
        emit_sse_immediate(assembler, 0x66, 0x73, 3U, XMM0, 8U, "psrldq");

        if (false == is_element)
        {
            emit_sse_immediate(assembler, 0x66, 0x73, 3U, XMM1, 8U, "psrldq");
        }
    }

    // XMM0 = 2 * n * m in the wide lanes.
    emit_widen(assembler, XMM0, size, false, false);
    emit_widen(assembler, XMM1, size, false, false);

    if (1U == size)
    {
        emit_sse38(assembler, 0x40, XMM0, XMM1, "pmulld");
    }
    else
    {
        emit_sse38(assembler, 0x28, XMM0, XMM1, "pmuldq");
    }

    emit_sse(assembler, 0x66, false, LANE_ADD[size + 1U], XMM0, XMM0, "padd");
    emit_broadcast_pattern(assembler, XMM2, lane_sign_pattern(size + 1U));
    emit_lanes_equal(assembler, size + 1U, XMM2, XMM0);
    emit_saturation_if_any(assembler, XMM2);
    emit_sse(assembler, 0x66, false, 0xEF, XMM0, XMM2, "pxor");

    if (operation != 0U)
    {
        emit_sse(assembler, 0U, false, 0x28, XMM1, XMM0, "movaps");

        if (true == is_scalar)
        {
            emit_load_element(assembler, XMM0, rd, size + 1U);
        }
        else
        {
            emit_load_vector(assembler, XMM0, rd, true);
        }

        emit_signed_saturating_add(assembler, size + 1U, 2U == operation);
    }

    if (true == is_scalar)
    {
        emit_store_element(assembler, XMM0, rd, size + 1U);
    }
    else
    {
        emit_store_vector(assembler, XMM0, rd, true);
    }

    return true;
}

/// SQRDMLAH and SQRDMLSH (vector, scalar and by element) on halfwords and words: Rd plus (or
/// minus) the rounded doubled high half of n * m, saturated once. In lanes of twice the size,
/// (2nm + 2^(e-1)) >> e is (nm + 2^(e-2)) >> (e-1) exactly, and adding Rd cannot overflow; the
/// sum then narrows with saturation, setting FPSR.QC when a lane saturates. Each group of 64
/// result bits goes to RAX or RDX until the end.
static bool
emit_fast_rounding_doubling_accumulate(bal_tier1_compiler_t *BAL_RESTRICT compiler,
                                       const uint32_t                     instruction)
{
    // Vector: 0 Q 1 01110 size 0 Rm 1000 S 1 Rn Rd, or the scalar 01 1 11110 ... By element:
    // 0 Q 1 01111 size L M Rm 11 S 1 H 0 Rn Rd, or the scalar 01 1 11111 ...
    const uint32_t size       = (instruction >> 22U) & 0x3U;
    const bool     is_vector  = 0x2E008400U == (instruction & 0xBF20F400U);
    const bool     is_scalar3 = 0x7E008400U == (instruction & 0xFF20F400U);
    const bool     is_element = 0x2F00D000U == (instruction & 0xBF00D400U);
    const bool     is_scalarE = 0x7F00D000U == (instruction & 0xFF00D400U);

    if ((false == is_vector && false == is_scalar3 && false == is_element && false == is_scalarE)
        || (size != 1U && size != 2U) || false == has_feature(compiler, BAL_TIER1_FEATURE_SSE42))
    {
        return false;
    }

    const bool     is_subtract = true == is_vector || true == is_scalar3
                                     ? 1U == ((instruction >> 11U) & 1U)
                                     : 1U == ((instruction >> 13U) & 1U);
    const bool     is_scalar   = true == is_scalar3 || true == is_scalarE;
    const bool     is_quad     = false == is_scalar && 1U == ((instruction >> 30U) & 1U);
    const bool     is_lane     = true == is_element || true == is_scalarE;
    const uint32_t rd          = instruction & 0x1FU;
    const uint32_t rn          = (instruction >> 5U) & 0x1FU;
    bal_x86_assembler_t *BAL_RESTRICT assembler = &compiler->assembler;

    // The register cache is emptied, so that RAX and RDX are free.
    begin_inline_sequence(compiler);

    for (uint32_t group = 0U; group < (true == is_quad ? 2U : 1U); ++group)
    {
        // XMM0 = n, XMM1 = m and XMM2 = Rd, widened.
        if (true == is_scalar)
        {
            emit_load_element(assembler, XMM0, rn, size);
            emit_load_element(assembler, XMM2, rd, size);
        }
        else
        {
            emit_load_vector(assembler, XMM0, rn, true);
            emit_load_vector(assembler, XMM2, rd, true);
        }

        emit_load_second_operand(assembler, instruction, size, is_lane, is_scalar);

        if (1U == group)
        {
            emit_sse_immediate(assembler, 0x66, 0x73, 3U, XMM0, 8U, "psrldq");
            emit_sse_immediate(assembler, 0x66, 0x73, 3U, XMM1, 8U, "psrldq");
            emit_sse_immediate(assembler, 0x66, 0x73, 3U, XMM2, 8U, "psrldq");
        }

        emit_widen(assembler, XMM0, size, false, false);
        emit_widen(assembler, XMM1, size, false, false);
        emit_widen(assembler, XMM2, size, false, false);
        emit_sse38(assembler, 1U == size ? 0x40 : 0x28, XMM0, XMM1, "pmulld/pmuldq");

        if (true == is_subtract)
        {
            emit_zero_vector(assembler, XMM3);
            emit_sse(assembler, 0x66, false, LANE_SUB[size + 1U], XMM3, XMM0, "psub");
            emit_sse(assembler, 0U, false, 0x28, XMM0, XMM3, "movaps");
        }

        emit_broadcast_pattern(
            assembler, XMM3, 1U == size ? 0x0000400000004000ULL : 0x0000000040000000ULL);
        emit_sse(assembler, 0x66, false, LANE_ADD[size + 1U], XMM0, XMM3, "padd");
        emit_lanes_shift_arithmetic(assembler, XMM0, size + 1U, 1U == size ? 15U : 31U, XMM3);
        emit_sse(assembler, 0x66, false, LANE_ADD[size + 1U], XMM0, XMM2, "padd");

        if (1U == size)
        {
            emit_sse(assembler, 0U, false, 0x28, XMM1, XMM0, "movaps");
            emit_sse(assembler, 0x66, false, 0x6B, XMM0, XMM0, "packssdw");
            emit_sse(assembler, 0U, false, 0x28, XMM3, XMM0, "movaps");
            emit_widen(assembler, XMM3, 1U, false, false);
            emit_saturation_unless_equal(assembler, XMM3, XMM1);
        }
        else
        {
            // Clamp the doublewords to the word range, then keep their low halves.
            emit_broadcast_pattern(assembler, XMM4, 0x000000007FFFFFFFULL);
            emit_sse(assembler, 0U, false, 0x28, XMM3, XMM0, "movaps");
            emit_lanes_greater(assembler, 3U, XMM3, XMM4);
            emit_sse(assembler, 0x66, false, 0xDB, XMM4, XMM3, "pand");
            emit_sse(assembler, 0U, false, 0x28, XMM5, XMM3, "movaps");
            emit_sse(assembler, 0x66, false, 0xDF, XMM3, XMM0, "pandn");
            emit_sse(assembler, 0x66, false, 0xEB, XMM3, XMM4, "por");
            emit_broadcast_pattern(assembler, XMM4, 0xFFFFFFFF80000000ULL);
            emit_sse(assembler, 0U, false, 0x28, XMM0, XMM4, "movaps");
            emit_lanes_greater(assembler, 3U, XMM0, XMM3);
            emit_sse(assembler, 0x66, false, 0xEB, XMM5, XMM0, "por");
            emit_sse(assembler, 0x66, false, 0xDB, XMM4, XMM0, "pand");
            emit_sse(assembler, 0x66, false, 0xDF, XMM0, XMM3, "pandn");
            emit_sse(assembler, 0x66, false, 0xEB, XMM0, XMM4, "por");
            emit_saturation_if_any(assembler, XMM5);
            emit_sse_immediate(assembler, 0x66, 0x70, XMM0, XMM0, 0x08, "pshufd");
        }

        emit_sse(assembler,
                 0x66,
                 true,
                 0x7E,
                 XMM0,
                 (uint8_t)(0U == group ? BAL_X86_RAX : BAL_X86_RDX),
                 "movq");
    }

    emit_sse(assembler, 0x66, true, 0x6E, XMM0, (uint8_t)BAL_X86_RAX, "movq");

    if (true == is_scalar)
    {
        emit_store_element(assembler, XMM0, rd, size);
        return true;
    }

    if (true == is_quad)
    {
        emit_sse(assembler, 0x66, true, 0x6E, XMM1, (uint8_t)BAL_X86_RDX, "movq");
        emit_sse(assembler, 0x66, false, 0x6C, XMM0, XMM1, "punpcklqdq");
    }

    emit_store_vector(assembler, XMM0, rd, is_quad);
    return true;
}

/// FMOV (scalar, immediate) in half, single and double precision: the interpreter expands the
/// immediate at translation time, and the code writes the constant.
static bool
emit_fast_move_immediate(bal_tier1_compiler_t *BAL_RESTRICT compiler, const uint32_t instruction)
{
    // M 0 S 11110 ftype 1 imm8 100 00000 Rd.
    const uint32_t ftype = (instruction >> 22U) & 0x3U;

    if (0x1E201000U != (instruction & 0xFF201FE0U) || 2U == ftype)
    {
        return false;
    }

    bal_cpu_t scratch;
    (void)memset(&scratch, 0, sizeof(scratch));
    (void)bal_execute_float(&scratch, instruction);
    const uint32_t        rd          = instruction & 0x1FU;
    const bal_x86_macro_t value_macro = {
        .opcode              = BAL_X86_MACRO_MOV_REGISTER_IMMEDIATE,
        .destination         = SCRATCH_A,
        .immediate_or_offset = scratch.v[rd][0],
    };
    bal_sliding_window_push(&compiler->window, value_macro);
    push_scalar_write(compiler, rd);
    return true;
}

/// FRECPE and FRSQRTE (vector and scalar) in double precision, lane by lane through general
/// purpose registers: the estimate from RECIPROCAL_ESTIMATES or
/// RECIPROCAL_SQUARE_ROOT_ESTIMATES by the top eight fraction bits (and the low exponent bit),
/// with the exponent 2045 - e or (3068 - e) / 2. Exponents the tables do not cover (zeros and
/// denormals, results that would be denormal, infinities and NaNs, and negative FRSQRTE
/// operands) take the slow path.
static bool
emit_fast_double_estimate(bal_tier1_compiler_t *BAL_RESTRICT compiler, const uint32_t instruction)
{
    // 0 Q U 01110 1 1 10000 11101 10 Rn Rd, or the scalar 01 U 11110 ...
    const bool is_scalar = 0x5EE1D800U == (instruction & 0xDFFFFC00U);

    if (0x4EE1D800U != (instruction & 0xDFFFFC00U) && false == is_scalar)
    {
        return false;
    }

    const bool                        is_root   = 1U == ((instruction >> 29U) & 1U);
    const uint32_t                    rd        = instruction & 0x1FU;
    const uint32_t                    rn        = (instruction >> 5U) & 0x1FU;
    bal_x86_assembler_t *BAL_RESTRICT assembler = &compiler->assembler;
    size_t                            slow_jumps[3];
    uint32_t                          count = 1U;

    // After begin_float_sequence() every general purpose register but RBP is free.
    slow_jumps[0] = begin_float_sequence(compiler);

    for (uint32_t lane = 0U; lane < (true == is_scalar ? 1U : 2U); ++lane)
    {
        const bal_x86_register_t result = 0U == lane ? BAL_X86_RCX : BAL_X86_RDX;

        // R8 = x; RAX = its exponent field with the sign (for FRSQRTE) or without it.
        bal_x86_emit_load_r64_rbp_offset(
            assembler, BAL_X86_R8, (int32_t)offsetof(bal_cpu_t, v[rn][lane]));
        bal_x86_emit_mov_r64_r64(assembler, BAL_X86_RAX, BAL_X86_R8);

        if (false == is_root)
        {
            bal_x86_emit_shl_r64_imm8(assembler, BAL_X86_RAX, 1U);
            bal_x86_emit_shr_r64_imm8(assembler, BAL_X86_RAX, 53U);
        }
        else
        {
            bal_x86_emit_shr_r64_imm8(assembler, BAL_X86_RAX, 52U);
        }

        // In range when the exponent less one is below 2044 (FRSQRTE: 2046, unsigned with the
        // sign above).
        bal_x86_emit_mov_r64_r64(assembler, SCRATCH_A, BAL_X86_RAX);
        bal_x86_emit_mov_r64_imm64(assembler, SCRATCH_B, 1U);
        bal_x86_emit_sub_r64_r64(assembler, SCRATCH_A, SCRATCH_B);
        bal_x86_emit_mov_r64_imm64(assembler, SCRATCH_B, true == is_root ? 2046U : 2044U);
        bal_x86_emit_sub_r64_r64(assembler, SCRATCH_A, SCRATCH_B);
        slow_jumps[count++] = emit_forward_jump(assembler, true, BAL_X86_COND_AE);

        // SCRATCH_A = the estimate.
        bal_x86_emit_mov_r64_r64(assembler, SCRATCH_A, BAL_X86_R8);
        bal_x86_emit_shr_r64_imm8(assembler, SCRATCH_A, 44U);
        bal_x86_emit_mov_r64_imm64(assembler, SCRATCH_B, true == is_root ? 0x1FFU : 0xFFU);
        bal_x86_emit_and_r64_r64(assembler, SCRATCH_A, SCRATCH_B);
        bal_x86_emit_mov_r64_imm64(assembler,
                                   SCRATCH_B,
                                   true == is_root
                                       ? (uint64_t)(uintptr_t)RECIPROCAL_SQUARE_ROOT_ESTIMATES
                                       : (uint64_t)(uintptr_t)RECIPROCAL_ESTIMATES);
        bal_x86_emit_add_r64_r64(assembler, SCRATCH_A, SCRATCH_B);
        bal_x86_emit_load_r32_mem8(assembler, SCRATCH_A, SCRATCH_A);
        bal_x86_emit_shl_r64_imm8(assembler, SCRATCH_A, 44U);

        // result = sign | (2045 - e) << 52 | estimate << 44, or ((3068 - e) / 2) << 52 | ...
        bal_x86_emit_mov_r64_imm64(assembler, result, true == is_root ? 3068U : 2045U);
        bal_x86_emit_sub_r64_r64(assembler, result, BAL_X86_RAX);

        if (true == is_root)
        {
            bal_x86_emit_shr_r64_imm8(assembler, result, 1U);
        }

        bal_x86_emit_shl_r64_imm8(assembler, result, 52U);
        bal_x86_emit_or_r64_r64(assembler, result, SCRATCH_A);

        if (false == is_root)
        {
            bal_x86_emit_shr_r64_imm8(assembler, BAL_X86_R8, 63U);
            bal_x86_emit_shl_r64_imm8(assembler, BAL_X86_R8, 63U);
            bal_x86_emit_or_r64_r64(assembler, result, BAL_X86_R8);
        }
    }

    bal_x86_emit_store_r64_rbp_offset(
        assembler, BAL_X86_RCX, (int32_t)offsetof(bal_cpu_t, v[rd][0]));

    if (true == is_scalar)
    {
        bal_x86_emit_xor_r32_r32(assembler, BAL_X86_RDX, BAL_X86_RDX);
    }

    bal_x86_emit_store_r64_rbp_offset(
        assembler, BAL_X86_RDX, (int32_t)offsetof(bal_cpu_t, v[rd][1]));
    emit_slow_path(compiler, instruction, float_helper(instruction), slow_jumps, count);
    return true;
}

/// SCVTF, UCVTF, FCVTZS and FCVTZU (fixed-point, vector and scalar) on halfwords: halves and
/// halfwords convert to single precision exactly, where scaling by a power of two is exact; the
/// conversion to half precision then rounds once, and the conversion to halfwords truncates
/// exactly and narrows, a lane out of range taking the slow path as in emit_fast_half_vector().
static bool
emit_fast_half_fixed(bal_tier1_compiler_t *BAL_RESTRICT compiler, const uint32_t instruction)
{
    // 0 Q U 011110 001x immb 111 oo 1 Rn Rd (oo = 00 SCVTF and UCVTF, 11 FCVTZS and FCVTZU), or
    // the scalar 01 U 111110 ...
    const uint32_t opcode    = (instruction >> 11U) & 0x3U;
    const bool     is_scalar = 0x5F10E400U == (instruction & 0xDFF0E400U);

    if ((0x0F10E400U != (instruction & 0x9FF0E400U) && false == is_scalar)
        || (opcode != 0U && opcode != 3U) || false == has_feature(compiler, BAL_TIER1_FEATURE_FMA)
        || false == has_feature(compiler, BAL_TIER1_FEATURE_SSE41))
    {
        return false;
    }

    const bool     is_to_int   = 3U == opcode;
    const bool     is_unsigned = 1U == ((instruction >> 29U) & 1U);
    const bool     is_quad     = false == is_scalar && 1U == ((instruction >> 30U) & 1U);
    const uint32_t fbits       = 32U - ((instruction >> 16U) & 0x7FU);
    const uint32_t rd          = instruction & 0x1FU;
    const uint32_t rn          = (instruction >> 5U) & 0x1FU;
    bal_x86_assembler_t *BAL_RESTRICT assembler = &compiler->assembler;
    size_t                            slow_jumps[4];
    uint32_t                          count = 1U;

    slow_jumps[0] = begin_float_sequence(compiler);

    for (uint32_t group = 0U; group < (true == is_quad ? 2U : 1U); ++group)
    {
        // XMM0 = the four lanes as singles.
        if (true == is_scalar)
        {
            emit_load_half(assembler, XMM4, rn);
        }
        else
        {
            emit_load_vector(assembler, XMM4, rn, is_quad);
        }

        if (1U == group)
        {
            emit_sse_immediate(assembler, 0x66, 0x73, 3U, XMM4, 8U, "psrldq");
        }

        if (true == is_to_int)
        {
            emit_halves_to_singles(assembler, XMM0, XMM4);
            emit_broadcast_pattern(
                assembler, XMM1, 0x0000000100000001ULL * ((127U + fbits) << 23U));
            emit_sse(assembler, 0U, false, 0x59, XMM0, XMM1, "mulps");

            // Truncate exactly, narrow with saturation and widen back: a lane out of range (or
            // a NaN or an infinity, the indefinite value) differs. Inexact comes only then.
            emit_sse3a(assembler, 0x08, XMM2, XMM0, 0x0B, "roundps");
            emit_sse(assembler, 0xF3, false, 0x5B, XMM2, XMM2, "cvttps2dq");
            emit_sse(assembler, 0U, false, 0x28, XMM3, XMM2, "movaps");

            if (true == is_unsigned)
            {
                emit_sse38(assembler, 0x2B, XMM3, XMM3, "packusdw");
            }
            else
            {
                emit_sse(assembler, 0x66, false, 0x6B, XMM3, XMM3, "packssdw");
            }

            emit_sse(assembler, 0U, false, 0x28, XMM5, XMM3, "movaps");
            emit_widen(assembler, XMM3, 1U, is_unsigned, false);
            emit_lanes_equal(assembler, 2U, XMM3, XMM2);
            emit_sse(assembler, 0U, false, 0x50, (uint8_t)SCRATCH_A, XMM3, "movmskps");
            bal_x86_emit_sub_r32_imm32(assembler, SCRATCH_A, 0xF);
            slow_jumps[count++] = emit_forward_jump(assembler, true, BAL_X86_COND_NE);
            emit_sse3a(assembler, 0x08, XMM3, XMM0, 0x03, "roundps");
            emit_sse(assembler, 0U, false, 0x28, XMM0, XMM5, "movaps");
        }
        else
        {
            emit_sse(assembler, 0U, false, 0x28, XMM0, XMM4, "movaps");
            emit_widen(assembler, XMM0, 1U, is_unsigned, false);
            emit_sse(assembler, 0U, false, 0x5B, XMM0, XMM0, "cvtdq2ps");
            emit_broadcast_pattern(
                assembler, XMM1, 0x0000000100000001ULL * ((127U - fbits) << 23U));
            emit_sse(assembler, 0U, false, 0x59, XMM0, XMM1, "mulps");
            emit_singles_to_halves(assembler, XMM0, XMM0);
        }

        emit_half_result(assembler, group);
    }

    emit_half_store(
        compiler, instruction, rd, is_quad, is_scalar, false == is_to_int, slow_jumps, count);
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
           || emit_fast_integer_to_float(compiler, instruction)
           || emit_fast_float_to_integer(compiler, instruction)
           || emit_fast_round(compiler, instruction) || emit_fast_move(compiler, instruction)
           || emit_fast_select(compiler, instruction) || emit_fast_duplicate(compiler, instruction)
           || emit_fast_vector_integer(compiler, instruction)
           || emit_fast_shift_immediate(compiler, instruction)
           || emit_fast_vector_misc(compiler, instruction)
           || emit_fast_float_misc(compiler, instruction)
           || emit_fast_vector_float_compare(compiler, instruction)
           || emit_fast_permute(compiler, instruction) || emit_fast_extract(compiler, instruction)
           || emit_fast_table(compiler, instruction)
           || emit_fast_modified_immediate(compiler, instruction)
           || emit_fast_element_move(compiler, instruction)
           || emit_fast_convert_precision(compiler, instruction)
           || emit_fast_float_minmax(compiler, instruction)
           || emit_fast_vector_convert(compiler, instruction)
           || emit_fast_vector_integer_more(compiler, instruction)
           || emit_fast_add_pair(compiler, instruction) || emit_fast_widening(compiler, instruction)
           || emit_fast_integer_by_element(compiler, instruction)
           || emit_fast_vector_misc_more(compiler, instruction)
           || emit_fast_across_lanes(compiler, instruction)
           || emit_fast_float_add_pair(compiler, instruction)
           || emit_fast_float_extreme_pair(compiler, instruction)
           || emit_fast_table_lookup(compiler, instruction)
           || emit_fast_byte_permute(compiler, instruction)
           || emit_fast_half_arithmetic(compiler, instruction)
           || emit_fast_vector_precision(compiler, instruction)
           || emit_fast_half_convert(compiler, instruction)
           || emit_fast_float_step(compiler, instruction)
           || emit_fast_float_compare_more(compiler, instruction)
           || emit_fast_estimate(compiler, instruction)
           || emit_fast_doubling_multiply(compiler, instruction)
           || emit_fast_dot_product(compiler, instruction)
           || emit_fast_saturating_shift_narrow(compiler, instruction)
           || emit_fast_conditional_compare(compiler, instruction)
           || emit_fast_crypto(compiler, instruction)
           || emit_fast_scalar_more(compiler, instruction)
           || emit_fast_half_compare(compiler, instruction)
           || emit_fast_half_vector(compiler, instruction)
           || emit_fast_register_shift(compiler, instruction)
           || emit_fast_count_leading(compiler, instruction)
           || emit_fast_narrow_high(compiler, instruction)
           || emit_fast_halving_subtract(compiler, instruction)
           || emit_fast_byte_multiply(compiler, instruction)
           || emit_fast_saturating_unary(compiler, instruction)
           || emit_fast_saturating_arithmetic(compiler, instruction)
           || emit_fast_mixed_saturating_add(compiler, instruction)
           || emit_fast_saturating_scalar(compiler, instruction)
           || emit_fast_sha(compiler, instruction) || emit_fast_half_fused(compiler, instruction)
           || emit_fast_round_away(compiler, instruction)
           || emit_fast_half_extreme(compiler, instruction)
           || emit_fast_half_arithmetic_more(compiler, instruction)
           || emit_fast_double_to_half(compiler, instruction)
           || emit_fast_half_estimate(compiler, instruction)
           || emit_fast_float_difference(compiler, instruction)
           || emit_fast_reciprocal_exponent(compiler, instruction)
           || emit_fast_convert_odd(compiler, instruction)
           || emit_fast_unsigned_estimate(compiler, instruction)
           || emit_fast_polynomial_multiply(compiler, instruction)
           || emit_fast_convert_lanes(compiler, instruction)
           || emit_fast_doubling_long(compiler, instruction)
           || emit_fast_rounding_doubling_accumulate(compiler, instruction)
           || emit_fast_mixed_saturating_add_more(compiler, instruction)
           || emit_fast_move_immediate(compiler, instruction)
           || emit_fast_double_estimate(compiler, instruction)
           || emit_fast_half_fixed(compiler, instruction);
}
