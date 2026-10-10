//! Measures which A64 instructions tier 1 translates correctly.
//!
//! For every entry of the decoder table, random encodings of that entry are run once through
//! Ballistic and once through Unicorn, starting from the same random register and memory state,
//! and the results are compared: the general purpose and SIMD&FP registers, NZCV, the FPSR flags,
//! the PC and guest memory. One CSV line is printed per entry:
//!
//!     index,mnemonic,mask,expected,status,translated,matched,samples
//!
//! where status is one of:
//!
//! - `verified`: tier 1 translated the samples and every result matched Unicorn,
//! - `wrong`: at least one translated sample did not match Unicorn,
//! - `unverified`: tier 1 translated the samples but Unicorn could not run any of them,
//! - `todo`: tier 1 rejected every sample.
//!
//! Pointer authentication runs with random keys, key enables (SCTLR_EL1) and address layouts
//! (TCR_EL1), set in both engines.
//!
//! Half of the general purpose registers hold addresses inside a mapped data area, so loads and
//! stores mostly reach memory. A memory fault or another guest exception, such as an SVC, raised
//! by both engines is a match when it is the same exception and the registers agree, as in the
//! fuzzer; one raised by only one of them is a mismatch.

#include "bal_decoder.h"
#include "bal_engine.h"
#include "bal_engine_flags.h"
#include "bal_log.h"
#include "generated/decoder_table.h"
#include "unicorn/unicorn.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define SAMPLES_PER_ENTRY  24U
#define ATTEMPTS_PER_ENTRY 256U
#define GUEST_MEMORY_SIZE  0x10000U
#define DATA_BASE          0x1000U
#define DATA_SIZE          0xC000U

// The FPSR cumulative flags (IOC, DZC, OFC, UFC, IXC, IDC) and QC.
#define FPSR_COMPARED_BITS 0x0800009FULL

// QEMU's exception numbers: an undefined instruction, an SVC, a data abort, which Unicorn raises
// for alignment faults, and a BRK.
#define UNICORN_EXCEPTION_UNDEFINED  1U
#define UNICORN_EXCEPTION_SVC        2U
#define UNICORN_EXCEPTION_DATA_ABORT 4U
#define UNICORN_EXCEPTION_BREAKPOINT 7U
#define UNICORN_EXCEPTION_NONE       0xFFFFFFFFU

typedef enum
{
    RUN_OK,
    RUN_MEMORY_FAULT,
    RUN_EXCEPTION,
    RUN_REJECTED,
} run_result_t;

typedef struct
{
    uint64_t x[32];
    uint64_t v[32][2];
    uint64_t pc;
    uint64_t fpsr;
    uint32_t nzcv;
    /// The bal_exception_t raised, or BAL_EXCEPTION_NONE.
    uint32_t exception;
    /// The pointer authentication keys and the SCTLR_EL1 and TCR_EL1 values, as in bal_cpu_t.
    uint64_t pac_keys[5][2];
    uint64_t sctlr_el1;
    uint64_t tcr_el1;
    /// Whether Unicorn needs the pointer authentication registers set.
    uint64_t has_pac_state;
} state_t;

/// SCTLR_EL1 as Unicorn resets it, without the key enables.
#define UNICORN_SCTLR_EL1 0xC50838ULL

static void
write_system_register(uc_engine *engine,
                      uint32_t   op0,
                      uint32_t   op1,
                      uint32_t   crn,
                      uint32_t   crm,
                      uint32_t   op2,
                      uint64_t   value)
{
    uc_arm64_cp_reg reg
        = { .crn = crn, .crm = crm, .op0 = op0, .op1 = op1, .op2 = op2, .val = value };
    (void)uc_reg_write(engine, UC_ARM64_REG_CP_REG, &reg);
}

/// Sets the keys, SCTLR_EL1 and TCR_EL1 in Unicorn, with SCR_EL3 and HCR_EL2 set (non-secure,
/// API and APK) so that pointer authentication does not trap to EL2 or EL3.
static void
write_pac_state(uc_engine *engine, const state_t *state)
{
    write_system_register(engine, 3, 6, 1, 1, 0, (3ULL << 16U) | (1ULL << 10U) | 1ULL);
    write_system_register(engine, 3, 4, 1, 1, 0, (3ULL << 40U) | (1ULL << 31U));
    write_system_register(engine, 3, 0, 1, 0, 0, UNICORN_SCTLR_EL1 | state->sctlr_el1);
    write_system_register(engine, 3, 0, 2, 0, 2, state->tcr_el1);

    // APIAKey, APIBKey, APDAKey, APDBKey and APGAKey: CRm 1, 1, 2, 2, 3 and op2 0 or 2 for the
    // low half, plus one for the high half.
    for (uint32_t key = 0U; key < 5U; ++key)
    {
        const uint32_t crm = key < 2U ? 1U : (key < 4U ? 2U : 3U);
        const uint32_t op2 = 1U == key || 3U == key ? 2U : 0U;
        write_system_register(engine, 3, 0, 2, crm, op2, state->pac_keys[key][0]);
        write_system_register(engine, 3, 0, 2, crm, op2 + 1U, state->pac_keys[key][1]);
    }
}

static uint64_t g_random_state = 0x2545F4914F6CDD1DULL;
static uint8_t  g_initial_memory[GUEST_MEMORY_SIZE];
static uint8_t  g_unicorn_memory[GUEST_MEMORY_SIZE];
static uint8_t  g_ballistic_memory[GUEST_MEMORY_SIZE];

static uint64_t
next_random(void)
{
    uint64_t value = (g_random_state += 0x9E3779B97F4A7C15ULL);
    value          = (value ^ (value >> 30U)) * 0xBF58476D1CE4E5B9ULL;
    value          = (value ^ (value >> 27U)) * 0x94D049BB133111EBULL;
    return value ^ (value >> 31U);
}

static void
on_exception(uc_engine *engine, uint32_t exception_number, void *user_data)
{
    *(uint32_t *)user_data = exception_number;
    (void)uc_emu_stop(engine);
}

static run_result_t
run_unicorn(uint32_t instruction, state_t *state)
{
    uc_engine *engine = NULL;

    if (uc_open(UC_ARCH_ARM64, UC_MODE_ARM, &engine) != UC_ERR_OK)
    {
        return RUN_REJECTED;
    }

    // The guest CPU implements the atomics, half precision and the other extensions of the max
    // model.
    (void)uc_ctl_set_cpu_model(engine, UC_CPU_ARM64_MAX);
    uint32_t exception_number = UNICORN_EXCEPTION_NONE;
    uc_hook  exception_hook   = 0U;
    (void)uc_hook_add(
        engine, &exception_hook, UC_HOOK_INTR, (void *)on_exception, &exception_number, 1U, 0U);

    if (0U != state->has_pac_state)
    {
        write_pac_state(engine, state);
    }

    (void)memcpy(g_unicorn_memory, g_initial_memory, sizeof(g_unicorn_memory));
    (void)memcpy(g_unicorn_memory, &instruction, sizeof(instruction));
    (void)uc_mem_map(engine, 0, GUEST_MEMORY_SIZE, UC_PROT_ALL);
    (void)uc_mem_write(engine, 0, g_unicorn_memory, GUEST_MEMORY_SIZE);

    for (int i = 0; i < 29; ++i)
    {
        (void)uc_reg_write(engine, UC_ARM64_REG_X0 + i, &state->x[i]);
    }

    (void)uc_reg_write(engine, UC_ARM64_REG_X29, &state->x[29]);
    (void)uc_reg_write(engine, UC_ARM64_REG_X30, &state->x[30]);
    (void)uc_reg_write(engine, UC_ARM64_REG_SP, &state->x[31]);
    (void)uc_reg_write(engine, UC_ARM64_REG_NZCV, &state->nzcv);

    for (int i = 0; i < 32; ++i)
    {
        (void)uc_reg_write(engine, UC_ARM64_REG_Q0 + i, state->v[i]);
    }

    const uc_err error = uc_emu_start(engine, 0, sizeof(instruction), 0, 1);

    for (int i = 0; i < 29; ++i)
    {
        (void)uc_reg_read(engine, UC_ARM64_REG_X0 + i, &state->x[i]);
    }

    (void)uc_reg_read(engine, UC_ARM64_REG_X29, &state->x[29]);
    (void)uc_reg_read(engine, UC_ARM64_REG_X30, &state->x[30]);
    (void)uc_reg_read(engine, UC_ARM64_REG_SP, &state->x[31]);
    (void)uc_reg_read(engine, UC_ARM64_REG_NZCV, &state->nzcv);
    (void)uc_reg_read(engine, UC_ARM64_REG_PC, &state->pc);

    for (int i = 0; i < 32; ++i)
    {
        (void)uc_reg_read(engine, UC_ARM64_REG_Q0 + i, state->v[i]);
    }

    uint32_t fpsr = 0U;
    (void)uc_reg_read(engine, UC_ARM64_REG_FPSR, &fpsr);
    state->fpsr = fpsr;
    (void)uc_mem_read(engine, 0, g_unicorn_memory, GUEST_MEMORY_SIZE);
    (void)uc_close(engine);

    if (UC_ERR_READ_UNMAPPED == error || UC_ERR_WRITE_UNMAPPED == error
        || UNICORN_EXCEPTION_DATA_ABORT == exception_number)
    {
        state->exception = BAL_EXCEPTION_DATA_ABORT;
        return RUN_MEMORY_FAULT;
    }

    switch (exception_number)
    {
        case UNICORN_EXCEPTION_SVC:
            state->exception = BAL_EXCEPTION_SUPERVISOR_CALL;
            return RUN_EXCEPTION;
        case UNICORN_EXCEPTION_BREAKPOINT:
            state->exception = BAL_EXCEPTION_BREAKPOINT;
            return RUN_EXCEPTION;
        case UNICORN_EXCEPTION_UNDEFINED:
            state->exception = BAL_EXCEPTION_UNDEFINED_INSTRUCTION;
            return RUN_EXCEPTION;
        default:
            break;
    }

    return UC_ERR_OK == error && UNICORN_EXCEPTION_NONE == exception_number ? RUN_OK : RUN_REJECTED;
}

static run_result_t
run_ballistic(uint32_t instruction, state_t *state)
{
    (void)memcpy(g_ballistic_memory, g_initial_memory, sizeof(g_ballistic_memory));
    (void)memcpy(g_ballistic_memory, &instruction, sizeof(instruction));

    bal_allocator_t allocator = { 0 };
    bal_allocator_default_init(&allocator);
    bal_memory_interface_t memory_interface = { 0 };

    if (bal_flat_translation_interface_init(
            &allocator, &memory_interface, g_ballistic_memory, sizeof(g_ballistic_memory))
        != BAL_SUCCESS)
    {
        return RUN_REJECTED;
    }

    bal_cpu_t cpu;
    (void)memset(&cpu, 0, sizeof(cpu));
    (void)memcpy(cpu.x, state->x, sizeof(cpu.x));
    (void)memcpy(cpu.v, state->v, sizeof(cpu.v));
    (void)memcpy(cpu.pac_keys, state->pac_keys, sizeof(cpu.pac_keys));
    cpu.sctlr_el1 = state->sctlr_el1;
    cpu.tcr_el1   = state->tcr_el1;
    cpu.flag_n    = (uint8_t)((state->nzcv >> 31U) & 1U);
    cpu.flag_z    = (uint8_t)((state->nzcv >> 30U) & 1U);
    cpu.flag_c    = (uint8_t)((state->nzcv >> 29U) & 1U);
    cpu.flag_v    = (uint8_t)((state->nzcv >> 28U) & 1U);

    bal_engine_t engine;
    (void)memset(&engine, 0, sizeof(engine));

    if (bal_engine_init(&engine, &cpu, &allocator, &memory_interface) != BAL_SUCCESS)
    {
        return RUN_REJECTED;
    }

    engine.flags |= BAL_ENGINE_FLAG_SINGLE_STEP;
    const bal_error_t status = bal_engine_run_thread(&engine);
    bal_engine_destroy(&engine);
    (void)bal_flat_translation_interface_destroy(&allocator, &memory_interface);

    if (status != BAL_SUCCESS && status != BAL_ERROR_MEMORY_FAULT
        && status != BAL_ERROR_GUEST_EXCEPTION)
    {
        return RUN_REJECTED;
    }

    (void)memcpy(state->x, cpu.x, sizeof(cpu.x));
    (void)memcpy(state->v, cpu.v, sizeof(cpu.v));
    state->nzcv = ((uint32_t)cpu.flag_n << 31U) | ((uint32_t)cpu.flag_z << 30U)
                  | ((uint32_t)cpu.flag_c << 29U) | ((uint32_t)cpu.flag_v << 28U);
    state->pc        = cpu.pc;
    state->fpsr      = cpu.fpsr;
    state->exception = cpu.exception;

    if (BAL_ERROR_GUEST_EXCEPTION == status)
    {
        return RUN_EXCEPTION;
    }

    return BAL_ERROR_MEMORY_FAULT == status ? RUN_MEMORY_FAULT : RUN_OK;
}

/// A random register value: half of the time an address inside the data area, mostly 16-byte
/// aligned so that accesses needing alignment can succeed.
static uint64_t
random_register(void)
{
    if (0U == (next_random() & 1U))
    {
        return next_random();
    }

    const uint64_t address = DATA_BASE + (next_random() % DATA_SIZE);
    return 0U != (next_random() % 4U) ? address & ~0xFULL : address;
}

/// A random pointer for pointer authentication: canonical in 48 bits, with a top byte, or random.
static uint64_t
random_pointer(void)
{
    switch (next_random() % 5U)
    {
        case 0:
            return random_register();
        case 1:
            return next_random() & 0x0000FFFFFFFFFFFFULL;
        case 2:
            return next_random() | 0xFFFF000000000000ULL;
        case 3:
            return (next_random() & 0x0000FFFFFFFFFFFFULL) | (next_random() << 56U);
        default:
            return next_random();
    }
}

/// Whether the PAC field of `pointer`, as AddPAC() sees it, holds copies of bit 55 only.
static bool
is_canonical_pointer(const state_t *state, const uint64_t pointer, const bool is_data)
{
    const bool     is_upper = 0U != ((pointer >> 55U) & 1U);
    const uint64_t tcr      = state->tcr_el1;
    const bool     tbi      = 0U != ((tcr >> (is_upper ? 38U : 37U)) & 1U);
    const bool     tbid     = 0U != ((tcr >> (is_upper ? 52U : 51U)) & 1U);
    const uint32_t bottom   = 64U - ((uint32_t)(tcr >> (is_upper ? 16U : 0U)) & 0x3FU);
    const uint32_t top      = tbi && (is_data || !tbid) ? 55U : 63U;
    const uint64_t field    = (~0ULL >> (63U - top)) & (~0ULL << bottom);
    return 0U == (pointer & field) || field == (pointer & field);
}

/// The pointer authentication cases Unicorn (QEMU 5.0) cannot verify: it corrupts bit 63 or 55
/// rather than bit 62 or 54 of the code it adds to a pointer with bad extension bits and picks
/// the range of such a pointer differently, it does not strip with XPAC* while every key is
/// disabled, and LDRAA or LDRAB with writeback to their own target register is CONSTRAINED
/// UNPREDICTABLE.
static bool
is_unverifiable_pac(const uint32_t instruction, const state_t *state)
{
    if (0xDAC10000U == (instruction & 0xFFFF0000U))
    {
        const uint32_t opcode = (instruction >> 10U) & 0x3FU;
        const uint32_t rd     = instruction & 0x1FU;

        if (opcode < 4U || (opcode >= 8U && opcode < 0xCU))
        {
            return false == is_canonical_pointer(state, state->x[rd], 0U != (opcode & 2U));
        }

        return opcode >= 0x10U && 0U == state->sctlr_el1;
    }

    // PACIA1716, PACIASP and the other signing hints (op2 bit 2 clear).
    if (0xD503211FU == (instruction & 0xFFFFFD1FU) && 0U == ((instruction >> 7U) & 1U))
    {
        const uint32_t target = 0U == ((instruction >> 9U) & 1U) ? 17U : 30U;
        return false == is_canonical_pointer(state, state->x[target], false);
    }

    if (0xD50320FFU == instruction)
    {
        return 0U == state->sctlr_el1;
    }

    if (0xF8200C00U == (instruction & 0xFF200C00U))
    {
        return (instruction & 0x1FU) == ((instruction >> 5U) & 0x1FU);
    }

    return false;
}

/// The PAC* instruction that signs the pointer `instruction` authenticates, with the same key and
/// modifier, and the register it signs (`target`). Zero when there is none.
static uint32_t
signing_instruction(const uint32_t instruction, uint32_t *target)
{
    const uint32_t rn = (instruction >> 5U) & 0x1FU;
    const uint32_t rd = instruction & 0x1FU;

    // AUTIA, AUTIB, AUTDA, AUTDB (opcode 0001xx) and their Z forms (0011xx).
    if (0xDAC10000U == (instruction & 0xFFFF0000U))
    {
        const uint32_t opcode = (instruction >> 10U) & 0x3FU;
        *target               = rd;

        if (opcode >= 0x4U && opcode <= 0x7U)
        {
            return 0xDAC10000U | ((opcode - 4U) << 10U) | (rn << 5U) | rd;
        }

        if (opcode >= 0xCU && opcode <= 0xFU)
        {
            return 0xDAC10000U | ((opcode - 4U) << 10U) | (31U << 5U) | rd;
        }

        return 0U;
    }

    // AUTIA1716 and AUTIB1716 sign X17 with X16; AUTIASP, AUTIAZ and the B forms X30 with SP or
    // zero.
    if (0xD503211FU == (instruction & 0xFFFFFD1FU))
    {
        const uint32_t op2 = (instruction >> 5U) & 7U;
        const uint32_t key = (op2 >> 1U) & 1U;

        if (op2 < 4U)
        {
            return 0U;
        }

        if (0U == ((instruction >> 9U) & 1U))
        {
            *target = 17U;
            return 0xDAC10000U | (key << 10U) | (16U << 5U) | 17U;
        }

        *target = 30U;
        return 0U != (op2 & 1U) ? 0xDAC10000U | (key << 10U) | (31U << 5U) | 30U
                                : 0xDAC10000U | ((8U + key) << 10U) | (31U << 5U) | 30U;
    }

    // LDRAA and LDRAB: PACDZA or PACDZB on the base.
    if (0xF8200400U == (instruction & 0xFF200400U))
    {
        const uint32_t key = 2U + ((instruction >> 23U) & 1U);
        *target            = rn;
        return 31U == rn ? 0U : 0xDAC10000U | ((8U + key) << 10U) | (31U << 5U) | rn;
    }

    // The authenticated branches: RETAA and RETAB sign X30 with SP, the others Xn with Xm, SP or
    // zero.
    if (0xD61F0800U == (instruction & 0xFE9FF800U))
    {
        const uint32_t key = (instruction >> 10U) & 1U;
        const uint32_t rm  = instruction & 0x1FU;

        if (0x00400000U == (instruction & 0x00600000U))
        {
            *target = 30U;
            return 0xDAC10000U | (key << 10U) | (31U << 5U) | 30U;
        }

        *target = rn;

        if (31U == rn)
        {
            return 0U;
        }

        return 0U != ((instruction >> 24U) & 1U)
                   ? 0xDAC10000U | (key << 10U) | (rm << 5U) | rn
                   : 0xDAC10000U | ((8U + key) << 10U) | (31U << 5U) | rn;
    }

    return 0U;
}

/// MRS, MSR (register) and MSR (immediate) of DIT and SSBS, which the guest CPU has and Unicorn
/// (QEMU 5.0) does not.
static bool
is_dit_or_ssbs(const uint32_t instruction)
{
    const uint32_t register_access = instruction & 0xFFDFFFE0U;
    const uint32_t immediate       = instruction & 0xFFFFF0FFU;
    return 0xD51B42A0U == register_access || 0xD51B42C0U == register_access
           || 0xD503405FU == immediate || 0xD503403FU == immediate;
}

/// Gives a pointer authentication sample random pointers, keys, enables and address layout.
static void
randomize_pac_state(const uint32_t instruction, state_t *state)
{
    static const uint64_t ENABLES[4] = { 1ULL << 31U, 1ULL << 30U, 1ULL << 27U, 1ULL << 13U };

    state->has_pac_state = 1U;

    for (uint32_t r = 0U; r < 32U; ++r)
    {
        state->x[r] = random_pointer();
    }

    for (uint32_t key = 0U; key < 5U; ++key)
    {
        state->pac_keys[key][0] = next_random();
        state->pac_keys[key][1] = next_random();
    }

    for (uint32_t key = 0U; key < 4U; ++key)
    {
        state->sctlr_el1 |= 0U != (next_random() % 4U) ? ENABLES[key] : 0U;
    }

    // T0SZ and T1SZ in 16 to 39, TBI0, TBI1, TBID0 and TBID1.
    state->tcr_el1 = (16U + next_random() % 24U) | ((16U + next_random() % 24U) << 16U)
                     | ((next_random() & 3U) << 37U) | ((next_random() & 3U) << 51U);

    // Unicorn authenticates LDRAA and LDRAB against SP rather than zero.
    if (0xF8200400U == (instruction & 0xFF200400U))
    {
        state->x[31] = 0U;
    }

    // Half of the authentications get a pointer Unicorn signed, so that they can succeed and the
    // branches land in mapped memory.
    uint32_t       target = 0U;
    const uint32_t signer = signing_instruction(instruction, &target);

    if (signer != 0U && 0U != (next_random() & 1U))
    {
        state->x[target]     = random_register();
        state_t signed_state = *state;

        if (RUN_OK == run_unicorn(signer, &signed_state))
        {
            state->x[target] = signed_state.x[target];
        }
    }
}

int
main(void)
{
    bal_logger_init_default();
    bal_thread_logger.min_level = BAL_LOG_LEVEL_NONE;

    for (uint32_t i = 0U; i < GUEST_MEMORY_SIZE; i += 8U)
    {
        const uint64_t value = next_random();
        (void)memcpy(&g_initial_memory[i], &value, sizeof(value));
    }

    for (uint32_t index = 0U; index < BAL_DECODER_ARM64_INSTRUCTIONS_SIZE; ++index)
    {
        const bal_decoder_instruction_metadata_t *entry = &g_bal_decoder_arm64_instructions[index];
        uint32_t samples = 0U, translated = 0U, compared = 0U, matched = 0U;

        for (uint32_t attempt = 0U; attempt < ATTEMPTS_PER_ENTRY && samples < SAMPLES_PER_ENTRY;
             ++attempt)
        {
            const uint32_t instruction = entry->expected | ((uint32_t)next_random() & ~entry->mask);

            // Only count encodings the decoder resolves to this very entry.
            if (bal_decode_arm64(instruction) != entry)
            {
                continue;
            }

            ++samples;
            state_t initial;
            (void)memset(&initial, 0, sizeof(initial));

            for (uint32_t r = 0U; r < 32U; ++r)
            {
                initial.x[r]    = random_register();
                initial.v[r][0] = next_random();
                initial.v[r][1] = next_random();
            }

            initial.nzcv = (uint32_t)(next_random() & 0xFU) << 28U;

            if (OPCODE_POINTER_AUTHENTICATION == entry->ir_opcode)
            {
                randomize_pac_state(instruction, &initial);
            }

            state_t expected = initial;
            state_t actual   = initial;

            const run_result_t ballistic_result = run_ballistic(instruction, &actual);

            if (RUN_REJECTED == ballistic_result)
            {
                continue;
            }

            ++translated;

            // Unicorn compares the zero-extended memory value in LDSMAX and LDSMIN narrower than
            // 64 bits, which the architecture compares signed, so it cannot verify them.
            // Unicorn also writes the 16-bit result of FCVTZS and FCVTZU (scalar, fixed-point) on
            // half precision sign-extended to 32 bits instead of clearing the register above it.
            const bool is_known_unicorn_bug
                = (0x38204000U == (instruction & 0x3F20EC00U) && (instruction >> 30U) != 3U)
                  || (0x5F00FC00U == (instruction & 0xDF80FC00U)
                      && 1U == ((instruction >> 20U) & 0x7U))
                  || (OPCODE_POINTER_AUTHENTICATION == entry->ir_opcode
                      && true == is_unverifiable_pac(instruction, &initial))
                  || true == is_dit_or_ssbs(instruction);

            if (true == is_known_unicorn_bug)
            {
                continue;
            }

            const run_result_t unicorn_result = run_unicorn(instruction, &expected);

            if (RUN_REJECTED == unicorn_result)
            {
                continue;
            }

            ++compared;
            const int is_match
                = ballistic_result == unicorn_result && expected.exception == actual.exception
                  && 0 == memcmp(expected.x, actual.x, sizeof(expected.x))
                  && 0 == memcmp(expected.v, actual.v, sizeof(expected.v))
                  && expected.nzcv == actual.nzcv && expected.pc == actual.pc
                  && (expected.fpsr & FPSR_COMPARED_BITS) == (actual.fpsr & FPSR_COMPARED_BITS)
                  && (RUN_MEMORY_FAULT == ballistic_result || RUN_EXCEPTION == ballistic_result
                      || 0 == memcmp(g_unicorn_memory, g_ballistic_memory, GUEST_MEMORY_SIZE));
            matched += is_match ? 1U : 0U;
        }

        const char *status = "todo";

        if (compared > matched)
        {
            status = "wrong";
        }
        else if (compared > 0U)
        {
            status = "verified";
        }
        else if (translated > 0U)
        {
            status = "unverified";
        }

        (void)printf("%u,%s,0x%08X,0x%08X,%s,%u,%u,%u\n",
                     index,
                     entry->name,
                     entry->mask,
                     entry->expected,
                     status,
                     translated,
                     matched,
                     samples);
    }

    return EXIT_SUCCESS;
}
