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
} state_t;

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
    cpu.flag_n = (uint8_t)((state->nzcv >> 31U) & 1U);
    cpu.flag_z = (uint8_t)((state->nzcv >> 30U) & 1U);
    cpu.flag_c = (uint8_t)((state->nzcv >> 29U) & 1U);
    cpu.flag_v = (uint8_t)((state->nzcv >> 28U) & 1U);

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

            initial.nzcv     = (uint32_t)(next_random() & 0xFU) << 28U;
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
                      && 1U == ((instruction >> 20U) & 0x7U));

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
