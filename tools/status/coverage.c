//! Measures which A64 instructions tier 1 translates correctly.
//!
//! For every entry of the decoder table, random encodings of that entry are run once through
//! Ballistic and once through Unicorn, starting from the same random register state, and the
//! results are compared. One CSV line is printed per entry:
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
//! Loads and stores are run against an unmapped address in both engines, so they only count once
//! tier 1 can translate them.

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
#define GUEST_MEMORY_SIZE  4096U

typedef struct
{
    uint64_t x[32];
    uint32_t nzcv;
    uint32_t pad;
    uint64_t pc;
} state_t;

static uint64_t g_random_state = 0x2545F4914F6CDD1DULL;

static uint64_t
next_random(void)
{
    uint64_t value = (g_random_state += 0x9E3779B97F4A7C15ULL);
    value          = (value ^ (value >> 30U)) * 0xBF58476D1CE4E5B9ULL;
    value          = (value ^ (value >> 27U)) * 0x94D049BB133111EBULL;
    return value ^ (value >> 31U);
}

static int
run_unicorn(uint32_t instruction, state_t *state)
{
    uc_engine *engine = NULL;

    if (uc_open(UC_ARCH_ARM64, UC_MODE_ARM, &engine) != UC_ERR_OK)
    {
        return -1;
    }

    (void)uc_mem_map(engine, 0, GUEST_MEMORY_SIZE, UC_PROT_ALL);
    (void)uc_mem_write(engine, 0, &instruction, sizeof(instruction));

    for (int i = 0; i < 29; ++i)
    {
        (void)uc_reg_write(engine, UC_ARM64_REG_X0 + i, &state->x[i]);
    }

    (void)uc_reg_write(engine, UC_ARM64_REG_X29, &state->x[29]);
    (void)uc_reg_write(engine, UC_ARM64_REG_X30, &state->x[30]);
    (void)uc_reg_write(engine, UC_ARM64_REG_SP, &state->x[31]);
    (void)uc_reg_write(engine, UC_ARM64_REG_NZCV, &state->nzcv);
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
    (void)uc_close(engine);
    return UC_ERR_OK == error ? 0 : 1;
}

// Returns 0 when tier 1 ran the instruction, 1 when it rejected it.
static int
run_ballistic(uint32_t instruction, state_t *state)
{
    static uint32_t guest_memory[GUEST_MEMORY_SIZE / sizeof(uint32_t)];
    (void)memset(guest_memory, 0, sizeof(guest_memory));
    guest_memory[0] = instruction;

    bal_allocator_t allocator = { 0 };
    bal_allocator_default_init(&allocator);
    bal_memory_interface_t memory_interface = { 0 };

    if (bal_flat_translation_interface_init(
            &allocator, &memory_interface, guest_memory, sizeof(guest_memory))
        != BAL_SUCCESS)
    {
        return -1;
    }

    bal_cpu_t cpu;
    (void)memset(&cpu, 0, sizeof(cpu));
    (void)memcpy(cpu.x, state->x, sizeof(cpu.x));
    cpu.flag_n = (uint8_t)((state->nzcv >> 31U) & 1U);
    cpu.flag_z = (uint8_t)((state->nzcv >> 30U) & 1U);
    cpu.flag_c = (uint8_t)((state->nzcv >> 29U) & 1U);
    cpu.flag_v = (uint8_t)((state->nzcv >> 28U) & 1U);

    bal_engine_t engine;
    (void)memset(&engine, 0, sizeof(engine));

    if (bal_engine_init(&engine, &cpu, &allocator, &memory_interface) != BAL_SUCCESS)
    {
        return -1;
    }

    engine.flags |= BAL_ENGINE_FLAG_SINGLE_STEP;
    const bal_error_t status = bal_engine_run_thread(&engine);
    bal_engine_destroy(&engine);
    (void)bal_flat_translation_interface_destroy(&allocator, &memory_interface);

    if (status != BAL_SUCCESS)
    {
        return 1;
    }

    (void)memcpy(state->x, cpu.x, sizeof(cpu.x));
    state->nzcv = ((uint32_t)cpu.flag_n << 31U) | ((uint32_t)cpu.flag_z << 30U)
                  | ((uint32_t)cpu.flag_c << 29U) | ((uint32_t)cpu.flag_v << 28U);
    state->pc = cpu.pc;
    return 0;
}

int
main(void)
{
    bal_logger_init_default();
    bal_thread_logger.min_level = BAL_LOG_LEVEL_NONE;

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

            for (uint32_t r = 0U; r < 32U; ++r)
            {
                initial.x[r] = next_random();
            }

            initial.nzcv     = (uint32_t)(next_random() & 0xFU) << 28U;
            initial.pc       = 0U;
            state_t expected = initial;
            state_t actual   = initial;

            if (run_ballistic(instruction, &actual) != 0)
            {
                continue;
            }

            ++translated;

            if (run_unicorn(instruction, &expected) != 0)
            {
                continue;
            }

            ++compared;
            const int is_match = 0 == memcmp(expected.x, actual.x, sizeof(expected.x))
                                 && expected.nzcv == actual.nzcv && expected.pc == actual.pc;
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
