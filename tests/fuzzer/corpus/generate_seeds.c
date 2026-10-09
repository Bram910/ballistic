#include "bal_decoder.h"
#include "bal_log.h"
#include "generated/decoder_table.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static const uint8_t REGISTER_SEEDS[] = { 0U, 1U, 15U, 30U, 31U };
#define REGISTER_SEEDS_SIZE (uint8_t)(sizeof(REGISTER_SEEDS) / sizeof(REGISTER_SEEDS[0]))

static const uint8_t CONDITION_SEEDS[] = { 0U, 1U, 7U, 14U, 15U };
#define CONDITION_SEEDS_SIZE (uint8_t)(sizeof(CONDITION_SEEDS) / sizeof(CONDITION_SEEDS[0]))

static uint32_t build_seed(const bal_decoder_instruction_metadata_t *BAL_RESTRICT
                                    instruction_metadata,
                           uint8_t  operand_index,
                           uint32_t value);

int
main(void)
{
    bal_logger_init_default();
    FILE *seeds_file = fopen("seeds.bin", "wb");

    if (NULL == seeds_file)
    {
        BAL_LOG_ERROR(&bal_thread_logger,
                      "Aborting process: failed to open seeds.bin for writing.");
        return EXIT_FAILURE;
    }

    const bal_decoder_instruction_metadata_t *BAL_RESTRICT metadata_cursor
        = g_bal_decoder_arm64_instructions;
    uint32_t seeds_written_to_file = 0U;

    for (uint32_t i = 0U; i < BAL_DECODER_ARM64_INSTRUCTIONS_SIZE; ++i)
    {
        uint32_t                                               seed = metadata_cursor->expected;
        const bal_decoder_instruction_metadata_t *BAL_RESTRICT decoded_instruction_metadata
            = bal_decode_arm64(seed);

        if (decoded_instruction_metadata != NULL)
        {
            (void)fwrite(&seed, sizeof(seed), 1U, seeds_file);
            ++seeds_written_to_file;
        }

        for (uint8_t operand_index = 0U; operand_index < BAL_OPERANDS_SIZE; ++operand_index)
        {
            const bal_decoder_operand_t *BAL_RESTRICT operand_cursor
                = &metadata_cursor->operands[operand_index];

            if (BAL_OPERAND_TYPE_NONE == operand_cursor->type)
            {
                continue;
            }

            if (BAL_OPERAND_TYPE_REGISTER_32 == operand_cursor->type
                || BAL_OPERAND_TYPE_REGISTER_64 == operand_cursor->type
                || BAL_OPERAND_TYPE_REGISTER_128 == operand_cursor->type)
            {
                for (uint8_t register_seed_index = 0U; register_seed_index < REGISTER_SEEDS_SIZE;
                     ++register_seed_index)
                {
                    seed = build_seed(
                        metadata_cursor, operand_index, REGISTER_SEEDS[register_seed_index]);

                    if (bal_decode_arm64(seed) != NULL)
                    {
                        (void)fwrite(&seed, sizeof(seed), 1U, seeds_file);
                        ++seeds_written_to_file;
                    }
                }
            }
            else if (BAL_OPERAND_TYPE_IMMEDIATE == operand_cursor->type)
            {
                const uint32_t max_value          = (1U << operand_cursor->bit_width) - 1U;
                const uint32_t immediate_seeds[3] = { 0U, max_value / 2U, max_value };

                for (uint32_t immediate_seed_index = 0U; immediate_seed_index < 3U;
                     ++immediate_seed_index)
                {
                    seed = build_seed(
                        metadata_cursor, operand_index, immediate_seeds[immediate_seed_index]);

                    if (bal_decode_arm64(seed) != NULL)
                    {
                        (void)fwrite(&seed, sizeof(seed), 1U, seeds_file);
                        ++seeds_written_to_file;
                    }
                }
            }
            else if (BAL_OPERAND_TYPE_CONDITION == operand_cursor->type)
            {
                for (uint32_t condition_seed_index = 0U;
                     condition_seed_index < CONDITION_SEEDS_SIZE;
                     ++condition_seed_index)
                {
                    seed = build_seed(
                        metadata_cursor, operand_index, CONDITION_SEEDS[condition_seed_index]);

                    if (bal_decode_arm64(seed) != NULL)
                    {
                        (void)fwrite(&seed, sizeof(seed), 1U, seeds_file);
                        ++seeds_written_to_file;
                    }
                }
            }
            else
            {
            }
        }

        ++metadata_cursor;
    }

    (void)fclose(seeds_file);
    BAL_LOG_INFO(&bal_thread_logger,
                 "Wrote %u seeds from %u instructions to seeds.bin.\n",
                 seeds_written_to_file,
                 (uint32_t)BAL_DECODER_ARM64_INSTRUCTIONS_SIZE);
    return EXIT_SUCCESS;
}

static uint32_t
build_seed(const bal_decoder_instruction_metadata_t *BAL_RESTRICT instruction_metadata,
           const uint8_t                                          operand_index,
           const uint32_t                                         value)
{
    const bal_decoder_operand_t *operand = &instruction_metadata->operands[operand_index];
    const uint32_t operand_mask = ((1U << operand->bit_width) - 1U) << operand->bit_position;
    const uint32_t encoded_seed = (instruction_metadata->expected & ~operand_mask)
                                  | ((value << operand->bit_position) & operand_mask);
    return encoded_seed;
}