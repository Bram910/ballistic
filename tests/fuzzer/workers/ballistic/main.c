#include "bal_attributes.h"
#include "bal_engine.h"
#include "bal_engine_flags.h"
#include "bal_fuzzer_ipc.h"
#include "bal_fuzzer_protocol.h"
#include "bal_fuzzer_state.h"
#include "bal_log.h"
#include <errno.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>

#define WORKER_GUEST_MEMORY 4096U
#define ARM64_RET_ENCODING  0xD65F03C0U

int
main(int argc, char **argv)
{
    if (1 == isatty(STDIN_FILENO) && argc != 2)
    {
        (void)fprintf(stderr, "Usage: %s [hex_instruction]\n", argv[0]);
        return 1;
    }

    bal_logger_init_default();
    bal_thread_logger.min_level    = BAL_LOG_LEVEL_NONE;
    bal_fuzzer_input_t    input    = {};
    bal_fuzzer_response_t response = {};

    bool program_launched_with_arguments = false;

    if (argc == 2)
    {
        program_launched_with_arguments = true;
        bal_thread_logger.min_level     = BAL_LOG_LEVEL_TRACE;
        const char *input_string        = argv[1];
        char       *end                 = NULL;
        errno                           = 0;

        const int           base  = 16;
        const unsigned long value = strtoul(input_string, &end, base);
        if (ERANGE == errno)
        {
            (void)fprintf(stderr, "Error: input value out of range.\n");
            return 1;
        }

        if (input_string == end)
        {
            (void)fprintf(stderr, "Error: No hex digits found in input.\n");
            return 1;
        }

        if (*end != '\0')
        {
            (void)fprintf(stderr, "Error: Trailing garbage characters found: '%s'\n", end);
            return 1;
        }

        if (value > UINT32_MAX)
        {
            (void)fprintf(stderr, "Error: Value 0x%lX exceeds 32-bit instruction size.\n", value);
            return 1;
        }

        const uint32_t instruction = (uint32_t)value;
        input.instructions[0]      = instruction;
        ++input.instruction_count;
    }

    for (;;)
    {
        if (false == program_launched_with_arguments)
        {
            if (bal_fuzzer_ipc_receive(STDIN_FILENO, &input, sizeof(input)) != BAL_SUCCESS)
            {
                return 0;
            }
        }

        response.message_id = input.message_id;
        response.status     = BAL_FUZZER_WORKER_OK;
        (void)memset(&response.final_state, 0, sizeof(response.final_state));

        uint32_t guest_memory[WORKER_GUEST_MEMORY / sizeof(uint32_t)];
        (void)memset(guest_memory, 0, sizeof(guest_memory));

        const uint32_t instruction_count = input.instruction_count;

        if (instruction_count > BAL_FUZZER_MAX_INSTRUCTIONS)
        {
            response.status = BAL_FUZZER_WORKER_ERROR_COMPILE_FAILED;

            if (false == program_launched_with_arguments)
            {
                (void)bal_fuzzer_ipc_send(STDOUT_FILENO, &response, sizeof(response));
            }

            continue;
        }

        (void)memcpy(guest_memory, input.instructions, instruction_count * sizeof(uint32_t));
        guest_memory[instruction_count] = ARM64_RET_ENCODING;
        bal_allocator_t allocator       = {};
        bal_allocator_default_init(&allocator);
        bal_memory_interface_t memory_interface = {};
        bal_error_t            status           = bal_flat_translation_interface_init(
            &allocator, &memory_interface, guest_memory, sizeof(guest_memory));

        if (status != BAL_SUCCESS)
        {
            response.status = BAL_FUZZER_WORKER_ERROR_COMPILE_FAILED;

            if (false == program_launched_with_arguments)
            {
                (void)bal_fuzzer_ipc_send(STDOUT_FILENO, &response, sizeof(response));
            }

            continue;
        }

        bal_cpu_t cpu = {};
        (void)memcpy(cpu.x, input.initial_state.x, sizeof(cpu.x));
        cpu.pc     = input.initial_state.pc;
        cpu.flag_c = input.initial_state.flag_c;
        cpu.flag_z = input.initial_state.flag_z;
        cpu.flag_n = input.initial_state.flag_n;
        cpu.flag_v = input.initial_state.flag_v;
        (void)memcpy(cpu.v, input.initial_state.v, sizeof(cpu.v));

        cpu.x[30]           = input.instruction_count * 4ULL;
        bal_engine_t engine = {};
        status              = bal_engine_init(&engine, &cpu, &allocator, &memory_interface);

        if (status != BAL_SUCCESS)
        {
            response.status = BAL_FUZZER_WORKER_ERROR_EXECUTION_FAILED;
            continue;
        }

        engine.flags |= BAL_ENGINE_FLAG_SINGLE_STEP;
        status = bal_engine_run_thread(&engine);

        if (BAL_ERROR_UNKNOWN_INSTRUCTION == status)
        {
            response.status = BAL_FUZZER_WORKER_ERROR_UNKNOWN_INSTRUCTION;
        }
        else if (BAL_ERROR_MEMORY_FAULT == status)
        {
            response.status = BAL_FUZZER_WORKER_MEMORY_FAULT;
        }
        else if (status != BAL_SUCCESS)
        {
            response.status = BAL_FUZZER_WORKER_ERROR_EXECUTION_FAILED;
        }
        else
        {
            response.status = BAL_FUZZER_WORKER_OK;
        }

        bal_fuzzer_state_capture_bal_cpu(&response.final_state, &cpu);
        bal_engine_destroy(&engine);
        (void)bal_flat_translation_interface_destroy(&allocator, &memory_interface);

        if (false == program_launched_with_arguments)
        {
            if (bal_fuzzer_ipc_send(STDOUT_FILENO, &response, sizeof(response)) != BAL_SUCCESS)
            {
                return 0;
            }
        }
        else
        {
            return 0;
        }
    }
}
