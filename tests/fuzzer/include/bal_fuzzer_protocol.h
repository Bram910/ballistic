#ifndef BALLISTIC_BAL_FUZZER_PROTOCOL_H
#define BALLISTIC_BAL_FUZZER_PROTOCOL_H

#include "backend/bal_cpu.h"
#include "bal_attributes.h"
#include <stdint.h>

#define BAL_FUZZER_MAX_INSTRUCTIONS 60U

BAL_ALIGNED(64) typedef struct
{
    uint64_t x[32];
    uint64_t pc;
    uint8_t  flag_c;
    uint8_t  flag_z;
    uint8_t  flag_n;
    uint8_t  flag_v;
    /// The bal_exception_t the instruction raised, or BAL_EXCEPTION_NONE.
    uint32_t exception;
    uint64_t instruction_count;
    uint64_t v[32][2];
    uint64_t pad1[29];
} bal_fuzzer_cpu_snapshot_t;

static_assert(sizeof(bal_fuzzer_cpu_snapshot_t) == sizeof(bal_cpu_t), "Struct size mismatch");

typedef enum
{
    BAL_FUZZER_WORKER_OK = 0,
    BAL_FUZZER_WORKER_ERROR_COMPILE_FAILED,
    BAL_FUZZER_WORKER_ERROR_EXECUTION_FAILED,
    BAL_FUZZER_WORKER_ERROR_TIMEOUT,
    BAl_FUZZER_WORKER_ERROR_CRASHED,
    BAL_FUZZER_WORKER_ERROR_UNKNOWN_INSTRUCTION,

    /// The instruction accessed memory the worker does not map. The final state is still sent so
    /// it can be compared with the other engine.
    BAL_FUZZER_WORKER_MEMORY_FAULT,

    /// The instruction raised another guest exception, such as an SVC; final_state.exception
    /// says which.
    BAL_FUZZER_WORKER_GUEST_EXCEPTION,
    BAL_FUZZER_WORKER_ERROR_END
} bal_fuzzer_worker_status_t;

/// Fuzz input sent to the worker.
BAL_ALIGNED(64) typedef struct
{
    uint64_t                  message_id;
    uint32_t                  instruction_count;
    uint32_t                  pad;
    uint32_t                  instructions[BAL_FUZZER_MAX_INSTRUCTIONS];
    bal_fuzzer_cpu_snapshot_t initial_state;
} bal_fuzzer_input_t;

/// Response sent from the worker.
BAL_ALIGNED(64) typedef struct
{
    uint64_t                   message_id; // Must match the input message_id.
    bal_fuzzer_worker_status_t status;
    char                       pad[52];
    bal_fuzzer_cpu_snapshot_t  final_state;
} bal_fuzzer_response_t;

#endif // BALLISTIC_BAL_FUZZER_PROTOCOL_H

/*** end of file ***/