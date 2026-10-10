//! Runs small guest programs through the engine to check the system level behaviour that has no
//! deterministic reference in Unicorn: the generic timer, guest exceptions and resuming after them,
//! the CRC32 and DC ZVA results against fixed values, and pointer authentication round trips.

#include "bal_engine.h"
#include "bal_engine_flags.h"
#include "bal_log.h"
#include "unity.h"
#include <string.h>

#define GUEST_MEMORY_SIZE 0x4000U
#define RET               0xD65F03C0U

static uint8_t                g_memory[GUEST_MEMORY_SIZE];
static bal_allocator_t        g_allocator;
static bal_memory_interface_t g_memory_interface;
static bal_cpu_t              g_cpu;
static uint32_t               g_counter_reads;

void
setUp(void)
{
    bal_thread_logger.min_level = BAL_LOG_LEVEL_NONE;
    (void)memset(g_memory, 0, sizeof(g_memory));
    (void)memset(&g_cpu, 0, sizeof(g_cpu));
    g_cpu.x[30]     = BAL_ENGINE_SENTINEL;
    g_counter_reads = 0U;
    bal_allocator_default_init(&g_allocator);
    TEST_ASSERT_EQUAL(BAL_SUCCESS,
                      bal_flat_translation_interface_init(
                          &g_allocator, &g_memory_interface, g_memory, sizeof(g_memory)));
}

void
tearDown(void)
{
    (void)bal_flat_translation_interface_destroy(&g_allocator, &g_memory_interface);
}

/// Places `count` instructions at guest address 0, followed by a RET to the sentinel.
static void
load_program(const uint32_t *instructions, const uint32_t count)
{
    (void)memcpy(g_memory, instructions, count * sizeof(uint32_t));
    const uint32_t ret = RET;
    (void)memcpy(g_memory + count * sizeof(uint32_t), &ret, sizeof(ret));
}

/// Runs the guest from its current PC until it returns to the sentinel or stops.
static bal_error_t
run(void)
{
    bal_engine_t engine;
    (void)memset(&engine, 0, sizeof(engine));
    TEST_ASSERT_EQUAL(BAL_SUCCESS,
                      bal_engine_init(&engine, &g_cpu, &g_allocator, &g_memory_interface));
    const bal_error_t status = bal_engine_run_thread(&engine);
    bal_engine_destroy(&engine);
    return status;
}

static uint64_t
read_test_counter(void *context)
{
    ++g_counter_reads;
    return *(const uint64_t *)context;
}

static void
test_Mrs_CounterRegistersReadHostCallback(void)
{
    // MRS X0, CNTPCT_EL0; MRS X1, CNTVCT_EL0.
    const uint32_t program[] = { 0xD53BE020U, 0xD53BE041U };
    load_program(program, 2U);
    uint64_t count        = 0x123456789ABCULL;
    g_cpu.read_counter    = read_test_counter;
    g_cpu.counter_context = &count;

    TEST_ASSERT_EQUAL(BAL_SUCCESS, run());
    TEST_ASSERT_EQUAL_HEX64(count, g_cpu.x[0]);
    TEST_ASSERT_EQUAL_HEX64(count, g_cpu.x[1]);
    TEST_ASSERT_EQUAL_UINT32(2U, g_counter_reads);
}

static void
test_Mrs_CounterFrequencyDefaultsToSwitchFrequency(void)
{
    // MRS X2, CNTFRQ_EL0.
    const uint32_t program[] = { 0xD53BE002U };
    load_program(program, 1U);

    TEST_ASSERT_EQUAL(BAL_SUCCESS, run());
    TEST_ASSERT_EQUAL_UINT64(BAL_DEFAULT_COUNTER_FREQUENCY, g_cpu.x[2]);
    TEST_ASSERT_EQUAL_UINT64(19200000ULL, g_cpu.x[2]);
}

static void
test_Mrs_CounterFrequencyComesFromCpu(void)
{
    const uint32_t program[] = { 0xD53BE002U };
    load_program(program, 1U);
    g_cpu.counter_frequency = 1000000U;

    TEST_ASSERT_EQUAL(BAL_SUCCESS, run());
    TEST_ASSERT_EQUAL_UINT64(1000000U, g_cpu.x[2]);
}

static void
test_Mrs_CounterWithoutCallbackUsesMonotonicHostClock(void)
{
    // MRS X0, CNTPCT_EL0; MRS X1, CNTPCT_EL0.
    const uint32_t program[] = { 0xD53BE020U, 0xD53BE021U };
    load_program(program, 2U);

    TEST_ASSERT_EQUAL(BAL_SUCCESS, run());
    TEST_ASSERT_NOT_EQUAL(0U, g_cpu.x[0]);
    TEST_ASSERT_TRUE(g_cpu.x[1] >= g_cpu.x[0]);
}

static void
test_Svc_StopsAfterTheCallAndResumes(void)
{
    // MOVZ X0, #7; SVC #0x45; MOVZ X0, #9.
    const uint32_t program[] = { 0xD28000E0U, 0xD40008A1U, 0xD2800120U };
    load_program(program, 3U);

    TEST_ASSERT_EQUAL(BAL_ERROR_GUEST_EXCEPTION, run());
    TEST_ASSERT_EQUAL_UINT32(BAL_EXCEPTION_SUPERVISOR_CALL, g_cpu.exception);
    TEST_ASSERT_EQUAL_HEX64(0x45U, g_cpu.exception_immediate);
    TEST_ASSERT_EQUAL_HEX64(8U, g_cpu.pc);
    TEST_ASSERT_EQUAL_UINT64(7U, g_cpu.x[0]);

    // The host handles the call, clears the exception and resumes after the SVC.
    g_cpu.exception = BAL_EXCEPTION_NONE;
    TEST_ASSERT_EQUAL(BAL_SUCCESS, run());
    TEST_ASSERT_EQUAL_UINT64(9U, g_cpu.x[0]);
}

static void
test_Brk_StopsOnTheInstruction(void)
{
    // NOP; BRK #3.
    const uint32_t program[] = { 0xD503201FU, 0xD4200060U };
    load_program(program, 2U);

    TEST_ASSERT_EQUAL(BAL_ERROR_GUEST_EXCEPTION, run());
    TEST_ASSERT_EQUAL_UINT32(BAL_EXCEPTION_BREAKPOINT, g_cpu.exception);
    TEST_ASSERT_EQUAL_HEX64(3U, g_cpu.exception_immediate);
    TEST_ASSERT_EQUAL_HEX64(4U, g_cpu.pc);
}

static void
test_Udf_RaisesUndefinedInstruction(void)
{
    // UDF #0x1234.
    const uint32_t program[] = { 0x00001234U };
    load_program(program, 1U);

    TEST_ASSERT_EQUAL(BAL_ERROR_GUEST_EXCEPTION, run());
    TEST_ASSERT_EQUAL_UINT32(BAL_EXCEPTION_UNDEFINED_INSTRUCTION, g_cpu.exception);
    TEST_ASSERT_EQUAL_HEX64(0x1234U, g_cpu.exception_immediate);
    TEST_ASSERT_EQUAL_HEX64(0U, g_cpu.pc);
}

/// Computes the CRC of "123456789" with an initial value of all ones, as the check values of the
/// CRC-32 and CRC-32C standards do; the instructions leave the final inversion to software.
static uint32_t
crc_of_check_string(const uint32_t crc_x, const uint32_t crc_b)
{
    // CRC32(C)X W0, W0, X1; CRC32(C)B W0, W0, W2.
    const uint32_t program[] = { crc_x, crc_b };
    load_program(program, 2U);
    g_cpu.x[0] = 0xFFFFFFFFU;
    (void)memcpy(&g_cpu.x[1], "12345678", 8U);
    g_cpu.x[2] = '9';

    TEST_ASSERT_EQUAL(BAL_SUCCESS, run());
    return ~(uint32_t)g_cpu.x[0];
}

static void
test_Crc32_MatchesTheStandardCheckValue(void)
{
    TEST_ASSERT_EQUAL_HEX32(0xCBF43926U, crc_of_check_string(0x9AC14C00U, 0x1AC24000U));
}

static void
test_Crc32c_MatchesTheStandardCheckValue(void)
{
    TEST_ASSERT_EQUAL_HEX32(0xE3069283U, crc_of_check_string(0x9AC15C00U, 0x1AC25000U));
}

static void
test_DcZva_ZeroesTheAlignedBlock(void)
{
    // DC ZVA, X1.
    const uint32_t program[] = { 0xD50B7421U };
    load_program(program, 1U);
    (void)memset(g_memory + 0x1000U, 0xAA, 0x100U);
    g_cpu.x[1] = 0x1045U;

    TEST_ASSERT_EQUAL(BAL_SUCCESS, run());
    TEST_ASSERT_EQUAL_HEX8(0xAAU, g_memory[0x103FU]);

    for (uint32_t i = 0x1040U; i < 0x1080U; ++i)
    {
        TEST_ASSERT_EQUAL_HEX8(0U, g_memory[i]);
    }

    TEST_ASSERT_EQUAL_HEX8(0xAAU, g_memory[0x1080U]);
}

static void
test_DcZva_UnmappedBlockRaisesDataAbort(void)
{
    const uint32_t program[] = { 0xD50B7421U };
    load_program(program, 1U);
    g_cpu.x[1] = 0x100000U;

    TEST_ASSERT_EQUAL(BAL_ERROR_MEMORY_FAULT, run());
    TEST_ASSERT_EQUAL_UINT32(BAL_EXCEPTION_DATA_ABORT, g_cpu.exception);
    TEST_ASSERT_EQUAL_HEX64(0U, g_cpu.pc);
}

/// Enables the A instruction key with the keys and layout the known answers were computed with:
/// APIAKey = 0x2222:0x1111 and 48-bit addresses in both ranges, without top-byte-ignore.
static void
enable_instruction_key_a(void)
{
    g_cpu.pac_keys[BAL_PAC_KEY_IA][0] = 0x1111U;
    g_cpu.pac_keys[BAL_PAC_KEY_IA][1] = 0x2222U;
    g_cpu.sctlr_el1                   = 1ULL << 31U;
    g_cpu.tcr_el1                     = 0x100010U;
}

static void
test_Pacia_MatchesKnownCode(void)
{
    // PACIA X0, X1. The code is the one QEMU computes for the same key, pointer and modifier.
    const uint32_t program[] = { 0xDAC10020U };
    load_program(program, 1U);
    enable_instruction_key_a();
    g_cpu.x[0] = 0x0000123456789ABCULL;
    g_cpu.x[1] = 0x55U;

    TEST_ASSERT_EQUAL(BAL_SUCCESS, run());
    TEST_ASSERT_EQUAL_HEX64(0xEA06123456789ABCULL, g_cpu.x[0]);
}

static void
test_Autia_FailureMakesThePointerNonCanonical(void)
{
    // AUTIA X0, X1 on a pointer without a code: the A-key error code lands in bits 62:61.
    const uint32_t program[] = { 0xDAC11020U };
    load_program(program, 1U);
    enable_instruction_key_a();
    g_cpu.x[0] = 0x0000123456789ABCULL;
    g_cpu.x[1] = 0x55U;

    TEST_ASSERT_EQUAL(BAL_SUCCESS, run());
    TEST_ASSERT_EQUAL_HEX64(0x2000123456789ABCULL, g_cpu.x[0]);
}

static void
test_Paciasp_AutiaspRestoreTheLinkRegister(void)
{
    // PACIASP; MOV X1, X30; AUTIASP; then the RET to the sentinel needs the restored X30.
    const uint32_t program[] = { 0xD503233FU, 0xAA1E03E1U, 0xD50323BFU };
    load_program(program, 3U);
    enable_instruction_key_a();
    g_cpu.x[31] = 0x3F00U;

    TEST_ASSERT_EQUAL(BAL_SUCCESS, run());
    TEST_ASSERT_NOT_EQUAL(BAL_ENGINE_SENTINEL, g_cpu.x[1]);
    TEST_ASSERT_EQUAL_HEX64(BAL_ENGINE_SENTINEL, g_cpu.x[30]);
}

static void
test_PointerAuthenticationHints_AreNopsWhileDisabled(void)
{
    // PACIASP; MOV X1, X30.
    const uint32_t program[] = { 0xD503233FU, 0xAA1E03E1U };
    load_program(program, 2U);

    TEST_ASSERT_EQUAL(BAL_SUCCESS, run());
    TEST_ASSERT_EQUAL_HEX64(BAL_ENGINE_SENTINEL, g_cpu.x[1]);
}

static void
test_Pacibsp_FollowsTheBKeyEnable(void)
{
    // PACIBSP; MOV X1, X30; AUTIBSP, with only the A key enabled and then with the B key too.
    const uint32_t program[] = { 0xD503237FU, 0xAA1E03E1U, 0xD50323FFU };
    load_program(program, 3U);
    enable_instruction_key_a();
    g_cpu.x[31] = 0x3F00U;

    TEST_ASSERT_EQUAL(BAL_SUCCESS, run());
    TEST_ASSERT_EQUAL_HEX64(BAL_ENGINE_SENTINEL, g_cpu.x[1]);

    g_cpu.pc        = 0U;
    g_cpu.x[30]     = BAL_ENGINE_SENTINEL;
    g_cpu.sctlr_el1 = 1ULL << 30U;
    TEST_ASSERT_EQUAL(BAL_SUCCESS, run());
    TEST_ASSERT_NOT_EQUAL(BAL_ENGINE_SENTINEL, g_cpu.x[1]);
    TEST_ASSERT_EQUAL_HEX64(BAL_ENGINE_SENTINEL, g_cpu.x[30]);
}

static void
test_IcIvauAndIsb_RunTheCodeTheGuestWrote(void)
{
    // Calls a function returning 1, overwrites its first instruction with MOVZ X0, #2, and
    // calls it again after IC IVAU and ISB.
    const uint32_t program[] = {
        0xAA1E03E5U, // MOV X5, X30
        0x9400003FU, // BL 0x100
        0xAA0003E3U, // MOV X3, X0
        0xB9000041U, // STR W1, [X2]
        0xD50B7522U, // IC IVAU, X2
        0xD5033FDFU, // ISB
        0x9400003AU, // BL 0x100
        0xAA0503FEU, // MOV X30, X5
        RET,
    };
    const uint32_t function[] = { 0xD2800020U, RET }; // MOVZ X0, #1; RET
    (void)memcpy(g_memory, program, sizeof(program));
    (void)memcpy(g_memory + 0x100U, function, sizeof(function));
    g_cpu.x[1] = 0xD2800040U;
    g_cpu.x[2] = 0x100U;

    TEST_ASSERT_EQUAL(BAL_SUCCESS, run());
    TEST_ASSERT_EQUAL_UINT64(1U, g_cpu.x[3]);
    TEST_ASSERT_EQUAL_UINT64(2U, g_cpu.x[0]);
}

static void
test_ClearCache_RetranslatesTheGuestCode(void)
{
    // MOVZ X0, #1, run once; then MOVZ X0, #2 in its place after bal_engine_clear_cache().
    const uint32_t program[] = { 0xD2800020U };
    load_program(program, 1U);
    bal_engine_t engine;
    (void)memset(&engine, 0, sizeof(engine));
    TEST_ASSERT_EQUAL(BAL_SUCCESS,
                      bal_engine_init(&engine, &g_cpu, &g_allocator, &g_memory_interface));
    TEST_ASSERT_EQUAL(BAL_SUCCESS, bal_engine_run_thread(&engine));
    TEST_ASSERT_EQUAL_UINT64(1U, g_cpu.x[0]);

    const uint32_t replacement = 0xD2800040U;
    (void)memcpy(g_memory, &replacement, sizeof(replacement));
    bal_engine_clear_cache(&engine);
    g_cpu.pc    = 0U;
    g_cpu.x[30] = BAL_ENGINE_SENTINEL;
    TEST_ASSERT_EQUAL(BAL_SUCCESS, bal_engine_run_thread(&engine));
    TEST_ASSERT_EQUAL_UINT64(2U, g_cpu.x[0]);
    bal_engine_destroy(&engine);
}

static void
test_Retaa_ReturnsThroughTheSignedLinkRegister(void)
{
    // PACIASP; RETAA.
    const uint32_t program[] = { 0xD503233FU, 0xD65F0BFFU };
    (void)memcpy(g_memory, program, sizeof(program));
    enable_instruction_key_a();
    g_cpu.x[31] = 0x3F00U;

    // RETAA authenticates X30 into the PC and leaves X30 signed.
    TEST_ASSERT_EQUAL(BAL_SUCCESS, run());
    TEST_ASSERT_NOT_EQUAL(BAL_ENGINE_SENTINEL, g_cpu.x[30]);
}

int
main(void)
{
    UNITY_BEGIN();
    RUN_TEST(test_Mrs_CounterRegistersReadHostCallback);
    RUN_TEST(test_Mrs_CounterFrequencyDefaultsToSwitchFrequency);
    RUN_TEST(test_Mrs_CounterFrequencyComesFromCpu);
    RUN_TEST(test_Mrs_CounterWithoutCallbackUsesMonotonicHostClock);
    RUN_TEST(test_Svc_StopsAfterTheCallAndResumes);
    RUN_TEST(test_Brk_StopsOnTheInstruction);
    RUN_TEST(test_Udf_RaisesUndefinedInstruction);
    RUN_TEST(test_Crc32_MatchesTheStandardCheckValue);
    RUN_TEST(test_Crc32c_MatchesTheStandardCheckValue);
    RUN_TEST(test_DcZva_ZeroesTheAlignedBlock);
    RUN_TEST(test_DcZva_UnmappedBlockRaisesDataAbort);
    RUN_TEST(test_Pacia_MatchesKnownCode);
    RUN_TEST(test_Autia_FailureMakesThePointerNonCanonical);
    RUN_TEST(test_Paciasp_AutiaspRestoreTheLinkRegister);
    RUN_TEST(test_PointerAuthenticationHints_AreNopsWhileDisabled);
    RUN_TEST(test_Pacibsp_FollowsTheBKeyEnable);
    RUN_TEST(test_Retaa_ReturnsThroughTheSignedLinkRegister);
    RUN_TEST(test_IcIvauAndIsb_RunTheCodeTheGuestWrote);
    RUN_TEST(test_ClearCache_RetranslatesTheGuestCode);
    return UNITY_END();
}
