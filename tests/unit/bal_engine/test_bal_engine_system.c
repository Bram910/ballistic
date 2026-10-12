//! Runs small guest programs through the engine to check the system level behaviour that has no
//! deterministic reference in Unicorn: the generic timer, guest exceptions and resuming after them,
//! the CRC32 and DC ZVA results against fixed values, and pointer authentication round trips.

#include "bal_engine.h"
#include "bal_engine_flags.h"
#include "bal_log.h"
#include "unity.h"
#include <stdbool.h>
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
test_UnsupportedInstruction_RaisesWhenReached(void)
{
    // MOVZ X0, #1; AT S1E1R, X1 (EL1 only); MOVZ X0, #2.
    const uint32_t program[] = { 0xD2800020U, 0xD5087801U, 0xD2800040U };
    load_program(program, 3U);

    TEST_ASSERT_EQUAL(BAL_ERROR_UNKNOWN_INSTRUCTION, run());
    TEST_ASSERT_EQUAL_UINT64(1U, g_cpu.x[0]);
    TEST_ASSERT_EQUAL_UINT32(BAL_EXCEPTION_UNSUPPORTED_INSTRUCTION, g_cpu.exception);
    TEST_ASSERT_EQUAL_HEX64(0xD5087801U, g_cpu.exception_immediate);
    TEST_ASSERT_EQUAL_HEX64(4U, g_cpu.pc);
}

static void
test_FetchPastMappedMemory_FaultsAtTheUnmappedAddress(void)
{
    // MOVZ X0, #1; MOVZ X0, #2 in the last eight bytes, falling through into unmapped memory.
    const uint32_t program[] = { 0xD2800020U, 0xD2800040U };
    (void)memcpy(g_memory + GUEST_MEMORY_SIZE - 8U, program, sizeof(program));
    g_cpu.pc = GUEST_MEMORY_SIZE - 8U;

    TEST_ASSERT_EQUAL(BAL_ERROR_MEMORY_FAULT, run());
    TEST_ASSERT_EQUAL_UINT64(2U, g_cpu.x[0]);
    TEST_ASSERT_EQUAL_HEX64(GUEST_MEMORY_SIZE, g_cpu.pc);
}

static void
test_DitAndSsbs_ReadBackWhatWasWritten(void)
{
    const uint32_t program[] = {
        0xD503415FU, // MSR DIT, #1
        0xD53B42A0U, // MRS X0, DIT
        0xD503413FU, // MSR SSBS, #1
        0xD53B42C1U, // MRS X1, SSBS
        0xD51B42BFU, // MSR DIT, XZR
        0xD53B42A2U, // MRS X2, DIT
        0xD51B42C4U, // MSR SSBS, X4
        0xD53B42C3U, // MRS X3, SSBS
    };
    load_program(program, 8U);
    g_cpu.x[4] = ~0ULL;

    // Only the field's own bit is kept: DIT is bit 24 and SSBS bit 12.
    TEST_ASSERT_EQUAL(BAL_SUCCESS, run());
    TEST_ASSERT_EQUAL_HEX64(1ULL << 24U, g_cpu.x[0]);
    TEST_ASSERT_EQUAL_HEX64(1ULL << 12U, g_cpu.x[1]);
    TEST_ASSERT_EQUAL_HEX64(0U, g_cpu.x[2]);
    TEST_ASSERT_EQUAL_HEX64(1ULL << 12U, g_cpu.x[3]);
}

static bal_engine_t g_looping_engine;

/// Stops g_looping_engine on its 100th counter read.
static uint64_t
stop_after_100_reads(void *context)
{
    (void)context;

    if (100U == ++g_counter_reads)
    {
        bal_engine_stop_thread(&g_looping_engine);
    }

    return g_counter_reads;
}

static void
test_StopThread_LeavesChainedBlocks(void)
{
    // MRS X0, CNTPCT_EL0; B back to it. The block chains to itself, so only the stop request
    // ends the loop.
    const uint32_t program[] = { 0xD53BE020U, 0x17FFFFFFU };
    (void)memcpy(g_memory, program, sizeof(program));
    g_cpu.read_counter = stop_after_100_reads;
    (void)memset(&g_looping_engine, 0, sizeof(g_looping_engine));
    TEST_ASSERT_EQUAL(
        BAL_SUCCESS, bal_engine_init(&g_looping_engine, &g_cpu, &g_allocator, &g_memory_interface));

    TEST_ASSERT_EQUAL(BAL_SUCCESS, bal_engine_run_thread(&g_looping_engine));
    TEST_ASSERT_EQUAL_UINT32(100U, g_counter_reads);
    TEST_ASSERT_EQUAL_UINT64(100U, g_cpu.x[0]);
    TEST_ASSERT_EQUAL_HEX64(0U, g_cpu.pc);
    bal_engine_destroy(&g_looping_engine);
}

static void
test_ChainedLoop_ComputesTheSameResult(void)
{
    // A loop whose block chains to itself through B.NE, with inline floating point, so the
    // chained jump must hand the host MXCSR back as an ordinary exit does.
    const uint32_t program[] = {
        0x8B010000U, // ADD X0, X0, X1
        0x1E212800U, // FADD S0, S0, S1
        0xF1000421U, // SUBS X1, X1, #1
        0x54FFFFA1U, // B.NE back to the ADD
    };
    load_program(program, 4U);
    g_cpu.x[1]    = 1000U;
    g_cpu.v[1][0] = 0x3F800000U; // 1.0

    TEST_ASSERT_EQUAL(BAL_SUCCESS, run());
    TEST_ASSERT_EQUAL_UINT64(500500U, g_cpu.x[0]);
    TEST_ASSERT_EQUAL_HEX64(0x447A0000U, g_cpu.v[0][0]); // 1000.0
    TEST_ASSERT_EQUAL_HEX64(0U, g_cpu.fpsr);
}

static void
test_IndirectBranches_ReachTheirTargets(void)
{
    // A loop that calls a function with BL and RET and branches through a two-entry jump table
    // with BR, so the indirect branches look up changing targets.
    static const uint32_t program[] = {
        0x94000010U, // 0x00: BL 0x40
        0x92400026U, // 0x04: AND X6, X1, #1
        0x8B060D07U, // 0x08: ADD X7, X8, X6, LSL #3
        0xD61F00E0U, // 0x0C: BR X7
        0U,          0U, 0U, 0U,
        0x91000442U, // 0x20: ADD X2, X2, #1
        0x14000003U, // 0x24: B 0x30
        0x91000463U, // 0x28: ADD X3, X3, #1
        0x14000001U, // 0x2C: B 0x30
        0xF1000421U, // 0x30: SUBS X1, X1, #1
        0x54FFFE61U, // 0x34: B.NE 0x00
        0xAA1403FEU, // 0x38: MOV X30, X20
        RET,         // 0x3C
        0x91000400U, // 0x40: ADD X0, X0, #1
        RET,         // 0x44
    };
    (void)memcpy(g_memory, program, sizeof(program));
    g_cpu.x[1]  = 1000U;
    g_cpu.x[8]  = 0x20U;
    g_cpu.x[20] = BAL_ENGINE_SENTINEL;

    TEST_ASSERT_EQUAL(BAL_SUCCESS, run());
    TEST_ASSERT_EQUAL_UINT64(1000U, g_cpu.x[0]);
    TEST_ASSERT_EQUAL_UINT64(500U, g_cpu.x[2]);
    TEST_ASSERT_EQUAL_UINT64(500U, g_cpu.x[3]);
}

static void
test_Reset_RetranslatesTheGuestCode(void)
{
    // MOVZ X0, #1, run once; then MOVZ X0, #2 in its place after bal_engine_reset().
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
    TEST_ASSERT_EQUAL(BAL_SUCCESS, bal_engine_reset(&engine));
    g_cpu.pc    = 0U;
    g_cpu.x[30] = BAL_ENGINE_SENTINEL;
    TEST_ASSERT_EQUAL(BAL_SUCCESS, bal_engine_run_thread(&engine));
    TEST_ASSERT_EQUAL_UINT64(2U, g_cpu.x[0]);
    bal_engine_destroy(&engine);
}

/// Guest page 0x10000 maps to g_page_a or, once g_use_page_b is set, to g_page_b; everything else
/// goes to the flat test memory.
static uint8_t                g_page_a[4096];
static uint8_t                g_page_b[4096];
static bool                   g_use_page_b;
static bal_memory_interface_t g_switching_interface;
static bal_engine_t           g_switching_engine;

static const uint8_t *
switching_translate(void *context, const bal_guest_address_t address, size_t *available)
{
    (void)context;

    if (address >= 0x10000U && address < 0x11000U)
    {
        *available = 0x11000U - (size_t)address;
        return (true == g_use_page_b ? g_page_b : g_page_a) + (address - 0x10000U);
    }

    return g_memory_interface.translate(&g_memory_interface, address, available);
}

static void
set_up_switching_memory(void)
{
    const uint64_t a = 0x1111U;
    const uint64_t b = 0x2222U;
    (void)memcpy(g_page_a + 8U, &a, sizeof(a));
    (void)memcpy(g_page_b + 8U, &b, sizeof(b));
    g_use_page_b                    = false;
    g_switching_interface.context   = &g_memory_interface;
    g_switching_interface.translate = switching_translate;
    (void)memset(&g_switching_engine, 0, sizeof(g_switching_engine));
    TEST_ASSERT_EQUAL(
        BAL_SUCCESS,
        bal_engine_init(&g_switching_engine, &g_cpu, &g_allocator, &g_switching_interface));
}

static void
test_MemoryMapChange_IsSeenByTheNextRun(void)
{
    // LDR X0, [X1], run with the page mapped to A and then to B.
    const uint32_t program[] = { 0xF9400020U };
    load_program(program, 1U);
    set_up_switching_memory();
    g_cpu.x[1] = 0x10008U;

    TEST_ASSERT_EQUAL(BAL_SUCCESS, bal_engine_run_thread(&g_switching_engine));
    TEST_ASSERT_EQUAL_HEX64(0x1111U, g_cpu.x[0]);

    g_use_page_b = true;
    g_cpu.pc     = 0U;
    g_cpu.x[30]  = BAL_ENGINE_SENTINEL;
    TEST_ASSERT_EQUAL(BAL_SUCCESS, bal_engine_run_thread(&g_switching_engine));
    TEST_ASSERT_EQUAL_HEX64(0x2222U, g_cpu.x[0]);
    bal_engine_destroy(&g_switching_engine);
}

/// Remaps the switching page while the engine runs.
static uint64_t
remap_while_running(void *context)
{
    (void)context;
    g_use_page_b = true;
    bal_engine_invalidate_memory(&g_switching_engine);
    return 0U;
}

static void
test_InvalidateMemory_TakesEffectAtTheNextBlock(void)
{
    const uint32_t program[] = {
        0xF9400020U, // LDR X0, [X1]
        0xD53BE025U, // MRS X5, CNTPCT_EL0, which remaps the page
        0x14000001U, // B to the next instruction, ending the block
        0xF9400022U, // LDR X2, [X1]
    };
    load_program(program, 4U);
    set_up_switching_memory();
    g_cpu.x[1]         = 0x10008U;
    g_cpu.read_counter = remap_while_running;

    TEST_ASSERT_EQUAL(BAL_SUCCESS, bal_engine_run_thread(&g_switching_engine));
    TEST_ASSERT_EQUAL_HEX64(0x1111U, g_cpu.x[0]);
    TEST_ASSERT_EQUAL_HEX64(0x2222U, g_cpu.x[2]);
    bal_engine_destroy(&g_switching_engine);
}

static void
test_DataAbort_WritesTheRegistersBack(void)
{
    // MOVZ X3, #7 stays in a host register until LDR X0, [X1] faults.
    const uint32_t program[] = { 0xD28000E3U, 0xF9400020U };
    load_program(program, 2U);
    g_cpu.x[1] = 0x200000U;

    TEST_ASSERT_EQUAL(BAL_ERROR_MEMORY_FAULT, run());
    TEST_ASSERT_EQUAL_UINT64(7U, g_cpu.x[3]);
    TEST_ASSERT_EQUAL_HEX64(4U, g_cpu.pc);
}

static void
test_FloatingPointFlags_FollowChainedBlocks(void)
{
    // Flags raised in one block are read by MRS FPSR in a later one, and MSR FPSR clears the ones
    // still pending. The outer loop runs three times, so the later passes go through chained
    // jumps.
    const uint32_t program[] = {
        0x1E241862U, // 0x00: FDIV S2, S3, S4, dividing by zero
        0x14000001U, // 0x04: B 0x08
        0x1E2708C5U, // 0x08: FMUL S5, S6, S7, overflowing
        0x14000001U, // 0x0C: B 0x10
        0xD53B4429U, // 0x10: MRS X9, FPSR
        0xD51B443FU, // 0x14: MSR FPSR, XZR
        0x14000001U, // 0x18: B 0x1C
        0x1E2A2928U, // 0x1C: FADD S8, S9, S10, exact
        0xD53B442AU, // 0x20: MRS X10, FPSR
        0xF1000694U, // 0x24: SUBS X20, X20, #1
        0x54FFFEC1U, // 0x28: B.NE 0x00
    };
    load_program(program, 11U);
    g_cpu.x[20]   = 3U;
    g_cpu.v[3][0] = 0x3F800000U; // 1.0
    g_cpu.v[6][0] = 0x7149F2CAU; // 1e30
    g_cpu.v[7][0] = 0x7149F2CAU;

    TEST_ASSERT_EQUAL(BAL_SUCCESS, run());
    TEST_ASSERT_EQUAL_HEX64(0x16U, g_cpu.x[9]); // DZC, OFC and IXC
    TEST_ASSERT_EQUAL_HEX64(0U, g_cpu.x[10]);
    TEST_ASSERT_EQUAL_HEX64(0U, g_cpu.fpsr);
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

static void
test_PacMemo_SigningAgainGivesTheSameCode(void)
{
    // PACIASP; MOV X1, X30; AUTIASP; PACIASP; MOV X2, X30; AUTIASP. The second signing and
    // authentication come from the memo the first ones filled.
    const uint32_t program[]
        = { 0xD503233FU, 0xAA1E03E1U, 0xD50323BFU, 0xD503233FU, 0xAA1E03E2U, 0xD50323BFU };
    load_program(program, 6U);
    enable_instruction_key_a();
    g_cpu.x[31] = 0x3F00U;

    TEST_ASSERT_EQUAL(BAL_SUCCESS, run());
    TEST_ASSERT_NOT_EQUAL(BAL_ENGINE_SENTINEL, g_cpu.x[1]);
    TEST_ASSERT_EQUAL_HEX64(g_cpu.x[1], g_cpu.x[2]);
    TEST_ASSERT_EQUAL_HEX64(BAL_ENGINE_SENTINEL, g_cpu.x[30]);
}

static void
test_PacMemo_FollowsKeyChangesBetweenRuns(void)
{
    // PACIASP; MOV X1, X30; AUTIASP, run twice in one engine with a new key in between; the
    // second run must sign with the new key, as a fresh engine does.
    const uint32_t program[] = { 0xD503233FU, 0xAA1E03E1U, 0xD50323BFU };
    load_program(program, 3U);
    enable_instruction_key_a();
    g_cpu.x[31] = 0x3F00U;

    bal_engine_t engine;
    (void)memset(&engine, 0, sizeof(engine));
    TEST_ASSERT_EQUAL(BAL_SUCCESS,
                      bal_engine_init(&engine, &g_cpu, &g_allocator, &g_memory_interface));
    TEST_ASSERT_EQUAL(BAL_SUCCESS, bal_engine_run_thread(&engine));
    const uint64_t first_code = g_cpu.x[1];

    g_cpu.pc                          = 0U;
    g_cpu.x[30]                       = BAL_ENGINE_SENTINEL;
    g_cpu.pac_keys[BAL_PAC_KEY_IA][0] = 0x3333U;
    TEST_ASSERT_EQUAL(BAL_SUCCESS, bal_engine_run_thread(&engine));
    const uint64_t second_code = g_cpu.x[1];
    bal_engine_destroy(&engine);

    g_cpu.pc    = 0U;
    g_cpu.x[30] = BAL_ENGINE_SENTINEL;
    TEST_ASSERT_EQUAL(BAL_SUCCESS, run());
    TEST_ASSERT_NOT_EQUAL(first_code, second_code);
    TEST_ASSERT_EQUAL_HEX64(g_cpu.x[1], second_code);
    TEST_ASSERT_EQUAL_HEX64(BAL_ENGINE_SENTINEL, g_cpu.x[30]);
}

/// Runs one instruction with X1 and X2 set, returning X0, which starts as all ones.
static uint64_t
run_one(const uint32_t instruction, const uint64_t x1, const uint64_t x2)
{
    load_program(&instruction, 1U);
    g_cpu.pc    = 0U;
    g_cpu.x[30] = BAL_ENGINE_SENTINEL;
    g_cpu.x[0]  = ~0ULL;
    g_cpu.x[1]  = x1;
    g_cpu.x[2]  = x2;
    TEST_ASSERT_EQUAL(BAL_SUCCESS, run());
    return g_cpu.x[0];
}

static void
test_Cssc_AbsCntCtz(void)
{
    TEST_ASSERT_EQUAL_HEX64(5U, run_one(0xDAC02020U, (uint64_t)-5, 0U));             // ABS X0, X1
    TEST_ASSERT_EQUAL_HEX64(0x80000000U, run_one(0x5AC02020U, 0xFF80000000ULL, 0U)); // ABS W0, W1
    TEST_ASSERT_EQUAL_HEX64(7U, run_one(0x5AC02020U, 0xFFFFFFF9U, 0U));              // ABS W0, W1
    TEST_ASSERT_EQUAL_HEX64(9U, run_one(0xDAC01C20U, 0x800000000000F0F0ULL, 0U));    // CNT X0, X1
    TEST_ASSERT_EQUAL_HEX64(0U, run_one(0x5AC01C20U, 0xFFFFFFFF00000000ULL, 0U));    // CNT W0, W1
    TEST_ASSERT_EQUAL_HEX64(32U, run_one(0x5AC01820U, 0xFF00000000ULL, 0U));         // CTZ W0, W1
    TEST_ASSERT_EQUAL_HEX64(8U, run_one(0xDAC01820U, 0x100U, 0U));                   // CTZ X0, X1
    TEST_ASSERT_EQUAL_HEX64(64U, run_one(0xDAC01820U, 0U, 0U));                      // CTZ X0, X1
}

static void
test_Cssc_MinimumMaximum(void)
{
    // SMAX X0, X1, #-3 and UMIN W0, W1, #200.
    TEST_ASSERT_EQUAL_HEX64((uint64_t)-3, run_one(0x91C3F420U, (uint64_t)-10, 0U));
    TEST_ASSERT_EQUAL_HEX64(200U, run_one(0x11CF2020U, 0xFFFFFFFFU, 0U));
    // SMAX W0, W1, #-3 on a negative W1 compares the low 32 bits as signed.
    TEST_ASSERT_EQUAL_HEX64(0xFFFFFFFDU, run_one(0x11C3F420U, 0xFFFFFFF0U, 0U));
    // SMIN X0, X1, X2 and UMAX W0, W1, W2.
    TEST_ASSERT_EQUAL_HEX64((uint64_t)-7, run_one(0x9AC26820U, 4U, (uint64_t)-7));
    TEST_ASSERT_EQUAL_HEX64(0x80000000U, run_one(0x1AC26420U, 0x1234U, 0x7780000000ULL));
}

static void
test_Rcpc3_WritebackFormsRoundTrip(void)
{
    // STLR W1, [X2, #-4]!; LDAPR W3, [X2], #4; STILP X1, X4, [X2, #-16]!;
    // LDIAPP X5, X6, [X2], #16.
    const uint32_t program[] = { 0x99800841U, 0x99C00843U, 0xD9040841U, 0xD9460845U };
    load_program(program, 4U);
    g_cpu.x[1] = 0x1122334455667788ULL;
    g_cpu.x[2] = 0x2010U;
    g_cpu.x[4] = 0x99AABBCCDDEEFF00ULL;

    TEST_ASSERT_EQUAL(BAL_SUCCESS, run());
    TEST_ASSERT_EQUAL_HEX64(0x55667788U, g_cpu.x[3]);
    TEST_ASSERT_EQUAL_HEX64(0x1122334455667788ULL, g_cpu.x[5]);
    TEST_ASSERT_EQUAL_HEX64(0x99AABBCCDDEEFF00ULL, g_cpu.x[6]);
    TEST_ASSERT_EQUAL_HEX64(0x2010U, g_cpu.x[2]);

    uint64_t pair[2];
    (void)memcpy(pair, g_memory + 0x2000U, sizeof(pair));
    TEST_ASSERT_EQUAL_HEX64(0x1122334455667788ULL, pair[0]);
    TEST_ASSERT_EQUAL_HEX64(0x99AABBCCDDEEFF00ULL, pair[1]);
}

static void
test_Rcpc3_VectorFormsRoundTrip(void)
{
    // STLUR Q0, [X2, #-16]; LDAPUR Q1, [X2, #-16]; LDAPUR S2, [X2, #-16];
    // STL1 {V0.D}[1], [X2]; LDAP1 {V3.D}[0], [X2].
    const uint32_t program[] = { 0x1D9F0840U, 0x1DDF0841U, 0x9D5F0842U, 0x4D018440U, 0x0D418443U };
    load_program(program, 5U);
    g_cpu.x[2]    = 0x2010U;
    g_cpu.v[0][0] = 0x0123456789ABCDEFULL;
    g_cpu.v[0][1] = 0xFEDCBA9876543210ULL;
    g_cpu.v[2][1] = ~0ULL;
    g_cpu.v[3][1] = 0x5555U;

    TEST_ASSERT_EQUAL(BAL_SUCCESS, run());
    TEST_ASSERT_EQUAL_HEX64(0x0123456789ABCDEFULL, g_cpu.v[1][0]);
    TEST_ASSERT_EQUAL_HEX64(0xFEDCBA9876543210ULL, g_cpu.v[1][1]);
    TEST_ASSERT_EQUAL_HEX64(0x89ABCDEFU, g_cpu.v[2][0]);
    TEST_ASSERT_EQUAL_HEX64(0U, g_cpu.v[2][1]);
    TEST_ASSERT_EQUAL_HEX64(0xFEDCBA9876543210ULL, g_cpu.v[3][0]);
    TEST_ASSERT_EQUAL_HEX64(0x5555U, g_cpu.v[3][1]);
}

static void
test_Lse128_OperationsReturnTheOldPair(void)
{
    // SWPP X1, X2, [X10]; LDSETP X3, X4, [X10]; LDCLRPAL X5, X6, [X10].
    const uint32_t program[] = { 0x19228141U, 0x19243143U, 0x19E61145U };
    load_program(program, 3U);
    const uint64_t initial[2] = { 0x1111111111111111ULL, 0x2222222222222222ULL };
    (void)memcpy(g_memory + 0x2000U, initial, sizeof(initial));
    g_cpu.x[10] = 0x2000U;
    g_cpu.x[1]  = 0x00000000000000F0ULL;
    g_cpu.x[2]  = 0x0F00000000000000ULL;
    g_cpu.x[3]  = 0x000000000000000FULL;
    g_cpu.x[4]  = 0x00F0000000000000ULL;
    g_cpu.x[5]  = 0x00000000000000F0ULL;
    g_cpu.x[6]  = 0x0F00000000000000ULL;

    TEST_ASSERT_EQUAL(BAL_SUCCESS, run());
    TEST_ASSERT_EQUAL_HEX64(0x1111111111111111ULL, g_cpu.x[1]);
    TEST_ASSERT_EQUAL_HEX64(0x2222222222222222ULL, g_cpu.x[2]);
    TEST_ASSERT_EQUAL_HEX64(0xF0U, g_cpu.x[3]);
    TEST_ASSERT_EQUAL_HEX64(0x0F00000000000000ULL, g_cpu.x[4]);
    TEST_ASSERT_EQUAL_HEX64(0xFFU, g_cpu.x[5]);
    TEST_ASSERT_EQUAL_HEX64(0x0FF0000000000000ULL, g_cpu.x[6]);

    uint64_t final[2];
    (void)memcpy(final, g_memory + 0x2000U, sizeof(final));
    TEST_ASSERT_EQUAL_HEX64(0x0FU, final[0]);
    TEST_ASSERT_EQUAL_HEX64(0x00F0000000000000ULL, final[1]);
}

static void
test_Lse128_MisalignedAddressRaisesDataAbort(void)
{
    // SWPP X1, X2, [X10] at an address that is only 8-byte aligned.
    const uint32_t program[] = { 0x19228141U };
    load_program(program, 1U);
    g_cpu.x[10] = 0x2008U;

    TEST_ASSERT_NOT_EQUAL(BAL_SUCCESS, run());
    TEST_ASSERT_EQUAL(BAL_EXCEPTION_DATA_ABORT, g_cpu.exception);
}

static void
test_Mops_ForwardCopyCopiesTheRange(void)
{
    // CPYFP [X0]!, [X1]!, X2!; CPYFM; CPYFE.
    const uint32_t program[] = { 0x19010440U, 0x19410440U, 0x19810440U };
    load_program(program, 3U);

    for (uint32_t i = 0U; i < 64U; ++i)
    {
        g_memory[0x2000U + i] = (uint8_t)(i + 1U);
    }

    g_cpu.x[0] = 0x3000U;
    g_cpu.x[1] = 0x2000U;
    g_cpu.x[2] = 37U;

    TEST_ASSERT_EQUAL(BAL_SUCCESS, run());
    TEST_ASSERT_EQUAL_MEMORY(g_memory + 0x2000U, g_memory + 0x3000U, 37U);
    TEST_ASSERT_EQUAL_UINT8(0U, g_memory[0x3000U + 37U]);
    TEST_ASSERT_EQUAL_HEX64(0x3025U, g_cpu.x[0]);
    TEST_ASSERT_EQUAL_HEX64(0x2025U, g_cpu.x[1]);
    TEST_ASSERT_EQUAL_HEX64(0U, g_cpu.x[2]);
    TEST_ASSERT_EQUAL_UINT8(1U, g_cpu.flag_c);
    TEST_ASSERT_EQUAL_UINT8(0U, g_cpu.flag_n);
}

static void
test_Mops_OverlappingCopyGoesBackwards(void)
{
    // CPYP [X0]!, [X1]!, X2!; CPYM; CPYE, with the destination 8 bytes above the source.
    const uint32_t program[] = { 0x1D010440U, 0x1D410440U, 0x1D810440U };
    load_program(program, 3U);
    uint8_t expected[40];

    for (uint32_t i = 0U; i < 40U; ++i)
    {
        g_memory[0x2000U + i] = (uint8_t)(i + 1U);
        expected[i]           = (uint8_t)(i < 8U ? i + 1U : i - 7U);
    }

    g_cpu.x[0] = 0x2008U;
    g_cpu.x[1] = 0x2000U;
    g_cpu.x[2] = 32U;

    TEST_ASSERT_EQUAL(BAL_SUCCESS, run());
    TEST_ASSERT_EQUAL_MEMORY(expected, g_memory + 0x2000U, sizeof(expected));
    // A backward copy leaves the registers at the start of the ranges and sets N.
    TEST_ASSERT_EQUAL_HEX64(0x2008U, g_cpu.x[0]);
    TEST_ASSERT_EQUAL_HEX64(0x2000U, g_cpu.x[1]);
    TEST_ASSERT_EQUAL_HEX64(0U, g_cpu.x[2]);
    TEST_ASSERT_EQUAL_UINT8(1U, g_cpu.flag_n);
    TEST_ASSERT_EQUAL_UINT8(1U, g_cpu.flag_c);
}

static void
test_Mops_SetFillsTheRange(void)
{
    // SETP [X0]!, X2!, X3; SETM; SETE.
    const uint32_t program[] = { 0x19C30440U, 0x19C34440U, 0x19C38440U };
    load_program(program, 3U);
    g_cpu.x[0] = 0x3000U;
    g_cpu.x[2] = 50U;
    g_cpu.x[3] = 0x12ABU;

    TEST_ASSERT_EQUAL(BAL_SUCCESS, run());

    for (uint32_t i = 0U; i < 50U; ++i)
    {
        TEST_ASSERT_EQUAL_HEX8(0xABU, g_memory[0x3000U + i]);
    }

    TEST_ASSERT_EQUAL_HEX8(0U, g_memory[0x3000U + 50U]);
    TEST_ASSERT_EQUAL_HEX64(0x3032U, g_cpu.x[0]);
    TEST_ASSERT_EQUAL_HEX64(0U, g_cpu.x[2]);
}

static void
test_Mops_UnmappedSetFaultsWithoutWriting(void)
{
    // SETP [X0]!, X2!, X3 running past the end of guest memory.
    const uint32_t program[] = { 0x19C30440U };
    load_program(program, 1U);
    g_cpu.x[0] = GUEST_MEMORY_SIZE - 0x100U;
    g_cpu.x[2] = 0x200U;
    g_cpu.x[3] = 0xABU;

    TEST_ASSERT_NOT_EQUAL(BAL_SUCCESS, run());
    TEST_ASSERT_EQUAL(BAL_EXCEPTION_DATA_ABORT, g_cpu.exception);
    TEST_ASSERT_EQUAL_HEX8(0U, g_memory[GUEST_MEMORY_SIZE - 0x100U]);
    TEST_ASSERT_EQUAL_HEX64(GUEST_MEMORY_SIZE - 0x100U, g_cpu.x[0]);
    TEST_ASSERT_EQUAL_HEX64(0x200U, g_cpu.x[2]);
}

static void
test_Rcw_OperationsSetTheFlags(void)
{
    // RCWSET X1, X2, [X10]; RCWCAS X3, X4, [X10] with a matching then a stale X3.
    const uint32_t program[] = { 0x3821B142U, 0x19230944U, 0x19250946U };
    load_program(program, 3U);
    const uint64_t initial = 0x10U;
    (void)memcpy(g_memory + 0x2000U, &initial, sizeof(initial));
    g_cpu.x[10] = 0x2000U;
    g_cpu.x[1]  = 0x0FU;
    g_cpu.x[3]  = 0x1FU;
    g_cpu.x[4]  = 0x77U;
    g_cpu.x[5]  = 0x1FU;
    g_cpu.x[6]  = 0x99U;

    TEST_ASSERT_EQUAL(BAL_SUCCESS, run());
    TEST_ASSERT_EQUAL_HEX64(0x10U, g_cpu.x[2]);
    TEST_ASSERT_EQUAL_HEX64(0x1FU, g_cpu.x[3]);
    // The second compare-and-swap sees 0x77 and fails, setting N and leaving memory alone.
    TEST_ASSERT_EQUAL_HEX64(0x77U, g_cpu.x[5]);
    TEST_ASSERT_EQUAL_UINT8(1U, g_cpu.flag_n);
    TEST_ASSERT_EQUAL_UINT8(1U, g_cpu.flag_c);
    TEST_ASSERT_EQUAL_UINT8(0U, g_cpu.flag_z);

    uint64_t final = 0U;
    (void)memcpy(&final, g_memory + 0x2000U, sizeof(final));
    TEST_ASSERT_EQUAL_HEX64(0x77U, final);
}

static void
test_Rcw_SoftChecksProtectBitZeroAndTheMask(void)
{
    // RCWSSET X1, X2, [X10]; RCWSSWP X3, X4, [X10]; RCWSCLR X5, X6, [X10].
    const uint32_t program[] = { 0x7821B142U, 0x7823A144U, 0x78259146U };
    load_program(program, 3U);
    const uint64_t initial = 0x101U;
    (void)memcpy(g_memory + 0x2000U, &initial, sizeof(initial));
    g_cpu.x[10]        = 0x2000U;
    g_cpu.rcwsmask_el1 = 0xF0U;
    g_cpu.x[1]         = 0x30U;   // Inside the mask: allowed.
    g_cpu.x[3]         = 0x1331U; // Changes bit 12, outside the mask: refused.
    g_cpu.x[5]         = 0x1U;    // Clears bit 0: refused.

    TEST_ASSERT_EQUAL(BAL_SUCCESS, run());
    TEST_ASSERT_EQUAL_HEX64(0x101U, g_cpu.x[2]);
    TEST_ASSERT_EQUAL_HEX64(0x131U, g_cpu.x[4]);
    TEST_ASSERT_EQUAL_HEX64(0x131U, g_cpu.x[6]);
    TEST_ASSERT_EQUAL_UINT8(0U, g_cpu.flag_c);

    uint64_t final = 0U;
    (void)memcpy(&final, g_memory + 0x2000U, sizeof(final));
    TEST_ASSERT_EQUAL_HEX64(0x131U, final);
}

static void
test_Mte_RegisterFormsClearTheTag(void)
{
    // ADDG X1, X2, #32, #3; SUBG SP, SP, #16, #0; IRG X3, X2; GMI X4, X2, X5; SUBPS X6, X7, X8.
    const uint32_t program[] = { 0x91820C41U, 0xD18103FFU, 0x9ADF1043U, 0x9AC51444U, 0xBAC800E6U };
    load_program(program, 5U);
    g_cpu.x[2]  = 0x0500000000001000ULL;
    g_cpu.x[5]  = 0x1U;
    g_cpu.x[7]  = 0xFF00000000000010ULL;
    g_cpu.x[8]  = 0x20U;
    g_cpu.x[31] = 0x3F00U;

    TEST_ASSERT_EQUAL(BAL_SUCCESS, run());
    // Allocation tag access is disabled at EL0, so the generated tags are zero.
    TEST_ASSERT_EQUAL_HEX64(0x1020U, g_cpu.x[1]);
    TEST_ASSERT_EQUAL_HEX64(0x3EF0U, g_cpu.x[31]);
    TEST_ASSERT_EQUAL_HEX64(0x1000U, g_cpu.x[3]);
    TEST_ASSERT_EQUAL_HEX64(0x21U, g_cpu.x[4]);
    // SUBPS subtracts the sign-extended 56-bit addresses: 0x10 - 0x20.
    TEST_ASSERT_EQUAL_HEX64(0xFFFFFFFFFFFFFFF0ULL, g_cpu.x[6]);
    TEST_ASSERT_EQUAL_UINT8(1U, g_cpu.flag_n);
    TEST_ASSERT_EQUAL_UINT8(0U, g_cpu.flag_c);
}

static void
test_Mte_TagStoresKeepTheirDataEffects(void)
{
    // STZG X1, [X2, #16]!; STZ2G X1, [X3], #32; STGP X4, X5, [X6, #16]; LDG X7, [X2].
    const uint32_t program[] = { 0xD9601C41U, 0xD9E02461U, 0x690094C4U, 0xD9600047U };
    load_program(program, 4U);
    (void)memset(g_memory + 0x2000U, 0xFF, 0x300U);
    g_cpu.x[2] = 0x2000U;
    g_cpu.x[3] = 0x2100U;
    g_cpu.x[4] = 0x1111111111111111ULL;
    g_cpu.x[5] = 0x2222222222222222ULL;
    g_cpu.x[6] = 0x2200U;
    g_cpu.x[7] = 0x0A00000000004000ULL;

    TEST_ASSERT_EQUAL(BAL_SUCCESS, run());
    TEST_ASSERT_EQUAL_HEX64(0x2010U, g_cpu.x[2]);
    TEST_ASSERT_EQUAL_HEX64(0x2120U, g_cpu.x[3]);
    TEST_ASSERT_EQUAL_HEX64(0x2200U, g_cpu.x[6]);
    TEST_ASSERT_EQUAL_HEX64(0x4000U, g_cpu.x[7]);
    TEST_ASSERT_EQUAL_HEX8(0xFFU, g_memory[0x200FU]);
    TEST_ASSERT_EQUAL_HEX8(0U, g_memory[0x2010U]);
    TEST_ASSERT_EQUAL_HEX8(0U, g_memory[0x201FU]);
    TEST_ASSERT_EQUAL_HEX8(0xFFU, g_memory[0x2020U]);
    TEST_ASSERT_EQUAL_HEX8(0U, g_memory[0x2100U]);
    TEST_ASSERT_EQUAL_HEX8(0U, g_memory[0x211FU]);
    TEST_ASSERT_EQUAL_HEX8(0xFFU, g_memory[0x2120U]);

    uint64_t pair[2];
    (void)memcpy(pair, g_memory + 0x2210U, sizeof(pair));
    TEST_ASSERT_EQUAL_HEX64(0x1111111111111111ULL, pair[0]);
    TEST_ASSERT_EQUAL_HEX64(0x2222222222222222ULL, pair[1]);
}

static void
test_Mte_UnalignedTagStoreRaisesDataAbort(void)
{
    // STG X1, [X8] at an address that is not 16-byte aligned.
    const uint32_t program[] = { 0xD9200901U };
    load_program(program, 1U);
    g_cpu.x[8] = 0x2208U;

    TEST_ASSERT_NOT_EQUAL(BAL_SUCCESS, run());
    TEST_ASSERT_EQUAL(BAL_EXCEPTION_DATA_ABORT, g_cpu.exception);
}

static void
test_Mte_SetgFillsWholeGranules(void)
{
    // SETGP [X0]!, X2!, X3; SETGM; SETGE, then again with a size that is not a granule multiple.
    const uint32_t program[] = { 0x1DC30440U, 0x1DC34440U, 0x1DC38440U };
    load_program(program, 3U);
    g_cpu.x[0] = 0x3000U;
    g_cpu.x[2] = 48U;
    g_cpu.x[3] = 0x5AU;

    TEST_ASSERT_EQUAL(BAL_SUCCESS, run());
    TEST_ASSERT_EQUAL_HEX8(0x5AU, g_memory[0x3000U]);
    TEST_ASSERT_EQUAL_HEX8(0x5AU, g_memory[0x302FU]);
    TEST_ASSERT_EQUAL_HEX8(0U, g_memory[0x3030U]);
    TEST_ASSERT_EQUAL_HEX64(0x3030U, g_cpu.x[0]);

    g_cpu.pc    = 0U;
    g_cpu.x[30] = BAL_ENGINE_SENTINEL;
    g_cpu.x[0]  = 0x3100U;
    g_cpu.x[2]  = 40U;
    TEST_ASSERT_NOT_EQUAL(BAL_SUCCESS, run());
    TEST_ASSERT_EQUAL(BAL_EXCEPTION_DATA_ABORT, g_cpu.exception);
    TEST_ASSERT_EQUAL_HEX8(0U, g_memory[0x3100U]);
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
    RUN_TEST(test_PacMemo_SigningAgainGivesTheSameCode);
    RUN_TEST(test_PacMemo_FollowsKeyChangesBetweenRuns);
    RUN_TEST(test_Cssc_AbsCntCtz);
    RUN_TEST(test_Cssc_MinimumMaximum);
    RUN_TEST(test_Rcpc3_WritebackFormsRoundTrip);
    RUN_TEST(test_Rcpc3_VectorFormsRoundTrip);
    RUN_TEST(test_Lse128_OperationsReturnTheOldPair);
    RUN_TEST(test_Lse128_MisalignedAddressRaisesDataAbort);
    RUN_TEST(test_Mops_ForwardCopyCopiesTheRange);
    RUN_TEST(test_Mops_OverlappingCopyGoesBackwards);
    RUN_TEST(test_Mops_SetFillsTheRange);
    RUN_TEST(test_Mops_UnmappedSetFaultsWithoutWriting);
    RUN_TEST(test_Rcw_OperationsSetTheFlags);
    RUN_TEST(test_Rcw_SoftChecksProtectBitZeroAndTheMask);
    RUN_TEST(test_Mte_RegisterFormsClearTheTag);
    RUN_TEST(test_Mte_TagStoresKeepTheirDataEffects);
    RUN_TEST(test_Mte_UnalignedTagStoreRaisesDataAbort);
    RUN_TEST(test_Mte_SetgFillsWholeGranules);
    RUN_TEST(test_IcIvauAndIsb_RunTheCodeTheGuestWrote);
    RUN_TEST(test_ClearCache_RetranslatesTheGuestCode);
    RUN_TEST(test_UnsupportedInstruction_RaisesWhenReached);
    RUN_TEST(test_FetchPastMappedMemory_FaultsAtTheUnmappedAddress);
    RUN_TEST(test_DitAndSsbs_ReadBackWhatWasWritten);
    RUN_TEST(test_StopThread_LeavesChainedBlocks);
    RUN_TEST(test_ChainedLoop_ComputesTheSameResult);
    RUN_TEST(test_IndirectBranches_ReachTheirTargets);
    RUN_TEST(test_Reset_RetranslatesTheGuestCode);
    RUN_TEST(test_MemoryMapChange_IsSeenByTheNextRun);
    RUN_TEST(test_InvalidateMemory_TakesEffectAtTheNextBlock);
    RUN_TEST(test_DataAbort_WritesTheRegistersBack);
    RUN_TEST(test_FloatingPointFlags_FollowChainedBlocks);
    return UNITY_END();
}
