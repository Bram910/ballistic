//! SVE differential tester. It runs random encodings of every SVE decoder table entry through
//! Ballistic and through the AArch64 reference (sve_reference.c) under qemu-aarch64 with 128-bit
//! vectors, and compares the general purpose registers, SP, NZCV, the FPSR flags, the Z and P
//! registers, FFR, a data area and whether the instruction faulted.
//!
//! Usage: bal_sve_diff REFERENCE [SAMPLES] [MNEMONICS] [REPORTS]
//!
//! REFERENCE is the built reference program, run with $BAL_SVE_QEMU (qemu-aarch64-static by
//! default). With $BAL_SVE_SPACE set to `sme` it tests the SME encoding space, and the SMSTART,
//! SMSTOP, SVCR and TPIDR2_EL0 forms of MSR and MRS, with random streaming mode, ZA and ZT0
//! state, which needs a qemu with SME2 (10.1 or later). With `all`, it tests every decoder entry
//! MNEMONICS names, which is how instructions newer than Unicorn are checked. $BAL_SVE_SEED, when
//! set, changes the random encodings and states. MNEMONICS is an optional comma separated list of
//! entries to test, and REPORTS the number of mismatches to describe. The output has one CSV line
//! per entry, in the format of the status tracker's coverage tool:
//!
//!     index,mnemonic,mask,expected,status,translated,matched,samples
//!
//! where status is `verified` (every compared sample matched), `wrong`, `todo` (Ballistic rejects
//! the encodings qemu runs) or `undefined` (both reject them). Mismatch reports start with `#`.
// mkdtemp() and the other POSIX functions.
#define _DEFAULT_SOURCE

#include "bal_decoder.h"
#include "bal_engine.h"
#include "bal_engine_flags.h"
#include "bal_log.h"
#include "generated/decoder_table.h"
#include "sve_test.h"
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

// Only the data area and one code page are mapped, as in the reference, so faults agree.
#define CODE_ADDRESS 0x10000000ULL

#define STATUS_SIGILL  4U
#define STATUS_SIGBUS  7U
#define STATUS_SIGSEGV 11U
/// The reference crashed (a qemu bug) on this test.
#define STATUS_CRASHED 255U

static bool     g_is_sme;
static bool     g_is_all;
static uint8_t  g_memory[SVE_DATA_SIZE];
static uint8_t  g_code[0x1000];
static void    *g_interface;
static uint64_t g_random = 0x2545F4914F6CDD1DULL;

static uint64_t
next_random(void)
{
    return sve_splitmix(&g_random);
}

/// Half of the registers hold addresses inside the data area, mostly 16-byte aligned.
static uint64_t
random_register(void)
{
    switch (next_random() % 4U)
    {
        case 0:
        case 1:
            return SVE_DATA_BASE + 0x1000U + (next_random() % 0x2000U & ~0xFULL);
        case 2:
            return next_random() % 64U;
        default:
            return next_random();
    }
}

static void
random_state(sve_state_t *state)
{
    memset(state, 0, sizeof(*state));

    for (uint32_t r = 0U; r < 31U; ++r)
    {
        state->x[r] = random_register();
    }

    state->sp   = SVE_DATA_BASE + 0x2000U;
    state->nzcv = (next_random() & 0xFU) << 28U;

    // Mostly the default FPCR; sometimes another rounding mode, flush-to-zero or default NaN.
    if (0U == next_random() % 4U)
    {
        state->fpcr = (next_random() & 0x3U) << 22U | (next_random() & 0x3U) << 24U;
    }

    for (uint32_t r = 0U; r < 32U; ++r)
    {
        state->v[r][0] = next_random();
        state->v[r][1] = next_random();
    }

    for (uint32_t r = 0U; r < 16U; ++r)
    {
        // All true, all false, or random.
        const uint64_t kind = next_random() % 4U;
        state->p[r]         = 0U == kind ? 0xFFFFU : (1U == kind ? 0U : (uint16_t)next_random());
    }

    state->ffr = 0U == next_random() % 2U ? 0xFFFFU : (uint16_t)next_random();

    if (true == g_is_sme)
    {
        // Mostly streaming mode with ZA on, which most SME instructions need.
        state->svcr       = 0U == next_random() % 4U ? next_random() % 3U : 3U;
        state->tpidr2_el0 = next_random();

        for (uint32_t i = 0U; i < sizeof(state->za); i += 8U)
        {
            const uint64_t value = next_random();
            memcpy(&state->za[i], &value, 8U);
        }

        for (uint32_t i = 0U; i < sizeof(state->zt0); i += 8U)
        {
            const uint64_t value = next_random();
            memcpy(&state->zt0[i], &value, 8U);
        }
    }
}

static const uint8_t *
translate(void *context, const bal_guest_address_t address, size_t *available)
{
    // Ballistic passes the interface itself, as the flat interface expects.
    if (context != g_interface)
    {
        fprintf(stderr, "translate called with %p, not the interface\n", context);
        abort();
    }

    if (address >= SVE_DATA_BASE && address < SVE_DATA_BASE + SVE_DATA_SIZE)
    {
        *available = (size_t)(SVE_DATA_BASE + SVE_DATA_SIZE - address);
        return g_memory + (address - SVE_DATA_BASE);
    }

    if (address >= CODE_ADDRESS && address < CODE_ADDRESS + sizeof(g_code))
    {
        *available = (size_t)(CODE_ADDRESS + sizeof(g_code) - address);
        return g_code + (address - CODE_ADDRESS);
    }

    *available = 0U;
    return NULL;
}

/// Runs `test` in Ballistic. Returns false if Ballistic rejected the instruction.
static bool
run_ballistic(const sve_test_t *test, sve_state_t *out)
{
    memset(g_code, 0, sizeof(g_code));
    memcpy(g_code, &test->instruction, 4U);
    memcpy(g_code + 4U, test->following, sizeof(test->following));
    const uint64_t end
        = CODE_ADDRESS
          + 4U * (1U + (0U != test->following[0] ? 1U : 0U) + (0U != test->following[1] ? 1U : 0U));
    sve_fill_data(g_memory, test->seed);

    bal_allocator_t allocator;
    memset(&allocator, 0, sizeof(allocator));
    bal_allocator_default_init(&allocator);
    bal_memory_interface_t memory_interface;
    memset(&memory_interface, 0, sizeof(memory_interface));
    memory_interface.context   = g_code;
    memory_interface.translate = translate;
    g_interface                = &memory_interface;

    static bal_cpu_t cpu;
    memset(&cpu, 0, sizeof(cpu));
    memcpy(cpu.x, test->state.x, sizeof(test->state.x));
    cpu.x[31]  = test->state.sp;
    cpu.pc     = CODE_ADDRESS;
    cpu.flag_n = (uint8_t)((test->state.nzcv >> 31U) & 1U);
    cpu.flag_z = (uint8_t)((test->state.nzcv >> 30U) & 1U);
    cpu.flag_c = (uint8_t)((test->state.nzcv >> 29U) & 1U);
    cpu.flag_v = (uint8_t)((test->state.nzcv >> 28U) & 1U);
    cpu.fpcr   = test->state.fpcr;
    cpu.fpsr   = test->state.fpsr;
    memcpy(cpu.v, test->state.v, sizeof(cpu.v));
    memcpy(cpu.p, test->state.p, sizeof(cpu.p));
    cpu.ffr = test->state.ffr;
    memcpy(cpu.za, test->state.za, sizeof(cpu.za));
    memcpy(cpu.zt0, test->state.zt0, sizeof(cpu.zt0));
    cpu.svcr       = test->state.svcr;
    cpu.tpidr2_el0 = test->state.tpidr2_el0;

    bal_engine_t engine;
    memset(&engine, 0, sizeof(engine));
    (void)bal_engine_init(&engine, &cpu, &allocator, &memory_interface);
    engine.flags |= BAL_ENGINE_FLAG_SINGLE_STEP;
    bal_error_t status = BAL_SUCCESS;

    // One instruction at a time until the last one has run.
    for (uint32_t step = 0U; step < 16U && BAL_SUCCESS == status && cpu.pc < end; ++step)
    {
        status = bal_engine_run_thread(&engine);
    }

    bal_engine_destroy(&engine);

    *out = test->state;
    memcpy(out->x, cpu.x, sizeof(out->x));
    out->sp   = cpu.x[31];
    out->nzcv = ((uint64_t)cpu.flag_n << 31U) | ((uint64_t)cpu.flag_z << 30U)
                | ((uint64_t)cpu.flag_c << 29U) | ((uint64_t)cpu.flag_v << 28U);
    out->fpsr = cpu.fpsr;
    memcpy(out->v, cpu.v, sizeof(out->v));
    memcpy(out->p, cpu.p, sizeof(out->p));
    out->ffr = cpu.ffr;
    memcpy(out->za, cpu.za, sizeof(out->za));
    memcpy(out->zt0, cpu.zt0, sizeof(out->zt0));
    out->svcr       = cpu.svcr;
    out->tpidr2_el0 = cpu.tpidr2_el0;
    out->data_hash  = sve_hash_data(g_memory);
    out->status     = 0U;

    if (BAL_ERROR_MEMORY_FAULT == status
        || (BAL_ERROR_GUEST_EXCEPTION == status && BAL_EXCEPTION_DATA_ABORT == cpu.exception))
    {
        out->status = STATUS_SIGSEGV;
    }
    else if (BAL_ERROR_GUEST_EXCEPTION == status
             && BAL_EXCEPTION_UNDEFINED_INSTRUCTION == cpu.exception)
    {
        // An SME instruction whose mode is off: translated, and a SIGILL to compare.
        out->status = STATUS_SIGILL;
    }
    else if (status != BAL_SUCCESS)
    {
        return false;
    }

    return true;
}

/// Compares two final states, describing the first difference in `why`.
static bool
same_state(const sve_state_t *a, const sve_state_t *b, char *why, size_t size)
{
    // SIGBUS (alignment) and SIGSEGV are both data aborts.
    const uint64_t status_a = STATUS_SIGBUS == a->status ? STATUS_SIGSEGV : a->status;
    const uint64_t status_b = STATUS_SIGBUS == b->status ? STATUS_SIGSEGV : b->status;

    if (status_a != status_b)
    {
        snprintf(why,
                 size,
                 "status %llu vs %llu",
                 (unsigned long long)status_a,
                 (unsigned long long)status_b);
        return false;
    }

    for (uint32_t r = 0U; r < 31U; ++r)
    {
        // The reference keeps its context pointer in a register the instruction does not name.
        if (r != b->base_register && a->x[r] != b->x[r])
        {
            snprintf(why,
                     size,
                     "x%u %016llx vs %016llx",
                     r,
                     (unsigned long long)a->x[r],
                     (unsigned long long)b->x[r]);
            return false;
        }
    }

    if (a->sp != b->sp || a->nzcv != b->nzcv)
    {
        snprintf(why,
                 size,
                 "sp/nzcv %llx/%llx vs %llx/%llx",
                 (unsigned long long)a->sp,
                 (unsigned long long)a->nzcv,
                 (unsigned long long)b->sp,
                 (unsigned long long)b->nzcv);
        return false;
    }

    // The cumulative exception flags and QC.
    if ((a->fpsr & 0x0800009FULL) != (b->fpsr & 0x0800009FULL))
    {
        snprintf(why,
                 size,
                 "fpsr %llx vs %llx",
                 (unsigned long long)a->fpsr,
                 (unsigned long long)b->fpsr);
        return false;
    }

    for (uint32_t r = 0U; r < 32U; ++r)
    {
        if (a->v[r][0] != b->v[r][0] || a->v[r][1] != b->v[r][1])
        {
            snprintf(why,
                     size,
                     "z%u %016llx%016llx vs %016llx%016llx",
                     r,
                     (unsigned long long)a->v[r][1],
                     (unsigned long long)a->v[r][0],
                     (unsigned long long)b->v[r][1],
                     (unsigned long long)b->v[r][0]);
            return false;
        }
    }

    for (uint32_t r = 0U; r < 16U; ++r)
    {
        if (a->p[r] != b->p[r])
        {
            snprintf(why, size, "p%u %04x vs %04x", r, a->p[r], b->p[r]);
            return false;
        }
    }

    // qemu keeps the RES0 bits of a value written to SVCR.
    if ((a->svcr & 3U) != (b->svcr & 3U) || a->tpidr2_el0 != b->tpidr2_el0)
    {
        snprintf(why,
                 size,
                 "svcr/tpidr2 %llx/%llx vs %llx/%llx",
                 (unsigned long long)a->svcr,
                 (unsigned long long)a->tpidr2_el0,
                 (unsigned long long)b->svcr,
                 (unsigned long long)b->tpidr2_el0);
        return false;
    }

    // ZA and ZT0 only hold values while ZA is on.
    if (0U != (b->svcr & 2U))
    {
        for (uint32_t i = 0U; i < sizeof(a->za); ++i)
        {
            if (a->za[i] != b->za[i])
            {
                snprintf(
                    why, size, "za[%u][%u] %02x vs %02x", i / 16U, i % 16U, a->za[i], b->za[i]);
                return false;
            }
        }

        if (0 != memcmp(a->zt0, b->zt0, sizeof(a->zt0)))
        {
            snprintf(why, size, "zt0 differs");
            return false;
        }
    }

    if (a->ffr != b->ffr || a->data_hash != b->data_hash)
    {
        snprintf(why,
                 size,
                 "ffr %04x vs %04x, data %s",
                 a->ffr,
                 b->ffr,
                 a->data_hash != b->data_hash ? "differs" : "same");
        return false;
    }

    return true;
}

static uint64_t
read_x(const sve_state_t *state, const uint32_t index, const bool is_sp)
{
    return 31U == index ? (true == is_sp ? state->sp : 0U) : state->x[index];
}

/// Where qemu departs from the architecture, in ways the architecture allows or by bugs. Those
/// tests are not compared:
///
/// - Its contiguous LDFF1 and LDNF1 read the governing predicate at byte reg_off / 8 instead of
///   at its 64-bit word, dropping elements when the loop starts in the upper half of the vector.
/// - Its first-fault loads give up on the elements on the page after the first one.
/// - Its WHILERW treats only equal addresses as conflict free, where the architecture (and later
///   qemu) divides the distance by the element size first.
/// - It runs the SME2 SQRSHR, UQRSHR and SQRSHRU forms (qemu 11) outside streaming mode, which
///   they need.
/// - Its SME vertical tile slice loads (qemu 11) leave the inactive elements after the last
///   active one unchanged instead of zeroing them.
/// - For an instruction qemu lacks, the reference runs an equivalent one without its alignment
///   checks, so a fault only Ballistic raises is not compared.
/// - qemu-user turns a guest address into a host one without checking the guest mapped it, so a
///   gather from a random address can read qemu's own memory where Ballistic faults.
static bool
is_known_qemu_difference(const sve_test_t  *test,
                         const sve_state_t *ballistic,
                         const sve_state_t *reference)
{
    const uint32_t instruction   = test->instruction;
    const uint32_t n             = (instruction >> 5U) & 0x1FU;
    const uint32_t m             = (instruction >> 16U) & 0x1FU;
    const bool     is_nonfault   = 0xA410A000U == (instruction & 0xFE10E000U);
    const bool     is_firstfault = 0xA4006000U == (instruction & 0xFE00E000U);

    if (0U != test->substitute && STATUS_SIGSEGV == ballistic->status && 0U == reference->status)
    {
        return true;
    }

    const bool is_narrowing_shift
        = 0xC1E0D400U == (instruction & 0xFFE0FC00U) || 0xC120D800U == (instruction & 0xFF20F800U);

    if (true == is_narrowing_shift && 0U == (test->state.svcr & 1U)
        && 0U != (test->flags & SVE_TEST_SME))
    {
        return true;
    }

    const bool is_tile_load
        = 0xE0000000U == (instruction & 0xFF200010U) || 0xE1C00000U == (instruction & 0xFFE00010U);

    if (true == is_tile_load && 0U != (instruction & 0x8000U))
    {
        const uint32_t bytes = 0xE1C00000U == (instruction & 0xFFE00010U)
                                   ? 16U
                                   : 1U << ((instruction >> 22U) & 0x3U);
        const uint16_t pg    = test->state.p[(instruction >> 10U) & 0x7U];
        uint16_t       lanes = 0U;

        for (uint32_t offset = 0U; offset < 16U; offset += bytes)
        {
            lanes |= (uint16_t)(1U << offset);
        }

        const uint16_t active = pg & lanes;
        return 0U != active && 0U == (active & (1U << (16U - bytes)));
    }

    if (true == is_nonfault || true == is_firstfault)
    {
        static const uint8_t ESIZE[16]
            = { 8, 16, 32, 64, 64, 16, 32, 64, 64, 32, 32, 64, 64, 32, 16, 64 };
        static const uint8_t MSIZE[16]
            = { 8, 8, 8, 8, 32, 16, 16, 16, 16, 16, 32, 32, 8, 8, 8, 64 };
        const uint32_t dtype = (instruction >> 21U) & 0xFU;
        const uint32_t bytes = ESIZE[dtype] / 8U;
        const uint16_t pg    = test->state.p[(instruction >> 10U) & 0x7U];
        uint32_t       first = 16U;

        for (uint32_t offset = 0U; offset < 16U && 16U == first; offset += bytes)
        {
            first = 0U != ((pg >> offset) & 1U) ? offset : first;
        }

        if (first >= 8U && first < 16U)
        {
            return true;
        }

        const uint64_t mbytes = MSIZE[dtype] / 8U;
        const uint64_t address
            = read_x(&test->state, n, true) + read_x(&test->state, m, false) * mbytes;
        const uint64_t end = address + (16U / bytes) * mbytes - 1U;
        return true == is_firstfault && (address >> 12U) != (end >> 12U);
    }

    if (0x25203010U == (instruction & 0xFF20FC10U))
    {
        const uint64_t a     = read_x(&test->state, n, false);
        const uint64_t b     = read_x(&test->state, m, false);
        const uint64_t diff  = a > b ? a - b : b - a;
        const uint64_t bytes = 1ULL << ((instruction >> 22U) & 0x3U);
        return 0U != diff && diff < bytes;
    }

    const bool is_gather = 0x84000000U == (instruction & 0xBE000000U);
    return true == is_gather && STATUS_SIGSEGV == ballistic->status && 0U == reference->status;
}

/// Whether `name` is in the comma separated `list`.
static bool
is_listed(const char *list, const char *name)
{
    const size_t length = strlen(name);

    for (const char *p = list; NULL != p; p = strchr(p, ','), p = NULL == p ? NULL : p + 1)
    {
        if (0 == strncmp(p, name, length) && (',' == p[length] || '\0' == p[length]))
        {
            return true;
        }
    }

    return false;
}

typedef struct
{
    uint32_t    entry;
    uint32_t    ballistic_ok;
    sve_test_t  test;
    sve_state_t ballistic;
    sve_state_t reference;
} record_t;

/// Runs the reference on records [`next`, `count`) until all have a result; a qemu crash leaves
/// the results before it, and the crashing test is marked and skipped.
static void
run_reference(record_t *records, const uint32_t count, const char *reference, const char *qemu)
{
    char directory[] = "/tmp/bal_sve_diff_XXXXXX";

    if (NULL == mkdtemp(directory))
    {
        perror("mkdtemp");
        exit(1);
    }

    char tests_path[64];
    char results_path[64];
    char command[1024];
    snprintf(tests_path, sizeof(tests_path), "%s/tests", directory);
    snprintf(results_path, sizeof(results_path), "%s/results", directory);
    snprintf(command,
             sizeof(command),
             "%s -cpu max,sve-default-vector-length=16%s %s < %s > %s 2> /dev/null",
             qemu,
             true == g_is_sme ? ",sme-default-vector-length=16,sme_fa64=on" : "",
             reference,
             tests_path,
             results_path);
    uint32_t next = 0U;

    while (next < count)
    {
        FILE *tests = fopen(tests_path, "wb");

        for (uint32_t i = next; i < count; ++i)
        {
            (void)fwrite(&records[i].test, sizeof(sve_test_t), 1U, tests);
        }

        (void)fclose(tests);
        (void)system(command);
        FILE *results = fopen(results_path, "rb");

        while (next < count
               && 1U == fread(&records[next].reference, sizeof(sve_state_t), 1U, results))
        {
            ++next;
        }

        (void)fclose(results);

        if (next < count)
        {
            records[next].reference.status = STATUS_CRASHED;
            ++next;
        }
    }

    (void)unlink(tests_path);
    (void)unlink(results_path);
    (void)rmdir(directory);
}

/// The record index of the SME system forms, past the decoder entries.
#define SYSTEM_ENTRY BAL_DECODER_ARM64_INSTRUCTIONS_SIZE

/// Whether `instruction` is a FEAT_MOPS CPY* or SET* instruction, and the stage field: bits 23:22
/// of the copies, 15:14 of the sets, with 00 the prologue, 01 the main and 10 the epilogue.
static bool
is_memory_operation(const uint32_t instruction, uint32_t *stage_shift)
{
    if (0x19000400U != (instruction & 0x3B200C00U))
    {
        return false;
    }

    // The sets have op1 = 11 in bits 23:22.
    *stage_shift = 0xC00000U == (instruction & 0xC00000U) ? 14U : 22U;
    return true;
}

/// Turns a test of a memory copy or set instruction into a test of its whole prologue, main and
/// epilogue sequence, whose result the architecture defines. The registers are distinct, the
/// size small and the addresses inside the data area, sometimes overlapping.
static void
make_sequence(sve_test_t *test, const uint32_t stage_shift)
{
    // Three distinct registers below 31, which the architecture requires.
    uint32_t regs[3];

    for (uint32_t i = 0U; i < 3U; ++i)
    {
        do
        {
            regs[i] = (uint32_t)(next_random() % 31U);
        } while ((i > 0U && regs[i] == regs[0]) || (i > 1U && regs[i] == regs[1]));
    }

    uint32_t first = test->instruction & ~(0x3U << stage_shift) & ~0x1F03FFU;
    first |= regs[0] | (regs[1] << 5U) | (regs[2] << 16U);
    test->instruction      = first;
    test->following[0]     = first | (1U << stage_shift);
    test->following[1]     = first | (2U << stage_shift);
    const uint32_t d       = first & 0x1FU;
    const uint32_t n       = (first >> 5U) & 0x1FU;
    const uint32_t s       = (first >> 16U) & 0x1FU;
    test->state.x[n % 31U] = next_random() % 160U;

    if (d < 31U)
    {
        test->state.x[d] = SVE_DATA_BASE + 0x1000U + next_random() % 0x800U;
    }

    if (s < 31U && 22U == stage_shift)
    {
        // Only CPY (bit 26 set), not the forward-only CPYF, handles overlapping regions.
        const uint64_t near    = test->state.x[d % 31U] + next_random() % 64U - 32U;
        const bool     memmove = 0U != (first & 0x4000000U);
        test->state.x[s]       = true == memmove && 0U == next_random() % 2U
                                     ? near
                                     : SVE_DATA_BASE + 0x2000U + next_random() % 0x800U;
    }
}

/// Sets a register field of `instruction` to a register other than those in `avoid`.
static uint32_t
replace_register(const uint32_t instruction, const uint32_t shift, const uint32_t avoid)
{
    uint32_t r;

    do
    {
        r = (uint32_t)(next_random() % 31U);
    } while (0U != (avoid & (1U << r)));

    return (instruction & ~(0x1FU << shift)) | (r << shift);
}

/// For the FEAT_LRCPC3 instructions, which qemu lacks, the equivalent ordinary instruction for the
/// reference to run, after making the registers ones the architecture does not leave
/// unpredictable. Returns zero for other instructions.
static uint32_t
rcpc3_substitute(uint32_t *instruction)
{
    uint32_t       i     = *instruction;
    const uint32_t n     = (i >> 5U) & 0x1FU;
    const bool     is_64 = 0U != (i & 0x40000000U);

    // LDAPUR and STLUR (SIMD&FP): size 011101 opc 0 imm9 10 Rn Rt are LDUR and STUR (size 111100
    // opc 0 imm9 00 Rn Rt).
    if (0x1D000800U == (i & 0x3F200C00U))
    {
        return (i & ~0x3F000C00U) | 0x3C000000U;
    }

    // LDAP1 and STL1 are LD1 and ST1 of one doubleword lane, with bit 16 clear.
    if (0x0D018400U == (i & 0xBFBFFC00U))
    {
        return i & ~0x10000U;
    }

    // LDAPR post-indexed and STLR pre-indexed by the access size are LDR and STR (immediate).
    if (0x99800800U == (i & 0xBFBFFC00U))
    {
        const bool     is_load = 0U != (i & 0x400000U);
        const uint32_t bytes   = true == is_64 ? 8U : 4U;

        if (n != 31U && (i & 0x1FU) == n)
        {
            i = replace_register(i, 0U, 1U << n);
        }

        *instruction        = i;
        const uint32_t imm9 = (true == is_load ? bytes : 0x200U - bytes) & 0x1FFU;
        return (true == is_64 ? 0xF8000000U : 0xB8000000U) | (true == is_load ? 0x400400U : 0xC00U)
               | (imm9 << 12U) | (i & 0x3FFU);
    }

    // LDIAPP and STILP are LDP and STP: post-indexed (loads) or pre-indexed (stores) by two
    // elements when bit 12 is clear, at Xn otherwise.
    if (0x99000800U == (i & 0xBFA0EC00U))
    {
        const bool     is_load = 0U != (i & 0x400000U);
        const bool     wback   = 0U == (i & 0x1000U);
        const uint32_t t       = i & 0x1FU;
        const uint32_t t2      = (i >> 16U) & 0x1FU;

        // Distinct registers, and none the base when it is written back.
        if ((true == is_load && t == t2) || (true == wback && n != 31U && (t == n || t2 == n)))
        {
            i = replace_register(i, 0U, (1U << n) | (1U << t2));
            i = replace_register(i, 16U, (1U << n) | (1U << (i & 0x1FU)));
        }

        *instruction        = i;
        const uint32_t imm7 = true == wback ? (true == is_load ? 2U : 0x7EU) : 0U;
        uint32_t       base = true == wback ? (true == is_load ? 0x28C00000U : 0x29800000U)
                                            : (true == is_load ? 0x29400000U : 0x29000000U);
        base |= true == is_64 ? 0x80000000U : 0U;
        return base | (imm7 << 15U) | (((i >> 16U) & 0x1FU) << 10U) | (i & 0x3FFU);
    }

    return 0U;
}

/// Fills `record` with a random test of `instruction` and runs it in Ballistic.
static void
add_record(record_t *record, const uint32_t entry, const uint32_t instruction)
{
    uint32_t stage_shift     = 0U;
    record->entry            = entry;
    record->test.instruction = instruction;
    record->test.flags       = true == g_is_sme ? SVE_TEST_SME : 0U;
    record->test.seed        = next_random();
    random_state(&record->test.state);

    if (true == g_is_all && true == is_memory_operation(instruction, &stage_shift))
    {
        make_sequence(&record->test, stage_shift);
    }

    if (true == g_is_all)
    {
        record->test.substitute = rcpc3_substitute(&record->test.instruction);
    }

    record->test.state.base_register = sve_base_register(record->test.instruction);
    record->ballistic_ok = true == run_ballistic(&record->test, &record->ballistic) ? 1U : 0U;
}

/// Describes a mismatch.
static void
report(const record_t *record, const char *why)
{
    const uint32_t     instruction = record->test.instruction;
    const sve_state_t *state       = &record->test.state;
    const uint32_t     n           = (instruction >> 5U) & 0x1FU;
    const uint32_t     m           = (instruction >> 16U) & 0x1FU;
    printf("# %s %08X: ballistic vs reference: %s\n",
           SYSTEM_ENTRY == record->entry ? "SVCR"
                                         : g_bal_decoder_arm64_instructions[record->entry].name,
           instruction,
           why);
    printf("#   xn/sp=%016llx xm=%016llx pg=%04x ffr %04x -> %04x vs %04x\n",
           (unsigned long long)read_x(state, n, true),
           (unsigned long long)read_x(state, m, false),
           state->p[(instruction >> 10U) & 0x7U],
           state->ffr,
           record->ballistic.ffr,
           record->reference.ffr);
    printf("#   zn=%016llx%016llx zm=%016llx%016llx\n",
           (unsigned long long)state->v[n][1],
           (unsigned long long)state->v[n][0],
           (unsigned long long)state->v[m][1],
           (unsigned long long)state->v[m][0]);
}

int
main(int argc, char **argv)
{
    if (argc < 2)
    {
        fprintf(stderr, "usage: %s REFERENCE [SAMPLES] [MNEMONICS] [REPORTS]\n", argv[0]);
        return 2;
    }

    if (NULL != getenv("BAL_SVE_SEED"))
    {
        g_random = strtoull(getenv("BAL_SVE_SEED"), NULL, 0);
    }

    const char *space = getenv("BAL_SVE_SPACE");
    g_is_sme          = NULL != space && 0 == strcmp(space, "sme");
    g_is_all          = NULL != space && 0 == strcmp(space, "all");
    bal_logger_init_default();
    bal_thread_logger.min_level = BAL_LOG_LEVEL_NONE;
    const char    *reference    = argv[1];
    const uint32_t samples      = argc > 2 ? (uint32_t)strtoul(argv[2], NULL, 10) : 24U;
    const char    *filter       = argc > 3 && '\0' != argv[3][0] ? argv[3] : NULL;
    const long     reports      = argc > 4 ? strtol(argv[4], NULL, 10) : 0;
    const char    *qemu
        = NULL != getenv("BAL_SVE_QEMU") ? getenv("BAL_SVE_QEMU") : "qemu-aarch64-static";

    record_t *records
        = calloc((size_t)(BAL_DECODER_ARM64_INSTRUCTIONS_SIZE + 1U) * samples, sizeof(record_t));
    uint32_t count = 0U;

    if (NULL == records)
    {
        perror("calloc");
        return 1;
    }

    for (uint32_t e = 0U; e < BAL_DECODER_ARM64_INSTRUCTIONS_SIZE; ++e)
    {
        const bal_decoder_instruction_metadata_t *entry = &g_bal_decoder_arm64_instructions[e];

        // The SVE encoding space, op0 = 0010, the SME one, op0 = 0000 with bit 31 set, or all.
        const bool is_in_space = true == g_is_all   ? true
                                 : true == g_is_sme ? 0x80000000U == (entry->expected & 0x9E000000U)
                                                    : 0x2U == ((entry->expected >> 25U) & 0xFU);

        if (false == is_in_space || (NULL != filter && false == is_listed(filter, entry->name)))
        {
            continue;
        }

        uint32_t taken = 0U;

        for (uint32_t attempt = 0U; attempt < samples * 16U && taken < samples; ++attempt)
        {
            const uint32_t instruction = entry->expected | ((uint32_t)next_random() & ~entry->mask);

            if (bal_decode_arm64(instruction) != entry)
            {
                continue;
            }

            add_record(&records[count++], e, instruction);
            ++taken;
        }
    }

    // SMSTART and SMSTOP (MSR SVCRSM, SVCRZA and SVCRSMZA, #imm), MSR and MRS SVCR and
    // TPIDR2_EL0, counted apart from the decoder entries they share with other MSR and MRS.
    if (true == g_is_sme && NULL == filter)
    {
        static const uint32_t FORMS[]
            = { 0xD503437FU, 0xD503427FU, 0xD503457FU, 0xD503447FU, 0xD503477FU,
                0xD503467FU, 0xD51B4240U, 0xD53B4240U, 0xD51BD0A0U, 0xD53BD0A0U };

        for (uint32_t i = 0U; i < samples; ++i)
        {
            const uint32_t form = FORMS[i % (sizeof(FORMS) / sizeof(FORMS[0]))];
            const uint32_t rt   = 0x1FU == (form & 0x1FU) ? 0x1FU : (uint32_t)(next_random() % 32U);
            add_record(&records[count++], SYSTEM_ENTRY, (form & ~0x1FU) | rt);
        }
    }

    run_reference(records, count, reference, qemu);

    // Per entry, and for the system forms last: compared, matched, samples, samples the reference
    // ran.
    static uint32_t tally[BAL_DECODER_ARM64_INSTRUCTIONS_SIZE + 1U][4];
    long            reported = 0;

    for (uint32_t i = 0U; i < count; ++i)
    {
        const record_t *record = &records[i];
        uint32_t       *t      = tally[record->entry];
        const uint64_t  status = record->reference.status;
        ++t[2];
        t[3] += STATUS_SIGILL != status && STATUS_CRASHED != status ? 1U : 0U;

        if (0U == record->ballistic_ok || STATUS_CRASHED == status
            || true
                   == is_known_qemu_difference(
                       &record->test, &record->ballistic, &record->reference))
        {
            continue;
        }

        ++t[0];
        char why[160] = "";

        // A memory copy or set prologue reports in NZCV which of the algorithms the architecture
        // allows the implementation uses: Ballistic's and qemu's differ.
        sve_state_t compared = record->reference;

        if (0U != record->test.following[0])
        {
            compared.nzcv = record->ballistic.nzcv;
        }

        if (true == same_state(&record->ballistic, &compared, why, sizeof(why)))
        {
            ++t[1];
        }
        else if (reported < reports)
        {
            ++reported;
            report(record, why);
        }
    }

    for (uint32_t e = 0U; e < BAL_DECODER_ARM64_INSTRUCTIONS_SIZE; ++e)
    {
        const uint32_t *t = tally[e];

        if (0U == t[2])
        {
            continue;
        }

        const char *status = "todo";

        if (t[0] > t[1])
        {
            status = "wrong";
        }
        else if (t[0] > 0U)
        {
            status = "verified";
        }
        else if (0U == t[3])
        {
            status = "undefined";
        }

        const bal_decoder_instruction_metadata_t *entry = &g_bal_decoder_arm64_instructions[e];
        printf("%u,%s,0x%08X,0x%08X,%s,%u,%u,%u\n",
               e,
               entry->name,
               entry->mask,
               entry->expected,
               status,
               t[0],
               t[1],
               t[2]);
    }

    if (0U != tally[SYSTEM_ENTRY][2])
    {
        printf("# SMSTART, SMSTOP, SVCR and TPIDR2_EL0: %u of %u compared samples matched\n",
               tally[SYSTEM_ENTRY][1],
               tally[SYSTEM_ENTRY][0]);
    }

    free(records);
    return 0;
}
