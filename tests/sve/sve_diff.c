//! SVE differential tester. It runs random encodings of every SVE decoder table entry through
//! Ballistic and through the AArch64 reference (sve_reference.c) under qemu-aarch64 with 128-bit
//! vectors, and compares the general purpose registers, SP, NZCV, the FPSR flags, the Z and P
//! registers, FFR, a data area and whether the instruction faulted.
//!
//! Usage: bal_sve_diff REFERENCE [SAMPLES] [MNEMONICS] [REPORTS]
//!
//! REFERENCE is the built reference program, run with $BAL_SVE_QEMU (qemu-aarch64-static by
//! default). $BAL_SVE_SEED, when set, changes the random encodings and states. MNEMONICS is an
//! optional comma separated list of entries to test, and REPORTS the number of mismatches to
//! describe. The output has one CSV line per entry, in the format of the status
//! tracker's coverage tool:
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

    bal_engine_t engine;
    memset(&engine, 0, sizeof(engine));
    (void)bal_engine_init(&engine, &cpu, &allocator, &memory_interface);
    engine.flags |= BAL_ENGINE_FLAG_SINGLE_STEP;
    const bal_error_t status = bal_engine_run_thread(&engine);
    bal_engine_destroy(&engine);

    *out = test->state;
    memcpy(out->x, cpu.x, sizeof(out->x));
    out->sp   = cpu.x[31];
    out->nzcv = ((uint64_t)cpu.flag_n << 31U) | ((uint64_t)cpu.flag_z << 30U)
                | ((uint64_t)cpu.flag_c << 29U) | ((uint64_t)cpu.flag_v << 28U);
    out->fpsr = cpu.fpsr;
    memcpy(out->v, cpu.v, sizeof(out->v));
    memcpy(out->p, cpu.p, sizeof(out->p));
    out->ffr       = cpu.ffr;
    out->data_hash = sve_hash_data(g_memory);
    out->status    = 0U;

    if (BAL_ERROR_MEMORY_FAULT == status
        || (BAL_ERROR_GUEST_EXCEPTION == status && BAL_EXCEPTION_DATA_ABORT == cpu.exception))
    {
        out->status = STATUS_SIGSEGV;
    }
    else if (BAL_ERROR_GUEST_EXCEPTION == status)
    {
        out->status = STATUS_SIGILL;
    }
    else if (status != BAL_SUCCESS)
    {
        return false;
    }

    return STATUS_SIGILL != out->status;
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

/// Where qemu 6.2 departs from the architecture, in ways the architecture allows or by bugs.
/// Those tests are not compared:
///
/// - Its contiguous LDFF1 and LDNF1 read the governing predicate at byte reg_off / 8 instead of
///   at its 64-bit word, dropping elements when the loop starts in the upper half of the vector.
/// - Its first-fault loads give up on the elements on the page after the first one.
/// - Its WHILERW treats only equal addresses as conflict free, where the architecture (and later
///   qemu) divides the distance by the element size first.
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
             "%s -cpu max,sve-default-vector-length=16 %s < %s > %s 2> /dev/null",
             qemu,
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

/// Describes a mismatch.
static void
report(const record_t *record, const char *why)
{
    const uint32_t     instruction = record->test.instruction;
    const sve_state_t *state       = &record->test.state;
    const uint32_t     n           = (instruction >> 5U) & 0x1FU;
    const uint32_t     m           = (instruction >> 16U) & 0x1FU;
    printf("# %s %08X: ballistic vs reference: %s\n",
           g_bal_decoder_arm64_instructions[record->entry].name,
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

    bal_logger_init_default();
    bal_thread_logger.min_level = BAL_LOG_LEVEL_NONE;
    const char    *reference    = argv[1];
    const uint32_t samples      = argc > 2 ? (uint32_t)strtoul(argv[2], NULL, 10) : 24U;
    const char    *filter       = argc > 3 && '\0' != argv[3][0] ? argv[3] : NULL;
    const long     reports      = argc > 4 ? strtol(argv[4], NULL, 10) : 0;
    const char    *qemu
        = NULL != getenv("BAL_SVE_QEMU") ? getenv("BAL_SVE_QEMU") : "qemu-aarch64-static";

    record_t *records
        = calloc((size_t)BAL_DECODER_ARM64_INSTRUCTIONS_SIZE * samples, sizeof(record_t));
    uint32_t count = 0U;

    if (NULL == records)
    {
        perror("calloc");
        return 1;
    }

    for (uint32_t e = 0U; e < BAL_DECODER_ARM64_INSTRUCTIONS_SIZE; ++e)
    {
        const bal_decoder_instruction_metadata_t *entry = &g_bal_decoder_arm64_instructions[e];

        // The SVE encoding space, op0 = 0010.
        if (0x2U != ((entry->expected >> 25U) & 0xFU)
            || (NULL != filter && false == is_listed(filter, entry->name)))
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

            record_t *record         = &records[count++];
            record->entry            = e;
            record->test.instruction = instruction;
            record->test.seed        = next_random();
            random_state(&record->test.state);
            record->test.state.base_register = sve_base_register(instruction);
            record->ballistic_ok
                = true == run_ballistic(&record->test, &record->ballistic) ? 1U : 0U;
            ++taken;
        }
    }

    run_reference(records, count, reference, qemu);

    // Per entry: compared, matched, samples, samples the reference ran.
    static uint32_t tally[BAL_DECODER_ARM64_INSTRUCTIONS_SIZE][4];
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

        if (true == same_state(&record->ballistic, &record->reference, why, sizeof(why)))
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

    free(records);
    return 0;
}
