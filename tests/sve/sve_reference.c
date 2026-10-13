//! The SVE reference: a freestanding AArch64 program for qemu-aarch64 -cpu max with 128-bit
//! vectors. It reads sve_test_t records from stdin, runs each instruction from the record's state
//! with generated load and store code around it, and writes the final sve_state_t to stdout.
#include "sve_test.h"
#include <stddef.h>

// The compiler calls these for structure copies.
void *
memcpy(void *destination, const void *source, size_t size)
{
    char       *d = destination;
    const char *s = source;

    for (size_t i = 0; i < size; ++i)
    {
        d[i] = s[i];
    }

    return destination;
}

void *
memset(void *destination, int value, size_t size)
{
    char *d = destination;

    for (size_t i = 0; i < size; ++i)
    {
        d[i] = (char)value;
    }

    return destination;
}

#define SIGILL  4
#define SIGBUS  7
#define SIGSEGV 11
#define SIGALRM 14

/// The seconds a test may run before the reference gives up on it, such as a branch to itself.
#define TEST_SECONDS 5

static long
syscall6(long number, long a, long b, long c, long d, long e, long f)
{
    register long x8 __asm__("x8") = number;
    register long x0 __asm__("x0") = a;
    register long x1 __asm__("x1") = b;
    register long x2 __asm__("x2") = c;
    register long x3 __asm__("x3") = d;
    register long x4 __asm__("x4") = e;
    register long x5 __asm__("x5") = f;
    __asm__ volatile("svc #0"
                     : "+r"(x0)
                     : "r"(x8), "r"(x1), "r"(x2), "r"(x3), "r"(x4), "r"(x5)
                     : "memory");
    return x0;
}

static long
syscall4(long number, long a, long b, long c, long d)
{
    register long x8 __asm__("x8") = number;
    register long x0 __asm__("x0") = a;
    register long x1 __asm__("x1") = b;
    register long x2 __asm__("x2") = c;
    register long x3 __asm__("x3") = d;
    __asm__ volatile("svc #0" : "+r"(x0) : "r"(x8), "r"(x1), "r"(x2), "r"(x3) : "memory");
    return x0;
}

static long
read_all(void *buffer, long size)
{
    long done = 0;

    while (done < size)
    {
        long n = syscall6(63, 0, (long)((char *)buffer + done), size - done, 0, 0, 0);

        if (n <= 0)
        {
            return done;
        }

        done += n;
    }

    return done;
}

static void
write_all(const void *buffer, long size)
{
    long done = 0;

    while (done < size)
    {
        long n = syscall6(64, 1, (long)((const char *)buffer + done), size - done, 0, 0, 0);

        if (n <= 0)
        {
            return;
        }

        done += n;
    }
}

static sve_state_t       g_context;
static uint32_t         *g_code;
static volatile uint64_t g_epilogue;
static uint32_t          g_count;

static void
emit(uint32_t word)
{
    g_code[g_count++] = word;
}

#define OFFSET(field)  ((uint32_t)offsetof(sve_state_t, field))
#define HOST_SP_OFFSET OFFSET(data_hash)

static void
emit_ldr_x(uint32_t t, uint32_t n, uint32_t offset)
{
    emit(0xF9400000U | ((offset / 8U) << 10U) | (n << 5U) | t);
}

static void
emit_str_x(uint32_t t, uint32_t n, uint32_t offset)
{
    emit(0xF9000000U | ((offset / 8U) << 10U) | (n << 5U) | t);
}

static void
emit_add_imm(uint32_t d, uint32_t n, uint32_t imm)
{
    emit(0x91000000U | (imm << 10U) | (n << 5U) | d);
}

static void
emit_vector_access(
    uint32_t base, uint32_t is_store, uint32_t is_predicate, uint32_t index, uint32_t t)
{
    const uint32_t opcode = (is_store ? 0xE5800000U : 0x85800000U) | (is_predicate ? 0U : 0x4000U);
    emit(opcode | ((index >> 3U) << 16U) | ((index & 7U) << 10U) | (base << 5U) | t);
}

/// The ZA array loads and stores: LDR and STR ZA[W12, #i], [Xn, #i, MUL VL] for the 16 slices,
/// then ZT0, from the context in Xr through Xn. W12 is free: the base is never W12-W15.
static void
emit_za_access(uint32_t r, uint32_t n, uint32_t is_store)
{
    emit(0xD280000CU); // MOV X12, #0
    emit_add_imm(n, r, OFFSET(za));
    for (uint32_t i = 0U; i < 16U; ++i)
    {
        emit((is_store ? 0xE1200000U : 0xE1000000U) | (n << 5U) | i);
    }
    emit_add_imm(n, r, OFFSET(zt0));
    emit((is_store ? 0xE13F8000U : 0xE11F8000U) | (n << 5U)); // LDR or STR ZT0, [Xn]
}

/// Builds: save the host registers, load the guest state from the context in Xr, the
/// instruction, store the guest state, restore the host and return.
static void
build(const sve_test_t *test, uint32_t r)
{
    const uint32_t instruction = test->instruction;
    const uint32_t is_sme      = 0U != (test->flags & SVE_TEST_SME);
    const uint32_t temp        = 0U == r ? 1U : 0U;
    g_count                    = 0U;

    emit(0xA9BF7BFDU); // STP X29, X30, [SP, #-16]!
    for (uint32_t i = 19U; i < 29U; i += 2U)
    {
        emit(0xA9BF0000U | ((i + 1U) << 10U) | (31U << 5U) | i); // STP Xi, Xi+1, [SP, #-16]!
    }
    emit_add_imm(1U, 31U, 0U);           // MOV X1, SP
    emit_str_x(1U, 0U, HOST_SP_OFFSET);  // the host SP in the context's padding
    emit(0xAA0003E0U | (0U << 16U) | r); // MOV Xr, X0

    // Streaming mode and ZA first: changing them resets the vector state and FPSR.
    if (is_sme)
    {
        emit_ldr_x(1U, r, OFFSET(svcr));
        emit(0xD51B4240U | 1U); // MSR SVCR, X1
        emit_ldr_x(1U, r, OFFSET(tpidr2_el0));
        emit(0xD51BD0A0U | 1U); // MSR TPIDR2_EL0, X1
        if (0U != (test->state.svcr & 2U))
        {
            emit_za_access(r, 1U, 0U);
        }
    }

    emit_ldr_x(1U, r, OFFSET(nzcv));
    emit(0xD51B4200U | 1U); // MSR NZCV, X1
    emit_ldr_x(1U, r, OFFSET(fpcr));
    emit(0xD51B4400U | 1U); // MSR FPCR, X1
    emit_ldr_x(1U, r, OFFSET(fpsr));
    emit(0xD51B4420U | 1U); // MSR FPSR, X1
    emit_add_imm(1U, r, OFFSET(v));
    for (uint32_t i = 0U; i < 32U; ++i)
    {
        emit_vector_access(1U, 0U, 0U, i, i);
    }
    emit_add_imm(1U, r, OFFSET(ffr));
    emit_vector_access(1U, 0U, 1U, 0U, 0U);
    emit(0x25289000U); // WRFFR P0.B
    emit_add_imm(1U, r, OFFSET(p));
    for (uint32_t i = 0U; i < 16U; ++i)
    {
        emit_vector_access(1U, 0U, 1U, i, i);
    }
    emit_ldr_x(1U, r, OFFSET(sp));
    emit_add_imm(31U, 1U, 0U); // MOV SP, X1
    for (uint32_t i = 0U; i < 31U; ++i)
    {
        if (i != r)
        {
            emit_ldr_x(i, r, OFFSET(x) + 8U * i);
        }
    }

    emit(0U != test->substitute ? test->substitute : instruction);
    for (uint32_t i = 0U; i < 2U && 0U != test->following[i]; ++i)
    {
        emit(test->following[i]);
    }
    g_epilogue = (uint64_t)(uintptr_t)&g_code[g_count];

    for (uint32_t i = 0U; i < 31U; ++i)
    {
        if (i != r)
        {
            emit_str_x(i, r, OFFSET(x) + 8U * i);
        }
    }
    emit_add_imm(temp, 31U, 0U); // MOV Xt, SP
    emit_str_x(temp, r, OFFSET(sp));
    emit(0xD53B4200U | temp); // MRS Xt, NZCV
    emit_str_x(temp, r, OFFSET(nzcv));
    emit(0xD53B4420U | temp); // MRS Xt, FPSR
    emit_str_x(temp, r, OFFSET(fpsr));
    emit_add_imm(temp, r, OFFSET(v));
    for (uint32_t i = 0U; i < 32U; ++i)
    {
        emit_vector_access(temp, 1U, 0U, i, i);
    }
    emit_add_imm(temp, r, OFFSET(p));
    for (uint32_t i = 0U; i < 16U; ++i)
    {
        emit_vector_access(temp, 1U, 1U, i, i);
    }
    emit(0x2519F000U); // RDFFR P0.B
    emit_add_imm(temp, r, OFFSET(ffr));
    emit_vector_access(temp, 1U, 1U, 0U, 0U);
    if (is_sme)
    {
        emit(0xD53BD0A0U | temp); // MRS Xt, TPIDR2_EL0
        emit_str_x(temp, r, OFFSET(tpidr2_el0));
        emit(0xD53B4240U | temp); // MRS Xt, SVCR
        emit_str_x(temp, r, OFFSET(svcr));
        // TBZ Xt, #1 over the ZA stores, which are only possible while ZA is on.
        emit(0x36080000U | (21U << 5U) | temp);
        emit_za_access(r, temp, 1U);
        emit(0xD51B4240U | 31U); // MSR SVCR, XZR
    }
    emit(0xD51B4400U | 31U); // MSR FPCR, XZR
    emit_ldr_x(temp, r, HOST_SP_OFFSET);
    emit_add_imm(31U, temp, 0U); // MOV SP, Xt
    for (uint32_t i = 27U; i >= 19U; i -= 2U)
    {
        emit(0xA8C10000U | ((i + 1U) << 10U) | (31U << 5U) | i); // LDP Xi, Xi+1, [SP], #16
    }
    emit(0xA8C17BFDU); // LDP X29, X30, [SP], #16
    emit(0xD65F03C0U); // RET

    // Make the new code visible to the instruction stream.
    for (uint32_t i = 0U; i < g_count; ++i)
    {
        __asm__ volatile("dc cvau, %0\n ic ivau, %0" : : "r"(&g_code[i]) : "memory");
    }
    __asm__ volatile("dsb ish\n isb" : : : "memory");
}

/// The signal handler: records the signal and resumes at the epilogue, which stores the state
/// the instruction left.
static void
handler(int signal, void *info, void *context)
{
    (void)info;
    g_context.status = (uint64_t)signal;
    // ucontext: uc_flags, uc_link, uc_stack (24), uc_sigmask (8), padding to 176, then
    // sigcontext: fault_address, regs[31], sp, pc.
    uint64_t *pc = (uint64_t *)((char *)context + 176 + 8 + 31 * 8 + 8);

    // A fault in the epilogue would come back here forever, and a test past its time may never
    // end; give up instead, which the driver reports as a reference crash on this test.
    if (SIGALRM == signal || *pc >= g_epilogue)
    {
        syscall6(94, 3, 0, 0, 0, 0, 0); // exit_group
    }

    *pc = g_epilogue;
}

typedef struct
{
    void (*handler)(int, void *, void *);
    unsigned long flags;
    void (*restorer)(void);
    unsigned long mask;
} kernel_sigaction_t;

void
_start(void)
{
    // The data area at its fixed guest address, and a page of code.
    syscall6(222, (long)SVE_DATA_BASE, (long)SVE_DATA_SIZE, 3, 0x32, -1, 0);
    g_code = (uint32_t *)syscall6(222, 0, 4096, 7, 0x22, -1, 0);

    // The guest SP is random, so the handler runs on its own stack.
    static uint8_t signal_stack[65536];
    struct
    {
        void         *sp;
        int           flags;
        int           pad;
        unsigned long size;
    } stack     = { signal_stack, 0, 0, sizeof(signal_stack) };
    long result = syscall6(132, (long)&stack, 0, 0, 0, 0, 0);

    if (result != 0)
    {
        syscall6(93, 100 - result, 0, 0, 0, 0, 0);
    }

    // SA_SIGINFO | SA_ONSTACK | SA_NODEFER; the kernel provides the return to the program.
    kernel_sigaction_t action     = { handler, 0x4U | 0x08000000U | 0x40000000U, 0, 0U };
    const int          signals[4] = { SIGILL, SIGBUS, SIGSEGV, SIGALRM };

    for (int i = 0; i < 4; ++i)
    {
        result = syscall4(134, signals[i], (long)&action, 0, 8);

        if (result != 0)
        {
            syscall6(93, 150 + 10 * i + (result == -22 ? 1 : 2), 0, 0, 0, 0, 0);
        }
    }

    static sve_test_t test;

    while (sizeof(test) == read_all(&test, sizeof(test)))
    {
        sve_fill_data((uint8_t *)SVE_DATA_BASE, test.seed);
        g_context               = test.state;
        g_context.status        = 0U;
        const uint32_t r        = sve_base_register(test.instruction);
        g_context.base_register = r;
        build(&test, r);

        // setitimer(ITIMER_REAL): SIGALRM after TEST_SECONDS, cleared once the test returns.
        long timer[4] = { 0, 0, TEST_SECONDS, 0 };
        syscall6(103, 0, (long)timer, 0, 0, 0, 0);
        ((void (*)(sve_state_t *))(uintptr_t)g_code)(&g_context);
        timer[2] = 0;
        syscall6(103, 0, (long)timer, 0, 0, 0, 0);
        g_context.data_hash = sve_hash_data((const uint8_t *)SVE_DATA_BASE);
        g_context.pad[0] = g_context.pad[1] = g_context.pad[2] = 0U;
        write_all(&g_context, sizeof(g_context));
    }

    syscall6(93, 0, 0, 0, 0, 0, 0);
}
