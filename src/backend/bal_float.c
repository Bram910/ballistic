#include "backend/bal_float.h"
#include "bal_attributes.h"

#include <fenv.h>
#include <math.h>
#include <string.h>

#if defined(__x86_64__) || defined(_M_X64)
#include <xmmintrin.h>

// Everything here computes with SSE, whose rounding mode and exception flags live in MXCSR.
// fesetround(), feclearexcept() and fetestexcept() also keep the x87 unit in step, which makes them
// many times slower than the MXCSR access they need; these replace them. Each MXCSR access is
// still slow, so the control bits (everything but the flags) are kept in g_control: each public
// entry point reloads them, and only changing the rounding mode, clearing the flags and testing
// them touch MXCSR.

static BAL_THREAD_LOCAL uint32_t g_control;

/// Reloads g_control from MXCSR, which code outside this file may have changed.
static void
load_control(void)
{
    g_control = _mm_getcsr() & ~0x3FU;
}

static int
sse_fegetround(void)
{
    static const int MODES[4] = { FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO };
    return MODES[(g_control >> 13U) & 3U];
}

/// Changing the mode also clears the flags, which every caller has read or is about to clear.
static int
sse_fesetround(const int mode)
{
    const uint32_t control = FE_DOWNWARD == mode     ? 1U
                             : FE_UPWARD == mode     ? 2U
                             : FE_TOWARDZERO == mode ? 3U
                                                     : 0U;
    const uint32_t wanted  = (g_control & ~0x6000U) | (control << 13U);

    if (wanted != g_control)
    {
        g_control = wanted;
        _mm_setcsr(wanted);
    }

    return 0;
}

static int
sse_feclearexcept(const int excepts)
{
    (void)excepts;
    _mm_setcsr(g_control);
    return 0;
}

static int
sse_fetestexcept(const int excepts)
{
    const uint32_t raised = _mm_getcsr();
    int            flags  = 0;
    flags |= 0U != (raised & 0x01U) ? FE_INVALID : 0;
    flags |= 0U != (raised & 0x04U) ? FE_DIVBYZERO : 0;
    flags |= 0U != (raised & 0x08U) ? FE_OVERFLOW : 0;
    flags |= 0U != (raised & 0x10U) ? FE_UNDERFLOW : 0;
    flags |= 0U != (raised & 0x20U) ? FE_INEXACT : 0;
    return flags & excepts;
}

#define fegetround    sse_fegetround
#define fesetround    sse_fesetround
#define feclearexcept sse_feclearexcept
#define fetestexcept  sse_fetestexcept
#else
/// The standard functions keep nothing to reload.
static void
load_control(void)
{
}
#endif

// The helper reads and changes the host floating-point environment, so the compiler must not
// constant-fold, reorder or speculate floating-point operations across the fenv calls.
#pragma STDC FENV_ACCESS ON

#define FPCR_AHP      (1ULL << 26U)
#define FPCR_DN       (1ULL << 25U)
#define FPCR_FZ       (1ULL << 24U)
#define FPCR_FZ16     (1ULL << 19U)
#define FPCR_RMODE(x) (((x) >> 22U) & 0x3U)

#define FPSR_IOC (1ULL << 0U)
#define FPSR_DZC (1ULL << 1U)
#define FPSR_OFC (1ULL << 2U)
#define FPSR_UFC (1ULL << 3U)
#define FPSR_IXC (1ULL << 4U)
#define FPSR_IDC (1ULL << 7U)

/// The scalar floating-point classes bal_execute_float() handles.
typedef enum
{
    FLOAT_UNSUPPORTED,
    /// FCVT[NPMZA][SU], SCVTF/UCVTF and FMOV between general purpose and SIMD&FP registers.
    FLOAT_INTEGER_CONVERSION,
    /// SCVTF/UCVTF and FCVTZS/FCVTZU with a fixed-point scale.
    FLOAT_FIXED_CONVERSION,
    /// FMOV, FABS, FNEG, FSQRT, FCVT and FRINT*.
    FLOAT_ONE_SOURCE,
    /// FCMP and FCMPE.
    FLOAT_COMPARE,
    /// FMOV (scalar, immediate).
    FLOAT_IMMEDIATE,
    /// FCCMP and FCCMPE.
    FLOAT_CONDITIONAL_COMPARE,
    /// FMUL, FDIV, FADD, FSUB, FMAX, FMIN, FMAXNM, FMINNM and FNMUL.
    FLOAT_TWO_SOURCE,
    /// FCSEL.
    FLOAT_CONDITIONAL_SELECT,
    /// FMADD, FMSUB, FNMADD and FNMSUB.
    FLOAT_THREE_SOURCE,
} float_class_t;

/// The precision of a scalar operation, from the ftype field.
typedef enum
{
    PRECISION_SINGLE  = 0,
    PRECISION_DOUBLE  = 1,
    PRECISION_INVALID = 2,
    PRECISION_HALF    = 3,
} precision_t;

static float_class_t
classify(const uint32_t instruction)
{
    // Scalar floating point: x 0 S 1111 x ftype ..., with S = 0. Bit 31 is sf for the
    // conversions between floating point and integers and M, which must be zero, otherwise.
    if (0x1E000000U != (instruction & 0x7E000000U))
    {
        return FLOAT_UNSUPPORTED;
    }

    if (1U == ((instruction >> 24U) & 1U))
    {
        return 0U == (instruction >> 31U) ? FLOAT_THREE_SOURCE : FLOAT_UNSUPPORTED;
    }

    if (0x1E000000U == (instruction & 0x5F200000U))
    {
        return FLOAT_FIXED_CONVERSION;
    }

    if (0x1E200000U == (instruction & 0x5F20FC00U))
    {
        return FLOAT_INTEGER_CONVERSION;
    }

    // The rest have M = 0 in bit 31.
    if (1U == (instruction >> 31U))
    {
        return FLOAT_UNSUPPORTED;
    }

    if (0x1E204000U == (instruction & 0x5F207C00U))
    {
        return FLOAT_ONE_SOURCE;
    }

    if (0x1E202000U == (instruction & 0x5F203C00U))
    {
        return FLOAT_COMPARE;
    }

    if (0x1E201000U == (instruction & 0x5F201FE0U))
    {
        return FLOAT_IMMEDIATE;
    }

    if (0x1E200400U == (instruction & 0x5F200C00U))
    {
        return FLOAT_CONDITIONAL_COMPARE;
    }

    if (0x1E200800U == (instruction & 0x5F200C00U))
    {
        return FLOAT_TWO_SOURCE;
    }

    if (0x1E200C00U == (instruction & 0x5F200C00U))
    {
        return FLOAT_CONDITIONAL_SELECT;
    }

    return FLOAT_UNSUPPORTED;
}

bool
bal_is_float_supported(const uint32_t instruction)
{
    const float_class_t instruction_class = classify(instruction);
    const precision_t   precision         = (precision_t)((instruction >> 22U) & 0x3U);
    const bool          is_half           = PRECISION_HALF == precision;

    if (PRECISION_INVALID == precision && instruction_class != FLOAT_INTEGER_CONVERSION)
    {
        return false;
    }

    switch (instruction_class)
    {
        case FLOAT_INTEGER_CONVERSION: {
            const uint32_t sf     = instruction >> 31U;
            const uint32_t rmode  = (instruction >> 19U) & 0x3U;
            const uint32_t opcode = (instruction >> 16U) & 0x7U;

            // FJCVTZS.
            if (6U == opcode && 3U == rmode)
            {
                return 0U == sf && PRECISION_DOUBLE == precision;
            }

            // FMOV: W with S or H, X with D or H, and X with the top half of a Q register.
            if (opcode >= 6U)
            {
                if (1U == rmode)
                {
                    return 1U == sf && PRECISION_INVALID == precision;
                }

                return 0U == rmode
                       && ((0U == sf && (PRECISION_SINGLE == precision || is_half))
                           || (1U == sf && (PRECISION_DOUBLE == precision || is_half)));
            }

            // The arithmetic conversions: NS/NU/AS/AU and SCVTF/UCVTF with rmode 00, and the
            // directed ones with rmode 01-11.
            if (PRECISION_INVALID == precision)
            {
                return false;
            }

            if (opcode >= 2U && opcode <= 3U)
            {
                return 0U == rmode;
            }

            if (opcode >= 4U)
            {
                return 0U == rmode;
            }

            return true;
        }

        case FLOAT_FIXED_CONVERSION: {
            const uint32_t sf     = instruction >> 31U;
            const uint32_t rmode  = (instruction >> 19U) & 0x3U;
            const uint32_t opcode = (instruction >> 16U) & 0x7U;
            const uint32_t scale  = (instruction >> 10U) & 0x3FU;

            if (0U == sf && scale < 32U)
            {
                return false;
            }

            return (0U == rmode && (2U == opcode || 3U == opcode)) || (3U == rmode && opcode <= 1U);
        }

        case FLOAT_ONE_SOURCE: {
            const uint32_t opcode = (instruction >> 15U) & 0x3FU;

            // BFCVT: single precision to BFloat16, with ftype 01.
            if (6U == opcode && PRECISION_DOUBLE == precision)
            {
                return true;
            }

            // FCVT between any two of half, single and double precision.
            if (4U == (opcode & 0x3CU))
            {
                const uint32_t target = opcode & 0x3U;
                return target != (uint32_t)precision && target != PRECISION_INVALID;
            }

            // FRINT32Z, FRINT32X, FRINT64Z and FRINT64X, in single and double precision.
            if (0x10U == (opcode & 0x3CU))
            {
                return false == is_half;
            }

            return opcode <= 3U || (opcode >= 8U && opcode <= 15U && opcode != 13U);
        }

        case FLOAT_COMPARE:
            return 0U == ((instruction >> 14U) & 0x3U) && 0U == (instruction & 0x7U);

        case FLOAT_IMMEDIATE:
        case FLOAT_CONDITIONAL_SELECT:
            return true;

        case FLOAT_CONDITIONAL_COMPARE:
        case FLOAT_THREE_SOURCE:
            return true;

        case FLOAT_TWO_SOURCE:
            return ((instruction >> 12U) & 0xFU) <= 8U;

        case FLOAT_UNSUPPORTED:
        default:
            return false;
    }
}

/// The bit layout of a precision.
typedef struct
{
    uint32_t bits;
    uint32_t fraction_bits;
    uint64_t sign_mask;
    uint64_t exponent_mask;
    uint64_t fraction_mask;
    uint64_t quiet_bit;
} layout_t;

static layout_t
layout_of(const precision_t precision)
{
    layout_t layout = { 0 };

    if (PRECISION_DOUBLE == precision)
    {
        layout.bits          = 64U;
        layout.fraction_bits = 52U;
    }
    else if (PRECISION_HALF == precision)
    {
        layout.bits          = 16U;
        layout.fraction_bits = 10U;
    }
    else
    {
        layout.bits          = 32U;
        layout.fraction_bits = 23U;
    }

    layout.sign_mask     = 1ULL << (layout.bits - 1U);
    layout.fraction_mask = (1ULL << layout.fraction_bits) - 1U;
    layout.exponent_mask = (layout.sign_mask - 1U) & ~layout.fraction_mask;
    layout.quiet_bit     = 1ULL << (layout.fraction_bits - 1U);
    return layout;
}

static bool
is_nan(const layout_t *layout, const uint64_t value)
{
    return layout->exponent_mask == (value & layout->exponent_mask)
           && 0U != (value & layout->fraction_mask);
}

static bool
is_signalling_nan(const layout_t *layout, const uint64_t value)
{
    return is_nan(layout, value) && 0U == (value & layout->quiet_bit);
}

static bool
is_infinity(const layout_t *layout, const uint64_t value)
{
    return layout->exponent_mask == (value & layout->exponent_mask)
           && 0U == (value & layout->fraction_mask);
}

static bool
is_zero(const layout_t *layout, const uint64_t value)
{
    return 0U == (value & ~layout->sign_mask);
}

static bool
is_denormal(const layout_t *layout, const uint64_t value)
{
    return 0U == (value & layout->exponent_mask) && 0U != (value & layout->fraction_mask);
}

static uint64_t
default_nan(const layout_t *layout)
{
    return layout->exponent_mask | layout->quiet_bit;
}

/// The execution context of one instruction: FPCR and the FPSR flags it raises.
typedef struct
{
    uint64_t    fpcr;
    uint64_t    flags;
    layout_t    layout;
    precision_t precision;
    uint32_t    pad;
} context_t;

/// Flushes a denormal input to zero when FPCR.FZ is set, raising Input Denormal. Half precision
/// uses FPCR.FZ16 instead and raises nothing.
static uint64_t
flush_input(context_t *context, const uint64_t value)
{
    if (PRECISION_HALF == context->precision)
    {
        return 0U != (context->fpcr & FPCR_FZ16) && true == is_denormal(&context->layout, value)
                   ? value & context->layout.sign_mask
                   : value;
    }

    if (0U != (context->fpcr & FPCR_FZ) && true == is_denormal(&context->layout, value))
    {
        context->flags |= FPSR_IDC;
        return value & context->layout.sign_mask;
    }

    return value;
}

/// Returns the NaN `value` quietened, or the default NaN when FPCR.DN is set.
static uint64_t
propagate_nan(context_t *context, const uint64_t value)
{
    if (true == is_signalling_nan(&context->layout, value))
    {
        context->flags |= FPSR_IOC;
    }

    if (0U != (context->fpcr & FPCR_DN))
    {
        return default_nan(&context->layout);
    }

    return value | context->layout.quiet_bit;
}

/// Applies the AArch64 NaN priority to up to three operands: the first signalling NaN, then the
/// first quiet NaN. Returns true and sets `result` if any operand is a NaN.
static bool
process_nans(context_t *context, const uint64_t operands[], const uint32_t count, uint64_t *result)
{
    for (uint32_t i = 0U; i < count; ++i)
    {
        if (true == is_signalling_nan(&context->layout, operands[i]))
        {
            // Any other signalling NaN raises Invalid Operation as well, which is the same flag.
            *result = propagate_nan(context, operands[i]);
            return true;
        }
    }

    for (uint32_t i = 0U; i < count; ++i)
    {
        if (true == is_nan(&context->layout, operands[i]))
        {
            *result = propagate_nan(context, operands[i]);
            return true;
        }
    }

    return false;
}

static int
host_rounding_mode(const uint32_t rmode)
{
    switch (rmode)
    {
        case 1U:
            return FE_UPWARD;
        case 2U:
            return FE_DOWNWARD;
        case 3U:
            return FE_TOWARDZERO;
        default:
            return FE_TONEAREST;
    }
}

/// Converts the host exception flags raised since the last feclearexcept() to FPSR bits.
static uint64_t
host_flags(void)
{
    const int raised = fetestexcept(FE_ALL_EXCEPT);
    uint64_t  flags  = 0U;
    flags |= 0 != (raised & FE_INVALID) ? FPSR_IOC : 0U;
    flags |= 0 != (raised & FE_DIVBYZERO) ? FPSR_DZC : 0U;
    flags |= 0 != (raised & FE_OVERFLOW) ? FPSR_OFC : 0U;
    flags |= 0 != (raised & FE_UNDERFLOW) ? FPSR_UFC : 0U;
    flags |= 0 != (raised & FE_INEXACT) ? FPSR_IXC : 0U;
    return flags;
}

/// Finishes an arithmetic result computed by the host: a NaN the host generated becomes the
/// AArch64 default NaN, and with FPCR.FZ a denormal result is flushed to zero, raising Underflow
/// but not Inexact.
static uint64_t
finish(context_t *context, uint64_t result, uint64_t flags)
{
    if (true == is_nan(&context->layout, result))
    {
        result = default_nan(&context->layout);
    }
    else if (0U != (context->fpcr & FPCR_FZ)
             && (true == is_denormal(&context->layout, result) || 0U != (flags & FPSR_UFC)))
    {
        // The host may already have rounded a tiny result to zero, raising Underflow.
        result = result & context->layout.sign_mask;
        flags  = (flags & ~(FPSR_IXC | FPSR_UFC)) | FPSR_UFC;
    }

    context->flags |= flags;
    return result;
}

static float
to_float(const uint64_t bits)
{
    const uint32_t narrow = (uint32_t)bits;
    float          value  = 0.0F;
    (void)memcpy(&value, &narrow, sizeof(value));
    return value;
}

static double
to_double(const uint64_t bits)
{
    double value = 0.0;
    (void)memcpy(&value, &bits, sizeof(value));
    return value;
}

static uint64_t
from_float(const float value)
{
    uint32_t bits = 0U;
    (void)memcpy(&bits, &value, sizeof(bits));
    return bits;
}

static uint64_t
from_double(const double value)
{
    uint64_t bits = 0U;
    (void)memcpy(&bits, &value, sizeof(bits));
    return bits;
}

/// Host arithmetic operations, applied in the operand precision.
typedef enum
{
    OPERATION_ADD,
    OPERATION_SUBTRACT,
    OPERATION_MULTIPLY,
    OPERATION_DIVIDE,
    OPERATION_SQUARE_ROOT,
    OPERATION_FUSED_MULTIPLY_ADD,
} operation_t;

/// Computes one host operation in the operand precision under the current host rounding mode.
static uint64_t
host_compute(const context_t  *context,
             const operation_t operation,
             const uint64_t    a,
             const uint64_t    b,
             const uint64_t    c)
{
    if (PRECISION_DOUBLE == context->precision)
    {
        const double x     = to_double(a);
        const double y     = to_double(b);
        const double z     = to_double(c);
        double       value = 0.0;

        switch (operation)
        {
            case OPERATION_ADD:
                value = x + y;
                break;
            case OPERATION_SUBTRACT:
                value = x - y;
                break;
            case OPERATION_MULTIPLY:
                value = x * y;
                break;
            case OPERATION_DIVIDE:
                value = x / y;
                break;
            case OPERATION_SQUARE_ROOT:
                value = sqrt(x);
                break;
            case OPERATION_FUSED_MULTIPLY_ADD:
            default:
                value = fma(x, y, z);
                break;
        }

        return from_double(value);
    }

    const float x     = to_float(a);
    const float y     = to_float(b);
    const float z     = to_float(c);
    float       value = 0.0F;

    switch (operation)
    {
        case OPERATION_ADD:
            value = x + y;
            break;
        case OPERATION_SUBTRACT:
            value = x - y;
            break;
        case OPERATION_MULTIPLY:
            value = x * y;
            break;
        case OPERATION_DIVIDE:
            value = x / y;
            break;
        case OPERATION_SQUARE_ROOT:
            value = sqrtf(x);
            break;
        case OPERATION_FUSED_MULTIPLY_ADD:
        default:
            value = fmaf(x, y, z);
            break;
    }

    return from_float(value);
}

/// Converts half precision `bits` to double, exactly. With `is_alternative`, the FPCR.AHP format,
/// exponent 31 holds normal numbers; otherwise it is an infinity (NaNs are the caller's).
static double
half_to_double(const uint64_t bits, const bool is_alternative)
{
    const double   sign     = 0U != (bits & 0x8000U) ? -1.0 : 1.0;
    const uint32_t exponent = (uint32_t)(bits >> 10U) & 0x1FU;
    const double   fraction = (double)(bits & 0x3FFU);

    if (0U == exponent)
    {
        return sign * ldexp(fraction, -24);
    }

    if (31U == exponent && false == is_alternative)
    {
        return sign * INFINITY;
    }

    return sign * ldexp(1024.0 + fraction, (int)exponent - 25);
}

/// Rounds a value to half precision as FPRound does, under the guest rounding mode.
///
/// `value` is the exact result truncated towards zero, with `sticky` set when that dropped
/// non-zero bits, so a result rounds once even when computing it took a rounding. `is_negative`
/// gives the sign, which matters for a zero. `is_alternative` selects the FPCR.AHP format, which
/// saturates instead of overflowing, and `can_flush` applies FPCR.FZ16, which conversions ignore.
static uint64_t
round_to_half(context_t   *context,
              const double value,
              const bool   sticky,
              const bool   is_negative,
              const bool   is_alternative,
              const bool   can_flush)
{
    const uint64_t sign       = true == is_negative ? 0x8000U : 0U;
    const double   magnitude  = fabs(value);
    const double   min_normal = 0x1p-14;
    const uint32_t rmode      = FPCR_RMODE(context->fpcr);

    if (0.0 == magnitude && false == sticky)
    {
        return sign;
    }

    // Tininess is decided before rounding.
    if (true == can_flush && 0U != (context->fpcr & FPCR_FZ16) && magnitude < min_normal)
    {
        context->flags |= FPSR_UFC;
        return sign;
    }

    // The quantum is the weight of the lowest fraction bit, fixed at 2^-24 for denormals.
    int exponent = 0;
    (void)frexp(magnitude, &exponent);
    exponent              = exponent - 1 < -14 ? -14 : exponent - 1;
    const double quantum  = ldexp(1.0, exponent - 10);
    const double scaled   = magnitude / quantum;
    const double integer  = floor(scaled);
    const double error    = scaled - integer;
    const bool   inexact  = error > 0.0 || true == sticky;
    bool         round_up = false;

    switch (rmode)
    {
        case 0U: // To nearest, ties to even.
            round_up
                = error > 0.5 || (0.5 == error && (true == sticky || 1.0 == fmod(integer, 2.0)));
            break;
        case 1U: // Towards +infinity.
            round_up = true == inexact && false == is_negative;
            break;
        case 2U: // Towards -infinity.
            round_up = true == inexact && true == is_negative;
            break;
        default: // Towards zero.
            break;
    }

    const double rounded = (integer + (true == round_up ? 1.0 : 0.0)) * quantum;
    const double maximum = true == is_alternative ? 131008.0 : 65504.0;

    if (rounded > maximum)
    {
        // The alternative format has no infinity and saturates with Invalid Operation.
        if (true == is_alternative)
        {
            context->flags |= FPSR_IOC;
            return sign | 0x7FFFU;
        }

        context->flags |= FPSR_OFC | FPSR_IXC;
        const bool is_to_infinity = 0U == rmode || (1U == rmode && false == is_negative)
                                    || (2U == rmode && true == is_negative);
        return sign | (true == is_to_infinity ? 0x7C00U : 0x7BFFU);
    }

    if (true == inexact)
    {
        context->flags |= FPSR_IXC | (magnitude < min_normal ? FPSR_UFC : 0U);
    }

    if (rounded < min_normal)
    {
        return sign | (uint64_t)ldexp(rounded, 24);
    }

    (void)frexp(rounded, &exponent);
    const double fraction = ldexp(rounded, 1 - exponent) - 1.0; // in [0, 1)
    return sign | ((uint64_t)(exponent - 1 + 15) << 10U) | (uint64_t)ldexp(fraction, 10);
}

/// Converts a finite or infinite operand of the context's precision to double, exactly.
static double
operand_to_double(const context_t *context, const uint64_t value)
{
    switch (context->precision)
    {
        case PRECISION_DOUBLE:
            return to_double(value);
        case PRECISION_HALF:
            return half_to_double(value, false);
        default:
            return (double)to_float(value);
    }
}

/// Computes one operation in double precision under the current host rounding mode.
static double
compute_double(const operation_t operation, const double x, const double y, const double z)
{
    switch (operation)
    {
        case OPERATION_ADD:
            return x + y;
        case OPERATION_SUBTRACT:
            return x - y;
        case OPERATION_MULTIPLY:
            return x * y;
        case OPERATION_DIVIDE:
            return x / y;
        case OPERATION_SQUARE_ROOT:
            return sqrt(x);
        case OPERATION_FUSED_MULTIPLY_ADD:
        default:
            return fma(x, y, z);
    }
}

/// host_operation() for half precision: the operation runs in double precision, once under the
/// guest rounding mode for the sign of a zero, Invalid Operation and Divide by Zero, and once
/// towards zero, whose result and Inexact flag round_to_half() turns into the half result.
static uint64_t
half_operation(context_t        *context,
               const operation_t operation,
               const uint64_t    a,
               const uint64_t    b,
               const uint64_t    c)
{
    const double x             = half_to_double(a, false);
    const double y             = half_to_double(b, false);
    const double z             = half_to_double(c, false);
    const int    previous_mode = fegetround();
    (void)fesetround(host_rounding_mode(FPCR_RMODE(context->fpcr)));
    (void)feclearexcept(FE_ALL_EXCEPT);
    const double   guest_result = compute_double(operation, x, y, z);
    const uint64_t flags        = host_flags() & (FPSR_IOC | FPSR_DZC);
    (void)fesetround(FE_TOWARDZERO);
    (void)feclearexcept(FE_ALL_EXCEPT);
    const double truncated = compute_double(operation, x, y, z);
    const bool   sticky    = 0 != fetestexcept(FE_INEXACT);
    (void)fesetround(previous_mode);
    context->flags |= flags;

    if (0 != isnan(guest_result))
    {
        return default_nan(&context->layout);
    }

    if (0 != isinf(guest_result))
    {
        return (0 != signbit(guest_result) ? 0x8000U : 0U) | 0x7C00U;
    }

    return round_to_half(context, truncated, sticky, 0 != signbit(guest_result), false, true);
}

/// Runs one host operation under the guest rounding mode and returns its result and flags.
static uint64_t
host_operation(context_t        *context,
               const operation_t operation,
               const uint64_t    a,
               const uint64_t    b,
               const uint64_t    c)
{
    if (PRECISION_HALF == context->precision)
    {
        return half_operation(context, operation, a, b, c);
    }

    const int previous_mode = fegetround();
    (void)fesetround(host_rounding_mode(FPCR_RMODE(context->fpcr)));
    (void)feclearexcept(FE_ALL_EXCEPT);
    uint64_t result = host_compute(context, operation, a, b, c);
    uint64_t flags  = host_flags();

    // AArch64 detects tininess before rounding and x86 after it, so an inexact result that
    // rounded up to the smallest normal may still have been tiny: then it raises Underflow, and
    // with FPCR.FZ finish() flushes it. Rounding towards zero gives a denormal exactly when the
    // exact result was tiny.
    const uint64_t smallest_normal = 1ULL << context->layout.fraction_bits;

    if (0U != (flags & FPSR_IXC) && smallest_normal == (result & ~context->layout.sign_mask))
    {
        (void)fesetround(FE_TOWARDZERO);
        const uint64_t truncated = host_compute(context, operation, a, b, c);

        if (true == is_denormal(&context->layout, truncated)
            || true == is_zero(&context->layout, truncated))
        {
            flags |= FPSR_UFC;
        }
    }

    (void)fesetround(previous_mode);
    return finish(context, result, flags);
}

/// Reads the low `bits` of SIMD&FP register `index`.
static uint64_t
read_vector(const bal_cpu_t *cpu, const uint32_t index, const uint32_t bits)
{
    const uint64_t value = cpu->v[index][0];
    return bits >= 64U ? value : value & ((1ULL << bits) - 1U);
}

/// Writes a scalar to SIMD&FP register `index`, clearing the rest of the register.
static void
write_vector(bal_cpu_t *cpu, const uint32_t index, const uint64_t value)
{
    cpu->v[index][0] = value;
    cpu->v[index][1] = 0U;
}

/// Reads a general purpose register, with register 31 as the zero register.
static uint64_t
read_general(const bal_cpu_t *cpu, const uint32_t index)
{
    return 31U == index ? 0U : cpu->x[index];
}

/// Writes a general purpose register, discarding writes to the zero register.
static void
write_general(bal_cpu_t *cpu, const uint32_t index, const uint64_t value)
{
    if (index != 31U)
    {
        cpu->x[index] = value;
    }
}

/// Evaluates an AArch64 condition code against the guest flags.
static bool
condition_holds(const bal_cpu_t *cpu, const uint32_t condition)
{
    const bool n      = cpu->flag_n != 0U;
    const bool z      = cpu->flag_z != 0U;
    const bool c      = cpu->flag_c != 0U;
    const bool v      = cpu->flag_v != 0U;
    bool       result = true;

    switch (condition >> 1U)
    {
        case 0U:
            result = z;
            break;
        case 1U:
            result = c;
            break;
        case 2U:
            result = n;
            break;
        case 3U:
            result = v;
            break;
        case 4U:
            result = c && false == z;
            break;
        case 5U:
            result = n == v;
            break;
        case 6U:
            result = n == v && false == z;
            break;
        default:
            return true;
    }

    return 1U == (condition & 1U) ? false == result : result;
}

static void
set_nzcv(bal_cpu_t *cpu, const uint32_t nzcv)
{
    cpu->flag_n = (uint8_t)((nzcv >> 3U) & 1U);
    cpu->flag_z = (uint8_t)((nzcv >> 2U) & 1U);
    cpu->flag_c = (uint8_t)((nzcv >> 1U) & 1U);
    cpu->flag_v = (uint8_t)(nzcv & 1U);
}

/// Returns true if `a` is less than `b`; neither is a NaN.
static bool
is_less(const context_t *context, const uint64_t a, const uint64_t b)
{
    if (PRECISION_SINGLE == context->precision)
    {
        return to_float(a) < to_float(b);
    }

    return operand_to_double(context, a) < operand_to_double(context, b);
}

/// Returns true if `a` equals `b`; neither is a NaN.
static bool
is_equal(const context_t *context, const uint64_t a, const uint64_t b)
{
    return (true == is_zero(&context->layout, a) && true == is_zero(&context->layout, b)) || a == b;
}

/// Compares two operands, returning the NZCV result of FCMP. `signal_all_nans` selects FCMPE.
static uint32_t
compare(context_t *context, uint64_t a, uint64_t b, const bool signal_all_nans)
{
    a = flush_input(context, a);
    b = flush_input(context, b);

    if (true == is_nan(&context->layout, a) || true == is_nan(&context->layout, b))
    {
        if (true == signal_all_nans || true == is_signalling_nan(&context->layout, a)
            || true == is_signalling_nan(&context->layout, b))
        {
            context->flags |= FPSR_IOC;
        }

        return 0x3U;
    }

    if (true == is_equal(context, a, b))
    {
        return 0x6U;
    }

    return true == is_less(context, a, b) ? 0x8U : 0x2U;
}

/// FMAX, FMIN, FMAXNM and FMINNM.
static uint64_t
minimum_maximum(
    context_t *context, uint64_t a, uint64_t b, const bool is_maximum, const bool is_number)
{
    a                      = flush_input(context, a);
    b                      = flush_input(context, b);
    const layout_t *layout = &context->layout;

    // FMAXNM and FMINNM return the number when exactly one operand is a quiet NaN.
    if (true == is_number)
    {
        const bool a_quiet = is_nan(layout, a) && false == is_signalling_nan(layout, a);
        const bool b_quiet = is_nan(layout, b) && false == is_signalling_nan(layout, b);

        if (true == a_quiet && false == is_nan(layout, b))
        {
            a = b;
        }
        else if (true == b_quiet && false == is_nan(layout, a))
        {
            b = a;
        }
    }

    const uint64_t operands[2] = { a, b };
    uint64_t       result      = 0U;

    if (true == process_nans(context, operands, 2U, &result))
    {
        return result;
    }

    if (true == is_zero(layout, a) && true == is_zero(layout, b))
    {
        // +0 is larger than -0.
        return true == is_maximum ? a & b : a | b;
    }

    const bool a_is_less = is_less(context, a, b);
    return a_is_less == is_maximum ? b : a;
}

/// Rounds to an integral value in floating point: FRINT* with `rmode` 0 (to nearest, ties to
/// even), 1 (towards +infinity), 2 (towards -infinity), 3 (towards zero), or 4 (to nearest, ties
/// away). `exact` raises Inexact for FRINTX.
static uint64_t
round_to_integral(context_t *context, uint64_t value, const uint32_t rmode, const bool exact)
{
    value                      = flush_input(context, value);
    const uint64_t operands[1] = { value };
    uint64_t       result      = 0U;

    if (true == process_nans(context, operands, 1U, &result))
    {
        return result;
    }

    if (true == is_infinity(&context->layout, value) || true == is_zero(&context->layout, value))
    {
        return value;
    }

    const int previous_mode = fegetround();
    (void)fesetround(host_rounding_mode(rmode > 3U ? 0U : rmode));
    (void)feclearexcept(FE_ALL_EXCEPT);

    if (PRECISION_HALF == context->precision)
    {
        // An integral half precision value is exactly representable.
        const double input   = half_to_double(value, false);
        const double integer = 4U == rmode ? round(input) : nearbyint(input);
        result = round_to_half(context, integer, false, 0 != signbit(integer), false, false);
        if (true == exact && integer != input)
        {
            context->flags |= FPSR_IXC;
        }
    }
    else if (PRECISION_DOUBLE == context->precision)
    {
        const double input = to_double(value);
        result             = from_double(4U == rmode ? round(input) : nearbyint(input));
        if (true == exact && to_double(result) != input)
        {
            context->flags |= FPSR_IXC;
        }
    }
    else
    {
        const float input = to_float(value);
        result            = from_float(4U == rmode ? roundf(input) : nearbyintf(input));
        if (true == exact && to_float(result) != input)
        {
            context->flags |= FPSR_IXC;
        }
    }

    (void)fesetround(previous_mode);
    return result;
}

/// FRINT32Z, FRINT32X, FRINT64Z and FRINT64X: round_to_integral() into the range of an
/// `intsize`-bit signed integer. NaNs, infinities and out of range values give the most negative
/// integer instead and raise Invalid Operation without Inexact.
static uint64_t
round_to_integral_bits(context_t     *context,
                       uint64_t       value,
                       const uint32_t rmode,
                       const uint32_t intsize)
{
    const layout_t *layout = &context->layout;
    const uint64_t  overflow
        = layout->sign_mask
          | ((uint64_t)((layout->exponent_mask >> layout->fraction_bits) / 2U + intsize - 1U)
             << layout->fraction_bits);
    const uint64_t previous = context->flags;

    value = flush_input(context, value);

    if (true == is_nan(layout, value) || true == is_infinity(layout, value))
    {
        context->flags |= FPSR_IOC;
        return overflow;
    }

    const uint64_t result = round_to_integral(context, value, rmode, true);
    const double   limit  = ldexp(1.0, (int)intsize - 1);
    const double   number = operand_to_double(context, result);

    if (number >= limit || number < -limit)
    {
        context->flags = previous | FPSR_IOC | (context->flags & FPSR_IDC);
        return overflow;
    }

    return result;
}

/// FPConvertBF(): rounds the single precision `value` to BFloat16 in the FPCR rounding mode,
/// raising the floating-point exceptions of the conversion.
static uint64_t
convert_to_bfloat16(context_t *context, uint64_t value)
{
    value = flush_input(context, value);

    if (true == is_nan(&context->layout, value))
    {
        if (true == is_signalling_nan(&context->layout, value))
        {
            context->flags |= FPSR_IOC;
        }

        return 0U != (context->fpcr & FPCR_DN) ? 0x7FC0U : ((value | 0x00400000U) >> 16U) & 0xFFFFU;
    }

    const uint64_t truncated = value >> 16U;
    const uint64_t low       = value & 0xFFFFU;

    // Infinities, zeros and values that need no rounding convert exactly.
    if (0U == low || true == is_infinity(&context->layout, value))
    {
        return truncated;
    }

    const bool is_negative = 0U != (value >> 31U);
    bool       is_round_up = false;

    switch (FPCR_RMODE(context->fpcr))
    {
        case 0U: // To nearest, ties to even.
            is_round_up = low > 0x8000U || (0x8000U == low && 0U != (truncated & 1U));
            break;
        case 1U: // Towards +infinity.
            is_round_up = false == is_negative;
            break;
        case 2U: // Towards -infinity.
            is_round_up = is_negative;
            break;
        default: // Towards zero.
            break;
    }

    const uint64_t result = truncated + (true == is_round_up ? 1U : 0U);
    context->flags |= FPSR_IXC;

    // A denormal input is tiny before rounding; rounding up the largest finite value overflows.
    if (0U == (value & context->layout.exponent_mask))
    {
        context->flags |= FPSR_UFC;
    }
    else if (0x7F80U == (result & 0x7FFFU))
    {
        context->flags |= FPSR_OFC;
    }

    return result;
}

/// Converts to a signed or unsigned integer of `bits` bits, scaling by 2^`fraction_bits` first,
/// with rounding mode `rmode` as in round_to_integral(). Out of range values and NaNs saturate and
/// raise Invalid Operation.
static uint64_t
convert_to_integer(context_t     *context,
                   uint64_t       value,
                   const uint32_t fraction_bits,
                   const uint32_t rmode,
                   const bool     is_unsigned,
                   const uint32_t bits)
{
    value = flush_input(context, value);

    if (true == is_nan(&context->layout, value))
    {
        context->flags |= FPSR_IOC;
        return 0U;
    }

    // A float widens to double exactly, and scaling by a power of two is exact until it
    // overflows to infinity, which still saturates.
    double input = operand_to_double(context, value);
    input        = ldexp(input, (int)fraction_bits);

    double rounded = 0.0;

    if (4U == rmode)
    {
        rounded = round(input);
    }
    else
    {
        const int previous_mode = fegetround();
        (void)fesetround(host_rounding_mode(rmode));
        rounded = nearbyint(input);
        (void)fesetround(previous_mode);
    }

    const double   limit   = ldexp(1.0, (int)(true == is_unsigned ? bits : bits - 1U));
    const double   minimum = true == is_unsigned ? 0.0 : -limit;
    const uint64_t mask    = bits >= 64U ? ~0ULL : (1ULL << bits) - 1U;

    // Saturate to the most negative or most positive value of the destination.
    if (rounded < minimum)
    {
        context->flags |= FPSR_IOC;
        return true == is_unsigned ? 0U : 1ULL << (bits - 1U);
    }

    if (rounded >= limit)
    {
        context->flags |= FPSR_IOC;
        return true == is_unsigned ? mask : mask >> 1U;
    }

    if (rounded != input)
    {
        context->flags |= FPSR_IXC;
    }

    if (true == is_unsigned)
    {
        return (rounded >= 9223372036854775808.0
                    ? (uint64_t)(rounded - 9223372036854775808.0) + 0x8000000000000000ULL
                    : (uint64_t)rounded)
               & mask;
    }

    return (uint64_t)(int64_t)rounded & mask;
}

/// Converts a signed or unsigned `bits`-bit integer, divided by 2^`fraction_bits`, under the
/// guest rounding mode.
static uint64_t
convert_from_integer(context_t     *context,
                     uint64_t       value,
                     const uint32_t fraction_bits,
                     const bool     is_unsigned,
                     const uint32_t bits)
{
    if (bits < 64U)
    {
        value &= (1ULL << bits) - 1U;

        if (false == is_unsigned && 0U != (value >> (bits - 1U)))
        {
            value |= ~((1ULL << bits) - 1U);
        }
    }

    const int previous_mode = fegetround();
    (void)fesetround(host_rounding_mode(FPCR_RMODE(context->fpcr)));
    (void)feclearexcept(FE_ALL_EXCEPT);
    uint64_t result = 0U;

    // Half precision converts to double towards zero, keeping a sticky bit, and rounds once to
    // half precision; the scaling is exact in double precision.
    if (PRECISION_HALF == context->precision)
    {
        (void)fesetround(FE_TOWARDZERO);
        const double converted = true == is_unsigned ? (double)value : (double)(int64_t)value;
        const bool   sticky    = 0 != fetestexcept(FE_INEXACT);
        (void)fesetround(previous_mode);
        const double scaled = ldexp(converted, -(int)fraction_bits);
        return round_to_half(context, scaled, sticky, scaled < 0.0, false, true);
    }

    // The conversion rounds once; the scaling by a power of two that follows is exact, since
    // the result stays far from the denormal range.
    if (PRECISION_DOUBLE == context->precision)
    {
        const double converted = true == is_unsigned ? (double)value : (double)(int64_t)value;
        result
            = from_double(0U == fraction_bits ? converted : ldexp(converted, -(int)fraction_bits));
    }
    else
    {
        const float converted = true == is_unsigned ? (float)value : (float)(int64_t)value;
        result
            = from_float(0U == fraction_bits ? converted : ldexpf(converted, -(int)fraction_bits));
    }

    const uint64_t flags = host_flags() & FPSR_IXC;
    (void)fesetround(previous_mode);
    context->flags |= flags;
    return result;
}

/// FCVT between half, single and double precision. Half precision operands are not flushed and
/// half precision results are not flushed (FPUnpackCV, FPRoundCV), and FPCR.AHP selects the
/// alternative half-precision format, which has no infinities or NaNs.
static uint64_t
convert_precision(context_t *context, uint64_t value, const precision_t target)
{
    const layout_t target_layout  = layout_of(target);
    const bool     is_alternative = 0U != (context->fpcr & FPCR_AHP);
    const uint64_t source_sign    = value & context->layout.sign_mask;

    if (PRECISION_HALF == context->precision)
    {
        const bool is_special
            = context->layout.exponent_mask == (value & context->layout.exponent_mask);

        if (false == is_special || true == is_alternative)
        {
            const double number = half_to_double(value, is_alternative);
            return PRECISION_DOUBLE == target ? from_double(number) : from_float((float)number);
        }
    }
    else
    {
        value = flush_input(context, value);

        if (PRECISION_HALF == target)
        {
            if (true == is_nan(&context->layout, value) && true == is_alternative)
            {
                context->flags |= FPSR_IOC;
                return 0U != source_sign ? 0x8000U : 0U;
            }

            if (true == is_infinity(&context->layout, value))
            {
                if (true == is_alternative)
                {
                    context->flags |= FPSR_IOC;
                    return (0U != source_sign ? 0x8000U : 0U) | 0x7FFFU;
                }

                return (0U != source_sign ? 0x8000U : 0U) | 0x7C00U;
            }

            if (false == is_nan(&context->layout, value))
            {
                const double number = operand_to_double(context, value);
                return round_to_half(
                    context, number, false, 0U != source_sign, is_alternative, false);
            }
        }
    }

    if (true == is_infinity(&context->layout, value))
    {
        return (0U != source_sign ? target_layout.sign_mask : 0U) | target_layout.exponent_mask;
    }

    if (true == is_nan(&context->layout, value))
    {
        if (true == is_signalling_nan(&context->layout, value))
        {
            context->flags |= FPSR_IOC;
        }

        if (0U != (context->fpcr & FPCR_DN))
        {
            return default_nan(&target_layout);
        }

        // The payload keeps its top bits; the result is quiet.
        const uint64_t sign
            = 0U != (value & context->layout.sign_mask) ? target_layout.sign_mask : 0U;
        uint64_t fraction = value & context->layout.fraction_mask;
        fraction          = target_layout.fraction_bits > context->layout.fraction_bits
                                ? fraction << (target_layout.fraction_bits - context->layout.fraction_bits)
                                : fraction >> (context->layout.fraction_bits - target_layout.fraction_bits);
        return sign | target_layout.exponent_mask | target_layout.quiet_bit | fraction;
    }

    if (PRECISION_DOUBLE == target)
    {
        return from_double((double)to_float(value));
    }

    const int previous_mode = fegetround();
    (void)fesetround(host_rounding_mode(FPCR_RMODE(context->fpcr)));
    (void)feclearexcept(FE_ALL_EXCEPT);
    const uint64_t result = from_float((float)to_double(value));
    const uint64_t flags  = host_flags();
    (void)fesetround(previous_mode);

    context->precision = PRECISION_SINGLE;
    context->layout    = target_layout;
    return finish(context, result, flags);
}

/// Expands the 8-bit immediate of FMOV (scalar, immediate) to `layout`.
static uint64_t
expand_immediate(const layout_t *layout, const uint32_t imm8)
{
    const uint32_t exponent_bits = layout->bits - layout->fraction_bits - 1U;
    const uint64_t sign          = (uint64_t)(imm8 >> 7U) << (layout->bits - 1U);
    const uint64_t b6            = (imm8 >> 6U) & 1U;
    uint64_t       exponent      = ((b6 ^ 1U) << (exponent_bits - 1U))
                        | ((0U - b6) & ((1ULL << (exponent_bits - 3U)) - 1U)) << 2U
                        | ((imm8 >> 4U) & 0x3U);
    const uint64_t fraction = (uint64_t)(imm8 & 0xFU) << (layout->fraction_bits - 4U);
    return sign | (exponent << layout->fraction_bits) | fraction;
}

/// FJCVTZS: converts a double to a 32-bit integer the JavaScript way, towards zero and modulo
/// 2^32. Sets `is_exact` when the value converts exactly, in range and is not negative zero.
static uint32_t
convert_javascript(context_t *context, uint64_t value, bool *is_exact)
{
    value                   = flush_input(context, value);
    const bool     negative = 0U != (value >> 63U);
    const uint32_t exponent = (uint32_t)((value >> 52U) & 0x7FFU);
    *is_exact               = false;

    // NaNs and infinities convert to zero.
    if (0x7FFU == exponent)
    {
        context->flags |= FPSR_IOC;
        return 0U;
    }

    // The value is mantissa * 2^shift; keep the low 32 bits of its integer part.
    const uint64_t mantissa = (value & 0xFFFFFFFFFFFFFULL) | (0U != exponent ? 1ULL << 52U : 0U);
    const int32_t  shift    = (int32_t)(0U != exponent ? exponent : 1U) - 1075;
    uint32_t       low      = 0U;
    bool           fraction = false;

    if (shift >= 0)
    {
        low = shift >= 32 ? 0U : (uint32_t)(mantissa << (uint32_t)shift);
    }
    else if (shift > -64)
    {
        low      = (uint32_t)(mantissa >> (uint32_t)-shift);
        fraction = 0U != (mantissa & ((1ULL << (uint32_t)-shift) - 1U));
    }
    else
    {
        fraction = 0U != mantissa;
    }

    const double truncated = trunc(to_double(value));

    if (truncated < -2147483648.0 || truncated > 2147483647.0)
    {
        context->flags |= FPSR_IOC;
    }
    else if (true == fraction)
    {
        context->flags |= FPSR_IXC;
    }
    else
    {
        *is_exact = false == negative || 0U != mantissa;
    }

    return true == negative ? 0U - low : low;
}

static void
execute_integer_conversion(bal_cpu_t *cpu, context_t *context, const uint32_t instruction)
{
    const uint32_t sf     = instruction >> 31U;
    const uint32_t rmode  = (instruction >> 19U) & 0x3U;
    const uint32_t opcode = (instruction >> 16U) & 0x7U;
    const uint32_t rn     = (instruction >> 5U) & 0x1FU;
    const uint32_t rd     = instruction & 0x1FU;
    const uint32_t bits   = 1U == sf ? 64U : 32U;

    if (6U == opcode && 3U == rmode)
    {
        bool           is_exact = false;
        const uint32_t result   = convert_javascript(context, read_vector(cpu, rn, 64U), &is_exact);
        write_general(cpu, rd, result);
        set_nzcv(cpu, true == is_exact ? 0x4U : 0U);
        return;
    }

    if (opcode >= 6U)
    {
        const bool to_vector = 7U == opcode;

        if (1U == rmode)
        {
            // FMOV Xd, Vn.D[1] and FMOV Vd.D[1], Xn; the low half is left unchanged.
            if (true == to_vector)
            {
                cpu->v[rd][1] = read_general(cpu, rn);
            }
            else
            {
                write_general(cpu, rd, cpu->v[rn][1]);
            }

            return;
        }

        const uint32_t width = context->layout.bits;

        if (true == to_vector)
        {
            write_vector(
                cpu, rd, read_general(cpu, rn) & ((width >= 64U ? 0U : 1ULL << width) - 1U));
        }
        else
        {
            write_general(cpu, rd, read_vector(cpu, rn, width));
        }

        return;
    }

    if (2U == opcode || 3U == opcode)
    {
        const uint64_t result
            = convert_from_integer(context, read_general(cpu, rn), 0U, 3U == opcode, bits);
        write_vector(cpu, rd, result);
        return;
    }

    // FCVTNS/NU (rmode 00, opcode 0/1), FCVTAS/AU (rmode 00, opcode 4/5), and FCVTPS/PU,
    // FCVTMS/MU, FCVTZS/ZU with rmode 01/10/11.
    const uint32_t rounding    = opcode >= 4U ? 4U : rmode;
    const bool     is_unsigned = 1U == (opcode & 1U);
    const uint64_t source      = read_vector(cpu, rn, context->layout.bits);
    write_general(cpu, rd, convert_to_integer(context, source, 0U, rounding, is_unsigned, bits));
}

static void
execute_fixed_conversion(bal_cpu_t *cpu, context_t *context, const uint32_t instruction)
{
    const uint32_t sf            = instruction >> 31U;
    const uint32_t rmode         = (instruction >> 19U) & 0x3U;
    const uint32_t opcode        = (instruction >> 16U) & 0x7U;
    const uint32_t fraction_bits = 64U - ((instruction >> 10U) & 0x3FU);
    const uint32_t rn            = (instruction >> 5U) & 0x1FU;
    const uint32_t rd            = instruction & 0x1FU;
    const uint32_t bits          = 1U == sf ? 64U : 32U;
    const bool     is_unsigned   = 1U == (opcode & 1U);

    if (0U == rmode)
    {
        write_vector(
            cpu,
            rd,
            convert_from_integer(context, read_general(cpu, rn), fraction_bits, is_unsigned, bits));
        return;
    }

    const uint64_t source = read_vector(cpu, rn, context->layout.bits);
    write_general(
        cpu, rd, convert_to_integer(context, source, fraction_bits, 3U, is_unsigned, bits));
}

static void
execute_one_source(bal_cpu_t *cpu, context_t *context, const uint32_t instruction)
{
    const uint32_t opcode = (instruction >> 15U) & 0x3FU;
    const uint32_t rn     = (instruction >> 5U) & 0x1FU;
    const uint32_t rd     = instruction & 0x1FU;
    const uint64_t value  = read_vector(cpu, rn, context->layout.bits);
    const uint64_t sign   = context->layout.sign_mask;

    switch (opcode)
    {
        case 0U:
            write_vector(cpu, rd, value); // FMOV
            return;
        case 1U:
            write_vector(cpu, rd, value & ~sign); // FABS
            return;
        case 2U:
            write_vector(cpu, rd, value ^ sign); // FNEG
            return;
        case 3U: {
            const uint64_t input       = flush_input(context, value);
            const uint64_t operands[1] = { input };
            uint64_t       result      = 0U;

            if (false == process_nans(context, operands, 1U, &result))
            {
                result = host_operation(context, OPERATION_SQUARE_ROOT, input, 0U, 0U);
            }

            write_vector(cpu, rd, result);
            return;
        }
        default:
            break;
    }

    if (6U == opcode && PRECISION_DOUBLE == context->precision)
    {
        // BFCVT: the operand is a single precision register.
        context->precision = PRECISION_SINGLE;
        context->layout    = layout_of(PRECISION_SINGLE);
        write_vector(cpu, rd, convert_to_bfloat16(context, read_vector(cpu, rn, 32U)));
        return;
    }

    if (4U == (opcode & 0x3CU))
    {
        write_vector(cpu, rd, convert_precision(context, value, (precision_t)(opcode & 0x3U)));
        return;
    }

    if (0x10U == (opcode & 0x3CU))
    {
        const uint32_t rmode = 0U == (opcode & 1U) ? 3U : FPCR_RMODE(context->fpcr);
        write_vector(
            cpu,
            rd,
            round_to_integral_bits(context, value, rmode, 0U == (opcode & 2U) ? 32U : 64U));
        return;
    }

    // FRINTN, P, M, Z (rmode 0-3), A (4), X and I (the FPCR rounding mode).
    const uint32_t rint_opcode = opcode & 0x7U;
    uint32_t       rmode       = rint_opcode;

    if (rint_opcode >= 6U)
    {
        rmode = FPCR_RMODE(context->fpcr);
    }

    write_vector(cpu, rd, round_to_integral(context, value, rmode, 6U == rint_opcode));
}

static void
execute_two_source(bal_cpu_t *cpu, context_t *context, const uint32_t instruction)
{
    const uint32_t opcode = (instruction >> 12U) & 0xFU;
    const uint32_t rm     = (instruction >> 16U) & 0x1FU;
    const uint32_t rn     = (instruction >> 5U) & 0x1FU;
    const uint32_t rd     = instruction & 0x1FU;
    const uint64_t a      = read_vector(cpu, rn, context->layout.bits);
    const uint64_t b      = read_vector(cpu, rm, context->layout.bits);

    if (opcode >= 4U && opcode <= 7U)
    {
        write_vector(cpu, rd, minimum_maximum(context, a, b, 0U == (opcode & 1U), opcode >= 6U));
        return;
    }

    static const operation_t OPERATIONS[4] = {
        OPERATION_MULTIPLY,
        OPERATION_DIVIDE,
        OPERATION_ADD,
        OPERATION_SUBTRACT,
    };
    const operation_t operation   = 8U == opcode ? OPERATION_MULTIPLY : OPERATIONS[opcode];
    const uint64_t    x           = flush_input(context, a);
    const uint64_t    y           = flush_input(context, b);
    const uint64_t    operands[2] = { x, y };
    uint64_t          result      = 0U;

    if (false == process_nans(context, operands, 2U, &result))
    {
        // Division by zero raises Divide by Zero only for a finite non-zero dividend, which the
        // host also does.
        result = host_operation(context, operation, x, y, 0U);
    }

    if (8U == opcode)
    {
        result ^= context->layout.sign_mask; // FNMUL negates the product, NaNs included.
    }

    write_vector(cpu, rd, result);
}

/// Returns addend + n * m with a single rounding, with the NaN priority of FPMulAdd(): the addend
/// first, and infinity times zero with a quiet NaN addend giving the default NaN.
static uint64_t
fused_multiply_add(context_t *context, uint64_t addend, uint64_t n, uint64_t m)
{
    addend = flush_input(context, addend);
    n      = flush_input(context, n);
    m      = flush_input(context, m);

    const layout_t *layout                 = &context->layout;
    const bool      is_infinity_times_zero = (is_infinity(layout, n) && is_zero(layout, m))
                                        || (is_zero(layout, n) && is_infinity(layout, m));
    uint64_t result = 0U;

    if (true == is_nan(layout, addend) && false == is_signalling_nan(layout, addend)
        && true == is_infinity_times_zero)
    {
        context->flags |= FPSR_IOC;
        return default_nan(layout);
    }

    const uint64_t operands[3] = { addend, n, m };

    if (false == process_nans(context, operands, 3U, &result))
    {
        result = host_operation(context, OPERATION_FUSED_MULTIPLY_ADD, n, m, addend);
    }

    return result;
}

static void
execute_three_source(bal_cpu_t *cpu, context_t *context, const uint32_t instruction)
{
    const bool     o1   = 1U == ((instruction >> 21U) & 1U);
    const bool     o0   = 1U == ((instruction >> 15U) & 1U);
    const uint32_t rm   = (instruction >> 16U) & 0x1FU;
    const uint32_t ra   = (instruction >> 10U) & 0x1FU;
    const uint32_t rn   = (instruction >> 5U) & 0x1FU;
    const uint32_t rd   = instruction & 0x1FU;
    const uint64_t sign = context->layout.sign_mask;

    // FMADD: a + n*m. FMSUB: a - n*m. FNMADD: -a - n*m. FNMSUB: -a + n*m. The negations apply to
    // the operands before NaN processing, so they flip the sign of a propagated NaN.
    uint64_t addend = read_vector(cpu, ra, context->layout.bits);
    uint64_t n      = read_vector(cpu, rn, context->layout.bits);
    uint64_t m      = read_vector(cpu, rm, context->layout.bits);
    addend          = true == o1 ? addend ^ sign : addend;
    n               = o0 != o1 ? n ^ sign : n;

    write_vector(cpu, rd, fused_multiply_add(context, addend, n, m));
}

uint64_t
bal_execute_float(bal_cpu_t *cpu, const uint32_t instruction)
{
    load_control();
    const float_class_t instruction_class = classify(instruction);
    precision_t         precision         = (precision_t)((instruction >> 22U) & 0x3U);

    // The top-half FMOV uses ftype 10 for a 64-bit lane.
    if (PRECISION_INVALID == precision)
    {
        precision = PRECISION_DOUBLE;
    }

    context_t context = {
        .fpcr      = cpu->fpcr,
        .flags     = 0U,
        .layout    = layout_of(precision),
        .precision = precision,
        .pad       = 0U,
    };

    switch (instruction_class)
    {
        case FLOAT_INTEGER_CONVERSION:
            execute_integer_conversion(cpu, &context, instruction);
            break;

        case FLOAT_FIXED_CONVERSION:
            execute_fixed_conversion(cpu, &context, instruction);
            break;

        case FLOAT_ONE_SOURCE:
            execute_one_source(cpu, &context, instruction);
            break;

        case FLOAT_COMPARE: {
            const uint32_t rm           = (instruction >> 16U) & 0x1FU;
            const uint32_t rn           = (instruction >> 5U) & 0x1FU;
            const bool     is_zero_form = 1U == ((instruction >> 3U) & 1U);
            const uint64_t a            = read_vector(cpu, rn, context.layout.bits);
            const uint64_t b
                = true == is_zero_form ? 0U : read_vector(cpu, rm, context.layout.bits);
            set_nzcv(cpu, compare(&context, a, b, 1U == ((instruction >> 4U) & 1U)));
            break;
        }

        case FLOAT_IMMEDIATE:
            write_vector(cpu,
                         instruction & 0x1FU,
                         expand_immediate(&context.layout, (instruction >> 13U) & 0xFFU));
            break;

        case FLOAT_CONDITIONAL_COMPARE: {
            const uint32_t rm        = (instruction >> 16U) & 0x1FU;
            const uint32_t condition = (instruction >> 12U) & 0xFU;
            const uint32_t rn        = (instruction >> 5U) & 0x1FU;

            if (true == condition_holds(cpu, condition))
            {
                const uint64_t a = read_vector(cpu, rn, context.layout.bits);
                const uint64_t b = read_vector(cpu, rm, context.layout.bits);
                set_nzcv(cpu, compare(&context, a, b, 1U == ((instruction >> 4U) & 1U)));
            }
            else
            {
                set_nzcv(cpu, instruction & 0xFU);
            }

            break;
        }

        case FLOAT_TWO_SOURCE:
            execute_two_source(cpu, &context, instruction);
            break;

        case FLOAT_CONDITIONAL_SELECT: {
            const uint32_t rm        = (instruction >> 16U) & 0x1FU;
            const uint32_t condition = (instruction >> 12U) & 0xFU;
            const uint32_t rn        = (instruction >> 5U) & 0x1FU;
            const uint32_t source    = true == condition_holds(cpu, condition) ? rn : rm;
            write_vector(cpu, instruction & 0x1FU, read_vector(cpu, source, context.layout.bits));
            break;
        }

        case FLOAT_THREE_SOURCE:
            execute_three_source(cpu, &context, instruction);
            break;

        case FLOAT_UNSUPPORTED:
        default:
            break;
    }

    cpu->fpsr |= context.flags;
    return 1U;
}

static precision_t
precision_of(const uint32_t esize)
{
    return 16U == esize ? PRECISION_HALF : (64U == esize ? PRECISION_DOUBLE : PRECISION_SINGLE);
}

static context_t
element_context(const uint64_t fpcr, const uint32_t esize)
{
    const precision_t precision = precision_of(esize);
    context_t         context   = {
                  .fpcr      = fpcr,
                  .flags     = 0U,
                  .layout    = layout_of(precision),
                  .precision = precision,
                  .pad       = 0U,
    };
    return context;
}

/// Returns +2.0 (FMULX, FRECPS) or +1.5 (FRSQRTS) with the given sign.
static uint64_t
constant(const context_t *context, const bool is_two, const uint64_t sign)
{
    if (PRECISION_DOUBLE == context->precision)
    {
        return sign | (true == is_two ? 0x4000000000000000ULL : 0x3FF8000000000000ULL);
    }

    if (PRECISION_HALF == context->precision)
    {
        return sign | (true == is_two ? 0x4000U : 0x3E00U);
    }

    return sign | (true == is_two ? 0x40000000ULL : 0x3FC00000ULL);
}

/// FRECPS (`is_reciprocal`) or FRSQRTS: 2 - a * b, or (3 - a * b) / 2, with a single rounding.
static uint64_t
reciprocal_step(context_t *context, uint64_t a, uint64_t b, const bool is_reciprocal)
{
    const layout_t *layout     = &context->layout;
    a                          = flush_input(context, a ^ layout->sign_mask);
    b                          = flush_input(context, b);
    const uint64_t operands[2] = { a, b };
    uint64_t       result      = 0U;

    if (true == process_nans(context, operands, 2U, &result))
    {
        return result;
    }

    if ((is_infinity(layout, a) && is_zero(layout, b))
        || (is_zero(layout, a) && is_infinity(layout, b)))
    {
        return constant(context, is_reciprocal, 0U);
    }

    if (true == is_infinity(layout, a) || true == is_infinity(layout, b))
    {
        return layout->exponent_mask | ((a ^ b) & layout->sign_mask);
    }

    if (PRECISION_HALF == context->precision)
    {
        // The halving is exact in double precision, so both forms are one fused operation there.
        const double x             = half_to_double(a, false);
        const double y             = half_to_double(b, false);
        const double addend        = true == is_reciprocal ? 2.0 : 3.0;
        const double scale         = true == is_reciprocal ? 1.0 : 0.5;
        const int    previous_mode = fegetround();
        (void)fesetround(host_rounding_mode(FPCR_RMODE(context->fpcr)));
        const double guest_result = fma(x, y, addend) * scale;
        (void)fesetround(FE_TOWARDZERO);
        (void)feclearexcept(FE_ALL_EXCEPT);
        const double truncated = fma(x, y, addend) * scale;
        const bool   sticky    = 0 != fetestexcept(FE_INEXACT);
        (void)fesetround(previous_mode);
        return round_to_half(context, truncated, sticky, 0 != signbit(guest_result), false, true);
    }

    const int previous_mode = fegetround();
    (void)fesetround(host_rounding_mode(FPCR_RMODE(context->fpcr)));
    (void)feclearexcept(FE_ALL_EXCEPT);

    // (3 - a * b) / 2 must round once, so the halving goes into an operand rather than the
    // result: halving a normal operand is exact. With both operands denormal the product is
    // negligible next to 3 and the final halving is exact instead.
    const uint64_t smallest_normal  = 1ULL << layout->fraction_bits;
    const bool     can_halve_a      = (a & ~layout->sign_mask) >= (smallest_normal << 1U);
    const bool     can_halve_b      = (b & ~layout->sign_mask) >= (smallest_normal << 1U);
    const bool     is_halved_inside = false == is_reciprocal && (can_halve_a || can_halve_b);

    if (PRECISION_DOUBLE == context->precision)
    {
        double x = to_double(a);
        double y = to_double(b);

        if (true == is_halved_inside)
        {
            x = true == can_halve_a ? x * 0.5 : x;
            y = true == can_halve_a ? y : y * 0.5;
        }

        const double value
            = fma(x, y, true == is_reciprocal ? 2.0 : (is_halved_inside ? 1.5 : 3.0));
        result = from_double(true == is_reciprocal || is_halved_inside ? value : value / 2.0);
    }
    else
    {
        float x = to_float(a);
        float y = to_float(b);

        if (true == is_halved_inside)
        {
            x = true == can_halve_a ? x * 0.5F : x;
            y = true == can_halve_a ? y : y * 0.5F;
        }

        const float value
            = fmaf(x, y, true == is_reciprocal ? 2.0F : (is_halved_inside ? 1.5F : 3.0F));
        result = from_float(true == is_reciprocal || is_halved_inside ? value : value / 2.0F);
    }

    const uint64_t flags = host_flags();
    (void)fesetround(previous_mode);
    return finish(context, result, flags);
}

/// The 9-bit reciprocal estimate of RecipEstimate(): `a` is in [256, 512).
static uint64_t
reciprocal_estimate_table(uint64_t a)
{
    a                = (a * 2U) + 1U;
    const uint64_t b = (1ULL << 19U) / a;
    return (b + 1U) / 2U;
}

/// The 9-bit reciprocal square root estimate of RecipSqrtEstimate(): `a` is in [128, 512).
static uint64_t
reciprocal_square_root_estimate_table(uint64_t a)
{
    if (a < 256U)
    {
        a = (a * 2U) + 1U;
    }
    else
    {
        a = (a >> 1U) << 1U;
        a = (a + 1U) * 2U;
    }

    // b is the smallest integer from 512 with a * (b + 1)^2 >= 2^28, so b + 1 is the ceiling of
    // sqrt(2^28 / a), at least 725 for these a. The square root is within one of it.
    uint64_t c = (uint64_t)sqrt((double)(1ULL << 28U) / (double)a);

    while (c > 0U && a * c * c >= (1ULL << 28U))
    {
        --c;
    }

    while (a * c * c < (1ULL << 28U))
    {
        ++c;
    }

    return c / 2U;
}

/// FRECPE and FRSQRTE, following FPRecipEstimate() and FPRSqrtEstimate() without FEAT_RPRES.
static uint64_t
estimate(context_t *context, uint64_t value, const bool is_square_root)
{
    const layout_t *layout     = &context->layout;
    value                      = flush_input(context, value);
    const uint64_t operands[1] = { value };
    uint64_t       result      = 0U;
    const uint64_t sign        = value & layout->sign_mask;
    const bool     is_double   = PRECISION_DOUBLE == context->precision;
    const bool     is_half     = PRECISION_HALF == context->precision;

    if (true == process_nans(context, operands, 1U, &result))
    {
        return result;
    }

    if (true == is_zero(layout, value))
    {
        context->flags |= FPSR_DZC;
        return sign | layout->exponent_mask;
    }

    if (true == is_square_root)
    {
        if (sign != 0U)
        {
            context->flags |= FPSR_IOC;
            return default_nan(layout);
        }

        if (true == is_infinity(layout, value))
        {
            return 0U;
        }
    }
    else if (true == is_infinity(layout, value))
    {
        return sign;
    }

    // The fraction is held in 52 bits whatever the precision.
    uint64_t fraction = value & layout->fraction_mask;
    int64_t  exponent = (int64_t)((value & layout->exponent_mask) >> layout->fraction_bits);
    fraction          = fraction << (52U - layout->fraction_bits);

    if (false == is_square_root)
    {
        // Too small a value (below 2^-16, 2^-128 or 2^-1024) overflows: to infinity or the
        // largest normal, by rounding mode.
        const uint64_t magnitude = value & ~layout->sign_mask;
        const bool     is_tiny   = true == is_double ? magnitude < 0x0004000000000000ULL
                                   : true == is_half ? magnitude < 0x0100U
                                                     : magnitude < 0x00200000ULL;

        if (true == is_tiny)
        {
            const uint32_t rmode = FPCR_RMODE(context->fpcr);
            const bool     is_to_infinity
                = 0U == rmode || (1U == rmode && 0U == sign) || (2U == rmode && sign != 0U);
            context->flags |= FPSR_OFC | FPSR_IXC;
            return true == is_to_infinity
                       ? sign | layout->exponent_mask
                       : sign | (layout->exponent_mask - (1ULL << layout->fraction_bits))
                             | layout->fraction_mask;
        }

        // At or above 2^14, 2^126 or 2^1022 the result is a denormal, which flushes to zero.
        const bool is_huge     = true == is_double ? magnitude >= 0x7FD0000000000000ULL
                                 : true == is_half ? magnitude >= 0x7400U
                                                   : magnitude >= 0x7E800000ULL;
        const bool is_flushing = 0U != (context->fpcr & (true == is_half ? FPCR_FZ16 : FPCR_FZ));

        if (true == is_flushing && true == is_huge)
        {
            context->flags |= FPSR_UFC;
            return sign;
        }

        if (0 == exponent)
        {
            if (0U == (fraction & (1ULL << 51U)))
            {
                exponent = -1;
                fraction = (fraction << 2U) & ((1ULL << 52U) - 1U);
            }
            else
            {
                fraction = (fraction << 1U) & ((1ULL << 52U) - 1U);
            }
        }

        const uint64_t scaled = 0x100U | (fraction >> 44U);
        int64_t        result_exponent
            = (true == is_double ? 2045 : (true == is_half ? 29 : 253)) - exponent;
        uint64_t result_fraction = (reciprocal_estimate_table(scaled) & 0xFFU) << 44U;

        if (0 == result_exponent)
        {
            result_fraction = (1ULL << 51U) | (result_fraction >> 1U);
        }
        else if (-1 == result_exponent)
        {
            result_fraction = (1ULL << 50U) | (result_fraction >> 2U);
            result_exponent = 0;
        }

        result_fraction >>= 52U - layout->fraction_bits;
        return sign | ((uint64_t)result_exponent << layout->fraction_bits) | result_fraction;
    }

    if (0 == exponent)
    {
        while (0U == (fraction & (1ULL << 51U)))
        {
            fraction = (fraction << 1U) & ((1ULL << 52U) - 1U);
            --exponent;
        }

        fraction = (fraction << 1U) & ((1ULL << 52U) - 1U);
    }

    // An even exponent scales to [256, 512), an odd one to [128, 256).
    const uint64_t scaled
        = 0 == (exponent & 1) ? 0x100U | (fraction >> 44U) : 0x80U | (fraction >> 45U);
    const int64_t result_exponent
        = ((true == is_double ? 3068 : (true == is_half ? 44 : 380)) - exponent) / 2;
    const uint64_t estimate_bits   = reciprocal_square_root_estimate_table(scaled) & 0xFFU;
    const uint64_t result_fraction = estimate_bits << (layout->fraction_bits - 8U);
    return ((uint64_t)result_exponent << layout->fraction_bits) | result_fraction;
}

/// FRECPX, following FPRecpX(): the sign with the inverted exponent and a zero fraction, or the
/// largest normal exponent for zeros and denormals.
static uint64_t
reciprocal_exponent(context_t *context, const uint64_t value)
{
    const layout_t *layout      = &context->layout;
    const uint64_t  operands[1] = { flush_input(context, value) };
    uint64_t        result      = 0U;

    if (true == process_nans(context, operands, 1U, &result))
    {
        return result;
    }

    const uint64_t sign     = value & layout->sign_mask;
    const uint64_t exponent = value & layout->exponent_mask;

    if (0U == exponent)
    {
        return sign | (layout->exponent_mask - (1ULL << layout->fraction_bits));
    }

    return sign | (~exponent & layout->exponent_mask);
}

/// FCMEQ, FCMGE, FCMGT, FACGE and FACGT: all ones when the comparison holds.
static uint64_t
compare_mask(context_t *context, uint64_t a, uint64_t b, const bal_float_operation_t operation)
{
    const layout_t *layout   = &context->layout;
    const uint64_t  all_ones = layout->bits >= 64U ? ~0ULL : (1ULL << layout->bits) - 1U;
    a                        = flush_input(context, a);
    b                        = flush_input(context, b);

    if (true == is_nan(layout, a) || true == is_nan(layout, b))
    {
        // FCMEQ is a quiet comparison; the ordered ones signal on every NaN.
        if (operation != BAL_FLOAT_COMPARE_EQUAL || true == is_signalling_nan(layout, a)
            || true == is_signalling_nan(layout, b))
        {
            context->flags |= FPSR_IOC;
        }

        return 0U;
    }

    if (BAL_FLOAT_ABSOLUTE_COMPARE_GREATER_EQUAL == operation
        || BAL_FLOAT_ABSOLUTE_COMPARE_GREATER == operation)
    {
        a &= ~layout->sign_mask;
        b &= ~layout->sign_mask;
    }

    const bool is_equal_result = is_equal(context, a, b);
    bool       holds           = false;

    switch (operation)
    {
        case BAL_FLOAT_COMPARE_EQUAL:
            holds = is_equal_result;
            break;
        case BAL_FLOAT_COMPARE_GREATER_EQUAL:
        case BAL_FLOAT_ABSOLUTE_COMPARE_GREATER_EQUAL:
            holds = is_equal_result || is_less(context, b, a);
            break;
        default:
            holds = false == is_equal_result && is_less(context, b, a);
            break;
    }

    return true == holds ? all_ones : 0U;
}

/// FMULX: a multiply where infinity times zero gives 2 with the sign of the product.
static uint64_t
multiply_extended(context_t *context, uint64_t a, uint64_t b)
{
    const layout_t *layout     = &context->layout;
    a                          = flush_input(context, a);
    b                          = flush_input(context, b);
    const uint64_t operands[2] = { a, b };
    uint64_t       result      = 0U;

    if (true == process_nans(context, operands, 2U, &result))
    {
        return result;
    }

    if ((is_infinity(layout, a) && is_zero(layout, b))
        || (is_zero(layout, a) && is_infinity(layout, b)))
    {
        return constant(context, true, (a ^ b) & layout->sign_mask);
    }

    return host_operation(context, OPERATION_MULTIPLY, a, b, 0U);
}

/// FADD, FSUB, FMUL, FDIV and FSQRT with NaN processing and FPCR.FZ.
static uint64_t
arithmetic(context_t *context, const operation_t operation, uint64_t a, uint64_t b)
{
    const uint32_t count       = OPERATION_SQUARE_ROOT == operation ? 1U : 2U;
    a                          = flush_input(context, a);
    b                          = 2U == count ? flush_input(context, b) : 0U;
    const uint64_t operands[2] = { a, b };
    uint64_t       result      = 0U;

    if (true == process_nans(context, operands, count, &result))
    {
        return result;
    }

    return host_operation(context, operation, a, b, 0U);
}

uint64_t
bal_float_element(const uint64_t              fpcr,
                  uint64_t                   *fpsr,
                  const uint32_t              esize,
                  const bal_float_operation_t operation,
                  const uint64_t              a,
                  const uint64_t              b,
                  const uint64_t              c)
{
    load_control();
    context_t context = element_context(fpcr, esize);
    uint64_t  result  = 0U;

    switch (operation)
    {
        case BAL_FLOAT_ADD:
            result = arithmetic(&context, OPERATION_ADD, a, b);
            break;
        case BAL_FLOAT_SUBTRACT:
            result = arithmetic(&context, OPERATION_SUBTRACT, a, b);
            break;
        case BAL_FLOAT_MULTIPLY:
            result = arithmetic(&context, OPERATION_MULTIPLY, a, b);
            break;
        case BAL_FLOAT_DIVIDE:
            result = arithmetic(&context, OPERATION_DIVIDE, a, b);
            break;
        case BAL_FLOAT_SQUARE_ROOT:
            result = arithmetic(&context, OPERATION_SQUARE_ROOT, a, 0U);
            break;
        case BAL_FLOAT_MAXIMUM:
        case BAL_FLOAT_MINIMUM:
        case BAL_FLOAT_MAXIMUM_NUMBER:
        case BAL_FLOAT_MINIMUM_NUMBER:
            result = minimum_maximum(
                &context,
                a,
                b,
                BAL_FLOAT_MAXIMUM == operation || BAL_FLOAT_MAXIMUM_NUMBER == operation,
                BAL_FLOAT_MAXIMUM_NUMBER == operation || BAL_FLOAT_MINIMUM_NUMBER == operation);
            break;
        case BAL_FLOAT_MULTIPLY_EXTENDED:
            result = multiply_extended(&context, a, b);
            break;
        case BAL_FLOAT_MULTIPLY_ADD:
            result = fused_multiply_add(&context, c, a, b);
            break;
        case BAL_FLOAT_ABSOLUTE_DIFFERENCE:
            result = arithmetic(&context, OPERATION_SUBTRACT, a, b) & ~context.layout.sign_mask;
            break;
        case BAL_FLOAT_RECIPROCAL_STEP:
            result = reciprocal_step(&context, a, b, true);
            break;
        case BAL_FLOAT_RECIPROCAL_SQUARE_ROOT_STEP:
            result = reciprocal_step(&context, a, b, false);
            break;
        case BAL_FLOAT_RECIPROCAL_ESTIMATE:
            result = estimate(&context, a, false);
            break;
        case BAL_FLOAT_RECIPROCAL_SQUARE_ROOT_ESTIMATE:
            result = estimate(&context, a, true);
            break;
        case BAL_FLOAT_RECIPROCAL_EXPONENT:
            result = reciprocal_exponent(&context, a);
            break;
        default:
            result = compare_mask(&context, a, b, operation);
            break;
    }

    *fpsr |= context.flags;
    return result;
}

uint64_t
bal_float_round_element(const uint64_t fpcr,
                        uint64_t      *fpsr,
                        const uint32_t esize,
                        const uint64_t value,
                        const uint32_t rmode,
                        const bool     exact)
{
    load_control();
    context_t      context = element_context(fpcr, esize);
    const uint64_t result  = round_to_integral(&context, value, rmode, exact);
    *fpsr |= context.flags;
    return result;
}

uint64_t
bal_float_to_bfloat16_element(const uint64_t fpcr, uint64_t *fpsr, const uint64_t value)
{
    load_control();
    context_t      context = element_context(fpcr, 32U);
    const uint64_t result  = convert_to_bfloat16(&context, value);
    *fpsr |= context.flags;
    return result;
}

uint64_t
bal_float_round_bits_element(const uint64_t fpcr,
                             uint64_t      *fpsr,
                             const uint32_t esize,
                             const uint64_t value,
                             const bool     is_exact,
                             const uint32_t intsize)
{
    load_control();
    context_t      context = element_context(fpcr, esize);
    const uint32_t rmode   = true == is_exact ? FPCR_RMODE(fpcr) : 3U;
    const uint64_t result  = round_to_integral_bits(&context, value, rmode, intsize);
    *fpsr |= context.flags;
    return result;
}

uint64_t
bal_float_to_integer_element(const uint64_t fpcr,
                             uint64_t      *fpsr,
                             const uint32_t esize,
                             const uint64_t value,
                             const uint32_t fraction_bits,
                             const uint32_t rmode,
                             const bool     is_unsigned,
                             const uint32_t bits)
{
    load_control();
    context_t      context = element_context(fpcr, esize);
    const uint64_t result
        = convert_to_integer(&context, value, fraction_bits, rmode, is_unsigned, bits);
    *fpsr |= context.flags;
    return result;
}

uint64_t
bal_float_from_integer_element(const uint64_t fpcr,
                               uint64_t      *fpsr,
                               const uint32_t esize,
                               const uint64_t value,
                               const uint32_t fraction_bits,
                               const bool     is_unsigned,
                               const uint32_t bits)
{
    load_control();
    context_t      context = element_context(fpcr, esize);
    const uint64_t result = convert_from_integer(&context, value, fraction_bits, is_unsigned, bits);
    *fpsr |= context.flags;
    return result;
}

uint64_t
bal_float_convert_element(const uint64_t fpcr,
                          uint64_t      *fpsr,
                          const uint32_t source_esize,
                          const uint32_t target_esize,
                          const uint64_t value)
{
    load_control();
    context_t      context = element_context(fpcr, source_esize);
    const uint64_t result  = convert_precision(&context, value, precision_of(target_esize));
    *fpsr |= context.flags;
    return result;
}

uint32_t
bal_unsigned_estimate_element(const uint32_t value, const bool is_square_root)
{
    // UnsignedRecipEstimate() saturates below 0.5, UnsignedRSqrtEstimate() below 0.25.
    if (false == is_square_root)
    {
        return 0U == (value >> 31U)
                   ? 0xFFFFFFFFU
                   : (uint32_t)(reciprocal_estimate_table(value >> 23U) & 0x1FFU) << 23U;
    }

    return 0U == (value >> 30U)
               ? 0xFFFFFFFFU
               : (uint32_t)(reciprocal_square_root_estimate_table(value >> 23U) & 0x1FFU) << 23U;
}

uint64_t
bal_float_narrow_odd_element(const uint64_t fpcr, uint64_t *fpsr, const uint64_t value)
{
    load_control();
    // Rounding to odd truncates and then sets the lowest bit of an inexact result. Overflow
    // gives the largest normal as truncation does, and a flushed result is exact zero.
    context_t      context = element_context((fpcr & ~(3ULL << 22U)) | (3ULL << 22U), 64U);
    uint64_t       result  = convert_precision(&context, value, PRECISION_SINGLE);
    const uint64_t flags   = context.flags;

    if (0U != (flags & FPSR_IXC))
    {
        result |= 1U;
    }

    *fpsr |= flags;
    return result;
}

/*** end of file ***/
