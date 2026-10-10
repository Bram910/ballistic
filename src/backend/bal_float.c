#include "backend/bal_float.h"

#include <fenv.h>
#include <math.h>
#include <string.h>

// The helper reads and changes the host floating-point environment, so the compiler must not
// constant-fold, reorder or speculate floating-point operations across the fenv calls.
#pragma STDC FENV_ACCESS ON

#define FPCR_DN       (1ULL << 25U)
#define FPCR_FZ       (1ULL << 24U)
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
            // directed ones with rmode 01-11. Half precision is not implemented yet.
            if (is_half || PRECISION_INVALID == precision)
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

            if (is_half || (0U == sf && scale < 32U))
            {
                return false;
            }

            return (0U == rmode && (2U == opcode || 3U == opcode)) || (3U == rmode && opcode <= 1U);
        }

        case FLOAT_ONE_SOURCE: {
            const uint32_t opcode = (instruction >> 15U) & 0x3FU;

            if (is_half)
            {
                return opcode <= 2U; // FMOV, FABS, FNEG
            }

            // FCVT between single and double precision, not to or from half precision.
            if (4U == (opcode & 0x3CU))
            {
                const uint32_t target = opcode & 0x3U;
                return target != (uint32_t)precision && target <= 1U;
            }

            return opcode <= 3U || (opcode >= 8U && opcode <= 15U && opcode != 13U);
        }

        case FLOAT_COMPARE:
            return false == is_half && 0U == ((instruction >> 14U) & 0x3U)
                   && 0U == (instruction & 0x7U);

        case FLOAT_IMMEDIATE:
        case FLOAT_CONDITIONAL_SELECT:
            return true;

        case FLOAT_CONDITIONAL_COMPARE:
        case FLOAT_THREE_SOURCE:
            return false == is_half;

        case FLOAT_TWO_SOURCE:
            return false == is_half && ((instruction >> 12U) & 0xFU) <= 8U;

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

/// Flushes a denormal input to zero when FPCR.FZ is set, raising Input Denormal.
static uint64_t
flush_input(context_t *context, const uint64_t value)
{
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

/// Runs one host operation under the guest rounding mode and returns its result and flags.
static uint64_t
host_operation(context_t        *context,
               const operation_t operation,
               const uint64_t    a,
               const uint64_t    b,
               const uint64_t    c)
{
    const int previous_mode = fegetround();
    (void)fesetround(host_rounding_mode(FPCR_RMODE(context->fpcr)));
    (void)feclearexcept(FE_ALL_EXCEPT);
    uint64_t result = 0U;

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

        result = from_double(value);
    }
    else
    {
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

        result = from_float(value);
    }

    const uint64_t flags = host_flags();
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
    if (PRECISION_DOUBLE == context->precision)
    {
        return to_double(a) < to_double(b);
    }

    return to_float(a) < to_float(b);
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

    if (PRECISION_DOUBLE == context->precision)
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
    double input
        = PRECISION_DOUBLE == context->precision ? to_double(value) : (double)to_float(value);
    input = ldexp(input, (int)fraction_bits);

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

    // The conversion rounds once; the scaling by a power of two that follows is exact, since
    // the result stays far from the denormal range.
    if (PRECISION_DOUBLE == context->precision)
    {
        const double converted = true == is_unsigned ? (double)value : (double)(int64_t)value;
        result                 = from_double(ldexp(converted, -(int)fraction_bits));
    }
    else
    {
        const float converted = true == is_unsigned ? (float)value : (float)(int64_t)value;
        result                = from_float(ldexpf(converted, -(int)fraction_bits));
    }

    const uint64_t flags = host_flags() & FPSR_IXC;
    (void)fesetround(previous_mode);
    context->flags |= flags;
    return result;
}

/// FCVT between single and double precision.
static uint64_t
convert_precision(context_t *context, uint64_t value, const precision_t target)
{
    value                        = flush_input(context, value);
    const layout_t target_layout = layout_of(target);

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

static void
execute_integer_conversion(bal_cpu_t *cpu, context_t *context, const uint32_t instruction)
{
    const uint32_t sf     = instruction >> 31U;
    const uint32_t rmode  = (instruction >> 19U) & 0x3U;
    const uint32_t opcode = (instruction >> 16U) & 0x7U;
    const uint32_t rn     = (instruction >> 5U) & 0x1FU;
    const uint32_t rd     = instruction & 0x1FU;
    const uint32_t bits   = 1U == sf ? 64U : 32U;

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

    if (4U == (opcode & 0x3CU))
    {
        write_vector(cpu, rd, convert_precision(context, value, (precision_t)(opcode & 0x3U)));
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
        write_vector(cpu, rd, default_nan(layout));
        return;
    }

    const uint64_t operands[3] = { addend, n, m };

    if (false == process_nans(context, operands, 3U, &result))
    {
        result = host_operation(context, OPERATION_FUSED_MULTIPLY_ADD, n, m, addend);
    }

    write_vector(cpu, rd, result);
}

uint64_t
bal_execute_float(bal_cpu_t *cpu, const uint32_t instruction)
{
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

/*** end of file ***/
