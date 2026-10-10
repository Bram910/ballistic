#include "backend/bal_simd.h"

#include "backend/bal_float.h"
#include <string.h>

#define FPSR_QC (1ULL << 27U)

/// A 128-bit SIMD&FP register: d[0] holds bits 63:0 and d[1] bits 127:64.
typedef struct
{
    uint64_t d[2];
} vector_t;

/// The state one instruction works on.
typedef struct
{
    bal_cpu_t *cpu;
    /// FPSR flags raised so far, including QC.
    uint64_t flags;
} simd_t;

static uint64_t
mask_of(const uint32_t esize)
{
    return esize >= 64U ? ~0ULL : (1ULL << esize) - 1U;
}

static int64_t
sign_extend(const uint64_t value, const uint32_t esize)
{
    const uint32_t shift = 64U - esize;
    return (int64_t)(value << shift) >> shift;
}

static uint64_t
get_element(const vector_t *vector, const uint32_t index, const uint32_t esize)
{
    const uint32_t bit = index * esize;
    return (vector->d[bit / 64U] >> (bit % 64U)) & mask_of(esize);
}

static void
set_element(vector_t *vector, const uint32_t index, const uint32_t esize, const uint64_t value)
{
    const uint32_t bit  = index * esize;
    const uint64_t mask = mask_of(esize) << (bit % 64U);
    uint64_t      *word = &vector->d[bit / 64U];
    *word               = (*word & ~mask) | ((value << (bit % 64U)) & mask);
}

static vector_t
read_vector(const simd_t *simd, const uint32_t index)
{
    vector_t vector = { { simd->cpu->v[index][0], simd->cpu->v[index][1] } };
    return vector;
}

/// Writes `datasize` bits of `vector` to register `index`, clearing the bits above them.
static void
write_vector(simd_t *simd, const uint32_t index, vector_t vector, const uint32_t datasize)
{
    if (datasize < 128U)
    {
        vector.d[1] = 0U;
        vector.d[0] &= mask_of(datasize);
    }

    simd->cpu->v[index][0] = vector.d[0];
    simd->cpu->v[index][1] = vector.d[1];
}

static uint64_t
read_general(const simd_t *simd, const uint32_t index)
{
    return 31U == index ? 0U : simd->cpu->x[index];
}

static void
write_general(simd_t *simd, const uint32_t index, const uint64_t value)
{
    if (index != 31U)
    {
        simd->cpu->x[index] = value;
    }
}

/// Saturates a signed value to `esize` bits, setting QC when it does not fit.
static uint64_t
saturate_signed(simd_t *simd, const int64_t value, const uint32_t esize)
{
    const int64_t maximum = (int64_t)(mask_of(esize) >> 1U);
    const int64_t minimum = -maximum - 1;

    if (value > maximum)
    {
        simd->flags |= FPSR_QC;
        return (uint64_t)maximum & mask_of(esize);
    }

    if (value < minimum)
    {
        simd->flags |= FPSR_QC;
        return (uint64_t)minimum & mask_of(esize);
    }

    return (uint64_t)value & mask_of(esize);
}

/// Saturates an unsigned value to `esize` bits, setting QC when it does not fit.
static uint64_t
saturate_unsigned(simd_t *simd, const uint64_t value, const uint32_t esize)
{
    if (value > mask_of(esize))
    {
        simd->flags |= FPSR_QC;
        return mask_of(esize);
    }

    return value;
}

/// Saturates a signed value to an unsigned `esize`-bit result (SQXTUN, SQSHLU).
static uint64_t
saturate_signed_to_unsigned(simd_t *simd, const int64_t value, const uint32_t esize)
{
    if (value < 0)
    {
        simd->flags |= FPSR_QC;
        return 0U;
    }

    return saturate_unsigned(simd, (uint64_t)value, esize);
}

/// SQADD/UQADD (`is_subtract` false) and SQSUB/UQSUB on `esize`-bit elements.
static uint64_t
saturating_add(simd_t        *simd,
               const uint64_t a,
               const uint64_t b,
               const uint32_t esize,
               const bool     is_unsigned,
               const bool     is_subtract)
{
    if (true == is_unsigned)
    {
        if (true == is_subtract)
        {
            if (a < b)
            {
                simd->flags |= FPSR_QC;
                return 0U;
            }

            return a - b;
        }

        const uint64_t sum = (a + b) & mask_of(esize);

        if (sum < a)
        {
            simd->flags |= FPSR_QC;
            return mask_of(esize);
        }

        return sum;
    }

    const int64_t x = sign_extend(a, esize);
    const int64_t y = sign_extend(b, esize);

    if (esize < 64U)
    {
        return saturate_signed(simd, true == is_subtract ? x - y : x + y, esize);
    }

    int64_t    result   = 0;
    const bool overflow = true == is_subtract ? __builtin_sub_overflow(x, y, &result)
                                              : __builtin_add_overflow(x, y, &result);

    if (true == overflow)
    {
        simd->flags |= FPSR_QC;
        return x < 0 ? 0x8000000000000000ULL : 0x7FFFFFFFFFFFFFFFULL;
    }

    return (uint64_t)result;
}

/// SUQADD (`is_unsigned` false): the signed accumulator `a` plus the unsigned `b`, saturated to
/// a signed result; USQADD: the unsigned `a` plus the signed `b`, saturated to an unsigned one.
static uint64_t
saturating_accumulate(
    simd_t *simd, const uint64_t a, const uint64_t b, const uint32_t esize, const bool is_unsigned)
{
    if (esize < 64U)
    {
        const int64_t sum = true == is_unsigned ? (int64_t)a + sign_extend(b, esize)
                                                : sign_extend(a, esize) + (int64_t)b;
        return true == is_unsigned ? saturate_signed_to_unsigned(simd, sum, esize)
                                   : saturate_signed(simd, sum, esize);
    }

    if (false == is_unsigned)
    {
        // a + b overflows exactly when b exceeds INT64_MAX - a, which fits in 64 bits for any a.
        if (b > 0x7FFFFFFFFFFFFFFFULL - a)
        {
            simd->flags |= FPSR_QC;
            return 0x7FFFFFFFFFFFFFFFULL;
        }

        return a + b;
    }

    if ((int64_t)b < 0)
    {
        const uint64_t magnitude = 0U - b;

        if (magnitude > a)
        {
            simd->flags |= FPSR_QC;
            return 0U;
        }

        return a - magnitude;
    }

    if (a + b < a)
    {
        simd->flags |= FPSR_QC;
        return ~0ULL;
    }

    return a + b;
}

/// Shifts an `esize`-bit element by a signed amount: left when positive, right when negative,
/// as SSHL, USHL, SRSHL, URSHL, SQSHL, UQSHL, SQRSHL, UQRSHL and the shifts by immediate do.
/// `saturate_unsigned_result` gives SQSHLU: a signed input saturated to an unsigned result.
static uint64_t
shift_element(simd_t        *simd,
              const uint64_t value,
              const uint32_t esize,
              const int64_t  shift,
              const bool     is_unsigned,
              const bool     is_rounding,
              const bool     is_saturating,
              const bool     saturate_unsigned_result)
{
    const uint64_t mask = mask_of(esize);

    if (shift >= 0)
    {
        if (false == is_saturating)
        {
            return shift >= (int64_t)esize ? 0U : (value << shift) & mask;
        }

        if (true == is_unsigned || true == saturate_unsigned_result)
        {
            if (true == saturate_unsigned_result && sign_extend(value, esize) < 0)
            {
                simd->flags |= FPSR_QC;
                return 0U;
            }

            if (0U == value)
            {
                return 0U;
            }

            if (shift >= (int64_t)esize || value > (mask >> shift))
            {
                simd->flags |= FPSR_QC;
                return mask;
            }

            return (value << shift) & mask;
        }

        const int64_t x = sign_extend(value, esize);

        if (0 == x)
        {
            return 0U;
        }

        const int64_t maximum = (int64_t)(mask >> 1U);
        const int64_t minimum = -maximum - 1;

        if (shift >= (int64_t)esize || x > (maximum >> shift) || x < (minimum >> shift))
        {
            simd->flags |= FPSR_QC;
            return (uint64_t)(x < 0 ? minimum : maximum) & mask;
        }

        return ((uint64_t)x << shift) & mask;
    }

    // Right shifts: (x + round) >> r, computed without the 65-bit intermediate as
    // (x >> r) + bit r-1 of x.
    const int64_t r = -shift;

    if (true == is_unsigned)
    {
        uint64_t result = r >= 64 ? 0U : value >> r;

        if (true == is_rounding && r - 1 < 64)
        {
            result += (value >> (r - 1)) & 1U;
        }

        return result & mask;
    }

    const int64_t x      = sign_extend(value, esize);
    int64_t       result = r >= 64 ? (x < 0 ? -1 : 0) : x >> r;

    if (true == is_rounding)
    {
        result += r - 1 >= 64 ? (x < 0 ? 1 : 0) : (int64_t)(((uint64_t)x >> (r - 1)) & 1U);
    }

    return (uint64_t)result & mask;
}

static uint64_t
count_leading_zeros(const uint64_t value, const uint32_t esize)
{
    if (0U == value)
    {
        return esize;
    }

    return (uint64_t)__builtin_clzll(value) - (64U - esize);
}

static uint64_t
reverse_bits(uint64_t value, const uint32_t esize)
{
    uint64_t result = 0U;

    for (uint32_t i = 0U; i < esize; ++i)
    {
        result = (result << 1U) | (value & 1U);
        value >>= 1U;
    }

    return result;
}

/// Polynomial multiply of two `esize`-bit values, keeping the low `result_bits` bits.
static uint64_t
polynomial_multiply(const uint64_t a,
                    const uint64_t b,
                    const uint32_t esize,
                    const uint32_t result_bits)
{
    uint64_t result = 0U;

    for (uint32_t i = 0U; i < esize; ++i)
    {
        if (0U != ((b >> i) & 1U))
        {
            result ^= a << i;
        }
    }

    return result & mask_of(result_bits);
}

/// SQDMULH and SQRDMULH (`is_rounding`) on 16- or 32-bit elements: the high half of twice the
/// product.
static uint64_t
doubling_multiply_high(
    simd_t *simd, const uint64_t a, const uint64_t b, const uint32_t esize, const bool is_rounding)
{
    // (2 * a * b + round) >> esize, halved so that it fits in 64 bits.
    const int64_t product = sign_extend(a, esize) * sign_extend(b, esize);
    const int64_t round   = true == is_rounding ? (int64_t)1 << (esize - 2U) : 0;
    return saturate_signed(simd, (product + round) >> (esize - 1U), esize);
}

/// SQRDMLAH and SQRDMLSH (`is_subtract`): the accumulator plus or minus the rounded high half
/// of twice the product, saturated once.
static uint64_t
doubling_multiply_accumulate(simd_t        *simd,
                             const uint64_t accumulator,
                             const uint64_t a,
                             const uint64_t b,
                             const uint32_t esize,
                             const bool     is_subtract)
{
    // ((accumulator << esize) +/- 2 * a * b + (1 << (esize - 1))) >> esize, halved.
    const int64_t product = sign_extend(a, esize) * sign_extend(b, esize);
    const int64_t base    = sign_extend(accumulator, esize) * ((int64_t)1 << (esize - 1U));
    const int64_t sum
        = (true == is_subtract ? base - product : base + product) + ((int64_t)1 << (esize - 2U));
    return saturate_signed(simd, sum >> (esize - 1U), esize);
}

/// SQDMULL on 16- or 32-bit elements: twice the product, as a double-width element.
static uint64_t
doubling_multiply_long(simd_t *simd, const uint64_t a, const uint64_t b, const uint32_t esize)
{
    const int64_t product = sign_extend(a, esize) * sign_extend(b, esize);

    // Only the square of the most negative value overflows when doubled.
    if (product == (int64_t)1 << (2U * esize - 2U))
    {
        simd->flags |= FPSR_QC;
        return mask_of(2U * esize) >> 1U;
    }

    return (uint64_t)(product * 2) & mask_of(2U * esize);
}

/// SQDMULL, SQDMLAL and SQDMLSL by their three different opcode (1101, 1001, 1011): the
/// doubled product, alone or saturating-added to or subtracted from the wide accumulator.
static uint64_t
doubling_long_operation(simd_t        *simd,
                        const uint32_t opcode,
                        const uint64_t accumulator,
                        const uint64_t a,
                        const uint64_t b,
                        const uint32_t esize)
{
    const uint64_t product = doubling_multiply_long(simd, a, b, esize);

    if (0xDU == opcode)
    {
        return product;
    }

    return saturating_add(simd, accumulator, product, 2U * esize, false, 0xBU == opcode);
}

/// SDOT and UDOT on one 32-bit element: the accumulator plus the four byte products.
static uint64_t
dot_product(const uint64_t accumulator, const uint64_t a, const uint64_t b, const bool is_unsigned)
{
    uint64_t sum = accumulator;

    for (uint32_t i = 0U; i < 4U; ++i)
    {
        const uint64_t x = (a >> (8U * i)) & 0xFFU;
        const uint64_t y = (b >> (8U * i)) & 0xFFU;
        sum += true == is_unsigned ? x * y : (uint64_t)(sign_extend(x, 8U) * sign_extend(y, 8U));
    }

    return sum & 0xFFFFFFFFU;
}

/// Evaluates the integer comparisons CMGT, CMGE, CMHI, CMHS, CMEQ and CMTST.
static uint64_t
compare_integers(const uint32_t kind, const uint64_t a, const uint64_t b, const uint32_t esize)
{
    bool holds = false;

    switch (kind)
    {
        case 0U: // CMGT
            holds = sign_extend(a, esize) > sign_extend(b, esize);
            break;
        case 1U: // CMGE
            holds = sign_extend(a, esize) >= sign_extend(b, esize);
            break;
        case 2U: // CMHI
            holds = a > b;
            break;
        case 3U: // CMHS
            holds = a >= b;
            break;
        case 4U: // CMEQ
            holds = a == b;
            break;
        case 5U: // CMTST
            holds = 0U != (a & b);
            break;
        case 6U: // CMLE (zero)
            holds = sign_extend(a, esize) <= sign_extend(b, esize);
            break;
        default: // CMLT (zero)
            holds = sign_extend(a, esize) < sign_extend(b, esize);
            break;
    }

    return true == holds ? mask_of(esize) : 0U;
}

/// The integer operations of the three same group. Returns false for encodings it does not
/// implement. `accumulator` is the destination element for MLA, MLS, SABA, UABA.
static bool
integer_three_same_element(simd_t        *simd,
                           const uint32_t u,
                           const uint32_t opcode,
                           const uint64_t a,
                           const uint64_t b,
                           const uint64_t accumulator,
                           const uint32_t esize,
                           uint64_t      *result)
{
    const uint64_t mask        = mask_of(esize);
    const bool     is_unsigned = 1U == u;
    const int64_t  x           = sign_extend(a, esize);
    const int64_t  y           = sign_extend(b, esize);

    switch (opcode)
    {
        case 0x00U: // SHADD, UHADD
        case 0x02U: // SRHADD, URHADD
        case 0x04U: // SHSUB, UHSUB
        {
            const uint64_t round = 0x02U == opcode ? 1U : 0U;

            if (0x04U == opcode)
            {
                *result = (true == is_unsigned ? (a - b) >> 1U : (uint64_t)((x - y) >> 1)) & mask;

                // a - b needs esize + 1 bits; for 64-bit elements recover the borrow.
                if (64U == esize)
                {
                    const uint64_t half = (a >> 1U) - (b >> 1U) - ((~a & b) & 1U);
                    *result             = true == is_unsigned
                                              ? half
                                              : (uint64_t)((x >> 1) - (y >> 1) - (int64_t)((~a & b) & 1U));
                }

                return true;
            }

            // (a + b + round) >> 1 without overflow.
            *result = true == is_unsigned
                          ? (a >> 1U) + (b >> 1U) + (((a & 1U) + (b & 1U) + round) >> 1U)
                          : (uint64_t)((x >> 1) + (y >> 1)
                                       + (int64_t)(((a & 1U) + (b & 1U) + round) >> 1U));
            *result &= mask;
            return true;
        }
        case 0x01U: // SQADD, UQADD
            *result = saturating_add(simd, a, b, esize, is_unsigned, false);
            return true;
        case 0x05U: // SQSUB, UQSUB
            *result = saturating_add(simd, a, b, esize, is_unsigned, true);
            return true;
        case 0x06U: // CMGT, CMHI
            *result = compare_integers(true == is_unsigned ? 2U : 0U, a, b, esize);
            return true;
        case 0x07U: // CMGE, CMHS
            *result = compare_integers(true == is_unsigned ? 3U : 1U, a, b, esize);
            return true;
        case 0x08U: // SSHL, USHL
        case 0x09U: // SQSHL, UQSHL
        case 0x0AU: // SRSHL, URSHL
        case 0x0BU: // SQRSHL, UQRSHL
            *result = shift_element(simd,
                                    a,
                                    esize,
                                    sign_extend(b & 0xFFU, 8U),
                                    is_unsigned,
                                    0U != (opcode & 0x2U),
                                    0U != (opcode & 0x1U),
                                    false);
            return true;
        case 0x0CU: // SMAX, UMAX
            *result = (true == is_unsigned ? a > b : x > y) ? a : b;
            return true;
        case 0x0DU: // SMIN, UMIN
            *result = (true == is_unsigned ? a < b : x < y) ? a : b;
            return true;
        case 0x0EU: // SABD, UABD
        case 0x0FU: // SABA, UABA
        {
            const bool     is_a_larger = true == is_unsigned ? a > b : x > y;
            const uint64_t difference  = (true == is_a_larger ? a - b : b - a) & mask;
            *result = 0x0FU == opcode ? (accumulator + difference) & mask : difference;
            return true;
        }
        case 0x10U: // ADD, SUB
            *result = (true == is_unsigned ? a - b : a + b) & mask;
            return true;
        case 0x11U: // CMTST, CMEQ
            *result = compare_integers(true == is_unsigned ? 4U : 5U, a, b, esize);
            return true;
        case 0x12U: // MLA, MLS
            *result = (true == is_unsigned ? accumulator - a * b : accumulator + a * b) & mask;
            return true;
        case 0x13U: // MUL, PMUL
            *result
                = true == is_unsigned ? polynomial_multiply(a, b, esize, esize) : (a * b) & mask;
            return true;
        case 0x16U: // SQDMULH, SQRDMULH
            *result = doubling_multiply_high(simd, a, b, esize, is_unsigned);
            return true;
        default:
            return false;
    }
}

/// Returns true if the three same `opcode` with `u` exists in the Advanced SIMD scalar group
/// for `size`.
static bool
is_scalar_three_same(const uint32_t u, const uint32_t opcode, const uint32_t size)
{
    switch (opcode)
    {
        case 0x01U: // SQADD, UQADD
        case 0x05U: // SQSUB, UQSUB
        case 0x09U: // SQSHL, UQSHL
        case 0x0BU: // SQRSHL, UQRSHL
            return true;
        case 0x06U: // CMGT, CMHI
        case 0x07U: // CMGE, CMHS
        case 0x08U: // SSHL, USHL
        case 0x0AU: // SRSHL, URSHL
        case 0x10U: // ADD, SUB
        case 0x11U: // CMTST, CMEQ
            return 3U == size;
        case 0x16U: // SQDMULH, SQRDMULH
            return 1U == size || 2U == size;
        default:
            (void)u;
            return false;
    }
}

/// The floating-point operations of the three same group, by U, a (size bit 1) and opcode
/// bits 2:0 of the 11xxx opcodes. Returns false for encodings it does not implement.
static bool
float_three_same_operation(const uint32_t         u,
                           const uint32_t         a,
                           const uint32_t         opcode,
                           bal_float_operation_t *operation,
                           bool                  *is_pairwise,
                           bool                  *is_accumulate)
{
    *is_pairwise       = false;
    *is_accumulate     = false;
    const uint32_t key = (u << 4U) | (a << 3U) | (opcode & 0x7U);

    switch (key)
    {
        case 0x00U:
            *operation = BAL_FLOAT_MAXIMUM_NUMBER;
            return true;
        case 0x01U:
            *operation     = BAL_FLOAT_MULTIPLY_ADD;
            *is_accumulate = true;
            return true;
        case 0x02U:
            *operation = BAL_FLOAT_ADD;
            return true;
        case 0x03U:
            *operation = BAL_FLOAT_MULTIPLY_EXTENDED;
            return true;
        case 0x04U:
            *operation = BAL_FLOAT_COMPARE_EQUAL;
            return true;
        case 0x06U:
            *operation = BAL_FLOAT_MAXIMUM;
            return true;
        case 0x07U:
            *operation = BAL_FLOAT_RECIPROCAL_STEP;
            return true;
        case 0x08U:
            *operation = BAL_FLOAT_MINIMUM_NUMBER;
            return true;
        case 0x09U: // FMLS: the multiply-add of a negated first operand.
            *operation     = BAL_FLOAT_MULTIPLY_ADD;
            *is_accumulate = true;
            return true;
        case 0x0AU:
            *operation = BAL_FLOAT_SUBTRACT;
            return true;
        case 0x0EU:
            *operation = BAL_FLOAT_MINIMUM;
            return true;
        case 0x0FU:
            *operation = BAL_FLOAT_RECIPROCAL_SQUARE_ROOT_STEP;
            return true;
        case 0x10U:
            *operation   = BAL_FLOAT_MAXIMUM_NUMBER;
            *is_pairwise = true;
            return true;
        case 0x12U:
            *operation   = BAL_FLOAT_ADD;
            *is_pairwise = true;
            return true;
        case 0x13U:
            *operation = BAL_FLOAT_MULTIPLY;
            return true;
        case 0x14U:
            *operation = BAL_FLOAT_COMPARE_GREATER_EQUAL;
            return true;
        case 0x15U:
            *operation = BAL_FLOAT_ABSOLUTE_COMPARE_GREATER_EQUAL;
            return true;
        case 0x16U:
            *operation   = BAL_FLOAT_MAXIMUM;
            *is_pairwise = true;
            return true;
        case 0x17U:
            *operation = BAL_FLOAT_DIVIDE;
            return true;
        case 0x18U:
            *operation   = BAL_FLOAT_MINIMUM_NUMBER;
            *is_pairwise = true;
            return true;
        case 0x1AU:
            *operation = BAL_FLOAT_ABSOLUTE_DIFFERENCE;
            return true;
        case 0x1CU:
            *operation = BAL_FLOAT_COMPARE_GREATER;
            return true;
        case 0x1DU:
            *operation = BAL_FLOAT_ABSOLUTE_COMPARE_GREATER;
            return true;
        case 0x1EU:
            *operation   = BAL_FLOAT_MINIMUM;
            *is_pairwise = true;
            return true;
        default:
            return false;
    }
}

/// Applies a floating-point element operation, with FMLS negating the first multiplicand.
static uint64_t
float_operation(simd_t                     *simd,
                const uint32_t              esize,
                const bal_float_operation_t operation,
                const bool                  negate_first,
                uint64_t                    a,
                const uint64_t              b,
                const uint64_t              accumulator)
{
    if (true == negate_first)
    {
        a ^= 1ULL << (esize - 1U);
    }

    return bal_float_element(simd->cpu->fpcr, &simd->flags, esize, operation, a, b, accumulator);
}

/// The floating-point operations of three same, on `esize`-bit elements: the 11xxx opcodes of the
/// single and double precision group, and the half precision group, whose opcode is three bits.
/// a is bit 23 in both.
static bool
float_three_same(simd_t        *simd,
                 const uint32_t instruction,
                 const bool     is_scalar,
                 const uint32_t esize)
{
    const uint32_t        q             = (instruction >> 30U) & 1U;
    const uint32_t        u             = (instruction >> 29U) & 1U;
    const uint32_t        a             = (instruction >> 23U) & 1U;
    const uint32_t        rm            = (instruction >> 16U) & 0x1FU;
    const uint32_t        opcode        = (instruction >> 11U) & 0x7U;
    const uint32_t        rn            = (instruction >> 5U) & 0x1FU;
    const uint32_t        rd            = instruction & 0x1FU;
    const uint32_t        datasize      = 1U == q ? 128U : 64U;
    const vector_t        n             = read_vector(simd, rn);
    const vector_t        m             = read_vector(simd, rm);
    const vector_t        d             = read_vector(simd, rd);
    vector_t              result        = { { 0U, 0U } };
    bal_float_operation_t operation     = BAL_FLOAT_ADD;
    bool                  is_pairwise   = false;
    bool                  is_accumulate = false;

    if (false == float_three_same_operation(u, a, opcode, &operation, &is_pairwise, &is_accumulate))
    {
        return false;
    }

    if (true == is_scalar)
    {
        // FMULX, FCMEQ, FRECPS, FRSQRTS, FCMGE, FACGE, FABD, FCMGT and FACGT.
        const bool is_allowed
            = BAL_FLOAT_MULTIPLY_EXTENDED == operation || BAL_FLOAT_RECIPROCAL_STEP == operation
              || BAL_FLOAT_RECIPROCAL_SQUARE_ROOT_STEP == operation
              || BAL_FLOAT_ABSOLUTE_DIFFERENCE == operation || operation >= BAL_FLOAT_COMPARE_EQUAL;

        if (false == is_allowed || true == is_pairwise)
        {
            return false;
        }

        set_element(&result,
                    0U,
                    esize,
                    float_operation(simd,
                                    esize,
                                    operation,
                                    false,
                                    get_element(&n, 0U, esize),
                                    get_element(&m, 0U, esize),
                                    0U));
        write_vector(simd, rd, result, esize);
        return true;
    }

    if (64U == esize && 0U == q)
    {
        return false;
    }

    const uint32_t elements     = datasize / esize;
    const bool     negate_first = true == is_accumulate && 1U == a;

    for (uint32_t e = 0U; e < elements; ++e)
    {
        uint64_t x = 0U;
        uint64_t y = 0U;

        if (true == is_pairwise)
        {
            // Pairs of adjacent elements of m:n.
            const uint32_t first = 2U * e;
            x                    = first < elements ? get_element(&n, first, esize)
                                                    : get_element(&m, first - elements, esize);
            y                    = first + 1U < elements ? get_element(&n, first + 1U, esize)
                                                         : get_element(&m, first + 1U - elements, esize);
        }
        else
        {
            x = get_element(&n, e, esize);
            y = get_element(&m, e, esize);
        }

        set_element(
            &result,
            e,
            esize,
            float_operation(simd, esize, operation, negate_first, x, y, get_element(&d, e, esize)));
    }

    write_vector(simd, rd, result, datasize);
    return true;
}

/// Advanced SIMD three same, vector and scalar.
static bool
three_same(simd_t *simd, const uint32_t instruction, const bool is_scalar)
{
    const uint32_t q        = (instruction >> 30U) & 1U;
    const uint32_t u        = (instruction >> 29U) & 1U;
    const uint32_t size     = (instruction >> 22U) & 0x3U;
    const uint32_t rm       = (instruction >> 16U) & 0x1FU;
    const uint32_t opcode   = (instruction >> 11U) & 0x1FU;
    const uint32_t rn       = (instruction >> 5U) & 0x1FU;
    const uint32_t rd       = instruction & 0x1FU;
    const uint32_t datasize = true == is_scalar ? 0U : (1U == q ? 128U : 64U);
    const vector_t n        = read_vector(simd, rn);
    const vector_t m        = read_vector(simd, rm);
    const vector_t d        = read_vector(simd, rd);
    vector_t       result   = { { 0U, 0U } };

    // Logical operations: AND, BIC, ORR, ORN, EOR, BSL, BIT, BIF.
    if (0x03U == opcode)
    {
        if (true == is_scalar)
        {
            return false;
        }

        for (uint32_t i = 0U; i < 2U; ++i)
        {
            const uint64_t x = n.d[i];
            const uint64_t y = m.d[i];
            const uint64_t z = d.d[i];
            uint64_t       r = 0U;

            switch ((u << 2U) | size)
            {
                case 0U:
                    r = x & y;
                    break;
                case 1U:
                    r = x & ~y;
                    break;
                case 2U:
                    r = x | y;
                    break;
                case 3U:
                    r = x | ~y;
                    break;
                case 4U:
                    r = x ^ y;
                    break;
                case 5U: // BSL: d selects between n and m.
                    r = (z & x) | (~z & y);
                    break;
                case 6U: // BIT: insert n where m is set.
                    r = (x & y) | (z & ~y);
                    break;
                default: // BIF: insert n where m is clear.
                    r = (z & y) | (x & ~y);
                    break;
            }

            result.d[i] = r;
        }

        write_vector(simd, rd, result, datasize);
        return true;
    }

    // Floating point: opcode 11xxx, with size<0> selecting double precision.
    if (0x18U == (opcode & 0x18U))
    {
        return float_three_same(simd, instruction, is_scalar, 1U == (size & 1U) ? 64U : 32U);
    }

    const uint32_t esize = 8U << size;

    if (true == is_scalar)
    {
        if (false == is_scalar_three_same(u, opcode, size))
        {
            return false;
        }

        uint64_t value = 0U;

        if (false
            == integer_three_same_element(simd,
                                          u,
                                          opcode,
                                          get_element(&n, 0U, esize),
                                          get_element(&m, 0U, esize),
                                          0U,
                                          esize,
                                          &value))
        {
            return false;
        }

        set_element(&result, 0U, esize, value);
        write_vector(simd, rd, result, esize);
        return true;
    }

    if (3U == size && 0U == q)
    {
        return false;
    }

    const uint32_t elements = datasize / esize;

    // SQDMULH and SQRDMULH only exist for halfwords and words.
    if (0x16U == opcode && (0U == size || 3U == size))
    {
        return false;
    }

    // Pairwise integer operations: SMAXP, SMINP, UMAXP, UMINP and ADDP.
    if (opcode >= 0x14U && opcode <= 0x17U && opcode != 0x16U)
    {
        if (0x17U == opcode && 1U == u)
        {
            return false; // ADDP has no U = 1 form.
        }

        if (3U == size && opcode != 0x17U)
        {
            return false;
        }

        for (uint32_t e = 0U; e < elements; ++e)
        {
            const uint32_t first = 2U * e;
            const uint64_t a     = first < elements ? get_element(&n, first, esize)
                                                    : get_element(&m, first - elements, esize);
            const uint64_t b     = first + 1U < elements
                                       ? get_element(&n, first + 1U, esize)
                                       : get_element(&m, first + 1U - elements, esize);
            uint64_t       value = 0U;

            if (0x17U == opcode)
            {
                value = (a + b) & mask_of(esize);
            }
            else
            {
                const bool is_maximum = 0x14U == opcode;
                const bool a_wins
                    = 1U == u ? (is_maximum ? a > b : a < b)
                              : (is_maximum ? sign_extend(a, esize) > sign_extend(b, esize)
                                            : sign_extend(a, esize) < sign_extend(b, esize));
                value = true == a_wins ? a : b;
            }

            set_element(&result, e, esize, value);
        }

        write_vector(simd, rd, result, datasize);
        return true;
    }

    // MUL, MLA, MLS, the halving additions, the maxima and minima and the absolute differences
    // have no 64-bit form; PMUL only works on bytes.
    const bool is_without_64bit = 0x12U == opcode || 0x13U == opcode || 0x00U == opcode
                                  || 0x02U == opcode || 0x04U == opcode
                                  || (opcode >= 0x0CU && opcode <= 0x0FU);

    if ((true == is_without_64bit && 3U == size) || (0x13U == opcode && 1U == u && size != 0U))
    {
        return false;
    }

    for (uint32_t e = 0U; e < elements; ++e)
    {
        uint64_t value = 0U;

        if (false
            == integer_three_same_element(simd,
                                          u,
                                          opcode,
                                          get_element(&n, e, esize),
                                          get_element(&m, e, esize),
                                          get_element(&d, e, esize),
                                          esize,
                                          &value))
        {
            return false;
        }

        set_element(&result, e, esize, value);
    }

    write_vector(simd, rd, result, datasize);
    return true;
}

/// Advanced SIMD three same extra, vector and scalar: SQRDMLAH, SQRDMLSH and the vector SDOT and
/// UDOT.
static bool
three_same_extra(simd_t *simd, const uint32_t instruction, const bool is_scalar)
{
    const uint32_t q        = (instruction >> 30U) & 1U;
    const uint32_t u        = (instruction >> 29U) & 1U;
    const uint32_t size     = (instruction >> 22U) & 0x3U;
    const uint32_t rm       = (instruction >> 16U) & 0x1FU;
    const uint32_t opcode   = (instruction >> 11U) & 0xFU;
    const uint32_t rn       = (instruction >> 5U) & 0x1FU;
    const uint32_t rd       = instruction & 0x1FU;
    const uint32_t datasize = 1U == q ? 128U : 64U;
    const vector_t n        = read_vector(simd, rn);
    const vector_t m        = read_vector(simd, rm);
    const vector_t d        = read_vector(simd, rd);
    vector_t       result   = { { 0U, 0U } };

    // SQRDMLAH (0000) and SQRDMLSH (0001) on halfwords and words.
    if (1U == u && opcode <= 0x1U && (1U == size || 2U == size))
    {
        const uint32_t esize    = 8U << size;
        const uint32_t elements = true == is_scalar ? 1U : datasize / esize;

        for (uint32_t e = 0U; e < elements; ++e)
        {
            set_element(&result,
                        e,
                        esize,
                        doubling_multiply_accumulate(simd,
                                                     get_element(&d, e, esize),
                                                     get_element(&n, e, esize),
                                                     get_element(&m, e, esize),
                                                     esize,
                                                     0x1U == opcode));
        }

        write_vector(simd, rd, result, true == is_scalar ? esize : datasize);
        return true;
    }

    // SDOT and UDOT (0010) on words.
    if (false == is_scalar && 0x2U == opcode && 2U == size)
    {
        for (uint32_t e = 0U; e < datasize / 32U; ++e)
        {
            set_element(&result,
                        e,
                        32U,
                        dot_product(get_element(&d, e, 32U),
                                    get_element(&n, e, 32U),
                                    get_element(&m, e, 32U),
                                    1U == u));
        }

        write_vector(simd, rd, result, datasize);
        return true;
    }

    return false;
}

/// Advanced SIMD scalar three different: SQDMLAL, SQDMLSL and SQDMULL.
static bool
scalar_three_different(simd_t *simd, const uint32_t instruction)
{
    const uint32_t u      = (instruction >> 29U) & 1U;
    const uint32_t size   = (instruction >> 22U) & 0x3U;
    const uint32_t rm     = (instruction >> 16U) & 0x1FU;
    const uint32_t opcode = (instruction >> 12U) & 0xFU;
    const uint32_t rn     = (instruction >> 5U) & 0x1FU;
    const uint32_t rd     = instruction & 0x1FU;
    const vector_t n      = read_vector(simd, rn);
    const vector_t m      = read_vector(simd, rm);
    const vector_t d      = read_vector(simd, rd);
    vector_t       result = { { 0U, 0U } };

    if (u != 0U || (size != 1U && size != 2U)
        || (opcode != 0x9U && opcode != 0xBU && opcode != 0xDU))
    {
        return false;
    }

    const uint32_t esize = 8U << size;
    set_element(&result,
                0U,
                2U * esize,
                doubling_long_operation(simd,
                                        opcode,
                                        get_element(&d, 0U, 2U * esize),
                                        get_element(&n, 0U, esize),
                                        get_element(&m, 0U, esize),
                                        esize));
    write_vector(simd, rd, result, 2U * esize);
    return true;
}

/// The FRINT* and FCVT* rounding mode of a two-register miscellaneous opcode.
static uint32_t
misc_rounding_mode(const uint32_t u, const uint32_t a, const uint32_t opcode, const bal_cpu_t *cpu)
{
    // opcode 11000/11001: FRINTN, FRINTM (a = 0), FRINTP, FRINTZ (a = 1); U = 1 gives FRINTA,
    // FRINTX (a = 0) and FRINTI (a = 1, opcode 11001). The FCVT forms use the same pattern.
    const uint32_t low = opcode & 0x1U;

    if (0x1CU == opcode) // FCVTAS, FCVTAU
    {
        return 4U;
    }

    if (0U == u)
    {
        return 0U == a ? (0U == low ? 0U : 2U) : (0U == low ? 1U : 3U);
    }

    if (0x18U == (opcode & 0x1EU))
    {
        if (0U == a)
        {
            return 0U == low ? 4U : (uint32_t)((cpu->fpcr >> 22U) & 0x3U); // FRINTA, FRINTX
        }

        return (uint32_t)((cpu->fpcr >> 22U) & 0x3U); // FRINTI
    }

    return 0U == a ? (0U == low ? 0U : 2U) : (0U == low ? 1U : 3U);
}

/// The floating-point operations of two-register miscellaneous on `esize`-bit elements, for the
/// single and double precision group and the half precision one; a is bit 23 in both.
static bool
float_two_register_misc(simd_t        *simd,
                        const uint32_t instruction,
                        const bool     is_scalar,
                        const uint32_t esize)
{
    const uint32_t q        = (instruction >> 30U) & 1U;
    const uint32_t u        = (instruction >> 29U) & 1U;
    const uint32_t opcode   = (instruction >> 12U) & 0x1FU;
    const uint32_t rn       = (instruction >> 5U) & 0x1FU;
    const uint32_t rd       = instruction & 0x1FU;
    const uint32_t datasize = true == is_scalar ? 64U : (1U == q ? 128U : 64U);
    const vector_t n        = read_vector(simd, rn);
    vector_t       result   = { { 0U, 0U } };
    const uint64_t fpcr     = simd->cpu->fpcr;

    const uint32_t a        = (instruction >> 23U) & 1U;
    const uint32_t elements = true == is_scalar ? 1U : datasize / esize;

    if (false == is_scalar && 64U == esize && 0U == q)
    {
        return false;
    }

    for (uint32_t e = 0U; e < elements; ++e)
    {
        const uint64_t x     = get_element(&n, e, esize);
        uint64_t       value = 0U;

        switch ((u << 6U) | (a << 5U) | opcode)
        {
            case 0x2CU: // FCMGT #0
                value = bal_float_element(
                    fpcr, &simd->flags, esize, BAL_FLOAT_COMPARE_GREATER, x, 0U, 0U);
                break;
            case 0x2DU: // FCMEQ #0
                value = bal_float_element(
                    fpcr, &simd->flags, esize, BAL_FLOAT_COMPARE_EQUAL, x, 0U, 0U);
                break;
            case 0x2EU: // FCMLT #0: 0 > x
                value = bal_float_element(
                    fpcr, &simd->flags, esize, BAL_FLOAT_COMPARE_GREATER, 0U, x, 0U);
                break;
            case 0x6CU: // FCMGE #0
                value = bal_float_element(
                    fpcr, &simd->flags, esize, BAL_FLOAT_COMPARE_GREATER_EQUAL, x, 0U, 0U);
                break;
            case 0x6DU: // FCMLE #0: 0 >= x
                value = bal_float_element(
                    fpcr, &simd->flags, esize, BAL_FLOAT_COMPARE_GREATER_EQUAL, 0U, x, 0U);
                break;
            case 0x2FU: // FABS
                if (true == is_scalar)
                {
                    return false;
                }
                value = x & (mask_of(esize) >> 1U);
                break;
            case 0x6FU: // FNEG
                if (true == is_scalar)
                {
                    return false;
                }
                value = x ^ (1ULL << (esize - 1U));
                break;
            case 0x7FU: // FSQRT
                if (true == is_scalar)
                {
                    return false;
                }
                value = bal_float_element(
                    fpcr, &simd->flags, esize, BAL_FLOAT_SQUARE_ROOT, x, 0U, 0U);
                break;
            case 0x3DU: // FRECPE
                value = bal_float_element(
                    fpcr, &simd->flags, esize, BAL_FLOAT_RECIPROCAL_ESTIMATE, x, 0U, 0U);
                break;
            case 0x7DU: // FRSQRTE
                value = bal_float_element(fpcr,
                                          &simd->flags,
                                          esize,
                                          BAL_FLOAT_RECIPROCAL_SQUARE_ROOT_ESTIMATE,
                                          x,
                                          0U,
                                          0U);
                break;
            case 0x3CU: // URECPE
            case 0x7CU: // URSQRTE
                if (true == is_scalar || esize != 32U)
                {
                    return false;
                }
                value = bal_unsigned_estimate_element((uint32_t)x,
                                                      0x7CU == ((u << 6U) | (a << 5U) | opcode));
                break;
            case 0x3FU: // FRECPX
                if (false == is_scalar)
                {
                    return false;
                }
                value = bal_float_element(
                    fpcr, &simd->flags, esize, BAL_FLOAT_RECIPROCAL_EXPONENT, x, 0U, 0U);
                break;
            case 0x18U: // FRINTN
            case 0x19U: // FRINTM
            case 0x38U: // FRINTP
            case 0x39U: // FRINTZ
            case 0x58U: // FRINTA
            case 0x59U: // FRINTX
            case 0x79U: // FRINTI
                if (true == is_scalar)
                {
                    return false;
                }
                value = bal_float_round_element(fpcr,
                                                &simd->flags,
                                                esize,
                                                x,
                                                misc_rounding_mode(u, a, opcode, simd->cpu),
                                                0x59U == ((u << 6U) | (a << 5U) | opcode));
                break;
            case 0x1AU: // FCVTNS
            case 0x1BU: // FCVTMS
            case 0x1CU: // FCVTAS
            case 0x3AU: // FCVTPS
            case 0x3BU: // FCVTZS
            case 0x5AU: // FCVTNU
            case 0x5BU: // FCVTMU
            case 0x5CU: // FCVTAU
            case 0x7AU: // FCVTPU
            case 0x7BU: // FCVTZU
                value = bal_float_to_integer_element(fpcr,
                                                     &simd->flags,
                                                     esize,
                                                     x,
                                                     0U,
                                                     misc_rounding_mode(u, a, opcode, simd->cpu),
                                                     1U == u,
                                                     esize);
                break;
            case 0x1DU: // SCVTF
            case 0x5DU: // UCVTF
                value = bal_float_from_integer_element(
                    fpcr, &simd->flags, esize, x, 0U, 1U == u, esize);
                break;
            default:
                return false;
        }

        set_element(&result, e, esize, value);
    }

    write_vector(simd, rd, result, true == is_scalar ? esize : datasize);
    return true;
}

/// Advanced SIMD two-register miscellaneous, vector and scalar.
static bool
two_register_misc(simd_t *simd, const uint32_t instruction, const bool is_scalar)
{
    const uint32_t q        = (instruction >> 30U) & 1U;
    const uint32_t u        = (instruction >> 29U) & 1U;
    const uint32_t size     = (instruction >> 22U) & 0x3U;
    const uint32_t opcode   = (instruction >> 12U) & 0x1FU;
    const uint32_t rn       = (instruction >> 5U) & 0x1FU;
    const uint32_t rd       = instruction & 0x1FU;
    const uint32_t datasize = true == is_scalar ? 64U : (1U == q ? 128U : 64U);
    const vector_t n        = read_vector(simd, rn);
    const vector_t d        = read_vector(simd, rd);
    vector_t       result   = { { 0U, 0U } };
    const uint64_t fpcr     = simd->cpu->fpcr;

    // Floating point: opcodes 01100-01111 with size<1> = 1 and 1x0xx-11111.
    const bool is_float = opcode >= 0x16U || (opcode >= 0x0CU && opcode <= 0x0FU && size >= 2U);

    if (true == is_float && opcode != 0x16U && opcode != 0x17U)
    {
        return float_two_register_misc(simd, instruction, is_scalar, 1U == (size & 1U) ? 64U : 32U);
    }

    // FCVTN, FCVTN2 to the narrower precision and FCVTL, FCVTL2 to the wider one: sz = 1 converts
    // between single and double precision, sz = 0 between half and single.
    if (0x16U == opcode || 0x17U == opcode)
    {
        // FCVTXN and FCVTXN2: double to single precision, rounding to odd.
        if (1U == u && 0x16U == opcode && 1U == size)
        {
            const uint32_t count = true == is_scalar ? 1U : 2U;
            const uint32_t part  = true == is_scalar ? 0U : q * count;
            result               = false == is_scalar && 1U == q ? d : result;

            for (uint32_t e = 0U; e < count; ++e)
            {
                set_element(&result,
                            part + e,
                            32U,
                            bal_float_narrow_odd_element(fpcr, &simd->flags, n.d[e]));
            }

            write_vector(simd, rd, result, true == is_scalar ? 32U : (1U == q ? 128U : 64U));
            return true;
        }

        // With size<1> set these encodings are BFCVTN (FEAT_BF16, which the guest CPU lacks) or
        // unallocated.
        if (true == is_scalar || 1U == u || 1U == (size >> 1U))
        {
            return false;
        }

        const uint32_t wide   = 1U == (size & 1U) ? 64U : 32U;
        const uint32_t narrow = wide / 2U;
        const uint32_t count  = 128U / wide;

        if (0x16U == opcode)
        {
            result = 1U == q ? d : result;

            for (uint32_t e = 0U; e < count; ++e)
            {
                set_element(&result,
                            q * count + e,
                            narrow,
                            bal_float_convert_element(
                                fpcr, &simd->flags, wide, narrow, get_element(&n, e, wide)));
            }

            write_vector(simd, rd, result, 1U == q ? 128U : 64U);
            return true;
        }

        for (uint32_t e = 0U; e < count; ++e)
        {
            set_element(
                &result,
                e,
                wide,
                bal_float_convert_element(
                    fpcr, &simd->flags, narrow, wide, get_element(&n, q * count + e, narrow)));
        }

        write_vector(simd, rd, result, 128U);
        return true;
    }

    const uint32_t esize    = 8U << size;
    const uint32_t elements = true == is_scalar ? 1U : datasize / esize;

    // Narrowing: XTN, SQXTN, UQXTN, SQXTUN write the lower or (Q = 1) upper half.
    if (0x12U == opcode || 0x14U == opcode)
    {
        if (3U == size || (0x12U == opcode && 0U == u && true == is_scalar))
        {
            return false;
        }

        const uint32_t narrow = esize;
        const uint32_t wide   = esize * 2U;
        const uint32_t count  = true == is_scalar ? 1U : 64U / narrow;
        const uint32_t base   = true == is_scalar ? 0U : q * count;
        result                = 1U == q && false == is_scalar ? d : result;

        for (uint32_t e = 0U; e < count; ++e)
        {
            const uint64_t x     = get_element(&n, e, wide);
            uint64_t       value = 0U;

            if (0x12U == opcode)
            {
                value = 0U == u ? x & mask_of(narrow) // XTN
                                : saturate_signed_to_unsigned(
                                      simd, sign_extend(x, wide), narrow); // SQXTUN
            }
            else
            {
                value = 0U == u ? saturate_signed(simd, sign_extend(x, wide), narrow) // SQXTN
                                : saturate_unsigned(simd, x, narrow);                 // UQXTN
            }

            set_element(&result, base + e, narrow, value);
        }

        write_vector(simd, rd, result, true == is_scalar ? narrow : (1U == q ? 128U : 64U));
        return true;
    }

    // SHLL, SHLL2: widen and shift left by the element size.
    if (0x13U == opcode && 1U == u)
    {
        if (true == is_scalar || 3U == size)
        {
            return false;
        }

        for (uint32_t e = 0U; e < 64U / esize; ++e)
        {
            set_element(
                &result, e, esize * 2U, get_element(&n, q * (64U / esize) + e, esize) << esize);
        }

        write_vector(simd, rd, result, 128U);
        return true;
    }

    // Pairwise add long: SADDLP, UADDLP, SADALP, UADALP.
    if (0x02U == opcode || 0x06U == opcode)
    {
        if (true == is_scalar || 3U == size)
        {
            return false;
        }

        const uint32_t wide = esize * 2U;

        for (uint32_t e = 0U; e < datasize / wide; ++e)
        {
            const uint64_t a = get_element(&n, 2U * e, esize);
            const uint64_t b = get_element(&n, 2U * e + 1U, esize);
            uint64_t       sum
                = 1U == u ? a + b : (uint64_t)(sign_extend(a, esize) + sign_extend(b, esize));

            if (0x06U == opcode)
            {
                sum += get_element(&d, e, wide);
            }

            set_element(&result, e, wide, sum & mask_of(wide));
        }

        write_vector(simd, rd, result, datasize);
        return true;
    }

    // NOT (size 00) and RBIT (size 01) work on bytes whatever the size field says.
    if (0x25U == ((u << 5U) | opcode))
    {
        if (true == is_scalar || size >= 2U)
        {
            return false;
        }

        for (uint32_t e = 0U; e < datasize / 8U; ++e)
        {
            const uint64_t x = get_element(&n, e, 8U);
            set_element(&result, e, 8U, 0U == size ? ~x & 0xFFU : reverse_bits(x, 8U));
        }

        write_vector(simd, rd, result, datasize);
        return true;
    }

    // The element-wise integer operations.
    const bool is_64bit_only = true == is_scalar;

    for (uint32_t e = 0U; e < elements; ++e)
    {
        const uint64_t x     = get_element(&n, e, esize);
        uint64_t       value = 0U;

        switch ((u << 5U) | opcode)
        {
            case 0x00U: // REV64
            case 0x20U: // REV32
            case 0x01U: // REV16
            {
                const uint32_t key       = (u << 5U) | opcode;
                const uint32_t container = 0x00U == key ? 64U : (0x01U == key ? 16U : 32U);

                if (true == is_scalar || esize >= container)
                {
                    return false;
                }

                const uint32_t per_container = container / esize;
                const uint32_t base          = (e / per_container) * per_container;
                value = get_element(&n, base + (per_container - 1U - (e - base)), esize);
                break;
            }
            case 0x04U: // CLS
                if (true == is_scalar || 3U == size)
                {
                    return false;
                }
                value = count_leading_zeros(x ^ (0U - (x >> (esize - 1U)) & mask_of(esize)), esize)
                        - 1U;
                break;
            case 0x24U: // CLZ
                if (true == is_scalar || 3U == size)
                {
                    return false;
                }
                value = count_leading_zeros(x, esize);
                break;
            case 0x05U: // CNT
                if (true == is_scalar || size != 0U)
                {
                    return false;
                }
                value = (uint64_t)__builtin_popcountll(x);
                break;
            case 0x07U: // SQABS
            case 0x27U: // SQNEG
            {
                const int64_t signed_x   = sign_extend(x, esize);
                const bool    is_minimum = signed_x == -(int64_t)(mask_of(esize) >> 1U) - 1;

                if (true == is_minimum)
                {
                    // Both saturate the most negative value to the most positive one.
                    simd->flags |= FPSR_QC;
                    value = mask_of(esize) >> 1U;
                    break;
                }

                value = 0U == u && signed_x >= 0 ? x : (0U - x) & mask_of(esize);
                break;
            }
            case 0x08U: // CMGT #0
            case 0x09U: // CMEQ #0
            case 0x0AU: // CMLT #0
            case 0x28U: // CMGE #0
            case 0x29U: // CMLE #0
            {
                if (true == is_64bit_only && size != 3U)
                {
                    return false;
                }

                static const uint32_t KINDS[] = { 0U, 4U, 7U };
                const uint32_t kind = 0U == u ? KINDS[opcode - 0x08U] : (0x08U == opcode ? 1U : 6U);
                value               = compare_integers(kind, x, 0U, esize);
                break;
            }
            case 0x03U: // SUQADD
            case 0x23U: // USQADD
                value = saturating_accumulate(simd, get_element(&d, e, esize), x, esize, 1U == u);
                break;
            case 0x0BU: // ABS
            case 0x2BU: // NEG
                if (true == is_64bit_only && size != 3U)
                {
                    return false;
                }
                value = 0x2BU == ((u << 5U) | opcode) || sign_extend(x, esize) < 0
                            ? (0U - x) & mask_of(esize)
                            : x;
                break;
            default:
                return false;
        }

        if (false == is_scalar && 3U == size && 0U == q)
        {
            return false;
        }

        set_element(&result, e, esize, value);
    }

    write_vector(simd, rd, result, true == is_scalar ? esize : datasize);
    return true;
}

/// Advanced SIMD across lanes.
static bool
across_lanes(simd_t *simd, const uint32_t instruction)
{
    const uint32_t q        = (instruction >> 30U) & 1U;
    const uint32_t u        = (instruction >> 29U) & 1U;
    const uint32_t size     = (instruction >> 22U) & 0x3U;
    const uint32_t opcode   = (instruction >> 12U) & 0x1FU;
    const uint32_t rn       = (instruction >> 5U) & 0x1FU;
    const uint32_t rd       = instruction & 0x1FU;
    const uint32_t datasize = 1U == q ? 128U : 64U;
    const vector_t n        = read_vector(simd, rn);
    vector_t       result   = { { 0U, 0U } };

    // FMAXNMV, FMINNMV, FMAXV, FMINV: single precision on 4S with U = 1 and half precision on 4H
    // and 8H with U = 0, reduced as a tree of adjacent pairs, as Reduce() does.
    if ((0x0CU == opcode || 0x0FU == opcode) && 0U == (size & 1U) && (0U == u || 1U == q))
    {
        const uint32_t              esize      = 1U == u ? 32U : 16U;
        const bool                  is_minimum = 1U == (size >> 1U);
        const bal_float_operation_t operation
            = 0x0CU == opcode ? (is_minimum ? BAL_FLOAT_MINIMUM_NUMBER : BAL_FLOAT_MAXIMUM_NUMBER)
                              : (is_minimum ? BAL_FLOAT_MINIMUM : BAL_FLOAT_MAXIMUM);
        uint64_t values[8];
        uint32_t count = datasize / esize;

        for (uint32_t e = 0U; e < count; ++e)
        {
            values[e] = get_element(&n, e, esize);
        }

        while (count > 1U)
        {
            for (uint32_t e = 0U; e < count / 2U; ++e)
            {
                values[e] = float_operation(
                    simd, esize, operation, false, values[2U * e], values[2U * e + 1U], 0U);
            }

            count /= 2U;
        }

        set_element(&result, 0U, esize, values[0]);
        write_vector(simd, rd, result, esize);
        return true;
    }

    const uint32_t esize    = 8U << size;
    const uint32_t elements = datasize / esize;

    if (3U == size || (2U == size && 0U == q))
    {
        return false;
    }

    if (0x03U == opcode) // SADDLV, UADDLV
    {
        uint64_t sum = 0U;

        for (uint32_t e = 0U; e < elements; ++e)
        {
            const uint64_t x = get_element(&n, e, esize);
            sum += 1U == u ? x : (uint64_t)sign_extend(x, esize);
        }

        set_element(&result, 0U, esize * 2U, sum & mask_of(esize * 2U));
        write_vector(simd, rd, result, esize * 2U);
        return true;
    }

    if (opcode != 0x0AU && opcode != 0x1AU && !(0x1BU == opcode && 0U == u))
    {
        return false;
    }

    uint64_t accumulator = get_element(&n, 0U, esize);

    for (uint32_t e = 1U; e < elements; ++e)
    {
        const uint64_t x = get_element(&n, e, esize);

        if (0x1BU == opcode) // ADDV
        {
            accumulator = (accumulator + x) & mask_of(esize);
            continue;
        }

        const bool is_maximum = 0x0AU == opcode;
        const bool x_wins
            = 1U == u ? (is_maximum ? x > accumulator : x < accumulator)
                      : (is_maximum ? sign_extend(x, esize) > sign_extend(accumulator, esize)
                                    : sign_extend(x, esize) < sign_extend(accumulator, esize));
        accumulator = true == x_wins ? x : accumulator;
    }

    set_element(&result, 0U, esize, accumulator);
    write_vector(simd, rd, result, esize);
    return true;
}

/// Advanced SIMD copy: DUP, INS, SMOV, UMOV; and the scalar DUP (element).
static bool
copy(simd_t *simd, const uint32_t instruction, const bool is_scalar)
{
    const uint32_t q    = (instruction >> 30U) & 1U;
    const uint32_t op   = (instruction >> 29U) & 1U;
    const uint32_t imm5 = (instruction >> 16U) & 0x1FU;
    const uint32_t imm4 = (instruction >> 11U) & 0xFU;
    const uint32_t rn   = (instruction >> 5U) & 0x1FU;
    const uint32_t rd   = instruction & 0x1FU;

    if (0U == (imm5 & 0xFU))
    {
        return false;
    }

    const uint32_t size     = (uint32_t)__builtin_ctz(imm5);
    const uint32_t esize    = 8U << size;
    const uint32_t index    = imm5 >> (size + 1U);
    const uint32_t datasize = 1U == q ? 128U : 64U;
    const vector_t n        = read_vector(simd, rn);
    vector_t       result   = { { 0U, 0U } };

    if (true == is_scalar)
    {
        set_element(&result, 0U, esize, get_element(&n, index, esize));
        write_vector(simd, rd, result, esize);
        return true;
    }

    if (1U == op)
    {
        // INS (element): Vd.T[index] = Vn.T[imm4 >> size].
        if (0U == q)
        {
            return false;
        }

        result = read_vector(simd, rd);
        set_element(&result, index, esize, get_element(&n, imm4 >> size, esize));
        write_vector(simd, rd, result, 128U);
        return true;
    }

    switch (imm4)
    {
        case 0x0U: // DUP (element)
        case 0x1U: // DUP (general)
        {
            if (3U == size && 0U == q)
            {
                return false;
            }

            const uint64_t value = 0x0U == imm4 ? get_element(&n, index, esize)
                                                : read_general(simd, rn) & mask_of(esize);

            for (uint32_t e = 0U; e < datasize / esize; ++e)
            {
                set_element(&result, e, esize, value);
            }

            write_vector(simd, rd, result, datasize);
            return true;
        }
        case 0x3U: // INS (general)
            if (0U == q)
            {
                return false;
            }

            result = read_vector(simd, rd);
            set_element(&result, index, esize, read_general(simd, rn));
            write_vector(simd, rd, result, 128U);
            return true;
        case 0x5U: // SMOV
        {
            // Wd from a byte or halfword (Q = 0), Xd from up to a word (Q = 1).
            if (size >= (1U == q ? 3U : 2U))
            {
                return false;
            }

            const int64_t value = sign_extend(get_element(&n, index, esize), esize);
            write_general(simd, rd, 1U == q ? (uint64_t)value : (uint64_t)value & 0xFFFFFFFFU);
            return true;
        }
        case 0x7U: // UMOV
            if ((1U == q) != (3U == size))
            {
                return false;
            }

            write_general(simd, rd, get_element(&n, index, esize));
            return true;
        default:
            return false;
    }
}

/// Advanced SIMD modified immediate: MOVI, MVNI, ORR, BIC and FMOV (vector, immediate).
static bool
modified_immediate(simd_t *simd, const uint32_t instruction)
{
    const uint32_t q        = (instruction >> 30U) & 1U;
    const uint32_t op       = (instruction >> 29U) & 1U;
    const uint32_t cmode    = (instruction >> 12U) & 0xFU;
    const uint32_t o2       = (instruction >> 11U) & 1U;
    const uint32_t rd       = instruction & 0x1FU;
    const uint64_t imm8     = (((instruction >> 16U) & 0x7U) << 5U) | ((instruction >> 5U) & 0x1FU);
    const uint32_t datasize = 1U == q ? 128U : 64U;
    uint64_t       imm64    = 0U;
    bool           is_orr   = false;
    bool           is_bic   = false;

    if (1U == o2 && !(0xFU == cmode && 0U == op))
    {
        return false;
    }

    switch (cmode >> 1U)
    {
        case 0U:
        case 1U:
        case 2U:
        case 3U: // 32-bit shifted immediate
        {
            const uint64_t word = imm8 << (8U * (cmode >> 1U));
            imm64               = word | (word << 32U);
            is_orr              = 1U == (cmode & 1U) && 0U == op;
            is_bic              = 1U == (cmode & 1U) && 1U == op;
            break;
        }
        case 4U:
        case 5U: // 16-bit shifted immediate
        {
            const uint64_t half = imm8 << (8U * ((cmode >> 1U) & 1U));
            imm64               = half | (half << 16U) | (half << 32U) | (half << 48U);
            is_orr              = 1U == (cmode & 1U) && 0U == op;
            is_bic              = 1U == (cmode & 1U) && 1U == op;
            break;
        }
        case 6U: // 32-bit shifting ones (MSL)
        {
            const uint64_t word
                = 0U == (cmode & 1U) ? (imm8 << 8U) | 0xFFU : (imm8 << 16U) | 0xFFFFU;
            imm64 = word | (word << 32U);
            break;
        }
        default:
            if (0xEU == cmode)
            {
                if (0U == op)
                {
                    imm64 = imm8 * 0x0101010101010101ULL;
                }
                else
                {
                    for (uint32_t i = 0U; i < 8U; ++i)
                    {
                        imm64 |= (0U != ((imm8 >> i) & 1U) ? 0xFFULL : 0U) << (8U * i);
                    }
                }
            }
            else if (0U == op)
            {
                if (1U == o2)
                {
                    // FMOV half: imm8 expanded as VFPExpandImm(imm8, 16), replicated.
                    const uint64_t b6       = (imm8 >> 6U) & 1U;
                    const uint64_t halfword = ((imm8 >> 7U) << 15U) | ((b6 ^ 1U) << 14U)
                                              | ((0U - b6) & 0x3U) << 12U | ((imm8 & 0x3FU) << 6U);
                    imm64 = halfword * 0x0001000100010001ULL;
                    break;
                }

                // FMOV single: imm8 expanded as VFPExpandImm(imm8, 32), replicated.
                const uint64_t b6   = (imm8 >> 6U) & 1U;
                const uint64_t word = ((imm8 >> 7U) << 31U) | ((b6 ^ 1U) << 30U)
                                      | ((0U - b6) & 0x1FU) << 25U | ((imm8 & 0x3FU) << 19U);
                imm64 = word | (word << 32U);
            }
            else
            {
                if (0U == q)
                {
                    return false;
                }

                const uint64_t b6 = (imm8 >> 6U) & 1U;
                imm64 = ((imm8 >> 7U) << 63U) | ((b6 ^ 1U) << 62U) | ((0U - b6) & 0xFFU) << 54U
                        | ((imm8 & 0x3FU) << 48U);
            }

            break;
    }

    vector_t result = read_vector(simd, rd);

    for (uint32_t i = 0U; i < 2U; ++i)
    {
        if (true == is_orr)
        {
            result.d[i] |= imm64;
        }
        else if (true == is_bic)
        {
            result.d[i] &= ~imm64;
        }
        else
        {
            // MVNI inverts the shifted and MSL forms.
            const bool is_inverted = 1U == op && cmode < 0xEU;
            result.d[i]            = true == is_inverted ? ~imm64 : imm64;
        }
    }

    write_vector(simd, rd, result, datasize);
    return true;
}

/// Advanced SIMD permute: UZP1, TRN1, ZIP1, UZP2, TRN2, ZIP2.
static bool
permute(simd_t *simd, const uint32_t instruction)
{
    const uint32_t q        = (instruction >> 30U) & 1U;
    const uint32_t size     = (instruction >> 22U) & 0x3U;
    const uint32_t rm       = (instruction >> 16U) & 0x1FU;
    const uint32_t opcode   = (instruction >> 12U) & 0x7U;
    const uint32_t rn       = (instruction >> 5U) & 0x1FU;
    const uint32_t rd       = instruction & 0x1FU;
    const uint32_t datasize = 1U == q ? 128U : 64U;
    const uint32_t esize    = 8U << size;
    const uint32_t elements = datasize / esize;
    const uint32_t part     = opcode >> 2U;
    const vector_t n        = read_vector(simd, rn);
    const vector_t m        = read_vector(simd, rm);
    vector_t       result   = { { 0U, 0U } };

    if ((3U == size && 0U == q) || 0U == (opcode & 0x3U))
    {
        return false;
    }

    for (uint32_t e = 0U; e < elements; ++e)
    {
        uint64_t value = 0U;

        switch (opcode & 0x3U)
        {
            case 1U: // UZP: the even (UZP1) or odd (UZP2) elements of n:m.
            {
                const uint32_t source = 2U * e + part;
                value                 = source < elements ? get_element(&n, source, esize)
                                                          : get_element(&m, source - elements, esize);
                break;
            }
            case 2U: // TRN
                value = 0U == (e & 1U) ? get_element(&n, (e & ~1U) + part, esize)
                                       : get_element(&m, (e & ~1U) + part, esize);
                break;
            default: // ZIP
            {
                const uint32_t source = part * (elements / 2U) + e / 2U;
                value                 = 0U == (e & 1U) ? get_element(&n, source, esize)
                                                       : get_element(&m, source, esize);
                break;
            }
        }

        set_element(&result, e, esize, value);
    }

    write_vector(simd, rd, result, datasize);
    return true;
}

/// EXT: bytes `imm4` onwards of m:n.
static bool
extract(simd_t *simd, const uint32_t instruction)
{
    const uint32_t q      = (instruction >> 30U) & 1U;
    const uint32_t rm     = (instruction >> 16U) & 0x1FU;
    const uint32_t imm4   = (instruction >> 11U) & 0xFU;
    const uint32_t rn     = (instruction >> 5U) & 0x1FU;
    const uint32_t rd     = instruction & 0x1FU;
    const uint32_t bytes  = 1U == q ? 16U : 8U;
    const vector_t n      = read_vector(simd, rn);
    const vector_t m      = read_vector(simd, rm);
    vector_t       result = { { 0U, 0U } };

    if (0U == q && imm4 >= 8U)
    {
        return false;
    }

    for (uint32_t e = 0U; e < bytes; ++e)
    {
        const uint32_t source = imm4 + e;
        set_element(&result,
                    e,
                    8U,
                    source < bytes ? get_element(&n, source, 8U)
                                   : get_element(&m, source - bytes, 8U));
    }

    write_vector(simd, rd, result, bytes * 8U);
    return true;
}

/// TBL and TBX with one to four table registers.
static bool
table_lookup(simd_t *simd, const uint32_t instruction)
{
    const uint32_t q       = (instruction >> 30U) & 1U;
    const uint32_t rm      = (instruction >> 16U) & 0x1FU;
    const uint32_t length  = ((instruction >> 13U) & 0x3U) + 1U;
    const bool     is_tbx  = 1U == ((instruction >> 12U) & 1U);
    const uint32_t rn      = (instruction >> 5U) & 0x1FU;
    const uint32_t rd      = instruction & 0x1FU;
    const uint32_t bytes   = 1U == q ? 16U : 8U;
    const vector_t indices = read_vector(simd, rm);
    vector_t       table[4];
    vector_t       result = true == is_tbx ? read_vector(simd, rd) : (vector_t) { { 0U, 0U } };

    if (0U != ((instruction >> 22U) & 0x3U))
    {
        return false;
    }

    for (uint32_t i = 0U; i < length; ++i)
    {
        table[i] = read_vector(simd, (rn + i) % 32U);
    }

    for (uint32_t e = 0U; e < bytes; ++e)
    {
        const uint32_t index = (uint32_t)get_element(&indices, e, 8U);

        if (index < length * 16U)
        {
            set_element(&result, e, 8U, get_element(&table[index / 16U], index % 16U, 8U));
        }
        else if (false == is_tbx)
        {
            set_element(&result, e, 8U, 0U);
        }
    }

    write_vector(simd, rd, result, bytes * 8U);
    return true;
}

/// Advanced SIMD shift by immediate, vector and scalar.
static bool
shift_immediate(simd_t *simd, const uint32_t instruction, const bool is_scalar)
{
    const uint32_t q        = (instruction >> 30U) & 1U;
    const uint32_t u        = (instruction >> 29U) & 1U;
    const uint32_t immh     = (instruction >> 19U) & 0xFU;
    const uint32_t immhb    = (instruction >> 16U) & 0x7FU;
    const uint32_t opcode   = (instruction >> 11U) & 0x1FU;
    const uint32_t rn       = (instruction >> 5U) & 0x1FU;
    const uint32_t rd       = instruction & 0x1FU;
    const uint32_t datasize = 1U == q ? 128U : 64U;
    const vector_t n        = read_vector(simd, rn);
    const vector_t d        = read_vector(simd, rd);
    vector_t       result   = { { 0U, 0U } };

    if (0U == immh)
    {
        return false;
    }

    const uint32_t size        = 31U - (uint32_t)__builtin_clz(immh);
    const uint32_t esize       = 8U << size;
    const uint32_t right_shift = 2U * esize - immhb;
    const uint32_t left_shift  = immhb - esize;

    // Narrowing and lengthening forms: SHRN, RSHRN, SQSHRN, SQRSHRN, SQSHRUN, SQRSHRUN, UQSHRN,
    // UQRSHRN (opcode 1000x/1001x), and SSHLL, USHLL (opcode 10100).
    if (opcode >= 0x10U && opcode <= 0x14U)
    {
        if (3U == size)
        {
            return false;
        }

        if (0x14U == opcode)
        {
            if (true == is_scalar)
            {
                return false;
            }

            for (uint32_t e = 0U; e < 64U / esize; ++e)
            {
                const uint64_t x = get_element(&n, q * (64U / esize) + e, esize);
                const uint64_t widened
                    = 1U == u ? x : (uint64_t)sign_extend(x, esize) & mask_of(esize * 2U);
                set_element(&result, e, esize * 2U, (widened << left_shift) & mask_of(esize * 2U));
            }

            write_vector(simd, rd, result, 128U);
            return true;
        }

        const bool     is_rounding = 1U == (opcode & 1U);
        const uint32_t kind
            = (u << 1U) | ((opcode >> 1U) & 1U); // 0 SHRN, 1 SQSHRN, 2 SQSHRUN, 3 UQSHRN
        const uint32_t wide  = esize * 2U;
        const uint32_t count = true == is_scalar ? 1U : 64U / esize;
        const uint32_t base  = true == is_scalar ? 0U : q * count;

        if (true == is_scalar && 0U == kind)
        {
            return false; // SHRN and RSHRN have no scalar form.
        }

        result = 1U == q && false == is_scalar ? d : result;

        for (uint32_t e = 0U; e < count; ++e)
        {
            const uint64_t x            = get_element(&n, e, wide);
            const bool     signed_input = kind != 3U;
            // The right shift, 2 * esize - immh:immb, is at most the narrow element size.
            const uint64_t shifted = shift_element(simd,
                                                   x,
                                                   wide,
                                                   -(int64_t)right_shift,
                                                   false == signed_input,
                                                   is_rounding,
                                                   false,
                                                   false);
            uint64_t       value   = 0U;

            switch (kind)
            {
                case 0U:
                    value = shifted & mask_of(esize);
                    break;
                case 1U:
                    value = saturate_signed(simd, sign_extend(shifted, wide), esize);
                    break;
                case 2U:
                    value = saturate_signed_to_unsigned(simd, sign_extend(shifted, wide), esize);
                    break;
                default:
                    value = saturate_unsigned(simd, shifted, esize);
                    break;
            }

            set_element(&result, base + e, esize, value);
        }

        write_vector(simd, rd, result, true == is_scalar ? esize : (1U == q ? 128U : 64U));
        return true;
    }

    // Fixed-point conversions: SCVTF, UCVTF (opcode 11100) and FCVTZS, FCVTZU (11111).
    if (0x1CU == opcode || 0x1FU == opcode)
    {
        if (0U == size || (false == is_scalar && 3U == size && 0U == q))
        {
            return false;
        }

        const uint32_t elements = true == is_scalar ? 1U : datasize / esize;

        for (uint32_t e = 0U; e < elements; ++e)
        {
            const uint64_t x = get_element(&n, e, esize);
            const uint64_t value
                = 0x1CU == opcode
                      ? bal_float_from_integer_element(
                            simd->cpu->fpcr, &simd->flags, esize, x, right_shift, 1U == u, esize)
                      : bal_float_to_integer_element(simd->cpu->fpcr,
                                                     &simd->flags,
                                                     esize,
                                                     x,
                                                     right_shift,
                                                     3U,
                                                     1U == u,
                                                     esize);
            set_element(&result, e, esize, value);
        }

        write_vector(simd, rd, result, true == is_scalar ? esize : datasize);
        return true;
    }

    if ((false == is_scalar && 3U == size && 0U == q)
        || (true == is_scalar && 3U != size && opcode != 0x0EU && opcode != 0x0CU))
    {
        return false;
    }

    const uint32_t elements = true == is_scalar ? 1U : datasize / esize;

    for (uint32_t e = 0U; e < elements; ++e)
    {
        const uint64_t x     = get_element(&n, e, esize);
        const uint64_t old   = get_element(&d, e, esize);
        uint64_t       value = 0U;

        switch (opcode)
        {
            case 0x00U: // SSHR, USHR
            case 0x02U: // SSRA, USRA
            case 0x04U: // SRSHR, URSHR
            case 0x06U: // SRSRA, URSRA
                value = shift_element(
                    simd, x, esize, -(int64_t)right_shift, 1U == u, opcode >= 0x04U, false, false);
                if (0x02U == opcode || 0x06U == opcode)
                {
                    value = (value + old) & mask_of(esize);
                }
                break;
            case 0x08U: // SRI (U = 1)
            {
                if (0U == u)
                {
                    return false;
                }

                const uint64_t keep = right_shift >= esize
                                          ? mask_of(esize)
                                          : ~(mask_of(esize) >> right_shift) & mask_of(esize);
                value               = (old & keep) | (right_shift >= esize ? 0U : x >> right_shift);
                break;
            }
            case 0x0AU: // SHL (U = 0), SLI (U = 1)
            {
                const uint64_t shifted = (x << left_shift) & mask_of(esize);
                value = 0U == u ? shifted : (old & ((1ULL << left_shift) - 1U)) | shifted;
                break;
            }
            case 0x0CU: // SQSHLU (U = 1)
                if (0U == u)
                {
                    return false;
                }
                value
                    = shift_element(simd, x, esize, (int64_t)left_shift, false, false, true, true);
                break;
            case 0x0EU: // SQSHL, UQSHL (immediate)
                value = shift_element(
                    simd, x, esize, (int64_t)left_shift, 1U == u, false, true, false);
                break;
            default:
                return false;
        }

        set_element(&result, e, esize, value);
    }

    write_vector(simd, rd, result, true == is_scalar ? esize : datasize);
    return true;
}

/// Advanced SIMD vector x indexed element, vector and scalar: FMUL, FMLA, FMLS, FMULX, MUL, MLA,
/// MLS, SMULL, UMULL, SMLAL, UMLAL, SMLSL, UMLSL, the saturating doubling multiplies, SDOT and
/// UDOT.
static bool
indexed_element(simd_t *simd, const uint32_t instruction, const bool is_scalar)
{
    const uint32_t q        = (instruction >> 30U) & 1U;
    const uint32_t u        = (instruction >> 29U) & 1U;
    const uint32_t size     = (instruction >> 22U) & 0x3U;
    const uint32_t l        = (instruction >> 21U) & 1U;
    const uint32_t m_bit    = (instruction >> 20U) & 1U;
    const uint32_t rm_low   = (instruction >> 16U) & 0xFU;
    const uint32_t opcode   = (instruction >> 12U) & 0xFU;
    const uint32_t h        = (instruction >> 11U) & 1U;
    const uint32_t rn       = (instruction >> 5U) & 0x1FU;
    const uint32_t rd       = instruction & 0x1FU;
    const uint32_t datasize = 1U == q ? 128U : 64U;
    const vector_t n        = read_vector(simd, rn);
    const vector_t d        = read_vector(simd, rd);
    vector_t       result   = { { 0U, 0U } };

    // Floating point: FMLA (0001), FMLS (0101), FMUL (1001, U = 0), FMULX (1001, U = 1).
    if (1U == (opcode & 0x3U) && opcode != 0xDU)
    {
        if (1U == size)
        {
            return false;
        }

        // Size 00 is half precision, whose index is H:L:M and whose Rm has four bits.
        const uint32_t esize = 0U == size ? 16U : (3U == size ? 64U : 32U);

        if ((64U == esize && 1U == l) || (false == is_scalar && 64U == esize && 0U == q)
            || (1U == u && opcode != 0x9U))
        {
            return false;
        }

        const uint32_t index    = 64U == esize   ? h
                                  : 16U == esize ? (h << 2U) | (l << 1U) | m_bit
                                                 : (h << 1U) | l;
        const uint32_t rm       = 16U == esize ? rm_low : (m_bit << 4U) | rm_low;
        const vector_t m        = read_vector(simd, rm);
        const uint64_t element  = get_element(&m, index, esize);
        const uint32_t elements = true == is_scalar ? 1U : datasize / esize;

        for (uint32_t e = 0U; e < elements; ++e)
        {
            const uint64_t a     = get_element(&n, e, esize);
            uint64_t       value = 0U;

            if (0x9U == opcode)
            {
                value = float_operation(simd,
                                        esize,
                                        1U == u ? BAL_FLOAT_MULTIPLY_EXTENDED : BAL_FLOAT_MULTIPLY,
                                        false,
                                        a,
                                        element,
                                        0U);
            }
            else
            {
                value = float_operation(simd,
                                        esize,
                                        BAL_FLOAT_MULTIPLY_ADD,
                                        0x5U == opcode,
                                        a,
                                        element,
                                        get_element(&d, e, esize));
            }

            set_element(&result, e, esize, value);
        }

        write_vector(simd, rd, result, true == is_scalar ? esize : datasize);
        return true;
    }

    if (0U == size || 3U == size)
    {
        return false;
    }

    const uint32_t esize   = 8U << size;
    const uint32_t index   = 1U == size ? (h << 2U) | (l << 1U) | m_bit : (h << 1U) | l;
    const uint32_t rm      = 1U == size ? rm_low : (m_bit << 4U) | rm_low;
    const vector_t m       = read_vector(simd, rm);
    const uint64_t element = get_element(&m, index, esize);

    // SQDMULH (U = 0, 1100), SQRDMULH (U = 0, 1101), SQRDMLAH (U = 1, 1101) and SQRDMLSH
    // (U = 1, 1111).
    if ((0U == u && (0xCU == opcode || 0xDU == opcode))
        || (1U == u && (0xDU == opcode || 0xFU == opcode)))
    {
        const uint32_t elements = true == is_scalar ? 1U : datasize / esize;

        for (uint32_t e = 0U; e < elements; ++e)
        {
            const uint64_t a = get_element(&n, e, esize);
            const uint64_t value
                = 0U == u ? doubling_multiply_high(simd, a, element, esize, 0xDU == opcode)
                          : doubling_multiply_accumulate(
                                simd, get_element(&d, e, esize), a, element, esize, 0xFU == opcode);
            set_element(&result, e, esize, value);
        }

        write_vector(simd, rd, result, true == is_scalar ? esize : datasize);
        return true;
    }

    // SQDMLAL (0011), SQDMLSL (0111) and SQDMULL (1011), with the "2" forms taking the upper
    // half of n.
    if (0U == u && (0x3U == opcode || 0x7U == opcode || 0xBU == opcode))
    {
        // The matching three different opcodes: 1001, 1011 and 1101.
        const uint32_t long_opcode = 0x3U == opcode ? 0x9U : (0x7U == opcode ? 0xBU : 0xDU);
        const uint32_t wide        = esize * 2U;
        const uint32_t elements    = true == is_scalar ? 1U : 64U / esize;
        const uint32_t part        = true == is_scalar ? 0U : q * elements;

        for (uint32_t e = 0U; e < elements; ++e)
        {
            set_element(&result,
                        e,
                        wide,
                        doubling_long_operation(simd,
                                                long_opcode,
                                                get_element(&d, e, wide),
                                                get_element(&n, part + e, esize),
                                                element,
                                                esize));
        }

        write_vector(simd, rd, result, true == is_scalar ? wide : 128U);
        return true;
    }

    // SDOT and UDOT (1110) on words, with a 32-bit index H:L into a 32-register Rm.
    if (false == is_scalar && 0xEU == opcode && 2U == size)
    {
        for (uint32_t e = 0U; e < datasize / 32U; ++e)
        {
            set_element(
                &result,
                e,
                32U,
                dot_product(get_element(&d, e, 32U), get_element(&n, e, 32U), element, 1U == u));
        }

        write_vector(simd, rd, result, datasize);
        return true;
    }

    if (true == is_scalar)
    {
        return false;
    }

    // MUL (U = 0, 1000), MLA (U = 1, 0000), MLS (U = 1, 0100).
    if ((0U == u && 0x8U == opcode) || (1U == u && (0x0U == opcode || 0x4U == opcode)))
    {
        for (uint32_t e = 0U; e < datasize / esize; ++e)
        {
            const uint64_t product = get_element(&n, e, esize) * element;
            const uint64_t old     = get_element(&d, e, esize);
            const uint64_t value
                = 0x8U == opcode ? product : (0x0U == opcode ? old + product : old - product);
            set_element(&result, e, esize, value & mask_of(esize));
        }

        write_vector(simd, rd, result, datasize);
        return true;
    }

    // SMULL, UMULL (1010), SMLAL, UMLAL (0010), SMLSL, UMLSL (0110), with the "2" forms taking
    // the upper half of n.
    if (0xAU == opcode || 0x2U == opcode || 0x6U == opcode)
    {
        const uint32_t wide     = esize * 2U;
        const uint32_t elements = 64U / esize;

        for (uint32_t e = 0U; e < elements; ++e)
        {
            const uint64_t a = get_element(&n, q * elements + e, esize);
            const uint64_t product
                = 1U == u ? a * element
                          : (uint64_t)(sign_extend(a, esize) * sign_extend(element, esize));
            const uint64_t old = get_element(&d, e, wide);
            const uint64_t value
                = 0xAU == opcode ? product : (0x2U == opcode ? old + product : old - product);
            set_element(&result, e, wide, value & mask_of(wide));
        }

        write_vector(simd, rd, result, 128U);
        return true;
    }

    return false;
}

/// Advanced SIMD three different: the long, wide and narrow additions, subtractions and
/// multiplications.
static bool
three_different(simd_t *simd, const uint32_t instruction)
{
    const uint32_t q      = (instruction >> 30U) & 1U;
    const uint32_t u      = (instruction >> 29U) & 1U;
    const uint32_t size   = (instruction >> 22U) & 0x3U;
    const uint32_t rm     = (instruction >> 16U) & 0x1FU;
    const uint32_t opcode = (instruction >> 12U) & 0xFU;
    const uint32_t rn     = (instruction >> 5U) & 0x1FU;
    const uint32_t rd     = instruction & 0x1FU;
    const vector_t n      = read_vector(simd, rn);
    const vector_t m      = read_vector(simd, rm);
    const vector_t d      = read_vector(simd, rd);
    vector_t       result = { { 0U, 0U } };

    // PMULL and PMULL2 on doublewords: a 128-bit carry-less product.
    if (0xEU == opcode && 0U == u && 3U == size)
    {
        const uint64_t a = n.d[q];
        const uint64_t b = m.d[q];

        for (uint32_t i = 0U; i < 64U; ++i)
        {
            if (0U != ((b >> i) & 1U))
            {
                result.d[0] ^= a << i;
                result.d[1] ^= 0U == i ? 0U : a >> (64U - i);
            }
        }

        write_vector(simd, rd, result, 128U);
        return true;
    }

    if (3U == size)
    {
        return false;
    }

    const uint32_t esize    = 8U << size;
    const uint32_t wide     = esize * 2U;
    const uint32_t elements = 64U / esize;
    const uint32_t part     = q * elements;

    // ADDHN, RADDHN, SUBHN, RSUBHN: the high halves of wide sums, written to the lower or upper
    // half of the destination.
    if (0x4U == opcode || 0x6U == opcode)
    {
        result = 1U == q ? d : result;

        for (uint32_t e = 0U; e < elements; ++e)
        {
            const uint64_t a     = get_element(&n, e, wide);
            const uint64_t b     = get_element(&m, e, wide);
            uint64_t       value = 0x4U == opcode ? a + b : a - b;

            if (1U == u)
            {
                value += 1ULL << (esize - 1U);
            }

            set_element(&result, part + e, esize, (value & mask_of(wide)) >> esize);
        }

        write_vector(simd, rd, result, 1U == q ? 128U : 64U);
        return true;
    }

    // SQDMLAL (1001), SQDMLSL (1011) and SQDMULL (1101) on halfwords and words.
    if (0x9U == opcode || 0xBU == opcode || 0xDU == opcode)
    {
        if (1U == u || 0U == size)
        {
            return false;
        }

        for (uint32_t e = 0U; e < elements; ++e)
        {
            set_element(&result,
                        e,
                        wide,
                        doubling_long_operation(simd,
                                                opcode,
                                                get_element(&d, e, wide),
                                                get_element(&n, part + e, esize),
                                                get_element(&m, part + e, esize),
                                                esize));
        }

        write_vector(simd, rd, result, 128U);
        return true;
    }

    if (0xFU == opcode || (0xEU == opcode && (1U == u || size != 0U)))
    {
        return false; // PMULL only takes bytes and doublewords.
    }

    for (uint32_t e = 0U; e < elements; ++e)
    {
        const uint64_t raw_b = get_element(&m, part + e, esize);
        const uint64_t b     = 1U == u ? raw_b : (uint64_t)sign_extend(raw_b, esize);
        const uint64_t old   = get_element(&d, e, wide);
        uint64_t       a     = 0U;

        // The W forms take a wide first operand.
        if (0x1U == opcode || 0x3U == opcode)
        {
            a = get_element(&n, e, wide);
        }
        else
        {
            const uint64_t raw_a = get_element(&n, part + e, esize);
            a                    = 1U == u ? raw_a : (uint64_t)sign_extend(raw_a, esize);
        }

        uint64_t value = 0U;

        switch (opcode)
        {
            case 0x0U: // SADDL, UADDL
            case 0x1U: // SADDW, UADDW
                value = a + b;
                break;
            case 0x2U: // SSUBL, USUBL
            case 0x3U: // SSUBW, USUBW
                value = a - b;
                break;
            case 0x5U: // SABAL, UABAL
            case 0x7U: // SABDL, UABDL
            {
                const bool     is_a_larger = 1U == u ? a > b : (int64_t)a > (int64_t)b;
                const uint64_t difference  = true == is_a_larger ? a - b : b - a;
                value                      = 0x5U == opcode ? old + difference : difference;
                break;
            }
            case 0x8U: // SMLAL, UMLAL
                value = old + a * b;
                break;
            case 0xAU: // SMLSL, UMLSL
                value = old - a * b;
                break;
            case 0xCU: // SMULL, UMULL
                value = a * b;
                break;
            default: // PMULL with bytes
                value = polynomial_multiply(get_element(&n, part + e, esize), raw_b, esize, wide);
                break;
        }

        set_element(&result, e, wide, value & mask_of(wide));
    }

    write_vector(simd, rd, result, 128U);
    return true;
}

/// Advanced SIMD scalar pairwise: ADDP and the floating-point FADDP, FMAXP, FMINP, FMAXNMP,
/// FMINNMP on the two elements of a vector.
static bool
scalar_pairwise(simd_t *simd, const uint32_t instruction)
{
    const uint32_t u      = (instruction >> 29U) & 1U;
    const uint32_t size   = (instruction >> 22U) & 0x3U;
    const uint32_t opcode = (instruction >> 12U) & 0x1FU;
    const uint32_t rn     = (instruction >> 5U) & 0x1FU;
    const uint32_t rd     = instruction & 0x1FU;
    const vector_t n      = read_vector(simd, rn);
    vector_t       result = { { 0U, 0U } };

    if (0U == u && 0x1BU == opcode)
    {
        if (size != 3U)
        {
            return false;
        }

        set_element(&result, 0U, 64U, n.d[0] + n.d[1]);
        write_vector(simd, rd, result, 64U);
        return true;
    }

    // Floating point: single or double precision with U = 1, half precision with U = 0, sz = 0.
    if (0U == u && 1U == (size & 1U))
    {
        return false;
    }

    const uint32_t        esize     = 0U == u ? 16U : (1U == (size & 1U) ? 64U : 32U);
    const bool            is_second = 1U == (size >> 1U);
    bal_float_operation_t operation = BAL_FLOAT_ADD;

    switch (opcode)
    {
        case 0x0CU:
            operation = true == is_second ? BAL_FLOAT_MINIMUM_NUMBER : BAL_FLOAT_MAXIMUM_NUMBER;
            break;
        case 0x0DU:
            if (true == is_second)
            {
                return false;
            }
            operation = BAL_FLOAT_ADD;
            break;
        case 0x0FU:
            operation = true == is_second ? BAL_FLOAT_MINIMUM : BAL_FLOAT_MAXIMUM;
            break;
        default:
            return false;
    }

    set_element(&result,
                0U,
                esize,
                float_operation(simd,
                                esize,
                                operation,
                                false,
                                get_element(&n, 0U, esize),
                                get_element(&n, 1U, esize),
                                0U));
    write_vector(simd, rd, result, esize);
    return true;
}

// The cryptographic extension.

/// The AES S-box and its inverse.
static const uint8_t AES_SBOX[256] = {
    0x63, 0x7C, 0x77, 0x7B, 0xF2, 0x6B, 0x6F, 0xC5, 0x30, 0x01, 0x67, 0x2B, 0xFE, 0xD7, 0xAB, 0x76,
    0xCA, 0x82, 0xC9, 0x7D, 0xFA, 0x59, 0x47, 0xF0, 0xAD, 0xD4, 0xA2, 0xAF, 0x9C, 0xA4, 0x72, 0xC0,
    0xB7, 0xFD, 0x93, 0x26, 0x36, 0x3F, 0xF7, 0xCC, 0x34, 0xA5, 0xE5, 0xF1, 0x71, 0xD8, 0x31, 0x15,
    0x04, 0xC7, 0x23, 0xC3, 0x18, 0x96, 0x05, 0x9A, 0x07, 0x12, 0x80, 0xE2, 0xEB, 0x27, 0xB2, 0x75,
    0x09, 0x83, 0x2C, 0x1A, 0x1B, 0x6E, 0x5A, 0xA0, 0x52, 0x3B, 0xD6, 0xB3, 0x29, 0xE3, 0x2F, 0x84,
    0x53, 0xD1, 0x00, 0xED, 0x20, 0xFC, 0xB1, 0x5B, 0x6A, 0xCB, 0xBE, 0x39, 0x4A, 0x4C, 0x58, 0xCF,
    0xD0, 0xEF, 0xAA, 0xFB, 0x43, 0x4D, 0x33, 0x85, 0x45, 0xF9, 0x02, 0x7F, 0x50, 0x3C, 0x9F, 0xA8,
    0x51, 0xA3, 0x40, 0x8F, 0x92, 0x9D, 0x38, 0xF5, 0xBC, 0xB6, 0xDA, 0x21, 0x10, 0xFF, 0xF3, 0xD2,
    0xCD, 0x0C, 0x13, 0xEC, 0x5F, 0x97, 0x44, 0x17, 0xC4, 0xA7, 0x7E, 0x3D, 0x64, 0x5D, 0x19, 0x73,
    0x60, 0x81, 0x4F, 0xDC, 0x22, 0x2A, 0x90, 0x88, 0x46, 0xEE, 0xB8, 0x14, 0xDE, 0x5E, 0x0B, 0xDB,
    0xE0, 0x32, 0x3A, 0x0A, 0x49, 0x06, 0x24, 0x5C, 0xC2, 0xD3, 0xAC, 0x62, 0x91, 0x95, 0xE4, 0x79,
    0xE7, 0xC8, 0x37, 0x6D, 0x8D, 0xD5, 0x4E, 0xA9, 0x6C, 0x56, 0xF4, 0xEA, 0x65, 0x7A, 0xAE, 0x08,
    0xBA, 0x78, 0x25, 0x2E, 0x1C, 0xA6, 0xB4, 0xC6, 0xE8, 0xDD, 0x74, 0x1F, 0x4B, 0xBD, 0x8B, 0x8A,
    0x70, 0x3E, 0xB5, 0x66, 0x48, 0x03, 0xF6, 0x0E, 0x61, 0x35, 0x57, 0xB9, 0x86, 0xC1, 0x1D, 0x9E,
    0xE1, 0xF8, 0x98, 0x11, 0x69, 0xD9, 0x8E, 0x94, 0x9B, 0x1E, 0x87, 0xE9, 0xCE, 0x55, 0x28, 0xDF,
    0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68, 0x41, 0x99, 0x2D, 0x0F, 0xB0, 0x54, 0xBB, 0x16,
};

static const uint8_t AES_INVERSE_SBOX[256] = {
    0x52, 0x09, 0x6A, 0xD5, 0x30, 0x36, 0xA5, 0x38, 0xBF, 0x40, 0xA3, 0x9E, 0x81, 0xF3, 0xD7, 0xFB,
    0x7C, 0xE3, 0x39, 0x82, 0x9B, 0x2F, 0xFF, 0x87, 0x34, 0x8E, 0x43, 0x44, 0xC4, 0xDE, 0xE9, 0xCB,
    0x54, 0x7B, 0x94, 0x32, 0xA6, 0xC2, 0x23, 0x3D, 0xEE, 0x4C, 0x95, 0x0B, 0x42, 0xFA, 0xC3, 0x4E,
    0x08, 0x2E, 0xA1, 0x66, 0x28, 0xD9, 0x24, 0xB2, 0x76, 0x5B, 0xA2, 0x49, 0x6D, 0x8B, 0xD1, 0x25,
    0x72, 0xF8, 0xF6, 0x64, 0x86, 0x68, 0x98, 0x16, 0xD4, 0xA4, 0x5C, 0xCC, 0x5D, 0x65, 0xB6, 0x92,
    0x6C, 0x70, 0x48, 0x50, 0xFD, 0xED, 0xB9, 0xDA, 0x5E, 0x15, 0x46, 0x57, 0xA7, 0x8D, 0x9D, 0x84,
    0x90, 0xD8, 0xAB, 0x00, 0x8C, 0xBC, 0xD3, 0x0A, 0xF7, 0xE4, 0x58, 0x05, 0xB8, 0xB3, 0x45, 0x06,
    0xD0, 0x2C, 0x1E, 0x8F, 0xCA, 0x3F, 0x0F, 0x02, 0xC1, 0xAF, 0xBD, 0x03, 0x01, 0x13, 0x8A, 0x6B,
    0x3A, 0x91, 0x11, 0x41, 0x4F, 0x67, 0xDC, 0xEA, 0x97, 0xF2, 0xCF, 0xCE, 0xF0, 0xB4, 0xE6, 0x73,
    0x96, 0xAC, 0x74, 0x22, 0xE7, 0xAD, 0x35, 0x85, 0xE2, 0xF9, 0x37, 0xE8, 0x1C, 0x75, 0xDF, 0x6E,
    0x47, 0xF1, 0x1A, 0x71, 0x1D, 0x29, 0xC5, 0x89, 0x6F, 0xB7, 0x62, 0x0E, 0xAA, 0x18, 0xBE, 0x1B,
    0xFC, 0x56, 0x3E, 0x4B, 0xC6, 0xD2, 0x79, 0x20, 0x9A, 0xDB, 0xC0, 0xFE, 0x78, 0xCD, 0x5A, 0xF4,
    0x1F, 0xDD, 0xA8, 0x33, 0x88, 0x07, 0xC7, 0x31, 0xB1, 0x12, 0x10, 0x59, 0x27, 0x80, 0xEC, 0x5F,
    0x60, 0x51, 0x7F, 0xA9, 0x19, 0xB5, 0x4A, 0x0D, 0x2D, 0xE5, 0x7A, 0x9F, 0x93, 0xC9, 0x9C, 0xEF,
    0xA0, 0xE0, 0x3B, 0x4D, 0xAE, 0x2A, 0xF5, 0xB0, 0xC8, 0xEB, 0xBB, 0x3C, 0x83, 0x53, 0x99, 0x61,
    0x17, 0x2B, 0x04, 0x7E, 0xBA, 0x77, 0xD6, 0x26, 0xE1, 0x69, 0x14, 0x63, 0x55, 0x21, 0x0C, 0x7D,
};

/// Multiplies two elements of GF(2^8) modulo the AES polynomial x^8 + x^4 + x^3 + x + 1.
static uint32_t
gf_multiply(uint32_t a, uint32_t b)
{
    uint32_t result = 0U;

    while (b != 0U)
    {
        if (0U != (b & 1U))
        {
            result ^= a;
        }

        a = ((a << 1U) ^ (0U != (a & 0x80U) ? 0x1BU : 0U)) & 0xFFU;
        b >>= 1U;
    }

    return result;
}

/// AESE, AESD, AESMC and AESIMC. The state's byte r + 4 * c is row r of column c.
static bool
aes(simd_t *simd, const uint32_t instruction)
{
    const uint32_t opcode = (instruction >> 12U) & 0x1FU;
    const uint32_t rn     = (instruction >> 5U) & 0x1FU;
    const uint32_t rd     = instruction & 0x1FU;
    const vector_t n      = read_vector(simd, rn);
    const vector_t d      = read_vector(simd, rd);
    vector_t       input  = n;
    vector_t       result = { { 0U, 0U } };

    if (opcode < 0x4U || opcode > 0x7U)
    {
        return false;
    }

    // AESE and AESD first add the round key held in the destination.
    if (opcode <= 0x5U)
    {
        input.d[0] ^= d.d[0];
        input.d[1] ^= d.d[1];
    }

    for (uint32_t c = 0U; c < 4U; ++c)
    {
        for (uint32_t r = 0U; r < 4U; ++r)
        {
            switch (opcode)
            {
                case 0x4U: // AESE: ShiftRows, then SubBytes.
                {
                    const uint64_t byte = get_element(&input, r + 4U * ((c + r) % 4U), 8U);
                    set_element(&result, r + 4U * c, 8U, AES_SBOX[byte]);
                    break;
                }
                case 0x5U: // AESD: InvShiftRows, then InvSubBytes.
                {
                    const uint64_t byte = get_element(&input, r + 4U * c, 8U);
                    set_element(&result, r + 4U * ((c + r) % 4U), 8U, AES_INVERSE_SBOX[byte]);
                    break;
                }
                default: // AESMC and AESIMC: each output byte mixes the four bytes of its column.
                {
                    static const uint32_t MIX[4]         = { 2U, 3U, 1U, 1U };
                    static const uint32_t INVERSE_MIX[4] = { 14U, 11U, 13U, 9U };
                    const uint32_t       *coefficients   = 0x6U == opcode ? MIX : INVERSE_MIX;
                    uint32_t              value          = 0U;

                    for (uint32_t j = 0U; j < 4U; ++j)
                    {
                        value ^= gf_multiply((uint32_t)get_element(&input, j + 4U * c, 8U),
                                             coefficients[(j - r) & 3U]);
                    }

                    set_element(&result, r + 4U * c, 8U, value);
                    break;
                }
            }
        }
    }

    write_vector(simd, rd, result, 128U);
    return true;
}

static uint32_t
rotate_left(const uint32_t value, const uint32_t amount)
{
    return (value << amount) | (value >> ((32U - amount) & 31U));
}

static uint32_t
rotate_right(const uint32_t value, const uint32_t amount)
{
    return rotate_left(value, (32U - amount) & 31U);
}

static uint32_t
sha_choose(const uint32_t x, const uint32_t y, const uint32_t z)
{
    return ((y ^ z) & x) ^ z;
}

static uint32_t
sha_majority(const uint32_t x, const uint32_t y, const uint32_t z)
{
    return (x & y) | ((x | y) & z);
}

static void
get_words(const vector_t *vector, uint32_t words[4])
{
    for (uint32_t e = 0U; e < 4U; ++e)
    {
        words[e] = (uint32_t)get_element(vector, e, 32U);
    }
}

static vector_t
from_words(const uint32_t words[4])
{
    vector_t vector = { { 0U, 0U } };

    for (uint32_t e = 0U; e < 4U; ++e)
    {
        set_element(&vector, e, 32U, words[e]);
    }

    return vector;
}

/// Four rounds of SHA-256 on the state halves `x` (A to D) and `y` (E to H). SHA256H keeps the
/// new `x`, SHA256H2 the new `y`.
static void
sha256_rounds(uint32_t x[4], uint32_t y[4], const uint32_t w[4])
{
    for (uint32_t e = 0U; e < 4U; ++e)
    {
        const uint32_t sigma1
            = rotate_right(y[0], 6U) ^ rotate_right(y[0], 11U) ^ rotate_right(y[0], 25U);
        const uint32_t sigma0
            = rotate_right(x[0], 2U) ^ rotate_right(x[0], 13U) ^ rotate_right(x[0], 22U);
        const uint32_t t = y[3] + sigma1 + sha_choose(y[0], y[1], y[2]) + w[e];
        x[3] += t;
        y[3] = t + sigma0 + sha_majority(x[0], x[1], x[2]);

        // Rotate Y:X left by one word.
        const uint32_t top_x = x[3];
        x[3]                 = x[2];
        x[2]                 = x[1];
        x[1]                 = x[0];
        x[0]                 = y[3];
        y[3]                 = y[2];
        y[2]                 = y[1];
        y[1]                 = y[0];
        y[0]                 = top_x;
    }
}

/// SHA three register: SHA1C, SHA1P, SHA1M, SHA1SU0, SHA256H, SHA256H2 and SHA256SU1.
static bool
sha_three_register(simd_t *simd, const uint32_t instruction)
{
    const uint32_t rm     = (instruction >> 16U) & 0x1FU;
    const uint32_t opcode = (instruction >> 12U) & 0x7U;
    const uint32_t rn     = (instruction >> 5U) & 0x1FU;
    const uint32_t rd     = instruction & 0x1FU;
    const vector_t n      = read_vector(simd, rn);
    const vector_t m      = read_vector(simd, rm);
    const vector_t d      = read_vector(simd, rd);
    uint32_t       x[4];
    uint32_t       y[4];
    uint32_t       w[4];
    vector_t       result = { { 0U, 0U } };

    get_words(&d, x);
    get_words(&n, y);
    get_words(&m, w);

    switch (opcode)
    {
        case 0x0U: // SHA1C
        case 0x1U: // SHA1P
        case 0x2U: // SHA1M
        {
            uint32_t e_value = y[0];

            for (uint32_t e = 0U; e < 4U; ++e)
            {
                uint32_t t = 0U;

                if (0x0U == opcode)
                {
                    t = sha_choose(x[1], x[2], x[3]);
                }
                else if (0x1U == opcode)
                {
                    t = x[1] ^ x[2] ^ x[3];
                }
                else
                {
                    t = sha_majority(x[1], x[2], x[3]);
                }

                e_value += rotate_left(x[0], 5U) + t + w[e];
                x[1] = rotate_left(x[1], 30U);

                // Rotate Y:X left by one word, Y being the single word e_value.
                const uint32_t top = x[3];
                x[3]               = x[2];
                x[2]               = x[1];
                x[1]               = x[0];
                x[0]               = e_value;
                e_value            = top;
            }

            result = from_words(x);
            break;
        }
        case 0x3U: // SHA1SU0
            result.d[0] = d.d[1] ^ d.d[0] ^ m.d[0];
            result.d[1] = n.d[0] ^ d.d[1] ^ m.d[1];
            break;
        case 0x4U: // SHA256H
            sha256_rounds(x, y, w);
            result = from_words(x);
            break;
        case 0x5U: // SHA256H2
            sha256_rounds(y, x, w);
            result = from_words(x);
            break;
        case 0x6U: // SHA256SU1
        {
            // T0 = m<31:0> : n<127:32>; the first two results feed the last two.
            const uint32_t t0[4] = { y[1], y[2], y[3], w[0] };
            uint32_t       words[4];

            for (uint32_t e = 0U; e < 4U; ++e)
            {
                const uint32_t source = e < 2U ? w[e + 2U] : words[e - 2U];
                const uint32_t sigma
                    = rotate_right(source, 17U) ^ rotate_right(source, 19U) ^ (source >> 10U);
                words[e] = sigma + x[e] + t0[e];
            }

            result = from_words(words);
            break;
        }
        default:
            return false;
    }

    write_vector(simd, rd, result, 128U);
    return true;
}

/// SHA two register: SHA1H, SHA1SU1 and SHA256SU0.
static bool
sha_two_register(simd_t *simd, const uint32_t instruction)
{
    const uint32_t opcode = (instruction >> 12U) & 0x1FU;
    const uint32_t rn     = (instruction >> 5U) & 0x1FU;
    const uint32_t rd     = instruction & 0x1FU;
    const vector_t n      = read_vector(simd, rn);
    const vector_t d      = read_vector(simd, rd);
    uint32_t       x[4];
    uint32_t       y[4];
    uint32_t       words[4];

    get_words(&d, x);
    get_words(&n, y);

    switch (opcode)
    {
        case 0x0U: // SHA1H
        {
            vector_t result = { { 0U, 0U } };
            set_element(&result, 0U, 32U, rotate_left(y[0], 30U));
            write_vector(simd, rd, result, 32U);
            return true;
        }
        case 0x1U: // SHA1SU1
        {
            // T = d EOR (n >> 32).
            const uint32_t t[4] = { x[0] ^ y[1], x[1] ^ y[2], x[2] ^ y[3], x[3] };
            words[0]            = rotate_left(t[0], 1U);
            words[1]            = rotate_left(t[1], 1U);
            words[2]            = rotate_left(t[2], 1U);
            words[3]            = rotate_left(t[3], 1U) ^ rotate_left(t[0], 2U);
            break;
        }
        case 0x2U: // SHA256SU0
        {
            // T = n<31:0> : d<127:32>.
            const uint32_t t[4] = { x[1], x[2], x[3], y[0] };

            for (uint32_t e = 0U; e < 4U; ++e)
            {
                const uint32_t sigma
                    = rotate_right(t[e], 7U) ^ rotate_right(t[e], 18U) ^ (t[e] >> 3U);
                words[e] = sigma + x[e];
            }

            break;
        }
        default:
            return false;
    }

    write_vector(simd, rd, from_words(words), 128U);
    return true;
}

/// Decodes and executes one instruction. Returns false if it is not implemented.
static bool
execute(simd_t *simd, const uint32_t instruction)
{
    // The cryptographic extension: AES, SHA-1 and SHA-256.
    if (0x4E280800U == (instruction & 0xFF3E0C00U))
    {
        return aes(simd, instruction);
    }

    if (0x5E000000U == (instruction & 0xFF208C00U))
    {
        return sha_three_register(simd, instruction);
    }

    if (0x5E280800U == (instruction & 0xFF3E0C00U))
    {
        return sha_two_register(simd, instruction);
    }

    // Advanced SIMD scalar: 01 U 1111x ...
    if (0x5E000000U == (instruction & 0xDE000000U))
    {
        // Half precision three same and two-register miscellaneous.
        if (0x5E400400U == (instruction & 0xDF60C400U))
        {
            return float_three_same(simd, instruction, true, 16U);
        }

        if (0x5E780800U == (instruction & 0xDF7E0C00U))
        {
            return float_two_register_misc(simd, instruction, true, 16U);
        }

        if (0x5E200400U == (instruction & 0xDF200400U))
        {
            return three_same(simd, instruction, true);
        }

        if (0x5E200800U == (instruction & 0xDF3E0C00U))
        {
            return two_register_misc(simd, instruction, true);
        }

        if (0x5E300800U == (instruction & 0xDF3E0C00U))
        {
            return scalar_pairwise(simd, instruction);
        }

        if (0x5E008400U == (instruction & 0xDF208400U))
        {
            return three_same_extra(simd, instruction, true);
        }

        if (0x5E200000U == (instruction & 0xDF200C00U))
        {
            return scalar_three_different(simd, instruction);
        }

        if (0x5E000400U == (instruction & 0xFFE0FC00U))
        {
            return copy(simd, instruction, true);
        }

        if (0x5F000400U == (instruction & 0xDF800400U))
        {
            return shift_immediate(simd, instruction, true);
        }

        if (0x5F000000U == (instruction & 0xDF000400U))
        {
            return indexed_element(simd, instruction, true);
        }

        return false;
    }

    // Advanced SIMD vector: 0 Q U 0111x ...
    if (0x0E000000U != (instruction & 0x9E000000U))
    {
        return false;
    }

    // Half precision three same and two-register miscellaneous.
    if (0x0E400400U == (instruction & 0x9F60C400U))
    {
        return float_three_same(simd, instruction, false, 16U);
    }

    if (0x0E780800U == (instruction & 0x9F7E0C00U))
    {
        return float_two_register_misc(simd, instruction, false, 16U);
    }

    if (0x0E200400U == (instruction & 0x9F200400U))
    {
        return three_same(simd, instruction, false);
    }

    if (0x0E200800U == (instruction & 0x9F3E0C00U))
    {
        return two_register_misc(simd, instruction, false);
    }

    if (0x0E300800U == (instruction & 0x9F3E0C00U))
    {
        return across_lanes(simd, instruction);
    }

    if (0x0E200000U == (instruction & 0x9F200C00U))
    {
        return three_different(simd, instruction);
    }

    if (0x0E008400U == (instruction & 0x9F208400U))
    {
        return three_same_extra(simd, instruction, false);
    }

    if (0x0E000400U == (instruction & 0x9FE08400U))
    {
        return copy(simd, instruction, false);
    }

    if (0x0E000800U == (instruction & 0xBF208C00U))
    {
        return permute(simd, instruction);
    }

    if (0x2E000000U == (instruction & 0xBFE08400U))
    {
        return extract(simd, instruction);
    }

    if (0x0E000000U == (instruction & 0xBF208C00U))
    {
        return table_lookup(simd, instruction);
    }

    if (0x0F000400U == (instruction & 0x9FF80400U))
    {
        return modified_immediate(simd, instruction);
    }

    if (0x0F000400U == (instruction & 0x9F800400U))
    {
        return shift_immediate(simd, instruction, false);
    }

    if (0x0F000000U == (instruction & 0x9F000400U))
    {
        return indexed_element(simd, instruction, false);
    }

    return false;
}

bool
bal_is_simd_supported(const uint32_t instruction)
{
    // Execute against scratch state: the operations are pure functions of the registers.
    bal_cpu_t cpu;
    (void)memset(&cpu, 0, sizeof(cpu));
    simd_t simd = { .cpu = &cpu, .flags = 0U };
    return execute(&simd, instruction);
}

uint64_t
bal_execute_simd(bal_cpu_t *cpu, const uint32_t instruction)
{
    simd_t simd = { .cpu = cpu, .flags = 0U };
    (void)execute(&simd, instruction);
    cpu->fpsr |= simd.flags;
    return 1U;
}

/// The shape of an Advanced SIMD structure load or store.
typedef struct
{
    bool is_load;
    bool is_single;
    bool is_replicate;
    bool has_writeback;
    /// Registers per structure, and how often the structures repeat (LD1 with several
    /// registers).
    uint32_t selem;
    uint32_t repeat;
    uint32_t esize;
    uint32_t datasize;
    /// The lane of a single-structure access.
    uint32_t index;
    uint32_t pad;
} structure_t;

/// Decodes the load/store vector structure classes. Returns false for unallocated encodings.
static bool
decode_structure(const uint32_t instruction, structure_t *structure)
{
    const uint32_t q    = (instruction >> 30U) & 1U;
    const uint32_t size = (instruction >> 10U) & 0x3U;
    (void)memset(structure, 0, sizeof(*structure));
    structure->is_load  = 1U == ((instruction >> 22U) & 1U);
    structure->datasize = 1U == q ? 128U : 64U;

    const bool is_multiple         = 0x0C000000U == (instruction & 0xBFBF0000U);
    const bool is_multiple_indexed = 0x0C800000U == (instruction & 0xBFA00000U);
    const bool is_single           = 0x0D000000U == (instruction & 0xBF9F0000U);
    const bool is_single_indexed   = 0x0D800000U == (instruction & 0xBF800000U);
    structure->has_writeback       = is_multiple_indexed || is_single_indexed;

    if (true == is_multiple || true == is_multiple_indexed)
    {
        static const uint8_t REPEAT[16] = { 1, 0, 4, 0, 1, 0, 3, 1, 1, 0, 2, 0, 0, 0, 0, 0 };
        static const uint8_t SELEM[16]  = { 4, 0, 1, 0, 3, 0, 1, 1, 2, 0, 1, 0, 0, 0, 0, 0 };
        const uint32_t       opcode     = (instruction >> 12U) & 0xFU;
        structure->repeat               = REPEAT[opcode];
        structure->selem                = SELEM[opcode];
        structure->esize                = 8U << size;

        // Structures of several 64-bit elements need a whole Q register.
        return structure->repeat != 0U && !(3U == size && 0U == q && structure->selem != 1U);
    }

    if (false == is_single && false == is_single_indexed)
    {
        return false;
    }

    const uint32_t opcode = (instruction >> 13U) & 0x7U;
    const uint32_t r      = (instruction >> 21U) & 1U;
    const uint32_t s      = (instruction >> 12U) & 1U;
    uint32_t       scale  = opcode >> 1U;
    structure->is_single  = true;
    structure->repeat     = 1U;
    structure->selem      = (((opcode & 1U) << 1U) | r) + 1U;

    switch (scale)
    {
        case 3U: // LD1R-LD4R
            if (false == structure->is_load || 1U == s)
            {
                return false;
            }

            scale                   = size;
            structure->is_replicate = true;
            break;
        case 0U:
            structure->index = (q << 3U) | (s << 2U) | size;
            break;
        case 1U:
            if (1U == (size & 1U))
            {
                return false;
            }

            structure->index = (q << 2U) | (s << 1U) | (size >> 1U);
            break;
        default:
            if (1U == (size >> 1U))
            {
                return false;
            }

            if (0U == size)
            {
                structure->index = (q << 1U) | s;
            }
            else
            {
                if (1U == s)
                {
                    return false;
                }

                structure->index = q;
                scale            = 3U;
            }

            break;
    }

    structure->esize = 8U << scale;
    return true;
}

bool
bal_is_simd_memory_supported(const uint32_t instruction)
{
    structure_t structure;
    return decode_structure(instruction, &structure);
}

/// Returns the host address of `bytes` bytes at `address`, or NULL if they are not all mapped.
static uint8_t *
translate_element(const bal_memory_interface_t *memory_interface,
                  const uint64_t                address,
                  const uint32_t                bytes)
{
    size_t         available_bytes = 0U;
    const uint8_t *host_address
        = memory_interface->translate((void *)memory_interface, address, &available_bytes);

    if (NULL == host_address || available_bytes < bytes)
    {
        return NULL;
    }

    return (uint8_t *)(uintptr_t)host_address;
}

uint64_t
bal_execute_simd_memory(bal_cpu_t                    *cpu,
                        const bal_memory_interface_t *memory_interface,
                        const uint32_t                instruction)
{
    structure_t structure;
    (void)decode_structure(instruction, &structure);

    simd_t         simd    = { .cpu = cpu, .flags = 0U };
    const uint32_t rt      = instruction & 0x1FU;
    const uint32_t rn      = (instruction >> 5U) & 0x1FU;
    const uint32_t rm      = (instruction >> 16U) & 0x1FU;
    const uint32_t ebytes  = structure.esize / 8U;
    const uint64_t address = cpu->x[rn]; // Register 31 is the stack pointer.
    uint64_t       offset  = 0U;

    if (true == structure.is_replicate)
    {
        uint32_t t = rt;

        for (uint32_t s = 0U; s < structure.selem; ++s)
        {
            const uint8_t *pointer = translate_element(memory_interface, address + offset, ebytes);

            if (NULL == pointer)
            {
                return 0U;
            }

            uint64_t element = 0U;
            (void)memcpy(&element, pointer, ebytes);
            vector_t result = { { 0U, 0U } };

            for (uint32_t e = 0U; e < structure.datasize / structure.esize; ++e)
            {
                set_element(&result, e, structure.esize, element);
            }

            write_vector(&simd, t, result, structure.datasize);
            offset += ebytes;
            t = (t + 1U) % 32U;
        }
    }
    else
    {
        // A single structure works on one lane of full registers; multiple structures on every
        // lane of datasize-bit registers.
        const uint32_t elements
            = true == structure.is_single ? 1U : structure.datasize / structure.esize;
        const uint32_t width = true == structure.is_single ? 128U : structure.datasize;

        for (uint32_t r = 0U; r < structure.repeat; ++r)
        {
            for (uint32_t e = 0U; e < elements; ++e)
            {
                uint32_t       t    = (rt + r) % 32U;
                const uint32_t lane = true == structure.is_single ? structure.index : e;

                for (uint32_t s = 0U; s < structure.selem; ++s)
                {
                    uint8_t *pointer
                        = translate_element(memory_interface, address + offset, ebytes);

                    if (NULL == pointer)
                    {
                        return 0U;
                    }

                    vector_t value = read_vector(&simd, t);

                    if (true == structure.is_load)
                    {
                        uint64_t element = 0U;
                        (void)memcpy(&element, pointer, ebytes);
                        set_element(&value, lane, structure.esize, element);
                        write_vector(&simd, t, value, width);
                    }
                    else
                    {
                        const uint64_t element = get_element(&value, lane, structure.esize);
                        (void)memcpy(pointer, &element, ebytes);
                    }

                    offset += ebytes;
                    t = (t + 1U) % 32U;
                }
            }
        }
    }

    if (true == structure.has_writeback)
    {
        cpu->x[rn] = address + (31U == rm ? offset : cpu->x[rm]);
    }

    return 1U;
}

/*** end of file ***/
