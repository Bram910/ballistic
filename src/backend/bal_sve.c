#include "backend/bal_sve.h"
#include "backend/bal_sve_shared.h"

#include "backend/bal_float.h"
#include "backend/bal_simd.h"
#include "bal_attributes.h"
#include <string.h>

// -----------------------------------------------------------------------------
// Elements and predicates
// -----------------------------------------------------------------------------

/// DecodePredCount(): the number of elements a predicate constraint `pattern` selects.
static uint32_t
pattern_count(const uint32_t pattern, const uint32_t esize)
{
    const uint32_t elements = elements_of(esize);

    if (pattern >= 1U && pattern <= 8U)
    {
        return pattern <= elements ? pattern : 0U;
    }

    switch (pattern)
    {
        case 0U: // POW2
        {
            uint32_t power = 1U;

            while (power * 2U <= elements)
            {
                power *= 2U;
            }

            return power;
        }
        case 9U:
        case 10U:
        case 11U:
        case 12U:
        case 13U: // VL16 to VL256
        {
            const uint32_t count = 16U << (pattern - 9U);
            return count <= elements ? count : 0U;
        }
        case 29U: // MUL4
            return elements - elements % 4U;
        case 30U: // MUL3
            return elements - elements % 3U;
        case 31U: // ALL
            return elements;
        default:
            return 0U;
    }
}

// -----------------------------------------------------------------------------
// Integer arithmetic
// -----------------------------------------------------------------------------

/// Saturates a signed value to `esize` bits.
static uint64_t
saturate_signed(const int64_t value, const uint32_t esize)
{
    const int64_t maximum = (int64_t)(mask_of(esize) >> 1U);
    const int64_t minimum = -maximum - 1;
    return (uint64_t)(value > maximum ? maximum : (value < minimum ? minimum : value))
           & mask_of(esize);
}

/// SQADD and SQSUB on one element of `esize` bits.
static uint64_t
signed_saturating_add(const uint64_t a, const uint64_t b, const uint32_t esize, const bool is_sub)
{
    if (esize < 64U)
    {
        const int64_t x = sign_extend(a, esize);
        const int64_t y = sign_extend(b, esize);
        return saturate_signed(true == is_sub ? x - y : x + y, esize);
    }

    // On 64 bits the sum overflows when its sign differs from both operands' (with the second
    // negated for a subtraction).
    const uint64_t result   = true == is_sub ? a - b : a + b;
    const uint64_t operand  = true == is_sub ? ~b : b;
    const bool     overflow = 0U != (((a ^ result) & (operand ^ result)) >> 63U);
    return false == overflow ? result
                             : (0U != (a >> 63U) ? 0x8000000000000000ULL : 0x7FFFFFFFFFFFFFFFULL);
}

/// ADD, SUB, SQADD, UQADD, SQSUB and UQSUB (vectors, unpredicated): 00000100 size 1 Zm 000 opc
/// Zn Zd.
static sve_result_t
integer_add_sub_vectors(sve_t *sve, const uint32_t instruction)
{
    const uint32_t opc   = (instruction >> 10U) & 0x7U;
    const uint32_t esize = 8U << ((instruction >> 22U) & 0x3U);

    if (2U == opc || 3U == opc)
    {
        return SVE_UNDEFINED;
    }

    const vector_t n      = read_z(sve, (instruction >> 5U) & 0x1FU);
    const vector_t m      = read_z(sve, (instruction >> 16U) & 0x1FU);
    vector_t       result = { { 0U, 0U } };

    for (uint32_t e = 0U; e < elements_of(esize); ++e)
    {
        const uint64_t a     = get_element(&n, e, esize);
        const uint64_t b     = get_element(&m, e, esize);
        uint64_t       value = 0U;

        switch (opc)
        {
            case 0U:
                value = a + b;
                break;
            case 1U:
                value = a - b;
                break;
            case 4U:
                value = signed_saturating_add(a, b, esize, false);
                break;
            case 5U:
                value = a + b;
                value = (value & mask_of(esize)) < a || (64U == esize && value < a) ? mask_of(esize)
                                                                                    : value;
                break;
            case 6U:
                value = signed_saturating_add(a, b, esize, true);
                break;
            default:
                value = a > b ? a - b : 0U;
                break;
        }

        set_element(&result, e, esize, value);
    }

    write_z(sve, instruction & 0x1FU, result);
    return SVE_OK;
}

// -----------------------------------------------------------------------------
// Shared integer helpers
// -----------------------------------------------------------------------------

static uint64_t
saturate_unsigned_add(const uint64_t a, const uint64_t b, const uint32_t esize, const bool is_sub)
{
    if (true == is_sub)
    {
        return a > b ? a - b : 0U;
    }

    const uint64_t sum = a + b;
    return (sum < a || sum > mask_of(esize)) ? mask_of(esize) : sum;
}

static uint64_t
absolute_difference(const uint64_t a, const uint64_t b, const uint32_t esize, const bool is_signed)
{
    if (true == is_signed)
    {
        const int64_t x = sign_extend(a, esize);
        const int64_t y = sign_extend(b, esize);
        return (x > y ? (uint64_t)x - (uint64_t)y : (uint64_t)y - (uint64_t)x) & mask_of(esize);
    }

    return a > b ? a - b : b - a;
}

static uint64_t
multiply_high(const uint64_t a, const uint64_t b, const uint32_t esize, const bool is_signed)
{
    if (true == is_signed)
    {
        const __int128 product = (__int128)sign_extend(a, esize) * (__int128)sign_extend(b, esize);
        return (uint64_t)(product >> esize) & mask_of(esize);
    }

    const unsigned __int128 product = (unsigned __int128)a * (unsigned __int128)b;
    return (uint64_t)(product >> esize) & mask_of(esize);
}

static uint64_t
divide(const uint64_t a, const uint64_t b, const uint32_t esize, const bool is_signed)
{
    if (0U == b)
    {
        return 0U;
    }

    if (false == is_signed)
    {
        return a / b;
    }

    const int64_t x = sign_extend(a, esize);
    const int64_t y = sign_extend(b, esize);

    // The most negative value divided by -1 overflows back to itself.
    if (-1 == y)
    {
        return (0U - (uint64_t)x) & mask_of(esize);
    }

    return (uint64_t)(x / y) & mask_of(esize);
}

static bool
is_greater(const uint64_t a, const uint64_t b, const uint32_t esize, const bool is_signed)
{
    return true == is_signed ? sign_extend(a, esize) > sign_extend(b, esize) : a > b;
}

/// Shifts by an amount that may reach or pass the element size.
static uint64_t
shift_element(const uint64_t value,
              const uint64_t amount,
              const uint32_t esize,
              const uint32_t kind)
{
    switch (kind)
    {
        case 0U: // ASR
        {
            const int64_t  signed_value = sign_extend(value, esize);
            const uint64_t shift        = amount >= esize ? esize - 1U : amount;
            return (uint64_t)(signed_value >> shift) & mask_of(esize);
        }
        case 1U: // LSR
            return amount >= esize ? 0U : value >> amount;
        default: // LSL
            return amount >= esize ? 0U : (value << amount) & mask_of(esize);
    }
}

static uint64_t
count_leading_zeros(const uint64_t value, const uint32_t esize)
{
    uint64_t count = 0U;

    for (uint32_t bit = esize; bit > 0U && 0U == ((value >> (bit - 1U)) & 1U); --bit)
    {
        ++count;
    }

    return count;
}

/// DecodeBitMasks() for the 13-bit N:immr:imms immediate, replicated to 64 bits. Returns false
/// for a reserved value.
static bool
decode_bit_mask(const uint32_t imm13, uint64_t *mask)
{
    const uint32_t n    = (imm13 >> 12U) & 1U;
    const uint32_t immr = (imm13 >> 6U) & 0x3FU;
    const uint32_t imms = imm13 & 0x3FU;
    const uint32_t bits = (n << 6U) | (~imms & 0x3FU);
    uint32_t       len  = 0U;

    for (uint32_t i = 7U; i > 0U; --i)
    {
        if (0U != ((bits >> (i - 1U)) & 1U))
        {
            len = i - 1U;
            break;
        }
    }

    if (0U == ((bits >> len) & 1U) || len < 1U)
    {
        return false;
    }

    const uint32_t esize  = 1U << len;
    const uint32_t levels = esize - 1U;
    const uint32_t s      = imms & levels;
    const uint32_t r      = immr & levels;

    if (s == levels)
    {
        return false;
    }

    const uint64_t welem   = mask_of(s + 1U);
    const uint64_t rotated = esize >= 64U
                                 ? (welem >> r) | (welem << ((64U - r) & 63U))
                                 : ((welem >> r) | (welem << (esize - r))) & mask_of(esize);
    uint64_t       value   = 0U;

    for (uint32_t shift = 0U; shift < 64U; shift += esize)
    {
        value |= (64U == esize ? rotated : rotated & mask_of(esize)) << shift;
    }

    *mask = value;
    return true;
}

/// VFPExpandImm() of the 8-bit immediate to `esize` bits (16, 32 or 64).
static uint64_t
expand_float_immediate(const uint32_t imm8, const uint32_t esize)
{
    const uint32_t e    = 16U == esize ? 5U : (32U == esize ? 8U : 11U);
    const uint32_t f    = esize - e - 1U;
    const uint64_t sign = (uint64_t)(imm8 >> 7U) << (esize - 1U);
    const uint64_t b6   = (imm8 >> 6U) & 1U;
    const uint64_t exp
        = ((b6 ^ 1U) << (e - 1U)) | ((0U - b6) & mask_of(e - 3U)) << 2U | ((imm8 >> 4U) & 0x3U);
    const uint64_t fraction = (uint64_t)(imm8 & 0xFU) << (f - 4U);
    return sign | (exp << f) | fraction;
}

// -----------------------------------------------------------------------------
// Integer arithmetic, predicated
// -----------------------------------------------------------------------------

/// The predicated binary group: 00000100 size 0 opc(5) 000 Pg Zm Zdn.
static sve_result_t
integer_binary_predicated(sve_t *sve, const uint32_t instruction)
{
    const uint32_t opc   = (instruction >> 16U) & 0x1FU;
    const uint32_t esize = size_at_22(instruction);

    switch (opc)
    {
        case 0x00U:
        case 0x01U:
        case 0x03U: // ADD, SUB, SUBR
        case 0x08U:
        case 0x09U:
        case 0x0AU:
        case 0x0BU: // SMAX, UMAX, SMIN, UMIN
        case 0x0CU:
        case 0x0DU: // SABD, UABD
        case 0x10U:
        case 0x12U:
        case 0x13U: // MUL, SMULH, UMULH
        case 0x18U:
        case 0x19U:
        case 0x1AU:
        case 0x1BU: // ORR, EOR, AND, BIC
            break;
        case 0x14U:
        case 0x15U:
        case 0x16U:
        case 0x17U: // SDIV, UDIV, SDIVR, UDIVR
            if (esize < 32U)
            {
                return SVE_UNDEFINED;
            }
            break;
        default:
            return SVE_UNDEFINED;
    }

    const uint32_t rdn = instruction & 0x1FU;
    const vector_t m   = read_z(sve, (instruction >> 5U) & 0x1FU);
    const uint16_t pg  = read_p(sve, (instruction >> 10U) & 0x7U);
    vector_t       dn  = read_z(sve, rdn);

    for (uint32_t e = 0U; e < elements_of(esize); ++e)
    {
        if (false == is_active(pg, e, esize))
        {
            continue;
        }

        const uint64_t a     = get_element(&dn, e, esize);
        const uint64_t b     = get_element(&m, e, esize);
        uint64_t       value = 0U;

        switch (opc)
        {
            case 0x00U:
                value = a + b;
                break;
            case 0x01U:
                value = a - b;
                break;
            case 0x03U:
                value = b - a;
                break;
            case 0x08U:
                value = true == is_greater(a, b, esize, true) ? a : b;
                break;
            case 0x09U:
                value = a > b ? a : b;
                break;
            case 0x0AU:
                value = true == is_greater(a, b, esize, true) ? b : a;
                break;
            case 0x0BU:
                value = a > b ? b : a;
                break;
            case 0x0CU:
                value = absolute_difference(a, b, esize, true);
                break;
            case 0x0DU:
                value = absolute_difference(a, b, esize, false);
                break;
            case 0x10U:
                value = a * b;
                break;
            case 0x12U:
                value = multiply_high(a, b, esize, true);
                break;
            case 0x13U:
                value = multiply_high(a, b, esize, false);
                break;
            case 0x14U:
                value = divide(a, b, esize, true);
                break;
            case 0x15U:
                value = divide(a, b, esize, false);
                break;
            case 0x16U:
                value = divide(b, a, esize, true);
                break;
            case 0x17U:
                value = divide(b, a, esize, false);
                break;
            case 0x18U:
                value = a | b;
                break;
            case 0x19U:
                value = a ^ b;
                break;
            case 0x1AU:
                value = a & b;
                break;
            default:
                value = a & ~b;
                break;
        }

        set_element(&dn, e, esize, value);
    }

    write_z(sve, rdn, dn);
    return SVE_OK;
}

/// The predicated reductions: 00000100 size 0 opc(5) 001 Pg Zn Vd. MOVPRFX (predicated) is
/// opc 1000M in the same group.
static sve_result_t
integer_reduction(sve_t *sve, const uint32_t instruction)
{
    const uint32_t opc   = (instruction >> 16U) & 0x1FU;
    const uint32_t esize = size_at_22(instruction);
    const vector_t n     = read_z(sve, (instruction >> 5U) & 0x1FU);
    const uint16_t pg    = read_p(sve, (instruction >> 10U) & 0x7U);
    const uint32_t rd    = instruction & 0x1FU;

    if (0x10U == (opc & 0x1EU))
    {
        // MOVPRFX (predicated): active elements of Zn, the others zeroed or kept (M = opc<0>).
        vector_t result = 0U != (opc & 1U) ? read_z(sve, rd) : (vector_t) { { 0U, 0U } };

        for (uint32_t e = 0U; e < elements_of(esize); ++e)
        {
            if (true == is_active(pg, e, esize))
            {
                set_element(&result, e, esize, get_element(&n, e, esize));
            }
        }

        write_z(sve, rd, result);
        return SVE_OK;
    }

    uint64_t accumulator = 0U;

    switch (opc)
    {
        case 0x00U: // SADDV
            if (64U == esize)
            {
                return SVE_UNDEFINED;
            }
            break;
        case 0x01U: // UADDV
            break;
        case 0x08U:
            accumulator = mask_of(esize) >> 1U ^ mask_of(esize);
            break; // SMAXV: INT_MIN
        case 0x09U:
            accumulator = 0U;
            break; // UMAXV
        case 0x0AU:
            accumulator = mask_of(esize) >> 1U;
            break; // SMINV: INT_MAX
        case 0x0BU:
            accumulator = mask_of(esize);
            break; // UMINV
        case 0x18U:
        case 0x19U:
            accumulator = 0U;
            break; // ORV, EORV
        case 0x1AU:
            accumulator = mask_of(esize);
            break; // ANDV
        default:
            return SVE_UNDEFINED;
    }

    for (uint32_t e = 0U; e < elements_of(esize); ++e)
    {
        if (false == is_active(pg, e, esize))
        {
            continue;
        }

        const uint64_t a = get_element(&n, e, esize);

        switch (opc)
        {
            case 0x00U:
                accumulator += (uint64_t)sign_extend(a, esize);
                break;
            case 0x01U:
                accumulator += a;
                break;
            case 0x08U:
                accumulator = true == is_greater(a, accumulator, esize, true) ? a : accumulator;
                break;
            case 0x09U:
                accumulator = a > accumulator ? a : accumulator;
                break;
            case 0x0AU:
                accumulator = true == is_greater(accumulator, a, esize, true) ? a : accumulator;
                break;
            case 0x0BU:
                accumulator = a < accumulator ? a : accumulator;
                break;
            case 0x18U:
                accumulator |= a;
                break;
            case 0x19U:
                accumulator ^= a;
                break;
            default:
                accumulator &= a;
                break;
        }
    }

    // SADDV and UADDV produce a doubleword; the others an element.
    const uint32_t bits   = opc <= 1U ? 64U : esize;
    vector_t       result = { { accumulator & mask_of(bits), 0U } };
    write_z(sve, rd, result);
    return SVE_OK;
}

/// The element size and shift of the predicated and unpredicated shifts by immediate, from
/// tsize and imm3. Returns 0 for a reserved tsize.
static uint32_t
shift_immediate_size(const uint32_t tsize, const uint32_t imm3, const bool is_left, uint32_t *shift)
{
    if (0U == tsize)
    {
        return 0U;
    }

    uint32_t esize = 8U;

    for (uint32_t bit = 3U; bit > 0U; --bit)
    {
        if (0U != ((tsize >> bit) & 1U))
        {
            esize = 8U << bit;
            break;
        }
    }

    const uint32_t value = (tsize << 3U) | imm3;
    *shift               = true == is_left ? value - esize : 2U * esize - value;
    return esize;
}

/// ASR, LSR, LSL and ASRD by immediate, predicated: 00000100 tszh 00 opc L U 100 Pg tszl imm3 Zdn.
static sve_result_t
shift_immediate_predicated(sve_t *sve, const uint32_t instruction)
{
    const uint32_t op      = (instruction >> 16U) & 0xFU; // opc:L:U
    const uint32_t tsize   = (((instruction >> 22U) & 0x3U) << 2U) | ((instruction >> 8U) & 0x3U);
    uint32_t       shift   = 0U;
    const bool     is_left = 0x3U == op;

    if (op != 0x0U && op != 0x1U && op != 0x3U && op != 0x4U)
    {
        return SVE_UNDEFINED;
    }

    const uint32_t esize = shift_immediate_size(tsize, (instruction >> 5U) & 0x7U, is_left, &shift);

    if (0U == esize)
    {
        return SVE_UNDEFINED;
    }

    const uint32_t rdn = instruction & 0x1FU;
    const uint16_t pg  = read_p(sve, (instruction >> 10U) & 0x7U);
    vector_t       dn  = read_z(sve, rdn);

    for (uint32_t e = 0U; e < elements_of(esize); ++e)
    {
        if (false == is_active(pg, e, esize))
        {
            continue;
        }

        const uint64_t a = get_element(&dn, e, esize);
        uint64_t       value;

        if (0x4U == op)
        {
            // ASRD: an arithmetic shift that rounds towards zero.
            int64_t signed_value = sign_extend(a, esize);

            if (signed_value < 0)
            {
                signed_value += (int64_t)((1ULL << shift) - 1U);
            }

            value = (uint64_t)(signed_value >> shift) & mask_of(esize);
        }
        else
        {
            value = shift_element(a, shift, esize, 0x0U == op ? 0U : (0x1U == op ? 1U : 2U));
        }

        set_element(&dn, e, esize, value);
    }

    write_z(sve, rdn, dn);
    return SVE_OK;
}

/// ASR, LSR, LSL and the reversed ASRR, LSRR and LSLR by vector, and the wide forms, predicated:
/// 00000100 size 01 W R L U 100 Pg Zm Zdn.
static sve_result_t
shift_vector_predicated(sve_t *sve, const uint32_t instruction)
{
    const bool     is_wide     = 0U != (instruction & 0x80000U);
    const bool     is_reversed = 0U != (instruction & 0x40000U);
    const uint32_t lu          = (instruction >> 16U) & 0x3U;
    const uint32_t esize       = size_at_22(instruction);

    if (2U == lu || (true == is_wide && (true == is_reversed || 64U == esize)))
    {
        return SVE_UNDEFINED;
    }

    const uint32_t kind = 0U == lu ? 0U : (1U == lu ? 1U : 2U);
    const uint32_t rdn  = instruction & 0x1FU;
    const vector_t m    = read_z(sve, (instruction >> 5U) & 0x1FU);
    const uint16_t pg   = read_p(sve, (instruction >> 10U) & 0x7U);
    vector_t       dn   = read_z(sve, rdn);

    for (uint32_t e = 0U; e < elements_of(esize); ++e)
    {
        if (false == is_active(pg, e, esize))
        {
            continue;
        }

        const uint64_t a = get_element(&dn, e, esize);
        const uint64_t b
            = true == is_wide ? get_element(&m, (e * esize) / 64U, 64U) : get_element(&m, e, esize);
        set_element(&dn,
                    e,
                    esize,
                    true == is_reversed ? shift_element(b, a, esize, kind)
                                        : shift_element(a, b, esize, kind));
    }

    write_z(sve, rdn, dn);
    return SVE_OK;
}

/// The predicated unary group: 00000100 size 01 1 opc(4) 101 Pg Zn Zd, from SXTB (0000) to NOT
/// (1110).
static sve_result_t
integer_unary_predicated(sve_t *sve, const uint32_t instruction)
{
    const uint32_t opc   = (instruction >> 16U) & 0xFU;
    const uint32_t esize = size_at_22(instruction);

    switch (opc)
    {
        case 0x0U:
        case 0x1U: // SXTB, UXTB
            if (8U == esize)
            {
                return SVE_UNDEFINED;
            }
            break;
        case 0x2U:
        case 0x3U: // SXTH, UXTH
            if (esize < 32U)
            {
                return SVE_UNDEFINED;
            }
            break;
        case 0x4U:
        case 0x5U: // SXTW, UXTW
            if (esize < 64U)
            {
                return SVE_UNDEFINED;
            }
            break;
        case 0xCU:
        case 0xDU: // FABS, FNEG
            if (8U == esize)
            {
                return SVE_UNDEFINED;
            }
            break;
        case 0xFU:
            return SVE_UNDEFINED;
        default:
            break;
    }

    const uint32_t rd = instruction & 0x1FU;
    const vector_t n  = read_z(sve, (instruction >> 5U) & 0x1FU);
    const uint16_t pg = read_p(sve, (instruction >> 10U) & 0x7U);
    vector_t       d  = read_z(sve, rd);

    for (uint32_t e = 0U; e < elements_of(esize); ++e)
    {
        if (false == is_active(pg, e, esize))
        {
            continue;
        }

        const uint64_t a = get_element(&n, e, esize);
        uint64_t       value;

        switch (opc)
        {
            case 0x0U:
                value = (uint64_t)sign_extend(a, 8U);
                break;
            case 0x1U:
                value = a & 0xFFU;
                break;
            case 0x2U:
                value = (uint64_t)sign_extend(a, 16U);
                break;
            case 0x3U:
                value = a & 0xFFFFU;
                break;
            case 0x4U:
                value = (uint64_t)sign_extend(a, 32U);
                break;
            case 0x5U:
                value = a & 0xFFFFFFFFU;
                break;
            case 0x6U:
                value = sign_extend(a, esize) < 0 ? 0U - a : a;
                break;
            case 0x7U:
                value = 0U - a;
                break;
            case 0x8U: // CLS: the leading bits equal to the sign bit, below it.
                value = count_leading_zeros((sign_extend(a, esize) < 0 ? ~a : a) & mask_of(esize),
                                            esize)
                        - 1U;
                break;
            case 0x9U:
                value = count_leading_zeros(a, esize);
                break;
            case 0xAU:
                value = (uint64_t)__builtin_popcountll(a);
                break;
            case 0xBU:
                value = 0U == a ? 1U : 0U;
                break;
            case 0xCU:
                value = a & (mask_of(esize) >> 1U);
                break;
            case 0xDU:
                value = a ^ (1ULL << (esize - 1U));
                break;
            default:
                value = ~a;
                break;
        }

        set_element(&d, e, esize, value);
    }

    write_z(sve, rd, d);
    return SVE_OK;
}

/// MLA and MLS (00000100 size 0 Zm 01 op Pg Zn Zda) and MAD and MSB (... 11 op Pg Za Zdn).
static sve_result_t
multiply_add_predicated(sve_t *sve, const uint32_t instruction)
{
    const uint32_t esize    = size_at_22(instruction);
    const bool     is_sub   = 0U != (instruction & 0x2000U);
    const bool     is_mad   = 0U != (instruction & 0x8000U);
    const uint32_t rd       = instruction & 0x1FU;
    const uint32_t r5       = (instruction >> 5U) & 0x1FU;
    const vector_t m        = read_z(sve, (instruction >> 16U) & 0x1FU);
    const vector_t operand5 = read_z(sve, r5);
    const uint16_t pg       = read_p(sve, (instruction >> 10U) & 0x7U);
    vector_t       d        = read_z(sve, rd);

    for (uint32_t e = 0U; e < elements_of(esize); ++e)
    {
        if (false == is_active(pg, e, esize))
        {
            continue;
        }

        // MLA: Zda + Zn * Zm. MAD: Za + Zdn * Zm.
        const uint64_t addend
            = true == is_mad ? get_element(&operand5, e, esize) : get_element(&d, e, esize);
        const uint64_t factor
            = true == is_mad ? get_element(&d, e, esize) : get_element(&operand5, e, esize);
        const uint64_t product = factor * get_element(&m, e, esize);
        set_element(&d, e, esize, true == is_sub ? addend - product : addend + product);
    }

    write_z(sve, rd, d);
    return SVE_OK;
}

// -----------------------------------------------------------------------------
// Integer arithmetic, unpredicated
// -----------------------------------------------------------------------------

/// AND, ORR, EOR and BIC (vectors, unpredicated): 00000100 opc 1 Zm 001100 Zn Zd.
static sve_result_t
bitwise_unpredicated(sve_t *sve, const uint32_t instruction)
{
    const vector_t n      = read_z(sve, (instruction >> 5U) & 0x1FU);
    const vector_t m      = read_z(sve, (instruction >> 16U) & 0x1FU);
    vector_t       result = n;

    for (uint32_t i = 0U; i < 2U; ++i)
    {
        switch ((instruction >> 22U) & 0x3U)
        {
            case 0U:
                result.d[i] = n.d[i] & m.d[i];
                break;
            case 1U:
                result.d[i] = n.d[i] | m.d[i];
                break;
            case 2U:
                result.d[i] = n.d[i] ^ m.d[i];
                break;
            default:
                result.d[i] = n.d[i] & ~m.d[i];
                break;
        }
    }

    write_z(sve, instruction & 0x1FU, result);
    return SVE_OK;
}

/// ASR, LSR and LSL by immediate (00000100 tszh 1 tszl imm3 1001 opc Zn Zd) and by wide
/// elements (00000100 size 1 Zm 1000 opc Zn Zd), unpredicated.
static sve_result_t
shift_unpredicated(sve_t *sve, const uint32_t instruction)
{
    const uint32_t opc     = (instruction >> 10U) & 0x3U;
    const bool     is_wide = 0U == (instruction & 0x1000U);

    if (2U == opc)
    {
        return SVE_UNDEFINED;
    }

    const uint32_t kind = 0U == opc ? 0U : (1U == opc ? 1U : 2U);
    const vector_t n    = read_z(sve, (instruction >> 5U) & 0x1FU);
    uint32_t       esize;
    uint32_t       shift = 0U;
    vector_t       m     = { { 0U, 0U } };

    if (true == is_wide)
    {
        esize = size_at_22(instruction);
        m     = read_z(sve, (instruction >> 16U) & 0x1FU);

        if (64U == esize)
        {
            return SVE_UNDEFINED;
        }
    }
    else
    {
        const uint32_t tsize
            = (((instruction >> 22U) & 0x3U) << 2U) | ((instruction >> 19U) & 0x3U);
        esize = shift_immediate_size(tsize, (instruction >> 16U) & 0x7U, 2U == kind, &shift);

        if (0U == esize)
        {
            return SVE_UNDEFINED;
        }
    }

    vector_t result = { { 0U, 0U } };

    for (uint32_t e = 0U; e < elements_of(esize); ++e)
    {
        const uint64_t amount = true == is_wide ? get_element(&m, (e * esize) / 64U, 64U) : shift;
        set_element(
            &result, e, esize, shift_element(get_element(&n, e, esize), amount, esize, kind));
    }

    write_z(sve, instruction & 0x1FU, result);
    return SVE_OK;
}

/// MUL, PMUL, SMULH and UMULH (vectors, unpredicated): 00000100 size 1 Zm 0110 opc Zn Zd.
static sve_result_t
multiply_unpredicated(sve_t *sve, const uint32_t instruction)
{
    const uint32_t opc   = (instruction >> 10U) & 0x3U;
    const uint32_t esize = size_at_22(instruction);

    if (1U == opc && esize != 8U)
    {
        return SVE_UNDEFINED;
    }

    const vector_t n      = read_z(sve, (instruction >> 5U) & 0x1FU);
    const vector_t m      = read_z(sve, (instruction >> 16U) & 0x1FU);
    vector_t       result = { { 0U, 0U } };

    for (uint32_t e = 0U; e < elements_of(esize); ++e)
    {
        const uint64_t a     = get_element(&n, e, esize);
        const uint64_t b     = get_element(&m, e, esize);
        uint64_t       value = 0U;

        switch (opc)
        {
            case 0U:
                value = a * b;
                break;
            case 1U: // PMUL: a carry-less product of bytes.
                for (uint32_t bit = 0U; bit < 8U; ++bit)
                {
                    value ^= 0U != ((b >> bit) & 1U) ? a << bit : 0U;
                }
                break;
            case 2U:
                value = multiply_high(a, b, esize, true);
                break;
            default:
                value = multiply_high(a, b, esize, false);
                break;
        }

        set_element(&result, e, esize, value);
    }

    write_z(sve, instruction & 0x1FU, result);
    return SVE_OK;
}

/// INDEX: 00000100 size 1 field 0100 op op field Zd, with the start and the step each an imm5 or
/// a general purpose register.
static sve_result_t
index_vector(sve_t *sve, const uint32_t instruction)
{
    const uint32_t esize        = size_at_22(instruction);
    const uint32_t field5       = (instruction >> 5U) & 0x1FU;
    const uint32_t field16      = (instruction >> 16U) & 0x1FU;
    const bool     is_start_reg = 0U != (instruction & 0x400U);
    const bool     is_step_reg  = 0U != (instruction & 0x800U);
    const uint64_t start
        = true == is_start_reg ? read_x(sve, field5) : (uint64_t)sign_extend(field5, 5U);
    const uint64_t step
        = true == is_step_reg ? read_x(sve, field16) : (uint64_t)sign_extend(field16, 5U);
    vector_t result = { { 0U, 0U } };

    for (uint32_t e = 0U; e < elements_of(esize); ++e)
    {
        set_element(&result, e, esize, start + (uint64_t)e * step);
    }

    write_z(sve, instruction & 0x1FU, result);
    return SVE_OK;
}

/// ADR: 00000100 opc 1 Zm 1010 msz Zn Zd. opc 00 and 01 add sign- or zero-extended words of Zm
/// to doublewords; 1x adds packed elements of 32 (opc<0> = 0) or 64 bits.
static sve_result_t
address_vector(sve_t *sve, const uint32_t instruction)
{
    const uint32_t opc    = (instruction >> 22U) & 0x3U;
    const uint32_t shift  = (instruction >> 10U) & 0x3U;
    const uint32_t esize  = 0U == (opc & 2U) ? 64U : (0U == (opc & 1U) ? 32U : 64U);
    const vector_t n      = read_z(sve, (instruction >> 5U) & 0x1FU);
    const vector_t m      = read_z(sve, (instruction >> 16U) & 0x1FU);
    vector_t       result = { { 0U, 0U } };

    for (uint32_t e = 0U; e < elements_of(esize); ++e)
    {
        uint64_t offset = get_element(&m, e, esize);

        if (0U == opc)
        {
            offset = (uint64_t)sign_extend(offset, 32U);
        }
        else if (1U == opc)
        {
            offset &= 0xFFFFFFFFU;
        }

        set_element(&result, e, esize, get_element(&n, e, esize) + (offset << shift));
    }

    write_z(sve, instruction & 0x1FU, result);
    return SVE_OK;
}

/// MOVPRFX (unpredicated): 00000100 00100000 101111 Zn Zd.
static sve_result_t
move_prefix(sve_t *sve, const uint32_t instruction)
{
    write_z(sve, instruction & 0x1FU, read_z(sve, (instruction >> 5U) & 0x1FU));
    return SVE_OK;
}

// -----------------------------------------------------------------------------
// Duplicates, copies and immediates
// -----------------------------------------------------------------------------

static vector_t
replicate(const uint64_t value, const uint32_t esize)
{
    vector_t result = { { 0U, 0U } };

    for (uint32_t e = 0U; e < elements_of(esize); ++e)
    {
        set_element(&result, e, esize, value);
    }

    return result;
}

/// DUP (scalar): 00000101 size 100000 001110 Rn Zd, with Rn = 31 meaning SP.
static sve_result_t
duplicate_scalar(sve_t *sve, const uint32_t instruction)
{
    const uint32_t esize = size_at_22(instruction);
    write_z(sve, instruction & 0x1FU, replicate(sve->cpu->x[(instruction >> 5U) & 0x1FU], esize));
    return SVE_OK;
}

/// The signed 8-bit immediate of DUP and CPY, shifted by 8 when sh (bit 13) is set. Returns false
/// for a shift on bytes.
static bool
shifted_immediate(const uint32_t instruction,
                  const uint32_t esize,
                  const bool     is_signed,
                  uint64_t      *value)
{
    const bool     is_shifted = 0U != (instruction & 0x2000U);
    const uint32_t imm8       = (instruction >> 5U) & 0xFFU;

    if (true == is_shifted && 8U == esize)
    {
        return false;
    }

    const uint64_t base = true == is_signed ? (uint64_t)sign_extend(imm8, 8U) : imm8;
    *value              = base << (true == is_shifted ? 8U : 0U);
    return true;
}

/// DUP (immediate) and FDUP: 00100101 size 111 00 0 11 sh imm8 Zd and ... 111 00 1 11 0 imm8 Zd.
static sve_result_t
duplicate_immediate(sve_t *sve, const uint32_t instruction)
{
    const uint32_t esize = size_at_22(instruction);
    uint64_t       value = 0U;

    if (0U != (instruction & 0x10000U))
    {
        if (8U == esize || 0U != (instruction & 0x2000U))
        {
            return SVE_UNDEFINED;
        }

        value = expand_float_immediate((instruction >> 5U) & 0xFFU, esize);
    }
    else if (false == shifted_immediate(instruction, esize, true, &value))
    {
        return SVE_UNDEFINED;
    }

    write_z(sve, instruction & 0x1FU, replicate(value, esize));
    return SVE_OK;
}

/// DUP (indexed): 00000101 imm2 1 tsz 001000 Zn Zd.
static sve_result_t
duplicate_indexed(sve_t *sve, const uint32_t instruction)
{
    const uint32_t tsz = (instruction >> 16U) & 0x1FU;

    if (0U == tsz)
    {
        return SVE_UNDEFINED;
    }

    const uint32_t lowest = (uint32_t)__builtin_ctz(tsz);
    const uint32_t esize  = 8U << lowest;
    const uint32_t imm    = (((instruction >> 22U) & 0x3U) << 5U) | tsz;
    const uint32_t index  = imm >> (lowest + 1U);
    const vector_t n      = read_z(sve, (instruction >> 5U) & 0x1FU);

    // A quadword element is the whole vector.
    if (128U == esize)
    {
        write_z(sve, instruction & 0x1FU, 0U == index ? n : (vector_t) { { 0U, 0U } });
        return SVE_OK;
    }

    const uint64_t value = index < elements_of(esize) ? get_element(&n, index, esize) : 0U;
    write_z(sve,
            instruction & 0x1FU,
            index < elements_of(esize) ? replicate(value, esize) : (vector_t) { { 0U, 0U } });
    return SVE_OK;
}

/// INSR (scalar, 00000101 size 100100 001110 Rm Zdn) and INSR (SIMD&FP scalar, ... 110100 ...
/// Vm Zdn): shifts Zdn up an element and inserts the value at element 0.
static sve_result_t
insert_element(sve_t *sve, const uint32_t instruction)
{
    const uint32_t esize  = size_at_22(instruction);
    const uint32_t rdn    = instruction & 0x1FU;
    const uint32_t rm     = (instruction >> 5U) & 0x1FU;
    const uint64_t value  = 0U != (instruction & 0x100000U) ? sve->cpu->v[rm][0] : read_x(sve, rm);
    const vector_t dn     = read_z(sve, rdn);
    vector_t       result = { { 0U, 0U } };

    for (uint32_t e = 1U; e < elements_of(esize); ++e)
    {
        set_element(&result, e, esize, get_element(&dn, e - 1U, esize));
    }

    set_element(&result, 0U, esize, value);
    write_z(sve, rdn, result);
    return SVE_OK;
}

/// DUPM and the bitwise immediates ORR, EOR and AND: 00000101 opc 0000 imm13 Zd(n).
static sve_result_t
bitwise_immediate(sve_t *sve, const uint32_t instruction)
{
    uint64_t mask = 0U;

    if (false == decode_bit_mask((instruction >> 5U) & 0x1FFFU, &mask))
    {
        return SVE_UNDEFINED;
    }

    const uint32_t rd     = instruction & 0x1FU;
    vector_t       result = read_z(sve, rd);

    for (uint32_t i = 0U; i < 2U; ++i)
    {
        switch ((instruction >> 22U) & 0x3U)
        {
            case 0U:
                result.d[i] |= mask;
                break;
            case 1U:
                result.d[i] ^= mask;
                break;
            case 2U:
                result.d[i] &= mask;
                break;
            default:
                result.d[i] = mask;
                break;
        }
    }

    write_z(sve, rd, result);
    return SVE_OK;
}

/// CPY (immediate) and FCPY: 00000101 size 01 Pg 0 M sh imm8 Zd and ... 01 Pg 110 imm8 Zd.
static sve_result_t
copy_immediate(sve_t *sve, const uint32_t instruction)
{
    const uint32_t esize      = size_at_22(instruction);
    const bool     is_float   = 0x6U == ((instruction >> 13U) & 0x7U);
    const bool     is_merging = true == is_float || 0U != (instruction & 0x4000U);
    uint64_t       value      = 0U;

    if (false == is_float && 0U != (instruction & 0x8000U))
    {
        return SVE_UNDEFINED;
    }

    if (true == is_float)
    {
        if (8U == esize)
        {
            return SVE_UNDEFINED;
        }

        value = expand_float_immediate((instruction >> 5U) & 0xFFU, esize);
    }
    else if (false == shifted_immediate(instruction, esize, true, &value))
    {
        return SVE_UNDEFINED;
    }

    const uint32_t rd     = instruction & 0x1FU;
    const uint16_t pg     = read_p(sve, (instruction >> 16U) & 0xFU);
    vector_t       result = true == is_merging ? read_z(sve, rd) : (vector_t) { { 0U, 0U } };

    for (uint32_t e = 0U; e < elements_of(esize); ++e)
    {
        if (true == is_active(pg, e, esize))
        {
            set_element(&result, e, esize, value);
        }
    }

    write_z(sve, rd, result);
    return SVE_OK;
}

/// CPY (scalar, 00000101 size 101000 101 Pg Rn Zd, Rn = 31 meaning SP) and CPY (SIMD&FP
/// scalar, ... 100000 100 Pg Vn Zd).
static sve_result_t
copy_scalar(sve_t *sve, const uint32_t instruction)
{
    const uint32_t esize      = size_at_22(instruction);
    const uint32_t rn         = (instruction >> 5U) & 0x1FU;
    const bool     is_general = 0U != (instruction & 0x80000U);
    const uint64_t value      = true == is_general ? sve->cpu->x[rn] : sve->cpu->v[rn][0];
    const uint32_t rd         = instruction & 0x1FU;
    const uint16_t pg         = read_p(sve, (instruction >> 10U) & 0x7U);
    vector_t       result     = read_z(sve, rd);

    for (uint32_t e = 0U; e < elements_of(esize); ++e)
    {
        if (true == is_active(pg, e, esize))
        {
            set_element(&result, e, esize, value);
        }
    }

    write_z(sve, rd, result);
    return SVE_OK;
}

/// SEL (vectors): 00000101 size 1 Zm 11 Pg Zn Zd.
static sve_result_t
select_vectors(sve_t *sve, const uint32_t instruction)
{
    const uint32_t esize  = size_at_22(instruction);
    const vector_t n      = read_z(sve, (instruction >> 5U) & 0x1FU);
    const vector_t m      = read_z(sve, (instruction >> 16U) & 0x1FU);
    const uint16_t pg     = read_p(sve, (instruction >> 10U) & 0xFU);
    vector_t       result = m;

    for (uint32_t e = 0U; e < elements_of(esize); ++e)
    {
        if (true == is_active(pg, e, esize))
        {
            set_element(&result, e, esize, get_element(&n, e, esize));
        }
    }

    write_z(sve, instruction & 0x1FU, result);
    return SVE_OK;
}

/// The unpredicated immediate arithmetic: ADD, SUB, SUBR, SQADD, UQADD, SQSUB, UQSUB (00100101
/// size 100 opc 11 sh imm8 Zdn), SMAX, UMAX, SMIN, UMIN (... 101 opc 11 0 imm8 Zdn) and MUL
/// (... 110 000 11 0 imm8 Zdn).
static sve_result_t
integer_immediate(sve_t *sve, const uint32_t instruction)
{
    const uint32_t group = (instruction >> 19U) & 0x3U;
    const uint32_t opc   = (instruction >> 16U) & 0x7U;
    const uint32_t esize = size_at_22(instruction);
    uint64_t       imm   = 0U;

    if (3U == group)
    {
        return SVE_UNDEFINED;
    }

    if (0U == group)
    {
        if (2U == opc || false == shifted_immediate(instruction, esize, false, &imm))
        {
            return SVE_UNDEFINED;
        }
    }
    else
    {
        // The min, max and multiply forms take no shift; the signed ones a signed imm8.
        const bool is_signed = (1U == group && 0U == (opc & 1U)) || 2U == group;

        if (0U != (instruction & 0x2000U) || (1U == group && opc > 3U)
            || (2U == group && opc != 0U))
        {
            return SVE_UNDEFINED;
        }

        const uint32_t imm8 = (instruction >> 5U) & 0xFFU;
        imm = (true == is_signed ? (uint64_t)sign_extend(imm8, 8U) : imm8) & mask_of(esize);
    }

    const uint32_t rdn    = instruction & 0x1FU;
    vector_t       result = read_z(sve, rdn);

    for (uint32_t e = 0U; e < elements_of(esize); ++e)
    {
        const uint64_t a = get_element(&result, e, esize);
        const uint64_t b = imm & mask_of(esize);
        uint64_t       value;

        if (0U == group)
        {
            switch (opc)
            {
                case 0U:
                    value = a + b;
                    break;
                case 1U:
                    value = a - b;
                    break;
                case 3U:
                    value = b - a;
                    break;
                // SQADD and SQSUB add an unsigned immediate to the signed element.
                case 4U:
                case 6U: {
                    const __int128 sum = (__int128)sign_extend(a, esize)
                                         + (6U == opc ? -(__int128)b : (__int128)b);
                    const __int128 maximum = (__int128)(mask_of(esize) >> 1U);
                    const __int128 clamped
                        = sum > maximum ? maximum : (sum < -maximum - 1 ? -maximum - 1 : sum);
                    value = (uint64_t)clamped;
                    break;
                }
                case 5U:
                    value = saturate_unsigned_add(a, b, esize, false);
                    break;
                default:
                    value = saturate_unsigned_add(a, b, esize, true);
                    break;
            }
        }
        else if (1U == group)
        {
            const bool is_signed = 0U == (opc & 1U);
            const bool is_max    = opc < 2U;
            const bool greater   = is_greater(a, b, esize, is_signed);
            value                = greater == is_max ? a : b;
        }
        else
        {
            value = a * b;
        }

        set_element(&result, e, esize, value);
    }

    write_z(sve, rdn, result);
    return SVE_OK;
}

// -----------------------------------------------------------------------------
// Permutes
// -----------------------------------------------------------------------------

/// The `esize / 8`-bit field of predicate element `index`.
static uint32_t
get_predicate_field(const uint16_t predicate, const uint32_t index, const uint32_t esize)
{
    const uint32_t bits = esize / 8U;
    return ((uint32_t)predicate >> (index * bits)) & ((1U << bits) - 1U);
}

static uint16_t
set_predicate_field(uint16_t       predicate,
                    const uint32_t index,
                    const uint32_t esize,
                    const uint32_t value)
{
    const uint32_t bits  = esize / 8U;
    const uint32_t shift = index * bits;
    const uint32_t mask  = ((1U << bits) - 1U) << shift;
    return (uint16_t)((predicate & ~mask) | ((value << shift) & mask));
}

/// The index of the last active element of `mask`, or -1.
static int32_t
last_active(const uint16_t mask, const uint32_t esize)
{
    int32_t last = -1;

    for (uint32_t e = 0U; e < elements_of(esize); ++e)
    {
        if (true == is_active(mask, e, esize))
        {
            last = (int32_t)e;
        }
    }

    return last;
}

/// ZIP1, ZIP2, UZP1, UZP2, TRN1 and TRN2 on elements `get` reads from the two operands, for
/// vectors (00000101 size 1 Zm 011 opc Zn Zd) and predicates (00000101 size 10 Pm 010 opc 0 Pn 0
/// Pd).
static uint32_t
permute_source(const uint32_t opc, const uint32_t e, const uint32_t elements, bool *from_second)
{
    const uint32_t half = elements / 2U;

    switch (opc)
    {
        case 0U: // ZIP1
        case 1U: // ZIP2
            *from_second = 1U == (e & 1U);
            return (0U == opc ? 0U : half) + e / 2U;
        case 2U: // UZP1
        case 3U: // UZP2
        {
            const uint32_t index = 2U * e + (opc - 2U);
            *from_second         = index >= elements;
            return index % elements;
        }
        default: // TRN1, TRN2
            *from_second = 1U == (e & 1U);
            return (e & ~1U) + (opc - 4U);
    }
}

static sve_result_t
permute_vectors(sve_t *sve, const uint32_t instruction)
{
    const uint32_t opc   = (instruction >> 10U) & 0x7U;
    const uint32_t esize = size_at_22(instruction);

    if (opc > 5U)
    {
        return SVE_UNDEFINED;
    }

    const vector_t n      = read_z(sve, (instruction >> 5U) & 0x1FU);
    const vector_t m      = read_z(sve, (instruction >> 16U) & 0x1FU);
    vector_t       result = { { 0U, 0U } };

    for (uint32_t e = 0U; e < elements_of(esize); ++e)
    {
        bool           from_second = false;
        const uint32_t source      = permute_source(opc, e, elements_of(esize), &from_second);
        set_element(&result, e, esize, get_element(true == from_second ? &m : &n, source, esize));
    }

    write_z(sve, instruction & 0x1FU, result);
    return SVE_OK;
}

static sve_result_t
permute_predicates(sve_t *sve, const uint32_t instruction)
{
    const uint32_t opc   = (instruction >> 10U) & 0x7U;
    const uint32_t esize = size_at_22(instruction);

    if (opc > 5U)
    {
        return SVE_UNDEFINED;
    }

    const uint16_t n      = read_p(sve, (instruction >> 5U) & 0xFU);
    const uint16_t m      = read_p(sve, (instruction >> 16U) & 0xFU);
    uint16_t       result = 0U;

    for (uint32_t e = 0U; e < elements_of(esize); ++e)
    {
        bool           from_second = false;
        const uint32_t source      = permute_source(opc, e, elements_of(esize), &from_second);
        result                     = set_predicate_field(
            result, e, esize, get_predicate_field(true == from_second ? m : n, source, esize));
    }

    sve->cpu->p[instruction & 0xFU] = result;
    return SVE_OK;
}

/// PUNPKLO and PUNPKHI: 00000101 00110 00 H 0100000 Pn 0 Pd.
static sve_result_t
unpack_predicate(sve_t *sve, const uint32_t instruction)
{
    const uint16_t n      = read_p(sve, (instruction >> 5U) & 0xFU);
    const uint32_t base   = 0U != (instruction & 0x10000U) ? 8U : 0U;
    uint16_t       result = 0U;

    for (uint32_t e = 0U; e < 8U; ++e)
    {
        result = set_predicate_field(result, e, 16U, (n >> (base + e)) & 1U);
    }

    sve->cpu->p[instruction & 0xFU] = result;
    return SVE_OK;
}

/// REV (predicate, 00000101 size 110100 0100000 Pn 0 Pd, and vector, ... 111000 001110 Zn Zd).
static sve_result_t
reverse_elements(sve_t *sve, const uint32_t instruction)
{
    const uint32_t esize    = size_at_22(instruction);
    const uint32_t elements = elements_of(esize);

    if (0U != (instruction & 0x4000U))
    {
        const uint16_t n      = read_p(sve, (instruction >> 5U) & 0xFU);
        uint16_t       result = 0U;

        for (uint32_t e = 0U; e < elements; ++e)
        {
            result = set_predicate_field(
                result, e, esize, get_predicate_field(n, elements - 1U - e, esize));
        }

        sve->cpu->p[instruction & 0xFU] = result;
        return SVE_OK;
    }

    const vector_t n      = read_z(sve, (instruction >> 5U) & 0x1FU);
    vector_t       result = { { 0U, 0U } };

    for (uint32_t e = 0U; e < elements; ++e)
    {
        set_element(&result, e, esize, get_element(&n, elements - 1U - e, esize));
    }

    write_z(sve, instruction & 0x1FU, result);
    return SVE_OK;
}

/// SUNPKLO, SUNPKHI, UUNPKLO and UUNPKHI: 00000101 size 1100 U H 001110 Zn Zd.
static sve_result_t
unpack_vector(sve_t *sve, const uint32_t instruction)
{
    const uint32_t esize = size_at_22(instruction);

    if (8U == esize)
    {
        return SVE_UNDEFINED;
    }

    const bool     is_unsigned = 0U != (instruction & 0x20000U);
    const uint32_t base        = 0U != (instruction & 0x10000U) ? elements_of(esize) : 0U;
    const vector_t n           = read_z(sve, (instruction >> 5U) & 0x1FU);
    vector_t       result      = { { 0U, 0U } };

    for (uint32_t e = 0U; e < elements_of(esize); ++e)
    {
        const uint64_t half = get_element(&n, base + e, esize / 2U);
        set_element(&result,
                    e,
                    esize,
                    true == is_unsigned ? half : (uint64_t)sign_extend(half, esize / 2U));
    }

    write_z(sve, instruction & 0x1FU, result);
    return SVE_OK;
}

/// REVB, REVH, REVW and RBIT (predicated): 00000101 size 1001 opc 100 Pg Zn Zd.
static sve_result_t
reverse_within_elements(sve_t *sve, const uint32_t instruction)
{
    const uint32_t opc   = (instruction >> 16U) & 0x3U;
    const uint32_t esize = size_at_22(instruction);
    // The size of the units reversed: bytes, halfwords, words or bits.
    const uint32_t unit = 3U == opc ? 1U : 8U << opc;

    if (unit != 1U && unit >= esize)
    {
        return SVE_UNDEFINED;
    }

    const uint32_t rd     = instruction & 0x1FU;
    const vector_t n      = read_z(sve, (instruction >> 5U) & 0x1FU);
    const uint16_t pg     = read_p(sve, (instruction >> 10U) & 0x7U);
    vector_t       result = read_z(sve, rd);

    for (uint32_t e = 0U; e < elements_of(esize); ++e)
    {
        if (false == is_active(pg, e, esize))
        {
            continue;
        }

        const uint64_t value    = get_element(&n, e, esize);
        uint64_t       reversed = 0U;

        for (uint32_t i = 0U; i < esize / unit; ++i)
        {
            const uint64_t part = (value >> (i * unit)) & mask_of(unit);
            reversed |= part << ((esize / unit - 1U - i) * unit);
        }

        set_element(&result, e, esize, reversed);
    }

    write_z(sve, rd, result);
    return SVE_OK;
}

/// TBL (00000101 size 1 Zm 001100 Zn Zd), the two-register TBL (... 001010 ...) and TBX (...
/// 001011 ...).
static sve_result_t
table_lookup(sve_t *sve, const uint32_t instruction)
{
    const uint32_t esize = size_at_22(instruction);
    const uint32_t mode  = (instruction >> 10U) & 0x7U; // 100 TBL, 010 two registers, 011 TBX

    if (mode < 2U || mode > 4U)
    {
        return SVE_UNDEFINED;
    }

    const uint32_t elements = elements_of(esize);
    const uint32_t rn       = (instruction >> 5U) & 0x1FU;
    const vector_t first    = read_z(sve, rn);
    const vector_t second   = read_z(sve, (rn + 1U) & 0x1FU);
    const vector_t indices  = read_z(sve, (instruction >> 16U) & 0x1FU);
    const uint32_t rd       = instruction & 0x1FU;
    vector_t       result   = 3U == mode ? read_z(sve, rd) : (vector_t) { { 0U, 0U } };
    const uint32_t table    = 2U == mode ? 2U * elements : elements;

    for (uint32_t e = 0U; e < elements; ++e)
    {
        const uint64_t index = get_element(&indices, e, esize);

        if (index < table)
        {
            const vector_t *source = index < elements ? &first : &second;
            set_element(
                &result, e, esize, get_element(source, (uint32_t)(index % elements), esize));
        }
        else if (mode != 3U)
        {
            set_element(&result, e, esize, 0U);
        }
    }

    write_z(sve, rd, result);
    return SVE_OK;
}

/// EXT (destructive, 00000101 001 imm8h Zm 000 imm8l Zn... Zdn, and constructive, 00000101 011
/// imm8h 000 imm8l Zn Zd, which takes Zn and Zn+1).
static sve_result_t
extract_vector(sve_t *sve, const uint32_t instruction)
{
    const bool     is_constructive = 0U != (instruction & 0x400000U);
    const uint32_t imm8 = (((instruction >> 16U) & 0x1FU) << 3U) | ((instruction >> 10U) & 0x7U);
    const uint32_t position = imm8 >= BAL_SVE_VECTOR_BYTES ? 0U : imm8;
    const uint32_t rd       = instruction & 0x1FU;
    const uint32_t rn       = (instruction >> 5U) & 0x1FU;
    const vector_t low      = true == is_constructive ? read_z(sve, rn) : read_z(sve, rd);
    const vector_t high
        = true == is_constructive ? read_z(sve, (rn + 1U) & 0x1FU) : read_z(sve, rn);
    vector_t result = { { 0U, 0U } };

    for (uint32_t e = 0U; e < BAL_SVE_VECTOR_BYTES; ++e)
    {
        const uint32_t source = e + position;
        set_element(&result,
                    e,
                    8U,
                    source < BAL_SVE_VECTOR_BYTES
                        ? get_element(&low, source, 8U)
                        : get_element(&high, source - BAL_SVE_VECTOR_BYTES, 8U));
    }

    write_z(sve, rd, result);
    return SVE_OK;
}

/// SPLICE (destructive, 00000101 size 101100 100 Pg Zm Zdn, and constructive, ... 101101 ...,
/// which takes Zn and Zn+1).
static sve_result_t
splice_vectors(sve_t *sve, const uint32_t instruction)
{
    const uint32_t esize           = size_at_22(instruction);
    const bool     is_constructive = 0U != (instruction & 0x10000U);
    const uint32_t rd              = instruction & 0x1FU;
    const uint32_t r5              = (instruction >> 5U) & 0x1FU;
    const uint16_t pg              = read_p(sve, (instruction >> 10U) & 0x7U);
    const vector_t first           = true == is_constructive ? read_z(sve, r5) : read_z(sve, rd);
    const vector_t second
        = true == is_constructive ? read_z(sve, (r5 + 1U) & 0x1FU) : read_z(sve, r5);
    const int32_t last   = last_active(pg, esize);
    vector_t      result = { { 0U, 0U } };
    uint32_t      x      = 0U;
    bool          active = false;

    for (int32_t e = 0; e <= last; ++e)
    {
        active = active || is_active(pg, (uint32_t)e, esize);

        if (true == active)
        {
            set_element(&result, x++, esize, get_element(&first, (uint32_t)e, esize));
        }
    }

    for (uint32_t e = 0U; x < elements_of(esize); ++e)
    {
        set_element(&result, x++, esize, get_element(&second, e, esize));
    }

    write_z(sve, rd, result);
    return SVE_OK;
}

/// COMPACT: 00000101 size 100001 100 Pg Zn Zd.
static sve_result_t
compact_vector(sve_t *sve, const uint32_t instruction)
{
    const uint32_t esize = size_at_22(instruction);

    if (esize < 32U)
    {
        return SVE_UNDEFINED;
    }

    const vector_t n      = read_z(sve, (instruction >> 5U) & 0x1FU);
    const uint16_t pg     = read_p(sve, (instruction >> 10U) & 0x7U);
    vector_t       result = { { 0U, 0U } };
    uint32_t       x      = 0U;

    for (uint32_t e = 0U; e < elements_of(esize); ++e)
    {
        if (true == is_active(pg, e, esize))
        {
            set_element(&result, x++, esize, get_element(&n, e, esize));
        }
    }

    write_z(sve, instruction & 0x1FU, result);
    return SVE_OK;
}

/// LASTA and LASTB to a general purpose register (00000101 size 10000 B 101 Pg Zn Rd) or a
/// SIMD&FP one (... 10001 B 100 Pg Zn Vd), and CLASTA and CLASTB to a vector (... 10100 B 100 Pg
/// Zm Zdn), a SIMD&FP register (... 10101 B 100 Pg Zm Vdn) or a general purpose one (... 11000 B
/// 101 Pg Zm Rdn).
static sve_result_t
last_element(sve_t *sve, const uint32_t instruction)
{
    const uint32_t esize          = size_at_22(instruction);
    const uint32_t kind           = (instruction >> 17U) & 0xFU; // bits 20:17
    const bool     is_before      = 0U != (instruction & 0x10000U);
    const bool     is_general     = 0U != (instruction & 0x2000U);
    const bool     is_conditional = kind >= 4U;
    const uint32_t rd             = instruction & 0x1FU;
    const vector_t n              = read_z(sve, (instruction >> 5U) & 0x1FU);
    const uint16_t pg             = read_p(sve, (instruction >> 10U) & 0x7U);
    const uint32_t elements       = elements_of(esize);
    int32_t        last           = last_active(pg, esize);

    // kind: 0000 LAST to general (with bit 13), 0001 LAST to SIMD&FP, 0100 CLAST vector, 0101
    // CLAST SIMD&FP, 1000 CLAST general (with bit 13).
    const bool to_vector = 4U == kind;
    const bool valid     = (0U == kind && true == is_general) || (1U == kind && false == is_general)
                       || (4U == kind && false == is_general) || (5U == kind && false == is_general)
                       || (8U == kind && true == is_general);

    if (false == valid)
    {
        return SVE_UNDEFINED;
    }

    if (true == is_conditional && last < 0)
    {
        // No active element: the destination keeps its low element, zero extended.
        if (true == to_vector)
        {
            return SVE_OK;
        }

        if (true == is_general)
        {
            write_x(sve, rd, read_x(sve, rd) & mask_of(esize));
        }
        else
        {
            vector_t result = { { sve->cpu->v[rd][0] & mask_of(esize), 0U } };
            write_z(sve, rd, result);
        }

        return SVE_OK;
    }

    if (false == is_before)
    {
        last = last + 1 >= (int32_t)elements ? 0 : last + 1;
    }
    else if (last < 0)
    {
        last = (int32_t)elements - 1;
    }

    const uint64_t value = get_element(&n, (uint32_t)last, esize);

    if (true == to_vector)
    {
        write_z(sve, rd, replicate(value, esize));
    }
    else if (true == is_general)
    {
        write_x(sve, rd, value);
    }
    else
    {
        vector_t result = { { value, 0U } };
        write_z(sve, rd, result);
    }

    return SVE_OK;
}

// -----------------------------------------------------------------------------
// Predicates and the first-fault register
// -----------------------------------------------------------------------------

/// PTRUE and PTRUES: 00100101 size 01100 S 111000 pattern 0 Pd.
static sve_result_t
predicate_true(sve_t *sve, const uint32_t instruction)
{
    const uint32_t esize  = 8U << ((instruction >> 22U) & 0x3U);
    const uint32_t count  = pattern_count((instruction >> 5U) & 0x1FU, esize);
    uint16_t       result = 0U;

    for (uint32_t e = 0U; e < elements_of(esize); ++e)
    {
        result = set_active(result, e, esize, e < count);
    }

    sve->cpu->p[instruction & 0xFU] = result;

    if (0U != (instruction & 0x10000U))
    {
        write_nzcv(sve, predicate_test(result, result, esize));
    }

    return SVE_OK;
}

/// PFALSE: 00100101 00011000 11100100 0000 Pd.
static sve_result_t
predicate_false(sve_t *sve, const uint32_t instruction)
{
    sve->cpu->p[instruction & 0xFU] = 0U;
    return SVE_OK;
}

/// RDFFR (unpredicated): 00100101 00011001 11110000 000 Pd.
static sve_result_t
read_ffr(sve_t *sve, const uint32_t instruction)
{
    sve->cpu->p[instruction & 0xFU] = sve->cpu->ffr;
    return SVE_OK;
}

/// RDFFR and RDFFRS (predicated): 00100101 0 S 011000 1111000 Pg 0 Pd.
static sve_result_t
read_ffr_predicated(sve_t *sve, const uint32_t instruction)
{
    const uint16_t mask             = sve->cpu->p[(instruction >> 5U) & 0xFU];
    const uint16_t result           = (uint16_t)(sve->cpu->ffr & mask);
    sve->cpu->p[instruction & 0xFU] = result;

    if (0U != (instruction & 0x400000U))
    {
        write_nzcv(sve, predicate_test(mask, result, 8U));
    }

    return SVE_OK;
}

/// SETFFR: 00100101 00101100 10010000 00000000.
static sve_result_t
set_ffr(sve_t *sve, const uint32_t instruction)
{
    (void)instruction;
    sve->cpu->ffr = 0xFFFFU;
    return SVE_OK;
}

/// WRFFR: 00100101 00101000 1001000 Pn 00000.
static sve_result_t
write_ffr(sve_t *sve, const uint32_t instruction)
{
    sve->cpu->ffr = sve->cpu->p[(instruction >> 5U) & 0xFU];
    return SVE_OK;
}

// -----------------------------------------------------------------------------
// Predicate operations
// -----------------------------------------------------------------------------

/// AND, BIC, EOR, SEL, ORR, ORN, NOR and NAND on predicates, with the flag-setting forms:
/// 00100101 op S 00 Pm 01 Pg o2 Pn o3 Pd.
static sve_result_t
predicate_logical(sve_t *sve, const uint32_t instruction)
{
    const uint32_t op = ((instruction >> 23U) & 1U) << 2U | ((instruction >> 9U) & 1U) << 1U
                        | ((instruction >> 4U) & 1U);
    const bool     setflags = 0U != (instruction & 0x400000U);
    const uint16_t m        = read_p(sve, (instruction >> 16U) & 0xFU);
    const uint16_t g        = read_p(sve, (instruction >> 10U) & 0xFU);
    const uint16_t n        = read_p(sve, (instruction >> 5U) & 0xFU);
    uint16_t       result   = 0U;

    switch (op)
    {
        case 0U:
            result = (uint16_t)(n & m & g);
            break; // AND
        case 1U:
            result = (uint16_t)(n & ~m & g);
            break; // BIC
        case 2U:
            result = (uint16_t)((n ^ m) & g);
            break; // EOR
        case 3U:   // SEL
            if (true == setflags)
            {
                return SVE_UNDEFINED;
            }
            result = (uint16_t)((n & g) | (m & ~g));
            break;
        case 4U:
            result = (uint16_t)((n | m) & g);
            break; // ORR
        case 5U:
            result = (uint16_t)((n | ~m) & g);
            break; // ORN
        case 6U:
            result = (uint16_t)(~(n | m) & g);
            break; // NOR
        default:
            result = (uint16_t)(~(n & m) & g);
            break; // NAND
    }

    sve->cpu->p[instruction & 0xFU] = result;

    if (true == setflags)
    {
        write_nzcv(sve, predicate_test(g, result, 8U));
    }

    return SVE_OK;
}

/// BRKA and BRKB (00100101 B S 010000 01 Pg 0 Pn M Pd) and BRKN (00100101 0 S 011000 01 Pg 0
/// Pn 0 Pdm).
static sve_result_t
predicate_break(sve_t *sve, const uint32_t instruction)
{
    const bool     setflags = 0U != (instruction & 0x400000U);
    const bool     is_next  = 0U != (instruction & 0x80000U);
    const bool     is_after = 0U == (instruction & 0x800000U);
    const bool     merging  = 0U != (instruction & 0x10U);
    const uint16_t g        = read_p(sve, (instruction >> 10U) & 0xFU);
    const uint16_t n        = read_p(sve, (instruction >> 5U) & 0xFU);
    const uint16_t d        = read_p(sve, instruction & 0xFU);
    uint16_t       result   = 0U;

    if (true == is_next)
    {
        // BRKN: keep Pdm if the last active element of Pn is true, else clear it.
        if (false == is_after || true == merging)
        {
            return SVE_UNDEFINED;
        }

        bool last = false;

        for (uint32_t e = 0U; e < 16U; ++e)
        {
            if (true == is_active(g, e, 8U))
            {
                last = is_active(n, e, 8U);
            }
        }

        result                          = true == last ? d : 0U;
        sve->cpu->p[instruction & 0xFU] = result;

        if (true == setflags)
        {
            write_nzcv(sve, predicate_test(0xFFFFU, result, 8U));
        }

        return SVE_OK;
    }

    if (true == setflags && true == merging)
    {
        return SVE_UNDEFINED;
    }

    bool brk = false;

    for (uint32_t e = 0U; e < 16U; ++e)
    {
        if (true == is_active(g, e, 8U))
        {
            const bool element = is_active(n, e, 8U);

            if (false == is_after)
            {
                brk = brk || element;
            }

            result = set_active(result, e, 8U, false == brk);

            if (true == is_after)
            {
                brk = brk || element;
            }
        }
        else if (true == merging)
        {
            result = set_active(result, e, 8U, is_active(d, e, 8U));
        }
    }

    sve->cpu->p[instruction & 0xFU] = result;

    if (true == setflags)
    {
        write_nzcv(sve, predicate_test(g, result, 8U));
    }

    return SVE_OK;
}

/// BRKPA and BRKPB: 00100101 0 S 00 Pm 11 Pg 0 Pn B Pd.
static sve_result_t
predicate_break_propagate(sve_t *sve, const uint32_t instruction)
{
    const bool     setflags  = 0U != (instruction & 0x400000U);
    const bool     is_before = 0U != (instruction & 0x10U);
    const uint16_t m         = read_p(sve, (instruction >> 16U) & 0xFU);
    const uint16_t g         = read_p(sve, (instruction >> 10U) & 0xFU);
    const uint16_t n         = read_p(sve, (instruction >> 5U) & 0xFU);
    bool           last      = false;
    uint16_t       result    = 0U;

    for (uint32_t e = 0U; e < 16U; ++e)
    {
        if (true == is_active(g, e, 8U))
        {
            last = is_active(n, e, 8U);
        }
    }

    for (uint32_t e = 0U; e < 16U; ++e)
    {
        if (true == is_active(g, e, 8U))
        {
            if (true == is_before)
            {
                last = last && false == is_active(m, e, 8U);
            }

            result = set_active(result, e, 8U, last);

            if (false == is_before)
            {
                last = last && false == is_active(m, e, 8U);
            }
        }
    }

    sve->cpu->p[instruction & 0xFU] = result;

    if (true == setflags)
    {
        write_nzcv(sve, predicate_test(g, result, 8U));
    }

    return SVE_OK;
}

/// PTEST: 00100101 01010000 11 Pg 0 Pn 00000.
static sve_result_t
predicate_test_instruction(sve_t *sve, const uint32_t instruction)
{
    write_nzcv(sve,
               predicate_test(read_p(sve, (instruction >> 10U) & 0xFU),
                              read_p(sve, (instruction >> 5U) & 0xFU),
                              8U));
    return SVE_OK;
}

/// PFIRST: 00100101 01011000 1100000 Pg 0 Pdn.
static sve_result_t
predicate_first(sve_t *sve, const uint32_t instruction)
{
    const uint16_t g      = read_p(sve, (instruction >> 5U) & 0xFU);
    uint16_t       result = read_p(sve, instruction & 0xFU);

    for (uint32_t e = 0U; e < 16U; ++e)
    {
        if (true == is_active(g, e, 8U))
        {
            result = set_active(result, e, 8U, true);
            break;
        }
    }

    sve->cpu->p[instruction & 0xFU] = result;
    write_nzcv(sve, predicate_test(g, result, 8U));
    return SVE_OK;
}

/// PNEXT: 00100101 size 011001 1100010 Pv 0 Pdn.
static sve_result_t
predicate_next(sve_t *sve, const uint32_t instruction)
{
    const uint32_t esize    = size_at_22(instruction);
    const uint16_t mask     = read_p(sve, (instruction >> 5U) & 0xFU);
    const uint16_t operand  = read_p(sve, instruction & 0xFU);
    const uint32_t elements = elements_of(esize);
    uint32_t       next     = 0U;

    for (uint32_t e = 0U; e < elements; ++e)
    {
        if (true == is_active(operand, e, esize))
        {
            next = e + 1U;
        }
    }

    while (next < elements && false == is_active(mask, next, esize))
    {
        ++next;
    }

    uint16_t result = 0U;

    if (next < elements)
    {
        result = set_active(result, next, esize, true);
    }

    sve->cpu->p[instruction & 0xFU] = result;
    write_nzcv(sve, predicate_test(mask, result, esize));
    return SVE_OK;
}

/// CNTP: 00100101 size 100000 10 Pg 0 Pn Rd.
static sve_result_t
count_predicate(sve_t *sve, const uint32_t instruction)
{
    const uint32_t esize = size_at_22(instruction);
    write_x(sve,
            instruction & 0x1FU,
            count_active(read_p(sve, (instruction >> 10U) & 0xFU),
                         read_p(sve, (instruction >> 5U) & 0xFU),
                         esize));
    return SVE_OK;
}

/// Adds `delta` (any sign) to `value`, saturating to a `bits`-bit signed or unsigned range, and
/// returns the result sign- or zero-extended to 64 bits.
static uint64_t
saturating_offset(const uint64_t value,
                  const int64_t  delta,
                  const uint32_t bits,
                  const bool     is_unsigned)
{
    const __int128 start = true == is_unsigned ? (__int128)(value & mask_of(bits))
                                               : (__int128)sign_extend(value, bits);
    __int128       sum   = start + delta;
    const __int128 high
        = true == is_unsigned ? (__int128)mask_of(bits) : (__int128)(mask_of(bits) >> 1U);
    const __int128 low = true == is_unsigned ? 0 : -high - 1;
    sum                = sum > high ? high : (sum < low ? low : sum);
    return true == is_unsigned ? (uint64_t)sum & mask_of(bits) : (uint64_t)(int64_t)sum;
}

/// INCP, DECP, SQINCP, UQINCP, SQDECP and UQDECP: 00100101 size 101 op D U 1000 opc Pm Rdn/Zdn,
/// by the number of active elements of Pm.
static sve_result_t
count_predicate_update(sve_t *sve, const uint32_t instruction)
{
    const uint32_t esize = size_at_22(instruction);
    // INCP and DECP have D in bit 16; the saturating forms D in bit 17 and U in bit 16.
    const bool is_plain = 0U != (instruction & 0x40000U);
    const bool is_decrement
        = true == is_plain ? 0U != (instruction & 0x10000U) : 0U != (instruction & 0x20000U);
    const bool     is_unsigned = false == is_plain && 0U != (instruction & 0x10000U);
    const bool     is_scalar   = 0U != (instruction & 0x800U);
    const uint32_t opc         = (instruction >> 9U) & 0x3U;
    const uint64_t count = count_active(0xFFFFU, read_p(sve, (instruction >> 5U) & 0xFU), esize);
    const int64_t  delta = true == is_decrement ? -(int64_t)count : (int64_t)count;
    const uint32_t rdn   = instruction & 0x1FU;

    if ((true == is_plain && (0U != (instruction & 0x20000U) || opc != 0U))
        || (false == is_scalar && opc != 0U)
        || (false == is_plain && true == is_scalar && 0U != (opc & 1U)))
    {
        return SVE_UNDEFINED;
    }

    if (true == is_scalar)
    {
        const uint64_t value = read_x(sve, rdn);

        if (true == is_plain)
        {
            write_x(sve, rdn, value + (uint64_t)delta);
        }
        else
        {
            // sf (bit 10) selects 64-bit saturation; otherwise the 32-bit result is extended.
            write_x(sve,
                    rdn,
                    saturating_offset(value, delta, 0U != (opc & 2U) ? 64U : 32U, is_unsigned));
        }

        return SVE_OK;
    }

    if (8U == esize)
    {
        return SVE_UNDEFINED;
    }

    vector_t result = read_z(sve, rdn);

    for (uint32_t e = 0U; e < elements_of(esize); ++e)
    {
        const uint64_t a = get_element(&result, e, esize);
        set_element(&result,
                    e,
                    esize,
                    true == is_plain ? a + (uint64_t)delta
                                     : saturating_offset(a, delta, esize, is_unsigned));
    }

    write_z(sve, rdn, result);
    return SVE_OK;
}

/// INC and DEC (scalar, 00000100 size 11 imm4 11100 D pattern Rdn, and vector, ... 11 imm4
/// 11000 D pattern Zdn), and the saturating SQINC, UQINC, SQDEC and UQDEC (scalar, ... 1 sf imm4
/// 1111 D U pattern Rdn, and vector, ... 10 imm4 1100 D U pattern Zdn).
static sve_result_t
count_update(sve_t *sve, const uint32_t instruction)
{
    const uint32_t esize     = size_at_22(instruction);
    const uint32_t pattern   = (instruction >> 5U) & 0x1FU;
    const uint64_t scale     = ((instruction >> 16U) & 0xFU) + 1U;
    const int64_t  amount    = (int64_t)(pattern_count(pattern, esize) * scale);
    const bool     is_scalar = 0U != (instruction & 0x2000U);
    const bool is_plain = 0x3U == ((instruction >> 20U) & 0x3U) && 0U == (instruction & 0x1000U);
    const bool is_decrement
        = true == is_plain ? 0U != (instruction & 0x400U) : 0U != (instruction & 0x800U);
    const bool     is_unsigned = false == is_plain && 0U != (instruction & 0x400U);
    const int64_t  delta       = true == is_decrement ? -amount : amount;
    const uint32_t rdn         = instruction & 0x1FU;

    if (true == is_scalar)
    {
        const uint64_t value = read_x(sve, rdn);

        if (true == is_plain)
        {
            if (0U != (instruction & 0x800U))
            {
                return SVE_UNDEFINED;
            }

            write_x(sve, rdn, value + (uint64_t)delta);
        }
        else
        {
            const uint32_t bits = 0U != (instruction & 0x100000U) ? 64U : 32U;
            write_x(sve, rdn, saturating_offset(value, delta, bits, is_unsigned));
        }

        return SVE_OK;
    }

    if (8U == esize || (true == is_plain && 0U != (instruction & 0x800U)))
    {
        return SVE_UNDEFINED;
    }

    vector_t result = read_z(sve, rdn);

    for (uint32_t e = 0U; e < elements_of(esize); ++e)
    {
        const uint64_t a = get_element(&result, e, esize);
        set_element(&result,
                    e,
                    esize,
                    true == is_plain ? a + (uint64_t)delta
                                     : saturating_offset(a, delta, esize, is_unsigned));
    }

    write_z(sve, rdn, result);
    return SVE_OK;
}

/// WHILELT, WHILELE, WHILELO, WHILELS and the decrementing WHILEGE, WHILEGT, WHILEHS and WHILEHI:
/// 00100101 size 1 Rm 000 sf U lt Rn eq Pd.
static sve_result_t
predicate_while(sve_t *sve, const uint32_t instruction)
{
    const uint32_t esize       = size_at_22(instruction);
    const uint32_t rsize       = 0U != (instruction & 0x1000U) ? 64U : 32U;
    const bool     is_unsigned = 0U != (instruction & 0x800U);
    const bool     is_less     = 0U != (instruction & 0x400U);
    // Bit 4 selects LE and LS for the incrementing forms but GT and HI for the decrementing ones.
    const bool     or_equal = (0U != (instruction & 0x10U)) == is_less;
    const uint64_t operand2 = read_x(sve, (instruction >> 16U) & 0x1FU) & mask_of(rsize);
    uint64_t       operand1 = read_x(sve, (instruction >> 5U) & 0x1FU) & mask_of(rsize);
    const uint32_t elements = elements_of(esize);
    uint16_t       result   = 0U;
    bool           last     = true;

    for (uint32_t i = 0U; i < elements; ++i)
    {
        const uint32_t e = true == is_less ? i : elements - 1U - i;
        const int      order
            = true == is_unsigned
                  ? (operand1 < operand2 ? -1 : (operand1 > operand2 ? 1 : 0))
                  : (sign_extend(operand1, rsize) < sign_extend(operand2, rsize)
                         ? -1
                         : (sign_extend(operand1, rsize) > sign_extend(operand2, rsize) ? 1 : 0));
        const bool condition = true == is_less ? (order < 0 || (true == or_equal && 0 == order))
                                               : (order > 0 || (true == or_equal && 0 == order));
        last                 = last && condition;
        result               = set_active(result, e, esize, last);
        operand1             = (true == is_less ? operand1 + 1U : operand1 - 1U) & mask_of(rsize);
    }

    sve->cpu->p[instruction & 0xFU] = result;
    write_nzcv(sve, predicate_test(0xFFFFU, result, esize));
    return SVE_OK;
}

/// WHILEWR and WHILERW: 00100101 size 1 Rm 001100 Rn rw Pd.
static sve_result_t
predicate_while_conflict(sve_t *sve, const uint32_t instruction)
{
    const uint32_t esize    = size_at_22(instruction);
    const bool     is_rw    = 0U != (instruction & 0x10U);
    const uint64_t operand1 = read_x(sve, (instruction >> 5U) & 0x1FU);
    const uint64_t operand2 = read_x(sve, (instruction >> 16U) & 0x1FU);
    __int128       diff     = (__int128)operand2 - (__int128)operand1;

    // WHILEWR stops at a write after read of an earlier element: a negative difference does not
    // conflict; WHILERW counts both directions.
    if (true == is_rw && diff < 0)
    {
        diff = -diff;
    }

    const __int128 limit  = diff / (esize / 8U);
    uint16_t       result = 0U;

    for (uint32_t e = 0U; e < elements_of(esize); ++e)
    {
        const bool active = true == is_rw ? (0 == limit || e < limit) : (limit <= 0 || e < limit);
        result            = set_active(result, e, esize, active);
    }

    sve->cpu->p[instruction & 0xFU] = result;
    write_nzcv(sve, predicate_test(0xFFFFU, result, esize));
    return SVE_OK;
}

/// CTERMEQ and CTERMNE: 00100101 1 sz 1 Rm 001000 Rn ne 0000.
static sve_result_t
compare_terminate(sve_t *sve, const uint32_t instruction)
{
    const uint32_t bits = 0U != (instruction & 0x400000U) ? 64U : 32U;
    const uint64_t a    = read_x(sve, (instruction >> 5U) & 0x1FU) & mask_of(bits);
    const uint64_t b    = read_x(sve, (instruction >> 16U) & 0x1FU) & mask_of(bits);
    const bool     term = 0U != (instruction & 0x10U) ? a != b : a == b;

    if (true == term)
    {
        sve->cpu->flag_n = 1U;
        sve->cpu->flag_v = 0U;
    }
    else
    {
        sve->cpu->flag_n = 0U;
        sve->cpu->flag_v = (uint8_t)(sve->cpu->flag_c ^ 1U);
    }

    return SVE_OK;
}

// -----------------------------------------------------------------------------
// Integer compares
// -----------------------------------------------------------------------------

/// Compares `a` with `b` by `kind`: 0 EQ, 1 NE, 2 GE, 3 GT, 4 LT, 5 LE, 6 HS, 7 HI, 8 LO, 9 LS.
/// EQ and NE are signed comparisons, which matters for negative immediates and doublewords.
static bool
compare_integers(
    const uint32_t kind, const int64_t sa, const int64_t sb, const uint64_t ua, const uint64_t ub)
{
    switch (kind)
    {
        case 0U:
            return sa == sb;
        case 1U:
            return sa != sb;
        case 2U:
            return sa >= sb;
        case 3U:
            return sa > sb;
        case 4U:
            return sa < sb;
        case 5U:
            return sa <= sb;
        case 6U:
            return ua >= ub;
        case 7U:
            return ua > ub;
        case 8U:
            return ua < ub;
        default:
            return ua <= ub;
    }
}

/// Writes the compare result of every element to Pd and sets the flags.
static sve_result_t
finish_compare(sve_t          *sve,
               const uint32_t  instruction,
               const uint32_t  esize,
               const uint32_t  kind,
               const vector_t *n,
               const vector_t *m,
               const bool      is_wide,
               const bool      is_immediate,
               const uint64_t  immediate)
{
    const uint16_t g      = read_p(sve, (instruction >> 10U) & 0x7U);
    uint16_t       result = 0U;

    for (uint32_t e = 0U; e < elements_of(esize); ++e)
    {
        if (false == is_active(g, e, esize))
        {
            continue;
        }

        const uint64_t a  = get_element(n, e, esize);
        const int64_t  sa = sign_extend(a, esize);
        int64_t        sb;
        uint64_t       ub;

        if (true == is_immediate)
        {
            sb = (int64_t)immediate;
            ub = immediate;
        }
        else if (true == is_wide)
        {
            ub = get_element(m, (e * esize) / 64U, 64U);
            sb = (int64_t)ub;
        }
        else
        {
            ub = get_element(m, e, esize);
            sb = sign_extend(ub, esize);
        }

        // Unsigned comparisons of a narrow element against a doubleword need the element zero
        // extended, which `a` is.
        result = set_active(result, e, esize, compare_integers(kind, sa, sb, a, ub));
    }

    sve->cpu->p[instruction & 0xFU] = result;
    write_nzcv(sve, predicate_test(g, result, esize));
    return SVE_OK;
}

/// CMP<cc> (vectors and wide elements): 00100100 size 0 Zm op 0 o2 Pg Zn ne Pd.
static sve_result_t
compare_vectors(sve_t *sve, const uint32_t instruction)
{
    static const uint32_t KINDS[8][2] = {
        { 6U, 7U }, // 000: HS, HI
        { 0U, 1U }, // 001: EQ, NE (wide)
        { 2U, 3U }, // 010: GE, GT (wide)
        { 4U, 5U }, // 011: LT, LE (wide)
        { 2U, 3U }, // 100: GE, GT
        { 0U, 1U }, // 101: EQ, NE
        { 6U, 7U }, // 110: HS, HI (wide)
        { 8U, 9U }, // 111: LO, LS (wide)
    };
    const uint32_t op      = (instruction >> 13U) & 0x7U;
    const uint32_t esize   = size_at_22(instruction);
    const bool     is_wide = op != 0U && op != 4U && op != 5U;

    if (true == is_wide && 64U == esize)
    {
        return SVE_UNDEFINED;
    }

    const vector_t n = read_z(sve, (instruction >> 5U) & 0x1FU);
    const vector_t m = read_z(sve, (instruction >> 16U) & 0x1FU);
    return finish_compare(
        sve, instruction, esize, KINDS[op][(instruction >> 4U) & 1U], &n, &m, is_wide, false, 0U);
}

/// CMP<cc> (signed immediate, 00100101 size 0 imm5 op 0 o2 Pg Zn ne Pd) and (unsigned
/// immediate, 00100100 size 1 imm7 lt Pg Zn ne Pd).
static sve_result_t
compare_immediate(sve_t *sve, const uint32_t instruction)
{
    const uint32_t esize       = size_at_22(instruction);
    const bool     is_unsigned = 0x24000000U == (instruction & 0xFF000000U);
    const vector_t n           = read_z(sve, (instruction >> 5U) & 0x1FU);
    uint32_t       kind;
    uint64_t       immediate;

    if (true == is_unsigned)
    {
        kind      = (0U != (instruction & 0x2000U) ? 8U : 6U) + ((instruction >> 4U) & 1U);
        immediate = (instruction >> 14U) & 0x7FU;
    }
    else
    {
        static const uint32_t KINDS[8][2] = {
            { 2U, 3U }, { 4U, 5U },   { 99U, 99U }, { 99U, 99U },
            { 0U, 1U }, { 99U, 99U }, { 99U, 99U }, { 99U, 99U },
        };
        kind      = KINDS[(instruction >> 13U) & 0x7U][(instruction >> 4U) & 1U];
        immediate = (uint64_t)sign_extend((instruction >> 16U) & 0x1FU, 5U);

        if (99U == kind)
        {
            return SVE_UNDEFINED;
        }
    }

    return finish_compare(sve, instruction, esize, kind, &n, &n, false, true, immediate);
}

// -----------------------------------------------------------------------------
// Element counts
// -----------------------------------------------------------------------------

/// RDVL: 00000100 10111111 01010 imm6 Rd.
static sve_result_t
read_vector_length(sve_t *sve, const uint32_t instruction)
{
    const int64_t imm = sign_extend((instruction >> 5U) & 0x3FU, 6U);
    write_x(sve, instruction & 0x1FU, (uint64_t)(imm * (int64_t)BAL_SVE_VECTOR_BYTES));
    return SVE_OK;
}

/// ADDVL and ADDPL: 00000100 0 op 1 Rn 01010 imm6 Rd, on SP-capable registers.
static sve_result_t
add_vector_length(sve_t *sve, const uint32_t instruction)
{
    const int64_t imm = sign_extend((instruction >> 5U) & 0x3FU, 6U);
    const int64_t bytes
        = 0U != (instruction & 0x400000U) ? BAL_SVE_VECTOR_BYTES / 8 : BAL_SVE_VECTOR_BYTES;
    const uint64_t base              = sve->cpu->x[(instruction >> 16U) & 0x1FU];
    sve->cpu->x[instruction & 0x1FU] = base + (uint64_t)(imm * bytes);
    return SVE_OK;
}

/// CNTB, CNTH, CNTW and CNTD: 00000100 size 10 imm4 111000 pattern Rd.
static sve_result_t
count_elements(sve_t *sve, const uint32_t instruction)
{
    const uint32_t esize = 8U << ((instruction >> 22U) & 0x3U);
    const uint64_t scale = ((instruction >> 16U) & 0xFU) + 1U;
    const uint64_t count = pattern_count((instruction >> 5U) & 0x1FU, esize);
    write_x(sve, instruction & 0x1FU, count * scale);
    return SVE_OK;
}

// -----------------------------------------------------------------------------
// Memory accesses
// -----------------------------------------------------------------------------

/// LDR and STR of a Z register (... 010 imm9l Rn Zt) or a predicate (... 000 imm9l Rn 0 Pt):
/// 1x00010110 imm9h ... with Rn = 31 meaning SP.
static sve_result_t
load_store_register(sve_t *sve, const uint32_t instruction)
{
    const bool    is_store  = 0U != (instruction & 0x20000000U);
    const bool    is_vector = 0U != (instruction & 0x4000U);
    const int64_t imm9
        = sign_extend(((instruction >> 13U) & 0x1F8U) | ((instruction >> 10U) & 0x7U), 9U);
    const uint32_t bytes = true == is_vector ? BAL_SVE_VECTOR_BYTES : BAL_SVE_VECTOR_BYTES / 8U;
    const uint64_t address
        = sve->cpu->x[(instruction >> 5U) & 0x1FU] + (uint64_t)(imm9 * (int64_t)bytes);
    const uint32_t t = instruction & 0x1FU;

    if (false == is_vector && 0U != (instruction & 0x10U))
    {
        return SVE_UNDEFINED;
    }

    if (false == is_mapped(sve, address, bytes))
    {
        return SVE_FAULT;
    }

    if (true == is_vector)
    {
        if (true == is_store)
        {
            memory_write(sve, address, 8U, sve->cpu->v[t][0]);
            memory_write(sve, address + 8U, 8U, sve->cpu->v[t][1]);
        }
        else
        {
            vector_t value
                = { { memory_read(sve, address, 8U), memory_read(sve, address + 8U, 8U) } };
            write_z(sve, t, value);
        }
    }
    else if (true == is_store)
    {
        memory_write(sve, address, 2U, sve->cpu->p[t & 0xFU]);
    }
    else
    {
        sve->cpu->p[t & 0xFU] = (uint16_t)memory_read(sve, address, 2U);
    }

    return SVE_OK;
}

/// The memory size, element size and signedness of a contiguous load's dtype.
static void
decode_dtype(const uint32_t dtype, uint32_t *msize, uint32_t *esize, bool *is_signed)
{
    static const uint8_t TABLE[16][3] = {
        { 8U, 8U, 0U },   { 8U, 16U, 0U },  { 8U, 32U, 0U },  { 8U, 64U, 0U },
        { 32U, 64U, 1U }, { 16U, 16U, 0U }, { 16U, 32U, 0U }, { 16U, 64U, 0U },
        { 16U, 64U, 1U }, { 16U, 32U, 1U }, { 32U, 32U, 0U }, { 32U, 64U, 0U },
        { 8U, 64U, 1U },  { 8U, 32U, 1U },  { 8U, 16U, 1U },  { 64U, 64U, 0U },
    };
    *msize     = TABLE[dtype][0];
    *esize     = TABLE[dtype][1];
    *is_signed = 0U != TABLE[dtype][2];
}

/// The kinds of fault handling a contiguous load has.
typedef enum
{
    LOAD_NORMAL,
    LOAD_FIRST_FAULT,
    LOAD_NON_FAULT,
} load_kind_t;

/// Loads `count` registers from Zt of interleaved structures of `esize`-bit elements, `msize`
/// bits each in memory, from `base`. Normal loads check every active element first; first-fault
/// and non-fault loads clear FFR from the first element they cannot read.
static sve_result_t
load_contiguous(sve_t            *sve,
                const uint32_t    t,
                const uint16_t    pg,
                const uint64_t    base,
                const uint32_t    msize,
                const uint32_t    esize,
                const bool        is_signed,
                const uint32_t    count,
                const load_kind_t kind)
{
    const uint32_t elements = elements_of(esize);
    const uint32_t mbytes   = msize / 8U;
    vector_t       results[4];

    if (LOAD_NORMAL == kind)
    {
        for (uint32_t e = 0U; e < elements; ++e)
        {
            if (true == is_active(pg, e, esize)
                && false == is_mapped(sve, base + (uint64_t)e * count * mbytes, count * mbytes))
            {
                return SVE_FAULT;
            }
        }
    }

    for (uint32_t r = 0U; r < count; ++r)
    {
        results[r].d[0] = 0U;
        results[r].d[1] = 0U;
    }

    bool first = true;

    for (uint32_t e = 0U; e < elements; ++e)
    {
        if (false == is_active(pg, e, esize))
        {
            continue;
        }

        for (uint32_t r = 0U; r < count; ++r)
        {
            const uint64_t address = base + ((uint64_t)e * count + r) * mbytes;

            if (false == is_mapped(sve, address, mbytes))
            {
                // The first active element of a first-fault load faults normally.
                if (LOAD_FIRST_FAULT == kind && true == first)
                {
                    return SVE_FAULT;
                }

                // Clear FFR from this element on; the later elements of the destination are
                // zero.
                for (uint32_t f = e; f < elements; ++f)
                {
                    sve->cpu->ffr = set_active(sve->cpu->ffr, f, esize, false);
                }

                write_z(sve, t, results[0]);
                return SVE_OK;
            }

            uint64_t value = memory_read(sve, address, mbytes);

            if (true == is_signed)
            {
                value = (uint64_t)sign_extend(value, msize);
            }

            set_element(&results[r], e, esize, value);
        }

        first = false;
    }

    for (uint32_t r = 0U; r < count; ++r)
    {
        write_z(sve, (t + r) & 0x1FU, results[r]);
    }

    return SVE_OK;
}

/// Stores `count` registers from Zt as interleaved structures, checking every active element
/// first.
static sve_result_t
store_contiguous(sve_t         *sve,
                 const uint32_t t,
                 const uint16_t pg,
                 const uint64_t base,
                 const uint32_t msize,
                 const uint32_t esize,
                 const uint32_t count)
{
    const uint32_t elements = elements_of(esize);
    const uint32_t mbytes   = msize / 8U;

    for (uint32_t e = 0U; e < elements; ++e)
    {
        if (true == is_active(pg, e, esize)
            && false == is_mapped(sve, base + (uint64_t)e * count * mbytes, count * mbytes))
        {
            return SVE_FAULT;
        }
    }

    vector_t sources[4];

    for (uint32_t r = 0U; r < count; ++r)
    {
        sources[r] = read_z(sve, (t + r) & 0x1FU);
    }

    for (uint32_t e = 0U; e < elements; ++e)
    {
        if (false == is_active(pg, e, esize))
        {
            continue;
        }

        for (uint32_t r = 0U; r < count; ++r)
        {
            memory_write(sve,
                         base + ((uint64_t)e * count + r) * mbytes,
                         mbytes,
                         get_element(&sources[r], e, esize));
        }
    }

    return SVE_OK;
}

/// The contiguous loads: LD1 (scalar plus immediate, 1010010 dtype 0 imm4 101 Pg Rn Zt, and
/// scalar plus scalar, ... dtype Rm 010 ...), LDFF1 (... dtype Rm 011 ...) and LDNF1 (... dtype
/// 1 imm4 101 ...).
static sve_result_t
load_contiguous_dtype(sve_t *sve, const uint32_t instruction)
{
    const uint32_t op = (instruction >> 13U) & 0x7U;
    uint32_t       msize;
    uint32_t       esize;
    bool           is_signed;
    decode_dtype((instruction >> 21U) & 0xFU, &msize, &esize, &is_signed);

    const uint64_t base = sve->cpu->x[(instruction >> 5U) & 0x1FU];
    const uint16_t pg   = read_p(sve, (instruction >> 10U) & 0x7U);
    uint64_t       address;
    load_kind_t    kind = LOAD_NORMAL;

    if (5U == op)
    {
        // Scalar plus immediate: imm4 times the bytes of one vector's worth of elements.
        const int64_t imm4 = sign_extend((instruction >> 16U) & 0xFU, 4U);
        address            = base + (uint64_t)(imm4 * (int64_t)(elements_of(esize) * (msize / 8U)));
        kind               = 0U != (instruction & 0x100000U) ? LOAD_NON_FAULT : LOAD_NORMAL;
    }
    else
    {
        const uint32_t rm = (instruction >> 16U) & 0x1FU;

        // LD1 needs an offset register; LDFF1 may take XZR.
        if (2U == op && 31U == rm)
        {
            return SVE_UNDEFINED;
        }

        address = base + (read_x(sve, rm) << __builtin_ctz(msize / 8U));
        kind    = 3U == op ? LOAD_FIRST_FAULT : LOAD_NORMAL;
    }

    return load_contiguous(
        sve, instruction & 0x1FU, pg, address, msize, esize, is_signed, 1U, kind);
}

/// LDNT1 and LD2, LD3, LD4 (scalar plus immediate, 1010010 msz num 0 imm4 111 Pg Rn Zt, and
/// scalar plus scalar, ... msz num Rm 110 ...).
static sve_result_t
load_structures(sve_t *sve, const uint32_t instruction)
{
    const uint32_t msize = 8U << ((instruction >> 23U) & 0x3U);
    const uint32_t count = ((instruction >> 21U) & 0x3U) + 1U;
    const uint64_t base  = sve->cpu->x[(instruction >> 5U) & 0x1FU];
    const uint16_t pg    = read_p(sve, (instruction >> 10U) & 0x7U);
    uint64_t       address;

    if (7U == ((instruction >> 13U) & 0x7U))
    {
        if (0U != (instruction & 0x100000U))
        {
            return SVE_UNDEFINED;
        }

        const int64_t imm4 = sign_extend((instruction >> 16U) & 0xFU, 4U);
        address            = base + (uint64_t)(imm4 * (int64_t)(count * BAL_SVE_VECTOR_BYTES));
    }
    else
    {
        const uint32_t rm = (instruction >> 16U) & 0x1FU;

        if (31U == rm)
        {
            return SVE_UNDEFINED;
        }

        address = base + (read_x(sve, rm) << __builtin_ctz(msize / 8U));
    }

    return load_contiguous(
        sve, instruction & 0x1FU, pg, address, msize, msize, false, count, LOAD_NORMAL);
}

/// LD1RQ (scalar plus immediate, 1010010 msz 00 0 imm4 001 Pg Rn Zt, and scalar plus scalar, ...
/// msz 00 Rm 000 ...): one predicated quadword.
static sve_result_t
load_replicate_quadword(sve_t *sve, const uint32_t instruction)
{
    const uint32_t msize = 8U << ((instruction >> 23U) & 0x3U);
    const uint64_t base  = sve->cpu->x[(instruction >> 5U) & 0x1FU];
    const uint16_t pg    = read_p(sve, (instruction >> 10U) & 0x7U);
    uint64_t       address;

    if (0U != (instruction & 0x2000U))
    {
        if (0U != (instruction & 0x100000U))
        {
            return SVE_UNDEFINED;
        }

        address = base + (uint64_t)(sign_extend((instruction >> 16U) & 0xFU, 4U) * 16);
    }
    else
    {
        const uint32_t rm = (instruction >> 16U) & 0x1FU;

        if (31U == rm)
        {
            return SVE_UNDEFINED;
        }

        address = base + (read_x(sve, rm) << __builtin_ctz(msize / 8U));
    }

    return load_contiguous(
        sve, instruction & 0x1FU, pg, address, msize, msize, false, 1U, LOAD_NORMAL);
}

/// LD1R: 1000010 dtypeh 1 imm6 1 dtypel Pg Rn Zt. One element, broadcast to the active
/// elements, read only if there is one.
static sve_result_t
load_broadcast(sve_t *sve, const uint32_t instruction)
{
    uint32_t msize;
    uint32_t esize;
    bool     is_signed;
    decode_dtype((((instruction >> 23U) & 0x3U) << 2U) | ((instruction >> 13U) & 0x3U),
                 &msize,
                 &esize,
                 &is_signed);

    const uint16_t pg      = read_p(sve, (instruction >> 10U) & 0x7U);
    const uint64_t address = sve->cpu->x[(instruction >> 5U) & 0x1FU]
                             + (uint64_t)((instruction >> 16U) & 0x3FU) * (msize / 8U);
    vector_t result = { { 0U, 0U } };
    bool     any    = false;

    for (uint32_t e = 0U; e < elements_of(esize); ++e)
    {
        any = any || is_active(pg, e, esize);
    }

    if (true == any)
    {
        if (false == is_mapped(sve, address, msize / 8U))
        {
            return SVE_FAULT;
        }

        uint64_t value = memory_read(sve, address, msize / 8U);
        value          = true == is_signed ? (uint64_t)sign_extend(value, msize) : value;

        for (uint32_t e = 0U; e < elements_of(esize); ++e)
        {
            if (true == is_active(pg, e, esize))
            {
                set_element(&result, e, esize, value);
            }
        }
    }

    write_z(sve, instruction & 0x1FU, result);
    return SVE_OK;
}

/// The contiguous stores: ST1 (scalar plus immediate, 1110010 msz size 0 imm4 111 Pg Rn Zt, and
/// scalar plus scalar, ... msz size Rm 010 ...), STNT1 and ST2, ST3, ST4 (... msz num 1 imm4 111
/// ..., and ... msz num Rm 011 ...).
static sve_result_t
store_contiguous_instruction(sve_t *sve, const uint32_t instruction)
{
    const uint32_t op        = (instruction >> 13U) & 0x7U;
    const uint32_t msize     = 8U << ((instruction >> 23U) & 0x3U);
    const uint64_t base      = sve->cpu->x[(instruction >> 5U) & 0x1FU];
    const uint16_t pg        = read_p(sve, (instruction >> 10U) & 0x7U);
    const bool     is_single = 2U == op || (7U == op && 0U == (instruction & 0x100000U));
    const uint32_t esize     = true == is_single ? 8U << ((instruction >> 21U) & 0x3U) : msize;
    const uint32_t count     = true == is_single ? 1U : ((instruction >> 21U) & 0x3U) + 1U;
    uint64_t       address;

    if (esize < msize)
    {
        return SVE_UNDEFINED;
    }

    if (7U == op)
    {
        const int64_t imm4 = sign_extend((instruction >> 16U) & 0xFU, 4U);
        address = base + (uint64_t)(imm4 * (int64_t)(count * elements_of(esize) * (msize / 8U)));
    }
    else
    {
        const uint32_t rm = (instruction >> 16U) & 0x1FU;

        if (31U == rm)
        {
            return SVE_UNDEFINED;
        }

        address = base + (read_x(sve, rm) << __builtin_ctz(msize / 8U));
    }

    return store_contiguous(sve, instruction & 0x1FU, pg, address, msize, esize, count);
}

/// The ways a gather or scatter forms its addresses.
typedef enum
{
    /// Xn plus the 32-bit offsets in Zm, extended and optionally scaled.
    ADDRESS_EXTENDED,
    /// Xn plus the 64-bit offsets in Zm, optionally scaled.
    ADDRESS_OFFSETS,
    /// Zn plus an immediate times the memory size.
    ADDRESS_VECTOR_IMMEDIATE,
    /// Zn plus Xm.
    ADDRESS_VECTOR_SCALAR,
} address_kind_t;

typedef struct
{
    address_kind_t kind;
    uint32_t       esize;
    uint32_t       msize;
    bool           is_scaled;
    bool           is_signed_offset;
    uint8_t        reserved[2];
} gather_t;

/// The address of element `e` of a gather or scatter.
static uint64_t
gather_address(const sve_t    *sve,
               const gather_t *gather,
               const uint32_t  instruction,
               const uint32_t  e)
{
    const uint32_t field = (instruction >> 16U) & 0x1FU;
    const vector_t zn    = read_z(sve, (instruction >> 5U) & 0x1FU);
    const uint32_t shift
        = true == gather->is_scaled ? (uint32_t)__builtin_ctz(gather->msize / 8U) : 0U;

    switch (gather->kind)
    {
        case ADDRESS_EXTENDED: {
            const vector_t zm     = read_z(sve, field);
            const uint64_t offset = get_element(&zm, e, gather->esize) & 0xFFFFFFFFU;
            const uint64_t extended
                = true == gather->is_signed_offset ? (uint64_t)sign_extend(offset, 32U) : offset;
            return sve->cpu->x[(instruction >> 5U) & 0x1FU] + (extended << shift);
        }
        case ADDRESS_OFFSETS: {
            const vector_t zm = read_z(sve, field);
            return sve->cpu->x[(instruction >> 5U) & 0x1FU] + (get_element(&zm, e, 64U) << shift);
        }
        case ADDRESS_VECTOR_IMMEDIATE:
            return get_element(&zn, e, gather->esize) + (uint64_t)field * (gather->msize / 8U);
        default:
            return get_element(&zn, e, gather->esize) + read_x(sve, field);
    }
}

/// Decodes the addressing of a gather (1x00010 msz ...) into `gather`. Returns false if it is
/// unallocated.
static bool
decode_gather(const uint32_t instruction, gather_t *gather, bool *is_unsigned, bool *is_first_fault)
{
    const uint32_t msz       = (instruction >> 23U) & 0x3U;
    const uint32_t opc       = (instruction >> 21U) & 0x3U;
    const bool     is_wide   = 0U != (instruction & 0x40000000U);
    gather->esize            = true == is_wide ? 64U : 32U;
    gather->msize            = 8U << msz;
    gather->is_scaled        = false;
    gather->is_signed_offset = false;
    *is_unsigned             = 0U != (instruction & 0x4000U);
    *is_first_fault          = 0U != (instruction & 0x2000U);

    if (0U == (instruction & 0x8000U))
    {
        // Xn plus extended 32-bit offsets: ... msz xs scaled Zm 0 U ff ...
        gather->kind             = ADDRESS_EXTENDED;
        gather->is_scaled        = 0U != (opc & 1U);
        gather->is_signed_offset = 0U != (opc & 2U);
    }
    else if (1U == opc)
    {
        gather->kind = ADDRESS_VECTOR_IMMEDIATE;
    }
    else if (true == is_wide && opc >= 2U)
    {
        gather->kind      = ADDRESS_OFFSETS;
        gather->is_scaled = 3U == opc;
    }
    else if (0U == opc)
    {
        // LDNT1 (vector plus scalar): 32-bit ... 10 U 0, 64-bit ... 1 U 0 0.
        gather->kind    = ADDRESS_VECTOR_SCALAR;
        *is_first_fault = false;

        if (true == is_wide ? 0U != (instruction & 0x2000U) : 0U != (instruction & 0x4000U))
        {
            return false;
        }

        *is_unsigned = 0U != (instruction & (true == is_wide ? 0x4000U : 0x2000U));
    }
    else
    {
        return false;
    }

    // Scaling bytes is the prefetch encoding; a sign extending load must widen.
    if ((true == gather->is_scaled && 0U == msz) || gather->msize > gather->esize
        || (false == *is_unsigned && gather->msize == gather->esize))
    {
        return false;
    }

    return true;
}

/// The gathers: LD1, LD1S, LDFF1, LDFF1S and LDNT1 with vector addresses.
static sve_result_t
load_gather(sve_t *sve, const uint32_t instruction)
{
    gather_t gather;
    bool     is_unsigned;
    bool     is_first_fault;

    if (false == decode_gather(instruction, &gather, &is_unsigned, &is_first_fault))
    {
        return SVE_UNDEFINED;
    }

    const uint16_t pg       = read_p(sve, (instruction >> 10U) & 0x7U);
    const uint32_t mbytes   = gather.msize / 8U;
    const uint32_t elements = elements_of(gather.esize);
    vector_t       result   = { { 0U, 0U } };
    bool           first    = true;

    for (uint32_t e = 0U; e < elements; ++e)
    {
        if (false == is_active(pg, e, gather.esize))
        {
            continue;
        }

        const uint64_t address = gather_address(sve, &gather, instruction, e);

        // Like qemu, a first-fault gather also gives up on a later element that crosses a page,
        // which the architecture allows.
        const bool is_crossing = ((address ^ (address + mbytes - 1U)) >> 12U) != 0U;

        if (false == is_mapped(sve, address, mbytes)
            || (true == is_first_fault && false == first && true == is_crossing))
        {
            if (false == is_first_fault || true == first)
            {
                return SVE_FAULT;
            }

            for (uint32_t f = e; f < elements; ++f)
            {
                sve->cpu->ffr = set_active(sve->cpu->ffr, f, gather.esize, false);
            }

            break;
        }

        const uint64_t value = memory_read(sve, address, mbytes);
        set_element(&result,
                    e,
                    gather.esize,
                    true == is_unsigned ? value : (uint64_t)sign_extend(value, gather.msize));
        first = false;
    }

    write_z(sve, instruction & 0x1FU, result);
    return SVE_OK;
}

/// The scatters: ST1 and STNT1 with vector addresses (1110010 msz opc ... 1 xs 0, ... 101 and
/// ... 001 ...). Every active element is checked before any is stored.
static sve_result_t
store_scatter(sve_t *sve, const uint32_t instruction)
{
    const uint32_t op  = (instruction >> 13U) & 0x7U;
    const uint32_t opc = (instruction >> 21U) & 0x3U;
    gather_t       gather;
    gather.msize            = 8U << ((instruction >> 23U) & 0x3U);
    gather.is_scaled        = false;
    gather.is_signed_offset = false;

    if (1U == op)
    {
        // STNT1 (vector plus scalar): ... msz 00 Rm 001 for 64-bit and ... msz 10 Rm 001 for
        // 32-bit.
        if (0U != (opc & 1U))
        {
            return SVE_UNDEFINED;
        }

        gather.kind  = ADDRESS_VECTOR_SCALAR;
        gather.esize = 0U != (opc & 2U) ? 32U : 64U;
    }
    else if (4U == (op & 0x5U))
    {
        // Extended offsets: bit 22 picks 32-bit elements, bit 21 scales and bit 14 signs.
        gather.kind             = ADDRESS_EXTENDED;
        gather.esize            = 0U != (opc & 2U) ? 32U : 64U;
        gather.is_scaled        = 0U != (opc & 1U);
        gather.is_signed_offset = 0U != (instruction & 0x4000U);
    }
    else if (5U == op)
    {
        gather.kind      = opc >= 2U ? ADDRESS_VECTOR_IMMEDIATE : ADDRESS_OFFSETS;
        gather.esize     = 3U == opc ? 32U : 64U;
        gather.is_scaled = 1U == opc;
    }
    else
    {
        return SVE_UNDEFINED;
    }

    if ((true == gather.is_scaled && 8U == gather.msize) || gather.msize > gather.esize)
    {
        return SVE_UNDEFINED;
    }

    const uint16_t pg       = read_p(sve, (instruction >> 10U) & 0x7U);
    const uint32_t mbytes   = gather.msize / 8U;
    const uint32_t elements = elements_of(gather.esize);
    const vector_t source   = read_z(sve, instruction & 0x1FU);
    uint64_t       addresses[4];

    for (uint32_t e = 0U; e < elements; ++e)
    {
        addresses[e] = gather_address(sve, &gather, instruction, e);

        if (true == is_active(pg, e, gather.esize) && false == is_mapped(sve, addresses[e], mbytes))
        {
            return SVE_FAULT;
        }
    }

    for (uint32_t e = 0U; e < elements; ++e)
    {
        if (true == is_active(pg, e, gather.esize))
        {
            memory_write(sve, addresses[e], mbytes, get_element(&source, e, gather.esize));
        }
    }

    return SVE_OK;
}

/// The prefetches: they do nothing here, but the scalar plus scalar form needs an offset register.
static sve_result_t
prefetch(sve_t *sve, const uint32_t instruction)
{
    (void)sve;
    const bool is_scalar_plus_scalar = 0x8400C000U == (instruction & 0xFE60E000U);
    return true == is_scalar_plus_scalar && 31U == ((instruction >> 16U) & 0x1FU) ? SVE_UNDEFINED
                                                                                  : SVE_OK;
}

// -----------------------------------------------------------------------------
// Floating point
// -----------------------------------------------------------------------------

/// 2^exponent, which must be a normal number of `esize` bits.
static uint64_t
fp_power_of_two(const int32_t exponent, const uint32_t esize)
{
    const uint32_t fraction = fp_fraction_bits(esize);
    const int32_t  bias     = (int32_t)mask_of(esize - 2U - fraction);
    return (uint64_t)(exponent + bias) << fraction;
}

/// FSCALE on one element: value * 2^scale with a single rounding.
static uint64_t
fp_scale(sve_t *sve, const uint32_t esize, const uint64_t value, int64_t scale)
{
    const uint32_t fraction = fp_fraction_bits(esize);
    const int32_t  bias     = (int32_t)mask_of(esize - 2U - fraction);
    const int32_t  minimum  = 1 - bias;

    // Multiplying by one processes NaNs and flushes denormal inputs.
    uint64_t result = fp(sve, esize, BAL_FLOAT_MULTIPLY, value, fp_power_of_two(0, esize), 0U);
    const uint64_t exponent = (result >> fraction) & mask_of(esize - 1U - fraction);

    if (0U == (result & mask_of(esize - 1U)) || exponent == mask_of(esize - 1U - fraction))
    {
        return result;
    }

    // Far enough to overflow or underflow anything; the loops below stay short.
    const int64_t limit = 4 * bias + (int64_t)fraction;
    scale               = scale > limit ? limit : (scale < -limit ? -limit : scale);

    // Scaling up is exact until it overflows, and stays overflowed.
    while (scale > 0)
    {
        const int64_t step = scale < bias ? scale : bias;
        result
            = fp(sve, esize, BAL_FLOAT_MULTIPLY, result, fp_power_of_two((int32_t)step, esize), 0U);
        scale -= step;
    }

    // Scaling down is exact while the result stays normal; the step that leaves the normal range
    // is the only one that rounds.
    while (scale < 0)
    {
        const int64_t room = (int64_t)((result >> fraction) & mask_of(esize - 1U - fraction)) - 1;

        if (room <= 0)
        {
            const int64_t step = scale < minimum ? minimum : scale;
            result             = fp(
                sve, esize, BAL_FLOAT_MULTIPLY, result, fp_power_of_two((int32_t)step, esize), 0U);
            break;
        }

        int64_t step = scale < -room ? -room : scale;
        step         = step < minimum ? minimum : step;
        result
            = fp(sve, esize, BAL_FLOAT_MULTIPLY, result, fp_power_of_two((int32_t)step, esize), 0U);
        scale -= step;
    }

    return result;
}

/// FADD, FSUB, FMUL, FTSMUL, FRECPS and FRSQRTS (unpredicated): 01100101 size 0 Zm 000 opc Zn Zd.
static sve_result_t
float_arithmetic_unpredicated(sve_t *sve, const uint32_t instruction)
{
    const uint32_t esize = fp_esize((instruction >> 22U) & 0x3U);
    const uint32_t opc   = (instruction >> 10U) & 0x7U;

    if (0U == esize || 4U == opc || 5U == opc)
    {
        return SVE_UNDEFINED;
    }

    static const bal_float_operation_t OPERATIONS[8] = {
        BAL_FLOAT_ADD,
        BAL_FLOAT_SUBTRACT,
        BAL_FLOAT_MULTIPLY,
        BAL_FLOAT_MULTIPLY,
        BAL_FLOAT_ADD,
        BAL_FLOAT_ADD,
        BAL_FLOAT_RECIPROCAL_STEP,
        BAL_FLOAT_RECIPROCAL_SQUARE_ROOT_STEP,
    };
    const vector_t n      = read_z(sve, (instruction >> 5U) & 0x1FU);
    const vector_t m      = read_z(sve, (instruction >> 16U) & 0x1FU);
    vector_t       result = { { 0U, 0U } };

    for (uint32_t e = 0U; e < elements_of(esize); ++e)
    {
        const uint64_t a = get_element(&n, e, esize);
        const uint64_t b = get_element(&m, e, esize);
        uint64_t       r = fp(sve, esize, OPERATIONS[opc], a, 3U == opc ? a : b, 0U);

        // FTSMUL: the square, with its sign from bit 0 of the other operand.
        if (3U == opc && false == fp_is_nan(r, esize))
        {
            r = (r & mask_of(esize - 1U)) | ((b & 1U) << (esize - 1U));
        }

        set_element(&result, e, esize, r);
    }

    write_z(sve, instruction & 0x1FU, result);
    return SVE_OK;
}

/// One predicated binary operation (opc of 01100101 size 00 opc 100 Pg Zm Zdn) on `a` (Zdn) and
/// `b`.
static uint64_t
float_binary(
    sve_t *sve, const uint32_t esize, const uint32_t opc, const uint64_t a, const uint64_t b)
{
    switch (opc)
    {
        case 0x0U:
            return fp(sve, esize, BAL_FLOAT_ADD, a, b, 0U);
        case 0x1U:
            return fp(sve, esize, BAL_FLOAT_SUBTRACT, a, b, 0U);
        case 0x2U:
            return fp(sve, esize, BAL_FLOAT_MULTIPLY, a, b, 0U);
        case 0x3U:
            return fp(sve, esize, BAL_FLOAT_SUBTRACT, b, a, 0U);
        case 0x4U:
            return fp(sve, esize, BAL_FLOAT_MAXIMUM_NUMBER, a, b, 0U);
        case 0x5U:
            return fp(sve, esize, BAL_FLOAT_MINIMUM_NUMBER, a, b, 0U);
        case 0x6U:
            return fp(sve, esize, BAL_FLOAT_MAXIMUM, a, b, 0U);
        case 0x7U:
            return fp(sve, esize, BAL_FLOAT_MINIMUM, a, b, 0U);
        case 0x8U:
            return fp(sve, esize, BAL_FLOAT_ABSOLUTE_DIFFERENCE, a, b, 0U);
        case 0x9U:
            return fp_scale(sve, esize, a, sign_extend(b, esize));
        case 0xAU:
            return fp(sve, esize, BAL_FLOAT_MULTIPLY_EXTENDED, a, b, 0U);
        case 0xCU:
            return fp(sve, esize, BAL_FLOAT_DIVIDE, b, a, 0U);
        default:
            return fp(sve, esize, BAL_FLOAT_DIVIDE, a, b, 0U);
    }
}

/// FADD, FSUB, FMUL, FSUBR, FMAXNM, FMINNM, FMAX, FMIN, FABD, FSCALE, FMULX, FDIVR and FDIV
/// (predicated): 01100101 size 00 opc 100 Pg Zm Zdn.
static sve_result_t
float_arithmetic_predicated(sve_t *sve, const uint32_t instruction)
{
    const uint32_t esize = fp_esize((instruction >> 22U) & 0x3U);
    const uint32_t opc   = (instruction >> 16U) & 0xFU;

    if (0U == esize || 0xBU == opc || opc >= 0xEU)
    {
        return SVE_UNDEFINED;
    }

    const uint16_t pg     = read_p(sve, (instruction >> 10U) & 0x7U);
    const vector_t m      = read_z(sve, (instruction >> 5U) & 0x1FU);
    vector_t       result = read_z(sve, instruction & 0x1FU);

    for (uint32_t e = 0U; e < elements_of(esize); ++e)
    {
        if (true == is_active(pg, e, esize))
        {
            set_element(
                &result,
                e,
                esize,
                float_binary(
                    sve, esize, opc, get_element(&result, e, esize), get_element(&m, e, esize)));
        }
    }

    write_z(sve, instruction & 0x1FU, result);
    return SVE_OK;
}

/// The FP immediates 0.5, 1.0, 2.0 and 0.0 of each size.
static uint64_t
fp_constant(const uint32_t esize, const uint32_t which)
{
    static const uint64_t CONSTANTS[3][4] = {
        { 0x3800U, 0x3C00U, 0x4000U, 0U },
        { 0x3F000000U, 0x3F800000U, 0x40000000U, 0U },
        { 0x3FE0000000000000ULL, 0x3FF0000000000000ULL, 0x4000000000000000ULL, 0U },
    };
    return CONSTANTS[16U == esize ? 0U : (32U == esize ? 1U : 2U)][which];
}

/// FADD, FSUB, FMUL, FSUBR, FMAXNM, FMINNM, FMAX and FMIN with an immediate: 01100101 size 011
/// opc 100 Pg 0000 i1 Zdn.
static sve_result_t
float_immediate(sve_t *sve, const uint32_t instruction)
{
    const uint32_t esize = fp_esize((instruction >> 22U) & 0x3U);
    const uint32_t opc   = (instruction >> 16U) & 0x7U;
    const bool     i1    = 0U != (instruction & 0x20U);

    if (0U == esize)
    {
        return SVE_UNDEFINED;
    }

    // 0.5 or 1.0, 0.5 or 2.0 for FMUL, and 0.0 or 1.0 for the maxima and minima.
    const uint32_t which
        = opc >= 4U ? (true == i1 ? 1U : 3U) : (true == i1 ? (2U == opc ? 2U : 1U) : 0U);
    const uint64_t constant = fp_constant(esize, which);
    const uint16_t pg       = read_p(sve, (instruction >> 10U) & 0x7U);
    vector_t       result   = read_z(sve, instruction & 0x1FU);

    for (uint32_t e = 0U; e < elements_of(esize); ++e)
    {
        if (true == is_active(pg, e, esize))
        {
            set_element(&result,
                        e,
                        esize,
                        float_binary(sve, esize, opc, get_element(&result, e, esize), constant));
        }
    }

    write_z(sve, instruction & 0x1FU, result);
    return SVE_OK;
}

/// FMLA, FMLS, FNMLA and FNMLS (01100101 size 1 Zm 0 opc Pg Zn Zda) and FMAD, FMSB, FNMAD and
/// FNMSB (01100101 size 1 Za 1 opc Pg Zm Zdn).
static sve_result_t
float_multiply_add(sve_t *sve, const uint32_t instruction)
{
    const uint32_t esize  = fp_esize((instruction >> 22U) & 0x3U);
    const uint32_t opc    = (instruction >> 13U) & 0x3U;
    const bool     is_mad = 0U != (instruction & 0x8000U);

    if (0U == esize)
    {
        return SVE_UNDEFINED;
    }

    const uint32_t d     = instruction & 0x1FU;
    const uint16_t pg    = read_p(sve, (instruction >> 10U) & 0x7U);
    const vector_t first = read_z(sve, true == is_mad ? d : (instruction >> 5U) & 0x1FU);
    const vector_t second
        = read_z(sve, true == is_mad ? (instruction >> 5U) & 0x1FU : (instruction >> 16U) & 0x1FU);
    const vector_t addend = read_z(sve, true == is_mad ? (instruction >> 16U) & 0x1FU : d);
    vector_t       result = read_z(sve, d);

    for (uint32_t e = 0U; e < elements_of(esize); ++e)
    {
        if (false == is_active(pg, e, esize))
        {
            continue;
        }

        uint64_t a = get_element(&first, e, esize);
        uint64_t c = get_element(&addend, e, esize);
        a          = 1U == opc || 2U == opc ? fp_negate(a, esize) : a;
        c          = opc >= 2U ? fp_negate(c, esize) : c;
        set_element(&result,
                    e,
                    esize,
                    fp(sve, esize, BAL_FLOAT_MULTIPLY_ADD, a, get_element(&second, e, esize), c));
    }

    write_z(sve, d, result);
    return SVE_OK;
}

/// One FP comparison: 0 GE, 1 GT, 2 EQ, 3 NE, 4 UO, 5 ACGE, 7 ACGT.
static bool
float_compare(
    sve_t *sve, const uint32_t esize, const uint32_t op, const uint64_t a, const uint64_t b)
{
    switch (op)
    {
        case 0U:
            return 0U != fp(sve, esize, BAL_FLOAT_COMPARE_GREATER_EQUAL, a, b, 0U);
        case 1U:
            return 0U != fp(sve, esize, BAL_FLOAT_COMPARE_GREATER, a, b, 0U);
        case 2U:
            return 0U != fp(sve, esize, BAL_FLOAT_COMPARE_EQUAL, a, b, 0U);
        case 3U:
            return 0U == fp(sve, esize, BAL_FLOAT_COMPARE_EQUAL, a, b, 0U);
        case 4U:
            // The quiet comparison raises the same exceptions as an unordered one.
            (void)fp(sve, esize, BAL_FLOAT_COMPARE_EQUAL, a, b, 0U);
            return true == fp_is_nan(a, esize) || true == fp_is_nan(b, esize);
        case 5U:
            return 0U != fp(sve, esize, BAL_FLOAT_ABSOLUTE_COMPARE_GREATER_EQUAL, a, b, 0U);
        default:
            return 0U != fp(sve, esize, BAL_FLOAT_ABSOLUTE_COMPARE_GREATER, a, b, 0U);
    }
}

/// FCMGE, FCMGT, FCMEQ, FCMNE, FCMUO, FACGE and FACGT (vectors, 01100101 size 0 Zm op 1 cmp Pg Zn
/// cmp Pd) and FCMGE, FCMGT, FCMLT, FCMLE, FCMEQ and FCMNE (with zero, 01100101 size 0100 eq lt
/// 001 Pg Zn ne Pd).
static sve_result_t
float_compare_vectors(sve_t *sve, const uint32_t instruction)
{
    const uint32_t esize     = fp_esize((instruction >> 22U) & 0x3U);
    const bool     with_zero = 0x65102000U == (instruction & 0xFF3CE000U);
    uint32_t       op;
    bool           swap = false;

    if (true == with_zero)
    {
        static const int8_t ZERO_OPS[8] = { 0, 1, 1, 0, 2, -1, 3, -1 };
        const uint32_t index = (((instruction >> 16U) & 0x3U) << 1U) | ((instruction >> 4U) & 1U);
        op                   = ZERO_OPS[index] < 0 ? 6U : (uint32_t)ZERO_OPS[index];
        swap                 = 2U == index || 3U == index;
    }
    else
    {
        op = (((instruction >> 15U) & 1U) << 2U) | (((instruction >> 13U) & 1U) << 1U)
             | ((instruction >> 4U) & 1U);
    }

    if (0U == esize || 6U == op)
    {
        return SVE_UNDEFINED;
    }

    const uint16_t pg = read_p(sve, (instruction >> 10U) & 0x7U);
    const vector_t n  = read_z(sve, (instruction >> 5U) & 0x1FU);
    const vector_t m  = read_z(sve, (instruction >> 16U) & 0x1FU);
    uint16_t       pd = 0U;

    for (uint32_t e = 0U; e < elements_of(esize); ++e)
    {
        if (true == is_active(pg, e, esize))
        {
            const uint64_t a = get_element(&n, e, esize);
            const uint64_t b = true == with_zero ? 0U : get_element(&m, e, esize);
            pd               = set_active(pd,
                            e,
                            esize,
                            true == swap ? float_compare(sve, esize, op, b, a)
                                                       : float_compare(sve, esize, op, a, b));
        }
    }

    sve->cpu->p[instruction & 0xFU] = pd;
    return SVE_OK;
}

/// The identity of a reduction: +0 for FADDV, the default NaN for FMAXNMV and FMINNMV, -infinity
/// for FMAXV and +infinity for FMINV.
static uint64_t
fp_identity(const uint32_t esize, const uint32_t opc)
{
    const uint32_t fraction = fp_fraction_bits(esize);
    const uint64_t infinity = mask_of(esize - 1U - fraction) << fraction;

    switch (opc)
    {
        case 4U:
        case 5U:
            return infinity | (1ULL << (fraction - 1U));
        case 6U:
            return infinity | (1ULL << (esize - 1U));
        case 7U:
            return infinity;
        default:
            return 0U;
    }
}

static const bal_float_operation_t REDUCTIONS[8] = {
    BAL_FLOAT_ADD,
    BAL_FLOAT_ADD,
    BAL_FLOAT_ADD,
    BAL_FLOAT_ADD,
    BAL_FLOAT_MAXIMUM_NUMBER,
    BAL_FLOAT_MINIMUM_NUMBER,
    BAL_FLOAT_MAXIMUM,
    BAL_FLOAT_MINIMUM,
};

/// The pairwise tree reduction of `count` values.
static uint64_t
fp_reduce(sve_t                      *sve,
          const uint32_t              esize,
          const bal_float_operation_t operation,
          const uint64_t             *values,
          const uint32_t              count)
{
    if (1U == count)
    {
        return values[0];
    }

    const uint64_t low  = fp_reduce(sve, esize, operation, values, count / 2U);
    const uint64_t high = fp_reduce(sve, esize, operation, values + count / 2U, count / 2U);
    return fp(sve, esize, operation, low, high, 0U);
}

/// FADDV, FMAXNMV, FMINNMV, FMAXV and FMINV: 01100101 size 000 opc 001 Pg Zn Vd.
static sve_result_t
float_reduction(sve_t *sve, const uint32_t instruction)
{
    const uint32_t esize = fp_esize((instruction >> 22U) & 0x3U);
    const uint32_t opc   = (instruction >> 16U) & 0x7U;

    if (0U == esize || (0U != opc && opc < 4U))
    {
        return SVE_UNDEFINED;
    }

    const uint16_t pg = read_p(sve, (instruction >> 10U) & 0x7U);
    const vector_t n  = read_z(sve, (instruction >> 5U) & 0x1FU);
    uint64_t       values[8];

    for (uint32_t e = 0U; e < elements_of(esize); ++e)
    {
        values[e]
            = true == is_active(pg, e, esize) ? get_element(&n, e, esize) : fp_identity(esize, opc);
    }

    const vector_t result
        = { { fp_reduce(sve, esize, REDUCTIONS[opc], values, elements_of(esize)), 0U } };
    write_z(sve, instruction & 0x1FU, result);
    return SVE_OK;
}

/// FADDA: 01100101 size 011000 001 Pg Zm Vdn, adding the active elements in order.
static sve_result_t
float_add_ordered(sve_t *sve, const uint32_t instruction)
{
    const uint32_t esize = fp_esize((instruction >> 22U) & 0x3U);

    if (0U == esize)
    {
        return SVE_UNDEFINED;
    }

    const uint16_t pg  = read_p(sve, (instruction >> 10U) & 0x7U);
    const vector_t m   = read_z(sve, (instruction >> 5U) & 0x1FU);
    uint64_t       sum = sve->cpu->v[instruction & 0x1FU][0] & mask_of(esize);

    for (uint32_t e = 0U; e < elements_of(esize); ++e)
    {
        if (true == is_active(pg, e, esize))
        {
            sum = fp(sve, esize, BAL_FLOAT_ADD, sum, get_element(&m, e, esize), 0U);
        }
    }

    const vector_t result = { { sum, 0U } };
    write_z(sve, instruction & 0x1FU, result);
    return SVE_OK;
}

/// FADDP, FMAXNMP, FMINNMP, FMAXP and FMINP: 01100100 size 010 opc 100 Pg Zm Zdn.
static sve_result_t
float_pairwise(sve_t *sve, const uint32_t instruction)
{
    const uint32_t esize = fp_esize((instruction >> 22U) & 0x3U);
    const uint32_t opc   = (instruction >> 16U) & 0x7U;

    if (0U == esize || (0U != opc && opc < 4U))
    {
        return SVE_UNDEFINED;
    }

    const uint16_t pg     = read_p(sve, (instruction >> 10U) & 0x7U);
    const vector_t n      = read_z(sve, instruction & 0x1FU);
    const vector_t m      = read_z(sve, (instruction >> 5U) & 0x1FU);
    vector_t       result = n;

    for (uint32_t e = 0U; e < elements_of(esize); ++e)
    {
        if (true == is_active(pg, e, esize))
        {
            const vector_t *source = 0U == (e & 1U) ? &n : &m;
            const uint32_t  pair   = e & ~1U;
            set_element(&result,
                        e,
                        esize,
                        fp(sve,
                           esize,
                           REDUCTIONS[opc],
                           get_element(source, pair, esize),
                           get_element(source, pair + 1U, esize),
                           0U));
        }
    }

    write_z(sve, instruction & 0x1FU, result);
    return SVE_OK;
}

/// FLOGB on one element: the unbiased exponent as a signed integer.
static uint64_t
fp_log_b(sve_t *sve, const uint32_t esize, const uint64_t value)
{
    const uint32_t fraction_bits = fp_fraction_bits(esize);
    const uint64_t exponent_mask = mask_of(esize - 1U - fraction_bits);
    const int64_t  bias          = (int64_t)(exponent_mask >> 1U);
    const uint64_t exponent      = (value >> fraction_bits) & exponent_mask;
    const uint64_t fraction      = value & mask_of(fraction_bits);
    const bool     flush = 0U != (sve->cpu->fpcr & (16U == esize ? (1ULL << 19U) : (1ULL << 24U)));

    if (0U == exponent)
    {
        if (0U != fraction)
        {
            if (false == flush)
            {
                return (uint64_t)(1 - bias
                                  - (int64_t)(fraction_bits
                                              - (63U - (uint32_t)__builtin_clzll(fraction))))
                       & mask_of(esize);
            }

            sve->flags |= FPSR_IDC;
        }
    }
    else if (exponent == exponent_mask)
    {
        if (0U == fraction)
        {
            return mask_of(esize - 1U);
        }
    }
    else
    {
        return (uint64_t)((int64_t)exponent - bias) & mask_of(esize);
    }

    // NaN or zero.
    sve->flags |= FPSR_IOC;
    return 1ULL << (esize - 1U);
}

/// FRINTN, FRINTP, FRINTM, FRINTZ, FRINTA, FRINTX and FRINTI (01100101 size 000 rmode 101 Pg Zn
/// Zd), FRECPX and FSQRT (... size 00110 op 101 ...) and FLOGB (01100101 00011 size 0 101 ...).
static sve_result_t
float_unary(sve_t *sve, const uint32_t instruction)
{
    const bool     is_log_b = 0x6518A000U == (instruction & 0xFFF9E000U);
    const uint32_t esize
        = fp_esize(true == is_log_b ? (instruction >> 17U) & 0x3U : (instruction >> 22U) & 0x3U);
    const uint32_t opc = (instruction >> 16U) & 0xFU;

    if (0U == esize || (false == is_log_b && 5U == opc))
    {
        return SVE_UNDEFINED;
    }

    const uint16_t pg         = read_p(sve, (instruction >> 10U) & 0x7U);
    const vector_t n          = read_z(sve, (instruction >> 5U) & 0x1FU);
    vector_t       result     = read_z(sve, instruction & 0x1FU);
    const uint32_t fpcr_rmode = (uint32_t)(sve->cpu->fpcr >> 22U) & 0x3U;

    for (uint32_t e = 0U; e < elements_of(esize); ++e)
    {
        if (false == is_active(pg, e, esize))
        {
            continue;
        }

        const uint64_t a = get_element(&n, e, esize);
        uint64_t       r;

        if (true == is_log_b)
        {
            r = fp_log_b(sve, esize, a);
        }
        else if (0xCU == opc)
        {
            r = fp(sve, esize, BAL_FLOAT_RECIPROCAL_EXPONENT, a, 0U, 0U);
        }
        else if (0xDU == opc)
        {
            r = fp(sve, esize, BAL_FLOAT_SQUARE_ROOT, a, 0U, 0U);
        }
        else
        {
            const uint32_t rmode = opc >= 6U ? fpcr_rmode : opc;
            r = bal_float_round_element(sve->cpu->fpcr, &sve->flags, esize, a, rmode, 6U == opc);
        }

        set_element(&result, e, esize, r);
    }

    write_z(sve, instruction & 0x1FU, result);
    return SVE_OK;
}

/// The kinds of conversion.
typedef enum
{
    CONVERT_FLOAT,
    CONVERT_ODD,
    CONVERT_BFLOAT,
    CONVERT_TO_INTEGER,
    CONVERT_FROM_INTEGER,
} convert_kind_t;

typedef struct
{
    uint32_t value;
    uint8_t  kind;
    uint8_t  from;
    uint8_t  to;
    uint8_t  container;
} convert_t;

/// The predicated conversions, by their encoding with Pg, Zn, Zd and the U bit (bit 16 of the
/// integer conversions) clear.
static const convert_t CONVERSIONS[] = {
    { 0x6588A000U, CONVERT_FLOAT, 32U, 16U, 32U },
    { 0x6589A000U, CONVERT_FLOAT, 16U, 32U, 32U },
    { 0x65C8A000U, CONVERT_FLOAT, 64U, 16U, 64U },
    { 0x65C9A000U, CONVERT_FLOAT, 16U, 64U, 64U },
    { 0x65CAA000U, CONVERT_FLOAT, 64U, 32U, 64U },
    { 0x65CBA000U, CONVERT_FLOAT, 32U, 64U, 64U },
    { 0x650AA000U, CONVERT_ODD, 64U, 32U, 64U },
    { 0x658AA000U, CONVERT_BFLOAT, 32U, 16U, 32U },
    { 0x655AA000U, CONVERT_TO_INTEGER, 16U, 16U, 16U },
    { 0x655CA000U, CONVERT_TO_INTEGER, 16U, 32U, 32U },
    { 0x655EA000U, CONVERT_TO_INTEGER, 16U, 64U, 64U },
    { 0x659CA000U, CONVERT_TO_INTEGER, 32U, 32U, 32U },
    { 0x65D8A000U, CONVERT_TO_INTEGER, 64U, 32U, 64U },
    { 0x65DCA000U, CONVERT_TO_INTEGER, 32U, 64U, 64U },
    { 0x65DEA000U, CONVERT_TO_INTEGER, 64U, 64U, 64U },
    { 0x6552A000U, CONVERT_FROM_INTEGER, 16U, 16U, 16U },
    { 0x6554A000U, CONVERT_FROM_INTEGER, 32U, 16U, 32U },
    { 0x6556A000U, CONVERT_FROM_INTEGER, 64U, 16U, 64U },
    { 0x6594A000U, CONVERT_FROM_INTEGER, 32U, 32U, 32U },
    { 0x65D0A000U, CONVERT_FROM_INTEGER, 32U, 64U, 64U },
    { 0x65D4A000U, CONVERT_FROM_INTEGER, 64U, 32U, 64U },
    { 0x65D6A000U, CONVERT_FROM_INTEGER, 64U, 64U, 64U },
};

/// One conversion of `value`.
static uint64_t
convert_element(sve_t *sve, const convert_t *convert, const bool is_unsigned, const uint64_t value)
{
    const uint64_t fpcr = sve->cpu->fpcr;
    const uint64_t a    = value & mask_of(convert->from);

    switch (convert->kind)
    {
        case CONVERT_FLOAT:
            return bal_float_convert_element(fpcr, &sve->flags, convert->from, convert->to, a);
        case CONVERT_ODD:
            return bal_float_narrow_odd_element(fpcr, &sve->flags, a);
        case CONVERT_BFLOAT:
            return bal_float_to_bfloat16_element(fpcr, &sve->flags, a);
        case CONVERT_TO_INTEGER: {
            const uint64_t r = bal_float_to_integer_element(
                fpcr, &sve->flags, convert->from, a, 0U, 3U, is_unsigned, convert->to);
            return true == is_unsigned
                       ? r & mask_of(convert->to)
                       : (uint64_t)sign_extend(r & mask_of(convert->to), convert->to);
        }
        default:
            return bal_float_from_integer_element(
                fpcr, &sve->flags, convert->to, a, 0U, is_unsigned, convert->from);
    }
}

/// FCVT, FCVTX, BFCVT, FCVTZS, FCVTZU, SCVTF and UCVTF: 01100101 opc 0 1x opc2 101 Pg Zn Zd.
static sve_result_t
float_convert(sve_t *sve, const uint32_t instruction)
{
    const uint32_t   key     = instruction & 0xFFFFE000U;
    const convert_t *convert = NULL;

    for (size_t i = 0U; i < sizeof(CONVERSIONS) / sizeof(CONVERSIONS[0]); ++i)
    {
        const bool is_integer = CONVERSIONS[i].kind >= CONVERT_TO_INTEGER;

        if (CONVERSIONS[i].value == (true == is_integer ? key & ~0x10000U : key))
        {
            convert = &CONVERSIONS[i];
        }
    }

    if (NULL == convert)
    {
        return SVE_UNDEFINED;
    }

    const bool     is_unsigned = 0U != (instruction & 0x10000U);
    const uint32_t esize       = convert->container;
    const uint16_t pg          = read_p(sve, (instruction >> 10U) & 0x7U);
    const vector_t n           = read_z(sve, (instruction >> 5U) & 0x1FU);
    vector_t       result      = read_z(sve, instruction & 0x1FU);

    for (uint32_t e = 0U; e < elements_of(esize); ++e)
    {
        if (true == is_active(pg, e, esize))
        {
            set_element(&result,
                        e,
                        esize,
                        convert_element(sve, convert, is_unsigned, get_element(&n, e, esize)));
        }
    }

    write_z(sve, instruction & 0x1FU, result);
    return SVE_OK;
}

/// FCVTNT, FCVTLT, FCVTXNT and BFCVTNT: 01100100 opc 0010 opc2 101 Pg Zn Zd, converting between the
/// odd narrow elements and the wide ones.
static sve_result_t
float_convert_top(sve_t *sve, const uint32_t instruction)
{
    static const convert_t TOP[] = {
        { 0x6488A000U, CONVERT_FLOAT, 32U, 16U, 32U },
        { 0x6489A000U, CONVERT_FLOAT, 16U, 32U, 32U },
        { 0x64CAA000U, CONVERT_FLOAT, 64U, 32U, 64U },
        { 0x64CBA000U, CONVERT_FLOAT, 32U, 64U, 64U },
        { 0x640AA000U, CONVERT_ODD, 64U, 32U, 64U },
        { 0x648AA000U, CONVERT_BFLOAT, 32U, 16U, 32U },
    };
    const convert_t *convert = NULL;

    for (size_t i = 0U; i < sizeof(TOP) / sizeof(TOP[0]); ++i)
    {
        convert = TOP[i].value == (instruction & 0xFFFFE000U) ? &TOP[i] : convert;
    }

    if (NULL == convert)
    {
        return SVE_UNDEFINED;
    }

    const uint32_t esize     = convert->container;
    const uint32_t half      = esize / 2U;
    const bool     is_narrow = convert->to < convert->from;
    const uint16_t pg        = read_p(sve, (instruction >> 10U) & 0x7U);
    const vector_t n         = read_z(sve, (instruction >> 5U) & 0x1FU);
    vector_t       result    = read_z(sve, instruction & 0x1FU);

    for (uint32_t e = 0U; e < elements_of(esize); ++e)
    {
        if (false == is_active(pg, e, esize))
        {
            continue;
        }

        if (true == is_narrow)
        {
            set_element(&result,
                        2U * e + 1U,
                        half,
                        convert_element(sve, convert, false, get_element(&n, e, esize)));
        }
        else
        {
            set_element(&result,
                        e,
                        esize,
                        convert_element(sve, convert, false, get_element(&n, 2U * e + 1U, half)));
        }
    }

    write_z(sve, instruction & 0x1FU, result);
    return SVE_OK;
}

/// FRECPE and FRSQRTE: 01100101 size 00111 op 001100 Zn Zd.
static sve_result_t
float_estimate(sve_t *sve, const uint32_t instruction)
{
    const uint32_t esize = fp_esize((instruction >> 22U) & 0x3U);

    if (0U == esize)
    {
        return SVE_UNDEFINED;
    }

    const bal_float_operation_t operation = 0U != (instruction & 0x10000U)
                                                ? BAL_FLOAT_RECIPROCAL_SQUARE_ROOT_ESTIMATE
                                                : BAL_FLOAT_RECIPROCAL_ESTIMATE;
    const vector_t              n         = read_z(sve, (instruction >> 5U) & 0x1FU);
    vector_t                    result    = { { 0U, 0U } };

    for (uint32_t e = 0U; e < elements_of(esize); ++e)
    {
        set_element(
            &result, e, esize, fp(sve, esize, operation, get_element(&n, e, esize), 0U, 0U));
    }

    write_z(sve, instruction & 0x1FU, result);
    return SVE_OK;
}

/// FEXPA: 00000100 size 100000 101110 Zn Zd. The fraction of 2^(i/N) from a table, with the
/// exponent from the next bits of the operand.
static sve_result_t
float_exponential(sve_t *sve, const uint32_t instruction)
{
    static const uint16_t HALF[32] = {
        0x0000U, 0x0016U, 0x002DU, 0x0045U, 0x005DU, 0x0075U, 0x008EU, 0x00A8U,
        0x00C2U, 0x00DCU, 0x00F8U, 0x0114U, 0x0130U, 0x014DU, 0x016BU, 0x0189U,
        0x01A8U, 0x01C8U, 0x01E8U, 0x0209U, 0x022BU, 0x024EU, 0x0271U, 0x0295U,
        0x02BAU, 0x02E0U, 0x0306U, 0x032EU, 0x0356U, 0x037FU, 0x03A9U, 0x03D4U,
    };
    static const uint32_t SINGLE[64] = {
        0x000000U, 0x0164D2U, 0x02CD87U, 0x043A29U, 0x05AAC3U, 0x071F62U, 0x08980FU, 0x0A14D5U,
        0x0B95C2U, 0x0D1ADFU, 0x0EA43AU, 0x1031DCU, 0x11C3D3U, 0x135A2BU, 0x14F4F0U, 0x16942DU,
        0x1837F0U, 0x19E046U, 0x1B8D3AU, 0x1D3EDAU, 0x1EF532U, 0x20B051U, 0x227043U, 0x243516U,
        0x25FED7U, 0x27CD94U, 0x29A15BU, 0x2B7A3AU, 0x2D583FU, 0x2F3B79U, 0x3123F6U, 0x3311C4U,
        0x3504F3U, 0x36FD92U, 0x38FBAFU, 0x3AFF5BU, 0x3D08A4U, 0x3F179AU, 0x412C4DU, 0x4346CDU,
        0x45672AU, 0x478D75U, 0x49B9BEU, 0x4BEC15U, 0x4E248CU, 0x506334U, 0x52A81EU, 0x54F35BU,
        0x5744FDU, 0x599D16U, 0x5BFBB8U, 0x5E60F5U, 0x60CCDFU, 0x633F89U, 0x65B907U, 0x68396AU,
        0x6AC0C7U, 0x6D4F30U, 0x6FE4BAU, 0x728177U, 0x75257DU, 0x77D0DFU, 0x7A83B3U, 0x7D3E0CU,
    };
    static const uint64_t DOUBLE[64] = {
        0x0000000000000ULL, 0x02C9A3E778061ULL, 0x059B0D3158574ULL, 0x0874518759BC8ULL,
        0x0B5586CF9890FULL, 0x0E3EC32D3D1A2ULL, 0x11301D0125B51ULL, 0x1429AAEA92DE0ULL,
        0x172B83C7D517BULL, 0x1A35BEB6FCB75ULL, 0x1D4873168B9AAULL, 0x2063B88628CD6ULL,
        0x2387A6E756238ULL, 0x26B4565E27CDDULL, 0x29E9DF51FDEE1ULL, 0x2D285A6E4030BULL,
        0x306FE0A31B715ULL, 0x33C08B26416FFULL, 0x371A7373AA9CBULL, 0x3A7DB34E59FF7ULL,
        0x3DEA64C123422ULL, 0x4160A21F72E2AULL, 0x44E086061892DULL, 0x486A2B5C13CD0ULL,
        0x4BFDAD5362A27ULL, 0x4F9B2769D2CA7ULL, 0x5342B569D4F82ULL, 0x56F4736B527DAULL,
        0x5AB07DD485429ULL, 0x5E76F15AD2148ULL, 0x6247EB03A5585ULL, 0x6623882552225ULL,
        0x6A09E667F3BCDULL, 0x6DFB23C651A2FULL, 0x71F75E8EC5F74ULL, 0x75FEB564267C9ULL,
        0x7A11473EB0187ULL, 0x7E2F336CF4E62ULL, 0x82589994CCE13ULL, 0x868D99B4492EDULL,
        0x8ACE5422AA0DBULL, 0x8F1AE99157736ULL, 0x93737B0CDC5E5ULL, 0x97D829FDE4E50ULL,
        0x9C49182A3F090ULL, 0xA0C667B5DE565ULL, 0xA5503B23E255DULL, 0xA9E6B5579FDBFULL,
        0xAE89F995AD3ADULL, 0xB33A2B84F15FBULL, 0xB7F76F2FB5E47ULL, 0xBCC1E904BC1D2ULL,
        0xC199BDD85529CULL, 0xC67F12E57D14BULL, 0xCB720DCEF9069ULL, 0xD072D4A07897CULL,
        0xD5818DCFBA487ULL, 0xDA9E603DB3285ULL, 0xDFC97337B9B5FULL, 0xE502EE78B3FF6ULL,
        0xEA4AFA2A490DAULL, 0xEFA1BEE615A27ULL, 0xF50765B6E4540ULL, 0xFA7C1819E90D8ULL,
    };
    const uint32_t esize = fp_esize((instruction >> 22U) & 0x3U);

    if (0U == esize)
    {
        return SVE_UNDEFINED;
    }

    const vector_t n      = read_z(sve, (instruction >> 5U) & 0x1FU);
    vector_t       result = { { 0U, 0U } };

    for (uint32_t e = 0U; e < elements_of(esize); ++e)
    {
        const uint64_t a = get_element(&n, e, esize);
        uint64_t       r;

        if (16U == esize)
        {
            r = (((a >> 5U) & 0x1FU) << 10U) | HALF[a & 0x1FU];
        }
        else if (32U == esize)
        {
            r = (((a >> 6U) & 0xFFU) << 23U) | SINGLE[a & 0x3FU];
        }
        else
        {
            r = (((a >> 6U) & 0x7FFU) << 52U) | DOUBLE[a & 0x3FU];
        }

        set_element(&result, e, esize, r);
    }

    write_z(sve, instruction & 0x1FU, result);
    return SVE_OK;
}

/// FTMAD: 01100101 size 010 imm3 100000 Zm Zdn, one step of a sine or cosine polynomial.
static sve_result_t
float_trig_multiply_add(sve_t *sve, const uint32_t instruction)
{
    static const uint64_t COEFFICIENTS[3][16] = {
        { 0x3C00U,
          0xB155U,
          0x2030U,
          0U,
          0U,
          0U,
          0U,
          0U,
          0x3C00U,
          0xB800U,
          0x293AU,
          0U,
          0U,
          0U,
          0U,
          0U },
        { 0x3F800000U,
          0xBE2AAAABU,
          0x3C088886U,
          0xB95008B9U,
          0x36369D6DU,
          0U,
          0U,
          0U,
          0x3F800000U,
          0xBF000000U,
          0x3D2AAAA6U,
          0xBAB60705U,
          0x37CD37CCU,
          0U,
          0U,
          0U },
        { 0x3FF0000000000000ULL,
          0xBFC5555555555543ULL,
          0x3F8111111110F30CULL,
          0xBF2A01A019B92FC6ULL,
          0x3EC71DE351F3D22BULL,
          0xBE5AE5E2B60F7B91ULL,
          0x3DE5D8408868552FULL,
          0x0000000000000000ULL,
          0x3FF0000000000000ULL,
          0xBFE0000000000000ULL,
          0x3FA5555555555536ULL,
          0xBF56C16C16C13A0BULL,
          0x3EFA01A019B1E8D8ULL,
          0xBE927E4F7282F468ULL,
          0x3E21EE96D2641B13ULL,
          0xBDA8F76380FBB401ULL },
    };
    const uint32_t esize = fp_esize((instruction >> 22U) & 0x3U);

    if (0U == esize)
    {
        return SVE_UNDEFINED;
    }

    const uint32_t table  = 16U == esize ? 0U : (32U == esize ? 1U : 2U);
    const uint32_t imm    = (instruction >> 16U) & 0x7U;
    const vector_t m      = read_z(sve, (instruction >> 5U) & 0x1FU);
    vector_t       result = read_z(sve, instruction & 0x1FU);

    for (uint32_t e = 0U; e < elements_of(esize); ++e)
    {
        uint64_t       b           = get_element(&m, e, esize);
        const bool     is_negative = 0U != (b >> (esize - 1U));
        const uint32_t index       = imm + (true == is_negative ? 8U : 0U);
        b                          = b & mask_of(esize - 1U);
        set_element(&result,
                    e,
                    esize,
                    fp(sve,
                       esize,
                       BAL_FLOAT_MULTIPLY_ADD,
                       get_element(&result, e, esize),
                       b,
                       COEFFICIENTS[table][index]));
    }

    write_z(sve, instruction & 0x1FU, result);
    return SVE_OK;
}

/// FTSSEL: 00000100 size 1 Zm 101100 Zn Zd. One, or the operand, with bit 1 of Zm negating it.
static sve_result_t
float_trig_select(sve_t *sve, const uint32_t instruction)
{
    const uint32_t esize = fp_esize((instruction >> 22U) & 0x3U);

    if (0U == esize)
    {
        return SVE_UNDEFINED;
    }

    const vector_t n      = read_z(sve, (instruction >> 5U) & 0x1FU);
    const vector_t m      = read_z(sve, (instruction >> 16U) & 0x1FU);
    vector_t       result = { { 0U, 0U } };

    for (uint32_t e = 0U; e < elements_of(esize); ++e)
    {
        const uint64_t b = get_element(&m, e, esize);
        uint64_t       r = 0U != (b & 1U) ? fp_constant(esize, 1U) : get_element(&n, e, esize);
        r                = 0U != (b & 2U) ? fp_negate(r, esize) : r;
        set_element(&result, e, esize, r);
    }

    write_z(sve, instruction & 0x1FU, result);
    return SVE_OK;
}

/// The complex pair products of FCMLA with rotation `rotation` added to the pairs of `result`,
/// for the active elements; `index` picks one pair of `m` for all pairs, or is negative.
static void
complex_multiply_add(sve_t          *sve,
                     vector_t       *result,
                     const vector_t *n,
                     const vector_t *m,
                     const uint32_t  esize,
                     const uint32_t  rotation,
                     const int32_t   index,
                     const uint16_t  pg)
{
    const uint32_t sel_a = rotation & 1U;
    const bool     neg_r = (rotation & 1U) != ((rotation >> 1U) & 1U);
    const bool     neg_i = 0U != (rotation & 2U);

    for (uint32_t pair = 0U; pair < elements_of(esize) / 2U; ++pair)
    {
        const uint32_t source = index < 0 ? pair : (uint32_t)index;
        const uint64_t a      = get_element(n, 2U * pair + sel_a, esize);
        uint64_t       b_r    = get_element(m, 2U * source + sel_a, esize);
        uint64_t       b_i    = get_element(m, 2U * source + (sel_a ^ 1U), esize);
        b_r                   = true == neg_r ? fp_negate(b_r, esize) : b_r;
        b_i                   = true == neg_i ? fp_negate(b_i, esize) : b_i;

        if (true == is_active(pg, 2U * pair, esize))
        {
            set_element(result,
                        2U * pair,
                        esize,
                        fp(sve,
                           esize,
                           BAL_FLOAT_MULTIPLY_ADD,
                           a,
                           b_r,
                           get_element(result, 2U * pair, esize)));
        }

        if (true == is_active(pg, 2U * pair + 1U, esize))
        {
            set_element(result,
                        2U * pair + 1U,
                        esize,
                        fp(sve,
                           esize,
                           BAL_FLOAT_MULTIPLY_ADD,
                           a,
                           b_i,
                           get_element(result, 2U * pair + 1U, esize)));
        }
    }
}

/// FCMLA (vectors): 01100100 size 0 Zm 0 rot Pg Zn Zda.
static sve_result_t
float_complex_multiply_add(sve_t *sve, const uint32_t instruction)
{
    const uint32_t esize = fp_esize((instruction >> 22U) & 0x3U);

    if (0U == esize)
    {
        return SVE_UNDEFINED;
    }

    const vector_t n      = read_z(sve, (instruction >> 5U) & 0x1FU);
    const vector_t m      = read_z(sve, (instruction >> 16U) & 0x1FU);
    vector_t       result = read_z(sve, instruction & 0x1FU);
    complex_multiply_add(sve,
                         &result,
                         &n,
                         &m,
                         esize,
                         (instruction >> 13U) & 0x3U,
                         -1,
                         read_p(sve, (instruction >> 10U) & 0x7U));
    write_z(sve, instruction & 0x1FU, result);
    return SVE_OK;
}

/// FCMLA (indexed): 01100100 1x1 ... 0001 rot Zn Zda, half precision with i2 and Zm in bits 18:16
/// or single precision with i1 and Zm in bits 19:16.
static sve_result_t
float_complex_multiply_add_indexed(sve_t *sve, const uint32_t instruction)
{
    const bool     is_single = 0U != (instruction & 0x400000U);
    const uint32_t esize     = true == is_single ? 32U : 16U;
    const uint32_t zm        = (instruction >> 16U) & (true == is_single ? 0xFU : 0x7U);
    const int32_t  index     = (int32_t)((instruction >> (true == is_single ? 20U : 19U))
                                    & (true == is_single ? 0x1U : 0x3U));
    const vector_t n         = read_z(sve, (instruction >> 5U) & 0x1FU);
    const vector_t m         = read_z(sve, zm);
    vector_t       result    = read_z(sve, instruction & 0x1FU);
    complex_multiply_add(sve, &result, &n, &m, esize, (instruction >> 10U) & 0x3U, index, 0xFFFFU);
    write_z(sve, instruction & 0x1FU, result);
    return SVE_OK;
}

/// FCADD: 01100100 size 00000 rot 100 Pg Zm Zdn.
static sve_result_t
float_complex_add(sve_t *sve, const uint32_t instruction)
{
    const uint32_t esize = fp_esize((instruction >> 22U) & 0x3U);

    if (0U == esize)
    {
        return SVE_UNDEFINED;
    }

    const bool     is_270 = 0U != (instruction & 0x10000U);
    const uint16_t pg     = read_p(sve, (instruction >> 10U) & 0x7U);
    const vector_t m      = read_z(sve, (instruction >> 5U) & 0x1FU);
    vector_t       result = read_z(sve, instruction & 0x1FU);

    for (uint32_t pair = 0U; pair < elements_of(esize) / 2U; ++pair)
    {
        const uint64_t real      = get_element(&m, 2U * pair, esize);
        const uint64_t imaginary = get_element(&m, 2U * pair + 1U, esize);
        const uint64_t to_real   = true == is_270 ? imaginary : fp_negate(imaginary, esize);
        const uint64_t to_imag   = true == is_270 ? fp_negate(real, esize) : real;

        if (true == is_active(pg, 2U * pair, esize))
        {
            set_element(
                &result,
                2U * pair,
                esize,
                fp(sve, esize, BAL_FLOAT_ADD, get_element(&result, 2U * pair, esize), to_real, 0U));
        }

        if (true == is_active(pg, 2U * pair + 1U, esize))
        {
            set_element(&result,
                        2U * pair + 1U,
                        esize,
                        fp(sve,
                           esize,
                           BAL_FLOAT_ADD,
                           get_element(&result, 2U * pair + 1U, esize),
                           to_imag,
                           0U));
        }
    }

    write_z(sve, instruction & 0x1FU, result);
    return SVE_OK;
}

/// FMLA, FMLS (01100100 1 size 1 ... 00000 op Zn Zda) and FMUL (... 001000 Zn Zd) by element:
/// half precision with i3h:i3l and Zm in bits 18:16, single with i2 and Zm in bits 18:16, double
/// with i1 and Zm in bits 19:16.
static sve_result_t
float_multiply_indexed(sve_t *sve, const uint32_t instruction)
{
    const uint32_t size = (instruction >> 22U) & 0x3U;
    uint32_t       esize;
    uint32_t       index;
    uint32_t       zm;

    if (size < 2U)
    {
        esize = 16U;
        index = ((instruction >> 20U) & 0x4U) | ((instruction >> 19U) & 0x3U);
        zm    = (instruction >> 16U) & 0x7U;
    }
    else if (2U == size)
    {
        esize = 32U;
        index = (instruction >> 19U) & 0x3U;
        zm    = (instruction >> 16U) & 0x7U;
    }
    else
    {
        esize = 64U;
        index = (instruction >> 20U) & 0x1U;
        zm    = (instruction >> 16U) & 0xFU;
    }

    const bool     is_multiply = 0U != (instruction & 0x2000U);
    const bool     is_subtract = 0U != (instruction & 0x400U);
    const vector_t n           = read_z(sve, (instruction >> 5U) & 0x1FU);
    const vector_t m           = read_z(sve, zm);
    const uint64_t b           = get_element(&m, index, esize);
    vector_t       result      = read_z(sve, instruction & 0x1FU);

    for (uint32_t e = 0U; e < elements_of(esize); ++e)
    {
        const uint64_t a = get_element(&n, e, esize);
        const uint64_t r = true == is_multiply ? fp(sve, esize, BAL_FLOAT_MULTIPLY, a, b, 0U)
                                               : fp(sve,
                                                    esize,
                                                    BAL_FLOAT_MULTIPLY_ADD,
                                                    true == is_subtract ? fp_negate(a, esize) : a,
                                                    b,
                                                    get_element(&result, e, esize));
        set_element(&result, e, esize, r);
    }

    write_z(sve, instruction & 0x1FU, result);
    return SVE_OK;
}

/// FMLALB, FMLALT, FMLSLB and FMLSLT (vectors, 01100100 101 Zm 10 S 00 T Zn Zda, and indexed,
/// 01100100 101 i3h:i3l Zm 01 S 0 i3l T Zn Zda), and BFMLALB, BFMLALT, BFMLSLB and BFMLSLT
/// (01100100 111 ...): the
/// even or odd half precision or BFloat16 elements, widened and multiplied into single precision
/// with one rounding.
static sve_result_t
float_widening_multiply_add(sve_t *sve, const uint32_t instruction)
{
    const bool     is_bfloat   = 0U != (instruction & 0x400000U);
    const bool     is_indexed  = 0U == (instruction & 0x8000U);
    const bool     is_subtract = 0U != (instruction & 0x2000U);
    const uint32_t top         = (instruction >> 10U) & 1U;

    const uint32_t zm     = (instruction >> 16U) & (true == is_indexed ? 0x7U : 0x1FU);
    const uint32_t index  = (((instruction >> 19U) & 0x3U) << 1U) | ((instruction >> 11U) & 1U);
    const uint64_t fpcr   = sve->cpu->fpcr;
    const vector_t n      = read_z(sve, (instruction >> 5U) & 0x1FU);
    const vector_t m      = read_z(sve, zm);
    vector_t       result = read_z(sve, instruction & 0x1FU);

    for (uint32_t e = 0U; e < 4U; ++e)
    {
        uint64_t a = get_element(&n, 2U * e + top, 16U);
        uint64_t b = get_element(&m, true == is_indexed ? index : 2U * e + top, 16U);
        a          = true == is_subtract ? a ^ 0x8000U : a;
        a          = true == is_bfloat ? a << 16U : bal_simd_half_to_single(fpcr, a);
        b          = true == is_bfloat ? b << 16U : bal_simd_half_to_single(fpcr, b);
        set_element(&result,
                    e,
                    32U,
                    fp(sve, 32U, BAL_FLOAT_MULTIPLY_ADD, a, b, get_element(&result, e, 32U)));
    }

    write_z(sve, instruction & 0x1FU, result);
    return SVE_OK;
}

/// BFDOT (vectors, 01100100 011 Zm 100000 Zn Zda, and indexed, 01100100 011 i2 Zm 010000 Zn Zda).
static sve_result_t
bfloat_dot(sve_t *sve, const uint32_t instruction)
{
    const bool     is_indexed = 0U == (instruction & 0x8000U);
    const uint32_t zm         = (instruction >> 16U) & (true == is_indexed ? 0x7U : 0x1FU);
    const uint32_t index      = (instruction >> 19U) & 0x3U;
    const vector_t n          = read_z(sve, (instruction >> 5U) & 0x1FU);
    const vector_t m          = read_z(sve, zm);
    vector_t       result     = read_z(sve, instruction & 0x1FU);

    for (uint32_t e = 0U; e < 4U; ++e)
    {
        const uint32_t pair = true == is_indexed ? index : e;
        set_element(&result,
                    e,
                    32U,
                    bal_simd_bfloat_dot_add((uint32_t)get_element(&result, e, 32U),
                                            (uint16_t)get_element(&n, 2U * e, 16U),
                                            (uint16_t)get_element(&n, 2U * e + 1U, 16U),
                                            (uint16_t)get_element(&m, 2U * pair, 16U),
                                            (uint16_t)get_element(&m, 2U * pair + 1U, 16U)));
    }

    write_z(sve, instruction & 0x1FU, result);
    return SVE_OK;
}

/// BFMMLA (01100100 011 Zm 111001 Zn Zda) and FMMLA on single precision (01100100 101 Zm 111001
/// Zn Zda): a 2x2 matrix product added to Zda.
static sve_result_t
matrix_multiply_add_float(sve_t *sve, const uint32_t instruction)
{
    const bool     is_bfloat = 0x6460E400U == (instruction & 0xFFE0FC00U);
    const vector_t n         = read_z(sve, (instruction >> 5U) & 0x1FU);
    const vector_t m         = read_z(sve, (instruction >> 16U) & 0x1FU);
    const vector_t d         = read_z(sve, instruction & 0x1FU);
    vector_t       result    = d;

    for (uint32_t i = 0U; i < 2U; ++i)
    {
        for (uint32_t j = 0U; j < 2U; ++j)
        {
            uint64_t sum = get_element(&d, 2U * i + j, 32U);

            if (true == is_bfloat)
            {
                for (uint32_t k = 0U; k < 2U; ++k)
                {
                    sum = bal_simd_bfloat_dot_add(
                        (uint32_t)sum,
                        (uint16_t)get_element(&n, 4U * i + 2U * k, 16U),
                        (uint16_t)get_element(&n, 4U * i + 2U * k + 1U, 16U),
                        (uint16_t)get_element(&m, 4U * j + 2U * k, 16U),
                        (uint16_t)get_element(&m, 4U * j + 2U * k + 1U, 16U));
                }
            }
            else
            {
                const uint64_t p0 = fp(sve,
                                       32U,
                                       BAL_FLOAT_MULTIPLY,
                                       get_element(&n, 2U * i, 32U),
                                       get_element(&m, 2U * j, 32U),
                                       0U);
                const uint64_t p1 = fp(sve,
                                       32U,
                                       BAL_FLOAT_MULTIPLY,
                                       get_element(&n, 2U * i + 1U, 32U),
                                       get_element(&m, 2U * j + 1U, 32U),
                                       0U);
                sum = fp(sve, 32U, BAL_FLOAT_ADD, sum, fp(sve, 32U, BAL_FLOAT_ADD, p0, p1, 0U), 0U);
            }

            set_element(&result, 2U * i + j, 32U, sum);
        }
    }

    write_z(sve, instruction & 0x1FU, result);
    return SVE_OK;
}

// -----------------------------------------------------------------------------
// SVE2 integer
// -----------------------------------------------------------------------------

/// SRSHL, URSHL, SQSHL, UQSHL, SQRSHL and UQRSHL on one element, by the whole signed element `b`
/// (unlike the Advanced SIMD forms, which take its low byte).
static uint64_t
shift_by_vector(const uint64_t a,
                const uint64_t b,
                const uint32_t esize,
                const bool     is_signed,
                const bool     is_rounding,
                const bool     is_saturating)
{
    const int64_t    amount = sign_extend(b, esize);
    const __int128   result = shift_wide(wide(a, esize, is_signed), amount, esize, is_rounding);
    const saturate_t mode   = false == is_saturating
                                  ? SATURATE_NONE
                                  : (true == is_signed ? SATURATE_SIGNED : SATURATE_UNSIGNED);
    return narrow_to(result, esize, mode);
}

/// The SVE2 predicated integer group: 01000100 size 0 opc 10 x Pg Zm Zdn, with the unary forms
/// (URECPE, URSQRTE, SQABS, SQNEG) taking Zn and Zd and the pairwise ones Zm and Zdn.
static sve_result_t
sve2_integer_predicated(sve_t *sve, const uint32_t instruction)
{
    const uint32_t esize  = size_at_22(instruction);
    const uint32_t opc    = (instruction >> 16U) & 0x1FU;
    const bool     is_b   = 0U != (instruction & 0x2000U);
    const uint32_t rd     = instruction & 0x1FU;
    const uint16_t pg     = read_p(sve, (instruction >> 10U) & 0x7U);
    const vector_t m      = read_z(sve, (instruction >> 5U) & 0x1FU);
    vector_t       d      = read_z(sve, rd);
    vector_t       result = d;

    if (true == is_b)
    {
        const bool is_estimate         = opc <= 1U;
        const bool is_pairwise         = 0x11U == opc || (opc >= 0x14U && opc <= 0x17U);
        const bool is_accumulate_pairs = 0x4U == opc || 0x5U == opc;

        if ((true == is_estimate && 32U != esize) || (true == is_accumulate_pairs && 8U == esize)
            || (false == is_estimate && false == is_pairwise && false == is_accumulate_pairs
                && 0x8U != opc && 0x9U != opc))
        {
            return SVE_UNDEFINED;
        }

        for (uint32_t e = 0U; e < elements_of(esize); ++e)
        {
            if (false == is_active(pg, e, esize))
            {
                continue;
            }

            uint64_t value;

            if (true == is_pairwise)
            {
                const vector_t *source    = 0U == (e & 1U) ? &d : &m;
                const uint64_t  x         = get_element(source, e & ~1U, esize);
                const uint64_t  y         = get_element(source, (e & ~1U) + 1U, esize);
                const bool      is_signed = 0U == (opc & 1U);

                if (0x11U == opc)
                {
                    value = x + y;
                }
                else if (opc <= 0x15U)
                {
                    value = true == is_greater(x, y, esize, is_signed) ? x : y;
                }
                else
                {
                    value = true == is_greater(x, y, esize, is_signed) ? y : x;
                }
            }
            else if (true == is_accumulate_pairs)
            {
                const bool     is_signed = 0x4U == opc;
                const uint32_t half      = esize / 2U;
                value                    = get_element(&d, e, esize)
                        + (uint64_t)(wide(get_element(&m, 2U * e, half), half, is_signed)
                                     + wide(get_element(&m, 2U * e + 1U, half), half, is_signed));
            }
            else if (true == is_estimate)
            {
                value = bal_unsigned_estimate_element((uint32_t)get_element(&m, e, 32U), 1U == opc);
            }
            else
            {
                // SQABS and SQNEG.
                const __int128 x = wide(get_element(&m, e, esize), esize, true);
                value = narrow_to(0x8U == opc ? (x < 0 ? -x : x) : -x, esize, SATURATE_SIGNED);
            }

            set_element(&result, e, esize, value);
        }

        write_z(sve, rd, result);
        return SVE_OK;
    }

    if (opc < 2U || 4U == opc || 5U == opc)
    {
        return SVE_UNDEFINED;
    }

    for (uint32_t e = 0U; e < elements_of(esize); ++e)
    {
        if (false == is_active(pg, e, esize))
        {
            continue;
        }

        uint64_t   a         = get_element(&d, e, esize);
        uint64_t   b         = get_element(&m, e, esize);
        const bool is_signed = 0U == (opc & 1U);
        uint64_t   value;

        if (opc < 0x10U)
        {
            // The shifts: opc<3> saturates, opc<2> reverses the operands and opc<1> rounds.
            const bool is_reversed = 0U != (opc & 0x4U);

            if (true == is_reversed)
            {
                const uint64_t t = a;
                a                = b;
                b                = t;
            }

            const bool is_saturating = opc >= 0x8U;
            const bool is_rounding   = 0U != (opc & 0x2U);
            value = shift_by_vector(a, b, esize, is_signed, is_rounding, is_saturating);
        }
        else if (opc < 0x18U)
        {
            // SHADD, UHADD, SHSUB, UHSUB, SRHADD, URHADD, SHSUBR and UHSUBR.
            const __int128 x = wide(a, esize, is_signed);
            const __int128 y = wide(b, esize, is_signed);
            __int128       r;

            switch (opc & 0x6U)
            {
                case 0x0U:
                    r = x + y;
                    break;
                case 0x2U:
                    r = x - y;
                    break;
                case 0x4U:
                    r = x + y + 1;
                    break;
                default:
                    r = y - x;
                    break;
            }

            value = (uint64_t)(r >> 1) & mask_of(esize);
        }
        else
        {
            // SQADD, UQADD, SQSUB, UQSUB, SUQADD, USQADD, SQSUBR and UQSUBR.
            const __int128 x = wide(a, esize, is_signed);
            __int128       r;

            switch (opc)
            {
                case 0x1CU:
                    r = x + wide(b, esize, false);
                    break;
                case 0x1DU:
                    r = x + wide(b, esize, true);
                    break;
                case 0x1AU:
                case 0x1BU:
                    r = x - wide(b, esize, is_signed);
                    break;
                case 0x1EU:
                case 0x1FU:
                    r = wide(b, esize, is_signed) - x;
                    break;
                default:
                    r = x + wide(b, esize, is_signed);
                    break;
            }

            value = narrow_to(r, esize, true == is_signed ? SATURATE_SIGNED : SATURATE_UNSIGNED);
        }

        set_element(&result, e, esize, value);
    }

    write_z(sve, rd, result);
    return SVE_OK;
}

/// SQRDMLAH and SQRDMLSH on one element: the accumulator plus or minus the rounded high half of
/// twice the product, saturated. The accumulator is a multiple of 2^esize in the sum, so it adds
/// after the shift.
static uint64_t
rounding_doubling_multiply_add(const uint64_t acc,
                               const uint64_t a,
                               const uint64_t b,
                               const uint32_t esize,
                               const bool     is_subtract)
{
    const __int128 product = wide(a, esize, true) * wide(b, esize, true);
    const __int128 high    = doubled_high(true == is_subtract ? -product : product, esize, true);
    return narrow_to(wide(acc, esize, true) + high, esize, SATURATE_SIGNED);
}

/// SQDMULH and SQRDMULH (vectors, unpredicated): 00000100 size 1 Zm 01110 R Zn Zd.
static sve_result_t
doubling_multiply_high_vectors(sve_t *sve, const uint32_t instruction)
{
    const uint32_t esize       = size_at_22(instruction);
    const bool     is_rounding = 0U != (instruction & 0x400U);
    const vector_t n           = read_z(sve, (instruction >> 5U) & 0x1FU);
    const vector_t m           = read_z(sve, (instruction >> 16U) & 0x1FU);
    vector_t       result      = { { 0U, 0U } };

    for (uint32_t e = 0U; e < elements_of(esize); ++e)
    {
        set_element(&result,
                    e,
                    esize,
                    doubling_multiply_high(
                        get_element(&n, e, esize), get_element(&m, e, esize), esize, is_rounding));
    }

    write_z(sve, instruction & 0x1FU, result);
    return SVE_OK;
}

/// EOR3, BCAX (00000100 0 op 1 Zm 001110 Zk Zdn) and BSL, BSL1N, BSL2N and NBSL (00000100 opc 1
/// Zm 001111 Zk Zdn), on the whole vector.
static sve_result_t
bitwise_ternary(sve_t *sve, const uint32_t instruction)
{
    const uint32_t opc    = (instruction >> 22U) & 0x3U;
    const bool     is_bsl = 0U != (instruction & 0x400U);

    if (false == is_bsl && opc >= 2U)
    {
        return SVE_UNDEFINED;
    }

    const vector_t dn     = read_z(sve, instruction & 0x1FU);
    const vector_t m      = read_z(sve, (instruction >> 16U) & 0x1FU);
    const vector_t k      = read_z(sve, (instruction >> 5U) & 0x1FU);
    vector_t       result = { { 0U, 0U } };

    for (uint32_t i = 0U; i < 2U; ++i)
    {
        const uint64_t a = dn.d[i];
        const uint64_t b = m.d[i];
        const uint64_t c = k.d[i];

        if (false == is_bsl)
        {
            result.d[i] = 0U == opc ? a ^ b ^ c : a ^ (b & ~c);
        }
        else
        {
            switch (opc)
            {
                case 0U:
                    result.d[i] = (a & c) | (b & ~c);
                    break;
                case 1U:
                    result.d[i] = (~a & c) | (b & ~c);
                    break;
                case 2U:
                    result.d[i] = (a & c) | (~b & ~c);
                    break;
                default:
                    result.d[i] = ~((a & c) | (b & ~c));
                    break;
            }
        }
    }

    write_z(sve, instruction & 0x1FU, result);
    return SVE_OK;
}

/// The element size and shift of a right shift by immediate encoded as tsz:imm3, with tsz of
/// `bits` bits; 0 for a reserved tsz.
static uint32_t
right_shift_immediate(const uint32_t tsz, const uint32_t imm3, uint32_t *shift)
{
    if (0U == tsz)
    {
        return 0U;
    }

    const uint32_t esize = 8U << (31U - (uint32_t)__builtin_clz(tsz));
    *shift               = 2U * esize - ((tsz << 3U) | imm3);
    return esize;
}

/// XAR: 00000100 tszh 1 tszl imm3 001101 Zm Zdn, exclusive or and rotate right.
static sve_result_t
exclusive_or_rotate(sve_t *sve, const uint32_t instruction)
{
    const uint32_t tsz   = (((instruction >> 22U) & 0x3U) << 2U) | ((instruction >> 19U) & 0x3U);
    uint32_t       shift = 0U;
    const uint32_t esize = right_shift_immediate(tsz, (instruction >> 16U) & 0x7U, &shift);

    if (0U == esize)
    {
        return SVE_UNDEFINED;
    }

    const vector_t dn     = read_z(sve, instruction & 0x1FU);
    const vector_t m      = read_z(sve, (instruction >> 5U) & 0x1FU);
    vector_t       result = { { 0U, 0U } };

    for (uint32_t e = 0U; e < elements_of(esize); ++e)
    {
        const uint64_t value = get_element(&dn, e, esize) ^ get_element(&m, e, esize);
        set_element(&result,
                    e,
                    esize,
                    shift == esize
                        ? value
                        : ((value >> shift) | (value << (esize - shift))) & mask_of(esize));
    }

    write_z(sve, instruction & 0x1FU, result);
    return SVE_OK;
}

/// The element size, index and Zm of a by-element form whose size is in bits 23:22: halfwords
/// take i3h:i3l (bits 22, 20:19) and Zm in bits 18:16, words i2 (bits 20:19) and Zm in bits
/// 18:16, doublewords i1 (bit 20) and Zm in bits 19:16.
static uint32_t
indexed_operand(const uint32_t instruction, uint32_t *index, uint32_t *zm)
{
    const uint32_t size = (instruction >> 22U) & 0x3U;

    if (size < 2U)
    {
        *index = (((instruction >> 22U) & 1U) << 2U) | ((instruction >> 19U) & 0x3U);
        *zm    = (instruction >> 16U) & 0x7U;
        return 16U;
    }

    if (2U == size)
    {
        *index = (instruction >> 19U) & 0x3U;
        *zm    = (instruction >> 16U) & 0x7U;
        return 32U;
    }

    *index = (instruction >> 20U) & 0x1U;
    *zm    = (instruction >> 16U) & 0xFU;
    return 64U;
}

/// MLA, MLS (01000100 1 size 1 ... 00001 S Zn Zda), SQRDMLAH, SQRDMLSH (... 00010 S ...),
/// SQDMULH, SQRDMULH (... 11110 R ...) and MUL (... 111110 ...) by element.
static sve_result_t
integer_multiply_indexed(sve_t *sve, const uint32_t instruction)
{
    uint32_t       index;
    uint32_t       zm;
    const uint32_t esize  = indexed_operand(instruction, &index, &zm);
    const uint32_t op     = (instruction >> 10U) & 0x3FU;
    const vector_t n      = read_z(sve, (instruction >> 5U) & 0x1FU);
    const vector_t m      = read_z(sve, zm);
    const vector_t d      = read_z(sve, instruction & 0x1FU);
    vector_t       result = d;

    // Within each 128-bit segment, here the whole vector.
    const uint64_t b = get_element(&m, index, esize);

    for (uint32_t e = 0U; e < elements_of(esize); ++e)
    {
        const uint64_t a   = get_element(&n, e, esize);
        const uint64_t acc = get_element(&d, e, esize);
        uint64_t       value;

        switch (op)
        {
            case 0x02U:
                value = acc + a * b;
                break;
            case 0x03U:
                value = acc - a * b;
                break;
            case 0x04U:
            case 0x05U:
                value = rounding_doubling_multiply_add(acc, a, b, esize, 0x05U == op);
                break;
            case 0x3CU:
            case 0x3DU:
                value = doubling_multiply_high(a, b, esize, 0x3DU == op);
                break;
            default:
                value = a * b;
                break;
        }

        set_element(&result, e, esize, value);
    }

    write_z(sve, instruction & 0x1FU, result);
    return SVE_OK;
}

/// The dot product of the four `esize / 4`-bit elements of `a` and `b` at element `e`, added to
/// `acc`; `a_signed` and `b_signed` pick the signedness of each.
static uint64_t
dot_four(const uint64_t  acc,
         const vector_t *a,
         const vector_t *b,
         const uint32_t  e,
         const uint32_t  be,
         const uint32_t  esize,
         const bool      a_signed,
         const bool      b_signed)
{
    const uint32_t small = esize / 4U;
    __int128       sum   = 0;

    for (uint32_t i = 0U; i < 4U; ++i)
    {
        sum += wide(get_element(a, 4U * e + i, small), small, a_signed)
               * wide(get_element(b, 4U * be + i, small), small, b_signed);
    }

    return (acc + (uint64_t)sum) & mask_of(esize);
}

/// SDOT and UDOT (vectors, 01000100 size 0 Zm 00000 U Zn Zda, and indexed, 01000100 1 size 1 ...
/// 00000 U ...), USDOT (vectors, 01000100 100 Zm 011110 ..., and indexed, 01000100 101 ... 00011
/// 0 ...) and SUDOT (indexed, ... 000111 ...).
static sve_result_t
integer_dot_product(sve_t *sve, const uint32_t instruction)
{
    const bool is_indexed = 0U != (instruction & 0x200000U);
    const bool is_mixed
        = 0x44807800U == (instruction & 0xFFE0FC00U) || 0U != (instruction & 0x1000U);
    uint32_t esize = 8U << ((instruction >> 22U) & 0x3U);
    uint32_t index = 0U;
    uint32_t zm    = (instruction >> 16U) & 0x1FU;

    if (true == is_indexed)
    {
        esize = indexed_operand(instruction, &index, &zm);
    }

    if (esize < 32U || (true == is_mixed && 32U != esize))
    {
        return SVE_UNDEFINED;
    }

    bool a_signed = 0U == (instruction & 0x400U);
    bool b_signed = a_signed;

    if (true == is_mixed)
    {
        // USDOT takes unsigned Zn and signed Zm; SUDOT the reverse.
        const bool is_sudot = true == is_indexed && 0U != (instruction & 0x400U);
        a_signed            = is_sudot;
        b_signed            = !is_sudot;
    }

    const vector_t n      = read_z(sve, (instruction >> 5U) & 0x1FU);
    const vector_t m      = read_z(sve, zm);
    vector_t       result = read_z(sve, instruction & 0x1FU);

    for (uint32_t e = 0U; e < elements_of(esize); ++e)
    {
        set_element(&result,
                    e,
                    esize,
                    dot_four(get_element(&result, e, esize),
                             &n,
                             &m,
                             e,
                             true == is_indexed ? index : e,
                             esize,
                             a_signed,
                             b_signed));
    }

    write_z(sve, instruction & 0x1FU, result);
    return SVE_OK;
}

/// One widening multiply-add kind of the SVE2 integer multiply-add group.
typedef enum
{
    WIDEN_MULTIPLY_ADD,
    WIDEN_MULTIPLY_SUBTRACT,
    WIDEN_MULTIPLY,
    WIDEN_DOUBLING_ADD,
    WIDEN_DOUBLING_SUBTRACT,
    WIDEN_DOUBLING,
} widen_kind_t;

/// The widening products of the even (`top` 0) or odd elements of Zn and either the same elements
/// of Zm or element `index` of each segment of Zm (when non-negative), into `esize`-bit elements.
static void
widening_multiply(vector_t          *result,
                  const vector_t    *n,
                  const vector_t    *m,
                  const uint32_t     esize,
                  const uint32_t     top_n,
                  const uint32_t     top_m,
                  const int32_t      index,
                  const bool         is_signed,
                  const widen_kind_t kind)
{
    const uint32_t half = esize / 2U;

    for (uint32_t e = 0U; e < elements_of(esize); ++e)
    {
        const __int128 a = wide(get_element(n, 2U * e + top_n, half), half, is_signed);
        const __int128 b = wide(
            get_element(m, index < 0 ? 2U * e + top_m : (uint32_t)index, half), half, is_signed);
        const __int128 product = a * b;
        const __int128 acc = wide(get_element(result, e, esize), esize, kind >= WIDEN_DOUBLING_ADD);
        // The doubled product saturates before it is accumulated, and the sum again.
        const __int128 doubled = wide(narrow_to(2 * product, esize, SATURATE_SIGNED), esize, true);
        uint64_t       value;

        switch (kind)
        {
            case WIDEN_MULTIPLY_ADD:
                value = narrow_to(acc + product, esize, SATURATE_NONE);
                break;
            case WIDEN_MULTIPLY_SUBTRACT:
                value = narrow_to(acc - product, esize, SATURATE_NONE);
                break;
            case WIDEN_MULTIPLY:
                value = narrow_to(product, esize, SATURATE_NONE);
                break;
            case WIDEN_DOUBLING_ADD:
                value = narrow_to(acc + doubled, esize, SATURATE_SIGNED);
                break;
            case WIDEN_DOUBLING_SUBTRACT:
                value = narrow_to(acc - doubled, esize, SATURATE_SIGNED);
                break;
            default:
                value = (uint64_t)doubled & mask_of(esize);
                break;
        }

        set_element(result, e, esize, value);
    }
}

/// The SVE2 integer multiply-add group on vectors: 01000100 size 0 Zm 0 op Zn Zda with SQDMLALBT,
/// SQDMLSLBT (00001 S), CDOT (0001 rot), CMLA (0010 rot), SQRDCMLAH (0011 rot), SMLALB to
/// UMLSLT (010 S U T), SQDMLALB to SQDMLSLT (0110 S T) and SQRDMLAH, SQRDMLSH (01110 S).
static sve_result_t
integer_multiply_add_vectors(sve_t *sve, const uint32_t instruction)
{
    const uint32_t esize  = size_at_22(instruction);
    const uint32_t op     = (instruction >> 10U) & 0x1FU;
    const vector_t n      = read_z(sve, (instruction >> 5U) & 0x1FU);
    const vector_t m      = read_z(sve, (instruction >> 16U) & 0x1FU);
    vector_t       result = read_z(sve, instruction & 0x1FU);

    if (op >= 0x10U && op < 0x1CU)
    {
        // The widening forms need a wider destination.
        if (8U == esize)
        {
            return SVE_UNDEFINED;
        }

        if (op < 0x18U)
        {
            const bool is_signed = 0U == (op & 0x2U);
            widening_multiply(&result,
                              &n,
                              &m,
                              esize,
                              op & 1U,
                              op & 1U,
                              -1,
                              is_signed,
                              0U != (op & 0x4U) ? WIDEN_MULTIPLY_SUBTRACT : WIDEN_MULTIPLY_ADD);
        }
        else
        {
            widening_multiply(&result,
                              &n,
                              &m,
                              esize,
                              op & 1U,
                              op & 1U,
                              -1,
                              true,
                              0U != (op & 0x2U) ? WIDEN_DOUBLING_SUBTRACT : WIDEN_DOUBLING_ADD);
        }
    }
    else if (0x2U == op || 0x3U == op)
    {
        // SQDMLALBT and SQDMLSLBT: the even elements of Zn and the odd ones of Zm.
        if (8U == esize)
        {
            return SVE_UNDEFINED;
        }

        widening_multiply(&result,
                          &n,
                          &m,
                          esize,
                          0U,
                          1U,
                          -1,
                          true,
                          0x3U == op ? WIDEN_DOUBLING_SUBTRACT : WIDEN_DOUBLING_ADD);
    }
    else if (0x1CU == op || 0x1DU == op)
    {
        for (uint32_t e = 0U; e < elements_of(esize); ++e)
        {
            set_element(&result,
                        e,
                        esize,
                        rounding_doubling_multiply_add(get_element(&result, e, esize),
                                                       get_element(&n, e, esize),
                                                       get_element(&m, e, esize),
                                                       esize,
                                                       0x1DU == op));
        }
    }
    else
    {
        return SVE_UNDEFINED;
    }

    write_z(sve, instruction & 0x1FU, result);
    return SVE_OK;
}

/// The widening forms by element: 01000100 1 size 1 ... op Zn Zda with SQDMLALB to SQDMLSLT
/// (001 S i T), SMLALB to UMLSLT (10 S U i T), SMULLB to UMULLT (110 U i T) and SQDMULLB,
/// SQDMULLT (1110 i T); halfwords take index i3h:i3l:i, words i2h:i2l.
static sve_result_t
integer_multiply_long_indexed(sve_t *sve, const uint32_t instruction)
{
    const bool     is_words = 0U != (instruction & 0x400000U);
    const uint32_t esize    = true == is_words ? 64U : 32U;
    const uint32_t zm       = (instruction >> 16U) & (true == is_words ? 0xFU : 0x7U);
    const uint32_t index
        = ((true == is_words ? (instruction >> 20U) & 0x1U : (instruction >> 19U) & 0x3U) << 1U)
          | ((instruction >> 11U) & 1U);
    const uint32_t op     = (instruction >> 12U) & 0xFU;
    const uint32_t top    = (instruction >> 10U) & 1U;
    const vector_t n      = read_z(sve, (instruction >> 5U) & 0x1FU);
    const vector_t m      = read_z(sve, zm);
    vector_t       result = read_z(sve, instruction & 0x1FU);
    widen_kind_t   kind;
    bool           is_signed = true;

    switch (op)
    {
        case 0x2U:
            kind = WIDEN_DOUBLING_ADD;
            break;
        case 0x3U:
            kind = WIDEN_DOUBLING_SUBTRACT;
            break;
        case 0x8U:
        case 0x9U:
            kind      = WIDEN_MULTIPLY_ADD;
            is_signed = 0x8U == op;
            break;
        case 0xAU:
        case 0xBU:
            kind      = WIDEN_MULTIPLY_SUBTRACT;
            is_signed = 0xAU == op;
            break;
        case 0xCU:
        case 0xDU:
            kind      = WIDEN_MULTIPLY;
            is_signed = 0xCU == op;
            result    = (vector_t) { { 0U, 0U } };
            break;
        case 0xEU:
            kind   = WIDEN_DOUBLING;
            result = (vector_t) { { 0U, 0U } };
            break;
        default:
            return SVE_UNDEFINED;
    }

    widening_multiply(&result, &n, &m, esize, top, 0U, (int32_t)index, is_signed, kind);
    write_z(sve, instruction & 0x1FU, result);
    return SVE_OK;
}

/// The real and imaginary parts of CMLA and SQRDCMLAH on the pair at `pair`: rotation 0 adds
/// re*re and re*im, 90 subtracts im*im and adds im*re, and so on.
static void
complex_integer_pair(const vector_t *n,
                     const vector_t *m,
                     const uint32_t  pair,
                     const uint32_t  mpair,
                     const uint32_t  esize,
                     const uint32_t  rotation,
                     uint64_t       *a_r,
                     uint64_t       *b_r,
                     bool           *neg_r,
                     uint64_t       *a_i,
                     uint64_t       *b_i,
                     bool           *neg_i)
{
    const uint32_t sel_a = rotation & 1U;
    *a_r                 = get_element(n, 2U * pair + sel_a, esize);
    *a_i                 = *a_r;
    *b_r                 = get_element(m, 2U * mpair + sel_a, esize);
    *b_i                 = get_element(m, 2U * mpair + (sel_a ^ 1U), esize);
    *neg_r               = 1U == rotation || 2U == rotation;
    *neg_i               = rotation >= 2U;
}

/// CMLA and SQRDCMLAH, vectors (01000100 size 0 Zm 001 S rot Zn Zda) and by element (01000100 1
/// size 1 ... 011 S rot ...: halfwords with i2 and Zm in bits 18:16, words with i1 and Zm in bits
/// 19:16).
static sve_result_t
complex_integer_multiply_add(sve_t *sve, const uint32_t instruction)
{
    const bool     is_indexed  = 0U != (instruction & 0x200000U);
    const bool     is_rounding = 0U != (instruction & 0x1000U);
    const uint32_t rotation    = (instruction >> 10U) & 0x3U;
    uint32_t       esize       = size_at_22(instruction);
    uint32_t       zm          = (instruction >> 16U) & 0x1FU;
    int32_t        index       = -1;

    if (true == is_indexed)
    {
        const bool is_words = 0U != (instruction & 0x400000U);
        esize               = true == is_words ? 32U : 16U;
        zm                  = (instruction >> 16U) & (true == is_words ? 0xFU : 0x7U);
        index               = (int32_t)((instruction >> (true == is_words ? 20U : 19U))
                          & (true == is_words ? 0x1U : 0x3U));
    }

    const vector_t n      = read_z(sve, (instruction >> 5U) & 0x1FU);
    const vector_t m      = read_z(sve, zm);
    vector_t       result = read_z(sve, instruction & 0x1FU);

    for (uint32_t pair = 0U; pair < elements_of(esize) / 2U; ++pair)
    {
        uint64_t a_r, b_r, a_i, b_i;
        bool     neg_r, neg_i;
        complex_integer_pair(&n,
                             &m,
                             pair,
                             index < 0 ? pair : (uint32_t)index,
                             esize,
                             rotation,
                             &a_r,
                             &b_r,
                             &neg_r,
                             &a_i,
                             &b_i,
                             &neg_i);

        for (uint32_t part = 0U; part < 2U; ++part)
        {
            const uint32_t e   = 2U * pair + part;
            const uint64_t acc = get_element(&result, e, esize);
            const uint64_t a   = 0U == part ? a_r : a_i;
            const uint64_t b   = 0U == part ? b_r : b_i;
            const bool     neg = 0U == part ? neg_r : neg_i;
            uint64_t       value;

            if (true == is_rounding)
            {
                value = rounding_doubling_multiply_add(acc, a, b, esize, neg);
            }
            else
            {
                const uint64_t product = a * b;
                value                  = true == neg ? acc - product : acc + product;
            }

            set_element(&result, e, esize, value);
        }
    }

    write_z(sve, instruction & 0x1FU, result);
    return SVE_OK;
}

/// CDOT, vectors (01000100 size 0 Zm 0001 rot Zn Zda) and by element (01000100 1 size 1 ... 0100
/// rot ...): the complex dot product of four 8-bit or 16-bit pairs per element.
static sve_result_t
complex_dot_product(sve_t *sve, const uint32_t instruction)
{
    const bool     is_indexed = 0U != (instruction & 0x200000U);
    const uint32_t rotation   = (instruction >> 10U) & 0x3U;
    uint32_t       esize      = size_at_22(instruction);
    uint32_t       zm         = (instruction >> 16U) & 0x1FU;
    int32_t        index      = -1;

    if (true == is_indexed)
    {
        const bool is_words = 0U != (instruction & 0x400000U);
        esize               = true == is_words ? 64U : 32U;
        zm                  = (instruction >> 16U) & (true == is_words ? 0xFU : 0x7U);
        index               = (int32_t)((instruction >> (true == is_words ? 20U : 19U))
                          & (true == is_words ? 0x1U : 0x3U));
    }

    if (esize < 32U)
    {
        return SVE_UNDEFINED;
    }

    const uint32_t small = esize / 4U;
    const uint32_t sel_a = rotation & 1U;
    // Rotations 0 and 270 subtract the imaginary products.
    const bool     neg_i  = 0U == rotation || 3U == rotation;
    const vector_t n      = read_z(sve, (instruction >> 5U) & 0x1FU);
    const vector_t m      = read_z(sve, zm);
    vector_t       result = read_z(sve, instruction & 0x1FU);

    for (uint32_t e = 0U; e < elements_of(esize); ++e)
    {
        const uint32_t source = index < 0 ? e : (uint32_t)index;
        __int128       sum    = wide(get_element(&result, e, esize), esize, true);

        for (uint32_t pair = 0U; pair < 2U; ++pair)
        {
            const __int128 n_r = wide(get_element(&n, 4U * e + 2U * pair, small), small, true);
            const __int128 n_i = wide(get_element(&n, 4U * e + 2U * pair + 1U, small), small, true);
            const __int128 m_a
                = wide(get_element(&m, 4U * source + 2U * pair + sel_a, small), small, true);
            const __int128 m_b
                = wide(get_element(&m, 4U * source + 2U * pair + (sel_a ^ 1U), small), small, true);
            sum += n_r * m_a;
            sum += (true == neg_i ? -n_i : n_i) * m_b;
        }

        set_element(&result, e, esize, narrow_to(sum, esize, SATURATE_NONE));
    }

    write_z(sve, instruction & 0x1FU, result);
    return SVE_OK;
}

/// SQSHL, UQSHL, SRSHR, URSHR and SQSHLU by immediate, predicated: 00000100 tszh 00 opc 100 Pg
/// tszl imm3 Zdn.
static sve_result_t
shift_immediate_saturating(sve_t *sve, const uint32_t instruction)
{
    const uint32_t opc     = (instruction >> 16U) & 0xFU;
    const uint32_t tsize   = (((instruction >> 22U) & 0x3U) << 2U) | ((instruction >> 8U) & 0x3U);
    const bool     is_left = 0x6U == opc || 0x7U == opc || 0xFU == opc;
    uint32_t       shift   = 0U;

    if (opc != 0x6U && opc != 0x7U && opc != 0xCU && opc != 0xDU && opc != 0xFU)
    {
        return SVE_UNDEFINED;
    }

    const uint32_t esize = shift_immediate_size(tsize, (instruction >> 5U) & 0x7U, is_left, &shift);

    if (0U == esize)
    {
        return SVE_UNDEFINED;
    }

    const bool       is_signed = 0x7U != opc && 0xDU != opc;
    const saturate_t mode      = 0x6U == opc
                                     ? SATURATE_SIGNED
                                     : (0x7U == opc || 0xFU == opc ? SATURATE_UNSIGNED : SATURATE_NONE);
    const uint16_t   pg        = read_p(sve, (instruction >> 10U) & 0x7U);
    vector_t         result    = read_z(sve, instruction & 0x1FU);

    for (uint32_t e = 0U; e < elements_of(esize); ++e)
    {
        if (true == is_active(pg, e, esize))
        {
            const __int128 value  = wide(get_element(&result, e, esize), esize, is_signed);
            const int64_t  amount = true == is_left ? (int64_t)shift : -(int64_t)shift;
            set_element(&result,
                        e,
                        esize,
                        narrow_to(shift_wide(value, amount, esize, false == is_left), esize, mode));
        }
    }

    write_z(sve, instruction & 0x1FU, result);
    return SVE_OK;
}

/// The SVE2 widening group: 01000101 size 0 Zm op Zn Zd with the long add, subtract and absolute
/// difference (00 op U T), the wide add and subtract (010 S U T), SQDMULLB/T (01100 T), PMULLB/T
/// (01101 T), SMULLB to UMULLT (0111 U T), SADDLBT, SSUBLBT and SSUBLTB (1000 S T) and the
/// absolute difference and accumulate long (1100 U T).
static sve_result_t
integer_widening(sve_t *sve, const uint32_t instruction)
{
    const uint32_t size      = (instruction >> 22U) & 0x3U;
    const uint32_t esize     = 8U << size;
    const uint32_t half      = esize / 2U;
    const uint32_t op        = (instruction >> 10U) & 0x3FU;
    const uint32_t top       = op & 1U;
    const bool     is_signed = 0U == (op & 0x2U);
    const vector_t n         = read_z(sve, (instruction >> 5U) & 0x1FU);
    const vector_t m         = read_z(sve, (instruction >> 16U) & 0x1FU);
    vector_t       result
        = 0x30U == (op & 0x3CU) ? read_z(sve, instruction & 0x1FU) : (vector_t) { { 0U, 0U } };

    // PMULLB and PMULLT on doublewords produce a quadword (FEAT_SVE_PMULL128).
    if (0U == size && 0x1AU == (op & 0x3EU))
    {
        const uint64_t    a       = n.d[top];
        const uint64_t    b       = m.d[top];
        unsigned __int128 product = 0U;

        for (uint32_t bit = 0U; bit < 64U; ++bit)
        {
            product ^= 0U != ((b >> bit) & 1U) ? (unsigned __int128)a << bit : 0U;
        }

        result.d[0] = (uint64_t)product;
        result.d[1] = (uint64_t)(product >> 64U);
        write_z(sve, instruction & 0x1FU, result);
        return SVE_OK;
    }

    if (0U == size || (0x1AU == (op & 0x3EU) && 2U == size))
    {
        return SVE_UNDEFINED;
    }

    const bool is_supported = op < 0x08U || (op >= 0x0CU && op < 0x20U) || 0x20U == op
                              || 0x22U == op || 0x23U == op || 0x30U == (op & 0x3CU);

    if (false == is_supported)
    {
        return SVE_UNDEFINED;
    }

    for (uint32_t e = 0U; e < elements_of(esize); ++e)
    {
        const uint64_t na = get_element(&n, 2U * e + top, half);
        const uint64_t mb = get_element(&m, 2U * e + top, half);
        __int128       value;

        if (op < 0x10U)
        {
            // SADDL, UADDL (000), SSUBL, USUBL (001) and SABDL, UABDL (011).
            const __int128 a    = wide(na, half, is_signed);
            const __int128 b    = wide(mb, half, is_signed);
            const uint32_t kind = op >> 2U;
            value = 0U == kind ? a + b : (1U == kind ? a - b : (a > b ? a - b : b - a));
        }
        else if (op < 0x18U)
        {
            // SADDW, UADDW, SSUBW and USUBW: the wide Zn element and a narrow Zm one.
            const __int128 a = wide(get_element(&n, e, esize), esize, is_signed);
            const __int128 b = wide(mb, half, is_signed);
            value            = 0U == (op & 0x4U) ? a + b : a - b;
        }
        else if (op < 0x1AU)
        {
            const __int128 product = wide(na, half, true) * wide(mb, half, true);
            value = wide(narrow_to(2 * product, esize, SATURATE_SIGNED), esize, true);
        }
        else if (op < 0x1CU)
        {
            uint64_t product = 0U;

            for (uint32_t bit = 0U; bit < half; ++bit)
            {
                product ^= 0U != ((mb >> bit) & 1U) ? na << bit : 0U;
            }

            value = (__int128)product;
        }
        else if (op < 0x20U)
        {
            value = wide(na, half, is_signed) * wide(mb, half, is_signed);
        }
        else if (op < 0x24U)
        {
            // SADDLBT (100000), SSUBLBT (100010) and SSUBLTB (100011): even Zn with odd Zm, or
            // odd Zn with even Zm.
            const uint32_t n_top = 0x23U == op ? 1U : 0U;
            const __int128 a     = wide(get_element(&n, 2U * e + n_top, half), half, true);
            const __int128 b     = wide(get_element(&m, 2U * e + (n_top ^ 1U), half), half, true);
            value                = 0x20U == op ? a + b : a - b;
        }
        else
        {
            // SABALB, SABALT, UABALB and UABALT.
            const __int128 a = wide(na, half, is_signed);
            const __int128 b = wide(mb, half, is_signed);
            value            = (__int128)get_element(&result, e, esize) + (a > b ? a - b : b - a);
        }

        set_element(&result, e, esize, narrow_to(value, esize, SATURATE_NONE));
    }

    write_z(sve, instruction & 0x1FU, result);
    return SVE_OK;
}

/// EORBT and EORTB: 01000101 size 0 Zm 10010 tb Zn Zd, exclusive or into the even or odd elements.
static sve_result_t
exclusive_or_interleaved(sve_t *sve, const uint32_t instruction)
{
    const uint32_t esize  = size_at_22(instruction);
    const uint32_t tb     = (instruction >> 10U) & 1U;
    const vector_t n      = read_z(sve, (instruction >> 5U) & 0x1FU);
    const vector_t m      = read_z(sve, (instruction >> 16U) & 0x1FU);
    vector_t       result = read_z(sve, instruction & 0x1FU);

    for (uint32_t pair = 0U; pair < elements_of(esize) / 2U; ++pair)
    {
        set_element(&result,
                    2U * pair + tb,
                    esize,
                    get_element(&n, 2U * pair + tb, esize)
                        ^ get_element(&m, 2U * pair + (tb ^ 1U), esize));
    }

    write_z(sve, instruction & 0x1FU, result);
    return SVE_OK;
}

/// SMMLA, USMMLA and UMMLA: 01000101 uns 0 Zm 100110 Zn Zda, a 2x8 by 8x2 byte matrix product.
static sve_result_t
matrix_multiply_add_integer(sve_t *sve, const uint32_t instruction)
{
    const uint32_t uns = (instruction >> 22U) & 0x3U;

    if (1U == uns)
    {
        return SVE_UNDEFINED;
    }

    const bool     n_signed = 0U == uns;
    const bool     m_signed = 0U == uns || 2U == uns;
    const vector_t n        = read_z(sve, (instruction >> 5U) & 0x1FU);
    const vector_t m        = read_z(sve, (instruction >> 16U) & 0x1FU);
    vector_t       result   = read_z(sve, instruction & 0x1FU);

    for (uint32_t i = 0U; i < 2U; ++i)
    {
        for (uint32_t j = 0U; j < 2U; ++j)
        {
            int64_t sum = 0;

            for (uint32_t k = 0U; k < 8U; ++k)
            {
                sum += (int64_t)wide(get_element(&n, 8U * i + k, 8U), 8U, n_signed)
                       * (int64_t)wide(get_element(&m, 8U * j + k, 8U), 8U, m_signed);
            }

            set_element(
                &result, 2U * i + j, 32U, get_element(&result, 2U * i + j, 32U) + (uint64_t)sum);
        }
    }

    write_z(sve, instruction & 0x1FU, result);
    return SVE_OK;
}

/// SSHLLB, SSHLLT, USHLLB and USHLLT: 01000101 0 tszh 0 tszl imm3 1010 U T Zn Zd.
static sve_result_t
shift_left_long(sve_t *sve, const uint32_t instruction)
{
    const uint32_t tsz = (((instruction >> 22U) & 1U) << 2U) | ((instruction >> 19U) & 0x3U);

    if (0U == tsz)
    {
        return SVE_UNDEFINED;
    }

    const uint32_t half      = 8U << (31U - (uint32_t)__builtin_clz(tsz));
    const uint32_t esize     = 2U * half;
    const uint32_t shift     = ((tsz << 3U) | ((instruction >> 16U) & 0x7U)) - half;
    const uint32_t top       = (instruction >> 10U) & 1U;
    const bool     is_signed = 0U == (instruction & 0x800U);
    const vector_t n         = read_z(sve, (instruction >> 5U) & 0x1FU);
    vector_t       result    = { { 0U, 0U } };

    for (uint32_t e = 0U; e < elements_of(esize); ++e)
    {
        const __int128 value = wide(get_element(&n, 2U * e + top, half), half, is_signed);
        set_element(
            &result, e, esize, narrow_to(value * ((__int128)1 << shift), esize, SATURATE_NONE));
    }

    write_z(sve, instruction & 0x1FU, result);
    return SVE_OK;
}

/// BEXT, BDEP and BGRP: 01000101 size 0 Zm 1011 opc Zn Zd.
static sve_result_t
bit_permute(sve_t *sve, const uint32_t instruction)
{
    const uint32_t esize = size_at_22(instruction);
    const uint32_t opc   = (instruction >> 10U) & 0x3U;

    if (3U == opc)
    {
        return SVE_UNDEFINED;
    }

    const vector_t n      = read_z(sve, (instruction >> 5U) & 0x1FU);
    const vector_t m      = read_z(sve, (instruction >> 16U) & 0x1FU);
    vector_t       result = { { 0U, 0U } };

    for (uint32_t e = 0U; e < elements_of(esize); ++e)
    {
        const uint64_t data       = get_element(&n, e, esize);
        const uint64_t mask       = get_element(&m, e, esize);
        uint64_t       low        = 0U;
        uint64_t       high       = 0U;
        uint32_t       low_count  = 0U;
        uint32_t       high_count = 0U;

        for (uint32_t bit = 0U; bit < esize; ++bit)
        {
            if (0U != ((mask >> bit) & 1U))
            {
                // BEXT and BGRP gather the selected bits, BDEP scatters the low bits to them.
                const bool is_deposit = 1U == opc;
                low |= (false == is_deposit ? (data >> bit) & 1U : (data >> low_count) & 1U)
                       << (false == is_deposit ? low_count : bit);
                ++low_count;
            }
            else
            {
                high |= ((data >> bit) & 1U) << high_count;
                ++high_count;
            }
        }

        uint64_t value = low;

        if (2U == opc)
        {
            // BGRP: the selected bits, then the others above them.
            value = low | (low_count < 64U ? high << low_count : 0U);
        }

        set_element(&result, e, esize, value & mask_of(esize));
    }

    write_z(sve, instruction & 0x1FU, result);
    return SVE_OK;
}

/// SABA and UABA: 01000101 size 0 Zm 11111 U Zn Zda.
static sve_result_t
absolute_difference_accumulate(sve_t *sve, const uint32_t instruction)
{
    const uint32_t esize     = size_at_22(instruction);
    const bool     is_signed = 0U == (instruction & 0x400U);
    const vector_t n         = read_z(sve, (instruction >> 5U) & 0x1FU);
    const vector_t m         = read_z(sve, (instruction >> 16U) & 0x1FU);
    vector_t       result    = read_z(sve, instruction & 0x1FU);

    for (uint32_t e = 0U; e < elements_of(esize); ++e)
    {
        set_element(
            &result,
            e,
            esize,
            get_element(&result, e, esize)
                + absolute_difference(
                    get_element(&n, e, esize), get_element(&m, e, esize), esize, is_signed));
    }

    write_z(sve, instruction & 0x1FU, result);
    return SVE_OK;
}

/// SSRA, USRA, SRSRA and URSRA (01000101 tszh 0 tszl imm3 1110 R U Zn Zda) and SRI and SLI
/// (... 11110 op ...).
static sve_result_t
shift_accumulate_insert(sve_t *sve, const uint32_t instruction)
{
    const uint32_t tsz = (((instruction >> 22U) & 0x3U) << 2U) | ((instruction >> 19U) & 0x3U);
    const uint32_t imm = (tsz << 3U) | ((instruction >> 16U) & 0x7U);
    const bool     is_insert = 0xF000U == (instruction & 0xF800U);
    const bool     is_left   = true == is_insert && 0U != (instruction & 0x400U);
    uint32_t       shift     = 0U;
    uint32_t       esize     = right_shift_immediate(tsz, (instruction >> 16U) & 0x7U, &shift);

    if (0U == esize)
    {
        return SVE_UNDEFINED;
    }

    if (true == is_left)
    {
        shift = imm - esize;
    }

    const bool     is_unsigned = false == is_insert && 0U != (instruction & 0x400U);
    const bool     is_rounding = false == is_insert && 0U != (instruction & 0x800U);
    const vector_t n           = read_z(sve, (instruction >> 5U) & 0x1FU);
    vector_t       result      = read_z(sve, instruction & 0x1FU);

    for (uint32_t e = 0U; e < elements_of(esize); ++e)
    {
        const uint64_t a   = get_element(&n, e, esize);
        const uint64_t acc = get_element(&result, e, esize);
        uint64_t       value;

        if (true == is_insert)
        {
            // SRI keeps the top `shift` bits of the destination; SLI the bottom ones.
            const uint64_t keep
                = true == is_left ? mask_of(shift)
                                  : (shift >= esize ? mask_of(esize) : ~(mask_of(esize) >> shift));
            const uint64_t moved
                = true == is_left ? a << shift : (shift >= esize ? 0U : a >> shift);
            value = (acc & keep) | (moved & ~keep);
        }
        else
        {
            const __int128 shifted = shift_wide(
                wide(a, esize, false == is_unsigned), -(int64_t)shift, esize, is_rounding);
            value = acc + (uint64_t)shifted;
        }

        set_element(&result, e, esize, value & mask_of(esize));
    }

    write_z(sve, instruction & 0x1FU, result);
    return SVE_OK;
}

/// ADCLB, ADCLT (01000101 0 sz 0 Zm 11010 T Zn Zda) and SBCLB, SBCLT (01000101 1 sz ...): add
/// with the carry in bit 0 of the odd elements of Zm, the carry out going there in the result.
static sve_result_t
add_carry_long(sve_t *sve, const uint32_t instruction)
{
    const uint32_t esize       = 0U != (instruction & 0x400000U) ? 64U : 32U;
    const bool     is_subtract = 0U != (instruction & 0x800000U);
    const uint32_t top         = (instruction >> 10U) & 1U;
    const vector_t n           = read_z(sve, (instruction >> 5U) & 0x1FU);
    const vector_t m           = read_z(sve, (instruction >> 16U) & 0x1FU);
    vector_t       result      = read_z(sve, instruction & 0x1FU);

    for (uint32_t pair = 0U; pair < elements_of(esize) / 2U; ++pair)
    {
        uint64_t       a            = get_element(&n, 2U * pair + top, esize);
        const uint64_t acc          = get_element(&result, 2U * pair, esize);
        const uint64_t carry        = get_element(&m, 2U * pair + 1U, esize) & 1U;
        a                           = true == is_subtract ? ~a & mask_of(esize) : a;
        const unsigned __int128 sum = (unsigned __int128)acc + a + carry;
        set_element(&result, 2U * pair, esize, (uint64_t)sum & mask_of(esize));
        set_element(&result, 2U * pair + 1U, esize, (uint64_t)(sum >> esize) & 1U);
    }

    write_z(sve, instruction & 0x1FU, result);
    return SVE_OK;
}

/// CADD and SQCADD: 01000101 size 0000 op 11011 rot Zm Zdn, adding Zm rotated by 90 or 270
/// degrees.
static sve_result_t
complex_integer_add(sve_t *sve, const uint32_t instruction)
{
    const uint32_t   esize  = size_at_22(instruction);
    const bool       is_270 = 0U != (instruction & 0x400U);
    const saturate_t mode   = 0U != (instruction & 0x10000U) ? SATURATE_SIGNED : SATURATE_NONE;
    const vector_t   m      = read_z(sve, (instruction >> 5U) & 0x1FU);
    vector_t         result = read_z(sve, instruction & 0x1FU);

    for (uint32_t pair = 0U; pair < elements_of(esize) / 2U; ++pair)
    {
        const __int128 real      = wide(get_element(&result, 2U * pair, esize), esize, true);
        const __int128 imaginary = wide(get_element(&result, 2U * pair + 1U, esize), esize, true);
        const __int128 m_real    = wide(get_element(&m, 2U * pair, esize), esize, true);
        const __int128 m_imag    = wide(get_element(&m, 2U * pair + 1U, esize), esize, true);
        set_element(&result,
                    2U * pair,
                    esize,
                    narrow_to(true == is_270 ? real + m_imag : real - m_imag, esize, mode));
        set_element(
            &result,
            2U * pair + 1U,
            esize,
            narrow_to(true == is_270 ? imaginary - m_real : imaginary + m_real, esize, mode));
    }

    write_z(sve, instruction & 0x1FU, result);
    return SVE_OK;
}

/// Writes `value` to the narrow element `e` of `result`: the bottom form (`top` 0) also clears
/// the odd elements, the top form leaves the even ones.
static void
set_narrow(vector_t      *result,
           const uint32_t e,
           const uint32_t half,
           const uint32_t top,
           const uint64_t value)
{
    set_element(result, 2U * e + top, half, value);

    if (0U == top)
    {
        set_element(result, 2U * e + 1U, half, 0U);
    }
}

/// The shift right narrow group: 01000101 0 tszh 1 tszl imm3 00 op U R T Zn Zd, with SQSHRUN,
/// SQRSHRUN (000 R), SHRN, RSHRN (001 R), SQSHRN, SQRSHRN (010 R) and UQSHRN, UQRSHRN (011 R).
static sve_result_t
shift_right_narrow(sve_t *sve, const uint32_t instruction)
{
    const uint32_t tsz   = (((instruction >> 22U) & 1U) << 2U) | ((instruction >> 19U) & 0x3U);
    uint32_t       shift = 0U;
    const uint32_t half  = right_shift_immediate(tsz, (instruction >> 16U) & 0x7U, &shift);

    if (0U == half)
    {
        return SVE_UNDEFINED;
    }

    const uint32_t   esize       = 2U * half;
    const uint32_t   op          = (instruction >> 11U) & 0x7U;
    const uint32_t   top         = (instruction >> 10U) & 1U;
    const bool       is_rounding = 0U != (op & 1U);
    const uint32_t   kind        = op >> 1U;
    const bool       is_signed   = 1U != kind && 3U != kind;
    const saturate_t mode
        = 0U == kind
              ? SATURATE_UNSIGNED
              : (1U == kind ? SATURATE_NONE : (2U == kind ? SATURATE_SIGNED : SATURATE_UNSIGNED));
    const vector_t n = read_z(sve, (instruction >> 5U) & 0x1FU);
    vector_t result  = 0U == top ? (vector_t) { { 0U, 0U } } : read_z(sve, instruction & 0x1FU);

    for (uint32_t e = 0U; e < elements_of(esize); ++e)
    {
        const __int128 value = shift_wide(
            wide(get_element(&n, e, esize), esize, is_signed), -(int64_t)shift, esize, is_rounding);
        set_narrow(&result, e, half, top, narrow_to(value, half, mode));
    }

    write_z(sve, instruction & 0x1FU, result);
    return SVE_OK;
}

/// SQXTNB, SQXTNT, UQXTNB, UQXTNT, SQXTUNB and SQXTUNT: 01000101 0 tszh 1 tszl 000 010 opc T Zn Zd.
static sve_result_t
extract_narrow(sve_t *sve, const uint32_t instruction)
{
    const uint32_t tsz = (((instruction >> 22U) & 1U) << 2U) | ((instruction >> 19U) & 0x3U);
    const uint32_t opc = (instruction >> 11U) & 0x3U;

    // Exactly one size bit is set.
    if (0U == tsz || 0U != (tsz & (tsz - 1U)) || 3U == opc)
    {
        return SVE_UNDEFINED;
    }

    const uint32_t   half      = 8U << (31U - (uint32_t)__builtin_clz(tsz));
    const uint32_t   esize     = 2U * half;
    const uint32_t   top       = (instruction >> 10U) & 1U;
    const bool       is_signed = 1U != opc;
    const saturate_t mode      = 0U == opc ? SATURATE_SIGNED : SATURATE_UNSIGNED;
    const vector_t   n         = read_z(sve, (instruction >> 5U) & 0x1FU);
    vector_t result = 0U == top ? (vector_t) { { 0U, 0U } } : read_z(sve, instruction & 0x1FU);

    for (uint32_t e = 0U; e < elements_of(esize); ++e)
    {
        set_narrow(&result,
                   e,
                   half,
                   top,
                   narrow_to(wide(get_element(&n, e, esize), esize, is_signed), half, mode));
    }

    write_z(sve, instruction & 0x1FU, result);
    return SVE_OK;
}

/// ADDHNB, ADDHNT, RADDHNB, RADDHNT, SUBHNB, SUBHNT, RSUBHNB and RSUBHNT: 01000101 size 1 Zm 011
/// S R T Zn Zd, the high halves of the wide sums or differences.
static sve_result_t
add_high_narrow(sve_t *sve, const uint32_t instruction)
{
    const uint32_t size = (instruction >> 22U) & 0x3U;

    if (0U == size)
    {
        return SVE_UNDEFINED;
    }

    const uint32_t esize       = 8U << size;
    const uint32_t half        = esize / 2U;
    const bool     is_subtract = 0U != (instruction & 0x1000U);
    const bool     is_rounding = 0U != (instruction & 0x800U);
    const uint32_t top         = (instruction >> 10U) & 1U;
    const vector_t n           = read_z(sve, (instruction >> 5U) & 0x1FU);
    const vector_t m           = read_z(sve, (instruction >> 16U) & 0x1FU);
    vector_t result = 0U == top ? (vector_t) { { 0U, 0U } } : read_z(sve, instruction & 0x1FU);

    for (uint32_t e = 0U; e < elements_of(esize); ++e)
    {
        const __int128 a   = wide(get_element(&n, e, esize), esize, false);
        const __int128 b   = wide(get_element(&m, e, esize), esize, false);
        __int128       sum = true == is_subtract ? a - b : a + b;
        sum += true == is_rounding ? (__int128)1 << (half - 1U) : 0;
        set_narrow(&result, e, half, top, narrow_to(sum >> half, half, SATURATE_NONE));
    }

    write_z(sve, instruction & 0x1FU, result);
    return SVE_OK;
}

/// MATCH and NMATCH: 01000101 size 1 Zm 100 Pg Zn op Pd, whether each active element of Zn is
/// in its segment of Zm, setting the flags.
static sve_result_t
match_elements(sve_t *sve, const uint32_t instruction)
{
    const uint32_t esize = size_at_22(instruction);

    if (esize > 16U)
    {
        return SVE_UNDEFINED;
    }

    const bool     is_negated = 0U != (instruction & 0x10U);
    const uint16_t pg         = read_p(sve, (instruction >> 10U) & 0x7U);
    const vector_t n          = read_z(sve, (instruction >> 5U) & 0x1FU);
    const vector_t m          = read_z(sve, (instruction >> 16U) & 0x1FU);
    uint16_t       pd         = 0U;

    for (uint32_t e = 0U; e < elements_of(esize); ++e)
    {
        if (false == is_active(pg, e, esize))
        {
            continue;
        }

        bool found = false;

        for (uint32_t j = 0U; j < elements_of(esize); ++j)
        {
            found = found || get_element(&n, e, esize) == get_element(&m, j, esize);
        }

        pd = set_active(pd, e, esize, found != is_negated);
    }

    sve->cpu->p[instruction & 0xFU] = pd;
    write_nzcv(sve, predicate_test(pg, pd, esize));
    return SVE_OK;
}

/// HISTSEG (01000101 001 Zm 101000 Zn Zd): for each byte, how many bytes of its segment of Zm
/// equal it.
static sve_result_t
histogram_segment(sve_t *sve, const uint32_t instruction)
{
    if (0U != (instruction & 0xC00000U))
    {
        return SVE_UNDEFINED;
    }

    const vector_t n      = read_z(sve, (instruction >> 5U) & 0x1FU);
    const vector_t m      = read_z(sve, (instruction >> 16U) & 0x1FU);
    vector_t       result = { { 0U, 0U } };

    for (uint32_t e = 0U; e < 16U; ++e)
    {
        uint64_t count = 0U;

        for (uint32_t j = 0U; j < 16U; ++j)
        {
            count += get_element(&n, e, 8U) == get_element(&m, j, 8U) ? 1U : 0U;
        }

        set_element(&result, e, 8U, count);
    }

    write_z(sve, instruction & 0x1FU, result);
    return SVE_OK;
}

/// HISTCNT (01000101 size 1 Zm 110 Pg Zn Zd): for each active element, how many active elements
/// of Zm up to it equal it; inactive elements are zero.
static sve_result_t
histogram_count(sve_t *sve, const uint32_t instruction)
{
    const uint32_t esize = size_at_22(instruction);

    if (esize < 32U)
    {
        return SVE_UNDEFINED;
    }

    const uint16_t pg     = read_p(sve, (instruction >> 10U) & 0x7U);
    const vector_t n      = read_z(sve, (instruction >> 5U) & 0x1FU);
    const vector_t m      = read_z(sve, (instruction >> 16U) & 0x1FU);
    vector_t       result = { { 0U, 0U } };

    for (uint32_t e = 0U; e < elements_of(esize); ++e)
    {
        if (false == is_active(pg, e, esize))
        {
            continue;
        }

        uint64_t count = 0U;

        for (uint32_t j = 0U; j <= e; ++j)
        {
            count += true == is_active(pg, j, esize)
                             && get_element(&n, e, esize) == get_element(&m, j, esize)
                         ? 1U
                         : 0U;
        }

        set_element(&result, e, esize, count);
    }

    write_z(sve, instruction & 0x1FU, result);
    return SVE_OK;
}

/// AESE, AESD, AESMC, AESIMC, SM4E, SM4EKEY and RAX1: with 128-bit vectors these are their
/// Advanced SIMD forms on the same registers.
static sve_result_t
crypto(sve_t *sve, const uint32_t instruction)
{
    const uint32_t d = instruction & 0x1FU;
    const uint32_t n = (instruction >> 5U) & 0x1FU;
    const uint32_t m = (instruction >> 16U) & 0x1FU;
    uint32_t       simd;

    switch (instruction & 0xFFFFFC00U)
    {
        case 0x4522E000U: // AESE Zdn, Zdn, Zm
            simd = 0x4E284800U | (n << 5U) | d;
            break;
        case 0x4522E400U: // AESD
            simd = 0x4E285800U | (n << 5U) | d;
            break;
        case 0x4523E000U: // SM4E
            simd = 0xCEC08400U | (n << 5U) | d;
            break;
        default:
            switch (instruction & 0xFFE0FC00U)
            {
                case 0x4520E000U: // AESMC Zdn
                case 0x4520E400U: // AESIMC
                    if (0U != (instruction & 0x1F03E0U))
                    {
                        return SVE_UNDEFINED;
                    }

                    simd = (0x4520E000U == (instruction & 0xFFE0FC00U) ? 0x4E286800U : 0x4E287800U)
                           | (d << 5U) | d;
                    break;
                case 0x4520F000U: // SM4EKEY
                    simd = 0xCE60C800U | (m << 16U) | (n << 5U) | d;
                    break;
                case 0x4520F400U: // RAX1
                    simd = 0xCE608C00U | (m << 16U) | (n << 5U) | d;
                    break;
                default:
                    return SVE_UNDEFINED;
            }
            break;
    }

    (void)bal_execute_simd(sve->cpu, simd);
    return SVE_OK;
}

// -----------------------------------------------------------------------------
// Decoding
// -----------------------------------------------------------------------------

static const sve_pattern_t PATTERNS[] = {
    { 0xFF3EE000U, 0x04068000U, shift_immediate_saturating },
    { 0xFF3EE000U, 0x040C8000U, shift_immediate_saturating },
    { 0xFF3FE000U, 0x040F8000U, shift_immediate_saturating },
    { 0xFF20F800U, 0x04207000U, doubling_multiply_high_vectors },
    { 0xFF20FC00U, 0x04203400U, exclusive_or_rotate },
    { 0xFF20FC00U, 0x04203800U, bitwise_ternary },
    { 0xFF20FC00U, 0x04203C00U, bitwise_ternary },
    { 0xFF20FC00U, 0x05203000U, table_lookup },
    { 0xFF20E000U, 0x04200000U, integer_add_sub_vectors },
    { 0xFF20E000U, 0x04000000U, integer_binary_predicated },
    { 0xFF20E000U, 0x04002000U, integer_reduction },
    { 0xFF30E000U, 0x04008000U, shift_immediate_predicated },
    { 0xFF30E000U, 0x04108000U, shift_vector_predicated },
    { 0xFF30E000U, 0x0410A000U, integer_unary_predicated },
    { 0xFF20C000U, 0x04004000U, multiply_add_predicated },
    { 0xFF20C000U, 0x0400C000U, multiply_add_predicated },
    { 0xFF20FC00U, 0x04203000U, bitwise_unpredicated },
    { 0xFF20E000U, 0x04208000U, shift_unpredicated },
    { 0xFF20F000U, 0x04206000U, multiply_unpredicated },
    { 0xFF20F000U, 0x04204000U, index_vector },
    { 0xFF20F000U, 0x0420A000U, address_vector },
    { 0xFFFFFC00U, 0x0420BC00U, move_prefix },
    { 0xFF3FFC00U, 0x05203800U, duplicate_scalar },
    { 0xFF3EC000U, 0x2538C000U, duplicate_immediate },
    { 0xFF20FC00U, 0x05202000U, duplicate_indexed },
    { 0xFF2FFC00U, 0x05243800U, insert_element },
    { 0xFF3C0000U, 0x05000000U, bitwise_immediate },
    { 0xFF300000U, 0x05100000U, copy_immediate },
    { 0xFF3FE000U, 0x0528A000U, copy_scalar },
    { 0xFF3FE000U, 0x05208000U, copy_scalar },
    { 0xFF20C000U, 0x0520C000U, select_vectors },
    { 0xFF20C000U, 0x2520C000U, integer_immediate },
    { 0xFF20E000U, 0x05206000U, permute_vectors },
    { 0xFF30E210U, 0x05204000U, permute_predicates },
    { 0xFFFEFE10U, 0x05304000U, unpack_predicate },
    { 0xFF3FFE10U, 0x05344000U, reverse_elements },
    { 0xFF3FFC00U, 0x05383800U, reverse_elements },
    { 0xFF3CFC00U, 0x05303800U, unpack_vector },
    { 0xFF3CE000U, 0x05248000U, reverse_within_elements },
    { 0xFF20F000U, 0x05202000U, table_lookup },
    { 0xFFA0E000U, 0x05200000U, extract_vector },
    { 0xFF3EE000U, 0x052C8000U, splice_vectors },
    { 0xFF3FE000U, 0x05218000U, compact_vector },
    { 0xFF20C000U, 0x05208000U, last_element },
    { 0xFF3EFC10U, 0x2518E000U, predicate_true },
    { 0xFFFFFFF0U, 0x2518E400U, predicate_false },
    { 0xFFFFFFF0U, 0x2519F000U, read_ffr },
    { 0xFFBFFE10U, 0x2518F000U, read_ffr_predicated },
    { 0xFFFFFFFFU, 0x252C9000U, set_ffr },
    { 0xFFFFFE1FU, 0x25289000U, write_ffr },
    { 0xFFFFF800U, 0x04BF5000U, read_vector_length },
    { 0xFFA0F800U, 0x04205000U, add_vector_length },
    { 0xFF30FC00U, 0x0420E000U, count_elements },
    { 0xFF30F800U, 0x0430E000U, count_update },
    { 0xFF30F800U, 0x0430C000U, count_update },
    { 0xFF20F000U, 0x0420F000U, count_update },
    { 0xFF30F000U, 0x0420C000U, count_update },
    { 0xFF30C000U, 0x25004000U, predicate_logical },
    { 0xFF37C200U, 0x25104000U, predicate_break },
    { 0xFFB0C200U, 0x2500C000U, predicate_break_propagate },
    { 0xFFFFC21FU, 0x2550C000U, predicate_test_instruction },
    { 0xFFFFFE10U, 0x2558C000U, predicate_first },
    { 0xFF3FFE10U, 0x2519C400U, predicate_next },
    { 0xFF3FC200U, 0x25208000U, count_predicate },
    { 0xFF38F000U, 0x25288000U, count_predicate_update },
    { 0xFF20E000U, 0x25200000U, predicate_while },
    { 0xFF20FC00U, 0x25203000U, predicate_while_conflict },
    { 0xFFA0FC0FU, 0x25A02000U, compare_terminate },
    { 0xFF200000U, 0x24000000U, compare_vectors },
    { 0xFF204000U, 0x25000000U, compare_immediate },
    { 0xFF200000U, 0x24200000U, compare_immediate },
    { 0xFFC0A000U, 0x85800000U, load_store_register },
    { 0xFFC0A000U, 0xE5800000U, load_store_register },
    { 0xFE00E000U, 0xA400A000U, load_contiguous_dtype },
    { 0xFE00C000U, 0xA4004000U, load_contiguous_dtype },
    { 0xFE00E000U, 0xA400E000U, load_structures },
    { 0xFE00E000U, 0xA400C000U, load_structures },
    { 0xFE60C000U, 0xA4000000U, load_replicate_quadword },
    { 0xFE408000U, 0x84408000U, load_broadcast },
    { 0xFE00E000U, 0xE400E000U, store_contiguous_instruction },
    { 0xFE00C000U, 0xE4004000U, store_contiguous_instruction },
    { 0xFE60E010U, 0x8400E000U, prefetch },
    { 0xFE60E010U, 0xC400E000U, prefetch },
    { 0xFFC08010U, 0x85C00000U, prefetch },
    { 0xFE60E010U, 0x8400C000U, prefetch },
    { 0xFFA08010U, 0x84200000U, prefetch },
    { 0xFFA08010U, 0xC4200000U, prefetch },
    { 0xFFE08010U, 0xC4608000U, prefetch },
    { 0xFE000000U, 0x84000000U, load_gather },
    { 0xFE000000U, 0xC4000000U, load_gather },
    { 0xFE000000U, 0xE4000000U, store_scatter },
    { 0xFF20E000U, 0x65000000U, float_arithmetic_unpredicated },
    { 0xFF30E000U, 0x65008000U, float_arithmetic_predicated },
    { 0xFF38E3C0U, 0x65188000U, float_immediate },
    { 0xFF38FC00U, 0x65108000U, float_trig_multiply_add },
    { 0xFF200000U, 0x65200000U, float_multiply_add },
    { 0xFF3CE000U, 0x65102000U, float_compare_vectors },
    { 0xFF204000U, 0x65004000U, float_compare_vectors },
    { 0xFF38E000U, 0x65002000U, float_reduction },
    { 0xFF3FE000U, 0x65182000U, float_add_ordered },
    { 0xFF38E000U, 0x64108000U, float_pairwise },
    { 0xFF38E000U, 0x6500A000U, float_unary },
    { 0xFF3EE000U, 0x650CA000U, float_unary },
    { 0xFFF9E000U, 0x6518A000U, float_unary },
    { 0xFF20E000U, 0x6500A000U, float_convert },
    { 0xFF3CE000U, 0x6408A000U, float_convert_top },
    { 0xFF3EFC00U, 0x650E3000U, float_estimate },
    { 0xFF3FFC00U, 0x0420B800U, float_exponential },
    { 0xFF20FC00U, 0x0420B000U, float_trig_select },
    { 0xFF3EE000U, 0x64008000U, float_complex_add },
    { 0xFF208000U, 0x64000000U, float_complex_multiply_add },
    { 0xFFA0F000U, 0x64A01000U, float_complex_multiply_add_indexed },
    { 0xFFA0F800U, 0x64200000U, float_multiply_indexed },
    { 0xFFA0F800U, 0x64A00000U, float_multiply_indexed },
    { 0xFFA0FC00U, 0x64202000U, float_multiply_indexed },
    { 0xFFA0FC00U, 0x64A02000U, float_multiply_indexed },
    { 0xFFA0D800U, 0x64A08000U, float_widening_multiply_add },
    { 0xFFA0D000U, 0x64A04000U, float_widening_multiply_add },
    { 0xFFE0FC00U, 0x64608000U, bfloat_dot },
    { 0xFFE0FC00U, 0x64604000U, bfloat_dot },
    { 0xFFE0FC00U, 0x6460E400U, matrix_multiply_add_float },
    { 0xFFE0FC00U, 0x64A0E400U, matrix_multiply_add_float },
    { 0xFF20C000U, 0x44008000U, sve2_integer_predicated },
    { 0xFF20F800U, 0x44000000U, integer_dot_product },
    { 0xFFE0FC00U, 0x44807800U, integer_dot_product },
    { 0xFF20F000U, 0x44001000U, complex_dot_product },
    { 0xFF20E000U, 0x44002000U, complex_integer_multiply_add },
    { 0xFF208000U, 0x44000000U, integer_multiply_add_vectors },
    { 0xFFA0F800U, 0x44A00000U, integer_dot_product },
    { 0xFFE0F800U, 0x44A01800U, integer_dot_product },
    { 0xFFA0F800U, 0x44200800U, integer_multiply_indexed },
    { 0xFFA0F800U, 0x44A00800U, integer_multiply_indexed },
    { 0xFFA0F800U, 0x44201000U, integer_multiply_indexed },
    { 0xFFA0F800U, 0x44A01000U, integer_multiply_indexed },
    { 0xFFA0F800U, 0x4420F000U, integer_multiply_indexed },
    { 0xFFA0F800U, 0x44A0F000U, integer_multiply_indexed },
    { 0xFFA0FC00U, 0x4420F800U, integer_multiply_indexed },
    { 0xFFA0FC00U, 0x44A0F800U, integer_multiply_indexed },
    { 0xFFA0F000U, 0x44A04000U, complex_dot_product },
    { 0xFFA0E000U, 0x44A06000U, complex_integer_multiply_add },
    { 0xFFA00000U, 0x44A00000U, integer_multiply_long_indexed },
    { 0xFF20F800U, 0x45009000U, exclusive_or_interleaved },
    { 0xFF20FC00U, 0x45009800U, matrix_multiply_add_integer },
    { 0xFFA0F000U, 0x4500A000U, shift_left_long },
    { 0xFF20F000U, 0x4500B000U, bit_permute },
    { 0xFF20F000U, 0x4500C000U, integer_widening },
    { 0xFF20F800U, 0x4500D000U, add_carry_long },
    { 0xFF3EF800U, 0x4500D800U, complex_integer_add },
    { 0xFF20F000U, 0x4500E000U, shift_accumulate_insert },
    { 0xFF20F800U, 0x4500F000U, shift_accumulate_insert },
    { 0xFF20F800U, 0x4500F800U, absolute_difference_accumulate },
    { 0xFF208000U, 0x45000000U, integer_widening },
    { 0xFF20F000U, 0x45008000U, integer_widening },
    { 0xFFA0C000U, 0x45200000U, shift_right_narrow },
    { 0xFFA7E000U, 0x45204000U, extract_narrow },
    { 0xFF20E000U, 0x45206000U, add_high_narrow },
    { 0xFF20E000U, 0x45208000U, match_elements },
    { 0xFF20FC00U, 0x4520A000U, histogram_segment },
    { 0xFF20E000U, 0x4520C000U, histogram_count },
    { 0xFFE0E000U, 0x4520E000U, crypto },
};

static sve_result_t
execute(sve_t *sve, const uint32_t instruction)
{
    // op0 = 0000 with bit 31 set is the SME encoding space.
    if (0x80000000U == (instruction & 0x9E000000U))
    {
        return sme_execute(sve, instruction);
    }

    // Some of the patterns below also match encodings SVE2.1 allocates.
    const sve_result_t result = sve2p1_execute(sve, instruction);

    if (SVE_UNDEFINED != result)
    {
        return result;
    }

    for (size_t i = 0U; i < sizeof(PATTERNS) / sizeof(PATTERNS[0]); ++i)
    {
        if (PATTERNS[i].value == (instruction & PATTERNS[i].mask))
        {
            return PATTERNS[i].handler(sve, instruction);
        }
    }

    return SVE_UNDEFINED;
}

/// A memory interface for checking support, which maps every address to scratch memory.
static const uint8_t *
scratch_translate(void *context, const bal_guest_address_t address, size_t *available_bytes)
{
    (void)context;
    static BAL_THREAD_LOCAL uint8_t scratch[8192];
    const size_t                    offset = (size_t)(address & 0xFFFU);
    *available_bytes                       = sizeof(scratch) - offset;
    return &scratch[offset];
}

bool
bal_is_sve_supported(const uint32_t instruction)
{
    bal_cpu_t cpu;
    (void)memset(&cpu, 0, sizeof(cpu));
    bal_memory_interface_t memory;
    (void)memset(&memory, 0, sizeof(memory));
    memory.translate = scratch_translate;
    // With streaming mode and ZA on, so that SME instructions run instead of being illegal.
    cpu.svcr  = SVCR_SM | SVCR_ZA;
    sve_t sve = { .cpu = &cpu, .memory = &memory, .flags = 0U };
    return execute(&sve, instruction) != SVE_UNDEFINED;
}

uint64_t
bal_execute_sve(bal_cpu_t                    *cpu,
                const bal_memory_interface_t *memory_interface,
                const uint32_t                instruction)
{
    sve_t              sve    = { .cpu = cpu, .memory = memory_interface, .flags = 0U };
    const sve_result_t result = execute(&sve, instruction);
    cpu->fpsr |= sve.flags;

    if (SVE_FAULT == result)
    {
        cpu->exception = BAL_EXCEPTION_DATA_ABORT;
        return 0U;
    }

    if (SVE_ILLEGAL == result)
    {
        cpu->exception           = BAL_EXCEPTION_UNDEFINED_INSTRUCTION;
        cpu->exception_immediate = 0U;
        return 0U;
    }

    return 1U;
}

/*** end of file ***/
