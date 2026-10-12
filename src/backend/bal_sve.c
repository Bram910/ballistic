#include "backend/bal_sve.h"

#include "bal_attributes.h"
#include <string.h>

/// The outcome of executing an instruction.
typedef enum
{
    SVE_OK,
    /// The encoding is unallocated or not implemented; nothing was changed.
    SVE_UNDEFINED,
    /// An access was not mapped.
    SVE_FAULT,
} sve_result_t;

/// A Z register: d[0] holds bits 63:0 and d[1] bits 127:64.
typedef struct
{
    uint64_t d[2];
} vector_t;

/// The state one instruction works on.
typedef struct
{
    bal_cpu_t                    *cpu;
    const bal_memory_interface_t *memory;
    /// FPSR flags raised so far.
    uint64_t flags;
} sve_t;

typedef sve_result_t (*sve_handler_t)(sve_t *sve, uint32_t instruction);

typedef struct
{
    uint32_t      mask;
    uint32_t      value;
    sve_handler_t handler;
} sve_pattern_t;

#define VECTOR_BITS 128U
#define FPSR_QC     (1ULL << 27U)

// -----------------------------------------------------------------------------
// Elements and predicates
// -----------------------------------------------------------------------------

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
read_z(const sve_t *sve, const uint32_t index)
{
    vector_t vector = { { sve->cpu->v[index][0], sve->cpu->v[index][1] } };
    return vector;
}

static void
write_z(sve_t *sve, const uint32_t index, const vector_t vector)
{
    sve->cpu->v[index][0] = vector.d[0];
    sve->cpu->v[index][1] = vector.d[1];
}

/// Whether element `index` of `esize` bits is active in `predicate`: its lowest bit.
static bool
is_active(const uint16_t predicate, const uint32_t index, const uint32_t esize)
{
    return 0U != ((predicate >> (index * (esize / 8U))) & 1U);
}

/// A predicate with element `index` set as `value` and its other bits for that element clear.
static uint16_t
set_active(uint16_t predicate, const uint32_t index, const uint32_t esize, const bool value)
{
    const uint32_t bits = esize / 8U;
    const uint32_t mask = ((1U << bits) - 1U) << (index * bits);
    predicate           = (uint16_t)(predicate & ~mask);
    return (uint16_t)(predicate | (true == value ? 1U << (index * bits) : 0U));
}

static uint32_t
elements_of(const uint32_t esize)
{
    return VECTOR_BITS / esize;
}

static void
write_nzcv(sve_t *sve, const uint32_t nzcv)
{
    sve->cpu->flag_n = (uint8_t)((nzcv >> 3U) & 1U);
    sve->cpu->flag_z = (uint8_t)((nzcv >> 2U) & 1U);
    sve->cpu->flag_c = (uint8_t)((nzcv >> 1U) & 1U);
    sve->cpu->flag_v = (uint8_t)(nzcv & 1U);
}

/// PredTest(): N is the first active element of `result` under `mask`, Z that none is set, C
/// that the last active one is clear.
static uint32_t
predicate_test(const uint16_t mask, const uint16_t result, const uint32_t esize)
{
    bool first = false;
    bool any   = false;
    bool last  = false;
    bool seen  = false;

    for (uint32_t e = 0U; e < elements_of(esize); ++e)
    {
        if (true == is_active(mask, e, esize))
        {
            const bool value = is_active(result, e, esize);

            if (false == seen)
            {
                first = value;
                seen  = true;
            }

            any  = any || value;
            last = value;
        }
    }

    return ((true == first ? 1U : 0U) << 3U) | ((true == any ? 0U : 1U) << 2U)
           | ((true == last ? 0U : 1U) << 1U);
}

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

static void
write_x(sve_t *sve, const uint32_t index, const uint64_t value)
{
    if (index != 31U)
    {
        sve->cpu->x[index] = value;
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
read_x(const sve_t *sve, const uint32_t index)
{
    return 31U == index ? 0U : sve->cpu->x[index];
}

static uint16_t
read_p(const sve_t *sve, const uint32_t index)
{
    return sve->cpu->p[index];
}

/// The esize of a two-bit size field at bit 22.
static uint32_t
size_at_22(const uint32_t instruction)
{
    return 8U << ((instruction >> 22U) & 0x3U);
}

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
    const uint64_t value  = index < elements_of(esize) ? get_element(&n, index, esize) : 0U;
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
// Decoding
// -----------------------------------------------------------------------------

static const sve_pattern_t PATTERNS[] = {
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
    { 0xFF3EFC10U, 0x2518E000U, predicate_true },
    { 0xFFFFFFF0U, 0x2518E400U, predicate_false },
    { 0xFFFFFFF0U, 0x2519F000U, read_ffr },
    { 0xFFBFFE10U, 0x2518F000U, read_ffr_predicated },
    { 0xFFFFFFFFU, 0x252C9000U, set_ffr },
    { 0xFFFFFE1FU, 0x25289000U, write_ffr },
    { 0xFFFFF800U, 0x04BF5000U, read_vector_length },
    { 0xFFA0F800U, 0x04205000U, add_vector_length },
    { 0xFF30FC00U, 0x0420E000U, count_elements },
};

static sve_result_t
execute(sve_t *sve, const uint32_t instruction)
{
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
    sve_t sve        = { .cpu = &cpu, .memory = &memory, .flags = 0U };
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
    return SVE_FAULT == result ? 0U : 1U;
}

/*** end of file ***/
