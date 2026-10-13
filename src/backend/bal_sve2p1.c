//! The SVE2.1 instructions, and the instructions of the SVE encoding space that SME and SME2 add
//! to streaming mode and SVE2.1 to non-streaming mode, at the 128-bit vector length: one 128-bit
//! segment, so the quadword operations work on the whole vector.

#include "backend/bal_simd.h"
#include "backend/bal_sve.h"
#include "backend/bal_sve_shared.h"

#include <string.h>

// -----------------------------------------------------------------------------
// Scalars and vector lengths
// -----------------------------------------------------------------------------

/// ADDSVL and ADDSPL (000001000 op 1 Rn 01011 imm6 Rd) add imm6 times the streaming vector or
/// predicate length in bytes to Xn or SP, and RDSVL (000001001 0 1 11111 01011 imm6 Rd) reads
/// imm6 times the streaming vector length. The streaming vector length is the vector length.
static sve_result_t
streaming_vector_length(sve_t *sve, const uint32_t instruction)
{
    const int64_t  imm = sign_extend((instruction >> 5U) & 0x3FU, 6U);
    const uint32_t d   = instruction & 0x1FU;

    if (0x04BF5800U == (instruction & 0xFFFFF800U))
    {
        write_x(sve, d, (uint64_t)(imm * (int64_t)BAL_SVE_VECTOR_BYTES));
        return SVE_OK;
    }

    // Register 31 is SP for both operands.
    const int64_t unit = 0U != (instruction & 0x400000U) ? (int64_t)BAL_SVE_VECTOR_BYTES / 8
                                                         : (int64_t)BAL_SVE_VECTOR_BYTES;
    sve->cpu->x[d]     = sve->cpu->x[(instruction >> 16U) & 0x1FU] + (uint64_t)(imm * unit);
    return SVE_OK;
}

// -----------------------------------------------------------------------------
// Quadword vector operations
// -----------------------------------------------------------------------------

/// ADDQV, SMAXQV, UMAXQV, SMINQV, UMINQV, ORQV, EORQV and ANDQV (00000100 size opc 001 Pg Zn Vd,
/// opc 000101, 0011 op U, or 0111 opc): each element of the one segment, or the identity of the
/// operation where it is inactive.
static sve_result_t
quadword_reduction(sve_t *sve, const uint32_t instruction)
{
    const uint32_t esize = size_at_22(instruction);
    const uint64_t ones  = mask_of(esize);
    uint64_t       identity;

    switch ((instruction >> 16U) & 0x3FU)
    {
        case 0x05U: // ADDQV
        case 0x0DU: // UMAXQV
        case 0x1CU: // ORQV
        case 0x1DU: // EORQV
            identity = 0U;
            break;
        case 0x0CU: // SMAXQV
            identity = 1ULL << (esize - 1U);
            break;
        case 0x0EU: // SMINQV
            identity = ones >> 1U;
            break;
        case 0x0FU: // UMINQV
        case 0x1EU: // ANDQV
            identity = ones;
            break;
        default:
            return SVE_UNDEFINED;
    }

    const uint16_t pg      = read_p(sve, (instruction >> 10U) & 0x7U);
    const vector_t operand = read_z(sve, (instruction >> 5U) & 0x1FU);
    vector_t       result  = { { 0U, 0U } };

    for (uint32_t e = 0U; e < elements_of(esize); ++e)
    {
        set_element(&result,
                    e,
                    esize,
                    true == is_active(pg, e, esize) ? get_element(&operand, e, esize) : identity);
    }

    write_z(sve, instruction & 0x1FU, result);
    return SVE_OK;
}

/// DUPQ: 00000101001 i1 tsz 001001 Zn Zd, the indexed element of each segment broadcast in it.
static sve_result_t
duplicate_quadword(sve_t *sve, const uint32_t instruction)
{
    const uint32_t tsz = (instruction >> 16U) & 0xFU;

    if (0U == tsz)
    {
        return SVE_UNDEFINED;
    }

    const uint32_t shift   = (uint32_t)__builtin_ctz(tsz);
    const uint32_t esize   = 8U << shift;
    const uint32_t index   = ((instruction >> 16U) & 0x1FU) >> (shift + 1U);
    const vector_t operand = read_z(sve, (instruction >> 5U) & 0x1FU);
    const uint64_t value   = get_element(&operand, index, esize);
    vector_t       result;

    for (uint32_t e = 0U; e < elements_of(esize); ++e)
    {
        set_element(&result, e, esize, value);
    }

    write_z(sve, instruction & 0x1FU, result);
    return SVE_OK;
}

/// EXTQ: 000001010110 imm4 001001 Zm Zdn, the 16 bytes from byte imm4 of Zm:Zdn.
static sve_result_t
extract_quadword(sve_t *sve, const uint32_t instruction)
{
    const uint32_t dn       = instruction & 0x1FU;
    const uint32_t position = (instruction >> 16U) & 0xFU;
    uint8_t        bytes[32];
    (void)memcpy(bytes, sve->cpu->v[dn], 16U);
    (void)memcpy(bytes + 16U, sve->cpu->v[(instruction >> 5U) & 0x1FU], 16U);
    (void)memcpy(sve->cpu->v[dn], bytes + position, 16U);
    return SVE_OK;
}

/// ZIPQ1, ZIPQ2, UZPQ1, UZPQ2 and TBLQ (01000100 size 0 Zm 111 opc Zn Zd, opc 00 H, 01 H and
/// 110), within each segment.
static sve_result_t
quadword_permute(sve_t *sve, const uint32_t instruction)
{
    const uint32_t opc      = (instruction >> 10U) & 0x7U;
    const uint32_t esize    = size_at_22(instruction);
    const uint32_t elements = elements_of(esize);
    const uint32_t pairs    = elements / 2U;
    const uint32_t part     = opc & 1U;
    const vector_t first    = read_z(sve, (instruction >> 5U) & 0x1FU);
    const vector_t second   = read_z(sve, (instruction >> 16U) & 0x1FU);
    vector_t       result   = { { 0U, 0U } };

    if (opc >= 4U && 6U != opc)
    {
        return SVE_UNDEFINED;
    }

    for (uint32_t p = 0U; p < pairs; ++p)
    {
        if (opc < 2U)
        {
            const uint32_t from = part * pairs + p;
            set_element(&result, 2U * p, esize, get_element(&first, from, esize));
            set_element(&result, 2U * p + 1U, esize, get_element(&second, from, esize));
        }
        else if (opc < 4U)
        {
            set_element(&result, p, esize, get_element(&first, 2U * p + part, esize));
            set_element(&result, pairs + p, esize, get_element(&second, 2U * p + part, esize));
        }
    }

    if (6U == opc)
    {
        for (uint32_t e = 0U; e < elements; ++e)
        {
            const uint64_t index = get_element(&second, e, esize);
            set_element(&result,
                        e,
                        esize,
                        index < elements ? get_element(&first, (uint32_t)index, esize) : 0U);
        }
    }

    write_z(sve, instruction & 0x1FU, result);
    return SVE_OK;
}

/// TBXQ: 00000101 size 1 Zm 001101 Zn Zd, TBLQ leaving the elements of Zd whose index is out of
/// range.
static sve_result_t
table_extension_quadword(sve_t *sve, const uint32_t instruction)
{
    const uint32_t esize   = size_at_22(instruction);
    const uint32_t d       = instruction & 0x1FU;
    const vector_t table   = read_z(sve, (instruction >> 5U) & 0x1FU);
    const vector_t indices = read_z(sve, (instruction >> 16U) & 0x1FU);
    vector_t       result  = read_z(sve, d);

    for (uint32_t e = 0U; e < elements_of(esize); ++e)
    {
        const uint64_t index = get_element(&indices, e, esize);

        if (index < elements_of(esize))
        {
            set_element(&result, e, esize, get_element(&table, (uint32_t)index, esize));
        }
    }

    write_z(sve, d, result);
    return SVE_OK;
}

/// REVD: 00000101 00 101110 100 Pg Zn Zd, which swaps the doublewords of each active quadword.
static sve_result_t
reverse_doublewords(sve_t *sve, const uint32_t instruction)
{
    if (true == is_active(read_p(sve, (instruction >> 10U) & 0x7U), 0U, 128U))
    {
        const vector_t operand = read_z(sve, (instruction >> 5U) & 0x1FU);
        const vector_t result  = { { operand.d[1], operand.d[0] } };
        write_z(sve, instruction & 0x1FU, result);
    }

    return SVE_OK;
}

// -----------------------------------------------------------------------------
// Integer arithmetic
// -----------------------------------------------------------------------------

/// SCLAMP and UCLAMP: 01000100 size 0 Zm 11000 U Zn Zd, Zd = Min(Max(Zn, Zd), Zm).
static sve_result_t
integer_clamp(sve_t *sve, const uint32_t instruction)
{
    const uint32_t esize     = size_at_22(instruction);
    const bool     is_signed = 0U == (instruction & 0x400U);
    const uint32_t d         = instruction & 0x1FU;
    const vector_t low       = read_z(sve, (instruction >> 5U) & 0x1FU);
    const vector_t high      = read_z(sve, (instruction >> 16U) & 0x1FU);
    vector_t       result    = read_z(sve, d);

    for (uint32_t e = 0U; e < elements_of(esize); ++e)
    {
        const __int128 value = wide(get_element(&result, e, esize), esize, is_signed);
        const __int128 lower = wide(get_element(&low, e, esize), esize, is_signed);
        const __int128 upper = wide(get_element(&high, e, esize), esize, is_signed);
        const __int128 above = value > lower ? value : lower;
        set_element(&result, e, esize, (uint64_t)(above < upper ? above : upper));
    }

    write_z(sve, d, result);
    return SVE_OK;
}

/// The two-way SDOT and UDOT of halfwords into words: 01000100000 Zm 11001 U Zn Zda, and by
/// element, 01000100100 i2 Zm 11001 U Zn Zda.
static sve_result_t
two_way_dot_product(sve_t *sve, const uint32_t instruction)
{
    const bool     is_signed  = 0U == (instruction & 0x400U);
    const bool     is_indexed = 0U != (instruction & 0x800000U);
    const uint32_t da         = instruction & 0x1FU;
    const uint32_t m          = (instruction >> 16U) & (true == is_indexed ? 0x7U : 0x1FU);
    const vector_t first      = read_z(sve, (instruction >> 5U) & 0x1FU);
    const vector_t second     = read_z(sve, m);
    vector_t       result     = read_z(sve, da);

    for (uint32_t e = 0U; e < 4U; ++e)
    {
        // The indexed form takes the pair of the segment's element i2.
        const uint32_t s   = true == is_indexed ? (instruction >> 19U) & 0x3U : e;
        __int128       sum = get_element(&result, e, 32U);

        for (uint32_t i = 0U; i < 2U; ++i)
        {
            sum += wide(get_element(&first, 2U * e + i, 16U), 16U, is_signed)
                   * wide(get_element(&second, 2U * s + i, 16U), 16U, is_signed);
        }

        set_element(&result, e, 32U, (uint64_t)sum);
    }

    write_z(sve, da, result);
    return SVE_OK;
}

/// The saturating narrows of two vectors into one, interleaved: SQCVTN, UQCVTN and SQCVTUN
/// (010001010 0 1 10 001010 opc 0 Zn 0 Zd, opc 00, 01 and 10), and SQRSHRN, UQRSHRN and
/// SQRSHRUN (010001011 0 1 1 imm4 00 op U 1 0 Zn 0 Zd), which round and shift right by 16 - imm4
/// first. Words narrow to halfwords.
static sve_result_t
narrow_pair(sve_t *sve, const uint32_t instruction)
{
    const bool is_shift = 0U != (instruction & 0x800000U);
    uint32_t   shift    = 0U;
    bool       is_signed_source;
    saturate_t mode;

    if (true == is_shift)
    {
        // op U = 00 SQRSHRUN, 10 SQRSHRN, 11 UQRSHRN.
        switch ((instruction >> 12U) & 0x3U)
        {
            case 0U:
                is_signed_source = true;
                mode             = SATURATE_UNSIGNED;
                break;
            case 2U:
                is_signed_source = true;
                mode             = SATURATE_SIGNED;
                break;
            case 3U:
                is_signed_source = false;
                mode             = SATURATE_UNSIGNED;
                break;
            default:
                return SVE_UNDEFINED;
        }

        shift = 16U - ((instruction >> 16U) & 0xFU);
    }
    else
    {
        switch ((instruction >> 11U) & 0x3U)
        {
            case 0U:
                is_signed_source = true;
                mode             = SATURATE_SIGNED;
                break;
            case 1U:
                is_signed_source = false;
                mode             = SATURATE_UNSIGNED;
                break;
            case 2U:
                is_signed_source = true;
                mode             = SATURATE_UNSIGNED;
                break;
            default:
                return SVE_UNDEFINED;
        }
    }

    const uint32_t n = (instruction >> 5U) & 0x1EU;
    vector_t       result;

    for (uint32_t i = 0U; i < 2U; ++i)
    {
        const vector_t operand = read_z(sve, n + i);

        for (uint32_t e = 0U; e < 4U; ++e)
        {
            __int128 value = wide(get_element(&operand, e, 32U), 32U, is_signed_source);

            if (0U != shift)
            {
                value = (value + ((__int128)1 << (shift - 1U))) >> shift;
            }

            set_element(&result, 2U * e + i, 16U, narrow_to(value, 16U, mode));
        }
    }

    write_z(sve, instruction & 0x1FU, result);
    return SVE_OK;
}

// -----------------------------------------------------------------------------
// Predicates
// -----------------------------------------------------------------------------

/// EncodePredCount(): the predicate-as-counter value for the first `count` of `elements`
/// elements of `esize` bits, or the last ones when `invert`.
static uint16_t
encode_predicate_count(const uint32_t esize, const uint32_t elements, uint32_t count, bool invert)
{
    if (0U == count)
    {
        return 0U;
    }

    if (true == invert)
    {
        count = elements - count;
    }
    else if (count == elements)
    {
        count  = 0U;
        invert = true;
    }

    const uint32_t shift = (uint32_t)__builtin_ctz(esize / 8U);
    return (uint16_t)((true == invert ? 0x8000U : 0U)
                      | (((count << (shift + 1U)) | (1U << shift)) & 0x7FFFU));
}

/// PredCountTest(): NZCV for a predicate-as-counter result.
static uint32_t
predicate_count_test(const uint32_t elements, const uint32_t count, const bool invert)
{
    const bool n = true == invert ? count == elements : 0U != count;
    const bool c = true == invert ? 0U == count : count != elements;
    return ((true == n ? 1U : 0U) << 3U) | ((0U == count ? 1U : 0U) << 2U)
           | ((true == c ? 1U : 0U) << 1U);
}

/// PTRUE (predicate as counter): 00100101 size 1000000111100000 1 0 PNd.
static sve_result_t
predicate_true_counter(sve_t *sve, const uint32_t instruction)
{
    const uint32_t esize = size_at_22(instruction);
    sve->cpu->p[8U + (instruction & 0x7U)]
        = encode_predicate_count(esize, elements_of(esize), elements_of(esize), false);
    return SVE_OK;
}

/// CNTP (predicate as counter): 00100101 size 100 000 10000 vl 1 PNn Rd, the active elements of
/// the two or four vectors the counter describes.
static sve_result_t
count_predicate_counter(sve_t *sve, const uint32_t instruction)
{
    const uint32_t esize = size_at_22(instruction);
    const uint32_t limit = elements_of(esize) * (2U << ((instruction >> 10U) & 1U));
    const uint64_t mask  = counter_to_predicate(read_p(sve, (instruction >> 5U) & 0xFU));
    uint64_t       sum   = 0U;

    for (uint32_t e = 0U; e < limit; ++e)
    {
        sum += (mask >> (e * (esize / 8U))) & 1U;
    }

    write_x(sve, instruction & 0x1FU, sum);
    return SVE_OK;
}

/// PEXT: 00100101 size 10000001110 0 imm2 PNn 1 Pd extracts part imm2 of the predicate a counter
/// in PN8-PN15 describes, and ... 1 0 i1 PNn 1 Pd parts 2 * i1 and 2 * i1 + 1 into Pd and Pd + 1.
static sve_result_t
predicate_extract(sve_t *sve, const uint32_t instruction)
{
    const bool     is_pair = 0U != (instruction & 0x400U);
    const uint32_t esize   = size_at_22(instruction);
    const uint32_t psize   = esize / 8U;
    const uint32_t count   = elements_of(esize);
    const uint32_t d       = instruction & 0xFU;
    const uint64_t mask    = counter_to_predicate(read_p(sve, 8U + ((instruction >> 5U) & 0x7U)));

    if (true == is_pair && 0U != (instruction & 0x200U))
    {
        return SVE_UNDEFINED;
    }

    const uint32_t part = (instruction >> 8U) & (true == is_pair ? 0x1U : 0x3U);

    for (uint32_t r = 0U; r < (true == is_pair ? 2U : 1U); ++r)
    {
        const uint32_t first  = (true == is_pair ? 2U * part + r : part) * count;
        uint16_t       result = 0U;

        for (uint32_t e = 0U; e < count; ++e)
        {
            result |= (uint16_t)(((mask >> ((first + e) * psize)) & 1U) << (e * psize));
        }

        sve->cpu->p[(d + r) & 0xFU] = result;
    }

    return SVE_OK;
}

/// PSEL: 00100101 i1 tszh 1 tszl Rv 01 Pn 0 Pm 0 Pd, Pn where element Wv + imm of Pm is active
/// and all false otherwise.
static sve_result_t
predicate_select(sve_t *sve, const uint32_t instruction)
{
    const uint32_t tsz = (((instruction >> 22U) & 1U) << 3U) | ((instruction >> 18U) & 0x7U);

    if (0U == tsz)
    {
        return SVE_UNDEFINED;
    }

    const uint32_t shift    = (uint32_t)__builtin_ctz(tsz);
    const uint32_t esize    = 8U << shift;
    const uint32_t imm5     = (((instruction >> 23U) & 1U) << 4U) | tsz;
    const uint64_t index    = (uint32_t)sve->cpu->x[12U + ((instruction >> 16U) & 0x3U)];
    const uint32_t element  = (uint32_t)((index + (imm5 >> (shift + 1U))) % elements_of(esize));
    const uint16_t selector = read_p(sve, (instruction >> 5U) & 0xFU);
    sve->cpu->p[instruction & 0xFU] = true == is_active(selector, element, esize)
                                          ? read_p(sve, (instruction >> 10U) & 0xFU)
                                          : 0U;
    return SVE_OK;
}

/// Whether element `e` of a WHILE comparison holds: `x` is the first operand moved by `e`.
static bool
while_condition(const uint64_t x, const uint64_t y, const bool is_unsigned, const uint32_t op)
{
    const bool less    = true == is_unsigned ? x < y : (int64_t)x < (int64_t)y;
    const bool greater = true == is_unsigned ? x > y : (int64_t)x > (int64_t)y;

    // op is lt:eq: GE, GT, LT, LE.
    switch (op)
    {
        case 0U:
            return false == less;
        case 1U:
            return true == greater;
        case 2U:
            return true == less;
        default:
            return false == greater;
    }
}

/// WHILEGE, WHILEGT, WHILEHI, WHILEHS, WHILELE, WHILELO, WHILELS and WHILELT with a
/// predicate-as-counter result for two or four vectors (00100101 size 1 Rm 01 vl 0 U lt Rn 1 eq
/// PNd) or a pair of predicates (00100101 size 1 Rm 0101 U lt Rn 1 Pd eq). The LT and LE forms
/// count up from Xn, the GE and GT forms down.
static sve_result_t
predicate_while_multiple(sve_t *sve, const uint32_t instruction)
{
    const bool     is_counter  = 0U == (instruction & 0x1000U);
    const uint32_t esize       = size_at_22(instruction);
    const bool     is_unsigned = 0U != (instruction & 0x800U);
    const bool     is_up       = 0U != (instruction & 0x400U);
    const uint32_t eq          = true == is_counter ? (instruction >> 3U) & 1U : instruction & 1U;
    const uint32_t op          = ((true == is_up ? 1U : 0U) << 1U) | eq;
    const uint32_t elements
        = elements_of(esize) * (true == is_counter ? 2U << ((instruction >> 13U) & 1U) : 2U);
    const uint64_t y     = read_x(sve, (instruction >> 16U) & 0x1FU);
    uint64_t       x     = read_x(sve, (instruction >> 5U) & 0x1FU);
    uint32_t       count = 0U;

    for (uint32_t i = 0U; i < elements; ++i)
    {
        if (false == while_condition(x, y, is_unsigned, op))
        {
            break;
        }

        ++count;
        x = true == is_up ? x + 1U : x - 1U;
    }

    if (true == is_counter)
    {
        sve->cpu->p[8U + (instruction & 0x7U)]
            = encode_predicate_count(esize, elements, count, false == is_up);
        write_nzcv(sve, predicate_count_test(elements, count, false == is_up));
        return SVE_OK;
    }

    // The first `count` elements of the pair are active counting up, the last counting down.
    const uint32_t psize  = esize / 8U;
    uint32_t       result = 0U;

    for (uint32_t e = 0U; e < elements; ++e)
    {
        const bool active = true == is_up ? e < count : e >= elements - count;
        result |= (true == active ? 1U : 0U) << (e * psize);
    }

    const uint32_t d    = instruction & 0xEU;
    sve->cpu->p[d]      = (uint16_t)result;
    sve->cpu->p[d + 1U] = (uint16_t)(result >> 16U);

    // PredTest() of the pair with an all-true mask.
    const bool first = 0U != (result & 1U);
    const bool last  = 0U != (result >> ((elements - 1U) * psize) & 1U);
    write_nzcv(sve,
               ((true == first ? 1U : 0U) << 3U) | ((0U == result ? 1U : 0U) << 2U)
                   | ((true == last ? 0U : 1U) << 1U));
    return SVE_OK;
}

/// The element size and part of PMOV: 00000101 opc 101 opc2 ..., with opc 00 and opc2 01 for
/// bytes, opc 00 and opc2 1 i1 for halfwords, opc 01 and i2 for words, and 1 i3h with i3l for
/// doublewords. Returns false for the unallocated opc 00 with opc2 00.
static bool
decode_predicate_move(const uint32_t instruction, uint32_t *esize, uint32_t *part)
{
    const uint32_t opc  = (instruction >> 22U) & 0x3U;
    const uint32_t opc2 = (instruction >> 17U) & 0x3U;

    if (opc >= 2U)
    {
        *esize = 64U;
        *part  = ((opc & 1U) << 2U) | opc2;
    }
    else if (1U == opc)
    {
        *esize = 32U;
        *part  = opc2;
    }
    else if (opc2 >= 2U)
    {
        *esize = 16U;
        *part  = opc2 & 1U;
    }
    else
    {
        *esize = 8U;
        *part  = 0U;
        return 1U == opc2;
    }

    return true;
}

/// PMOV between a predicate and a part of a vector: to a predicate (00000101 opc 101 opc2
/// 0001110 Zn 0 Pd), each element from one bit of the part, and to a vector (... 1001110 0 Pn
/// Zd), each bit of the part from one element, the rest of the vector zero for part 0 and kept
/// otherwise.
static sve_result_t
predicate_move(sve_t *sve, const uint32_t instruction)
{
    uint32_t esize;
    uint32_t part;

    if (false == decode_predicate_move(instruction, &esize, &part))
    {
        return SVE_UNDEFINED;
    }

    const uint32_t elements = elements_of(esize);
    const uint32_t first    = elements * part;

    if (0U == (instruction & 0x10000U))
    {
        const vector_t operand = read_z(sve, (instruction >> 5U) & 0x1FU);
        uint16_t       result  = 0U;

        for (uint32_t e = 0U; e < elements; ++e)
        {
            const uint32_t bit = first + e;
            result
                |= (uint16_t)(((operand.d[bit / 64U] >> (bit % 64U)) & 1U) << (e * (esize / 8U)));
        }

        sve->cpu->p[instruction & 0xFU] = result;
        return SVE_OK;
    }

    const uint32_t d       = instruction & 0x1FU;
    const uint16_t operand = read_p(sve, (instruction >> 5U) & 0xFU);
    vector_t       result  = { { 0U, 0U } };

    if (0U != part)
    {
        result = read_z(sve, d);
    }

    for (uint32_t e = 0U; e < elements; ++e)
    {
        const uint32_t bit = first + e;
        result.d[bit / 64U] &= ~(1ULL << (bit % 64U));
        result.d[bit / 64U] |= (true == is_active(operand, e, esize) ? 1ULL : 0ULL) << (bit % 64U);
    }

    write_z(sve, d, result);
    return SVE_OK;
}

// -----------------------------------------------------------------------------
// Loads and stores
// -----------------------------------------------------------------------------

/// LD1W and LD1D into quadwords (1010010 1 msz 001 imm4 001 Pg Rn Zt, and 1010010 1 msz 00 Rm
/// 100 ...), which zero-extend one word or doubleword into each quadword, and ST1W and ST1D from
/// quadwords (1110010 1 msz opc 0 imm4 111 ..., and 1110010 1 msz 0 0 Rm 010 ..., with opc 00
/// for words and 10 for doublewords), which store its low word or doubleword.
static sve_result_t
quadword_element_memory(sve_t *sve, const uint32_t instruction)
{
    const bool     is_store     = 0U != (instruction & 0x40000000U);
    const uint32_t mbytes       = 0U != (instruction & 0x800000U) ? 8U : 4U;
    const uint32_t op           = (instruction >> 13U) & 0x7U;
    const bool     is_immediate = (true == is_store ? 7U : 1U) == op;
    const uint32_t t            = instruction & 0x1FU;
    const uint32_t m            = (instruction >> 16U) & 0x1FU;
    uint64_t       address      = sve->cpu->x[(instruction >> 5U) & 0x1FU];

    if (true == is_immediate)
    {
        address += (uint64_t)(sign_extend(m & 0xFU, 4U) * (int64_t)mbytes);
    }
    else if (31U == m)
    {
        return SVE_UNDEFINED;
    }
    else
    {
        address += sve->cpu->x[m] * mbytes;
    }

    const bool is_active_element = is_active(read_p(sve, (instruction >> 10U) & 0x7U), 0U, 128U);

    if (true == is_active_element && false == is_mapped(sve, address, mbytes))
    {
        return SVE_FAULT;
    }

    if (true == is_store)
    {
        if (true == is_active_element)
        {
            memory_write(sve, address, mbytes, sve->cpu->v[t][0]);
        }

        return SVE_OK;
    }

    const vector_t result
        = { { true == is_active_element ? memory_read(sve, address, mbytes) : 0U, 0U } };
    write_z(sve, t, result);
    return SVE_OK;
}

/// LD2Q, LD3Q and LD4Q (1010010 num 001 imm4 111 Pg Rn Zt, and 1010010 num 01 Rm 100 ...) and
/// ST2Q, ST3Q and ST4Q (11100100 num 00 imm4 000 ..., and 11100100 num 1 Rm 000 ...): two to four
/// consecutive registers as interleaved quadwords, imm4 counting groups of them and Xm quadwords.
static sve_result_t
quadword_structures(sve_t *sve, const uint32_t instruction)
{
    const bool     is_store = 0U != (instruction & 0x40000000U);
    const uint32_t count    = ((instruction >> (true == is_store ? 22U : 23U)) & 0x3U) + 1U;
    const bool     is_immediate
        = true == is_store ? 0U == (instruction & 0x200000U) : 7U == ((instruction >> 13U) & 0x7U);
    const uint32_t t       = instruction & 0x1FU;
    const uint32_t m       = (instruction >> 16U) & 0x1FU;
    uint64_t       address = sve->cpu->x[(instruction >> 5U) & 0x1FU];

    // num = 00 is another instruction, or unallocated.
    if (1U == count)
    {
        return SVE_UNDEFINED;
    }

    if (true == is_immediate)
    {
        address += (uint64_t)(sign_extend(m & 0xFU, 4U) * (int64_t)(count * 16U));
    }
    else if (31U == m)
    {
        return SVE_UNDEFINED;
    }
    else
    {
        address += sve->cpu->x[m] * 16U;
    }

    const bool is_active_element = is_active(read_p(sve, (instruction >> 10U) & 0x7U), 0U, 128U);

    if (true == is_active_element && false == is_mapped(sve, address, count * 16U))
    {
        return SVE_FAULT;
    }

    for (uint32_t r = 0U; r < count; ++r)
    {
        const uint32_t z  = (t + r) & 0x1FU;
        const uint64_t at = address + r * 16U;

        if (true == is_store)
        {
            if (true == is_active_element)
            {
                memory_write(sve, at, 8U, sve->cpu->v[z][0]);
                memory_write(sve, at + 8U, 8U, sve->cpu->v[z][1]);
            }
        }
        else
        {
            const vector_t value
                = { { true == is_active_element ? memory_read(sve, at, 8U) : 0U,
                      true == is_active_element ? memory_read(sve, at + 8U, 8U) : 0U } };
            write_z(sve, z, value);
        }
    }

    return SVE_OK;
}

/// LD1Q (11000100000 Rm 101 Pg Zn Zt) and ST1Q (11100100001 Rm 001 ...): the quadword at the low
/// doubleword of Zn plus Xm, which XZR leaves out.
static sve_result_t
quadword_gather_scatter(sve_t *sve, const uint32_t instruction)
{
    const bool     is_store = 0U != (instruction & 0x20000000U);
    const uint32_t t        = instruction & 0x1FU;
    const uint64_t address
        = sve->cpu->v[(instruction >> 5U) & 0x1FU][0] + read_x(sve, (instruction >> 16U) & 0x1FU);
    const bool is_active_element = is_active(read_p(sve, (instruction >> 10U) & 0x7U), 0U, 128U);

    if (true == is_active_element && false == is_mapped(sve, address, 16U))
    {
        return SVE_FAULT;
    }

    if (true == is_store)
    {
        if (true == is_active_element)
        {
            memory_write(sve, address, 8U, sve->cpu->v[t][0]);
            memory_write(sve, address + 8U, 8U, sve->cpu->v[t][1]);
        }

        return SVE_OK;
    }

    const vector_t value
        = { { true == is_active_element ? memory_read(sve, address, 8U) : 0U,
              true == is_active_element ? memory_read(sve, address + 8U, 8U) : 0U } };
    write_z(sve, t, value);
    return SVE_OK;
}

// -----------------------------------------------------------------------------
// Floating point
// -----------------------------------------------------------------------------

#define FPSR_OFC (1ULL << 2U)
#define FPSR_UFC (1ULL << 3U)
#define FPSR_IXC (1ULL << 4U)

#define FPCR_FZ (1ULL << 24U)
#define FPCR_DN (1ULL << 25U)

/// FADDQV, FMAXNMQV, FMINNMQV, FMAXQV and FMINQV: 01100100 size 010 opc 101 Pg Zn Vd, opc 000,
/// 100, 101, 110 and 111. Reducing the one segment leaves each element as it is, so the result
/// is the active elements and the identity of the operation for the others.
static sve_result_t
float_quadword_reduction(sve_t *sve, const uint32_t instruction)
{
    const uint32_t size = (instruction >> 22U) & 0x3U;
    const uint32_t opc  = (instruction >> 16U) & 0x7U;

    if (0U == size || (0U != opc && opc < 4U))
    {
        return SVE_UNDEFINED;
    }

    const uint32_t esize    = 8U << size;
    const uint32_t fraction = fp_fraction_bits(esize);
    const uint64_t infinity = mask_of(esize - 1U - fraction) << fraction;
    uint64_t       identity;

    switch (opc)
    {
        case 0U: // FADDQV: +0.
            identity = 0U;
            break;
        case 6U: // FMAXQV: -infinity.
            identity = infinity | (1ULL << (esize - 1U));
            break;
        case 7U: // FMINQV: +infinity.
            identity = infinity;
            break;
        default: // FMAXNMQV and FMINNMQV: the default NaN.
            identity = infinity | (1ULL << (fraction - 1U));
            break;
    }

    const uint16_t pg      = read_p(sve, (instruction >> 10U) & 0x7U);
    const vector_t operand = read_z(sve, (instruction >> 5U) & 0x1FU);
    vector_t       result  = { { 0U, 0U } };

    for (uint32_t e = 0U; e < elements_of(esize); ++e)
    {
        set_element(&result,
                    e,
                    esize,
                    true == is_active(pg, e, esize) ? get_element(&operand, e, esize) : identity);
    }

    write_z(sve, instruction & 0x1FU, result);
    return SVE_OK;
}

/// FCLAMP (01100100 size 1 Zm 001001 Zn Zd) and BFCLAMP (size 00): Zd = FMINNM(FMAXNM(Zn, Zd),
/// Zm).
static sve_result_t
float_clamp(sve_t *sve, const uint32_t instruction)
{
    const uint32_t size   = (instruction >> 22U) & 0x3U;
    const uint32_t esize  = 0U == size ? 0U : 8U << size;
    const uint32_t bits   = 0U == esize ? 16U : esize;
    const uint32_t d      = instruction & 0x1FU;
    const vector_t low    = read_z(sve, (instruction >> 5U) & 0x1FU);
    const vector_t high   = read_z(sve, (instruction >> 16U) & 0x1FU);
    vector_t       result = read_z(sve, d);

    for (uint32_t e = 0U; e < elements_of(bits); ++e)
    {
        const uint64_t above = float_extreme(
            sve, esize, 1U, get_element(&low, e, bits), get_element(&result, e, bits));
        set_element(
            &result, e, bits, float_extreme(sve, esize, 3U, above, get_element(&high, e, bits)));
    }

    write_z(sve, d, result);
    return SVE_OK;
}

/// FPDot() on half precision pairs: a[0] * b[0] + a[1] * b[1] rounded once to single precision.
/// The halves widen exactly, keeping NaN payloads, so the products are exact single precision
/// values; a NaN operand gives the first signalling one, or else the first quiet one, quietened.
static uint64_t
half_pair_dot(sve_t *sve, const uint64_t a[2], const uint64_t b[2])
{
    const uint64_t fpcr       = sve->cpu->fpcr;
    const uint64_t singles[4] = {
        bal_simd_half_to_single(fpcr, a[0]),
        bal_simd_half_to_single(fpcr, a[1]),
        bal_simd_half_to_single(fpcr, b[0]),
        bal_simd_half_to_single(fpcr, b[1]),
    };

    for (uint32_t pass = 0U; pass < 2U; ++pass)
    {
        for (uint32_t i = 0U; i < 4U; ++i)
        {
            const bool is_signalling = 0U == (singles[i] & 0x400000U);

            if (true == fp_is_nan(singles[i], 32U) && (0U == pass) == is_signalling)
            {
                sve->flags |= true == is_signalling ? FPSR_IOC : 0U;
                return 0U != (fpcr & FPCR_DN) ? 0x7FC00000U : singles[i] | 0x400000U;
            }
        }
    }

    // Infinity times zero gives the default NaN and Invalid Operation, which the sum keeps.
    const uint64_t first  = fp(sve, 32U, BAL_FLOAT_MULTIPLY, singles[0], singles[2], 0U);
    const uint64_t second = fp(sve, 32U, BAL_FLOAT_MULTIPLY, singles[1], singles[3], 0U);
    return fp(sve, 32U, BAL_FLOAT_ADD, first, second, 0U);
}

/// FDOT of half precision pairs into single precision (01100100 001 Zm 100000 Zn Zda, and by
/// element, 01100100 001 i2 Zm 010000 Zn Zda): each product pair rounded once, then added to
/// Zda.
static sve_result_t
half_dot_product(sve_t *sve, const uint32_t instruction)
{
    const bool     is_indexed = 0U == (instruction & 0x8000U);
    const uint32_t da         = instruction & 0x1FU;
    const vector_t n          = read_z(sve, (instruction >> 5U) & 0x1FU);
    const vector_t m      = read_z(sve, (instruction >> 16U) & (true == is_indexed ? 0x7U : 0x1FU));
    vector_t       result = read_z(sve, da);

    for (uint32_t e = 0U; e < 4U; ++e)
    {
        const uint32_t pair = true == is_indexed ? (instruction >> 19U) & 0x3U : e;
        const uint64_t a[2] = { get_element(&n, 2U * e, 16U), get_element(&n, 2U * e + 1U, 16U) };
        const uint64_t b[2]
            = { get_element(&m, 2U * pair, 16U), get_element(&m, 2U * pair + 1U, 16U) };
        const uint64_t product = half_pair_dot(sve, a, b);
        set_element(&result,
                    e,
                    32U,
                    fp(sve, 32U, BAL_FLOAT_ADD, get_element(&result, e, 32U), product, 0U));
    }

    write_z(sve, da, result);
    return SVE_OK;
}

/// A BFloat16 operand, unpacked as the single precision value it widens to.
typedef struct
{
    uint64_t bits;
    double   value;
    bool     is_nan;
    bool     is_signalling;
    bool     is_infinity;
    bool     is_zero;
    bool     is_negative;
    uint8_t  pad[3];
} bfloat_operand_t;

/// FPUnpack() of a BFloat16 value widened to single precision: denormals flush to zero under
/// FPCR.FZ, raising Input Denormal.
static bfloat_operand_t
bfloat_unpack(sve_t *sve, const uint64_t value)
{
    const uint64_t   bits     = bfloat_to_single(value);
    const uint32_t   exponent = (uint32_t)(bits >> 23U) & 0xFFU;
    const uint64_t   fraction = bits & 0x7FFFFFU;
    bfloat_operand_t operand;
    (void)memset(&operand, 0, sizeof(operand));
    operand.bits        = bits;
    operand.is_negative = 0U != (bits >> 31U);

    if (0xFFU == exponent)
    {
        operand.is_nan        = 0U != fraction;
        operand.is_signalling = 0U != fraction && 0U == (fraction & 0x400000U);
        operand.is_infinity   = 0U == fraction;
        return operand;
    }

    if (0U == exponent && (0U == fraction || 0U != (sve->cpu->fpcr & FPCR_FZ)))
    {
        sve->flags |= 0U != fraction ? FPSR_IDC : 0U;
        operand.is_zero = true;
        operand.value   = true == operand.is_negative ? -0.0 : 0.0;
        return operand;
    }

    const double magnitude
        = 0U == exponent ? __builtin_ldexp((double)fraction, -149)
                         : __builtin_ldexp((double)(fraction | 0x800000U), (int)exponent - 150);
    operand.value = true == operand.is_negative ? -magnitude : magnitude;
    return operand;
}

/// `first` + `second` rounded to odd in double precision, which keeps the exact sum's rounding
/// to BFloat16, its inexactness and its tininess.
static double
sum_to_odd(const double first, const double second)
{
    const double sum     = first + second;
    const double partial = sum - first;
    const double error   = (first - (sum - partial)) + (second - partial);
    uint64_t     bits;
    (void)memcpy(&bits, &sum, sizeof(bits));

    if (0.0 != error && 0U == (bits & 1U))
    {
        return __builtin_nextafter(sum, error > 0.0 ? __builtin_inf() : -__builtin_inf());
    }

    return sum;
}

/// FPRoundBF(): a nonzero finite value, rounded to odd in double precision from the exact
/// result, rounded to BFloat16 under FPCR, raising Underflow (for a tiny inexact value, or a
/// tiny value flushed to zero under FPCR.FZ), Overflow and Inexact.
static uint64_t
bfloat_round(sve_t *sve, const double value)
{
    const uint64_t fpcr  = sve->cpu->fpcr;
    const uint32_t rmode = (uint32_t)(fpcr >> 22U) & 0x3U;
    uint64_t       bits;
    (void)memcpy(&bits, &value, sizeof(bits));
    const bool     is_negative = 0U != (bits >> 63U);
    const uint64_t sign        = true == is_negative ? 0x8000U : 0U;
    int32_t        exponent    = (int32_t)((bits >> 52U) & 0x7FFU) - 1023;
    const uint64_t mantissa    = (bits & 0xFFFFFFFFFFFFFULL) | (1ULL << 52U);
    const bool     is_tiny     = exponent < -126;

    if (true == is_tiny && 0U != (fpcr & FPCR_FZ))
    {
        sve->flags |= FPSR_UFC;
        return sign;
    }

    // Seven fraction bits, fewer below the normal range.
    const uint32_t drop = 45U + (true == is_tiny ? (uint32_t)(-126 - exponent) : 0U);
    uint64_t       kept = 0U;
    bool           is_inexact;
    bool           is_above_half = false;
    bool           is_half       = false;

    if (drop < 64U)
    {
        kept                     = mantissa >> drop;
        const uint64_t remainder = mantissa & ((1ULL << drop) - 1U);
        const uint64_t half      = 1ULL << (drop - 1U);
        is_inexact               = 0U != remainder;
        is_above_half            = remainder > half;
        is_half                  = remainder == half;
    }
    else
    {
        // Far below half of the smallest denormal.
        is_inexact = true;
    }

    if (true == is_tiny && true == is_inexact)
    {
        sve->flags |= FPSR_UFC;
    }

    bool round_up        = false;
    bool overflow_to_inf = false;

    switch (rmode)
    {
        case 0U:
            round_up        = is_above_half || (is_half && 0U != (kept & 1U));
            overflow_to_inf = true;
            break;
        case 1U:
            round_up        = is_inexact && false == is_negative;
            overflow_to_inf = false == is_negative;
            break;
        case 2U:
            round_up        = is_inexact && true == is_negative;
            overflow_to_inf = true == is_negative;
            break;
        default:
            break;
    }

    sve->flags |= true == is_inexact ? FPSR_IXC : 0U;
    kept += true == round_up ? 1U : 0U;

    if (true == is_tiny)
    {
        // A denormal, whose fraction carries into the smallest normal.
        return sign | kept;
    }

    if (0U != (kept >> 8U))
    {
        kept >>= 1U;
        ++exponent;
    }

    if (exponent > 127)
    {
        sve->flags |= FPSR_OFC | FPSR_IXC;
        return sign | (true == overflow_to_inf ? 0x7F80U : 0x7F7FU);
    }

    return sign | ((uint64_t)(exponent + 127) << 7U) | (kept & 0x7FU);
}

/// The BFloat16 arithmetic of FEAT_SVE_B16B16.
typedef enum
{
    BFLOAT_ADD,
    BFLOAT_SUBTRACT,
    BFLOAT_MULTIPLY,
    /// BFMulAdd(): addend + a * b.
    BFLOAT_MULTIPLY_ADD,
} bfloat_kind_t;

/// BFAdd(), BFSub(), BFMul() and BFMulAdd() under FPCR: single precision rules on the widened
/// operands, with the exact result rounded once to BFloat16.
static uint64_t
bfloat_compute(sve_t              *sve,
               const bfloat_kind_t kind,
               const uint64_t      addend_bits,
               const uint64_t      a_bits,
               const uint64_t      b_bits)
{
    const bool       is_fma = BFLOAT_MULTIPLY_ADD == kind;
    const uint64_t   fpcr   = sve->cpu->fpcr;
    bfloat_operand_t addend;
    (void)memset(&addend, 0, sizeof(addend));

    if (true == is_fma)
    {
        addend = bfloat_unpack(sve, addend_bits);
    }

    const bfloat_operand_t a      = bfloat_unpack(sve, a_bits);
    const bfloat_operand_t b      = bfloat_unpack(sve, b_bits);
    const bool is_product_invalid = (a.is_infinity && b.is_zero) || (a.is_zero && b.is_infinity);

    // BFMulAdd() gives the default NaN for a quiet NaN addend and infinity times zero.
    if (true == is_fma && addend.is_nan && false == addend.is_signalling
        && true == is_product_invalid)
    {
        sve->flags |= FPSR_IOC;
        return 0x7FC0U;
    }

    // FPProcessNaNs() and FPProcessNaNs3(): the addend first, signalling NaNs before quiet ones.
    const bfloat_operand_t *operands[3] = { &addend, &a, &b };

    for (uint32_t pass = 0U; pass < 2U; ++pass)
    {
        for (uint32_t i = true == is_fma ? 0U : 1U; i < 3U; ++i)
        {
            if (true == operands[i]->is_nan && (0U == pass) == operands[i]->is_signalling)
            {
                sve->flags |= 0U == pass ? FPSR_IOC : 0U;
                return 0U != (fpcr & FPCR_DN) ? 0x7FC0U : (operands[i]->bits >> 16U) | 0x40U;
            }
        }
    }

    const uint32_t rmode        = (uint32_t)(fpcr >> 22U) & 0x3U;
    const uint64_t rounded_zero = 2U == rmode ? 0x8000U : 0U;
    double         first;
    double         second;
    bool           first_negative;
    bool           second_negative;
    bool           first_infinity;
    bool           second_infinity;
    bool           first_zero;
    bool           second_zero;

    if (BFLOAT_MULTIPLY == kind)
    {
        const bool is_negative = a.is_negative != b.is_negative;

        if (true == is_product_invalid)
        {
            sve->flags |= FPSR_IOC;
            return 0x7FC0U;
        }

        if (true == a.is_infinity || true == b.is_infinity)
        {
            return true == is_negative ? 0xFF80U : 0x7F80U;
        }

        if (true == a.is_zero || true == b.is_zero)
        {
            return true == is_negative ? 0x8000U : 0U;
        }

        // A product of two BFloat16 values is exact in double precision.
        return bfloat_round(sve, a.value * b.value);
    }

    if (true == is_fma)
    {
        // The addend and the product, whose sign and kind the factors give.
        first           = addend.value;
        first_negative  = addend.is_negative;
        first_infinity  = addend.is_infinity;
        first_zero      = addend.is_zero;
        second          = a.value * b.value;
        second_negative = a.is_negative != b.is_negative;
        second_infinity = a.is_infinity || b.is_infinity;
        second_zero     = a.is_zero || b.is_zero;

        if (true == is_product_invalid)
        {
            sve->flags |= FPSR_IOC;
            return 0x7FC0U;
        }
    }
    else
    {
        // BFSub() negates the second operand only after its NaN checks.
        const bool is_subtract = BFLOAT_SUBTRACT == kind;
        first                  = a.value;
        first_negative         = a.is_negative;
        first_infinity         = a.is_infinity;
        first_zero             = a.is_zero;
        second                 = true == is_subtract ? -b.value : b.value;
        second_negative        = b.is_negative != is_subtract;
        second_infinity        = b.is_infinity;
        second_zero            = b.is_zero;
    }

    if (true == first_infinity && true == second_infinity && first_negative != second_negative)
    {
        sve->flags |= FPSR_IOC;
        return 0x7FC0U;
    }

    if (true == first_infinity || true == second_infinity)
    {
        const bool is_negative = true == first_infinity ? first_negative : second_negative;
        return true == is_negative ? 0xFF80U : 0x7F80U;
    }

    if (true == first_zero && true == second_zero && first_negative == second_negative)
    {
        return true == first_negative ? 0x8000U : 0U;
    }

    const double sum = sum_to_odd(first, second);
    return 0.0 == sum ? rounded_zero : bfloat_round(sve, sum);
}

/// The FEAT_SVE_B16B16 arithmetic on BFloat16 elements: BFADD, BFSUB and BFMUL unpredicated
/// (01100101 00 0 Zm 000 opc Zn Zd, opc 000, 001 and 010) and predicated (01100101 00 00 opc 100
/// Pg Zm Zdn, opc 0000, 0001 and 0010), and BFMAXNM, BFMINNM, BFMAX and BFMIN (predicated opc
/// 0100 to 0111).
static sve_result_t
bfloat_binary(sve_t *sve, const uint32_t instruction)
{
    static const bfloat_kind_t KINDS[3]      = { BFLOAT_ADD, BFLOAT_SUBTRACT, BFLOAT_MULTIPLY };
    const bool                 is_predicated = 0U != (instruction & 0x8000U);
    const uint32_t             opc
        = true == is_predicated ? (instruction >> 16U) & 0xFU : (instruction >> 10U) & 0x7U;

    if (3U == opc || opc > 7U || (false == is_predicated && opc > 2U))
    {
        return SVE_UNDEFINED;
    }

    const uint32_t d     = instruction & 0x1FU;
    const vector_t first = read_z(sve, true == is_predicated ? d : (instruction >> 5U) & 0x1FU);
    const vector_t second
        = read_z(sve, (instruction >> (true == is_predicated ? 5U : 16U)) & 0x1FU);
    const uint16_t pg = true == is_predicated ? read_p(sve, (instruction >> 10U) & 0x7U) : 0xFFFFU;
    vector_t       result = first;

    for (uint32_t e = 0U; e < 8U; ++e)
    {
        if (false == is_active(pg, e, 16U))
        {
            continue;
        }

        const uint64_t a = get_element(&first, e, 16U);
        const uint64_t b = get_element(&second, e, 16U);
        // Predicated opc 01xx: 0100 FMAXNM, 0101 FMINNM, 0110 FMAX, 0111 FMIN.
        static const uint32_t EXTREMES[4] = { 1U, 3U, 0U, 2U };
        set_element(&result,
                    e,
                    16U,
                    opc >= 4U ? float_extreme(sve, 0U, EXTREMES[opc - 4U], a, b)
                              : bfloat_compute(sve, KINDS[opc], 0U, a, b));
    }

    write_z(sve, d, result);
    return SVE_OK;
}

/// BFMLA and BFMLS: predicated (01100101 00 1 Zm 00 op Pg Zn Zda) and by element (01100100 0 i3h
/// 1 i3l Zm 0000 1 op Zn Zda), Zda + Zn * Zm with Zn negated for BFMLS, and BFMUL by element
/// (01100100 0 i3h 1 i3l Zm 001010 Zn Zd).
static sve_result_t
bfloat_multiply_add(sve_t *sve, const uint32_t instruction)
{
    const bool is_indexed  = 0x64000000U == (instruction & 0xFF000000U);
    const bool is_multiply = true == is_indexed && 0U != (instruction & 0x2000U);
    const bool is_subtract
        = false == is_multiply && 0U != (instruction & (true == is_indexed ? 0x400U : 0x2000U));
    const uint32_t index  = (((instruction >> 22U) & 1U) << 2U) | ((instruction >> 19U) & 0x3U);
    const uint32_t d      = instruction & 0x1FU;
    const vector_t n      = read_z(sve, (instruction >> 5U) & 0x1FU);
    const vector_t m      = read_z(sve, (instruction >> 16U) & (true == is_indexed ? 0x7U : 0x1FU));
    const uint16_t pg     = true == is_indexed ? 0xFFFFU : read_p(sve, (instruction >> 10U) & 0x7U);
    vector_t       result = read_z(sve, d);

    for (uint32_t e = 0U; e < 8U; ++e)
    {
        if (false == is_active(pg, e, 16U))
        {
            continue;
        }

        uint64_t       a = get_element(&n, e, 16U);
        const uint64_t b = get_element(&m, true == is_indexed ? index : e, 16U);
        a                = true == is_subtract ? a ^ 0x8000U : a;
        set_element(
            &result,
            e,
            16U,
            true == is_multiply
                ? bfloat_compute(sve, BFLOAT_MULTIPLY, 0U, a, b)
                : bfloat_compute(sve, BFLOAT_MULTIPLY_ADD, get_element(&result, e, 16U), a, b));
    }

    write_z(sve, d, result);
    return SVE_OK;
}

// -----------------------------------------------------------------------------
// FEAT_F64MM
// -----------------------------------------------------------------------------

/// The FEAT_F64MM instructions, which need vectors of at least 256 bits and are undefined at
/// this length: LD1ROB, LD1ROH, LD1ROW and LD1ROD (1010010 msz 01 0 imm4 001 Pg Rn Zt, and
/// 1010010 msz 01 Rm 000 ..., where Rm may not be XZR), FMMLA of doublewords, and ZIP1, ZIP2,
/// UZP1, UZP2, TRN1 and TRN2 of quadwords (00000101101 Zm 000 opc H Zn Zd, opc 10 unallocated).
static sve_result_t
needs_longer_vectors(sve_t *sve, const uint32_t instruction)
{
    (void)sve;

    const bool is_load_register       = 0xA4200000U == (instruction & 0xFE60E000U);
    const bool is_unallocated_permute = 0x05A01000U == (instruction & 0xFFE0F800U);

    if ((true == is_load_register && 31U == ((instruction >> 16U) & 0x1FU))
        || true == is_unallocated_permute)
    {
        return SVE_UNDEFINED;
    }

    return SVE_ILLEGAL;
}

static const sve_pattern_t PATTERNS[] = {
    { 0xFFA0F800U, 0x04205800U, streaming_vector_length },
    { 0xFFFFF800U, 0x04BF5800U, streaming_vector_length },
    { 0xFF20E000U, 0x04002000U, quadword_reduction },
    { 0xFFE0FC00U, 0x05202400U, duplicate_quadword },
    { 0xFFF0FC00U, 0x05602400U, extract_quadword },
    { 0xFF20E000U, 0x4400E000U, quadword_permute },
    { 0xFF20FC00U, 0x05203400U, table_extension_quadword },
    { 0xFFFFE000U, 0x052E8000U, reverse_doublewords },
    { 0xFF20F800U, 0x4400C000U, integer_clamp },
    { 0xFF60F800U, 0x4400C800U, two_way_dot_product },
    { 0xFFFFE420U, 0x45314000U, narrow_pair },
    { 0xFFF0CC20U, 0x45B00800U, narrow_pair },
    { 0xFF3FFFF8U, 0x25207810U, predicate_true_counter },
    { 0xFF3FFA00U, 0x25208200U, count_predicate_counter },
    { 0xFF3FF810U, 0x25207010U, predicate_extract },
    { 0xFF20C210U, 0x25204000U, predicate_select },
    { 0xFF20D010U, 0x25204010U, predicate_while_multiple },
    { 0xFF20F010U, 0x25205010U, predicate_while_multiple },
    { 0xFF38FC10U, 0x05283800U, predicate_move },
    { 0xFF38FE00U, 0x05293800U, predicate_move },
    { 0xFF70E000U, 0xA5102000U, quadword_element_memory },
    { 0xFF60E000U, 0xA5008000U, quadword_element_memory },
    { 0xFFF0E000U, 0xE500E000U, quadword_element_memory },
    { 0xFFF0E000U, 0xE5C0E000U, quadword_element_memory },
    { 0xFFE0E000U, 0xE5004000U, quadword_element_memory },
    { 0xFFE0E000U, 0xE5C04000U, quadword_element_memory },
    { 0xFE70E000U, 0xA410E000U, quadword_structures },
    { 0xFE60E000U, 0xA4208000U, quadword_structures },
    { 0xFF30E000U, 0xE4000000U, quadword_structures },
    { 0xFF20E000U, 0xE4200000U, quadword_structures },
    { 0xFFE0E000U, 0xC400A000U, quadword_gather_scatter },
    { 0xFFE0E000U, 0xE4202000U, quadword_gather_scatter },
    { 0xFE70E000U, 0xA4202000U, needs_longer_vectors },
    { 0xFE60E000U, 0xA4200000U, needs_longer_vectors },
    { 0xFFE0FC00U, 0x64E0E400U, needs_longer_vectors },
    { 0xFFE0E000U, 0x05A00000U, needs_longer_vectors },
    { 0xFF38E000U, 0x6410A000U, float_quadword_reduction },
    { 0xFF20FC00U, 0x64202400U, float_clamp },
    { 0xFFE0FC00U, 0x64208000U, half_dot_product },
    { 0xFFE0FC00U, 0x64204000U, half_dot_product },
    { 0xFFF0E000U, 0x65008000U, bfloat_binary },
    { 0xFFE0E000U, 0x65000000U, bfloat_binary },
    { 0xFFE0C000U, 0x65200000U, bfloat_multiply_add },
    { 0xFFA0F800U, 0x64200800U, bfloat_multiply_add },
    { 0xFFA0FC00U, 0x64202800U, bfloat_multiply_add },
};

sve_result_t
sve2p1_execute(sve_t *sve, const uint32_t instruction)
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

/*** end of file ***/
