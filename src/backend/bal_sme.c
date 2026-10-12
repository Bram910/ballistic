#include "backend/bal_simd.h"
#include "backend/bal_sve.h"
#include "backend/bal_sve_shared.h"

#include <string.h>

/// The FPSR value that entering or leaving streaming mode sets: the cumulative exception flags
/// and QC.
#define STREAMING_FPSR 0x0800009FULL

// -----------------------------------------------------------------------------
// Modes
// -----------------------------------------------------------------------------

/// SetPSTATE_SVCR(): changes PSTATE.SM and PSTATE.ZA, resetting the state each change resets.
static void
set_svcr(bal_cpu_t *cpu, const uint64_t value)
{
    if ((cpu->svcr ^ value) & SVCR_SM)
    {
        (void)memset(cpu->v, 0, sizeof(cpu->v));
        (void)memset(cpu->p, 0, sizeof(cpu->p));
        cpu->ffr  = 0U;
        cpu->fpsr = STREAMING_FPSR;
    }

    if (0U == (cpu->svcr & SVCR_ZA) && 0U != (value & SVCR_ZA))
    {
        (void)memset(cpu->za, 0, sizeof(cpu->za));
        (void)memset(cpu->zt0, 0, sizeof(cpu->zt0));
    }

    cpu->svcr = value & (SVCR_SM | SVCR_ZA);
}

uint64_t
bal_execute_svcr_write(bal_cpu_t *cpu, const uint32_t instruction)
{
    // MSR SVCR, Xt has bit 20 set; MSR SVCRSM, SVCRZA and SVCRSMZA, #imm take CRm = 0 ZA SM imm.
    if (0U != (instruction & 0x100000U))
    {
        const uint32_t rt = instruction & 0x1FU;
        set_svcr(cpu, 31U == rt ? 0U : cpu->x[rt]);
    }
    else
    {
        const uint64_t crm   = (instruction >> 8U) & 0xFU;
        const uint64_t mask  = (crm >> 1U) & (SVCR_SM | SVCR_ZA);
        const uint64_t value = 0U != (crm & 1U) ? mask : 0U;
        set_svcr(cpu, (cpu->svcr & ~mask) | value);
    }

    return 0U;
}

// -----------------------------------------------------------------------------
// ZA
// -----------------------------------------------------------------------------

/// The streaming vector length in bytes, which is also the number of ZA slices.
#define SVL_BYTES 16U

/// Whether the modes an instruction needs are on: ZA storage, and streaming mode when
/// `needs_streaming`.
static bool
is_enabled(const sve_t *sve, const bool needs_streaming)
{
    const uint64_t svcr = sve->cpu->svcr;
    return 0U != (svcr & SVCR_ZA) && (false == needs_streaming || 0U != (svcr & SVCR_SM));
}

/// The address of element `e` of slice `slice` of tile `tile`, whose elements are `bytes` wide:
/// horizontal slice s is ZA row s * bytes + tile, and vertical slice s is column s of those
/// rows.
static uint8_t *
tile_element(const sve_t   *sve,
             const uint32_t bytes,
             const uint32_t tile,
             const uint32_t slice,
             const bool     is_vertical,
             const uint32_t e)
{
    const uint32_t row    = (true == is_vertical ? e : slice) * bytes + tile;
    const uint32_t column = true == is_vertical ? slice : e;
    return &sve->cpu->za[row][column * bytes];
}

/// Reads an element of up to 8 bytes from ZA.
static uint64_t
read_za(const uint8_t *element, const uint32_t bytes)
{
    uint64_t value = 0U;
    (void)memcpy(&value, element, bytes);
    return value;
}

static void
write_za(uint8_t *element, const uint32_t bytes, const uint64_t value)
{
    (void)memcpy(element, &value, bytes);
}

/// A slice index: W12-W15 picked by `rv`, plus `offset`, modulo the number of slices.
static uint32_t
slice_index(const sve_t *sve, const uint32_t rv, const uint32_t offset, const uint32_t slices)
{
    return (uint32_t)((sve->cpu->x[12U + rv] + offset) % slices);
}

/// The tile and slice offset of the four-bit field at bits 3:0 of a tile slice instruction for
/// elements of `bytes` bytes: the tile takes the high bits and the offset the rest, all of it
/// for quadwords.
static void
tile_and_offset(const uint32_t field, const uint32_t bytes, uint32_t *tile, uint32_t *offset)
{
    const uint32_t tile_bits = (uint32_t)__builtin_ctz(bytes);
    *tile                    = 16U == bytes ? field : field >> (4U - tile_bits);
    *offset                  = 16U == bytes ? 0U : field & ((1U << (4U - tile_bits)) - 1U);
}

/// LD1B, LD1H, LD1W, LD1D and LD1Q (1110000 msz L Rm V Rs Pg Rn 0 ZAt:off) and ST1B, ST1H,
/// ST1W, ST1D and ST1Q (... msz 1 ...): one horizontal or vertical tile slice from or to memory
/// at Xn + Xm * size, with inactive elements zero when loading.
static sve_result_t
tile_slice_memory(sve_t *sve, const uint32_t instruction)
{
    const bool     is_quad  = 0xE1C00000U == (instruction & 0xFFC00000U);
    const uint32_t bytes    = true == is_quad ? 16U : 1U << ((instruction >> 22U) & 0x3U);
    const bool     is_store = 0U != (instruction & 0x200000U);

    if (false == is_enabled(sve, true))
    {
        return SVE_ILLEGAL;
    }

    uint32_t tile;
    uint32_t offset;
    tile_and_offset(instruction & 0xFU, bytes, &tile, &offset);
    const uint32_t slices      = SVL_BYTES / bytes;
    const bool     is_vertical = 0U != (instruction & 0x8000U);
    const uint32_t slice       = slice_index(sve, (instruction >> 13U) & 0x3U, offset, slices);
    const uint16_t pg          = read_p(sve, (instruction >> 10U) & 0x7U);
    const uint64_t base        = sve->cpu->x[(instruction >> 5U) & 0x1FU]
                          + read_x(sve, (instruction >> 16U) & 0x1FU) * bytes;

    for (uint32_t e = 0U; e < slices; ++e)
    {
        if (true == is_active(pg, e, 8U * bytes)
            && false == is_mapped(sve, base + e * bytes, bytes))
        {
            return SVE_FAULT;
        }
    }

    for (uint32_t e = 0U; e < slices; ++e)
    {
        uint8_t       *element = tile_element(sve, bytes, tile, slice, is_vertical, e);
        const bool     active  = is_active(pg, e, 8U * bytes);
        const uint64_t address = base + e * bytes;

        for (uint32_t i = 0U; i < bytes; i += 8U)
        {
            const uint32_t part = bytes < 8U ? bytes : 8U;

            if (true == is_store)
            {
                if (true == active)
                {
                    memory_write(sve, address + i, part, read_za(element + i, part));
                }
            }
            else
            {
                write_za(
                    element + i, part, true == active ? memory_read(sve, address + i, part) : 0U);
            }
        }
    }

    return SVE_OK;
}

/// LDR and STR of a ZA slice (1110000100 0 00000 0 Rv 000 Rn 0 off4, STR with bit 21) and of
/// ZT0 (11100001 0 0 011111 100000 Rn 00000).
static sve_result_t
za_vector_memory(sve_t *sve, const uint32_t instruction)
{
    const bool     is_store = 0U != (instruction & 0x200000U);
    const bool     is_zt0   = 0xE11F8000U == (instruction & 0xFFDFFC1FU);
    const uint64_t base     = sve->cpu->x[(instruction >> 5U) & 0x1FU];

    if (false == is_enabled(sve, false))
    {
        return SVE_ILLEGAL;
    }

    const uint32_t offset  = true == is_zt0 ? 0U : instruction & 0xFU;
    const uint32_t bytes   = true == is_zt0 ? sizeof(sve->cpu->zt0) : SVL_BYTES;
    const uint64_t address = base + offset * SVL_BYTES;
    uint8_t       *data
        = true == is_zt0
              ? sve->cpu->zt0
              : sve->cpu->za[slice_index(sve, (instruction >> 13U) & 0x3U, offset, SVL_BYTES)];

    if (false == is_mapped(sve, address, bytes))
    {
        return SVE_FAULT;
    }

    for (uint32_t i = 0U; i < bytes; i += 8U)
    {
        if (true == is_store)
        {
            memory_write(sve, address + i, 8U, read_za(data + i, 8U));
        }
        else
        {
            write_za(data + i, 8U, memory_read(sve, address + i, 8U));
        }
    }

    return SVE_OK;
}

/// ZERO { mask } (11000000 00001000 00000000 imm8), clearing the doubleword tiles in the mask,
/// and ZERO { ZT0 } (11000000 01001000 00000000 00000001).
static sve_result_t
zero_tiles(sve_t *sve, const uint32_t instruction)
{
    if (false == is_enabled(sve, false))
    {
        return SVE_ILLEGAL;
    }

    if (0xC0480001U == instruction)
    {
        (void)memset(sve->cpu->zt0, 0, sizeof(sve->cpu->zt0));
        return SVE_OK;
    }

    // Doubleword tile t is ZA rows t and t + 8.
    for (uint32_t row = 0U; row < SVL_BYTES; ++row)
    {
        if (0U != (instruction & (1U << (row % 8U))))
        {
            (void)memset(sve->cpu->za[row], 0, SVL_BYTES);
        }
    }

    return SVE_OK;
}

/// MOVA from a vector to a tile slice (11000000 size 00000 Q V Rs Pg Zn 0 ZAd:off) and from a
/// tile slice to a vector (11000000 size 00001 Q V Rs Pg 0 ZAn:off Zd), merging under Pg.
static sve_result_t
move_tile_vector(sve_t *sve, const uint32_t instruction)
{
    const uint32_t size      = (instruction >> 22U) & 0x3U;
    const bool     is_quad   = 3U == size && 0U != (instruction & 0x10000U);
    const bool     to_vector = 0U != (instruction & 0x20000U);
    const uint32_t bytes     = true == is_quad ? 16U : 1U << size;

    // Q is only allocated for doublewords.
    if (3U != size && 0U != (instruction & 0x10000U))
    {
        return SVE_UNDEFINED;
    }

    if (false == is_enabled(sve, true))
    {
        return SVE_ILLEGAL;
    }

    uint32_t tile;
    uint32_t offset;
    tile_and_offset(
        true == to_vector ? (instruction >> 5U) & 0xFU : instruction & 0xFU, bytes, &tile, &offset);
    const uint32_t slices      = SVL_BYTES / bytes;
    const bool     is_vertical = 0U != (instruction & 0x8000U);
    const uint32_t slice       = slice_index(sve, (instruction >> 13U) & 0x3U, offset, slices);
    const uint16_t pg          = read_p(sve, (instruction >> 10U) & 0x7U);
    const uint32_t z = true == to_vector ? instruction & 0x1FU : (instruction >> 5U) & 0x1FU;
    uint8_t        vector[SVL_BYTES];
    (void)memcpy(vector, sve->cpu->v[z], SVL_BYTES);

    for (uint32_t e = 0U; e < slices; ++e)
    {
        if (true == is_active(pg, e, 8U * bytes))
        {
            uint8_t *element = tile_element(sve, bytes, tile, slice, is_vertical, e);

            if (true == to_vector)
            {
                (void)memcpy(&vector[e * bytes], element, bytes);
            }
            else
            {
                (void)memcpy(element, &vector[e * bytes], bytes);
            }
        }
    }

    if (true == to_vector)
    {
        (void)memcpy(sve->cpu->v[z], vector, SVL_BYTES);
    }

    return SVE_OK;
}

/// ADDHA and ADDVA: 11000000 1 sz 01000 0 V Pm Pn Zn 0 0 ZAda, adding Zn to every row (ADDHA)
/// or column (ADDVA) of a word or doubleword tile, where Pn allows the row and Pm the column.
static sve_result_t
add_to_tile(sve_t *sve, const uint32_t instruction)
{
    const bool     is_double   = 0U != (instruction & 0x400000U);
    const uint32_t bytes       = true == is_double ? 8U : 4U;
    const bool     is_vertical = 0U != (instruction & 0x10000U);

    if (false == is_double && 0U != (instruction & 0x4U))
    {
        return SVE_UNDEFINED;
    }

    if (false == is_enabled(sve, true))
    {
        return SVE_ILLEGAL;
    }

    const uint32_t tile   = instruction & (true == is_double ? 0x7U : 0x3U);
    const uint32_t slices = SVL_BYTES / bytes;
    const uint16_t pn     = read_p(sve, (instruction >> 10U) & 0x7U);
    const uint16_t pm     = read_p(sve, (instruction >> 13U) & 0x7U);
    const vector_t n      = read_z(sve, (instruction >> 5U) & 0x1FU);

    for (uint32_t row = 0U; row < slices; ++row)
    {
        for (uint32_t column = 0U; column < slices; ++column)
        {
            if (true == is_active(pn, row, 8U * bytes) && true == is_active(pm, column, 8U * bytes))
            {
                uint8_t       *element = tile_element(sve, bytes, tile, row, false, column);
                const uint64_t addend
                    = get_element(&n, true == is_vertical ? row : column, 8U * bytes);
                write_za(element, bytes, read_za(element, bytes) + addend);
            }
        }
    }

    return SVE_OK;
}

/// The FPCR of the ZA floating-point operations, which produce default NaNs.
#define FPCR_DN (1ULL << 25U)

/// FMOPA and FMOPS on single (10000000 100 Zm Pm Pn Zn S 00 ZAda) and double precision (10000000
/// 110 Zm Pm Pn Zn S 0 ZAda): the outer product of Zn and Zm added to or subtracted from a tile,
/// where Pn allows the row and Pm the column. The exceptions are not recorded in FPSR.
static sve_result_t
float_outer_product(sve_t *sve, const uint32_t instruction)
{
    const bool     is_double   = 0U != (instruction & 0x400000U);
    const uint32_t esize       = true == is_double ? 64U : 32U;
    const bool     is_subtract = 0U != (instruction & 0x10U);

    if (false == is_enabled(sve, true))
    {
        return SVE_ILLEGAL;
    }

    const uint32_t bytes  = esize / 8U;
    const uint32_t tile   = instruction & (true == is_double ? 0x7U : 0x3U);
    const uint32_t slices = SVL_BYTES / bytes;
    const uint16_t pn     = read_p(sve, (instruction >> 10U) & 0x7U);
    const uint16_t pm     = read_p(sve, (instruction >> 13U) & 0x7U);
    const vector_t n      = read_z(sve, (instruction >> 5U) & 0x1FU);
    const vector_t m      = read_z(sve, (instruction >> 16U) & 0x1FU);
    const uint64_t fpcr   = sve->cpu->fpcr | FPCR_DN;
    uint64_t       flags  = 0U;

    for (uint32_t row = 0U; row < slices; ++row)
    {
        for (uint32_t column = 0U; column < slices; ++column)
        {
            if (true == is_active(pn, row, esize) && true == is_active(pm, column, esize))
            {
                uint8_t *element = tile_element(sve, bytes, tile, row, false, column);
                uint64_t a       = get_element(&n, row, esize);
                a                = true == is_subtract ? fp_negate(a, esize) : a;
                write_za(element,
                         bytes,
                         bal_float_element(fpcr,
                                           &flags,
                                           esize,
                                           BAL_FLOAT_MULTIPLY_ADD,
                                           a,
                                           get_element(&m, column, esize),
                                           read_za(element, bytes)));
            }
        }
    }

    return SVE_OK;
}

/// SMOPA, SMOPS, SUMOPA, SUMOPS, USMOPA, USMOPS, UMOPA and UMOPS on four-way bytes into words
/// (1010000 u0 10 u1 Zm Pm Pn Zn S 00 ZAda) and halfwords into doublewords (... 11 ... S 0
/// ZAda): each tile element gets the sum of four products of its row's Zn elements and its
/// column's Zm elements, where Pn and Pm allow each narrow element.
static sve_result_t
integer_outer_product(sve_t *sve, const uint32_t instruction)
{
    const bool     is_double   = 0U != (instruction & 0x400000U);
    const bool     is_subtract = 0U != (instruction & 0x10U);
    const bool     n_unsigned  = 0U != (instruction & 0x1000000U);
    const bool     m_unsigned  = 0U != (instruction & 0x200000U);
    const uint32_t bytes       = true == is_double ? 8U : 4U;
    const uint32_t small       = 2U * bytes;

    if (false == is_enabled(sve, true))
    {
        return SVE_ILLEGAL;
    }

    const uint32_t tile   = instruction & (true == is_double ? 0x7U : 0x3U);
    const uint32_t slices = SVL_BYTES / bytes;
    const uint16_t pn     = read_p(sve, (instruction >> 10U) & 0x7U);
    const uint16_t pm     = read_p(sve, (instruction >> 13U) & 0x7U);
    const vector_t n      = read_z(sve, (instruction >> 5U) & 0x1FU);
    const vector_t m      = read_z(sve, (instruction >> 16U) & 0x1FU);

    for (uint32_t row = 0U; row < slices; ++row)
    {
        for (uint32_t column = 0U; column < slices; ++column)
        {
            uint8_t *element = tile_element(sve, bytes, tile, row, false, column);
            uint64_t sum     = read_za(element, bytes);

            for (uint32_t k = 0U; k < 4U; ++k)
            {
                if (true == is_active(pn, 4U * row + k, small)
                    && true == is_active(pm, 4U * column + k, small))
                {
                    const __int128 product
                        = wide(get_element(&n, 4U * row + k, small), small, false == n_unsigned)
                          * wide(
                              get_element(&m, 4U * column + k, small), small, false == m_unsigned);
                    sum = true == is_subtract ? sum - (uint64_t)product : sum + (uint64_t)product;
                }
            }

            write_za(element, bytes, sum);
        }
    }

    return SVE_OK;
}

/// The operand pair of a widening outer product at `pair`: its two narrow elements, each zero
/// when its predicate element is inactive. Returns a bit per active element.
static uint32_t
outer_pair(const vector_t *vector,
           const uint16_t  predicate,
           const uint32_t  pair,
           const uint32_t  narrow,
           uint64_t        halves[2])
{
    uint32_t active = 0U;

    for (uint32_t k = 0U; k < 2U; ++k)
    {
        const bool is_on = is_active(predicate, 2U * pair + k, narrow);
        halves[k]        = true == is_on ? get_element(vector, 2U * pair + k, narrow) : 0U;
        active |= true == is_on ? 1U << k : 0U;
    }

    return active;
}

/// A single precision value as a double, exactly.
static double
single_to_double(const uint64_t bits)
{
    float          value;
    const uint32_t word = (uint32_t)bits;
    (void)memcpy(&value, &word, sizeof(value));
    return (double)value;
}

/// Rounds the exact value `scaled` * 2^-48 to single precision in rounding mode `rmode` (FPCR
/// RMode). Its magnitude is below 2^82, so the result is a normal number or zero.
static uint64_t
round_fixed_to_single(const __int128 scaled, const uint32_t rmode)
{
    const uint64_t    sign = scaled < 0 ? 0x80000000U : 0U;
    unsigned __int128 magnitude
        = scaled < 0 ? (unsigned __int128)(-scaled) : (unsigned __int128)scaled;
    int32_t msb = 127;

    while (0U == ((magnitude >> msb) & 1U))
    {
        --msb;
    }

    // Keep 24 significant bits; the rest decides the rounding.
    const int32_t           shift    = msb - 23;
    unsigned __int128       mantissa = shift > 0 ? magnitude >> shift : magnitude << -shift;
    const unsigned __int128 remainder
        = shift > 0 ? magnitude & (((unsigned __int128)1 << shift) - 1U) : 0U;
    const unsigned __int128 half     = shift > 0 ? (unsigned __int128)1 << (shift - 1) : 0U;
    bool                    round_up = false;

    if (0U != remainder)
    {
        switch (rmode)
        {
            case 0U: // to nearest, ties to even
                round_up = remainder > half || (remainder == half && 0U != (mantissa & 1U));
                break;
            case 1U: // towards plus infinity
                round_up = 0U == sign;
                break;
            case 2U: // towards minus infinity
                round_up = 0U != sign;
                break;
            default: // towards zero
                break;
        }
    }

    int32_t exponent = msb - 48;
    mantissa += true == round_up ? 1U : 0U;

    if (0U != (mantissa >> 24U))
    {
        mantissa >>= 1U;
        ++exponent;
    }

    return sign | ((uint64_t)(exponent + 127) << 23U) | ((uint64_t)mantissa & 0x7FFFFFU);
}

/// The dot product of two half precision pairs, a0 * b0 + a1 * b1, rounded once to single
/// precision, with default NaNs. Denormal inputs flush to zero under FPCR.FZ16.
static uint64_t
half_dot(const uint64_t fpcr, const uint64_t a[2], const uint64_t b[2])
{
    double   products[2];
    bool     is_nan               = false;
    bool     is_infinite          = false;
    bool     infinity_negative[2] = { false, false };
    uint32_t infinities           = 0U;

    for (uint32_t k = 0U; k < 2U; ++k)
    {
        const double x = single_to_double(bal_simd_half_to_single(fpcr, a[k]));
        const double y = single_to_double(bal_simd_half_to_single(fpcr, b[k]));

        if (x != x || y != y
            || ((x == 0.0 || y == 0.0) && (__builtin_isinf(x) || __builtin_isinf(y))))
        {
            is_nan = true;
        }
        else if (__builtin_isinf(x) || __builtin_isinf(y))
        {
            infinity_negative[infinities++] = (x < 0.0) != (y < 0.0);
            is_infinite                     = true;
        }

        products[k] = x * y;
    }

    if (true == is_nan || (2U == infinities && infinity_negative[0] != infinity_negative[1]))
    {
        return 0x7FC00000U;
    }

    if (true == is_infinite)
    {
        return (true == infinity_negative[0] ? 0x80000000U : 0U) | 0x7F800000U;
    }

    const __int128 sum
        = (__int128)__builtin_ldexp(products[0], 48) + (__int128)__builtin_ldexp(products[1], 48);
    const uint32_t rmode = (uint32_t)(fpcr >> 22U) & 0x3U;

    if (0 == sum)
    {
        // Two zero products keep their common sign; anything else that cancels is +0, or -0
        // when rounding towards minus infinity.
        const bool negative0 = __builtin_signbit(products[0]) != 0;
        const bool negative1 = __builtin_signbit(products[1]) != 0;
        const bool both_zero = 0.0 == products[0] && 0.0 == products[1];
        const bool negative  = true == both_zero ? (negative0 && negative1)
                                                      || (negative0 != negative1 && 2U == rmode)
                                                 : 2U == rmode;
        return true == negative ? 0x80000000U : 0U;
    }

    return round_fixed_to_single(sum, rmode);
}

/// The widening outer products into single precision tiles: FMOPA and FMOPS on half precision
/// (10000001 101 Zm Pm Pn Zn S 00 ZAda) and BFMOPA and BFMOPS on BFloat16 (10000001 100 ...).
/// Each tile element gets the dot product of a pair of its row's Zn elements and its column's
/// Zm elements, with inactive elements zero.
static sve_result_t
widening_float_outer_product(sve_t *sve, const uint32_t instruction)
{
    const bool is_half     = 0U != (instruction & 0x200000U);
    const bool is_subtract = 0U != (instruction & 0x10U);

    if (false == is_enabled(sve, true))
    {
        return SVE_ILLEGAL;
    }

    const uint32_t tile  = instruction & 0x3U;
    const uint16_t pn    = read_p(sve, (instruction >> 10U) & 0x7U);
    const uint16_t pm    = read_p(sve, (instruction >> 13U) & 0x7U);
    const vector_t n     = read_z(sve, (instruction >> 5U) & 0x1FU);
    const vector_t m     = read_z(sve, (instruction >> 16U) & 0x1FU);
    const uint64_t fpcr  = sve->cpu->fpcr | FPCR_DN;
    uint64_t       flags = 0U;

    for (uint32_t row = 0U; row < 4U; ++row)
    {
        for (uint32_t column = 0U; column < 4U; ++column)
        {
            uint64_t a[2];
            uint64_t b[2];

            // An element changes when a pair of row and column elements are both active.
            if (0U == (outer_pair(&n, pn, row, 16U, a) & outer_pair(&m, pm, column, 16U, b)))
            {
                continue;
            }

            a[0] ^= true == is_subtract ? 0x8000U : 0U;
            a[1] ^= true == is_subtract ? 0x8000U : 0U;
            uint8_t       *element = tile_element(sve, 4U, tile, row, false, column);
            const uint64_t sum     = read_za(element, 4U);
            uint64_t       result;

            if (true == is_half)
            {
                result = bal_float_element(
                    fpcr, &flags, 32U, BAL_FLOAT_ADD, sum, half_dot(fpcr, a, b), 0U);
            }
            else
            {
                result = bal_simd_bfloat_dot_add(
                    (uint32_t)sum, (uint16_t)a[0], (uint16_t)a[1], (uint16_t)b[0], (uint16_t)b[1]);
            }

            write_za(element, 4U, result);
        }
    }

    return SVE_OK;
}

/// FMOPA and FMOPS on half precision tiles (10000001 100 Zm Pm Pn Zn S 100 ZAda): a fused
/// multiply-add per element, where Pn allows the row and Pm the column.
static sve_result_t
half_outer_product(sve_t *sve, const uint32_t instruction)
{
    const bool is_subtract = 0U != (instruction & 0x10U);

    if (false == is_enabled(sve, true))
    {
        return SVE_ILLEGAL;
    }

    const uint32_t tile  = instruction & 0x1U;
    const uint16_t pn    = read_p(sve, (instruction >> 10U) & 0x7U);
    const uint16_t pm    = read_p(sve, (instruction >> 13U) & 0x7U);
    const vector_t n     = read_z(sve, (instruction >> 5U) & 0x1FU);
    const vector_t m     = read_z(sve, (instruction >> 16U) & 0x1FU);
    const uint64_t fpcr  = sve->cpu->fpcr | FPCR_DN;
    uint64_t       flags = 0U;

    for (uint32_t row = 0U; row < 8U; ++row)
    {
        for (uint32_t column = 0U; column < 8U; ++column)
        {
            if (true == is_active(pn, row, 16U) && true == is_active(pm, column, 16U))
            {
                uint8_t *element = tile_element(sve, 2U, tile, row, false, column);
                uint64_t a       = get_element(&n, row, 16U);
                a                = true == is_subtract ? a ^ 0x8000U : a;
                write_za(element,
                         2U,
                         bal_float_element(fpcr,
                                           &flags,
                                           16U,
                                           BAL_FLOAT_MULTIPLY_ADD,
                                           a,
                                           get_element(&m, column, 16U),
                                           read_za(element, 2U)));
            }
        }
    }

    return SVE_OK;
}

/// BMOPA and BMOPS (10000000 100 Zm Pm Pn Zn S 10 ZAda): adds to or subtracts from each word of
/// a tile the number of equal bits of its row's Zn word and its column's Zm word.
static sve_result_t
bit_outer_product(sve_t *sve, const uint32_t instruction)
{
    const bool is_subtract = 0U != (instruction & 0x10U);

    if (false == is_enabled(sve, true))
    {
        return SVE_ILLEGAL;
    }

    const uint32_t tile = instruction & 0x3U;
    const uint16_t pn   = read_p(sve, (instruction >> 10U) & 0x7U);
    const uint16_t pm   = read_p(sve, (instruction >> 13U) & 0x7U);
    const vector_t n    = read_z(sve, (instruction >> 5U) & 0x1FU);
    const vector_t m    = read_z(sve, (instruction >> 16U) & 0x1FU);

    for (uint32_t row = 0U; row < 4U; ++row)
    {
        for (uint32_t column = 0U; column < 4U; ++column)
        {
            if (true == is_active(pn, row, 32U) && true == is_active(pm, column, 32U))
            {
                uint8_t       *element = tile_element(sve, 4U, tile, row, false, column);
                const uint32_t same
                    = ~(uint32_t)(get_element(&n, row, 32U) ^ get_element(&m, column, 32U));
                const uint64_t count = (uint64_t)__builtin_popcount(same);
                const uint64_t sum   = read_za(element, 4U);
                write_za(element, 4U, true == is_subtract ? sum - count : sum + count);
            }
        }
    }

    return SVE_OK;
}

/// The two-way SMOPA, SMOPS, UMOPA and UMOPS on halfwords into words (1010000 U 100 Zm Pm Pn Zn
/// S 10 ZAda): each word of a tile gets the sum of two products of its row's Zn halfwords and
/// its column's Zm halfwords, where Pn and Pm allow each halfword.
static sve_result_t
integer_outer_product_pairs(sve_t *sve, const uint32_t instruction)
{
    const bool is_unsigned = 0U != (instruction & 0x1000000U);
    const bool is_subtract = 0U != (instruction & 0x10U);

    if (false == is_enabled(sve, true))
    {
        return SVE_ILLEGAL;
    }

    const uint32_t tile = instruction & 0x3U;
    const uint16_t pn   = read_p(sve, (instruction >> 10U) & 0x7U);
    const uint16_t pm   = read_p(sve, (instruction >> 13U) & 0x7U);
    const vector_t n    = read_z(sve, (instruction >> 5U) & 0x1FU);
    const vector_t m    = read_z(sve, (instruction >> 16U) & 0x1FU);

    for (uint32_t row = 0U; row < 4U; ++row)
    {
        for (uint32_t column = 0U; column < 4U; ++column)
        {
            uint8_t *element = tile_element(sve, 4U, tile, row, false, column);
            uint64_t sum     = read_za(element, 4U);

            for (uint32_t k = 0U; k < 2U; ++k)
            {
                if (true == is_active(pn, 2U * row + k, 16U)
                    && true == is_active(pm, 2U * column + k, 16U))
                {
                    const __int128 product
                        = wide(get_element(&n, 2U * row + k, 16U), 16U, false == is_unsigned)
                          * wide(get_element(&m, 2U * column + k, 16U), 16U, false == is_unsigned);
                    sum = true == is_subtract ? sum - (uint64_t)product : sum + (uint64_t)product;
                }
            }

            write_za(element, 4U, sum);
        }
    }

    return SVE_OK;
}

/// Copies tile slice `slice` of tile `tile` to `out`, or from `in` when `in` is not NULL.
static void
access_slice(const sve_t   *sve,
             const uint32_t bytes,
             const uint32_t tile,
             const uint32_t slice,
             const bool     is_vertical,
             uint8_t        out[SVL_BYTES],
             const uint8_t *in)
{
    for (uint32_t e = 0U; e < SVL_BYTES / bytes; ++e)
    {
        uint8_t *element = tile_element(sve, bytes, tile, slice, is_vertical, e);

        if (NULL != in)
        {
            (void)memcpy(element, &in[e * bytes], bytes);
        }
        else
        {
            (void)memcpy(&out[e * bytes], element, bytes);
        }
    }
}

/// The first ZA array vector of a vector group: (W8-W11 picked by `rv`, plus `offset`) modulo
/// the stride between the groups' vectors, 16 divided by `groups`.
static uint32_t
array_vector(const sve_t *sve, const uint32_t rv, const uint32_t offset, const uint32_t groups)
{
    return (uint32_t)((uint32_t)(sve->cpu->x[8U + rv] + offset) % (SVL_BYTES / groups));
}

/// MOVA and MOVAZ between two or four tile slices and vectors, and between ZA array vector
/// groups and vectors: 11000000 size 0001 D 0 V Rs 0 G N ... with D set from ZA to vectors, N
/// set for four registers, and G set for the ZA array forms (size 00, V 0); MOVAZ (from ZA,
/// bits 9:8 = 10) zeroes what it reads.
static sve_result_t
move_multiple(sve_t *sve, const uint32_t instruction)
{
    const uint32_t size       = (instruction >> 22U) & 0x3U;
    const bool     to_vector  = 0U != (instruction & 0x20000U);
    const bool     is_array   = 0U != (instruction & 0x800U);
    const uint32_t nreg       = 0U != (instruction & 0x400U) ? 4U : 2U;
    const bool     is_zeroing = true == to_vector && 0U != (instruction & 0x200U);
    const uint32_t bytes      = 1U << size;
    const uint32_t field      = true == to_vector ? (instruction >> 5U) & 0x7U : instruction & 0x7U;
    // The tile and offset field: three bits for two registers or doubleword tiles, two otherwise.
    const uint32_t width = 2U == nreg || 8U == bytes ? 3U : 2U;

    // Vector numbers are multiples of the register count, and the to-ZA forms have bits 5:3
    // clear.
    const uint32_t vector_bits
        = true == to_vector ? instruction & (nreg - 1U) : (instruction >> 5U) & (nreg - 1U);

    if ((true == is_array && (0U != size || 0U != (instruction & 0x8000U)))
        || (true == to_vector && 0U != (instruction & 0x100U)) || 0U != vector_bits
        || (false == to_vector && 0U != (instruction & 0x38U))
        || (false == is_array && 3U != width && 0U != (field & 0x4U))
        || (false == is_array && 4U == nreg && 8U == bytes))
    {
        return SVE_UNDEFINED;
    }

    if (false == is_enabled(sve, true))
    {
        return SVE_ILLEGAL;
    }

    const uint32_t z     = true == to_vector ? instruction & 0x1FU : (instruction >> 5U) & 0x1FU;
    const uint32_t first = z & ~(nreg - 1U);
    uint8_t        vectors[4][SVL_BYTES];

    if (false == to_vector)
    {
        for (uint32_t r = 0U; r < nreg; ++r)
        {
            (void)memcpy(vectors[r], sve->cpu->v[first + r], SVL_BYTES);
        }
    }

    for (uint32_t r = 0U; r < nreg; ++r)
    {
        if (true == is_array)
        {
            const uint32_t vector = array_vector(sve, (instruction >> 13U) & 0x3U, field, nreg)
                                    + r * (SVL_BYTES / nreg);

            if (true == to_vector)
            {
                (void)memcpy(vectors[r], sve->cpu->za[vector], SVL_BYTES);
            }
            else
            {
                (void)memcpy(sve->cpu->za[vector], vectors[r], SVL_BYTES);
            }

            if (true == is_zeroing)
            {
                (void)memset(sve->cpu->za[vector], 0, SVL_BYTES);
            }

            continue;
        }

        const uint32_t tile_bits = (uint32_t)__builtin_ctz(bytes);
        const uint32_t tile      = field >> (width - (tile_bits < width ? tile_bits : width));
        const uint32_t offset
            = tile_bits >= width ? 0U : (field & ((1U << (width - tile_bits)) - 1U)) * nreg;
        const uint32_t slices   = SVL_BYTES / bytes;
        const uint64_t index    = sve->cpu->x[12U + ((instruction >> 13U) & 0x3U)] & 0xFFFFFFFFU;
        const uint32_t slice    = (uint32_t)((index - index % nreg + offset) % slices) + r;
        const bool     vertical = 0U != (instruction & 0x8000U);
        access_slice(sve,
                     bytes,
                     tile_bits >= width ? field : tile,
                     slice,
                     vertical,
                     vectors[r],
                     true == to_vector ? NULL : vectors[r]);

        if (true == is_zeroing)
        {
            static const uint8_t ZEROS[SVL_BYTES] = { 0U };
            access_slice(
                sve, bytes, tile_bits >= width ? field : tile, slice, vertical, NULL, ZEROS);
        }
    }

    if (true == to_vector)
    {
        for (uint32_t r = 0U; r < nreg; ++r)
        {
            (void)memcpy(sve->cpu->v[first + r], vectors[r], SVL_BYTES);
        }
    }

    return SVE_OK;
}

/// MOVAZ to one vector: 11000000 size 00001 Q V Rs 000 1 ZAn:off Zd, which zeroes the slice it
/// reads.
static sve_result_t
move_zeroing(sve_t *sve, const uint32_t instruction)
{
    const uint32_t size    = (instruction >> 22U) & 0x3U;
    const bool     is_quad = 0U != (instruction & 0x10000U);

    if (true == is_quad && 3U != size)
    {
        return SVE_UNDEFINED;
    }

    if (false == is_enabled(sve, true))
    {
        return SVE_ILLEGAL;
    }

    const uint32_t bytes = true == is_quad ? 16U : 1U << size;
    uint32_t       tile;
    uint32_t       offset;
    tile_and_offset((instruction >> 5U) & 0xFU, bytes, &tile, &offset);
    const uint32_t       slices   = SVL_BYTES / bytes;
    const bool           vertical = 0U != (instruction & 0x8000U);
    const uint32_t       slice    = slice_index(sve, (instruction >> 13U) & 0x3U, offset, slices);
    static const uint8_t ZEROS[SVL_BYTES] = { 0U };
    uint8_t              vector[SVL_BYTES];
    access_slice(sve, bytes, tile, slice, vertical, vector, NULL);
    access_slice(sve, bytes, tile, slice, vertical, NULL, ZEROS);
    (void)memcpy(sve->cpu->v[instruction & 0x1FU], vector, SVL_BYTES);
    return SVE_OK;
}

/// MOVT between a general register and a doubleword of ZT0: 11000000 0100 11 D 0 off3 0011111
/// Rt, from ZT0 with D clear.
static sve_result_t
move_table(sve_t *sve, const uint32_t instruction)
{
    if (false == is_enabled(sve, false))
    {
        return SVE_ILLEGAL;
    }

    const uint32_t offset = 8U * ((instruction >> 12U) & 0x7U);
    const uint32_t t      = instruction & 0x1FU;

    if (0U != (instruction & 0x20000U))
    {
        write_za(&sve->cpu->zt0[offset], 8U, read_x(sve, t));
    }
    else
    {
        write_x(sve, t, read_za(&sve->cpu->zt0[offset], 8U));
    }

    return SVE_OK;
}

/// ZERO of ZA array vector groups (SME2.1): 11000000 00001 1 opc<2:1> opc<0> Rv 000 000000 0 off,
/// zeroing `nvec` consecutive vectors in each of `groups` groups.
static sve_result_t
zero_vectors(sve_t *sve, const uint32_t instruction)
{
    static const uint8_t GROUPS[8]  = { 2U, 1U, 2U, 4U, 4U, 1U, 2U, 4U };
    static const uint8_t VECTORS[8] = { 1U, 2U, 2U, 2U, 1U, 4U, 4U, 4U };
    const uint32_t       opc = (((instruction >> 16U) & 0x3U) << 1U) | ((instruction >> 15U) & 1U);
    const uint32_t       groups = GROUPS[opc];
    const uint32_t       nvec   = VECTORS[opc];
    // The offset field narrows as the groups grow: off3, off3:0, off2:0, o1:00 and so on.
    const uint32_t field_bits
        = 1U == nvec ? 3U : (2U == nvec ? (1U == groups ? 3U : 2U) : (1U == groups ? 2U : 1U));
    const uint32_t field = instruction & 0x7U;

    if (0U != (field >> field_bits))
    {
        return SVE_UNDEFINED;
    }

    if (false == is_enabled(sve, true))
    {
        return SVE_ILLEGAL;
    }

    const uint32_t stride = SVL_BYTES / groups;
    uint32_t       vector = array_vector(sve, (instruction >> 13U) & 0x3U, field * nvec, groups);
    vector -= vector % nvec;

    for (uint32_t r = 0U; r < groups; ++r)
    {
        for (uint32_t i = 0U; i < nvec; ++i)
        {
            (void)memset(sve->cpu->za[vector + i], 0, SVL_BYTES);
        }

        vector += stride;
    }

    return SVE_OK;
}

/// LUTI2 and LUTI4: elements of the 2-bit or 4-bit indexes in a segment of Zn look up words of
/// ZT0, truncated to the element size, into one (11000000 11001 ...), two or four consecutive
/// (11000000 10001 ...) or two or four strided (11000000 10011 ...) vectors.
static sve_result_t
table_lookup_zt0(sve_t *sve, const uint32_t instruction)
{
    const bool     is_single  = 0U != (instruction & 0x400000U);
    const bool     is_strided = false == is_single && 0U != (instruction & 0x100000U);
    const bool     is_two_bit = 0U != (instruction & 0x40000U);
    const uint32_t isize      = true == is_two_bit ? 2U : 4U;
    const uint32_t size       = (instruction >> 12U) & 0x3U;
    // Bit 14 marks two registers and bits 15:14 = 10 four.
    const uint32_t nreg = true == is_single ? 1U : (0U != (instruction & 0x4000U) ? 2U : 4U);
    uint32_t       imm;

    if (true == is_single)
    {
        imm = true == is_two_bit ? (instruction >> 14U) & 0xFU : (instruction >> 14U) & 0x7U;
    }
    else if (2U == nreg)
    {
        imm = true == is_two_bit ? (instruction >> 15U) & 0x7U : (instruction >> 15U) & 0x3U;
    }
    else
    {
        imm = true == is_two_bit ? (instruction >> 16U) & 0x3U : (instruction >> 16U) & 0x1U;
    }

    // The unallocated sizes: doublewords; for four registers of 4-bit indexes, bytes; and for
    // the strided forms, all but bytes and halfwords, and all but halfwords for four of 4-bit.
    const bool is_unallocated
        = 3U == size || (4U == nreg && false == is_two_bit && 0U == size)
          || (true == is_strided
              && (2U == size || (4U == nreg && false == is_two_bit && 1U != size)));
    const uint32_t misaligned_bits = true == is_strided ? (2U == nreg ? 0x8U : 0xCU) : nreg - 1U;
    const bool     is_misaligned   = false == is_single && 0U != (instruction & misaligned_bits);

    // The single register forms need bits 11:10 clear, like the others.
    if (true == is_unallocated || true == is_misaligned || 0U != (instruction & 0xC00U)
        || (false == is_single && 4U == nreg && 0U == (instruction & 0x8000U)))
    {
        return SVE_UNDEFINED;
    }

    if (false == is_enabled(sve, true))
    {
        return SVE_ILLEGAL;
    }

    const uint32_t esize    = 8U << size;
    const uint32_t elements = VECTOR_BITS / esize;
    const uint32_t segments = esize / (isize * nreg);
    const uint32_t segment  = imm % segments;
    const vector_t indexes  = read_z(sve, (instruction >> 5U) & 0x1FU);
    const uint32_t field    = instruction & 0x1FU;
    // Strided registers: D:0:Zd for two (stride 8) and D:00:Zd for four (stride 4).
    uint32_t       d = true == is_strided ? (field & 0x10U) | (field & (2U == nreg ? 0x7U : 0x3U))
                                          : (field & ~(nreg - 1U));
    const uint32_t dstride = true == is_strided ? (2U == nreg ? 8U : 4U) : 1U;

    for (uint32_t r = 0U; r < nreg; ++r)
    {
        const uint32_t base   = (segment * nreg + r) * elements;
        vector_t       result = read_z(sve, d);

        for (uint32_t e = 0U; e < elements; ++e)
        {
            const uint32_t index = (uint32_t)get_element(&indexes, base + e, isize);
            set_element(&result, e, esize, read_za(&sve->cpu->zt0[4U * index], 4U));
        }

        write_z(sve, d, result);
        d += dstride;
    }

    return SVE_OK;
}

/// CounterToPredicate(): the predicate a predicate-as-counter register describes, for four
/// vectors, one 16-bit predicate per 16 bits of the result. The low set bit of bits 3:0 gives
/// the element size, the bits above it up to bit 6 the number of leading active elements, and
/// bit 15 inverts.
static uint64_t
counter_to_predicate(const uint16_t counter)
{
    if (0U == (counter & 0xFU))
    {
        return 0U;
    }

    const uint32_t shift  = (uint32_t)__builtin_ctz(counter & 0xFU);
    const uint32_t psize  = 1U << shift;
    const uint32_t count  = (counter >> (shift + 1U)) & ((1U << (6U - shift)) - 1U);
    const bool     invert = 0U != (counter & 0x8000U);
    uint64_t       result = 0U;

    for (uint32_t e = 0U; e < 64U / psize; ++e)
    {
        if ((e < count) != invert)
        {
            result |= 1ULL << (e * psize);
        }
    }

    return result;
}

/// The SME2 multi-vector contiguous loads and stores: LD1, LDNT1, ST1 and STNT1 of two or four
/// consecutive (1010000 0 0 1 S imm4, or 0 0 S Rm, ...) or strided (1010000 1 ...) vectors,
/// governed by a predicate-as-counter register PN8-PN15. Inactive elements load as zero.
static sve_result_t
multi_vector_memory(sve_t *sve, const uint32_t instruction)
{
    const bool     is_strided   = 0U != (instruction & 0x1000000U);
    const bool     is_immediate = 0U != (instruction & 0x400000U);
    const bool     is_store     = 0U != (instruction & 0x200000U);
    const uint32_t nreg         = 0U != (instruction & 0x8000U) ? 4U : 2U;
    const uint32_t mbytes       = 1U << ((instruction >> 13U) & 0x3U);
    const uint32_t field        = instruction & 0x1FU;
    uint32_t       t;
    uint32_t       stride;

    if (true == is_immediate && 0U != (instruction & 0x100000U))
    {
        return SVE_UNDEFINED;
    }

    if (true == is_strided)
    {
        // T:0:Zt for two registers (stride 8) and T:00:Zt for four (stride 4), N in bit 3.
        if (4U == nreg && 0U != (instruction & 0x4U))
        {
            return SVE_UNDEFINED;
        }

        t      = (field & 0x10U) | (field & (2U == nreg ? 0x7U : 0x3U));
        stride = 2U == nreg ? 8U : 4U;
    }
    else
    {
        // Zt:0 or Zt:00, N in bit 0.
        if (4U == nreg && 0U != (instruction & 0x2U))
        {
            return SVE_UNDEFINED;
        }

        t      = field & (2U == nreg ? 0x1EU : 0x1CU);
        stride = 1U;
    }

    // The consecutive forms are also SVE2.1 instructions, which do not need streaming mode.
    if (true == is_strided && 0U == (sve->cpu->svcr & SVCR_SM))
    {
        return SVE_ILLEGAL;
    }

    const uint32_t esize    = 8U * mbytes;
    const uint32_t elements = SVL_BYTES / mbytes;
    const uint64_t mask     = counter_to_predicate(read_p(sve, 8U + ((instruction >> 10U) & 0x7U)));
    const uint64_t base     = sve->cpu->x[(instruction >> 5U) & 0x1FU];
    const uint64_t offset   = true == is_immediate
                                  ? (uint64_t)(sign_extend((instruction >> 16U) & 0xFU, 4U)
                                             * (int64_t)(nreg * elements))
                                  : read_x(sve, (instruction >> 16U) & 0x1FU);

    for (uint32_t r = 0U; r < nreg; ++r)
    {
        for (uint32_t e = 0U; e < elements; ++e)
        {
            const uint64_t address = base + (offset + r * elements + e) * mbytes;

            if (true == is_active((uint16_t)(mask >> (16U * r)), e, esize)
                && false == is_mapped(sve, address, mbytes))
            {
                return SVE_FAULT;
            }
        }
    }

    vector_t values[4];

    for (uint32_t r = 0U; r < nreg; ++r)
    {
        const uint16_t predicate = (uint16_t)(mask >> (16U * r));
        const uint32_t z         = (t + r * stride) & 0x1FU;
        values[r]                = true == is_store ? read_z(sve, z) : (vector_t) { { 0U, 0U } };

        for (uint32_t e = 0U; e < elements; ++e)
        {
            const uint64_t address = base + (offset + r * elements + e) * mbytes;

            if (false == is_active(predicate, e, esize))
            {
                continue;
            }

            if (true == is_store)
            {
                memory_write(sve, address, mbytes, get_element(&values[r], e, esize));
            }
            else
            {
                set_element(&values[r], e, esize, memory_read(sve, address, mbytes));
            }
        }
    }

    if (false == is_store)
    {
        for (uint32_t r = 0U; r < nreg; ++r)
        {
            write_z(sve, (t + r * stride) & 0x1FU, values[r]);
        }
    }

    return SVE_OK;
}

// -----------------------------------------------------------------------------
// SME2 multi-vector operations
// -----------------------------------------------------------------------------

/// Whether streaming mode is on, which the multi-vector operations need.
static bool
is_streaming(const sve_t *sve)
{
    return 0U != (sve->cpu->svcr & SVCR_SM);
}

/// A BFloat16 value as the single precision value it equals.
static uint64_t
bfloat_to_single(const uint64_t value)
{
    return (value & 0xFFFFU) << 16U;
}

/// One floating-point maximum or minimum (`kind` 0 FMAX, 1 FMAXNM, 2 FMIN, 3 FMINNM) of
/// `esize` bits, BFloat16 when `esize` is 0.
static uint64_t
float_extreme(
    sve_t *sve, const uint32_t esize, const uint32_t kind, const uint64_t a, const uint64_t b)
{
    static const bal_float_operation_t OPERATIONS[4] = {
        BAL_FLOAT_MAXIMUM,
        BAL_FLOAT_MAXIMUM_NUMBER,
        BAL_FLOAT_MINIMUM,
        BAL_FLOAT_MINIMUM_NUMBER,
    };

    if (0U == esize)
    {
        // The result is one of the operands, or a NaN with the low half of a single clear.
        return fp(sve, 32U, OPERATIONS[kind], bfloat_to_single(a), bfloat_to_single(b), 0U) >> 16U;
    }

    return fp(sve, esize, OPERATIONS[kind], a, b, 0U);
}

/// The SME2 multi-vector operations on two or four consecutive vectors, destructive, with one
/// vector (11000001 size 10 Zm 1010 N Q opc Zdn ...) or as many vectors (11000001 size 1 Zm
/// 1011 N Q ...) for the second operands: SMAX, UMAX, SMIN, UMIN (opc 0000 op, U in bit 0),
/// FMAX, FMAXNM, FMIN, FMINNM and the BFloat16 forms (0100 op, o2 in bit 0), SRSHL and URSHL
/// (10001), ADD (11000, one vector only) and SQDMULH (Q).
static sve_result_t
multi_vector_binary(sve_t *sve, const uint32_t instruction)
{
    const bool     is_multi   = 0U != (instruction & 0x1000U);
    const uint32_t nreg       = 0U != (instruction & 0x800U) ? 4U : 2U;
    const bool     is_sqdmulh = 0U != (instruction & 0x400U);
    const uint32_t size       = (instruction >> 22U) & 0x3U;
    const uint32_t opc        = (instruction >> 5U) & 0x1FU;
    const uint32_t low        = instruction & 0x1U;
    const uint32_t dn         = instruction & (2U == nreg ? 0x1EU : 0x1CU);
    uint32_t       m;

    if (true == is_multi)
    {
        // Zm:0 in bits 20:17 with bit 16 clear, or Zm:00 in bits 20:18 with bits 17:16 clear.
        if (0U != (instruction & (2U == nreg ? 0x10000U : 0x30000U)))
        {
            return SVE_UNDEFINED;
        }

        m = (instruction >> 16U) & (2U == nreg ? 0x1EU : 0x1CU);
    }
    else
    {
        m = (instruction >> 16U) & 0xFU;
    }

    uint32_t kind;

    if (true == is_sqdmulh)
    {
        kind = 0U == opc && 0U == low ? 4U : 99U;
    }
    else if (0x0U == (opc >> 1U))
    {
        kind = 0U; // integer maximum and minimum
    }
    else if (0x4U == (opc >> 1U))
    {
        kind = 1U; // floating-point maximum and minimum
    }
    else if (0x11U == opc)
    {
        kind = 2U; // rounding shifts
    }
    else if (0x18U == opc && false == is_multi && 0U == low)
    {
        kind = 3U; // ADD
    }
    else
    {
        kind = 99U;
    }

    if (99U == kind || (4U == nreg && 0U != (instruction & 0x2U)))
    {
        return SVE_UNDEFINED;
    }

    if (false == is_streaming(sve))
    {
        return SVE_ILLEGAL;
    }

    // BFloat16 (size 00 of the floating-point forms) has 16-bit elements.
    const uint32_t esize     = 1U == kind && 0U == size ? 16U : 8U << size;
    const uint32_t elements  = elements_of(esize);
    const bool     is_signed = 0U == low;
    vector_t       results[4];

    for (uint32_t r = 0U; r < nreg; ++r)
    {
        const vector_t a = read_z(sve, dn + r);
        const vector_t b = read_z(sve, true == is_multi ? m + r : m);

        for (uint32_t e = 0U; e < elements; ++e)
        {
            const uint64_t x = get_element(&a, e, esize);
            const uint64_t y = get_element(&b, e, esize);
            uint64_t       value;

            switch (kind)
            {
                case 0U: {
                    const bool is_minimum = 0U != (opc & 1U);
                    const bool greater    = wide(x, esize, is_signed) > wide(y, esize, is_signed);
                    value                 = greater != is_minimum ? x : y;
                    break;
                }
                case 1U: {
                    const uint32_t extreme = (low << 1U) | (opc & 1U);
                    value = float_extreme(sve, 0U == size ? 0U : esize, extreme, x, y);
                    break;
                }
                case 2U:
                    value = narrow_to(
                        shift_wide(wide(x, esize, is_signed), sign_extend(y, esize), esize, true),
                        esize,
                        SATURATE_NONE);
                    break;
                case 3U:
                    value = x + y;
                    break;
                default:
                    value = doubling_multiply_high(x, y, esize, false);
                    break;
            }

            set_element(&results[r], e, esize, value);
        }
    }

    for (uint32_t r = 0U; r < nreg; ++r)
    {
        write_z(sve, dn + r, results[r]);
    }

    return SVE_OK;
}

/// SCLAMP, UCLAMP (11000001 size 1 Zm 1100 N 1 Zn Zd U) and FCLAMP and BFCLAMP (... 1100 N 0
/// ...): each of two or four consecutive vectors clamped between Zn and Zm.
static sve_result_t
multi_vector_clamp(sve_t *sve, const uint32_t instruction)
{
    const uint32_t nreg       = 0U != (instruction & 0x800U) ? 4U : 2U;
    const bool     is_integer = 0U != (instruction & 0x400U);
    const uint32_t size       = (instruction >> 22U) & 0x3U;
    const uint32_t d          = instruction & (2U == nreg ? 0x1EU : 0x1CU);
    const bool     is_signed  = 0U == (instruction & 0x1U);

    if ((false == is_integer && 0U != (instruction & 0x1U))
        || (4U == nreg && 0U != (instruction & 0x2U)))
    {
        return SVE_UNDEFINED;
    }

    if (false == is_streaming(sve))
    {
        return SVE_ILLEGAL;
    }

    const uint32_t esize = false == is_integer && 0U == size ? 16U : 8U << size;
    const vector_t low   = read_z(sve, (instruction >> 5U) & 0x1FU);
    const vector_t high  = read_z(sve, (instruction >> 16U) & 0x1FU);

    for (uint32_t r = 0U; r < nreg; ++r)
    {
        vector_t result = read_z(sve, d + r);

        for (uint32_t e = 0U; e < elements_of(esize); ++e)
        {
            const uint64_t x = get_element(&result, e, esize);
            const uint64_t a = get_element(&low, e, esize);
            const uint64_t b = get_element(&high, e, esize);
            uint64_t       value;

            if (true == is_integer)
            {
                value = wide(a, esize, is_signed) > wide(x, esize, is_signed) ? a : x;
                value = wide(value, esize, is_signed) > wide(b, esize, is_signed) ? b : value;
            }
            else
            {
                const uint32_t fsize = 0U == size ? 0U : esize;
                value = float_extreme(sve, fsize, 3U, float_extreme(sve, fsize, 1U, a, x), b);
            }

            set_element(&result, e, esize, value);
        }

        write_z(sve, d + r, result);
    }

    return SVE_OK;
}

/// The single precision conversions and roundings of two or four consecutive vectors: FRINTN,
/// FRINTP, FRINTM and FRINTA (11000001 10 1 N1 opc 111000 ...), FCVTZS and FCVTZU (11000001 00 1 N
/// 0001 111000 ... U ...) and SCVTF and UCVTF (... 0010 ...).
static sve_result_t
multi_vector_convert(sve_t *sve, const uint32_t instruction)
{
    const uint32_t nreg     = 0U != (instruction & 0x100000U) ? 4U : 2U;
    const bool     is_round = 0U != (instruction & 0x800000U);
    // FRINT has bits 21:19 = 1N1, the conversions 1N0.
    const bool     is_frint_form = 0U != (instruction & 0x80000U);
    const uint32_t opc           = (instruction >> 16U) & 0x7U;
    const uint32_t n             = (instruction >> 5U) & (2U == nreg ? 0x1EU : 0x1CU);
    const uint32_t d             = instruction & (2U == nreg ? 0x1EU : 0x1CU);
    const bool     is_unsigned   = 0U != (instruction & 0x20U);

    // FRINTN (000), FRINTP (001), FRINTM (010) and FRINTA (100); FCVTZ (001) and SCVTF (010).
    const bool is_allocated = true == is_round
                                  ? (opc <= 2U || 4U == opc) && 0U == (instruction & 0x20U)
                                  : 1U == opc || 2U == opc;

    if (false == is_allocated || is_round != is_frint_form
        || 0U != (instruction & (2U == nreg ? 0x1U : 0x43U)))
    {
        return SVE_UNDEFINED;
    }

    if (false == is_streaming(sve))
    {
        return SVE_ILLEGAL;
    }

    static const uint32_t RMODES[5] = { 0U, 1U, 2U, 0U, 4U };
    const uint64_t        fpcr      = sve->cpu->fpcr;

    for (uint32_t r = 0U; r < nreg; ++r)
    {
        vector_t value = read_z(sve, n + r);

        for (uint32_t e = 0U; e < 4U; ++e)
        {
            const uint64_t x = get_element(&value, e, 32U);
            uint64_t       result;

            if (true == is_round)
            {
                result = bal_float_round_element(fpcr, &sve->flags, 32U, x, RMODES[opc], false);
            }
            else if (1U == opc)
            {
                result = bal_float_to_integer_element(
                    fpcr, &sve->flags, 32U, x, 0U, 3U, is_unsigned, 32U);
            }
            else
            {
                result = bal_float_from_integer_element(
                    fpcr, &sve->flags, 32U, x, 0U, is_unsigned, 32U);
            }

            set_element(&value, e, 32U, result);
        }

        write_z(sve, d + r, value);
    }

    return SVE_OK;
}

/// FCVT, FCVTN, BFCVT and BFCVTN from two single precision vectors to one (11000001 0 op
/// 100000111000 Zn N Zd), in halves or interleaved (N), and FCVT and FCVTL from one half
/// precision vector to two (11000001 10100000111000 Zn Zd L).
static sve_result_t
multi_vector_float_convert(sve_t *sve, const uint32_t instruction)
{
    const bool is_widening = 0U != (instruction & 0x800000U);

    if (true == is_widening && 0U != (instruction & 0x400000U))
    {
        return SVE_UNDEFINED;
    }

    if (false == is_streaming(sve))
    {
        return SVE_ILLEGAL;
    }

    const uint64_t fpcr = sve->cpu->fpcr;

    if (true == is_widening)
    {
        const bool     is_interleaved = 0U != (instruction & 0x1U);
        const uint32_t d              = instruction & 0x1EU;
        const vector_t source         = read_z(sve, (instruction >> 5U) & 0x1FU);
        vector_t       results[2];

        for (uint32_t e = 0U; e < 8U; ++e)
        {
            const uint64_t value = bal_float_convert_element(
                fpcr, &sve->flags, 16U, 32U, get_element(&source, e, 16U));

            if (true == is_interleaved)
            {
                set_element(&results[e & 1U], e / 2U, 32U, value);
            }
            else
            {
                set_element(&results[e / 4U], e % 4U, 32U, value);
            }
        }

        write_z(sve, d, results[0]);
        write_z(sve, d + 1U, results[1]);
        return SVE_OK;
    }

    const bool     is_bfloat      = 0U != (instruction & 0x400000U);
    const bool     is_interleaved = 0U != (instruction & 0x20U);
    const uint32_t n              = (instruction >> 5U) & 0x1EU;
    vector_t       result         = { { 0U, 0U } };

    for (uint32_t r = 0U; r < 2U; ++r)
    {
        const vector_t source = read_z(sve, n + r);

        for (uint32_t e = 0U; e < 4U; ++e)
        {
            const uint64_t x     = get_element(&source, e, 32U);
            const uint64_t value = true == is_bfloat
                                       ? bal_float_to_bfloat16_element(fpcr, &sve->flags, x)
                                       : bal_float_convert_element(fpcr, &sve->flags, 32U, 16U, x);
            set_element(&result, true == is_interleaved ? 2U * e + r : 4U * r + e, 16U, value);
        }
    }

    write_z(sve, instruction & 0x1FU, result);
    return SVE_OK;
}

/// SUNPK and UUNPK (11000001 size 1 N 0101111000 Zn Zd U): the elements of one or two vectors
/// extended to twice their size into two or four.
static sve_result_t
multi_vector_unpack(sve_t *sve, const uint32_t instruction)
{
    const uint32_t nreg = 0U != (instruction & 0x100000U) ? 4U : 2U;
    const uint32_t size = (instruction >> 22U) & 0x3U;

    if (0U == size || (4U == nreg && 0U != (instruction & 0x22U)))
    {
        return SVE_UNDEFINED;
    }

    if (false == is_streaming(sve))
    {
        return SVE_ILLEGAL;
    }

    const uint32_t esize     = 8U << size;
    const uint32_t half      = esize / 2U;
    const uint32_t elements  = elements_of(esize);
    const bool     is_signed = 0U == (instruction & 0x1U);
    const uint32_t n         = (instruction >> 5U) & (2U == nreg ? 0x1FU : 0x1EU);
    const uint32_t d         = instruction & (2U == nreg ? 0x1EU : 0x1CU);
    vector_t       results[4];

    for (uint32_t r = 0U; r < nreg / 2U; ++r)
    {
        const vector_t source = read_z(sve, n + r);

        for (uint32_t i = 0U; i < 2U; ++i)
        {
            for (uint32_t e = 0U; e < elements; ++e)
            {
                set_element(
                    &results[2U * r + i],
                    e,
                    esize,
                    narrow_to(wide(get_element(&source, i * elements + e, half), half, is_signed),
                              esize,
                              SATURATE_NONE));
            }
        }
    }

    for (uint32_t r = 0U; r < nreg; ++r)
    {
        write_z(sve, d + r, results[r]);
    }

    return SVE_OK;
}

/// ZIP and UZP of two vectors into two (11000001 size 1 Zm 11010 Q Zn Zd op) and of four into
/// four (11000001 size 1 10110 Q 111000 Zn 00 Zd op 0), with quadwords for Q.
static sve_result_t
multi_vector_permute(sve_t *sve, const uint32_t instruction)
{
    // Four registers have bits 15:13 = 111, two 110; Q is bit 16 or bit 10.
    const bool is_four = 0U != (instruction & 0x2000U);
    const bool is_quad
        = true == is_four ? 0U != (instruction & 0x10000U) : 0U != (instruction & 0x400U);
    const uint32_t esize  = true == is_quad ? 128U : 8U << ((instruction >> 22U) & 0x3U);
    const uint32_t nreg   = true == is_four ? 4U : 2U;
    const bool     is_uzp = 0U != (instruction & (true == is_four ? 0x2U : 0x1U));

    if ((true == is_quad && 0U != (instruction & 0xC00000U))
        || (true == is_four && 0U != (instruction & 0x61U)))
    {
        return SVE_UNDEFINED;
    }

    // The elements of one register must split into the groups.
    if (VECTOR_BITS < esize * nreg)
    {
        return SVE_UNDEFINED;
    }

    if (false == is_streaming(sve))
    {
        return SVE_ILLEGAL;
    }

    const uint32_t groups = VECTOR_BITS / (esize * nreg);
    const uint32_t d      = instruction & (true == is_four ? 0x1CU : 0x1EU);
    vector_t       sources[4];
    vector_t       results[4];

    for (uint32_t r = 0U; r < nreg; ++r)
    {
        const uint32_t source = true == is_four ? ((instruction >> 5U) & 0x1CU) + r
                                                : (0U == r ? (instruction >> 5U) & 0x1FU
                                                           : (instruction >> 16U) & 0x1FU);
        sources[r]            = read_z(sve, source);
    }

    for (uint32_t r = 0U; r < nreg; ++r)
    {
        for (uint32_t q = 0U; q < groups; ++q)
        {
            for (uint32_t i = 0U; i < nreg; ++i)
            {
                // ZIP puts element base + q of each source in turn; UZP deals each source's
                // groups out to the results.
                const uint32_t to   = true == is_uzp ? r * groups + q : nreg * q + i;
                const uint32_t from = true == is_uzp ? nreg * q + i : r * groups + q;
                const uint32_t into = true == is_uzp ? i : r;
                const uint32_t of   = true == is_uzp ? r : i;

                if (128U == esize)
                {
                    results[into] = sources[of];
                }
                else
                {
                    set_element(&results[into], to, esize, get_element(&sources[of], from, esize));
                }
            }
        }
    }

    for (uint32_t r = 0U; r < nreg; ++r)
    {
        write_z(sve, d + r, results[r]);
    }

    return SVE_OK;
}

/// SEL of two or four consecutive vectors (11000001 size 1 Zm 0100 PNg Zn 0 Zd 0, or ... 01100
/// ...), from Zn where the predicate-as-counter register is active and Zm elsewhere.
static sve_result_t
multi_vector_select(sve_t *sve, const uint32_t instruction)
{
    const uint32_t nreg = 0U != (instruction & 0x10000U) ? 4U : 2U;

    if (0U != (instruction & (2U == nreg ? 0x21U : 0x20063U)))
    {
        return SVE_UNDEFINED;
    }

    if (false == is_streaming(sve))
    {
        return SVE_ILLEGAL;
    }

    const uint32_t esize     = 8U << ((instruction >> 22U) & 0x3U);
    const uint32_t mask_bits = 2U == nreg ? 0x1EU : 0x1CU;
    const uint32_t n         = (instruction >> 5U) & mask_bits;
    const uint32_t m         = (instruction >> 16U) & mask_bits;
    const uint32_t d         = instruction & mask_bits;
    const uint64_t mask = counter_to_predicate(read_p(sve, 8U + ((instruction >> 10U) & 0x7U)));
    vector_t       results[4];

    for (uint32_t r = 0U; r < nreg; ++r)
    {
        const vector_t a = read_z(sve, n + r);
        const vector_t b = read_z(sve, m + r);

        for (uint32_t e = 0U; e < elements_of(esize); ++e)
        {
            const bool active = is_active((uint16_t)(mask >> (16U * r)), e, esize);
            set_element(&results[r], e, esize, get_element(true == active ? &a : &b, e, esize));
        }
    }

    for (uint32_t r = 0U; r < nreg; ++r)
    {
        write_z(sve, d + r, results[r]);
    }

    return SVE_OK;
}

/// The saturating narrowing conversions and shifts of two or four vectors into one: SQCVT,
/// UQCVT and SQCVTU of two (11000001 0 op 100011111000 Zn U Zd) and four (11000001 sz op
/// 110011111000 Zn N U Zd), and SQRSHR, UQRSHR and SQRSHRU of two (11000001111 op imm4 110101 Zn
/// U Zd) and four (11000001 tsize 1 imm5 11011 N Zn op U Zd), in quarters or interleaved (N).
static sve_result_t
multi_vector_narrow(sve_t *sve, const uint32_t instruction)
{
    const bool is_shift_four = 0xD800U == (instruction & 0xF800U);
    const bool is_shift_two  = 0xD400U == (instruction & 0xFC00U);
    const bool is_shift      = true == is_shift_four || true == is_shift_two;
    // Two registers: the shifts above, and the conversions with bit 20 clear.
    const bool is_two      = true == is_shift ? is_shift_two : 0U == (instruction & 0x100000U);
    const bool is_unsigned = 0U != (instruction & 0x20U);
    bool       op;
    uint32_t   esize;
    uint32_t   shift          = 0U;
    bool       is_interleaved = false;

    if (true == is_shift)
    {
        op = 0U != (instruction & (true == is_two ? 0x100000U : 0x40U));

        if (true == is_two)
        {
            esize = 16U;
            shift = 16U - ((instruction >> 16U) & 0xFU);
        }
        else
        {
            const uint32_t tsize = (instruction >> 22U) & 0x3U;

            if (0U == tsize)
            {
                return SVE_UNDEFINED;
            }

            esize          = 1U == tsize ? 8U : 16U;
            shift          = 8U * esize - ((tsize << 5U) | ((instruction >> 16U) & 0x1FU));
            is_interleaved = 0U != (instruction & 0x400U);
        }
    }
    else
    {
        op = 0U != (instruction & 0x400000U);

        if (true == is_two)
        {
            esize = 16U;
        }
        else
        {
            esize          = 8U << ((instruction >> 23U) & 1U);
            is_interleaved = 0U != (instruction & 0x40U);
        }
    }

    // The unsigned conversions of a signed source have op set and U clear.
    if (true == op && true == is_unsigned)
    {
        return SVE_UNDEFINED;
    }

    if (false == is_streaming(sve))
    {
        return SVE_ILLEGAL;
    }

    const uint32_t   nreg      = true == is_two ? 2U : 4U;
    const uint32_t   wide_size = esize * nreg;
    const uint32_t   elements  = VECTOR_BITS / wide_size;
    const uint32_t   n         = (instruction >> 5U) & (2U == nreg ? 0x1EU : 0x1CU);
    const saturate_t mode = true == is_unsigned || true == op ? SATURATE_UNSIGNED : SATURATE_SIGNED;
    vector_t         result = { { 0U, 0U } };

    for (uint32_t r = 0U; r < nreg; ++r)
    {
        const vector_t source = read_z(sve, n + r);

        for (uint32_t e = 0U; e < elements; ++e)
        {
            __int128 value
                = wide(get_element(&source, e, wide_size), wide_size, false == is_unsigned);

            if (0U != shift)
            {
                value = (value + ((__int128)1 << (shift - 1U))) >> shift;
            }

            set_element(&result,
                        true == is_interleaved ? nreg * e + r : r * elements + e,
                        esize,
                        narrow_to(value, esize, mode));
        }
    }

    write_z(sve, instruction & 0x1FU, result);
    return SVE_OK;
}

// -----------------------------------------------------------------------------
// Decoding
// -----------------------------------------------------------------------------

static const sve_pattern_t PATTERNS[] = {
    { 0xFF000010U, 0xE0000000U, tile_slice_memory },
    { 0xFFC00010U, 0xE1C00000U, tile_slice_memory },
    { 0xFFDF9C10U, 0xE1000000U, za_vector_memory },
    { 0xFFDFFC1FU, 0xE11F8000U, za_vector_memory },
    { 0xFFFFFF00U, 0xC0080000U, zero_tiles },
    { 0xFFFFFFFFU, 0xC0480001U, zero_tiles },
    { 0xFF3E0010U, 0xC0000000U, move_tile_vector },
    { 0xFF3E0200U, 0xC0020000U, move_tile_vector },
    { 0xFFBE0018U, 0xC0900000U, add_to_tile },
    { 0xFFE0000CU, 0x80800000U, float_outer_product },
    { 0xFFE00008U, 0x80C00000U, float_outer_product },
    { 0xFEC0000CU, 0xA0800000U, integer_outer_product },
    { 0xFEC00008U, 0xA0C00000U, integer_outer_product },
    { 0xFFC0000CU, 0x81800000U, widening_float_outer_product },
    { 0xFFE0000EU, 0x81800008U, half_outer_product },
    { 0xFFE0000CU, 0x80800008U, bit_outer_product },
    { 0xFEE0000CU, 0xA0800008U, integer_outer_product_pairs },
    { 0xFF3F1000U, 0xC0060000U, move_multiple },
    { 0xFF3F1000U, 0xC0040000U, move_multiple },
    { 0xFF3E1E00U, 0xC0020200U, move_zeroing },
    { 0xFFFD8FE0U, 0xC04C03E0U, move_table },
    { 0xFFFC1FF8U, 0xC00C0000U, zero_vectors },
    { 0xFFF80000U, 0xC0C80000U, table_lookup_zt0 },
    { 0xFFE80000U, 0xC0880000U, table_lookup_zt0 },
    { 0xFE800000U, 0xA0000000U, multi_vector_memory },
    { 0xFFE0FC00U, 0xC1E0D400U, multi_vector_narrow },
    { 0xFF20F800U, 0xC120D800U, multi_vector_narrow },
    { 0xFFBFFC00U, 0xC123E000U, multi_vector_narrow },
    { 0xFF3FFC00U, 0xC133E000U, multi_vector_narrow },
    { 0xFFBFFC00U, 0xC120E000U, multi_vector_float_convert },
    { 0xFFFFFC00U, 0xC1A0E000U, multi_vector_float_convert },
    { 0xFF2FFC00U, 0xC125E000U, multi_vector_unpack },
    { 0xFF3EFC00U, 0xC136E000U, multi_vector_permute },
    { 0xFFE8FC00U, 0xC1A8E000U, multi_vector_convert },
    { 0xFFE8FC00U, 0xC120E000U, multi_vector_convert },
    { 0xFF20F800U, 0xC120D000U, multi_vector_permute },
    { 0xFF20F000U, 0xC120C000U, multi_vector_clamp },
    { 0xFF30F000U, 0xC120A000U, multi_vector_binary },
    { 0xFF20F000U, 0xC120B000U, multi_vector_binary },
    { 0xFF20E000U, 0xC1208000U, multi_vector_select },
};

sve_result_t
sme_execute(sve_t *sve, const uint32_t instruction)
{
    for (size_t i = 0U; i < sizeof(PATTERNS) / sizeof(PATTERNS[0]); ++i)
    {
        if (NULL != PATTERNS[i].handler && PATTERNS[i].value == (instruction & PATTERNS[i].mask))
        {
            return PATTERNS[i].handler(sve, instruction);
        }
    }

    return SVE_UNDEFINED;
}

/*** end of file ***/
