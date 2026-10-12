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
