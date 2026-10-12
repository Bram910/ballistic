#include "backend/bal_pointer_authentication.h"

#include "bal_attributes.h"
#include <string.h>

/// SCTLR_EL1 bits that enable the instruction and data keys.
#define SCTLR_ENIA (1ULL << 31U)
#define SCTLR_ENIB (1ULL << 30U)
#define SCTLR_ENDA (1ULL << 27U)
#define SCTLR_ENDB (1ULL << 13U)

/// TCR_EL1 top-byte-ignore bits, for addresses with bit 55 clear (0) and set (1).
#define TCR_TBI0  (1ULL << 37U)
#define TCR_TBI1  (1ULL << 38U)
#define TCR_TBID0 (1ULL << 51U)
#define TCR_TBID1 (1ULL << 52U)

/// The range of TCR_EL1.TxSZ for the 4KB granule without FEAT_LVA and FEAT_TTST.
#define MINIMUM_TSZ 16U
#define MAXIMUM_TSZ 39U

// QARMA5 as ComputePAC() specifies it. The 64-bit state is sixteen 4-bit cells, cell 0 being
// bits 63:60 and cell 15 bits 3:0; the four rows of the 4x4 cell matrix are its 16-bit quarters,
// row 0 the top one. Every step works on all cells at once.

/// The S-box applied to both cells of a byte, and its inverse.
static const uint8_t SUBSTITUTION[256] = {
    0xBBU, 0xB6U, 0xB8U, 0xBFU, 0xBCU, 0xB0U, 0xB9U, 0xBEU, 0xB3U, 0xB7U, 0xB4U, 0xB5U, 0xBDU,
    0xB2U, 0xB1U, 0xBAU, 0x6BU, 0x66U, 0x68U, 0x6FU, 0x6CU, 0x60U, 0x69U, 0x6EU, 0x63U, 0x67U,
    0x64U, 0x65U, 0x6DU, 0x62U, 0x61U, 0x6AU, 0x8BU, 0x86U, 0x88U, 0x8FU, 0x8CU, 0x80U, 0x89U,
    0x8EU, 0x83U, 0x87U, 0x84U, 0x85U, 0x8DU, 0x82U, 0x81U, 0x8AU, 0xFBU, 0xF6U, 0xF8U, 0xFFU,
    0xFCU, 0xF0U, 0xF9U, 0xFEU, 0xF3U, 0xF7U, 0xF4U, 0xF5U, 0xFDU, 0xF2U, 0xF1U, 0xFAU, 0xCBU,
    0xC6U, 0xC8U, 0xCFU, 0xCCU, 0xC0U, 0xC9U, 0xCEU, 0xC3U, 0xC7U, 0xC4U, 0xC5U, 0xCDU, 0xC2U,
    0xC1U, 0xCAU, 0x0BU, 0x06U, 0x08U, 0x0FU, 0x0CU, 0x00U, 0x09U, 0x0EU, 0x03U, 0x07U, 0x04U,
    0x05U, 0x0DU, 0x02U, 0x01U, 0x0AU, 0x9BU, 0x96U, 0x98U, 0x9FU, 0x9CU, 0x90U, 0x99U, 0x9EU,
    0x93U, 0x97U, 0x94U, 0x95U, 0x9DU, 0x92U, 0x91U, 0x9AU, 0xEBU, 0xE6U, 0xE8U, 0xEFU, 0xECU,
    0xE0U, 0xE9U, 0xEEU, 0xE3U, 0xE7U, 0xE4U, 0xE5U, 0xEDU, 0xE2U, 0xE1U, 0xEAU, 0x3BU, 0x36U,
    0x38U, 0x3FU, 0x3CU, 0x30U, 0x39U, 0x3EU, 0x33U, 0x37U, 0x34U, 0x35U, 0x3DU, 0x32U, 0x31U,
    0x3AU, 0x7BU, 0x76U, 0x78U, 0x7FU, 0x7CU, 0x70U, 0x79U, 0x7EU, 0x73U, 0x77U, 0x74U, 0x75U,
    0x7DU, 0x72U, 0x71U, 0x7AU, 0x4BU, 0x46U, 0x48U, 0x4FU, 0x4CU, 0x40U, 0x49U, 0x4EU, 0x43U,
    0x47U, 0x44U, 0x45U, 0x4DU, 0x42U, 0x41U, 0x4AU, 0x5BU, 0x56U, 0x58U, 0x5FU, 0x5CU, 0x50U,
    0x59U, 0x5EU, 0x53U, 0x57U, 0x54U, 0x55U, 0x5DU, 0x52U, 0x51U, 0x5AU, 0xDBU, 0xD6U, 0xD8U,
    0xDFU, 0xDCU, 0xD0U, 0xD9U, 0xDEU, 0xD3U, 0xD7U, 0xD4U, 0xD5U, 0xDDU, 0xD2U, 0xD1U, 0xDAU,
    0x2BU, 0x26U, 0x28U, 0x2FU, 0x2CU, 0x20U, 0x29U, 0x2EU, 0x23U, 0x27U, 0x24U, 0x25U, 0x2DU,
    0x22U, 0x21U, 0x2AU, 0x1BU, 0x16U, 0x18U, 0x1FU, 0x1CU, 0x10U, 0x19U, 0x1EU, 0x13U, 0x17U,
    0x14U, 0x15U, 0x1DU, 0x12U, 0x11U, 0x1AU, 0xABU, 0xA6U, 0xA8U, 0xAFU, 0xACU, 0xA0U, 0xA9U,
    0xAEU, 0xA3U, 0xA7U, 0xA4U, 0xA5U, 0xADU, 0xA2U, 0xA1U, 0xAAU,
};

static const uint8_t INVERSE_SUBSTITUTION[256] = {
    0x55U, 0x5EU, 0x5DU, 0x58U, 0x5AU, 0x5BU, 0x51U, 0x59U, 0x52U, 0x56U, 0x5FU, 0x50U, 0x54U,
    0x5CU, 0x57U, 0x53U, 0xE5U, 0xEEU, 0xEDU, 0xE8U, 0xEAU, 0xEBU, 0xE1U, 0xE9U, 0xE2U, 0xE6U,
    0xEFU, 0xE0U, 0xE4U, 0xECU, 0xE7U, 0xE3U, 0xD5U, 0xDEU, 0xDDU, 0xD8U, 0xDAU, 0xDBU, 0xD1U,
    0xD9U, 0xD2U, 0xD6U, 0xDFU, 0xD0U, 0xD4U, 0xDCU, 0xD7U, 0xD3U, 0x85U, 0x8EU, 0x8DU, 0x88U,
    0x8AU, 0x8BU, 0x81U, 0x89U, 0x82U, 0x86U, 0x8FU, 0x80U, 0x84U, 0x8CU, 0x87U, 0x83U, 0xA5U,
    0xAEU, 0xADU, 0xA8U, 0xAAU, 0xABU, 0xA1U, 0xA9U, 0xA2U, 0xA6U, 0xAFU, 0xA0U, 0xA4U, 0xACU,
    0xA7U, 0xA3U, 0xB5U, 0xBEU, 0xBDU, 0xB8U, 0xBAU, 0xBBU, 0xB1U, 0xB9U, 0xB2U, 0xB6U, 0xBFU,
    0xB0U, 0xB4U, 0xBCU, 0xB7U, 0xB3U, 0x15U, 0x1EU, 0x1DU, 0x18U, 0x1AU, 0x1BU, 0x11U, 0x19U,
    0x12U, 0x16U, 0x1FU, 0x10U, 0x14U, 0x1CU, 0x17U, 0x13U, 0x95U, 0x9EU, 0x9DU, 0x98U, 0x9AU,
    0x9BU, 0x91U, 0x99U, 0x92U, 0x96U, 0x9FU, 0x90U, 0x94U, 0x9CU, 0x97U, 0x93U, 0x25U, 0x2EU,
    0x2DU, 0x28U, 0x2AU, 0x2BU, 0x21U, 0x29U, 0x22U, 0x26U, 0x2FU, 0x20U, 0x24U, 0x2CU, 0x27U,
    0x23U, 0x65U, 0x6EU, 0x6DU, 0x68U, 0x6AU, 0x6BU, 0x61U, 0x69U, 0x62U, 0x66U, 0x6FU, 0x60U,
    0x64U, 0x6CU, 0x67U, 0x63U, 0xF5U, 0xFEU, 0xFDU, 0xF8U, 0xFAU, 0xFBU, 0xF1U, 0xF9U, 0xF2U,
    0xF6U, 0xFFU, 0xF0U, 0xF4U, 0xFCU, 0xF7U, 0xF3U, 0x05U, 0x0EU, 0x0DU, 0x08U, 0x0AU, 0x0BU,
    0x01U, 0x09U, 0x02U, 0x06U, 0x0FU, 0x00U, 0x04U, 0x0CU, 0x07U, 0x03U, 0x45U, 0x4EU, 0x4DU,
    0x48U, 0x4AU, 0x4BU, 0x41U, 0x49U, 0x42U, 0x46U, 0x4FU, 0x40U, 0x44U, 0x4CU, 0x47U, 0x43U,
    0xC5U, 0xCEU, 0xCDU, 0xC8U, 0xCAU, 0xCBU, 0xC1U, 0xC9U, 0xC2U, 0xC6U, 0xCFU, 0xC0U, 0xC4U,
    0xCCU, 0xC7U, 0xC3U, 0x75U, 0x7EU, 0x7DU, 0x78U, 0x7AU, 0x7BU, 0x71U, 0x79U, 0x72U, 0x76U,
    0x7FU, 0x70U, 0x74U, 0x7CU, 0x77U, 0x73U, 0x35U, 0x3EU, 0x3DU, 0x38U, 0x3AU, 0x3BU, 0x31U,
    0x39U, 0x32U, 0x36U, 0x3FU, 0x30U, 0x34U, 0x3CU, 0x37U, 0x33U,
};
/// The cell permutation: output cell i is input cell CELL_SHUFFLE[i], and the inverse.
static const uint8_t CELL_SHUFFLE[16]
    = { 0U, 11U, 6U, 13U, 10U, 1U, 12U, 7U, 5U, 14U, 3U, 8U, 15U, 4U, 9U, 2U };
static const uint8_t INVERSE_CELL_SHUFFLE[16]
    = { 0U, 5U, 15U, 10U, 13U, 8U, 2U, 7U, 11U, 14U, 4U, 1U, 6U, 3U, 9U, 12U };

/// The tweak permutation and its inverse, and the cells the tweak's LFSR steps after it.
static const uint8_t TWEAK_SHUFFLE[16]
    = { 6U, 5U, 14U, 15U, 0U, 1U, 2U, 3U, 7U, 12U, 13U, 4U, 8U, 9U, 10U, 11U };
static const uint8_t INVERSE_TWEAK_SHUFFLE[16]
    = { 4U, 5U, 6U, 7U, 11U, 1U, 0U, 8U, 12U, 13U, 14U, 15U, 9U, 10U, 2U, 3U };

/// Cells 0, 1, 3, 4, 8, 11 and 13.
#define TWEAK_ROTATED_CELLS 0xFF0FF000F00F0F00ULL

/// Every cell's lowest bit, and its three highest.
#define CELL_LOW_BITS  0x1111111111111111ULL
#define CELL_HIGH_BITS 0xEEEEEEEEEEEEEEEEULL

static const uint64_t ROUND_CONSTANTS[5] = {
    0x0000000000000000ULL, 0x13198A2E03707344ULL, 0xA4093822299F31D0ULL,
    0x082EFA98EC4E6C89ULL, 0x452821E638D01377ULL,
};
static const uint64_t ALPHA = 0xC0AC29B7C97C50DDULL;

static uint64_t
substitute(const uint64_t state, const uint8_t table[256])
{
    uint64_t result = 0U;

    for (uint32_t byte = 0U; byte < 64U; byte += 8U)
    {
        result |= (uint64_t)table[(state >> byte) & 0xFFU] << byte;
    }

    return result;
}

static uint64_t
permute_cells(const uint64_t state, const uint8_t permutation[16])
{
    uint64_t result = 0U;

    for (uint32_t cell = 0U; cell < 16U; ++cell)
    {
        const uint64_t value = (state >> (60U - 4U * permutation[cell])) & 0xFU;
        result |= value << (60U - 4U * cell);
    }

    return result;
}

static uint64_t
rotate_left_64(const uint64_t value, const uint32_t amount)
{
    return (value << amount) | (value >> (64U - amount));
}

/// PACMult: each column times the involutory matrix circ(0, rho, rho^2, rho), rho rotating a
/// cell left by one bit. Rotating the state left by 16 bits brings row r + 1 into row r.
static uint64_t
multiply_columns(const uint64_t state)
{
    const uint64_t rho1 = ((state << 1U) & CELL_HIGH_BITS) | ((state >> 3U) & CELL_LOW_BITS);
    const uint64_t rho2
        = ((state << 2U) & 0xCCCCCCCCCCCCCCCCULL) | ((state >> 2U) & 0x3333333333333333ULL);
    return rotate_left_64(rho1, 16U) ^ rotate_left_64(rho2, 32U) ^ rotate_left_64(rho1, 48U);
}

/// TweakShuffle: the tweak permutation, then an LFSR step (b3, b2, b1, b0) to
/// (b0 ^ b1, b3, b2, b1) on some cells.
static uint64_t
shuffle_tweak(const uint64_t tweak)
{
    const uint64_t shuffled = permute_cells(tweak, TWEAK_SHUFFLE);
    const uint64_t stepped  = ((shuffled >> 1U) & 0x7777777777777777ULL)
                             | (((shuffled ^ (shuffled >> 1U)) & CELL_LOW_BITS) << 3U);
    return (shuffled & ~TWEAK_ROTATED_CELLS) | (stepped & TWEAK_ROTATED_CELLS);
}

/// TweakInvShuffle: the inverse LFSR step, then the inverse permutation.
static uint64_t
unshuffle_tweak(const uint64_t tweak)
{
    const uint64_t stepped
        = ((tweak << 1U) & CELL_HIGH_BITS) | (((tweak >> 3U) ^ tweak) & CELL_LOW_BITS);
    const uint64_t unstepped = (tweak & ~TWEAK_ROTATED_CELLS) | (stepped & TWEAK_ROTATED_CELLS);
    return permute_cells(unstepped, INVERSE_TWEAK_SHUFFLE);
}

/// ComputePAC(): QARMA5 encryption of `data` under the tweak `modifier` and the key halves
/// `key0` (the high half) and `key1` (the low half).
static uint64_t
compute_pac_uncached(const uint64_t data,
                     const uint64_t modifier,
                     const uint64_t key0,
                     const uint64_t key1)
{
    const uint64_t modified_key0
        = (key0 << 63U) | ((key0 >> 2U) << 1U) | (((key0 >> 63U) ^ (key0 >> 1U)) & 1U);
    uint64_t tweak = modifier;
    uint64_t state = data ^ key0;

    for (uint32_t i = 0U; i < 5U; ++i)
    {
        state ^= key1 ^ tweak ^ ROUND_CONSTANTS[i];

        if (i > 0U)
        {
            state = multiply_columns(permute_cells(state, CELL_SHUFFLE));
        }

        state = substitute(state, SUBSTITUTION);
        tweak = shuffle_tweak(tweak);
    }

    state ^= modified_key0 ^ tweak;
    state = substitute(multiply_columns(permute_cells(state, CELL_SHUFFLE)), SUBSTITUTION);
    state = multiply_columns(permute_cells(state, CELL_SHUFFLE));
    state ^= key1;
    state = permute_cells(state, INVERSE_CELL_SHUFFLE);
    state = permute_cells(multiply_columns(substitute(state, INVERSE_SUBSTITUTION)),
                          INVERSE_CELL_SHUFFLE);
    state ^= key0 ^ tweak;

    for (uint32_t i = 0U; i < 5U; ++i)
    {
        state = substitute(state, INVERSE_SUBSTITUTION);

        if (i < 4U)
        {
            state = permute_cells(multiply_columns(state), INVERSE_CELL_SHUFFLE);
        }

        tweak = unshuffle_tweak(tweak);
        state ^= ROUND_CONSTANTS[4U - i] ^ key1 ^ tweak ^ ALPHA;
    }

    return state ^ modified_key0;
}

/// A recently computed code. Games sign the same return addresses at the same stack depths over
/// and over, so a small cache saves most of the cipher's work.
typedef struct
{
    uint64_t data;
    uint64_t modifier;
    uint64_t key0;
    uint64_t key1;
    uint64_t pac;
} pac_cache_entry_t;

#define PAC_CACHE_ENTRIES 256U

static BAL_THREAD_LOCAL pac_cache_entry_t g_pac_cache[PAC_CACHE_ENTRIES];

/// ComputePAC(), through the per-thread cache. The zero-initialised entries hold the code of all
/// zeroes, which is computed rather than assumed, so the cache needs no valid bit.
static uint64_t
compute_pac(const uint64_t data, const uint64_t modifier, const uint64_t key0, const uint64_t key1)
{
    const uint64_t     hash  = (data ^ (modifier * 0x9E3779B97F4A7C15ULL)) * 0xBF58476D1CE4E5B9ULL;
    pac_cache_entry_t *entry = &g_pac_cache[hash >> 56U];

    if (entry->data != data || entry->modifier != modifier || entry->key0 != key0
        || entry->key1 != key1 || (0U == (data | modifier | key0 | key1) && 0U == entry->pac))
    {
        entry->data     = data;
        entry->modifier = modifier;
        entry->key0     = key0;
        entry->key1     = key1;
        entry->pac      = compute_pac_uncached(data, modifier, key0, key1);
    }

    return entry->pac;
}

typedef enum
{
    KEY_IA,
    KEY_IB,
    KEY_DA,
    KEY_DB,
    KEY_GA,
} key_t;

/// The address layout of a pointer: whether its top byte is ignored and its lowest PAC bit.
typedef struct
{
    uint32_t bottom_bit;
    bool     is_top_byte_ignored;
    uint8_t  pad[3];
} layout_t;

/// EffectiveTBI(): whether the top byte of an address in the upper (`is_upper`) or lower range
/// is ignored. TBID keeps the top byte of instruction addresses in the code.
static bool
is_top_byte_ignored(const bal_cpu_t *cpu, const bool is_upper, const bool is_data)
{
    const uint64_t tcr  = cpu->tcr_el1;
    const bool     tbi  = 0U != (tcr & (true == is_upper ? TCR_TBI1 : TCR_TBI0));
    const bool     tbid = 0U != (tcr & (true == is_upper ? TCR_TBID1 : TCR_TBID0));
    return tbi && (true == is_data || false == tbid);
}

/// CalculateBottomPACBit(): 64 - TxSZ of the selected range, TxSZ clamped to its valid values.
static uint32_t
bottom_pac_bit(const bal_cpu_t *cpu, const bool is_upper)
{
    uint32_t tsz = (uint32_t)(cpu->tcr_el1 >> (true == is_upper ? 16U : 0U)) & 0x3FU;
    tsz          = tsz < MINIMUM_TSZ ? MINIMUM_TSZ : (tsz > MAXIMUM_TSZ ? MAXIMUM_TSZ : tsz);
    return 64U - tsz;
}

/// The layout of a pointer whose range bit 55 selects, as Auth() and Strip() see it.
static layout_t
layout_of(const bal_cpu_t *cpu, const uint64_t pointer, const bool is_data)
{
    const bool     is_upper = 0U != ((pointer >> 55U) & 1U);
    const layout_t layout   = {
          .bottom_bit          = bottom_pac_bit(cpu, is_upper),
          .is_top_byte_ignored = is_top_byte_ignored(cpu, is_upper, is_data),
    };
    return layout;
}

static uint64_t
bit_range_mask(const uint32_t top, const uint32_t bottom)
{
    return (~0ULL >> (63U - top)) & (~0ULL << bottom);
}

/// The pointer with its PAC field replaced by copies of bit 55.
static uint64_t
strip(const uint64_t pointer, const layout_t *layout)
{
    const uint32_t top       = true == layout->is_top_byte_ignored ? 55U : 63U;
    const uint64_t field     = bit_range_mask(top, layout->bottom_bit);
    const bool     extension = 0U != ((pointer >> 55U) & 1U);
    return (pointer & ~field) | (true == extension ? field : 0U);
}

static void
get_key(const bal_cpu_t *cpu, const key_t key, uint64_t *high, uint64_t *low)
{
    *low  = cpu->pac_keys[key][0];
    *high = cpu->pac_keys[key][1];
}

/// AddPAC(): inserts the code for `pointer` under `modifier` and `key`.
static uint64_t
add_pac(const bal_cpu_t *cpu, const uint64_t pointer, const uint64_t modifier, const key_t key)
{
    const bool     is_data = KEY_DA == key || KEY_DB == key;
    const bool     tbi     = is_top_byte_ignored(cpu, 0U != ((pointer >> 55U) & 1U), is_data);
    const uint32_t top     = true == tbi ? 55U : 63U;

    // The range comes from bit 55 when either range ignores the top byte and from bit 63
    // otherwise, which settles pointers whose bits 55 and 63 disagree. It is kept in bit 55.
    const bool any_tbi
        = is_top_byte_ignored(cpu, false, is_data) || is_top_byte_ignored(cpu, true, is_data);
    const bool     selection = 0U != ((pointer >> (true == any_tbi ? 55U : 63U)) & 1U);
    const layout_t layout    = {
           .bottom_bit          = bottom_pac_bit(cpu, selection),
           .is_top_byte_ignored = tbi,
    };
    const uint64_t field    = bit_range_mask(top, layout.bottom_bit);
    const uint64_t extended = (pointer & ~field) | (true == selection ? field : 0U);
    uint64_t       high     = 0U;
    uint64_t       low      = 0U;

    get_key(cpu, key, &high, &low);
    uint64_t pac = compute_pac(extended, modifier, high, low);

    // A pointer whose extension bits disagree gets a corrupted code, so that authenticating it
    // fails.
    const uint64_t extension = pointer & field;

    if (extension != 0U && extension != field)
    {
        pac ^= 1ULL << (top - 1U);
    }

    const uint64_t code_mask = field & ~(1ULL << 55U);
    const uint64_t result    = (pointer & ~field) | (pac & code_mask);
    return result | (true == selection ? 1ULL << 55U : 0U);
}

/// Auth(): the stripped pointer if its code matches, and otherwise the stripped pointer with an
/// error code that makes it non-canonical.
static uint64_t
authenticate(const bal_cpu_t *cpu, const uint64_t pointer, const uint64_t modifier, const key_t key)
{
    const bool     is_data  = KEY_DA == key || KEY_DB == key;
    const layout_t layout   = layout_of(cpu, pointer, is_data);
    const uint32_t top      = true == layout.is_top_byte_ignored ? 55U : 63U;
    const uint64_t original = strip(pointer, &layout);
    uint64_t       high     = 0U;
    uint64_t       low      = 0U;

    get_key(cpu, key, &high, &low);
    const uint64_t pac       = compute_pac(original, modifier, high, low);
    const uint64_t code_mask = bit_range_mask(top, layout.bottom_bit) & ~(1ULL << 55U);

    if (0U == ((pac ^ pointer) & code_mask))
    {
        return original;
    }

    // The error code is the key number followed by its inverse: 01 for A keys, 10 for B keys.
    const uint64_t error_code = KEY_IB == key || KEY_DB == key ? 2U : 1U;
    const uint32_t error_bit  = true == layout.is_top_byte_ignored ? 53U : 61U;
    return (original & ~(3ULL << error_bit)) | (error_code << error_bit);
}

static bool
is_key_enabled(const bal_cpu_t *cpu, const key_t key)
{
    static const uint64_t ENABLE_BITS[4] = { SCTLR_ENIA, SCTLR_ENIB, SCTLR_ENDA, SCTLR_ENDB };
    return 0U != (cpu->sctlr_el1 & ENABLE_BITS[key]);
}

/// AddPACIA() and the other AddPAC* and Auth* functions: unchanged when the key is disabled.
static uint64_t
sign_or_authenticate(const bal_cpu_t *cpu,
                     const uint64_t   pointer,
                     const uint64_t   modifier,
                     const key_t      key,
                     const bool       is_authenticate)
{
    if (false == is_key_enabled(cpu, key))
    {
        return pointer;
    }

    return true == is_authenticate ? authenticate(cpu, pointer, modifier, key)
                                   : add_pac(cpu, pointer, modifier, key);
}

void
bal_pac_memo_attach(bal_pac_memo_t *memo, bal_cpu_t *cpu)
{
    const uint64_t configuration[8] = {
        cpu->pac_keys[KEY_IA][0],
        cpu->pac_keys[KEY_IA][1],
        cpu->pac_keys[KEY_IB][0],
        cpu->pac_keys[KEY_IB][1],
        cpu->tcr_el1,
        0U,
        0U,
        0U,
    };

    if (0 != memcmp(memo->configuration, configuration, sizeof(configuration)))
    {
        (void)memset(memo->entries, 0, sizeof(memo->entries));
        (void)memcpy(memo->configuration, configuration, sizeof(configuration));
    }

    cpu->jit_pac_memo = (uint64_t)(uintptr_t)memo->entries;
}

/// Records `plain` and `signed_pointer` in the memo when signing the one gives the other and
/// authenticating the other gives the one back.
static void
remember(const bal_cpu_t *cpu,
         const uint64_t   plain,
         const uint64_t   signed_pointer,
         const uint64_t   modifier,
         const key_t      key)
{
    bal_pac_memo_entry_t *memo = (bal_pac_memo_entry_t *)(uintptr_t)cpu->jit_pac_memo;

    if (NULL == memo || add_pac(cpu, plain, modifier, key) != signed_pointer
        || authenticate(cpu, signed_pointer, modifier, key) != plain)
    {
        return;
    }

    bal_pac_memo_entry_t *entry = &memo[BAL_PAC_MEMO_INDEX(plain, modifier)];
    entry->key                  = (uint64_t)key + 1U;
    entry->modifier             = modifier;
    entry->plain                = plain;
    entry->signed_pointer       = signed_pointer;
}

/// sign_or_authenticate() for the forms translated code looks up in the memo, recording the
/// result there.
static uint64_t
sign_or_authenticate_remembered(const bal_cpu_t *cpu,
                                const uint64_t   pointer,
                                const uint64_t   modifier,
                                const key_t      key,
                                const bool       is_authenticate)
{
    const uint64_t result = sign_or_authenticate(cpu, pointer, modifier, key, is_authenticate);

    if (true == is_key_enabled(cpu, key))
    {
        if (true == is_authenticate)
        {
            remember(cpu, result, pointer, modifier, key);
        }
        else
        {
            remember(cpu, pointer, result, modifier, key);
        }
    }

    return result;
}

static uint64_t
read_register(const bal_cpu_t *cpu, const uint32_t index)
{
    return 31U == index ? 0U : cpu->x[index];
}

/// Reads Xn or, for register 31, SP.
static uint64_t
read_register_or_sp(const bal_cpu_t *cpu, const uint32_t index)
{
    return cpu->x[index];
}

static void
write_register(bal_cpu_t *cpu, const uint32_t index, const uint64_t value)
{
    if (index != 31U)
    {
        cpu->x[index] = value;
    }
}

/// PACIA, PACIB, PACDA, PACDB, AUTIA, AUTIB, AUTDA, AUTDB, their Z forms, XPACI and XPACD:
/// 1 1 0 11010110 00001 opcode Rn Rd.
static bool
is_data_processing(const uint32_t instruction)
{
    if (0xDAC10000U != (instruction & 0xFFFF0000U))
    {
        return false;
    }

    const uint32_t opcode = (instruction >> 10U) & 0x3FU;
    const uint32_t rn     = (instruction >> 5U) & 0x1FU;

    // The Z forms (001xxx) and XPACI, XPACD (01000x) take no Rn.
    return opcode < 0x08U || (opcode < 0x12U && 31U == rn);
}

static bool
is_pacga(const uint32_t instruction)
{
    return 0x9AC03000U == (instruction & 0xFFE0FC00U);
}

/// The hint-space forms: the even ones of HINT #8 to #15 (CRm = 0001, the 1716 forms), #24 to #31
/// (CRm = 0011, the Z and SP forms) and #7 (XPACLRI).
static bool
is_hint(const uint32_t instruction)
{
    if (0xD50320FFU == instruction)
    {
        return true;
    }

    const bool is_1716 = 0U == ((instruction >> 9U) & 1U);
    return 0xD503211FU == (instruction & 0xFFFFFD1FU)
           && (false == is_1716 || 0U == ((instruction >> 5U) & 1U));
}

bool
bal_is_authenticated_branch(const uint32_t instruction)
{
    // 1101011 Z 0 op 11111 0000 1 M Rn Rm, with op 00 (BR), 01 (BLR) or 10 (RET).
    if (0xD61F0800U != (instruction & 0xFE9FF800U))
    {
        return false;
    }

    const bool     has_modifier = 0U != ((instruction >> 24U) & 1U);
    const uint32_t op           = (instruction >> 21U) & 0x3U;
    const uint32_t rn           = (instruction >> 5U) & 0x1FU;
    const uint32_t rm           = instruction & 0x1FU;

    if (3U == op)
    {
        return false;
    }

    // RETAA and RETAB: Z = 0, op = 10, Rn = Rm = 11111.
    if (2U == op)
    {
        return false == has_modifier && 31U == rn && 31U == rm;
    }

    // BRAAZ and BLRAAZ have Rm = 11111.
    return true == has_modifier || 31U == rm;
}

bool
bal_is_authenticated_load(const uint32_t instruction)
{
    // 11 111 0 00 M S 1 imm9 W 1 Rn Rt.
    return 0xF8200400U == (instruction & 0xFF200400U);
}

bool
bal_is_pointer_authentication_supported(const uint32_t instruction)
{
    return is_data_processing(instruction) || is_pacga(instruction) || is_hint(instruction)
           || bal_is_authenticated_branch(instruction);
}

uint64_t
bal_execute_pointer_authentication(bal_cpu_t *cpu, const uint32_t instruction)
{
    if (true == is_data_processing(instruction))
    {
        const uint32_t opcode = (instruction >> 10U) & 0x3FU;
        const uint32_t rn     = (instruction >> 5U) & 0x1FU;
        const uint32_t rd     = instruction & 0x1FU;
        const uint64_t value  = read_register(cpu, rd);

        if (opcode >= 0x10U)
        {
            // XPACI and XPACD.
            const layout_t layout = layout_of(cpu, value, 0x11U == opcode);
            write_register(cpu, rd, strip(value, &layout));
            return 1U;
        }

        // Bits 1:0 pick the key, bit 2 authentication, bit 3 a zero modifier.
        const key_t    key      = (key_t)(opcode & 0x3U);
        const uint64_t modifier = 0U != (opcode & 0x8U) ? 0U : read_register_or_sp(cpu, rn);
        write_register(
            cpu, rd, sign_or_authenticate(cpu, value, modifier, key, 0U != (opcode & 0x4U)));
        return 1U;
    }

    if (true == is_pacga(instruction))
    {
        const uint32_t rm   = (instruction >> 16U) & 0x1FU;
        const uint32_t rn   = (instruction >> 5U) & 0x1FU;
        const uint32_t rd   = instruction & 0x1FU;
        uint64_t       high = 0U;
        uint64_t       low  = 0U;

        get_key(cpu, KEY_GA, &high, &low);
        const uint64_t pac
            = compute_pac(read_register(cpu, rn), read_register_or_sp(cpu, rm), high, low);
        write_register(cpu, rd, pac & 0xFFFFFFFF00000000ULL);
        return 1U;
    }

    if (true == is_hint(instruction))
    {
        // XPACLRI strips X30 as an instruction address.
        if (0xD50320FFU == instruction)
        {
            const layout_t layout = layout_of(cpu, cpu->x[30], false);
            cpu->x[30]            = strip(cpu->x[30], &layout);
            return 1U;
        }

        // op2 bit 1 picks the B key and bit 2 authentication. CRm = 0001 works on X17 with the
        // modifier X16; CRm = 0011 on X30 with zero (op2 bit 0 clear) or SP.
        const uint32_t op2             = (instruction >> 5U) & 0x7U;
        const bool     is_1716         = 0U == ((instruction >> 9U) & 1U);
        const key_t    key             = 0U != (op2 & 0x2U) ? KEY_IB : KEY_IA;
        const bool     is_authenticate = 0U != (op2 & 0x4U);
        const uint32_t target          = true == is_1716 ? 17U : 30U;
        const uint64_t modifier        = true == is_1716      ? cpu->x[16]
                                         : 0U != (op2 & 0x1U) ? cpu->x[31]
                                                              : 0U;
        cpu->x[target]
            = sign_or_authenticate_remembered(cpu, cpu->x[target], modifier, key, is_authenticate);
        return 1U;
    }

    // The authenticated branches: M (bit 10) picks the B key; RETAA and RETAB authenticate X30
    // against SP, the Z = 1 forms against Xm or SP, and BRAAZ and BLRAAZ against zero.
    const bool     is_return    = 0x00400000U == (instruction & 0x01600000U);
    const bool     has_modifier = 0U != ((instruction >> 24U) & 1U);
    const key_t    key          = 0U != ((instruction >> 10U) & 1U) ? KEY_IB : KEY_IA;
    const uint32_t rn           = true == is_return ? 30U : (instruction >> 5U) & 0x1FU;
    const uint64_t modifier     = true == is_return ? cpu->x[31]
                                  : true == has_modifier ? read_register_or_sp(cpu, instruction & 0x1FU)
                                                         : 0U;
    cpu->pc                     = true == is_return
                                      ? sign_or_authenticate_remembered(cpu, cpu->x[30], modifier, key, true)
                                      : sign_or_authenticate(cpu, read_register(cpu, rn), modifier, key, true);
    return 1U;
}

uint64_t
bal_execute_authenticated_load(bal_cpu_t                    *cpu,
                               const bal_memory_interface_t *memory_interface,
                               const uint32_t                instruction)
{
    const uint32_t rn            = (instruction >> 5U) & 0x1FU;
    const uint32_t rt            = instruction & 0x1FU;
    const bool     has_writeback = 0U != ((instruction >> 11U) & 1U);
    const key_t    key           = 0U != ((instruction >> 23U) & 1U) ? KEY_DB : KEY_DA;

    // The offset is S:imm9 scaled by 8.
    const uint64_t offset_field = ((instruction >> 12U) & 0x1FFU) | ((instruction >> 13U) & 0x200U);
    const int64_t  offset       = (int64_t)(offset_field << 54U) >> 51;
    const uint64_t base    = sign_or_authenticate(cpu, read_register_or_sp(cpu, rn), 0U, key, true);
    const uint64_t address = base + (uint64_t)offset;

    size_t         available_bytes = 0U;
    const uint8_t *host_address
        = memory_interface->translate((void *)memory_interface, address, &available_bytes);

    if (NULL == host_address || available_bytes < 8U)
    {
        return 0U;
    }

    uint64_t value = 0U;
    (void)memcpy(&value, host_address, sizeof(value));

    if (true == has_writeback)
    {
        cpu->x[rn] = address;
    }

    write_register(cpu, rt, value);
    return 1U;
}

/*** end of file ***/
