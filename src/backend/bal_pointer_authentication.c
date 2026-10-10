#include "backend/bal_pointer_authentication.h"

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
// bits 63:60 and cell 15 bits 3:0.

static const uint8_t SUBSTITUTION[16] = { 0xBU, 0x6U, 0x8U, 0xFU, 0xCU, 0x0U, 0x9U, 0xEU,
                                          0x3U, 0x7U, 0x4U, 0x5U, 0xDU, 0x2U, 0x1U, 0xAU };

/// The cell permutation: output cell i is input cell CELL_SHUFFLE[i].
static const uint8_t CELL_SHUFFLE[16]
    = { 0U, 11U, 6U, 13U, 10U, 1U, 12U, 7U, 5U, 14U, 3U, 8U, 15U, 4U, 9U, 2U };

/// The tweak permutation, and the output cells the tweak's LFSR then steps.
static const uint8_t TWEAK_SHUFFLE[16]
    = { 6U, 5U, 14U, 15U, 0U, 1U, 2U, 3U, 7U, 12U, 13U, 4U, 8U, 9U, 10U, 11U };
static const uint16_t TWEAK_ROTATED_CELLS
    = (1U << 0U) | (1U << 1U) | (1U << 3U) | (1U << 4U) | (1U << 8U) | (1U << 11U) | (1U << 13U);

static const uint64_t ROUND_CONSTANTS[5] = {
    0x0000000000000000ULL, 0x13198A2E03707344ULL, 0xA4093822299F31D0ULL,
    0x082EFA98EC4E6C89ULL, 0x452821E638D01377ULL,
};
static const uint64_t ALPHA = 0xC0AC29B7C97C50DDULL;

static uint32_t
get_cell(const uint64_t state, const uint32_t cell)
{
    return (uint32_t)(state >> (60U - 4U * cell)) & 0xFU;
}

static uint64_t
with_cell(const uint64_t state, const uint32_t cell, const uint32_t value)
{
    const uint32_t shift = 60U - 4U * cell;
    return (state & ~(0xFULL << shift)) | ((uint64_t)(value & 0xFU) << shift);
}

static uint64_t
substitute(const uint64_t state, const bool is_inverse)
{
    uint64_t result = 0U;

    for (uint32_t cell = 0U; cell < 16U; ++cell)
    {
        const uint32_t value = get_cell(state, cell);
        uint32_t       out   = SUBSTITUTION[value];

        if (true == is_inverse)
        {
            for (uint32_t i = 0U; i < 16U; ++i)
            {
                if (SUBSTITUTION[i] == value)
                {
                    out = i;
                }
            }
        }

        result = with_cell(result, cell, out);
    }

    return result;
}

static uint64_t
shuffle_cells(const uint64_t state, const bool is_inverse)
{
    uint64_t result = 0U;

    for (uint32_t cell = 0U; cell < 16U; ++cell)
    {
        if (true == is_inverse)
        {
            result = with_cell(result, CELL_SHUFFLE[cell], get_cell(state, cell));
        }
        else
        {
            result = with_cell(result, cell, get_cell(state, CELL_SHUFFLE[cell]));
        }
    }

    return result;
}

static uint32_t
rotate_cell(const uint32_t cell, const uint32_t amount)
{
    return ((cell << amount) | (cell >> (4U - amount))) & 0xFU;
}

/// PACMult: each column times the involutory matrix circ(0, rho, rho^2, rho), rho being a one
/// bit rotation of a cell.
static uint64_t
multiply_columns(const uint64_t state)
{
    uint64_t result = 0U;

    for (uint32_t column = 0U; column < 4U; ++column)
    {
        uint32_t rows[4];

        for (uint32_t row = 0U; row < 4U; ++row)
        {
            rows[row] = get_cell(state, 4U * row + column);
        }

        for (uint32_t row = 0U; row < 4U; ++row)
        {
            const uint32_t value = rotate_cell(rows[(row + 1U) % 4U], 1U)
                                   ^ rotate_cell(rows[(row + 2U) % 4U], 2U)
                                   ^ rotate_cell(rows[(row + 3U) % 4U], 1U);
            result = with_cell(result, 4U * row + column, value);
        }
    }

    return result;
}

/// TweakShuffle and its inverse: the tweak permutation with an LFSR step on some cells.
static uint64_t
shuffle_tweak(const uint64_t tweak, const bool is_inverse)
{
    uint64_t result = 0U;

    if (false == is_inverse)
    {
        for (uint32_t cell = 0U; cell < 16U; ++cell)
        {
            uint32_t value = get_cell(tweak, TWEAK_SHUFFLE[cell]);

            if (0U != ((TWEAK_ROTATED_CELLS >> cell) & 1U))
            {
                // (b3, b2, b1, b0) becomes (b0 ^ b1, b3, b2, b1).
                value = (value >> 1U) | (((value ^ (value >> 1U)) & 1U) << 3U);
            }

            result = with_cell(result, cell, value);
        }

        return result;
    }

    for (uint32_t cell = 0U; cell < 16U; ++cell)
    {
        uint32_t value = get_cell(tweak, cell);

        if (0U != ((TWEAK_ROTATED_CELLS >> cell) & 1U))
        {
            value = ((value << 1U) & 0xEU) | (((value >> 3U) ^ value) & 1U);
        }

        result = with_cell(result, TWEAK_SHUFFLE[cell], value);
    }

    return result;
}

/// ComputePAC(): QARMA5 encryption of `data` under the tweak `modifier` and the key halves
/// `key0` (the high half) and `key1` (the low half).
static uint64_t
compute_pac(const uint64_t data, const uint64_t modifier, const uint64_t key0, const uint64_t key1)
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
            state = multiply_columns(shuffle_cells(state, false));
        }

        state = substitute(state, false);
        tweak = shuffle_tweak(tweak, false);
    }

    state ^= modified_key0 ^ tweak;
    state = substitute(multiply_columns(shuffle_cells(state, false)), false);
    state = multiply_columns(shuffle_cells(state, false));
    state ^= key1;
    state = shuffle_cells(state, true);
    state = shuffle_cells(multiply_columns(substitute(state, true)), true);
    state ^= key0 ^ tweak;

    for (uint32_t i = 0U; i < 5U; ++i)
    {
        state = substitute(state, true);

        if (i < 4U)
        {
            state = shuffle_cells(multiply_columns(state), true);
        }

        tweak = shuffle_tweak(tweak, true);
        state ^= ROUND_CONSTANTS[4U - i] ^ key1 ^ tweak ^ ALPHA;
    }

    return state ^ modified_key0;
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
        cpu->x[target] = sign_or_authenticate(cpu, cpu->x[target], modifier, key, is_authenticate);
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
    cpu->pc = sign_or_authenticate(cpu, read_register(cpu, rn), modifier, key, true);
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
