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
