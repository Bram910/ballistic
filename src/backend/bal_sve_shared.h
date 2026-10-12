//! The types and helpers the SVE and SME interpreters (bal_sve.c and bal_sme.c) share. Private
//! to the backend.
#ifndef BALLISTIC_BAL_SVE_SHARED_H
#define BALLISTIC_BAL_SVE_SHARED_H

#include "backend/bal_float.h"
#include "backend/bal_sve.h"
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/// The outcome of executing an instruction.
typedef enum
{
    SVE_OK,
    /// The encoding is unallocated or not implemented; nothing was changed.
    SVE_UNDEFINED,
    /// An access was not mapped.
    SVE_FAULT,
    /// An SME instruction needs streaming mode or ZA storage, which is off.
    SVE_ILLEGAL,
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

#define FPSR_QC (1ULL << 27U)

static inline uint64_t
mask_of(const uint32_t esize)
{
    return esize >= 64U ? ~0ULL : (1ULL << esize) - 1U;
}

static inline int64_t
sign_extend(const uint64_t value, const uint32_t esize)
{
    const uint32_t shift = 64U - esize;
    return (int64_t)(value << shift) >> shift;
}

static inline uint64_t
get_element(const vector_t *vector, const uint32_t index, const uint32_t esize)
{
    const uint32_t bit = index * esize;
    return (vector->d[bit / 64U] >> (bit % 64U)) & mask_of(esize);
}

static inline void
set_element(vector_t *vector, const uint32_t index, const uint32_t esize, const uint64_t value)
{
    const uint32_t bit  = index * esize;
    const uint64_t mask = mask_of(esize) << (bit % 64U);
    uint64_t      *word = &vector->d[bit / 64U];
    *word               = (*word & ~mask) | ((value << (bit % 64U)) & mask);
}

static inline vector_t
read_z(const sve_t *sve, const uint32_t index)
{
    vector_t vector = { { sve->cpu->v[index][0], sve->cpu->v[index][1] } };
    return vector;
}

static inline void
write_z(sve_t *sve, const uint32_t index, const vector_t vector)
{
    sve->cpu->v[index][0] = vector.d[0];
    sve->cpu->v[index][1] = vector.d[1];
}

/// Whether element `index` of `esize` bits is active in `predicate`: its lowest bit.
static inline bool
is_active(const uint16_t predicate, const uint32_t index, const uint32_t esize)
{
    return 0U != ((predicate >> (index * (esize / 8U))) & 1U);
}

/// A predicate with element `index` set as `value` and its other bits for that element clear.
static inline uint16_t
set_active(uint16_t predicate, const uint32_t index, const uint32_t esize, const bool value)
{
    const uint32_t bits = esize / 8U;
    const uint32_t mask = ((1U << bits) - 1U) << (index * bits);
    predicate           = (uint16_t)(predicate & ~mask);
    return (uint16_t)(predicate | (true == value ? 1U << (index * bits) : 0U));
}

static inline uint32_t
elements_of(const uint32_t esize)
{
    return VECTOR_BITS / esize;
}

static inline void
write_nzcv(sve_t *sve, const uint32_t nzcv)
{
    sve->cpu->flag_n = (uint8_t)((nzcv >> 3U) & 1U);
    sve->cpu->flag_z = (uint8_t)((nzcv >> 2U) & 1U);
    sve->cpu->flag_c = (uint8_t)((nzcv >> 1U) & 1U);
    sve->cpu->flag_v = (uint8_t)(nzcv & 1U);
}

/// PredTest(): N is the first active element of `result` under `mask`, Z that none is set, C
/// that the last active one is clear.
static inline uint32_t
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

static inline void
write_x(sve_t *sve, const uint32_t index, const uint64_t value)
{
    if (index != 31U)
    {
        sve->cpu->x[index] = value;
    }
}

static inline uint64_t
read_x(const sve_t *sve, const uint32_t index)
{
    return 31U == index ? 0U : sve->cpu->x[index];
}

static inline uint16_t
read_p(const sve_t *sve, const uint32_t index)
{
    return sve->cpu->p[index];
}

/// The esize of a two-bit size field at bit 22.
static inline uint32_t
size_at_22(const uint32_t instruction)
{
    return 8U << ((instruction >> 22U) & 0x3U);
}

static inline uint64_t
count_active(const uint16_t mask, const uint16_t operand, const uint32_t esize)
{
    uint64_t count = 0U;

    for (uint32_t e = 0U; e < elements_of(esize); ++e)
    {
        count
            += true == is_active(mask, e, esize) && true == is_active(operand, e, esize) ? 1U : 0U;
    }

    return count;
}

/// The host address of `bytes` bytes at `address` when they are contiguous, or NULL.
static inline uint8_t *
host_span(const sve_t *sve, const uint64_t address, const uint64_t bytes)
{
    // Like the other helpers, pass the interface itself; the flat interface reads its context.
    size_t         available = 0U;
    const uint8_t *host      = sve->memory->translate(
        (void *)(uintptr_t)sve->memory, (bal_guest_address_t)address, &available);
    return NULL != host && available >= bytes ? (uint8_t *)(uintptr_t)host : NULL;
}

/// Whether all `bytes` bytes at `address` are mapped.
static inline bool
is_mapped(const sve_t *sve, const uint64_t address, const uint64_t bytes)
{
    for (uint64_t i = 0U; i < bytes; ++i)
    {
        if (NULL == host_span(sve, address + i, 1U))
        {
            return false;
        }
    }

    return true;
}

/// Reads `bytes` (1 to 8) little-endian bytes. The caller has checked they are mapped.
static inline uint64_t
memory_read(const sve_t *sve, const uint64_t address, const uint32_t bytes)
{
    uint64_t       value = 0U;
    const uint8_t *host  = host_span(sve, address, bytes);

    for (uint32_t i = 0U; i < bytes; ++i)
    {
        const uint8_t byte = NULL != host ? host[i] : *host_span(sve, address + i, 1U);
        value |= (uint64_t)byte << (8U * i);
    }

    return value;
}

static inline void
memory_write(const sve_t *sve, const uint64_t address, const uint32_t bytes, const uint64_t value)
{
    uint8_t *host = host_span(sve, address, bytes);

    for (uint32_t i = 0U; i < bytes; ++i)
    {
        uint8_t *byte = NULL != host ? &host[i] : host_span(sve, address + i, 1U);
        *byte         = (uint8_t)(value >> (8U * i));
    }
}

#define FPSR_IOC (1ULL << 0U)

#define FPSR_IDC (1ULL << 7U)

static inline uint64_t
fp(sve_t                      *sve,
   const uint32_t              esize,
   const bal_float_operation_t operation,
   const uint64_t              a,
   const uint64_t              b,
   const uint64_t              c)
{
    return bal_float_element(sve->cpu->fpcr, &sve->flags, esize, operation, a, b, c);
}

static inline uint64_t
fp_negate(const uint64_t value, const uint32_t esize)
{
    return value ^ (1ULL << (esize - 1U));
}

static inline uint32_t
fp_fraction_bits(const uint32_t esize)
{
    return 16U == esize ? 10U : (32U == esize ? 23U : 52U);
}

static inline bool
fp_is_nan(const uint64_t value, const uint32_t esize)
{
    const uint32_t fraction = fp_fraction_bits(esize);
    const uint64_t exponent = mask_of(esize - 1U - fraction);
    return ((value >> fraction) & exponent) == exponent && 0U != (value & mask_of(fraction));
}

/// The FP element size of a size field: 1, 2 and 3 are half, single and double precision.
static inline uint32_t
fp_esize(const uint32_t size)
{
    return 0U == size ? 0U : 8U << size;
}

/// How a wide intermediate result returns to an element.
typedef enum
{
    SATURATE_NONE,
    SATURATE_SIGNED,
    SATURATE_UNSIGNED,
} saturate_t;

/// An element as a wide integer.
static inline __int128
wide(const uint64_t value, const uint32_t esize, const bool is_signed)
{
    return true == is_signed ? (__int128)sign_extend(value, esize)
                             : (__int128)(value & mask_of(esize));
}

/// A wide result as an `esize`-bit element, truncated or saturated.
static inline uint64_t
narrow_to(const __int128 value, const uint32_t esize, const saturate_t mode)
{
    if (SATURATE_UNSIGNED == mode)
    {
        return value < 0 ? 0U
                         : (value > (__int128)mask_of(esize) ? mask_of(esize) : (uint64_t)value);
    }

    if (SATURATE_SIGNED == mode)
    {
        const __int128 maximum = (__int128)(mask_of(esize) >> 1U);
        const __int128 minimum = -maximum - 1;
        const __int128 clamped = value > maximum ? maximum : (value < minimum ? minimum : value);
        return (uint64_t)clamped & mask_of(esize);
    }

    return (uint64_t)value & mask_of(esize);
}

/// `value`, an element of `esize` bits widened, shifted left by `amount` (right when negative),
/// rounding right shifts when `is_rounding`. Left shifts past the element keep only the sign.
static inline __int128
shift_wide(__int128 value, const int64_t amount, const uint32_t esize, const bool is_rounding)
{
    if (amount >= 0)
    {
        if (amount >= (int64_t)esize)
        {
            const __int128 big = (__int128)1 << 100U;
            return 0 == value ? 0 : (value > 0 ? big : -big);
        }

        return value * ((__int128)1 << amount);
    }

    int64_t shift = -amount;
    shift         = shift > (int64_t)esize + 1 ? (int64_t)esize + 1 : shift;

    if (true == is_rounding)
    {
        value += (__int128)1 << (shift - 1);
    }

    return value >> shift;
}

/// The high half of twice `product` plus `rounding`, as (2 * product + rounding) >> esize
/// without overflowing on 64-bit elements.
static inline __int128
doubled_high(const __int128 product, const uint32_t esize, const bool is_rounding)
{
    if (64U == esize)
    {
        return (product + (true == is_rounding ? (__int128)1 << 62U : 0)) >> 63U;
    }

    return (2 * product + (true == is_rounding ? (__int128)1 << (esize - 1U) : 0)) >> esize;
}

/// SQDMULH and SQRDMULH on one element: the high half of twice the product, saturated.
static inline uint64_t
doubling_multiply_high(const uint64_t a,
                       const uint64_t b,
                       const uint32_t esize,
                       const bool     is_rounding)
{
    const __int128 product = wide(a, esize, true) * wide(b, esize, true);
    return narrow_to(doubled_high(product, esize, is_rounding), esize, SATURATE_SIGNED);
}

/// PSTATE.SM and PSTATE.ZA in bal_cpu_t::svcr.
#define SVCR_SM 1U
#define SVCR_ZA 2U

/// Executes an instruction of the SME encoding space (bal_sme.c).
sve_result_t sme_execute(sve_t *sve, uint32_t instruction);

#endif /* BALLISTIC_BAL_SVE_SHARED_H */

/*** end of file ***/
