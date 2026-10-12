#include "backend/bal_atomic.h"

#include <string.h>

/// The instruction classes bal_execute_atomic() handles.
typedef enum
{
    ATOMIC_UNSUPPORTED,
    /// LDXR/LDAXR/STXR/STLXR and their byte and halfword forms.
    ATOMIC_EXCLUSIVE,
    /// LDXP/LDAXP/STXP/STLXP.
    ATOMIC_EXCLUSIVE_PAIR,
    /// LDAR/LDLAR/STLR/STLLR and their byte and halfword forms.
    ATOMIC_ORDERED,
    /// CAS/CASA/CASL/CASAL and their byte and halfword forms.
    ATOMIC_COMPARE_SWAP,
    /// CASP/CASPA/CASPL/CASPAL.
    ATOMIC_COMPARE_SWAP_PAIR,
    /// LDADD/LDCLR/LDEOR/LDSET/LDSMAX/LDSMIN/LDUMAX/LDUMIN and SWP, in every ordering and size.
    ATOMIC_MEMORY_OPERATION,
    /// LDAPR and its byte and halfword forms.
    ATOMIC_LOAD_ACQUIRE_PC,
    /// LDAPUR/STLUR and their sign-extending, byte and halfword forms.
    ATOMIC_ORDERED_UNSCALED,
    /// CLREX.
    ATOMIC_CLEAR_EXCLUSIVE,
    /// FEAT_LRCPC3: LDAPR with a post-index and STLR with a pre-index writeback.
    ATOMIC_ORDERED_WRITEBACK,
    /// FEAT_LRCPC3: LDIAPP and STILP.
    ATOMIC_ORDERED_PAIR,
    /// FEAT_LRCPC3: LDAPUR and STLUR on SIMD&FP registers.
    ATOMIC_ORDERED_UNSCALED_VECTOR,
    /// FEAT_LRCPC3: LDAP1 and STL1, one 64-bit lane.
    ATOMIC_ORDERED_LANE,
} atomic_class_t;

static atomic_class_t
classify(const uint32_t instruction)
{
    const uint32_t size = instruction >> 30U;
    const uint32_t rs   = (instruction >> 16U) & 0x1FU;
    const uint32_t rt   = instruction & 0x1FU;

    // CLREX: 1101 0101 0000 0011 0011 CRm 010 11111.
    if (0xD503305FU == (instruction & 0xFFFFF0FFU))
    {
        return ATOMIC_CLEAR_EXCLUSIVE;
    }

    // Load/store exclusive, ordered and compare-and-swap: size 001000 o2 L o1 Rs o0 Rt2 Rn Rt.
    if (0x08000000U == (instruction & 0x3F000000U))
    {
        const bool o2 = 1U == ((instruction >> 23U) & 1U);
        const bool o1 = 1U == ((instruction >> 21U) & 1U);

        if (false == o2 && false == o1)
        {
            return ATOMIC_EXCLUSIVE;
        }

        if (true == o2)
        {
            return true == o1 ? ATOMIC_COMPARE_SWAP : ATOMIC_ORDERED;
        }

        if (size >= 2U)
        {
            return ATOMIC_EXCLUSIVE_PAIR;
        }

        // CASP needs even register pairs; odd ones are unallocated.
        return 0U == (rs & 1U) && 0U == (rt & 1U) ? ATOMIC_COMPARE_SWAP_PAIR : ATOMIC_UNSUPPORTED;
    }

    // Atomic memory operations: size 111 V 00 A R 1 Rs o3 opc 00 Rn Rt, with V = 0.
    if (0x38200000U == (instruction & 0x3F200C00U))
    {
        const uint32_t o3  = (instruction >> 15U) & 1U;
        const uint32_t opc = (instruction >> 12U) & 0x7U;

        if (0U == o3 || 0U == opc)
        {
            return ATOMIC_MEMORY_OPERATION;
        }

        // LDAPR is A = 1, R = 0 with Rs = 11111; the other o3 = 1 encodings are unallocated or
        // belong to extensions tier 1 does not implement.
        if (4U == opc && 0x2U == ((instruction >> 22U) & 0x3U) && 31U == rs)
        {
            return ATOMIC_LOAD_ACQUIRE_PC;
        }

        return ATOMIC_UNSUPPORTED;
    }

    // FEAT_LRCPC3 LDAPR (post-index) and STLR (pre-index): 1 x 011001 1 L 0 00000 100010 Rn Rt.
    if (0x99800800U == (instruction & 0xBFBFFC00U))
    {
        return ATOMIC_ORDERED_WRITEBACK;
    }

    // FEAT_LRCPC3 LDIAPP and STILP: 1 x 011001 0 L 0 Rt2 000 o 10 Rn Rt, with o = 0 for the
    // writeback forms.
    if (0x99000800U == (instruction & 0xBFA0EC00U))
    {
        return ATOMIC_ORDERED_PAIR;
    }

    // FEAT_LRCPC3 LDAPUR and STLUR (SIMD&FP): size 011101 opc 0 imm9 10 Rn Rt; opc 1x is the
    // 128-bit form, which only size 00 has.
    if (0x1D000800U == (instruction & 0x3F200C00U))
    {
        return 0U == size || 0U == (instruction & 0x00800000U) ? ATOMIC_ORDERED_UNSCALED_VECTOR
                                                               : ATOMIC_UNSUPPORTED;
    }

    // FEAT_LRCPC3 LDAP1 and STL1: 0 Q 001101 0 L 0 00001 100001 Rn Rt.
    if (0x0D018400U == (instruction & 0xBFBFFC00U))
    {
        return ATOMIC_ORDERED_LANE;
    }

    // LDAPR/STLR (unscaled immediate): size 011001 opc 0 imm9 00 Rn Rt.
    if (0x19000000U == (instruction & 0x3F200C00U))
    {
        const uint32_t opc = (instruction >> 22U) & 0x3U;

        // 64-bit accesses cannot sign-extend, and 32-bit ones only into an X register.
        if ((3U == size && opc >= 2U) || (2U == size && 3U == opc))
        {
            return ATOMIC_UNSUPPORTED;
        }

        return ATOMIC_ORDERED_UNSCALED;
    }

    return ATOMIC_UNSUPPORTED;
}

bool
bal_is_atomic_supported(const uint32_t instruction)
{
    return classify(instruction) != ATOMIC_UNSUPPORTED;
}

/// Reads a general purpose register, with register 31 as the zero register.
static uint64_t
read_register(const bal_cpu_t *BAL_RESTRICT cpu, const uint32_t index)
{
    return 31U == index ? 0U : cpu->x[index];
}

/// Writes a general purpose register, discarding writes to the zero register.
static void
write_register(bal_cpu_t *BAL_RESTRICT cpu, const uint32_t index, const uint64_t value)
{
    if (index != 31U)
    {
        cpu->x[index] = value;
    }
}

static uint64_t
size_mask(const uint32_t bytes)
{
    return bytes >= 8U ? ~0ULL : (1ULL << (bytes * 8U)) - 1U;
}

static int64_t
sign_extend(const uint64_t value, const uint32_t bytes)
{
    const uint32_t shift = 64U - (bytes * 8U);
    return (int64_t)(value << shift) >> shift;
}

/// Returns the host address of `bytes` bytes at `address`, or NULL if they are not all mapped.
static uint8_t *
translate(const bal_memory_interface_t *BAL_RESTRICT memory_interface,
          const uint64_t                             address,
          const uint64_t                             bytes)
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

static bool
is_host_aligned(const uint8_t *pointer, const uint32_t bytes)
{
    return 0U == ((uintptr_t)pointer & (bytes - 1U));
}

/// Loads `bytes` (1 to 8) from `pointer` as one atomic access when the host address allows it.
static uint64_t
load(const uint8_t *pointer, const uint32_t bytes)
{
    if (false == is_host_aligned(pointer, bytes))
    {
        uint64_t value = 0U;
        (void)memcpy(&value, pointer, bytes);
        return value;
    }

    switch (bytes)
    {
        case 1U:
            return __atomic_load_n(pointer, __ATOMIC_SEQ_CST);
        case 2U:
            return __atomic_load_n((const uint16_t *)(const void *)pointer, __ATOMIC_SEQ_CST);
        case 4U:
            return __atomic_load_n((const uint32_t *)(const void *)pointer, __ATOMIC_SEQ_CST);
        default:
            return __atomic_load_n((const uint64_t *)(const void *)pointer, __ATOMIC_SEQ_CST);
    }
}

/// Stores the low `bytes` (1 to 8) of `value` to `pointer` as one atomic access when the host
/// address allows it.
static void
store(uint8_t *pointer, const uint32_t bytes, const uint64_t value)
{
    if (false == is_host_aligned(pointer, bytes))
    {
        (void)memcpy(pointer, &value, bytes);
        return;
    }

    switch (bytes)
    {
        case 1U:
            __atomic_store_n(pointer, (uint8_t)value, __ATOMIC_SEQ_CST);
            break;
        case 2U:
            __atomic_store_n((uint16_t *)(void *)pointer, (uint16_t)value, __ATOMIC_SEQ_CST);
            break;
        case 4U:
            __atomic_store_n((uint32_t *)(void *)pointer, (uint32_t)value, __ATOMIC_SEQ_CST);
            break;
        default:
            __atomic_store_n((uint64_t *)(void *)pointer, value, __ATOMIC_SEQ_CST);
            break;
    }
}

/// Replaces the `bytes` (1 to 8) at `pointer` with `desired` if they equal `expected`. Returns
/// the previous contents; the swap happened if they equal `expected`.
static uint64_t
compare_swap(uint8_t       *pointer,
             const uint32_t bytes,
             const uint64_t expected,
             const uint64_t desired)
{
    switch (bytes)
    {
        case 1U: {
            uint8_t old = (uint8_t)expected;
            (void)__atomic_compare_exchange_n(
                pointer, &old, (uint8_t)desired, false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);
            return old;
        }
        case 2U: {
            uint16_t old = (uint16_t)expected;
            (void)__atomic_compare_exchange_n((uint16_t *)(void *)pointer,
                                              &old,
                                              (uint16_t)desired,
                                              false,
                                              __ATOMIC_SEQ_CST,
                                              __ATOMIC_SEQ_CST);
            return old;
        }
        case 4U: {
            uint32_t old = (uint32_t)expected;
            (void)__atomic_compare_exchange_n((uint32_t *)(void *)pointer,
                                              &old,
                                              (uint32_t)desired,
                                              false,
                                              __ATOMIC_SEQ_CST,
                                              __ATOMIC_SEQ_CST);
            return old;
        }
        default: {
            uint64_t old = expected;
            (void)__atomic_compare_exchange_n((uint64_t *)(void *)pointer,
                                              &old,
                                              desired,
                                              false,
                                              __ATOMIC_SEQ_CST,
                                              __ATOMIC_SEQ_CST);
            return old;
        }
    }
}

/// The 16-byte form of compare_swap(). `value` holds the expected low and high halves on entry
/// and the previous contents on return. Returns true if the swap happened.
static bool
compare_swap_16(uint8_t *pointer, uint64_t value[2], const uint64_t desired[2])
{
    // The guest address is 16-byte aligned, so the host address only is not when the memory
    // interface maps guest memory at an odd host offset. CMPXCHG16B would fault there, so fall
    // back to a plain compare and copy.
    if (false == is_host_aligned(pointer, 16U))
    {
        uint64_t current[2];
        (void)memcpy(current, pointer, sizeof(current));
        const bool is_equal = current[0] == value[0] && current[1] == value[1];

        if (true == is_equal)
        {
            (void)memcpy(pointer, desired, sizeof(current));
        }

        value[0] = current[0];
        value[1] = current[1];
        return is_equal;
    }

#if defined(__x86_64__)
    bool is_equal = false;
    __asm__ __volatile__("lock cmpxchg16b %1"
                         : "=@ccz"(is_equal),
                           "+m"(*(volatile uint64_t(*)[2])(void *)pointer),
                           "+a"(value[0]),
                           "+d"(value[1])
                         : "b"(desired[0]), "c"(desired[1])
                         : "memory");
    return is_equal;
#else
    unsigned __int128 expected = ((unsigned __int128)value[1] << 64U) | (unsigned __int128)value[0];
    const unsigned __int128 replacement
        = ((unsigned __int128)desired[1] << 64U) | (unsigned __int128)desired[0];
    const bool is_equal = __atomic_compare_exchange_n((unsigned __int128 *)(void *)pointer,
                                                      &expected,
                                                      replacement,
                                                      false,
                                                      __ATOMIC_SEQ_CST,
                                                      __ATOMIC_SEQ_CST);
    value[0]            = (uint64_t)expected;
    value[1]            = (uint64_t)(expected >> 64U);
    return is_equal;
#endif
}

/// Loads 16 bytes as one atomic access, by swapping them with themselves.
static void
load_16(uint8_t *pointer, uint64_t value[2])
{
    value[0]               = 0U;
    value[1]               = 0U;
    const uint64_t zero[2] = { 0U, 0U };
    (void)compare_swap_16(pointer, value, zero);
}

/// Computes the new memory value of an LSE atomic memory operation.
static uint64_t
apply_memory_operation(const uint32_t o3,
                       const uint32_t opc,
                       const uint64_t old,
                       const uint64_t operand,
                       const uint32_t bytes)
{
    if (1U == o3)
    {
        return operand; // SWP
    }

    switch (opc)
    {
        case 0U:
            return old + operand; // LDADD
        case 1U:
            return old & ~operand; // LDCLR
        case 2U:
            return old ^ operand; // LDEOR
        case 3U:
            return old | operand; // LDSET
        case 4U:
            return sign_extend(old, bytes) > sign_extend(operand, bytes) ? old : operand; // LDSMAX
        case 5U:
            return sign_extend(old, bytes) < sign_extend(operand, bytes) ? old : operand; // LDSMIN
        case 6U:
            return old > operand ? old : operand; // LDUMAX
        default:
            return old < operand ? old : operand; // LDUMIN
    }
}

uint64_t
bal_execute_atomic(bal_cpu_t *BAL_RESTRICT                    cpu,
                   const bal_memory_interface_t *BAL_RESTRICT memory_interface,
                   const uint32_t                             instruction)
{
    const atomic_class_t instruction_class = classify(instruction);
    const uint32_t       size              = instruction >> 30U;
    const uint32_t       rt                = instruction & 0x1FU;
    const uint32_t       rn                = (instruction >> 5U) & 0x1FU;
    const uint32_t       rt2               = (instruction >> 10U) & 0x1FU;
    const uint32_t       rs                = (instruction >> 16U) & 0x1FU;
    const bool           is_load           = 1U == ((instruction >> 22U) & 1U);

    // The base register is the stack pointer when it is register 31, which x[31] holds.
    const uint64_t address = cpu->x[rn];
    uint32_t       bytes   = 1U << size;

    switch (instruction_class)
    {
        case ATOMIC_CLEAR_EXCLUSIVE:
            cpu->exclusive_state = 0U;
            return 1U;

        case ATOMIC_EXCLUSIVE:
        case ATOMIC_EXCLUSIVE_PAIR: {
            const bool     is_pair        = ATOMIC_EXCLUSIVE_PAIR == instruction_class;
            const uint32_t register_bytes = true == is_pair ? (2U == size ? 4U : 8U) : bytes;
            bytes                         = true == is_pair ? register_bytes * 2U : bytes;

            if (true == is_load)
            {
                uint8_t *pointer = 0U == (address & (bytes - 1U))
                                       ? translate(memory_interface, address, bytes)
                                       : NULL;

                if (NULL == pointer)
                {
                    return 0U;
                }

                uint64_t value[2] = { 0U, 0U };

                if (16U == bytes)
                {
                    load_16(pointer, value);
                    write_register(cpu, rt, value[0]);
                    write_register(cpu, rt2, value[1]);
                }
                else
                {
                    value[0] = load(pointer, bytes);

                    if (true == is_pair)
                    {
                        write_register(cpu, rt, value[0] & 0xFFFFFFFFU);
                        write_register(cpu, rt2, value[0] >> 32U);
                    }
                    else
                    {
                        write_register(cpu, rt, value[0]);
                    }
                }

                cpu->exclusive_state    = 1U;
                cpu->exclusive_address  = address;
                cpu->exclusive_value[0] = value[0];

                // Only a 16-byte load-exclusive replaces the high half of the reserved value, as
                // in QEMU, so a 16-byte store-exclusive after a narrower load-exclusive, which is
                // CONSTRAINED UNPREDICTABLE, compares against the same value as Unicorn.
                if (16U == bytes)
                {
                    cpu->exclusive_value[1] = value[1];
                }

                return 1U;
            }

            // A store-exclusive to another address than the reservation fails without accessing
            // memory, and every store-exclusive clears the reservation.
            uint64_t status = 1U;

            if (1U == cpu->exclusive_state && address == cpu->exclusive_address)
            {
                uint8_t *pointer = 0U == (address & (bytes - 1U))
                                       ? translate(memory_interface, address, bytes)
                                       : NULL;

                if (NULL == pointer)
                {
                    return 0U;
                }

                if (16U == bytes)
                {
                    uint64_t expected[2] = { cpu->exclusive_value[0], cpu->exclusive_value[1] };
                    const uint64_t desired[2] = { read_register(cpu, rt), read_register(cpu, rt2) };
                    status = true == compare_swap_16(pointer, expected, desired) ? 0U : 1U;
                }
                else
                {
                    uint64_t value = read_register(cpu, rt) & size_mask(register_bytes);

                    if (true == is_pair)
                    {
                        value |= read_register(cpu, rt2) << 32U;
                    }

                    // A store-exclusive of another size than its load-exclusive is CONSTRAINED
                    // UNPREDICTABLE. Like QEMU, the swap compares the low `bytes` of the reserved
                    // value, but success needs the whole value to match, so a narrower store can
                    // write memory and still report failure.
                    const uint64_t expected = cpu->exclusive_value[0] & size_mask(bytes);
                    const uint64_t old      = compare_swap(pointer, bytes, expected, value);
                    status                  = old == cpu->exclusive_value[0] ? 0U : 1U;
                }
            }

            cpu->exclusive_state = 0U;
            write_register(cpu, rs, status);
            return 1U;
        }

        case ATOMIC_ORDERED:
        case ATOMIC_LOAD_ACQUIRE_PC:
        case ATOMIC_ORDERED_UNSCALED: {
            // These need no alignment beyond what plain loads and stores need.
            uint64_t target   = address;
            uint32_t extend   = 0U; // 0: zero-extend, 32/64: sign-extend into a W/X register.
            bool     is_store = ATOMIC_ORDERED == instruction_class && false == is_load;

            if (ATOMIC_ORDERED_UNSCALED == instruction_class)
            {
                const uint32_t opc  = (instruction >> 22U) & 0x3U;
                const uint32_t imm9 = (instruction >> 12U) & 0x1FFU;

                // WARNING: Shifting the 9-bit field to the top and back sign-extends it.
                target   = address + (uint64_t)(int64_t)((int32_t)(imm9 << 23U) >> 23U);
                is_store = 0U == opc;
                extend   = 2U == opc ? 64U : (3U == opc ? 32U : 0U);
            }

            uint8_t *pointer = translate(memory_interface, target, bytes);

            if (NULL == pointer)
            {
                return 0U;
            }

            if (true == is_store)
            {
                store(pointer, bytes, read_register(cpu, rt));
                return 1U;
            }

            uint64_t value = load(pointer, bytes);

            if (64U == extend)
            {
                value = (uint64_t)sign_extend(value, bytes);
            }
            else if (32U == extend)
            {
                value = (uint64_t)sign_extend(value, bytes) & 0xFFFFFFFFU;
            }

            write_register(cpu, rt, value);
            return 1U;
        }

        case ATOMIC_COMPARE_SWAP:
        case ATOMIC_COMPARE_SWAP_PAIR: {
            const bool     is_pair        = ATOMIC_COMPARE_SWAP_PAIR == instruction_class;
            const uint32_t register_bytes = true == is_pair ? (0U == size ? 4U : 8U) : bytes;
            bytes                         = true == is_pair ? register_bytes * 2U : bytes;

            uint8_t *pointer = 0U == (address & (bytes - 1U))
                                   ? translate(memory_interface, address, bytes)
                                   : NULL;

            if (NULL == pointer)
            {
                return 0U;
            }

            if (16U == bytes)
            {
                uint64_t       value[2]   = { read_register(cpu, rs), read_register(cpu, rs + 1U) };
                const uint64_t desired[2] = { read_register(cpu, rt), read_register(cpu, rt + 1U) };
                (void)compare_swap_16(pointer, value, desired);
                write_register(cpu, rs, value[0]);
                write_register(cpu, rs + 1U, value[1]);
                return 1U;
            }

            const uint64_t mask     = size_mask(register_bytes);
            uint64_t       expected = read_register(cpu, rs) & mask;
            uint64_t       desired  = read_register(cpu, rt) & mask;

            if (true == is_pair)
            {
                expected |= read_register(cpu, rs + 1U) << 32U;
                desired |= read_register(cpu, rt + 1U) << 32U;
            }

            const uint64_t old = compare_swap(pointer, bytes, expected, desired);

            if (true == is_pair)
            {
                write_register(cpu, rs, old & 0xFFFFFFFFU);
                write_register(cpu, rs + 1U, old >> 32U);
            }
            else
            {
                write_register(cpu, rs, old);
            }

            return 1U;
        }

        case ATOMIC_MEMORY_OPERATION: {
            uint8_t *pointer = 0U == (address & (bytes - 1U))
                                   ? translate(memory_interface, address, bytes)
                                   : NULL;

            if (NULL == pointer)
            {
                return 0U;
            }

            const uint32_t o3      = (instruction >> 15U) & 1U;
            const uint32_t opc     = (instruction >> 12U) & 0x7U;
            const uint64_t mask    = size_mask(bytes);
            const uint64_t operand = read_register(cpu, rs) & mask;
            uint64_t       old     = load(pointer, bytes);

            for (;;)
            {
                const uint64_t desired
                    = apply_memory_operation(o3, opc, old, operand, bytes) & mask;
                const uint64_t current = compare_swap(pointer, bytes, old, desired);

                if (current == old)
                {
                    break;
                }

                old = current;
            }

            write_register(cpu, rt, old);
            return 1U;
        }

        case ATOMIC_ORDERED_WRITEBACK: {
            // LDAPR loads from Xn and then adds the size to it; STLR subtracts the size first and
            // stores there.
            const bool     is_store = 0U == (instruction & 0x00400000U);
            const uint64_t target   = true == is_store ? address - bytes : address;
            uint8_t       *pointer  = translate(memory_interface, target, bytes);

            if (NULL == pointer)
            {
                return 0U;
            }

            if (true == is_store)
            {
                store(pointer, bytes, read_register(cpu, rt));
                cpu->x[rn] = target;
            }
            else
            {
                const uint64_t value = load(pointer, bytes);
                cpu->x[rn]           = address + bytes;
                write_register(cpu, rt, value);
            }

            return 1U;
        }

        case ATOMIC_ORDERED_PAIR: {
            // LDIAPP loads from Xn and, in its writeback form, then adds the pair size; STILP
            // subtracts it first in its writeback form.
            const uint32_t register_bytes = 4U << (size & 1U);
            const bool     is_load_pair   = 0U != (instruction & 0x00400000U);
            const bool     has_writeback  = 0U == (instruction & 0x00001000U);
            const uint64_t pair_bytes     = 2U * register_bytes;
            const uint64_t target
                = false == is_load_pair && true == has_writeback ? address - pair_bytes : address;
            uint8_t *pointer = translate(memory_interface, target, pair_bytes);

            if (NULL == pointer)
            {
                return 0U;
            }

            const uint64_t mask = size_mask(register_bytes);

            if (true == is_load_pair)
            {
                const uint64_t first  = load(pointer, register_bytes);
                const uint64_t second = load(pointer + register_bytes, register_bytes);

                if (true == has_writeback)
                {
                    cpu->x[rn] = address + pair_bytes;
                }

                write_register(cpu, rt, first & mask);
                write_register(cpu, rs, second & mask);
            }
            else
            {
                const uint64_t first  = read_register(cpu, rt);
                const uint64_t second = read_register(cpu, rs);
                store(pointer, register_bytes, first);
                store(pointer + register_bytes, register_bytes, second);

                if (true == has_writeback)
                {
                    cpu->x[rn] = target;
                }
            }

            return 1U;
        }

        case ATOMIC_ORDERED_UNSCALED_VECTOR: {
            const uint32_t opc  = (instruction >> 22U) & 0x3U;
            const uint32_t imm9 = (instruction >> 12U) & 0x1FFU;

            // WARNING: Shifting the 9-bit field to the top and back sign-extends it.
            const uint64_t target = address + (uint64_t)(int64_t)((int32_t)(imm9 << 23U) >> 23U);
            bytes                 = opc >= 2U ? 16U : bytes;
            uint8_t *pointer      = translate(memory_interface, target, bytes);

            if (NULL == pointer)
            {
                return 0U;
            }

            if (0U == (opc & 1U))
            {
                if (16U == bytes)
                {
                    store(pointer, 8U, cpu->v[rt][0]);
                    store(pointer + 8U, 8U, cpu->v[rt][1]);
                }
                else
                {
                    store(pointer, bytes, cpu->v[rt][0]);
                }

                return 1U;
            }

            // A load clears the register above the element.
            cpu->v[rt][1] = 16U == bytes ? load(pointer + 8U, 8U) : 0U;
            cpu->v[rt][0] = load(pointer, 16U == bytes ? 8U : bytes);
            return 1U;
        }

        case ATOMIC_ORDERED_LANE: {
            const uint32_t lane    = (instruction >> 30U) & 1U;
            uint8_t       *pointer = translate(memory_interface, address, 8U);

            if (NULL == pointer)
            {
                return 0U;
            }

            if (0U != (instruction & 0x00400000U))
            {
                cpu->v[rt][lane] = load(pointer, 8U);
            }
            else
            {
                store(pointer, 8U, cpu->v[rt][lane]);
            }

            return 1U;
        }

        case ATOMIC_UNSUPPORTED:
        default:
            return 0U;
    }
}

/*** end of file ***/
