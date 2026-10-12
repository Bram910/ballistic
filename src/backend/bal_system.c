// clock_gettime() needs POSIX.1b on the hosts that have it.
#define _POSIX_C_SOURCE 199309L

#include "backend/bal_system.h"

#include "bal_platform.h"
#include <string.h>

#if BAL_PLATFORM_WINDOWS
#include <windows.h>
#else
#include <time.h>
#endif

/// The bit-reversed CRC-32 (ISO-HDLC) and CRC-32C (Castagnoli) polynomials.
#define CRC32_POLYNOMIAL  0xEDB88320U
#define CRC32C_POLYNOMIAL 0x82F63B78U

static bool
is_crc32(const uint32_t instruction)
{
    // sf 0 0 11010110 Rm 010 C sz Rn Rd; X operands (sf = 1) exactly when sz = 11.
    const uint32_t size = (instruction >> 10U) & 0x3U;
    return 0x1AC04000U == (instruction & 0x7FE0E000U)
           && (instruction >> 31U) == (3U == size ? 1U : 0U);
}

/// FEAT_MTE ADDG and SUBG: 1 op 0 100011 0 uimm6 00 uimm4 Xn Xd.
static bool
is_tag_arithmetic(const uint32_t instruction)
{
    return 0x91800000U == (instruction & 0xBFC0C000U);
}

/// FEAT_MTE SUBP and SUBPS (1 S 0 11010110 Xm 000000 Xn Xd), IRG (... 000100 ...) and GMI
/// (... 000101 ...).
static bool
is_tag_register(const uint32_t instruction)
{
    return 0x9AC00000U == (instruction & 0xDFE0FC00U) || 0x9AC01000U == (instruction & 0xFFE0F800U);
}

/// FEAT_CSSC ABS, CNT and CTZ: sf 1 0 11010110 00000 00 opcode Rn Rd, with opcode 000110 (CTZ),
/// 000111 (CNT) or 001000 (ABS).
static bool
is_cssc_one_source(const uint32_t instruction)
{
    const uint32_t opcode = (instruction >> 10U) & 0x3FU;
    return 0x5AC00000U == (instruction & 0x7FFFC000U) && opcode >= 0x6U && opcode <= 0x8U;
}

/// FEAT_CSSC SMAX, UMAX, SMIN and UMIN: sf 0 0 11010110 Rm 0110 op Rn Rd (register) and
/// sf 0 0 1000111 00 op imm8 Rn Rd (immediate).
static bool
is_cssc_minimum_maximum(const uint32_t instruction)
{
    return 0x1AC06000U == (instruction & 0x7FE0F000U) || 0x11C00000U == (instruction & 0x7FF00000U);
}

static bool
is_cfinv(const uint32_t instruction)
{
    return 0xD500401FU == instruction;
}

static bool
is_rmif(const uint32_t instruction)
{
    // 1 0 1 11010000 imm6 00001 Rn 0 mask.
    return 0xBA000400U == (instruction & 0xFFE07C10U);
}

static bool
is_setf(const uint32_t instruction)
{
    // 0 0 1 11010000 000000 sz 0010 Rn 0 1101.
    return 0x3A00080DU == (instruction & 0xFFFFBC1FU);
}

static bool
is_flag_format(const uint32_t instruction)
{
    // AXFLAG (0xD500405F) and XAFLAG (0xD500403F).
    return 0xD500401FU == (instruction & 0xFFFFFF9FU)
           && (0x20U == (instruction & 0x60U) || 0x40U == (instruction & 0x60U));
}

bool
bal_is_integer_helper_supported(const uint32_t instruction)
{
    return is_crc32(instruction) || is_cfinv(instruction) || is_rmif(instruction)
           || is_setf(instruction) || is_flag_format(instruction) || is_cssc_one_source(instruction)
           || is_cssc_minimum_maximum(instruction) || is_tag_arithmetic(instruction)
           || is_tag_register(instruction);
}

static uint64_t
read_register(const bal_cpu_t *cpu, const uint32_t index)
{
    return 31U == index ? 0U : cpu->x[index];
}

/// Updates a bit-reversed CRC with the low `bytes` bytes of `value`, least significant first.
static uint32_t
crc_update(uint32_t crc, const uint64_t value, const uint32_t bytes, const uint32_t polynomial)
{
    for (uint32_t i = 0U; i < bytes; ++i)
    {
        crc ^= (uint32_t)(value >> (8U * i)) & 0xFFU;

        for (uint32_t bit = 0U; bit < 8U; ++bit)
        {
            crc = 0U != (crc & 1U) ? (crc >> 1U) ^ polynomial : crc >> 1U;
        }
    }

    return crc;
}

uint64_t
bal_execute_integer_helper(bal_cpu_t *cpu, const uint32_t instruction)
{
    const uint32_t rn = (instruction >> 5U) & 0x1FU;

    if (true == is_crc32(instruction))
    {
        const uint32_t rd         = instruction & 0x1FU;
        const uint32_t rm         = (instruction >> 16U) & 0x1FU;
        const uint32_t bytes      = 1U << ((instruction >> 10U) & 0x3U);
        const bool     is_c       = 1U == ((instruction >> 12U) & 1U);
        const uint32_t polynomial = true == is_c ? CRC32C_POLYNOMIAL : CRC32_POLYNOMIAL;
        const uint32_t crc        = crc_update(
            (uint32_t)read_register(cpu, rn), read_register(cpu, rm), bytes, polynomial);

        if (rd != 31U)
        {
            cpu->x[rd] = crc;
        }

        return 1U;
    }

    if (true == is_cfinv(instruction))
    {
        cpu->flag_c ^= 1U;
        return 1U;
    }

    if (true == is_tag_arithmetic(instruction) || true == is_tag_register(instruction))
    {
        // Allocation tag access is disabled at EL0 (SCTLR_EL1.ATA0 = 0), so ADDG, SUBG and IRG
        // produce tag 0. Register 31 is SP for every address operand, and for the destination
        // of ADDG, SUBG and IRG.
        const uint64_t tag_mask = 0xFULL << 56U;
        const uint32_t rd       = instruction & 0x1FU;
        const uint32_t rm       = (instruction >> 16U) & 0x1FU;
        const uint64_t address  = cpu->x[rn];

        if (true == is_tag_arithmetic(instruction))
        {
            const uint64_t offset = (uint64_t)((instruction >> 16U) & 0x3FU) << 4U;
            const bool     is_sub = 0U != (instruction & 0x40000000U);
            cpu->x[rd] = ((true == is_sub) ? address - offset : address + offset) & ~tag_mask;
            return 1U;
        }

        switch ((instruction >> 10U) & 0x3FU)
        {
            case 0x4U: // IRG
                cpu->x[rd] = address & ~tag_mask;
                return 1U;
            case 0x5U: // GMI
                if (rd != 31U)
                {
                    cpu->x[rd] = read_register(cpu, rm) | (1ULL << ((address >> 56U) & 0xFU));
                }
                return 1U;
            default: // SUBP and SUBPS on the 56-bit addresses, sign-extended.
            {
                const uint64_t first  = (uint64_t)((int64_t)(address << 8U) >> 8U);
                const uint64_t second = (uint64_t)((int64_t)(cpu->x[rm] << 8U) >> 8U);
                const uint64_t result = first - second;

                if (0U != (instruction & 0x20000000U))
                {
                    cpu->flag_n = (uint8_t)(result >> 63U);
                    cpu->flag_z = (uint8_t)(0U == result ? 1U : 0U);
                    cpu->flag_c = (uint8_t)(first >= second ? 1U : 0U);
                    cpu->flag_v = (uint8_t)(((first ^ second) & (first ^ result)) >> 63U);
                }

                if (rd != 31U)
                {
                    cpu->x[rd] = result;
                }

                return 1U;
            }
        }
    }

    if (true == is_cssc_one_source(instruction) || true == is_cssc_minimum_maximum(instruction))
    {
        const uint32_t rd       = instruction & 0x1FU;
        const bool     is_64bit = 0U != (instruction >> 31U);
        const uint32_t bits     = true == is_64bit ? 64U : 32U;
        const uint64_t mask     = true == is_64bit ? ~0ULL : 0xFFFFFFFFULL;
        const uint64_t value    = read_register(cpu, rn) & mask;
        uint64_t       result   = 0U;

        if (true == is_cssc_one_source(instruction))
        {
            switch ((instruction >> 10U) & 0x3FU)
            {
                case 0x6U: // CTZ
                    result = 0U == value ? bits : (uint64_t)__builtin_ctzll(value);
                    break;
                case 0x7U: // CNT
                    result = (uint64_t)__builtin_popcountll(value);
                    break;
                default: // ABS
                {
                    const bool is_negative = 0U != ((value >> (bits - 1U)) & 1U);
                    result                 = true == is_negative ? (0U - value) & mask : value;
                    break;
                }
            }
        }
        else
        {
            // The immediate forms take a signed (SMAX, SMIN) or unsigned imm8.
            const bool     is_immediate = 0x11C00000U == (instruction & 0x7FF00000U);
            const uint32_t op
                = true == is_immediate ? (instruction >> 18U) & 0x3U : (instruction >> 10U) & 0x3U;
            const bool is_unsigned = 0U != (op & 0x1U);
            const bool is_minimum  = 0U != (op & 0x2U);
            uint64_t   other       = 0U;

            if (true == is_immediate)
            {
                const uint64_t imm8 = (instruction >> 10U) & 0xFFU;
                other = true == is_unsigned ? imm8 : (uint64_t)(int64_t)(int8_t)imm8 & mask;
            }
            else
            {
                other = read_register(cpu, (instruction >> 16U) & 0x1FU) & mask;
            }

            // Compare as unsigned, or as signed by flipping the sign bits.
            const uint64_t sign      = true == is_unsigned ? 0U : 1ULL << (bits - 1U);
            const bool     is_larger = (value ^ sign) > (other ^ sign);

            result = is_larger != is_minimum ? value : other;
        }

        if (rd != 31U)
        {
            cpu->x[rd] = result;
        }

        return 1U;
    }

    if (true == is_flag_format(instruction))
    {
        const uint8_t z = cpu->flag_z;
        const uint8_t c = cpu->flag_c;

        if (0x40U == (instruction & 0x60U))
        {
            // AXFLAG: from the Arm floating-point condition flags to the external format.
            cpu->flag_z = (uint8_t)(z | cpu->flag_v);
            cpu->flag_c = (uint8_t)(c & (cpu->flag_v ^ 1U));
            cpu->flag_n = 0U;
            cpu->flag_v = 0U;
        }
        else
        {
            // XAFLAG: back from the external format.
            cpu->flag_n = (uint8_t)((c | z) ^ 1U);
            cpu->flag_z = (uint8_t)(z & c);
            cpu->flag_c = (uint8_t)(c | z);
            cpu->flag_v = (uint8_t)((c ^ 1U) & z);
        }

        return 1U;
    }

    if (true == is_rmif(instruction))
    {
        // Rotate Xn right by imm6 and copy the selected low four bits into N, Z, C and V.
        const uint32_t shift    = (instruction >> 15U) & 0x3FU;
        const uint32_t mask     = instruction & 0xFU;
        const uint64_t value    = read_register(cpu, rn);
        const uint64_t rotated  = 0U == shift ? value : (value >> shift) | (value << (64U - shift));
        uint8_t       *flags[4] = { &cpu->flag_v, &cpu->flag_c, &cpu->flag_z, &cpu->flag_n };

        for (uint32_t i = 0U; i < 4U; ++i)
        {
            if (0U != ((mask >> i) & 1U))
            {
                *flags[i] = (uint8_t)((rotated >> i) & 1U);
            }
        }

        return 1U;
    }

    // SETF8 and SETF16: N and Z from the low byte or halfword, V from the bit above it differing
    // from its top bit; C is unchanged.
    const uint32_t bits  = 0U == ((instruction >> 14U) & 1U) ? 8U : 16U;
    const uint64_t value = read_register(cpu, rn);
    const uint64_t top   = (value >> (bits - 1U)) & 1U;
    cpu->flag_n          = (uint8_t)top;
    cpu->flag_z          = (uint8_t)(0U == (value & ((1ULL << bits) - 1U)) ? 1U : 0U);
    cpu->flag_v          = (uint8_t)(((value >> bits) & 1U) ^ top);
    return 1U;
}

/// Returns the host's monotonic clock as a count at `frequency` Hz.
static uint64_t
read_host_counter(const uint64_t frequency)
{
#if BAL_PLATFORM_WINDOWS
    LARGE_INTEGER count;
    LARGE_INTEGER host_frequency;
    (void)QueryPerformanceCounter(&count);
    (void)QueryPerformanceFrequency(&host_frequency);
    const uint64_t ticks      = (uint64_t)count.QuadPart;
    const uint64_t host_hertz = (uint64_t)host_frequency.QuadPart;
    return (ticks / host_hertz) * frequency + ((ticks % host_hertz) * frequency) / host_hertz;
#else
    struct timespec now = { 0 };
    (void)clock_gettime(CLOCK_MONOTONIC, &now);
    return (uint64_t)now.tv_sec * frequency + ((uint64_t)now.tv_nsec * frequency) / 1000000000U;
#endif
}

uint64_t
bal_execute_counter_read(bal_cpu_t *cpu, const uint32_t instruction)
{
    const uint32_t rt = instruction & 0x1FU;
    const uint64_t frequency
        = 0U == cpu->counter_frequency ? BAL_DEFAULT_COUNTER_FREQUENCY : cpu->counter_frequency;
    uint64_t value = frequency;

    // op2 selects CNTFRQ_EL0 (0), CNTPCT_EL0 (1) or CNTVCT_EL0 (2); the virtual counter has no
    // offset at EL0.
    if (((instruction >> 5U) & 0x7U) != 0U)
    {
        value = NULL != cpu->read_counter ? cpu->read_counter(cpu->counter_context)
                                          : read_host_counter(frequency);
    }

    if (rt != 31U)
    {
        cpu->x[rt] = value;
    }

    return 1U;
}

/// FEAT_MOPS CPYP, CPYM, CPYE and their CPYF and option forms:
/// 00 011 F 01 op1 0 Rs op2 01 Rn Rd, with op1 00 (prologue), 01 (main) or 10 (epilogue).
static bool
is_memory_copy(const uint32_t instruction)
{
    return 0x19000400U == (instruction & 0xFB200C00U) && ((instruction >> 22U) & 0x3U) != 3U;
}

/// FEAT_MOPS SETP, SETM, SETE and their option forms: 00 011 G 01 11 0 Rs op2 01 Rn Rd, with
/// op2<3:2> 00 (prologue), 01 (main) or 10 (epilogue); G = 1 for the FEAT_MTE SETGP, SETGM and
/// SETGE, which also set the allocation tags.
static bool
is_memory_set(const uint32_t instruction)
{
    return 0x19C00400U == (instruction & 0xFBE00C00U) && ((instruction >> 14U) & 0x3U) != 3U;
}

/// FEAT_MTE LDG: 11011001 011 imm9 00 Xn Xt.
static bool
is_tag_load(const uint32_t instruction)
{
    return 0xD9600000U == (instruction & 0xFFE00C00U);
}

/// FEAT_MTE STG, STZG, ST2G and STZ2G: 11011001 two zero 1 imm9 op2 Xn Xt, with op2 01
/// (post-index), 10 (signed offset) or 11 (pre-index); op2 00 is LDG and the EL1 bulk forms.
static bool
is_tag_store(const uint32_t instruction)
{
    return 0xD9200000U == (instruction & 0xFF200000U) && 0U != (instruction & 0x00000C00U);
}

/// FEAT_MTE STGP: 0110100 op 0 simm7 Xt2 Xn Xt, with op 01 (post-index), 10 (signed offset) or 11
/// (pre-index).
static bool
is_tag_store_pair(const uint32_t instruction)
{
    return 0x68000000U == (instruction & 0xFE400000U) && 0U != (instruction & 0x01800000U);
}

bool
bal_is_memory_operation_supported(const uint32_t instruction)
{
    const uint32_t d = instruction & 0x1FU;
    const uint32_t n = (instruction >> 5U) & 0x1FU;
    const uint32_t s = (instruction >> 16U) & 0x1FU;

    if (true == is_tag_load(instruction) || true == is_tag_store(instruction)
        || true == is_tag_store_pair(instruction))
    {
        return true;
    }

    // Overlapping registers and the zero register as an address or size are constrained
    // unpredictable; they are undefined here. SET may take its data from XZR.
    const bool is_distinct = s != n && s != d && n != d && d != 31U && n != 31U;

    if (true == is_memory_copy(instruction))
    {
        return true == is_distinct && s != 31U;
    }

    return true == is_memory_set(instruction) && true == is_distinct;
}

/// Returns true if the `size` bytes from `address` are all mapped by `memory_interface`.
static bool
is_range_mapped(const bal_memory_interface_t *memory_interface, uint64_t address, uint64_t size)
{
    while (size > 0U)
    {
        size_t available_bytes = 0U;

        if (NULL == memory_interface->translate((void *)memory_interface, address, &available_bytes)
            || 0U == available_bytes)
        {
            return false;
        }

        const uint64_t step = available_bytes < size ? available_bytes : size;
        address += step;
        size -= step;
    }

    return true;
}

/// The host address of `size` bytes at `address` when they are contiguous in host memory, and
/// NULL otherwise.
static uint8_t *
contiguous(const bal_memory_interface_t *memory_interface,
           const uint64_t                address,
           const uint64_t                size)
{
    size_t         available_bytes = 0U;
    const uint8_t *host_address
        = memory_interface->translate((void *)memory_interface, address, &available_bytes);
    return NULL != host_address && available_bytes >= size ? (uint8_t *)(uintptr_t)host_address
                                                           : NULL;
}

static uint8_t *
byte_at(const bal_memory_interface_t *memory_interface, const uint64_t address)
{
    return contiguous(memory_interface, address, 1U);
}

/// Copies `size` bytes from `from` to `to` as memmove() does. Copying the whole range as one
/// block is a valid choice of the IMPLEMENTATION DEFINED block size, and gives the same result
/// as copying in the direction the prologue picked. Returns false, copying nothing, if either
/// range is not mapped.
static bool
copy_memory(const bal_memory_interface_t *memory_interface,
            const uint64_t                to,
            const uint64_t                from,
            const uint64_t                size)
{
    if (0U == size)
    {
        return true;
    }

    if (false == is_range_mapped(memory_interface, from, size)
        || false == is_range_mapped(memory_interface, to, size))
    {
        return false;
    }

    uint8_t *destination = contiguous(memory_interface, to, size);
    uint8_t *source      = contiguous(memory_interface, from, size);

    if (NULL != destination && NULL != source)
    {
        (void)memmove(destination, source, size);
        return true;
    }

    // Byte by byte, in the direction that does not overwrite source bytes before reading them.
    for (uint64_t i = 0U; i < size; ++i)
    {
        const uint64_t offset = to > from ? size - 1U - i : i;
        uint8_t       *target = byte_at(memory_interface, to + offset);
        *target               = *byte_at(memory_interface, from + offset);
    }

    return true;
}

/// Fills `size` bytes from `to` with `data`. Returns false, writing nothing, if they are not
/// mapped.
static bool
set_memory(const bal_memory_interface_t *memory_interface,
           const uint64_t                to,
           const uint64_t                size,
           const uint8_t                 data)
{
    if (false == is_range_mapped(memory_interface, to, size))
    {
        return false;
    }

    uint64_t address   = to;
    uint64_t remaining = size;

    while (remaining > 0U)
    {
        size_t   available_bytes = 0U;
        uint8_t *host            = (uint8_t *)(uintptr_t)memory_interface->translate(
            (void *)memory_interface, address, &available_bytes);
        const uint64_t step = available_bytes < remaining ? available_bytes : remaining;
        (void)memset(host, data, step);
        address += step;
        remaining -= step;
    }

    return true;
}

/// LDG, STG, STZG, ST2G, STZ2G and STGP with allocation tag access disabled: LDG reads tag 0 and
/// the tag stores write no tag, but the addresses are still checked, STZG and STZ2G still zero
/// their granules and STGP still stores its registers. Returns zero, changing nothing, for an
/// unmapped address or an unaligned tag store.
static uint64_t
execute_tag_memory(bal_cpu_t                    *cpu,
                   const bal_memory_interface_t *memory_interface,
                   const uint32_t                instruction)
{
    const uint32_t rt      = instruction & 0x1FU;
    const uint32_t rn      = (instruction >> 5U) & 0x1FU;
    const uint64_t base    = cpu->x[rn];
    const bool     is_pair = is_tag_store_pair(instruction);

    // Both offsets count tag granules of 16 bytes.
    const int64_t offset
        = true == is_pair ? (int64_t)((int32_t)(((instruction >> 15U) & 0x7FU) << 25U) >> 25) * 16
                          : (int64_t)((int32_t)(((instruction >> 12U) & 0x1FFU) << 23U) >> 23) * 16;

    if (true == is_tag_load(instruction))
    {
        const uint64_t address = (base + (uint64_t)offset) & ~0xFULL;

        if (false == is_range_mapped(memory_interface, address, 16U))
        {
            return 0U;
        }

        if (rt != 31U)
        {
            cpu->x[rt] &= ~(0xFULL << 56U);
        }

        return 1U;
    }

    // op 01 is post-index and 11 pre-index, both with writeback; 10 is a signed offset.
    const uint32_t mode
        = true == is_pair ? (instruction >> 23U) & 0x3U : (instruction >> 10U) & 0x3U;
    const uint64_t address = 1U == mode ? base : base + (uint64_t)offset;
    // ST2G and STZ2G cover two granules; STGP stores 16 bytes of data in one.
    const bool     is_double = false == is_pair && 0U != (instruction & 0x00800000U);
    const uint64_t bytes     = true == is_double ? 32U : 16U;

    if (0U != (address & 0xFU) || false == is_range_mapped(memory_interface, address, bytes))
    {
        return 0U;
    }

    if (true == is_pair)
    {
        const uint64_t data[2]
            = { 31U == rt ? 0U : cpu->x[rt],
                31U == ((instruction >> 10U) & 0x1FU) ? 0U : cpu->x[(instruction >> 10U) & 0x1FU] };
        uint8_t *host = contiguous(memory_interface, address, 16U);

        if (NULL == host)
        {
            for (uint32_t i = 0U; i < 16U; ++i)
            {
                *byte_at(memory_interface, address + i)
                    = (uint8_t)(data[i / 8U] >> (8U * (i % 8U)));
            }
        }
        else
        {
            (void)memcpy(host, data, sizeof(data));
        }
    }
    else if (0U != (instruction & 0x00400000U))
    {
        (void)set_memory(memory_interface, address, bytes, 0U);
    }

    if (mode != 2U)
    {
        cpu->x[rn] = 1U == mode ? base + (uint64_t)offset : address;
    }

    return 1U;
}

static void
write_nzcv(bal_cpu_t *cpu, const uint32_t nzcv)
{
    cpu->flag_n = (uint8_t)((nzcv >> 3U) & 1U);
    cpu->flag_z = (uint8_t)((nzcv >> 2U) & 1U);
    cpu->flag_c = (uint8_t)((nzcv >> 1U) & 1U);
    cpu->flag_v = (uint8_t)(nzcv & 1U);
}

uint64_t
bal_execute_memory_operation(bal_cpu_t                    *cpu,
                             const bal_memory_interface_t *memory_interface,
                             const uint32_t                instruction)
{
    if (true == is_tag_load(instruction) || true == is_tag_store(instruction)
        || true == is_tag_store_pair(instruction))
    {
        return execute_tag_memory(cpu, memory_interface, instruction);
    }

    const uint32_t d       = instruction & 0x1FU;
    const uint32_t n       = (instruction >> 5U) & 0x1FU;
    const uint32_t s       = (instruction >> 16U) & 0x1FU;
    const bool     is_copy = is_memory_copy(instruction);
    const uint32_t stage
        = true == is_copy ? (instruction >> 22U) & 0x3U : (instruction >> 14U) & 0x3U;
    uint64_t to   = cpu->x[d];
    uint64_t size = cpu->x[n];

    // This implementation uses option B, which the prologue reports with C set: the registers
    // keep the start of the range, and N gives the direction of a copy. The prologue does the
    // whole operation, leaving nothing to the main and epilogue instructions; they finish an
    // operation that was interrupted between the instructions of a sequence.
    if (0U != stage && 0U != size && 0U == cpu->flag_c)
    {
        // Option A parameters, from another implementation: a MismatchedMemCpyException or
        // MismatchedMemSetException, reported as a data abort.
        return 0U;
    }

    if (false == is_copy)
    {
        const bool is_tagged = 0U != (instruction & 0x04000000U);

        if (0U == stage && 0U != (size >> 63U))
        {
            size = true == is_tagged ? 0x7FFFFFFFFFFFFFF0ULL : 0x7FFFFFFFFFFFFFFFULL;
        }

        // SETG works on whole tag granules: an unaligned address or size is an alignment fault.
        if (true == is_tagged && (0U != (size & 0xFU) || (0U != size && 0U != (to & 0xFU))))
        {
            return 0U;
        }

        const uint8_t data = (uint8_t)(31U == s ? 0U : cpu->x[s]);

        if (false == set_memory(memory_interface, to, size, data))
        {
            return 0U;
        }

        cpu->x[d] = to + size;
        cpu->x[n] = 0U;

        if (0U == stage)
        {
            write_nzcv(cpu, 0x2U);
        }

        return 1U;
    }

    uint64_t from       = cpu->x[s];
    bool     is_forward = 0U == cpu->flag_n;

    if (0U == stage)
    {
        if (0U != (size >> 55U))
        {
            size = 0x007FFFFFFFFFFFFFULL;
        }

        // CPYF copies forwards. CPY copies backwards when the source starts below an
        // overlapping destination, as memmove() does; the 56-bit addresses decide.
        const uint64_t mask        = 0x00FFFFFFFFFFFFFFULL;
        const bool     is_cpyf     = 0U == (instruction & 0x04000000U);
        const uint64_t source      = from & mask;
        const uint64_t destination = to & mask;
        is_forward
            = true == is_cpyf || false == (source < destination && source + size > destination);

        if (false == is_forward)
        {
            to += size;
            from += size;
        }
    }

    const uint64_t start_to   = true == is_forward ? to : to - size;
    const uint64_t start_from = true == is_forward ? from : from - size;

    // A fault leaves the registers and flags as they were before the instruction.
    if (false == copy_memory(memory_interface, start_to, start_from, size))
    {
        return 0U;
    }

    if (0U == stage)
    {
        write_nzcv(cpu, true == is_forward ? 0x2U : 0xAU);
    }

    cpu->x[d] = true == is_forward ? to + size : to - size;
    cpu->x[s] = true == is_forward ? from + size : from - size;
    cpu->x[n] = 0U;
    return 1U;
}

uint64_t
bal_execute_cache_zero(bal_cpu_t                    *cpu,
                       const bal_memory_interface_t *memory_interface,
                       const uint32_t                instruction)
{
    const uint64_t address
        = read_register(cpu, instruction & 0x1FU) & ~(uint64_t)(BAL_CACHE_ZERO_BLOCK_BYTES - 1U);
    size_t         available_bytes = 0U;
    const uint8_t *host_address
        = memory_interface->translate((void *)memory_interface, address, &available_bytes);

    if (NULL == host_address || available_bytes < BAL_CACHE_ZERO_BLOCK_BYTES)
    {
        return 0U;
    }

    (void)memset((uint8_t *)(uintptr_t)host_address, 0, BAL_CACHE_ZERO_BLOCK_BYTES);
    return 1U;
}

/*** end of file ***/
