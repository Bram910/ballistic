#include "backend/bal_system.h"

#include <string.h>

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

bool
bal_is_integer_helper_supported(const uint32_t instruction)
{
    return is_crc32(instruction) || is_cfinv(instruction) || is_rmif(instruction)
           || is_setf(instruction);
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
