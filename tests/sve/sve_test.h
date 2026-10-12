//! The record format shared by the SVE reference (sve_reference.c, AArch64 under qemu) and the
//! Ballistic side (sve_diff.c).
#ifndef SVE_TEST_H
#define SVE_TEST_H

#include <stdint.h>

#define SVE_DATA_BASE 0x200000ULL
#define SVE_DATA_SIZE 0x4000ULL

typedef struct
{
    uint64_t x[31];
    uint64_t sp;
    uint64_t nzcv;
    uint64_t fpcr;
    uint64_t fpsr;
    uint64_t v[32][2];
    uint16_t p[16];
    uint16_t ffr;
    uint16_t pad[3];
    /// The FNV-1a hash of the data area after the instruction.
    uint64_t data_hash;
    /// Zero, or the signal the instruction raised (4 SIGILL, 7 SIGBUS, 11 SIGSEGV).
    uint64_t status;
    /// The register the reference keeps its context pointer in, which the instruction never
    /// names; it is not compared.
    uint64_t base_register;
} sve_state_t;

typedef struct
{
    uint32_t    instruction;
    uint32_t    pad;
    uint64_t    seed;
    sve_state_t state;
} sve_test_t;

static inline uint64_t
sve_splitmix(uint64_t *state)
{
    uint64_t value = (*state += 0x9E3779B97F4A7C15ULL);
    value          = (value ^ (value >> 30U)) * 0xBF58476D1CE4E5B9ULL;
    value          = (value ^ (value >> 27U)) * 0x94D049BB133111EBULL;
    return value ^ (value >> 31U);
}

static inline void
sve_fill_data(uint8_t *data, uint64_t seed)
{
    for (uint64_t i = 0U; i < SVE_DATA_SIZE; i += 8U)
    {
        uint64_t value = sve_splitmix(&seed);

        for (uint64_t b = 0U; b < 8U; ++b)
        {
            data[i + b] = (uint8_t)(value >> (8U * b));
        }
    }
}

static inline uint64_t
sve_hash_data(const uint8_t *data)
{
    uint64_t hash = 0xCBF29CE484222325ULL;

    for (uint64_t i = 0U; i < SVE_DATA_SIZE; ++i)
    {
        hash = (hash ^ data[i]) * 0x100000001B3ULL;
    }

    return hash;
}

/// The general register the reference may use as its base: one the instruction's register
/// fields (bits 4:0, 9:5, 14:10 and 20:16) do not name.
static inline uint32_t
sve_base_register(const uint32_t instruction)
{
    for (uint32_t r = 30U; r > 0U; --r)
    {
        if (r != (instruction & 0x1FU) && r != ((instruction >> 5U) & 0x1FU)
            && r != ((instruction >> 10U) & 0x1FU) && r != ((instruction >> 16U) & 0x1FU))
        {
            return r;
        }
    }

    return 0U;
}

#endif
