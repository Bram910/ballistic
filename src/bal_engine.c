#include "bal_engine.h"
#include "backend/x86/bal_x86_tier1_compiler.h"
#include "bal_decoder.h"
#include "bal_engine_flags.h"
#include "bal_log.h"
#include "bal_platform.h"
#include <stdatomic.h>
#include <stdbool.h>
#include <stddef.h>
#include <string.h>

#define MAX_INSTRUCTIONS 65535

// Not sure what exact value to put here.
//
#define MAX_GUEST_REGISTERS 128

/// The size of an ARM Instruction in bytes.
#define ARM_INSTRUCTION_SIZE_BYTES 4

/// Helper macro to align `x` UP to the nearest memory alignment.
#define BAL_ALIGN_UP(x, memory_alignment) \
    (((x) + ((memory_alignment) - 1)) & ~((memory_alignment) - 1))

#define MEMORY_ALIGNMENT        64U
#define TIER1_BUFFER_SIZE_BYTES (1024 * 1024 * 16) // 16 MiB

#define BLOCK_CACHE_SETS 4096 // 256 KB total footprint.
#define BLOCK_CACHE_WAYS 4
#define BLOCK_CACHE_MASK (BLOCK_CACHE_SETS - 1)

/// Represents a single entry (way) in the block cache.
typedef struct
{
    /// The GVA of the compiled block.
    bal_guest_address_t guest_address;

    /// Pointer to the JIT compiled host machine code. A NULL value indicates an empty/invalid
    /// cache slot.
    void *host_code;
} block_cache_entry_t;

static_assert(16 == sizeof(block_cache_entry_t), "Block cache entry size mismatch");

/// Represents a single set in the 4-way associative block cache.
typedef struct
{
    block_cache_entry_t ways[BLOCK_CACHE_WAYS];
} block_cache_set_t;

static_assert(64 == sizeof(block_cache_set_t), "Block cache set size mismatch");

typedef struct
{
    bal_tier1_compiler_t    tier1_compiler;
    bal_executable_buffer_t tier1_buffer;
    size_t                  tier1_buffer_size;
    char                    pad[40];
    bal_jit_debug_context_t debug_context;

    BAL_ALIGNED(64) struct
    {
        atomic_bool is_executing;
        atomic_bool stop_requested;
        atomic_bool clear_cache_requested;
        char        pad1[61];
    } thread_state;

    BAL_ALIGNED(64) block_cache_set_t block_cache[BLOCK_CACHE_SETS];

    /// The blocks indirect branches jump to directly; see bal_tier1_lookup_entry_t.
    BAL_ALIGNED(64) bal_tier1_lookup_entry_t lookup_table[BAL_TIER1_LOOKUP_ENTRIES];
} internal_engine_state_t;

// Disable address sanitizers when executing jit code because it introduces very subtle but
// dangerous bugs.
__attribute__((no_sanitize("address", "undefined"))) static void execute_jit_block(
    bal_jit_block_t block, bal_cpu_t *BAL_RESTRICT cpu);

/// Looks up a JIT compiled host block by its GVA. Returns a pointer to the host JIT code if
/// found, NULL on a cache miss.
static void *block_cache_lookup(const block_cache_set_t *BAL_RESTRICT cache,
                                const bal_guest_address_t             pc);

/// Inserts a newly compiled JIT block into cache.
static void block_cache_insert(block_cache_set_t *BAL_RESTRICT cache,
                               const bal_guest_address_t       pc,
                               void                           *host_code);

static void link_block(internal_engine_state_t *BAL_RESTRICT engine_state,
                       uint64_t                              link_site,
                       const void                           *entry_point);

BAL_COLD bal_error_t
bal_engine_init(bal_engine_t *BAL_RESTRICT                 engine,
                bal_cpu_t *BAL_RESTRICT                    cpu,
                const bal_allocator_t *BAL_RESTRICT        allocator,
                const bal_memory_interface_t *BAL_RESTRICT memory_interface)
{
    // Check every pointer that will be used in the Engine.
    // This is tedious, but I currently don't give a fuck. Telling the user exactly what's
    // wrong is way better than seeing a random segfault in your terminal then going on Github
    // asking me why Ballistic is crashing on your system.

    if (BAL_UNLIKELY(NULL == engine))
    {
        return BAL_ERROR_INVALID_ARGUMENT;
    }

    if (BAL_UNLIKELY(NULL == cpu))
    {
        BAL_LOG_ERROR(&bal_thread_logger, "Aborting function: CPU is NULL");
        engine->status = BAL_ERROR_INVALID_ARGUMENT;
        return engine->status;
    }

    if (BAL_UNLIKELY(NULL == allocator))
    {
        BAL_LOG_ERROR(&bal_thread_logger, "Aborting function: Allocator is NULL");
        engine->status = BAL_ERROR_INVALID_ARGUMENT;
        return engine->status;
    }

    if (BAL_UNLIKELY(NULL == allocator->allocate))
    {
        BAL_LOG_ERROR(&bal_thread_logger, "Aborting function: Allocator->Allocate() is NULL");
        engine->status = BAL_ERROR_INVALID_ARGUMENT;
        return engine->status;
    }

    if (BAL_UNLIKELY(NULL == allocator->free))
    {
        BAL_LOG_ERROR(&bal_thread_logger, "Aborting function: Allocator->Free() is NULL");
        engine->status = BAL_ERROR_INVALID_ARGUMENT;
        return engine->status;
    }

    if (BAL_UNLIKELY(NULL == allocator->allocate_executable))
    {
        BAL_LOG_ERROR(&bal_thread_logger,
                      "Aborting function: Allocator->Allocate_Executable() is NULL");
        engine->status = BAL_ERROR_INVALID_ARGUMENT;
        return engine->status;
    }

    if (BAL_UNLIKELY(NULL == allocator->protect_rw))
    {
        BAL_LOG_ERROR(&bal_thread_logger, "Aborting function: Allocator->Protect_RW() is NULL");
        engine->status = BAL_ERROR_INVALID_ARGUMENT;
        return engine->status;
    }

    if (BAL_UNLIKELY(NULL == allocator->protect_rx))
    {
        BAL_LOG_ERROR(&bal_thread_logger, "Aborting function: Allocator->Protect_RX() is NULL");
        engine->status = BAL_ERROR_INVALID_ARGUMENT;
        return engine->status;
    }

    if (BAL_UNLIKELY(NULL == allocator->free_executable))
    {
        BAL_LOG_ERROR(&bal_thread_logger,
                      "Aborting function: Allocator->Free_Executable function is NULL");
        engine->status = BAL_ERROR_INVALID_ARGUMENT;
        return engine->status;
    }

    if (BAL_UNLIKELY(NULL == memory_interface))
    {
        BAL_LOG_ERROR(&bal_thread_logger, "Aborting function: Memory_Interface is NULL");
        engine->status = BAL_ERROR_INVALID_ARGUMENT;
        return engine->status;
    }

    if (BAL_UNLIKELY(NULL == memory_interface->context))
    {
        BAL_LOG_ERROR(&bal_thread_logger, "Aborting function: Memory_Interface->Context is NULL");
        engine->status = BAL_ERROR_INVALID_ARGUMENT;
        return engine->status;
    }

    if (BAL_UNLIKELY(NULL == memory_interface->translate))
    {
        BAL_LOG_ERROR(&bal_thread_logger,
                      "Aborting function: Memory_Interface->Translate function is NULL");
        engine->status = BAL_ERROR_INVALID_ARGUMENT;
        return engine->status;
    }

    internal_engine_state_t *BAL_RESTRICT internal_engine_state
        = (internal_engine_state_t *)allocator->allocate(
            allocator->context, MEMORY_ALIGNMENT, sizeof(internal_engine_state_t));

    if (BAL_UNLIKELY(NULL == internal_engine_state))
    {
        BAL_LOG_ERROR(&bal_thread_logger,
                      "Aborting function: Failed to allocate Internal Engine State");
        engine->status = BAL_ERROR_ALLOCATION_FAILED;
        return engine->status;
    }

    (void)memset(internal_engine_state, 0, sizeof(internal_engine_state_t));

    internal_engine_state->tier1_buffer_size = TIER1_BUFFER_SIZE_BYTES;
    internal_engine_state->tier1_buffer      = allocator->allocate_executable(
        allocator->context, 4096, internal_engine_state->tier1_buffer_size);

    if (NULL == internal_engine_state->tier1_buffer.rw_pointer)
    {
        BAL_LOG_ERROR(&bal_thread_logger,
                      "Aborting function: Failed to allocate Internal Tier 1 Buffer");
        engine->status = BAL_ERROR_ALLOCATION_FAILED;
        return engine->status;
    }

#if BAL_PLATFORM_LINUX || BAL_PLATFORM_WINDOWS

    const bal_error_t debug_status
        = bal_jit_debug_init(allocator, &internal_engine_state->debug_context);

    if (BAL_SUCCESS == debug_status)
    {
        bal_jit_debug_register_signal_handler(&internal_engine_state->debug_context,
                                              internal_engine_state->tier1_buffer.rx_pointer,
                                              internal_engine_state->tier1_buffer_size);
        cpu->debug_context = &internal_engine_state->debug_context;
    }
    else
    {
        BAL_LOG_WARN(&bal_thread_logger,
                     "Failed to initialize JIT debug context, debugging disabled.");
        memset(&internal_engine_state->debug_context, 0, sizeof(bal_jit_debug_context_t));
    }

#endif // BAL_PLATFORM_LINUX || BAL_PLATFORM_WINDOWS

    const bal_error_t status = bal_tier1_compiler_init(&internal_engine_state->tier1_compiler,
                                                       internal_engine_state->tier1_buffer,
                                                       internal_engine_state->tier1_buffer_size,
                                                       &internal_engine_state->debug_context);

    if (status != BAL_SUCCESS)
    {
        BAL_LOG_ERROR(&bal_thread_logger, "Aborting function: Failed to initialize Tier1 Compiler");
        allocator->free(allocator->context,
                        &internal_engine_state->tier1_buffer,
                        sizeof(internal_engine_state_t));
        allocator->free(allocator->context, internal_engine_state, sizeof(internal_engine_state_t));
        engine->status = status;
        return engine->status;
    }

    atomic_init(&internal_engine_state->thread_state.is_executing, false);
    atomic_init(&internal_engine_state->thread_state.stop_requested, false);
    atomic_init(&internal_engine_state->thread_state.clear_cache_requested, false);

    engine->cpu              = cpu;
    engine->allocator        = allocator;
    engine->memory_interface = memory_interface;
    engine->engine_state     = internal_engine_state;
    engine->status           = BAL_SUCCESS;
    engine->flags            = 0;
    return engine->status;
}

bal_error_t
bal_engine_run_thread(bal_engine_t *engine)
{
    internal_engine_state_t *internal_engine_state
        = (internal_engine_state_t *)engine->engine_state;
    bool expected = false;
    if (!atomic_compare_exchange_strong_explicit(&internal_engine_state->thread_state.is_executing,
                                                 &expected,
                                                 false,
                                                 memory_order_acq_rel,
                                                 memory_order_acquire))
    {
        BAL_LOG_ERROR(&bal_thread_logger, "Aborting function: Engine is already running");
        return BAL_ERROR_ENGINE_ALREADY_RUNNING;
    }

    // The link jump of the exit the last block left through, which the next block is chained to.
    uint64_t link_site            = 0U;
    engine->cpu->jit_lookup_table = (uint64_t)(uintptr_t)internal_engine_state->lookup_table;

    while (true)
    {
        // The requests that made chained blocks return are looked at below.
        (void)__atomic_exchange_n(&engine->cpu->jit_exit_requested, 0U, __ATOMIC_ACQUIRE);

        if (BAL_UNLIKELY(atomic_load_explicit(&internal_engine_state->thread_state.stop_requested,
                                              memory_order_acquire)))
        {
            BAL_LOG_INFO(&bal_thread_logger, "Engine stop requested. Exiting execution loop.");
            break;
        }

        // An ISB after an IC IVAU makes the code the guest wrote visible to it, so the
        // translations of the old code go.
        if (BAL_UNLIKELY(0U != engine->cpu->instruction_barrier))
        {
            engine->cpu->instruction_barrier = 0U;

            if (0U != engine->cpu->instruction_cache_invalidated)
            {
                engine->cpu->instruction_cache_invalidated = 0U;
                atomic_store_explicit(&internal_engine_state->thread_state.clear_cache_requested,
                                      true,
                                      memory_order_release);
            }
        }

        if (BAL_UNLIKELY(atomic_load_explicit(
                &internal_engine_state->thread_state.clear_cache_requested, memory_order_acquire)))
        {
            BAL_LOG_INFO(&bal_thread_logger,
                         "Clear cache requested. Resetting JIT layout and block cache.");

            // Resetting clears the code buffer, which is executable-only between translations.
            engine->allocator->protect_rw(engine->allocator->context,
                                          internal_engine_state->tier1_buffer,
                                          internal_engine_state->tier1_buffer_size);
            bal_tier1_compiler_reset(&internal_engine_state->tier1_compiler);
            (void)memset(
                internal_engine_state->block_cache, 0, sizeof(internal_engine_state->block_cache));
            (void)memset(internal_engine_state->lookup_table,
                         0,
                         sizeof(internal_engine_state->lookup_table));
            link_site = 0U;
            atomic_store_explicit(&internal_engine_state->thread_state.clear_cache_requested,
                                  false,
                                  memory_order_release);
        }

        if (engine->flags & BAL_ENGINE_FLAG_INTERRUPT_PENDING)
        {
            BAL_LOG_INFO(&bal_thread_logger, "Interrupt pending, yielding to host.");
            engine->flags &= ~BAL_ENGINE_FLAG_INTERRUPT_PENDING;
            break;
        }

        // TODO: Implement Block linking
        const bal_guest_address_t pc = engine->cpu->pc;

        if (BAL_UNLIKELY(BAL_ENGINE_SENTINEL == pc))
        {
            BAL_LOG_INFO(&bal_thread_logger, "Safe exit triggered by sentinel PC value.");
            break;
        }

        void *BAL_RESTRICT entry_point = NULL;

        if (BAL_LIKELY(!(engine->flags & BAL_ENGINE_FLAG_DISABLE_BLOCK_CACHE)))
        {
            entry_point = block_cache_lookup(internal_engine_state->block_cache, pc);
        }

        if (NULL == entry_point)
        {
            if (engine->flags & BAL_ENGINE_FLAG_LOG_BLOCKS)
            {
                BAL_LOG_INFO(
                    &bal_thread_logger, "Fetching block at PC 0x%016llX", (unsigned long long)pc);
            }

            engine->allocator->protect_rw(engine->allocator->context,
                                          internal_engine_state->tier1_buffer,
                                          internal_engine_state->tier1_buffer_size);
            size_t max_instructions = MAX_INSTRUCTIONS;

            if (engine->flags & BAL_ENGINE_FLAG_SINGLE_STEP)
            {
                max_instructions = 1;
            }

            entry_point = bal_tier1_compiler_translate(&internal_engine_state->tier1_compiler,
                                                       engine->memory_interface,
                                                       pc,
                                                       max_instructions,
                                                       engine->flags);

            if (BAL_UNLIKELY(NULL == entry_point))
            {
                if (internal_engine_state->tier1_compiler.assembler.status
                    == BAL_ERROR_INSTRUCTION_OVERFLOW)
                {
                    BAL_LOG_INFO(&bal_thread_logger,
                                 "Tier 1 executable buffer full. Resetting JIT layout cache");
                    bal_tier1_compiler_reset(&internal_engine_state->tier1_compiler);
                    (void)memset(internal_engine_state->block_cache,
                                 0,
                                 sizeof(internal_engine_state->block_cache));
                    (void)memset(internal_engine_state->lookup_table,
                                 0,
                                 sizeof(internal_engine_state->lookup_table));
                    link_site = 0U;
                    entry_point
                        = bal_tier1_compiler_translate(&internal_engine_state->tier1_compiler,
                                                       engine->memory_interface,
                                                       pc,
                                                       max_instructions,
                                                       engine->flags);
                }

                if (BAL_UNLIKELY(NULL == entry_point))
                {
                    BAL_LOG_ERROR(&bal_thread_logger,
                                  "Aborting function: Failed to compile block at PC 0x%0llx",
                                  // WARNING: C standard guarantees unsigned long long is at least
                                  // 64 bits wide.
                                  (unsigned long long)pc);
                    engine->status = internal_engine_state->tier1_compiler.status;
                    break;
                }
            }

            if (BAL_UNLIKELY(NULL == entry_point))
            {
                BAL_LOG_ERROR(&bal_thread_logger,
                              "Aborting function: Failed to compile block at PC 0x%0llX",
                              (unsigned long long)pc);
                engine->status = internal_engine_state->tier1_compiler.status;
                break;
            }
        }

        // Chain the exit the last block left through to this block, so that it runs without
        // returning here from now on.
        if (0U != link_site)
        {
            engine->allocator->protect_rw(engine->allocator->context,
                                          internal_engine_state->tier1_buffer,
                                          internal_engine_state->tier1_buffer_size);
            link_block(internal_engine_state, link_site, entry_point);
            link_site = 0U;
        }

        engine->allocator->protect_rx(engine->allocator->context,
                                      internal_engine_state->tier1_buffer,
                                      internal_engine_state->tier1_buffer_size);

        if (BAL_LIKELY(!(engine->flags & BAL_ENGINE_FLAG_DISABLE_BLOCK_CACHE)))
        {
            block_cache_insert(internal_engine_state->block_cache, pc, entry_point);

            // Indirect branches to this PC can now jump straight in.
            bal_tier1_lookup_entry_t *BAL_RESTRICT lookup
                = &internal_engine_state
                       ->lookup_table[(pc >> 2U) & (BAL_TIER1_LOOKUP_ENTRIES - 1U)];
            lookup->guest_address = pc;
            lookup->host_code     = (uint8_t *)entry_point + BAL_TIER1_BLOCK_PROLOGUE_SIZE;
        }

        const bal_jit_block_t compiled_block = (bal_jit_block_t)entry_point;

        if (engine->flags & BAL_ENGINE_FLAG_LOG_BLOCKS)
        {
            BAL_LOG_INFO(&bal_thread_logger, "Executing JIT Block at %p", entry_point);
        }

        execute_jit_block(compiled_block, engine->cpu);
        link_site                  = engine->cpu->jit_link_site;
        engine->cpu->jit_link_site = 0U;

        // The block stopped on a guest exception; see bal_exception_t for what pc holds.
        if (BAL_UNLIKELY(engine->cpu->exception != BAL_EXCEPTION_NONE))
        {
            BAL_LOG_INFO(&bal_thread_logger,
                         "Guest exception %u at PC 0x%016llX",
                         engine->cpu->exception,
                         (unsigned long long)engine->cpu->pc);
            engine->status = BAL_EXCEPTION_DATA_ABORT == engine->cpu->exception
                                 ? BAL_ERROR_MEMORY_FAULT
                             : BAL_EXCEPTION_UNSUPPORTED_INSTRUCTION == engine->cpu->exception
                                 ? BAL_ERROR_UNKNOWN_INSTRUCTION
                                 : BAL_ERROR_GUEST_EXCEPTION;
            break;
        }

        if (engine->flags & BAL_ENGINE_FLAG_SINGLE_STEP)
        {
            break;
        }
    }

    atomic_store_explicit(
        &internal_engine_state->thread_state.is_executing, false, memory_order_release);
    return engine->status;
}

void
bal_engine_stop_thread(bal_engine_t *BAL_RESTRICT engine)
{
    if (BAL_UNLIKELY(NULL == engine))
    {
        return;
    }

    if (BAL_UNLIKELY(NULL == engine->engine_state))
    {
        BAL_LOG_ERROR(&bal_thread_logger, "Aborting function: engine state is NULL");
        engine->status = BAL_ERROR_INVALID_ARGUMENT;
        return;
    }

    internal_engine_state_t *BAL_RESTRICT engine_state
        = (internal_engine_state_t *)engine->engine_state;
    atomic_store_explicit(&engine_state->thread_state.stop_requested, true, memory_order_release);
    bal_engine_request_exit(engine);
}

bal_error_t
bal_engine_reset(bal_engine_t *BAL_RESTRICT engine)
{
    if (BAL_UNLIKELY(NULL == engine))
    {
        return BAL_ERROR_INVALID_ARGUMENT;
    }

    engine->status = BAL_SUCCESS;
    internal_engine_state_t *BAL_RESTRICT engine_state
        = (internal_engine_state_t *)engine->engine_state;
    (void)memset(&engine_state->tier1_buffer, 0, engine_state->tier1_buffer_size);
    bal_tier1_compiler_reset(&engine_state->tier1_compiler);
    return engine->status;
}

void
bal_engine_destroy(bal_engine_t *engine)
{
    if (BAL_UNLIKELY(NULL == engine))
    {
        return;
    }

    if (NULL == engine->engine_state)
    {
        // Already destroyed or never fully initialized.
        memset(engine, 0, sizeof(bal_engine_t));
        return;
    }

    internal_engine_state_t *BAL_RESTRICT internal_engine_state
        = (internal_engine_state_t *)engine->engine_state;

    const bal_allocator_t *BAL_RESTRICT allocator = engine->allocator;

    if (NULL != allocator)
    {
        if (NULL != allocator->free_executable
            && NULL != internal_engine_state->tier1_buffer.rw_pointer)
        {
            BAL_LOG_INFO(&bal_thread_logger, "Freeing Tier 1 executable buffer.");
            allocator->free_executable(allocator->context,
                                       internal_engine_state->tier1_buffer,
                                       internal_engine_state->tier1_buffer_size);
        }

        if (NULL != allocator->free)
        {
            BAL_LOG_INFO(&bal_thread_logger, "Freeing internal engine state.");

#if BAL_PLATFORM_LINUX || BAL_PLATFORM_WINDOWS

            bal_jit_debug_unregister_signal_handler(&internal_engine_state->debug_context);

#endif // BAL_PLATFORM_LINUX || BAL_PLATFORM_WINDOWS

            bal_jit_debug_destroy(allocator, &internal_engine_state->debug_context);
            allocator->free(
                allocator->context, internal_engine_state, sizeof(internal_engine_state_t));
        }
    }
    else
    {
        BAL_LOG_WARN(&bal_thread_logger,
                     "Allocator is NULL during engine destroy. Internal memory leaked.");
    }

    memset(engine, 0, sizeof(bal_engine_t));
}

bool
bal_engine_is_running(bal_engine_t *engine)
{
    if (BAL_UNLIKELY(NULL == engine))
    {
        return false;
    }

    if (BAL_UNLIKELY(NULL == engine->engine_state))
    {
        BAL_LOG_ERROR(&bal_thread_logger, "Aborting function: engine state is NULL");
        engine->status = BAL_ERROR_INVALID_ARGUMENT;
        return false;
    }

    internal_engine_state_t *BAL_RESTRICT engine_state
        = (internal_engine_state_t *)engine->engine_state;
    return atomic_load_explicit(&engine_state->thread_state.is_executing, memory_order_acquire);
}

void
bal_engine_clear_cache(bal_engine_t *engine)
{
    if (BAL_UNLIKELY(NULL == engine))
    {
        return;
    }

    if (BAL_UNLIKELY(NULL == engine->engine_state))
    {
        BAL_LOG_ERROR(&bal_thread_logger, "Aborting function: engine state is NULL");
        engine->status = BAL_ERROR_INVALID_ARGUMENT;
        return;
    }

    internal_engine_state_t *BAL_RESTRICT engine_state
        = (internal_engine_state_t *)engine->engine_state;
    atomic_store_explicit(
        &engine_state->thread_state.clear_cache_requested, true, memory_order_release);
    bal_engine_request_exit(engine);
}

void
bal_engine_request_exit(bal_engine_t *engine)
{
    if (BAL_UNLIKELY(NULL == engine) || BAL_UNLIKELY(NULL == engine->cpu))
    {
        return;
    }

    __atomic_store_n(&engine->cpu->jit_exit_requested, 1U, __ATOMIC_RELEASE);
}

/// Points the jump at `link_site`, the host address of an exit's link jump, at the block whose
/// code starts at `entry_point`, past its prologue. Both are in the code buffer's executable view;
/// the write goes through its writable one.
static void
link_block(internal_engine_state_t *BAL_RESTRICT engine_state,
           const uint64_t                        link_site,
           const void                           *entry_point)
{
    uint8_t *const rx_base = (uint8_t *)engine_state->tier1_buffer.rx_pointer;
    uint8_t *const rw_base = (uint8_t *)engine_state->tier1_buffer.rw_pointer;
    uint8_t *const site    = (uint8_t *)(uintptr_t)link_site;
    const uint8_t *target  = (const uint8_t *)entry_point + BAL_TIER1_BLOCK_PROLOGUE_SIZE;

    // JMP rel32: E9 and a displacement from the end of the five-byte instruction.
    const int32_t displacement = (int32_t)(target - (site + 5));
    uint8_t      *field        = rw_base + (site - rx_base) + 1;
    (void)memcpy(field, &displacement, sizeof(displacement));
}

static void
execute_jit_block(const bal_jit_block_t block, bal_cpu_t *cpu)
{
    block(cpu);
}

BAL_HOT void *
block_cache_lookup(const block_cache_set_t *BAL_RESTRICT cache, const bal_guest_address_t pc)
{
    const uint32_t                          set_index = (uint32_t)(pc >> 2) & BLOCK_CACHE_MASK;
    const block_cache_entry_t *BAL_RESTRICT ways      = cache[set_index].ways;

    for (uint32_t i = 0; i < BLOCK_CACHE_WAYS; ++i)
    {
        if (ways[i].guest_address == pc && ways[i].host_code != NULL)
        {
            return ways[i].host_code;
        }
    }

    return NULL;
}

BAL_HOT void
block_cache_insert(block_cache_set_t *BAL_RESTRICT cache,
                   const bal_guest_address_t       pc,
                   void *BAL_RESTRICT              host_code)
{
    const uint32_t                    set_index = (uint32_t)(pc >> 2) & BLOCK_CACHE_MASK;
    block_cache_entry_t *BAL_RESTRICT ways      = cache[set_index].ways;

    for (uint32_t i = 0; i < BLOCK_CACHE_WAYS; ++i)
    {
        if (ways[i].guest_address == pc)
        {
            ways[i].host_code = host_code;
            return;
        }
    }

    for (uint32_t i = 0; i < BLOCK_CACHE_WAYS; ++i)
    {
        if (NULL == ways[i].host_code)
        {
            ways[i].guest_address = pc;
            ways[i].host_code     = host_code;
            return;
        }
    }

    // All 4 ways are full. Use the lower bits of the PC as a cheap starting index.
    const uint32_t evict_way      = (pc >> 4) & (BLOCK_CACHE_WAYS - 1);
    ways[evict_way].guest_address = pc;
    ways[evict_way].host_code     = host_code;
}