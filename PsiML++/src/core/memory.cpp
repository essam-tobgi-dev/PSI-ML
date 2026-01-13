#include "../../include/core/memory.h"
#include "../../include/core/logging.h"
#include "../../include/core/device.h"
#include <cstdlib>
#include <cstring>
#include <new>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <malloc.h>
#elif defined(__APPLE__)
#include <sys/mman.h>
#include <mach/mach.h>
#elif defined(__linux__)
#include <sys/mman.h>
#include <unistd.h>
#endif
#include "../../include/core/exception.h"

namespace psi {
    namespace core {

        // SystemAllocator implementation
        SystemAllocator::SystemAllocator(device_id_t device_id) : device_id_(device_id) {
            PSI_DEBUG_NAMED("memory", "Created SystemAllocator for device " + std::to_string(device_id));
        }

        void* SystemAllocator::allocate(memory_size_t size, usize alignment) {
            if (size == 0) {
                return nullptr;
            }

            // Ensure alignment is a power of 2 and at least sizeof(void*)
            if (alignment == 0) {
                alignment = Config::memory_alignment;
            }

            if ((alignment & (alignment - 1)) != 0) {
                PSI_THROW_MEMORY("Alignment must be a power of 2");
            }

            if (alignment < sizeof(void*)) {
                alignment = sizeof(void*);
            }

            void* ptr = aligned_alloc(size, alignment);
            if (!ptr) {
                PSI_THROW_MEMORY("Failed to allocate " + std::to_string(size) + " bytes with alignment " + std::to_string(alignment));
            }

            // Zero-initialize the memory
            std::memset(ptr, 0, size);

            stats_.record_allocation(size);

            PSI_DEBUG_NAMED("memory", "Allocated " + std::to_string(size) + " bytes at " +
                std::to_string(reinterpret_cast<uintptr_t>(ptr)));

            return ptr;
        }

        void SystemAllocator::deallocate(void* ptr, memory_size_t size) {
            if (!ptr) {
                return;
            }

            aligned_free(ptr);
            stats_.record_deallocation(size);

            PSI_DEBUG_NAMED("memory", "Deallocated " + std::to_string(size) + " bytes at " +
                std::to_string(reinterpret_cast<uintptr_t>(ptr)));
        }

        void* SystemAllocator::aligned_alloc(memory_size_t size, usize alignment) {
#ifdef _WIN32
            // Use _aligned_malloc on Windows
            return _aligned_malloc(size, alignment);

#elif defined(__APPLE__) || defined(__linux__)
            // Use posix_memalign on POSIX systems
            void* ptr = nullptr;
            int result = posix_memalign(&ptr, alignment, size);
            if (result != 0) {
                return nullptr;
            }
            return ptr;

#else
            // Fallback: manual alignment
            if (alignment <= sizeof(void*)) {
                return std::malloc(size);
            }

            // Allocate extra space for alignment and metadata
            usize total_size = size + alignment + sizeof(void*);
            void* raw_ptr = std::malloc(total_size);
            if (!raw_ptr) {
                return nullptr;
            }

            // Calculate aligned address
            uintptr_t addr = reinterpret_cast<uintptr_t>(raw_ptr) + sizeof(void*);
            addr = align_size(addr, alignment);
            void* aligned_ptr = reinterpret_cast<void*>(addr);

            // Store original pointer for deallocation
            void** metadata = static_cast<void**>(aligned_ptr) - 1;
            *metadata = raw_ptr;

            return aligned_ptr;
#endif
        }

        void SystemAllocator::aligned_free(void* ptr) {
            if (!ptr) {
                return;
            }

#ifdef _WIN32
            _aligned_free(ptr);

#elif defined(__APPLE__) || defined(__linux__)
            std::free(ptr);

#else
            // Fallback: retrieve original pointer
            void** metadata = static_cast<void**>(ptr) - 1;
            std::free(*metadata);
#endif
        }

        // PoolAllocator implementation
        PoolAllocator::PoolAllocator(memory_size_t block_size, usize blocks_per_chunk, device_id_t device_id)
            : block_size_(align_size(std::max(block_size, sizeof(Block)), sizeof(void*)))
            , blocks_per_chunk_(blocks_per_chunk)
            , device_id_(device_id)
            , chunks_(nullptr)
            , free_blocks_(nullptr) {

            PSI_DEBUG_NAMED("memory", "Created PoolAllocator: block_size=" + std::to_string(block_size_) +
                ", blocks_per_chunk=" + std::to_string(blocks_per_chunk_));

            allocate_new_chunk();
        }

        PoolAllocator::~PoolAllocator() {
            free_all_chunks();
        }

        void* PoolAllocator::allocate(memory_size_t size, usize alignment) {
            if (size > block_size_) {
                PSI_THROW_MEMORY("Requested size " + std::to_string(size) +
                    " exceeds pool block size " + std::to_string(block_size_));
            }

            if (!free_blocks_) {
                allocate_new_chunk();
                if (!free_blocks_) {
                    PSI_THROW_MEMORY("Failed to allocate new chunk for pool allocator");
                }
            }

            Block* block = free_blocks_;
            free_blocks_ = free_blocks_->next;

            void* ptr = static_cast<void*>(block);
            std::memset(ptr, 0, block_size_);

            stats_.record_allocation(block_size_);

            PSI_DEBUG_NAMED("memory", "Pool allocated " + std::to_string(block_size_) + " bytes at " +
                std::to_string(reinterpret_cast<uintptr_t>(ptr)));

            return ptr;
        }

        void PoolAllocator::deallocate(void* ptr, memory_size_t size) {
            if (!ptr) {
                return;
            }

            Block* block = static_cast<Block*>(ptr);
            block->next = free_blocks_;
            free_blocks_ = block;

            stats_.record_deallocation(block_size_);

            PSI_DEBUG_NAMED("memory", "Pool deallocated " + std::to_string(block_size_) + " bytes at " +
                std::to_string(reinterpret_cast<uintptr_t>(ptr)));
        }

        memory_size_t PoolAllocator::get_allocated_size() const {
            return stats_.total_allocated.load();
        }

        usize PoolAllocator::get_free_blocks() const {
            usize count = 0;
            Block* current = free_blocks_;
            while (current) {
                ++count;
                current = current->next;
            }
            return count;
        }

        void PoolAllocator::clear() {
            free_all_chunks();
            chunks_ = nullptr;
            free_blocks_ = nullptr;
            stats_.reset();
            allocate_new_chunk();
        }

        void PoolAllocator::allocate_new_chunk() {
            memory_size_t chunk_size = block_size_ * blocks_per_chunk_;

            // Allocate memory for chunk metadata and blocks
            memory_size_t total_size = sizeof(Chunk) + chunk_size;
            void* chunk_memory = std::malloc(total_size);
            if (!chunk_memory) {
                PSI_THROW_MEMORY("Failed to allocate pool chunk of size " + std::to_string(total_size));
            }

            // Initialize chunk
            Chunk* chunk = static_cast<Chunk*>(chunk_memory);
            chunk->next = chunks_;
            chunk->memory = static_cast<char*>(chunk_memory) + sizeof(Chunk);
            chunk->block_count = blocks_per_chunk_;
            chunks_ = chunk;

            // Initialize free list
            char* block_ptr = static_cast<char*>(chunk->memory);
            for (usize i = 0; i < blocks_per_chunk_; ++i) {
                Block* block = reinterpret_cast<Block*>(block_ptr);
                block->next = free_blocks_;
                free_blocks_ = block;
                block_ptr += block_size_;
            }

            PSI_DEBUG_NAMED("memory", "Allocated new pool chunk: " + std::to_string(blocks_per_chunk_) +
                " blocks of " + std::to_string(block_size_) + " bytes each");
        }

        void PoolAllocator::free_all_chunks() {
            while (chunks_) {
                Chunk* next = chunks_->next;
                std::free(chunks_);
                chunks_ = next;
            }
            free_blocks_ = nullptr;
        }

        // MemoryManager implementation
        MemoryManager::MemoryManager() : tracking_enabled_(true) {
            initialize_default_allocators();
            PSI_INFO_NAMED("memory", "Initialized MemoryManager");
        }

        Allocator* MemoryManager::get_allocator(device_id_t device_id) const {
            if (device_id == -1) {
                device_id = get_current_device()->get_id();
            }

            auto it = allocators_.find(device_id);
            if (it != allocators_.end()) {
                return it->second.get();
            }

            // Create allocator on demand
            auto allocator = std::make_unique<SystemAllocator>(device_id);
            Allocator* result = allocator.get();
            allocators_[device_id] = std::move(allocator);

            PSI_DEBUG_NAMED("memory", "Created on-demand allocator for device " + std::to_string(device_id));
            return result;
        }

        void MemoryManager::set_allocator(device_id_t device_id, std::unique_ptr<Allocator> allocator) {
            allocators_[device_id] = std::move(allocator);
            PSI_INFO_NAMED("memory", "Set custom allocator for device " + std::to_string(device_id));
        }

        void* MemoryManager::allocate(memory_size_t size, usize alignment, device_id_t device_id) {
            if (size == 0) {
                return nullptr;
            }

            Allocator* allocator = get_allocator(device_id);
            return allocator->allocate(size, alignment);
        }

        void MemoryManager::deallocate(void* ptr, memory_size_t size, device_id_t device_id) {
            if (!ptr) {
                return;
            }

            Allocator* allocator = get_allocator(device_id);
            allocator->deallocate(ptr, size);
        }

        MemoryStats MemoryManager::get_global_stats() const {
            MemoryStats global_stats;

            for (const auto& [device_id, allocator] : allocators_) {
                MemoryStats device_stats = allocator->get_stats();
                global_stats.total_allocated += device_stats.total_allocated.load();
                global_stats.peak_allocated += device_stats.peak_allocated.load();
                global_stats.allocation_count += device_stats.allocation_count.load();
                global_stats.deallocation_count += device_stats.deallocation_count.load();
            }

            return global_stats;
        }

        MemoryStats MemoryManager::get_device_stats(device_id_t device_id) const {
            Allocator* allocator = get_allocator(device_id);
            return allocator->get_stats();
        }

        void MemoryManager::reset_stats() {
            for (const auto& [device_id, allocator] : allocators_) {
                allocator->reset_stats();
            }
            PSI_INFO_NAMED("memory", "Reset memory statistics");
        }

        void MemoryManager::print_stats() const {
            MemoryStats global_stats = get_global_stats();

            std::ostringstream oss;
            oss << "Memory Statistics:\n";
            oss << "  Total Allocated: " << (global_stats.total_allocated.load() / 1024) << " KB\n";
            oss << "  Peak Allocated: " << (global_stats.peak_allocated.load() / 1024) << " KB\n";
            oss << "  Allocation Count: " << global_stats.allocation_count.load() << "\n";
            oss << "  Deallocation Count: " << global_stats.deallocation_count.load() << "\n";
            oss << "  Outstanding Allocations: " << (global_stats.allocation_count.load() -
                global_stats.deallocation_count.load()) << "\n";

            oss << "\nPer-Device Statistics:\n";
            for (const auto& [device_id, allocator] : allocators_) {
                MemoryStats device_stats = allocator->get_stats();
                oss << "  Device " << device_id << ":\n";
                oss << "    Allocated: " << (device_stats.total_allocated.load() / 1024) << " KB\n";
                oss << "    Peak: " << (device_stats.peak_allocated.load() / 1024) << " KB\n";
                oss << "    Allocations: " << device_stats.allocation_count.load() << "\n";
                oss << "    Deallocations: " << device_stats.deallocation_count.load() << "\n";
            }

            PSI_INFO_NAMED("memory", oss.str());
        }

        void MemoryManager::initialize_default_allocators() {
            // Create default system allocator for device 0 (CPU)
            auto system_allocator = std::make_unique<SystemAllocator>(0);
            allocators_[0] = std::move(system_allocator);

            PSI_DEBUG_NAMED("memory", "Initialized default system allocator for device 0");
        }

    } // namespace core
} // namespace psi