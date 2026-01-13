#pragma once

#include "types.h"
#include "config.h"
#include "device.h"
#include <memory>
#include <cstdlib>
#include <new>
#include <atomic>
#include <map>

namespace psi {
    namespace core {

        // Memory alignment utilities
        PSI_FORCE_INLINE usize align_size(usize size, usize alignment) {
            return (size + alignment - 1) & ~(alignment - 1);
        }

        PSI_FORCE_INLINE bool is_aligned(const void* ptr, usize alignment) {
            return (reinterpret_cast<uintptr_t>(ptr) % alignment) == 0;
        }

        // Memory allocation statistics
        struct MemoryStats {
            std::atomic<memory_size_t> total_allocated{ 0 };
            std::atomic<memory_size_t> peak_allocated{ 0 };
            std::atomic<u64> allocation_count{ 0 };
            std::atomic<u64> deallocation_count{ 0 };

            MemoryStats() = default;

            void record_allocation(memory_size_t size) {
                total_allocated += size;
                allocation_count++;

                memory_size_t current = total_allocated.load();
                memory_size_t expected = peak_allocated.load();
                while (current > expected &&
                    !peak_allocated.compare_exchange_weak(expected, current)) {
                    // Keep trying until we successfully update peak or current is no longer > expected
                }
            }

            void record_deallocation(memory_size_t size) {
                total_allocated -= size;
                deallocation_count++;
            }

            void reset() {
                total_allocated = 0;
                peak_allocated = 0;
                allocation_count = 0;
                deallocation_count = 0;
            }

            MemoryStats(const MemoryStats& other)
                : total_allocated(other.total_allocated.load())
                , peak_allocated(other.peak_allocated.load())
                , allocation_count(other.allocation_count.load())
                , deallocation_count(other.deallocation_count.load()) {}

            MemoryStats& operator=(const MemoryStats& other) {
                if (this != &other) {
                    total_allocated.store(other.total_allocated.load());
                    peak_allocated.store(other.peak_allocated.load());
                    allocation_count.store(other.allocation_count.load());
                    deallocation_count.store(other.deallocation_count.load());
                }
                return *this;
            }
        };

        // Memory allocator interface
        class Allocator {
        public:
            virtual ~Allocator() = default;

            virtual void* allocate(memory_size_t size, usize alignment = Config::memory_alignment) = 0;
            virtual void deallocate(void* ptr, memory_size_t size) = 0;

            virtual memory_size_t get_allocated_size() const = 0;
            virtual MemoryStats get_stats() const = 0;
            virtual void reset_stats() = 0;

            // Device association
            virtual device_id_t get_device_id() const = 0;
            virtual DeviceType get_device_type() const = 0;

            // Convenience methods
            template<typename T>
            T* allocate(usize count) {
                return static_cast<T*>(allocate(count * sizeof(T), alignof(T)));
            }

            template<typename T>
            void deallocate(T* ptr, usize count) {
                deallocate(static_cast<void*>(ptr), count * sizeof(T));
            }
        };

        // System allocator (uses malloc/free with alignment)
        class SystemAllocator : public Allocator {
        public:
            explicit SystemAllocator(device_id_t device_id = 0);
            ~SystemAllocator() override = default;

            void* allocate(memory_size_t size, usize alignment = Config::memory_alignment) override;
            void deallocate(void* ptr, memory_size_t size) override;

            memory_size_t get_allocated_size() const override {
                return stats_.total_allocated.load();
            }

            MemoryStats get_stats() const override { return stats_; }
            void reset_stats() override { stats_.reset(); }

            device_id_t get_device_id() const override { return device_id_; }
            DeviceType get_device_type() const override { return DeviceType::CPU; }

        private:
            device_id_t device_id_;
            mutable MemoryStats stats_;

            void* aligned_alloc(memory_size_t size, usize alignment);
            void aligned_free(void* ptr);
        };

        // Pool allocator for frequent small allocations
        class PoolAllocator : public Allocator {
        public:
            explicit PoolAllocator(memory_size_t block_size, usize blocks_per_chunk = 1024,
                device_id_t device_id = 0);
            ~PoolAllocator() override;

            void* allocate(memory_size_t size, usize alignment = Config::memory_alignment) override;
            void deallocate(void* ptr, memory_size_t size) override;

            memory_size_t get_allocated_size() const override;
            MemoryStats get_stats() const override { return stats_; }
            void reset_stats() override { stats_.reset(); }

            device_id_t get_device_id() const override { return device_id_; }
            DeviceType get_device_type() const override { return DeviceType::CPU; }

            // Pool-specific methods
            memory_size_t get_block_size() const { return block_size_; }
            usize get_free_blocks() const;
            void clear();

        private:
            struct Block {
                Block* next;
            };

            struct Chunk {
                Chunk* next;
                void* memory;
                usize block_count;
            };

            memory_size_t block_size_;
            usize blocks_per_chunk_;
            device_id_t device_id_;

            Chunk* chunks_;
            Block* free_blocks_;
            mutable MemoryStats stats_;

            void allocate_new_chunk();
            void free_all_chunks();
        };

        // Memory manager singleton
        class MemoryManager {
        public:
            static MemoryManager& instance() {
                static MemoryManager manager;
                return manager;
            }

            // Get allocator for device
            PSI_NODISCARD Allocator* get_allocator(device_id_t device_id = -1) const;

            // Set custom allocator for device
            void set_allocator(device_id_t device_id, std::unique_ptr<Allocator> allocator);

            // Global allocation functions
            void* allocate(memory_size_t size, usize alignment = Config::memory_alignment,
                device_id_t device_id = -1);
            void deallocate(void* ptr, memory_size_t size, device_id_t device_id = -1);

            // Template allocation functions
            template<typename T>
            T* allocate(usize count, device_id_t device_id = -1) {
                return static_cast<T*>(allocate(count * sizeof(T), alignof(T), device_id));
            }

            template<typename T>
            void deallocate(T* ptr, usize count, device_id_t device_id = -1) {
                deallocate(static_cast<void*>(ptr), count * sizeof(T), device_id);
            }

            // Memory statistics
            MemoryStats get_global_stats() const;
            MemoryStats get_device_stats(device_id_t device_id) const;
            void reset_stats();
            void print_stats() const;

            // Configuration
            void enable_tracking(bool enable = true) { tracking_enabled_ = enable; }
            PSI_NODISCARD bool is_tracking_enabled() const { return tracking_enabled_; }

        private:
            MemoryManager();
            ~MemoryManager() = default;
            MemoryManager(const MemoryManager&) = delete;
            MemoryManager& operator=(const MemoryManager&) = delete;

            mutable std::map<device_id_t, std::unique_ptr<Allocator>> allocators_;
            bool tracking_enabled_;

            void initialize_default_allocators();
        };

        // Global convenience functions
        PSI_NODISCARD inline void* allocate(memory_size_t size,
            usize alignment = Config::memory_alignment,
            device_id_t device_id = -1) {
            return MemoryManager::instance().allocate(size, alignment, device_id);
        }

        inline void deallocate(void* ptr, memory_size_t size, device_id_t device_id = -1) {
            MemoryManager::instance().deallocate(ptr, size, device_id);
        }

        template<typename T>
        PSI_NODISCARD T* allocate(usize count, device_id_t device_id = -1) {
            return MemoryManager::instance().allocate<T>(count, device_id);
        }

        template<typename T>
        void deallocate(T* ptr, usize count, device_id_t device_id = -1) {
            MemoryManager::instance().deallocate<T>(ptr, count, device_id);
        }

        // RAII memory wrapper
        template<typename T>
        class Memory {
        public:
            Memory() : ptr_(nullptr), size_(0), device_id_(-1) {}

            explicit Memory(usize count, device_id_t device_id = -1)
                : ptr_(allocate<T>(count, device_id))
                , size_(count)
                , device_id_(device_id) {
            }

            ~Memory() {
                if (ptr_) {
                    deallocate<T>(ptr_, size_, device_id_);
                }
            }

            // Move semantics
            Memory(Memory&& other) noexcept
                : ptr_(other.ptr_)
                , size_(other.size_)
                , device_id_(other.device_id_) {
                other.ptr_ = nullptr;
                other.size_ = 0;
            }

            Memory& operator=(Memory&& other) noexcept {
                if (this != &other) {
                    if (ptr_) {
                        deallocate<T>(ptr_, size_, device_id_);
                    }
                    ptr_ = other.ptr_;
                    size_ = other.size_;
                    device_id_ = other.device_id_;
                    other.ptr_ = nullptr;
                    other.size_ = 0;
                }
                return *this;
            }

            // Delete copy semantics
            Memory(const Memory&) = delete;
            Memory& operator=(const Memory&) = delete;

            // Access
            PSI_NODISCARD T* get() const { return ptr_; }
            PSI_NODISCARD T* data() const { return ptr_; }
            PSI_NODISCARD usize size() const { return size_; }
            PSI_NODISCARD device_id_t device_id() const { return device_id_; }

            // Operators
            T& operator[](usize index) { return ptr_[index]; }
            const T& operator[](usize index) const { return ptr_[index]; }

            explicit operator bool() const { return ptr_ != nullptr; }

            // Release ownership
            T* release() {
                T* result = ptr_;
                ptr_ = nullptr;
                size_ = 0;
                return result;
            }

            // Reset
            void reset(usize count = 0, device_id_t device_id = -1) {
                if (ptr_) {
                    deallocate<T>(ptr_, size_, device_id_);
                }
                if (count > 0) {
                    ptr_ = allocate<T>(count, device_id);
                    size_ = count;
                    device_id_ = device_id;
                }
                else {
                    ptr_ = nullptr;
                    size_ = 0;
                    device_id_ = -1;
                }
            }

        private:
            T* ptr_;
            usize size_;
            device_id_t device_id_;
        };

        // Memory alignment macros
#define PSI_ALIGNED(alignment) alignas(alignment)
#define PSI_CACHE_ALIGNED PSI_ALIGNED(PSI_CACHE_LINE_SIZE)

    } // namespace core
} // namespace psi