#include "../include/core/memory.h"
#include "../include/core/device.h"
#include <iostream>
#include <cassert>

using namespace psi::core;

void test_alignment_utilities() {
    std::cout << "Testing alignment utilities..." << std::endl;

    // Test align_size
    assert(align_size(10, 16) == 16);
    assert(align_size(16, 16) == 16);
    assert(align_size(17, 16) == 32);
    assert(align_size(32, 16) == 32);

    // Test is_aligned
    int aligned_array[4];
    void* aligned_ptr = static_cast<void*>(aligned_array);
    // The array should be aligned to at least sizeof(int)
    assert(is_aligned(aligned_ptr, sizeof(int)));

    std::cout << "  Alignment utilities: PASSED" << std::endl;
}

void test_memory_stats() {
    std::cout << "Testing MemoryStats..." << std::endl;

    MemoryStats stats;

    // Initial state
    assert(stats.total_allocated == 0);
    assert(stats.peak_allocated == 0);
    assert(stats.allocation_count == 0);
    assert(stats.deallocation_count == 0);

    // Record allocations
    stats.record_allocation(1024);
    assert(stats.total_allocated == 1024);
    assert(stats.peak_allocated == 1024);
    assert(stats.allocation_count == 1);

    stats.record_allocation(2048);
    assert(stats.total_allocated == 3072);
    assert(stats.peak_allocated == 3072);
    assert(stats.allocation_count == 2);

    // Record deallocation
    stats.record_deallocation(1024);
    assert(stats.total_allocated == 2048);
    assert(stats.peak_allocated == 3072);  // Peak should remain
    assert(stats.deallocation_count == 1);

    // Reset
    stats.reset();
    assert(stats.total_allocated == 0);
    assert(stats.peak_allocated == 0);

    std::cout << "  MemoryStats: PASSED" << std::endl;
}

void test_system_allocator() {
    std::cout << "Testing SystemAllocator..." << std::endl;

    SystemAllocator allocator(0);

    // Test basic allocation
    void* ptr = allocator.allocate(1024, 16);
    assert(ptr != nullptr);
    assert(is_aligned(ptr, 16));

    // Check stats
    auto stats = allocator.get_stats();
    assert(stats.total_allocated == 1024);
    assert(stats.allocation_count == 1);

    // Test deallocation
    allocator.deallocate(ptr, 1024);
    stats = allocator.get_stats();
    assert(stats.total_allocated == 0);
    assert(stats.deallocation_count == 1);

    // Test template allocation
    int* int_array = allocator.Allocator::allocate<int>(100);
    assert(int_array != nullptr);

    allocator.Allocator::deallocate<int>(int_array, 100);

    // Test device info
    assert(allocator.get_device_id() == 0);
    assert(allocator.get_device_type() == DeviceType::CPU);

    std::cout << "  SystemAllocator: PASSED" << std::endl;
}

void test_pool_allocator() {
    std::cout << "Testing PoolAllocator..." << std::endl;

    const usize block_size = 64;
    const usize blocks_per_chunk = 10;

    PoolAllocator allocator(block_size, blocks_per_chunk, 0);

    // Test block size
    assert(allocator.get_block_size() == block_size);

    // Allocate several blocks
    void* ptr1 = allocator.allocate(block_size, 16);
    void* ptr2 = allocator.allocate(block_size, 16);
    void* ptr3 = allocator.allocate(block_size, 16);

    assert(ptr1 != nullptr);
    assert(ptr2 != nullptr);
    assert(ptr3 != nullptr);
    assert(ptr1 != ptr2);
    assert(ptr2 != ptr3);

    // Check stats
    auto stats = allocator.get_stats();
    assert(stats.allocation_count == 3);

    // Deallocate
    allocator.deallocate(ptr1, block_size);
    allocator.deallocate(ptr2, block_size);
    allocator.deallocate(ptr3, block_size);

    stats = allocator.get_stats();
    assert(stats.deallocation_count == 3);

    // Clear pool
    allocator.clear();

    std::cout << "  PoolAllocator: PASSED" << std::endl;
}

void test_memory_manager() {
    std::cout << "Testing MemoryManager..." << std::endl;

    auto& manager = MemoryManager::instance();

    // Enable tracking
    manager.enable_tracking(true);
    assert(manager.is_tracking_enabled());

    // Test global allocation
    void* ptr = manager.allocate(2048, 16, 0);
    assert(ptr != nullptr);
    assert(is_aligned(ptr, 16));

    // Test template allocation
    float* float_array = manager.allocate<float>(256, 0);
    assert(float_array != nullptr);

    // Check stats
    auto stats = manager.get_device_stats(0);
    assert(stats.allocation_count >= 2);

    // Deallocate
    manager.deallocate(ptr, 2048, 0);
    manager.deallocate<float>(float_array, 256, 0);

    // Print stats
    std::cout << "  Memory statistics:" << std::endl;
    manager.print_stats();

    std::cout << "  MemoryManager: PASSED" << std::endl;
}

void test_global_functions() {
    std::cout << "Testing global memory functions..." << std::endl;

    // Test allocation
    void* ptr = allocate(512, 32, 0);
    assert(ptr != nullptr);
    assert(is_aligned(ptr, 32));

    // Test template allocation
    double* double_array = allocate<double>(128, 0);
    assert(double_array != nullptr);

    // Initialize some data
    for (usize i = 0; i < 128; ++i) {
        double_array[i] = static_cast<double>(i);
    }

    // Verify data
    assert(double_array[0] == 0.0);
    assert(double_array[127] == 127.0);

    // Deallocate
    deallocate(ptr, 512, 0);
    deallocate<double>(double_array, 128, 0);

    std::cout << "  Global functions: PASSED" << std::endl;
}

void test_memory_raii() {
    std::cout << "Testing Memory RAII wrapper..." << std::endl;

    {
        // Test basic construction
        Memory<int> mem(100, 0);
        assert(mem.get() != nullptr);
        assert(mem.size() == 100);
        assert(mem.device_id() == 0);
        assert(static_cast<bool>(mem));

        // Test data access
        for (usize i = 0; i < 100; ++i) {
            mem[i] = static_cast<int>(i);
        }

        // Verify data
        assert(mem[0] == 0);
        assert(mem[99] == 99);

        // Test move semantics
        Memory<int> mem2(std::move(mem));
        assert(mem2.get() != nullptr);
        assert(mem.get() == nullptr);  // Moved from
        assert(mem2[99] == 99);

        // Memory should be automatically freed when mem2 goes out of scope
    }

    {
        // Test reset
        Memory<float> mem(50, 0);
        mem.reset(100, 0);
        assert(mem.size() == 100);

        mem.reset(0);
        assert(mem.get() == nullptr);
        assert(mem.size() == 0);
    }

    std::cout << "  Memory RAII wrapper: PASSED" << std::endl;
}

void test_memory_leaks() {
    std::cout << "Testing for memory leaks..." << std::endl;

    auto& manager = MemoryManager::instance();
    manager.reset_stats();

    // Allocate and deallocate multiple times
    const int iterations = 100;
    for (int i = 0; i < iterations; ++i) {
        void* ptr = allocate(1024, 16, 0);
        assert(ptr != nullptr);
        deallocate(ptr, 1024, 0);
    }

    auto stats = manager.get_device_stats(0);
    std::cout << "  Allocations: " << stats.allocation_count << std::endl;
    std::cout << "  Deallocations: " << stats.deallocation_count << std::endl;
    std::cout << "  Current allocated: " << stats.total_allocated << " bytes" << std::endl;
    std::cout << "  Peak allocated: " << stats.peak_allocated << " bytes" << std::endl;

    // All memory should be deallocated
    assert(stats.total_allocated == 0);

    std::cout << "  No memory leaks detected: PASSED" << std::endl;
}

int main() {
    std::cout << "\n=== Memory Management Tests ===" << std::endl;
    std::cout << std::endl;

    try {
        test_alignment_utilities();
        std::cout << std::endl;

        test_memory_stats();
        std::cout << std::endl;

        test_system_allocator();
        std::cout << std::endl;

        test_pool_allocator();
        std::cout << std::endl;

        test_memory_manager();
        std::cout << std::endl;

        test_global_functions();
        std::cout << std::endl;

        test_memory_raii();
        std::cout << std::endl;

        test_memory_leaks();
        std::cout << std::endl;

        std::cout << "=== All Memory Tests PASSED ===" << std::endl;
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        return 1;
    }
}
