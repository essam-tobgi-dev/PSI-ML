#include "../include/core/device.h"
#include "../include/core/logging.h"
#include <iostream>
#include <cassert>

using namespace psi::core;

void test_device_info() {
    std::cout << "Testing DeviceInfo..." << std::endl;

    DeviceInfo info;
    assert(info.type == DeviceType::Unknown);
    assert(info.num_cores == 0);
    assert(info.name == "Unknown");

    std::cout << "  DeviceInfo default construction: PASSED" << std::endl;
}

void test_cpu_device() {
    std::cout << "Testing CPUDevice..." << std::endl;

    CPUDevice cpu;

    // Test basic properties
    assert(cpu.get_type() == DeviceType::CPU);
    assert(cpu.get_id() >= 0);
    assert(!cpu.get_name().empty());
    assert(cpu.get_num_cores() > 0);

    std::cout << "  CPU Device Name: " << cpu.get_name() << std::endl;
    std::cout << "  CPU Cores: " << cpu.get_num_cores() << std::endl;
    std::cout << "  CPU Device ID: " << cpu.get_id() << std::endl;

    // Test available memory
    auto available = cpu.get_available_memory();
    std::cout << "  Available Memory: " << available << " bytes" << std::endl;
    assert(available > 0);

    // Test utilization (should return a value between 0 and 100)
    auto util = cpu.get_utilization();
    std::cout << "  CPU Utilization: " << util << "%" << std::endl;
    assert(util >= 0.0f && util <= 100.0f);

    // Test synchronize (should be a no-op for CPU)
    cpu.synchronize();

    std::cout << "  CPUDevice operations: PASSED" << std::endl;
}

void test_device_capabilities() {
    std::cout << "Testing Device Capabilities..." << std::endl;

    CPUDevice cpu;
    DeviceInfo info = cpu.get_info();

    std::cout << "  Available capabilities:" << std::endl;

    if (has_capability(info.capabilities, DeviceCapability::SIMD)) {
        std::cout << "    - SIMD support" << std::endl;
    }
    if (has_capability(info.capabilities, DeviceCapability::AVX)) {
        std::cout << "    - AVX support" << std::endl;
    }
    if (has_capability(info.capabilities, DeviceCapability::AVX2)) {
        std::cout << "    - AVX2 support" << std::endl;
    }
    if (has_capability(info.capabilities, DeviceCapability::AVX512)) {
        std::cout << "    - AVX-512 support" << std::endl;
    }
    if (has_capability(info.capabilities, DeviceCapability::FMA)) {
        std::cout << "    - FMA support" << std::endl;
    }
    if (has_capability(info.capabilities, DeviceCapability::Multithreading)) {
        std::cout << "    - Multithreading support" << std::endl;
    }
    if (has_capability(info.capabilities, DeviceCapability::OpenMP)) {
        std::cout << "    - OpenMP support" << std::endl;
    }

    std::cout << "  Device capabilities detection: PASSED" << std::endl;
}

void test_device_manager() {
    std::cout << "Testing DeviceManager..." << std::endl;

    auto& manager = DeviceManager::instance();

    // Test getting devices
    const auto& devices = manager.get_devices();
    assert(!devices.empty());
    std::cout << "  Total devices: " << devices.size() << std::endl;

    // Test default device
    auto* default_dev = manager.get_default_device();
    assert(default_dev != nullptr);
    assert(default_dev->get_type() == DeviceType::CPU);
    std::cout << "  Default device: " << default_dev->get_name() << std::endl;

    // Test current device
    auto* current_dev = manager.get_current_device();
    assert(current_dev != nullptr);

    // Test getting CPU devices
    auto cpu_devices = manager.get_devices_by_type(DeviceType::CPU);
    assert(!cpu_devices.empty());
    std::cout << "  CPU devices: " << cpu_devices.size() << std::endl;

    // Print device information
    manager.print_device_info();

    std::cout << "  DeviceManager operations: PASSED" << std::endl;
}

void test_global_functions() {
    std::cout << "Testing global convenience functions..." << std::endl;

    auto* default_dev = get_default_device();
    assert(default_dev != nullptr);

    auto* current_dev = get_current_device();
    assert(current_dev != nullptr);

    auto cpu_devices = get_cpu_devices();
    assert(!cpu_devices.empty());

    std::cout << "  Global functions: PASSED" << std::endl;
}

void test_device_context() {
    std::cout << "Testing DeviceContext..." << std::endl;

    auto* initial_device = get_current_device();
    assert(initial_device != nullptr);
    device_id_t initial_id = initial_device->get_id();

    {
        DeviceContext ctx(initial_id);
        auto* ctx_device = get_current_device();
        assert(ctx_device->get_id() == initial_id);
    }

    // After context destruction, device should be restored
    auto* restored_device = get_current_device();
    assert(restored_device->get_id() == initial_id);

    std::cout << "  DeviceContext RAII: PASSED" << std::endl;
}

int main() {
    std::cout << "\n=== Device Management Tests ===" << std::endl;
    std::cout << std::endl;

    try {
        test_device_info();
        std::cout << std::endl;

        test_cpu_device();
        std::cout << std::endl;

        test_device_capabilities();
        std::cout << std::endl;

        test_device_manager();
        std::cout << std::endl;

        test_global_functions();
        std::cout << std::endl;

        test_device_context();
        std::cout << std::endl;

        std::cout << "=== All Device Tests PASSED ===" << std::endl;
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        return 1;
    }
}
