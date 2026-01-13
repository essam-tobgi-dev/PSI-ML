#include "../../include/core/device.h"
#include "../../include/core/logging.h"
#include <thread>
#include <sstream>

#ifdef _WIN32
    #include <windows.h>
    #include <intrin.h>
#elif defined(__APPLE__)
    #include <sys/sysctl.h>
    #include <mach/mach.h>
    #include <mach/vm_statistics.h>
    #include <mach/mach_types.h>
    #include <mach/vm_map.h>
#elif defined(__linux__)
    #include <sys/sysinfo.h>
    #include <unistd.h>
    #include <fstream>
    #include <string>
#endif

// Platform-specific includes for CPU feature detection
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    #ifdef _MSC_VER
        #include <intrin.h>
    #else
        #include <cpuid.h>
        #include <x86intrin.h>
    #endif
#elif defined(__aarch64__) || defined(__arm__)
    #ifdef __linux__
        #include <sys/auxv.h>
        #include <asm/hwcap.h>
    #endif
#endif

namespace psi {
namespace core {

// CPUDevice implementation
CPUDevice::CPUDevice() {
    info_.id = 0;
    info_.type = DeviceType::CPU;
    info_.capabilities = DeviceCapability::None;
    
    detect_cpu_info();
    detect_capabilities();
    
    PSI_INFO_NAMED("device", "Initialized CPU device: " + info_.name);
    PSI_INFO_NAMED("device", "CPU cores: " + std::to_string(info_.num_cores));
    PSI_INFO_NAMED("device", "Total memory: " + std::to_string(info_.total_memory / (1024*1024)) + " MB");
}

void CPUDevice::detect_cpu_info() {
    // Detect number of cores
    info_.num_cores = std::thread::hardware_concurrency();
    if (info_.num_cores == 0) {
        info_.num_cores = 1;  // Fallback
    }
    
    // Detect total memory
#ifdef _WIN32
    MEMORYSTATUSEX memInfo;
    memInfo.dwLength = sizeof(MEMORYSTATUSEX);
    GlobalMemoryStatusEx(&memInfo);
    info_.total_memory = static_cast<memory_size_t>(memInfo.ullTotalPhys);
    
    // Get CPU name from registry or WMI (simplified)
    info_.name = "CPU (Windows)";
    
#elif defined(__APPLE__)
    // Get memory size
    int mib[2];
    int64_t physical_memory;
    size_t length;
    
    mib[0] = CTL_HW;
    mib[1] = HW_MEMSIZE;
    length = sizeof(int64_t);
    sysctl(mib, 2, &physical_memory, &length, NULL, 0);
    info_.total_memory = static_cast<memory_size_t>(physical_memory);
    
    // Get CPU name
    char cpu_name[256];
    size_t cpu_name_len = sizeof(cpu_name);
    sysctlbyname("machdep.cpu.brand_string", &cpu_name, &cpu_name_len, NULL, 0);
    info_.name = std::string(cpu_name);
    
#elif defined(__linux__)
    // Get memory from /proc/meminfo
    std::ifstream meminfo("/proc/meminfo");
    std::string line;
    while (std::getline(meminfo, line)) {
        if (line.find("MemTotal:") != std::string::npos) {
            std::istringstream iss(line);
            std::string label, size_str, unit;
            iss >> label >> size_str >> unit;
            u64 size_kb = std::stoull(size_str);
            info_.total_memory = size_kb * 1024;  // Convert KB to bytes
            break;
        }
    }
    
    // Get CPU name from /proc/cpuinfo
    std::ifstream cpuinfo("/proc/cpuinfo");
    while (std::getline(cpuinfo, line)) {
        if (line.find("model name") != std::string::npos) {
            size_t colon_pos = line.find(':');
            if (colon_pos != std::string::npos) {
                info_.name = line.substr(colon_pos + 2);  // Skip ": "
            }
            break;
        }
    }
    
    if (info_.name.empty()) {
        info_.name = "CPU (Linux)";
    }
    
#else
    // Generic fallback
    info_.total_memory = 1024 * 1024 * 1024;  // 1GB fallback
    info_.name = "CPU (Unknown)";
#endif

    info_.available_memory = info_.total_memory;
    
    // Estimate peak FLOPS (very rough)
    info_.peak_flops = static_cast<f32>(info_.num_cores * 2.0e9);  // 2 GHz assumption
}

void CPUDevice::detect_capabilities() {
    info_.capabilities = DeviceCapability::Multithreading;

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    // x86/x64 CPUID detection
    u32 regs[4] = {0, 0, 0, 0};

#ifdef _MSC_VER
    __cpuid(reinterpret_cast<int*>(regs), 1);

    // Check SSE/AVX support
    if (regs[3] & (1 << 25)) {  // SSE
        info_.capabilities = info_.capabilities | DeviceCapability::SIMD;
    }

    if (regs[2] & (1 << 28)) {  // AVX
        info_.capabilities = info_.capabilities | DeviceCapability::AVX;
    }

    // Check AVX2 support
    __cpuid(reinterpret_cast<int*>(regs), 7);

    if (regs[1] & (1 << 5)) {  // AVX2
        info_.capabilities = info_.capabilities | DeviceCapability::AVX2;
    }

    if (regs[1] & (1 << 16)) {  // AVX-512F
        info_.capabilities = info_.capabilities | DeviceCapability::AVX512;
    }

    if (regs[1] & (1 << 12)) {  // FMA3
        info_.capabilities = info_.capabilities | DeviceCapability::FMA;
    }
#else
    // GCC/Clang/MinGW
    if (__get_cpuid(1, &regs[0], &regs[1], &regs[2], &regs[3])) {
        // Check SSE/AVX support
        if (regs[3] & (1 << 25)) {  // SSE
            info_.capabilities = info_.capabilities | DeviceCapability::SIMD;
        }

        if (regs[2] & (1 << 28)) {  // AVX
            info_.capabilities = info_.capabilities | DeviceCapability::AVX;
        }
    }

    // Check AVX2 support
    if (__get_cpuid_count(7, 0, &regs[0], &regs[1], &regs[2], &regs[3])) {
        if (regs[1] & (1 << 5)) {  // AVX2
            info_.capabilities = info_.capabilities | DeviceCapability::AVX2;
        }

        if (regs[1] & (1 << 16)) {  // AVX-512F
            info_.capabilities = info_.capabilities | DeviceCapability::AVX512;
        }

        if (regs[1] & (1 << 12)) {  // FMA3
            info_.capabilities = info_.capabilities | DeviceCapability::FMA;
        }
    }
#endif
    
#elif defined(__aarch64__) || defined(__arm__)
    // ARM NEON detection
    info_.capabilities = info_.capabilities | DeviceCapability::SIMD;
    
#ifdef __linux__
    // Check for advanced ARM features using getauxval
    unsigned long hwcap = getauxval(AT_HWCAP);
    if (hwcap & HWCAP_ASIMD) {
        info_.capabilities = info_.capabilities | DeviceCapability::SIMD;
    }
#endif

#endif

    // Check for OpenMP support
#ifdef _OPENMP
    info_.capabilities = info_.capabilities | DeviceCapability::OpenMP;
#endif

    // BLAS support depends on compile-time flags
#if PSI_ENABLE_BLAS
    info_.capabilities = info_.capabilities | DeviceCapability::BLAS;
#endif
}

memory_size_t CPUDevice::get_available_memory() const {
#ifdef _WIN32
    MEMORYSTATUSEX memInfo;
    memInfo.dwLength = sizeof(MEMORYSTATUSEX);
    GlobalMemoryStatusEx(&memInfo);
    return static_cast<memory_size_t>(memInfo.ullAvailPhys);
    
#elif defined(__APPLE__)
    vm_size_t page_size;
    vm_statistics64_data_t vm_stat;
    mach_port_t mach_port = mach_host_self();
    mach_msg_type_number_t count = sizeof(vm_stat) / sizeof(natural_t);
    
    if (KERN_SUCCESS == host_page_size(mach_port, &page_size) &&
        KERN_SUCCESS == host_statistics64(mach_port, HOST_VM_INFO,
                                         (host_info64_t)&vm_stat, &count)) {
        return static_cast<memory_size_t>(vm_stat.free_count * page_size);
    }
    return info_.total_memory / 2;  // Fallback estimate
    
#elif defined(__linux__)
    struct sysinfo sys_info;
    if (sysinfo(&sys_info) == 0) {
        return static_cast<memory_size_t>(sys_info.freeram * sys_info.mem_unit);
    }
    return info_.total_memory / 2;  // Fallback estimate
    
#else
    return info_.total_memory / 2;  // Generic fallback
#endif
}

f32 CPUDevice::get_utilization() const {
    // CPU utilization detection is complex and platform-specific
    // For now, return a placeholder value
    // In a real implementation, this would sample CPU usage over time
    return 0.1f;  // 10% placeholder
}

// DeviceManager implementation
DeviceManager::DeviceManager() : default_device_(nullptr) {
    detect_cpu_devices();
    
    if (!devices_.empty()) {
        default_device_ = devices_[0].get();
        PSI_INFO_NAMED("device", "Set default device to: " + default_device_->get_name());
    }
}

Device* DeviceManager::get_device(device_id_t id) const {
    for (const auto& device : devices_) {
        if (device->get_id() == id) {
            return device.get();
        }
    }
    return nullptr;
}

void DeviceManager::set_default_device(device_id_t id) {
    Device* device = get_device(id);
    if (device) {
        default_device_ = device;
        PSI_INFO_NAMED("device", "Changed default device to: " + device->get_name());
    } else {
        PSI_ERROR_NAMED("device", "Cannot set default device: invalid device ID " + std::to_string(id));
    }
}

std::vector<Device*> DeviceManager::get_devices_by_type(DeviceType type) const {
    std::vector<Device*> result;
    for (const auto& device : devices_) {
        if (device->get_type() == type) {
            result.push_back(device.get());
        }
    }
    return result;
}

Device* DeviceManager::get_current_device() const {
    thread_local Device* current_device = default_device_;
    return current_device ? current_device : default_device_;
}

void DeviceManager::set_current_device(device_id_t id) {
    Device* device = get_device(id);
    if (device) {
        thread_local Device* current_device = device;
        (void)current_device;  // Suppress unused variable warning
        PSI_DEBUG_NAMED("device", "Set current device for thread: " + device->get_name());
    } else {
        PSI_ERROR_NAMED("device", "Cannot set current device: invalid device ID " + std::to_string(id));
    }
}

void DeviceManager::refresh_devices() {
    PSI_INFO_NAMED("device", "Refreshing device list");
    devices_.clear();
    default_device_ = nullptr;
    
    detect_cpu_devices();
    
    if (!devices_.empty()) {
        default_device_ = devices_[0].get();
    }
}

void DeviceManager::print_device_info() const {
    std::ostringstream oss;
    oss << "Available devices:\n";
    
    for (const auto& device : devices_) {
        DeviceInfo info = device->get_info();
        oss << "  Device " << info.id << ": " << info.name << "\n";
        oss << "    Type: " << ((info.type == DeviceType::CPU) ? "CPU" : "Unknown") << "\n";
        oss << "    Cores: " << info.num_cores << "\n";
        oss << "    Total Memory: " << (info.total_memory / (1024*1024)) << " MB\n";
        oss << "    Available Memory: " << (device->get_available_memory() / (1024*1024)) << " MB\n";
        
        // Print capabilities
        oss << "    Capabilities: ";
        bool first = true;
        if (has_capability(info.capabilities, DeviceCapability::SIMD)) {
            oss << (first ? "" : ", ") << "SIMD"; first = false;
        }
        if (has_capability(info.capabilities, DeviceCapability::AVX)) {
            oss << (first ? "" : ", ") << "AVX"; first = false;
        }
        if (has_capability(info.capabilities, DeviceCapability::AVX2)) {
            oss << (first ? "" : ", ") << "AVX2"; first = false;
        }
        if (has_capability(info.capabilities, DeviceCapability::AVX512)) {
            oss << (first ? "" : ", ") << "AVX512"; first = false;
        }
        if (has_capability(info.capabilities, DeviceCapability::FMA)) {
            oss << (first ? "" : ", ") << "FMA"; first = false;
        }
        if (has_capability(info.capabilities, DeviceCapability::Multithreading)) {
            oss << (first ? "" : ", ") << "Multithreading"; first = false;
        }
        if (has_capability(info.capabilities, DeviceCapability::OpenMP)) {
            oss << (first ? "" : ", ") << "OpenMP"; first = false;
        }
        if (has_capability(info.capabilities, DeviceCapability::BLAS)) {
            oss << (first ? "" : ", ") << "BLAS"; first = false;
        }
        if (first) oss << "None";
        oss << "\n";
    }
    
    PSI_INFO_NAMED("device", oss.str());
}

void DeviceManager::detect_cpu_devices() {
    // Create CPU device
    auto cpu_device = std::make_unique<CPUDevice>();
    devices_.push_back(std::move(cpu_device));
    
    PSI_INFO_NAMED("device", "Detected " + std::to_string(devices_.size()) + " CPU device(s)");
}

} // namespace core
} // namespace psi