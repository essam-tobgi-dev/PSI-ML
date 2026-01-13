#pragma once

#include "types.h"
#include "config.h"
#include <string>
#include <vector>
#include <memory>

namespace psi {
    namespace core {

        // Device types
        enum class DeviceType : u8 {
            CPU = 0,
            GPU = 1,      // For future GPU support
            Unknown = 255
        };

        // Device capability flags
        enum class DeviceCapability : u32 {
            None = 0,
            SIMD = 1 << 0,       // SIMD instructions (SSE, AVX, NEON)
            AVX = 1 << 1,        // AVX support
            AVX2 = 1 << 2,       // AVX2 support
            AVX512 = 1 << 3,     // AVX-512 support
            FMA = 1 << 4,        // Fused multiply-add
            Multithreading = 1 << 5,
            OpenMP = 1 << 6,     // OpenMP support
            BLAS = 1 << 7,       // External BLAS library
        };

        // Bitwise operations for capability flags
        constexpr DeviceCapability operator|(DeviceCapability lhs, DeviceCapability rhs) {
            return static_cast<DeviceCapability>(
                static_cast<u32>(lhs) | static_cast<u32>(rhs)
                );
        }

        constexpr DeviceCapability operator&(DeviceCapability lhs, DeviceCapability rhs) {
            return static_cast<DeviceCapability>(
                static_cast<u32>(lhs) & static_cast<u32>(rhs)
                );
        }

        constexpr bool has_capability(DeviceCapability caps, DeviceCapability flag) {
            return (caps & flag) == flag;
        }

        // Device information structure
        struct DeviceInfo {
            device_id_t id;
            DeviceType type;
            std::string name;
            DeviceCapability capabilities;
            u32 num_cores;
            memory_size_t total_memory;
            memory_size_t available_memory;
            f32 peak_flops;  // Peak floating point operations per second

            DeviceInfo()
                : id(-1)
                , type(DeviceType::Unknown)
                , name("Unknown")
                , capabilities(DeviceCapability::None)
                , num_cores(0)
                , total_memory(0)
                , available_memory(0)
                , peak_flops(0.0f) {
            }
        };

        // Abstract device interface
        class Device {
        public:
            virtual ~Device() = default;

            virtual DeviceInfo get_info() const = 0;
            virtual device_id_t get_id() const = 0;
            virtual DeviceType get_type() const = 0;
            virtual bool has_capability(DeviceCapability cap) const = 0;

            virtual void synchronize() = 0;
            virtual memory_size_t get_available_memory() const = 0;
            virtual f32 get_utilization() const = 0;  // Current utilization percentage

            // Device name for debugging/logging
            virtual std::string get_name() const = 0;
        };

        // CPU device implementation
        class CPUDevice : public Device {
        public:
            CPUDevice();
            ~CPUDevice() override = default;

            DeviceInfo get_info() const override { return info_; }
            device_id_t get_id() const override { return info_.id; }
            DeviceType get_type() const override { return DeviceType::CPU; }
            bool has_capability(DeviceCapability cap) const override {
                return ::psi::core::has_capability(info_.capabilities, cap);
            }

            void synchronize() override {}  // No-op for CPU
            memory_size_t get_available_memory() const override;
            f32 get_utilization() const override;
            std::string get_name() const override { return info_.name; }

            // CPU-specific methods
            u32 get_num_cores() const { return info_.num_cores; }
            bool supports_simd() const {
                return has_capability(DeviceCapability::SIMD);
            }
            bool supports_avx() const {
                return has_capability(DeviceCapability::AVX);
            }
            bool supports_avx2() const {
                return has_capability(DeviceCapability::AVX2);
            }

        private:
            DeviceInfo info_;
            void detect_capabilities();
            void detect_cpu_info();
        };

        // Device manager singleton
        class DeviceManager {
        public:
            static DeviceManager& instance() {
                static DeviceManager manager;
                return manager;
            }

            // Get available devices
            PSI_NODISCARD const std::vector<std::unique_ptr<Device>>& get_devices() const {
                return devices_;
            }

            // Get device by ID
            PSI_NODISCARD Device* get_device(device_id_t id) const;

            // Get default device (first CPU device)
            PSI_NODISCARD Device* get_default_device() const {
                return default_device_;
            }

            // Set default device
            void set_default_device(device_id_t id);

            // Get devices by type
            PSI_NODISCARD std::vector<Device*> get_devices_by_type(DeviceType type) const;

            // Get current device (thread-local)
            PSI_NODISCARD Device* get_current_device() const;

            // Set current device (thread-local)
            void set_current_device(device_id_t id);

            // Device detection
            void refresh_devices();

            // Print device information
            void print_device_info() const;

        private:
            DeviceManager();
            ~DeviceManager() = default;
            DeviceManager(const DeviceManager&) = delete;
            DeviceManager& operator=(const DeviceManager&) = delete;

            std::vector<std::unique_ptr<Device>> devices_;
            Device* default_device_;

            void detect_cpu_devices();
            // void detect_gpu_devices(); // For future GPU support
        };

        // Global convenience functions
        PSI_NODISCARD inline Device* get_default_device() {
            return DeviceManager::instance().get_default_device();
        }

        PSI_NODISCARD inline Device* get_current_device() {
            return DeviceManager::instance().get_current_device();
        }

        inline void set_device(device_id_t id) {
            DeviceManager::instance().set_current_device(id);
        }

        PSI_NODISCARD inline std::vector<Device*> get_cpu_devices() {
            return DeviceManager::instance().get_devices_by_type(DeviceType::CPU);
        }

        // Device context RAII helper
        class DeviceContext {
        public:
            explicit DeviceContext(device_id_t device_id)
                : previous_device_(get_current_device()) {
                set_device(device_id);
            }

            ~DeviceContext() {
                if (previous_device_) {
                    set_device(previous_device_->get_id());
                }
            }

            DeviceContext(const DeviceContext&) = delete;
            DeviceContext& operator=(const DeviceContext&) = delete;

        private:
            Device* previous_device_;
        };

        // Macro for device context
#define PSI_DEVICE_CONTEXT(device_id) \
    ::psi::core::DeviceContext _psi_device_ctx(device_id)

    } // namespace core
} // namespace psi