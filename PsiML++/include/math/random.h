#pragma once

#include "../core/types.h"
#include "../core/config.h"
#include "../core/memory.h"
#include "../core/exception.h"
#include "../core/device.h"
#include "vector.h"
#include "matrix.h"
#include "tensor.h"
#include <random>
#include <chrono>
#include <algorithm>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace psi {
    namespace math {

        // Random number generator types
        enum class GeneratorType : core::u8 {
            MersenneTwister = 0,    // std::mt19937
            LinearCongruential = 1, // std::minstd_rand
            XORShift = 2,           // Custom XORShift implementation
            PCG = 3                 // Custom PCG implementation (if available)
        };

        // Distribution types
        enum class DistributionType : core::u8 {
            Uniform = 0,
            Normal = 1,
            Bernoulli = 2,
            Exponential = 3,
            Gamma = 4,
            Beta = 5,
            Binomial = 6,
            Poisson = 7
        };

        // Base random number generator interface
        class RandomGenerator {
        public:
            virtual ~RandomGenerator() = default;
            virtual void seed(core::u64 seed_value) = 0;
            virtual core::u32 next_u32() = 0;
            virtual core::u64 next_u64() = 0;
            virtual core::f64 next_f64() = 0;  // [0, 1)
            virtual core::f32 next_f32() = 0;  // [0, 1)
            virtual GeneratorType type() const = 0;
            virtual std::string name() const = 0;
        };

        // Mersenne Twister implementation
        class MersenneTwisterGenerator : public RandomGenerator {
        public:
            explicit MersenneTwisterGenerator(core::u64 seed = 0) : gen_(seed) {
                if (seed == 0) {
                    auto now = std::chrono::high_resolution_clock::now();
                    auto time_seed = static_cast<core::u64>(now.time_since_epoch().count());
                    gen_.seed(time_seed);
                }
            }

            void seed(core::u64 seed_value) override {
                gen_.seed(static_cast<std::mt19937_64::result_type>(seed_value));
            }

            core::u32 next_u32() override {
                return static_cast<core::u32>(gen_());
            }

            core::u64 next_u64() override {
                return gen_();
            }

            core::f64 next_f64() override {
                return uniform_01_dist_(gen_);
            }

            core::f32 next_f32() override {
                return static_cast<core::f32>(uniform_01_dist_(gen_));
            }

            GeneratorType type() const override { return GeneratorType::MersenneTwister; }
            std::string name() const override { return "MersenneTwister"; }

        private:
            std::mt19937_64 gen_;
            std::uniform_real_distribution<core::f64> uniform_01_dist_{ 0.0, 1.0 };
        };

        // XORShift implementation (simple and fast)
        class XORShiftGenerator : public RandomGenerator {
        public:
            explicit XORShiftGenerator(core::u64 seed = 0) : state_(seed) {
                if (seed == 0) {
                    auto now = std::chrono::high_resolution_clock::now();
                    state_ = static_cast<core::u64>(now.time_since_epoch().count());
                }
                // Ensure state is not zero
                if (state_ == 0) state_ = 1;
            }

            void seed(core::u64 seed_value) override {
                state_ = seed_value;
                if (state_ == 0) state_ = 1;
            }

            core::u32 next_u32() override {
                return static_cast<core::u32>(next_u64());
            }

            core::u64 next_u64() override {
                state_ ^= state_ << 13;
                state_ ^= state_ >> 7;
                state_ ^= state_ << 17;
                return state_;
            }

            core::f64 next_f64() override {
                return static_cast<core::f64>(next_u64()) / static_cast<core::f64>(UINT64_MAX);
            }

            core::f32 next_f32() override {
                return static_cast<core::f32>(next_u32()) / static_cast<core::f32>(UINT32_MAX);
            }

            GeneratorType type() const override { return GeneratorType::XORShift; }
            std::string name() const override { return "XORShift"; }

        private:
            core::u64 state_;
        };

        // Random number engine manager
        class Random {
        public:
            explicit Random(GeneratorType gen_type = GeneratorType::MersenneTwister,
                core::u64 seed = 0)
                : generator_(create_generator(gen_type, seed)) {
            }

            // Seed management
            void seed(core::u64 seed_value) {
                generator_->seed(seed_value);
            }

            void random_seed() {
                auto now = std::chrono::high_resolution_clock::now();
                auto time_seed = static_cast<core::u64>(now.time_since_epoch().count());
                generator_->seed(time_seed);
            }

            // Basic random number generation
            PSI_NODISCARD core::f32 uniform() {
                return generator_->next_f32();
            }

            PSI_NODISCARD core::f64 uniform_f64() {
                return generator_->next_f64();
            }

            PSI_NODISCARD core::f32 uniform(core::f32 min_val, core::f32 max_val) {
                return min_val + uniform() * (max_val - min_val);
            }

            PSI_NODISCARD core::f64 uniform(core::f64 min_val, core::f64 max_val) {
                return min_val + uniform_f64() * (max_val - min_val);
            }

            PSI_NODISCARD core::i32 uniform_int(core::i32 min_val, core::i32 max_val) {
                PSI_ASSERT(min_val <= max_val, "min_val must be <= max_val");
                core::u32 range = static_cast<core::u32>(max_val - min_val + 1);
                return min_val + static_cast<core::i32>(generator_->next_u32() % range);
            }

            // Normal distribution (Box-Muller transform)
            PSI_NODISCARD core::f32 normal(core::f32 mean = 0.0f, core::f32 stddev = 1.0f) {
                return static_cast<core::f32>(normal_f64(mean, stddev));
            }

            PSI_NODISCARD core::f64 normal_f64(core::f64 mean = 0.0, core::f64 stddev = 1.0) {
                static bool has_spare = false;
                static core::f64 spare;

                if (has_spare) {
                    has_spare = false;
                    return spare * stddev + mean;
                }

                has_spare = true;
                core::f64 u = uniform_f64();
                core::f64 v = uniform_f64();
                core::f64 mag = stddev * std::sqrt(-2.0 * std::log(u));
                spare = mag * std::cos(2.0 * M_PI * v);
                return mag * std::sin(2.0 * M_PI * v) + mean;
            }

            // Exponential distribution
            PSI_NODISCARD core::f32 exponential(core::f32 lambda = 1.0f) {
                return -std::log(1.0f - uniform()) / lambda;
            }

            PSI_NODISCARD core::f64 exponential_f64(core::f64 lambda = 1.0) {
                return -std::log(1.0 - uniform_f64()) / lambda;
            }

            // Bernoulli distribution
            PSI_NODISCARD bool bernoulli(core::f64 p = 0.5) {
                return uniform_f64() < p;
            }

            // Gamma distribution (using Marsaglia and Tsang's method)
            PSI_NODISCARD core::f32 gamma(core::f32 shape, core::f32 scale = 1.0f) {
                return static_cast<core::f32>(gamma_f64(shape, scale));
            }

            PSI_NODISCARD core::f64 gamma_f64(core::f64 shape, core::f64 scale = 1.0) {
                if (shape < 1.0) {
                    while (true) {
                        core::f64 u = uniform_f64();
                        core::f64 x = std::pow(u, 1.0 / shape);
                        core::f64 v = uniform_f64();
                        if (v <= (2.0 - x) / 2.0) {
                            return x * scale;
                        }
                        if (x <= 1.0 && v <= std::exp(-x)) {
                            return x * scale;
                        }
                    }
                }
                else {
                    // Marsaglia and Tsang's method
                    core::f64 d = shape - 1.0 / 3.0;
                    core::f64 c = 1.0 / std::sqrt(9.0 * d);

                    while (true) {
                        core::f64 x = normal_f64();
                        core::f64 v = 1.0 + c * x;
                        if (v <= 0.0) continue;

                        v = v * v * v;
                        core::f64 u = uniform_f64();
                        if (u < 1.0 - 0.0331 * x * x * x * x) {
                            return d * v * scale;
                        }
                        if (std::log(u) < 0.5 * x * x + d * (1.0 - v + std::log(v))) {
                            return d * v * scale;
                        }
                    }
                }
            }

            // Beta distribution
            PSI_NODISCARD core::f32 beta(core::f32 alpha, core::f32 beta_param) {
                return static_cast<core::f32>(beta_f64(alpha, beta_param));
            }

            PSI_NODISCARD core::f64 beta_f64(core::f64 alpha, core::f64 beta_param) {
                core::f64 x = gamma_f64(alpha, 1.0);
                core::f64 y = gamma_f64(beta_param, 1.0);
                return x / (x + y);
            }

            // Poisson distribution
            PSI_NODISCARD core::i32 poisson(core::f64 lambda) {
                if (lambda < 30.0) {
                    // Use Knuth's algorithm
                    core::f64 L = std::exp(-lambda);
                    core::f64 p = 1.0;
                    core::i32 k = 0;

                    do {
                        k++;
                        p *= uniform_f64();
                    } while (p > L);

                    return k - 1;
                }
                else {
                    // Use normal approximation for large lambda
                    core::f64 g = std::sqrt(lambda) * normal_f64() + lambda;
                    return static_cast<core::i32>(std::max(0.0, g));
                }
            }

            // Vector and Matrix generation
            template<typename T>
            Vector<T> uniform_vector(core::usize size, T min_val = T{ 0 }, T max_val = T{ 1 },
                core::device_id_t device_id = 0) {
                Vector<T> result(size, device_id);
                for (core::usize i = 0; i < size; ++i) {
                    if constexpr (std::is_floating_point_v<T>) {
                        result[i] = static_cast<T>(uniform(static_cast<core::f32>(min_val),
                            static_cast<core::f32>(max_val)));
                    }
                    else {
                        result[i] = static_cast<T>(uniform_int(static_cast<core::i32>(min_val),
                            static_cast<core::i32>(max_val)));
                    }
                }
                return result;
            }

            template<typename T>
            Vector<T> normal_vector(core::usize size, T mean = T{ 0 }, T stddev = T{ 1 },
                core::device_id_t device_id = 0) {
                static_assert(std::is_floating_point_v<T>, "Normal distribution requires floating point type");
                Vector<T> result(size, device_id);
                for (core::usize i = 0; i < size; ++i) {
                    result[i] = static_cast<T>(normal(static_cast<core::f32>(mean),
                        static_cast<core::f32>(stddev)));
                }
                return result;
            }

            template<typename T>
            Matrix<T> uniform_matrix(core::usize rows, core::usize cols,
                T min_val = T{ 0 }, T max_val = T{ 1 },
                core::device_id_t device_id = 0) {
                Matrix<T> result(rows, cols, device_id);
                for (core::usize i = 0; i < rows * cols; ++i) {
                    if constexpr (std::is_floating_point_v<T>) {
                        result[i] = static_cast<T>(uniform(static_cast<core::f32>(min_val),
                            static_cast<core::f32>(max_val)));
                    }
                    else {
                        result[i] = static_cast<T>(uniform_int(static_cast<core::i32>(min_val),
                            static_cast<core::i32>(max_val)));
                    }
                }
                return result;
            }

            template<typename T>
            Matrix<T> normal_matrix(core::usize rows, core::usize cols,
                T mean = T{ 0 }, T stddev = T{ 1 },
                core::device_id_t device_id = 0) {
                static_assert(std::is_floating_point_v<T>, "Normal distribution requires floating point type");
                Matrix<T> result(rows, cols, device_id);
                for (core::usize i = 0; i < rows * cols; ++i) {
                    result[i] = static_cast<T>(normal(static_cast<core::f32>(mean),
                        static_cast<core::f32>(stddev)));
                }
                return result;
            }

            template<typename T>
            Tensor<T> uniform_tensor(const Shape& shape, T min_val = T{ 0 }, T max_val = T{ 1 },
                core::device_id_t device_id = 0) {
                Tensor<T> result(shape, device_id);
                for (core::usize i = 0; i < result.size(); ++i) {
                    if constexpr (std::is_floating_point_v<T>) {
                        result[i] = static_cast<T>(uniform(static_cast<core::f32>(min_val),
                            static_cast<core::f32>(max_val)));
                    }
                    else {
                        result[i] = static_cast<T>(uniform_int(static_cast<core::i32>(min_val),
                            static_cast<core::i32>(max_val)));
                    }
                }
                return result;
            }

            template<typename T>
            Tensor<T> normal_tensor(const Shape& shape, T mean = T{ 0 }, T stddev = T{ 1 },
                core::device_id_t device_id = 0) {
                static_assert(std::is_floating_point_v<T>, "Normal distribution requires floating point type");
                Tensor<T> result(shape, device_id);
                for (core::usize i = 0; i < result.size(); ++i) {
                    result[i] = static_cast<T>(normal(static_cast<core::f32>(mean),
                        static_cast<core::f32>(stddev)));
                }
                return result;
            }

            // Special initialization methods for neural networks
            template<typename T>
            Matrix<T> xavier_uniform(core::usize rows, core::usize cols, core::device_id_t device_id = 0) {
                static_assert(std::is_floating_point_v<T>, "Xavier initialization requires floating point type");
                T limit = static_cast<T>(std::sqrt(6.0 / (rows + cols)));
                return uniform_matrix<T>(rows, cols, -limit, limit, device_id);
            }

            template<typename T>
            Matrix<T> xavier_normal(core::usize rows, core::usize cols, core::device_id_t device_id = 0) {
                static_assert(std::is_floating_point_v<T>, "Xavier initialization requires floating point type");
                T stddev = static_cast<T>(std::sqrt(2.0 / (rows + cols)));
                return normal_matrix<T>(rows, cols, T{ 0 }, stddev, device_id);
            }

            template<typename T>
            Matrix<T> he_uniform(core::usize rows, core::usize cols, core::device_id_t device_id = 0) {
                static_assert(std::is_floating_point_v<T>, "He initialization requires floating point type");
                T limit = static_cast<T>(std::sqrt(6.0 / rows));
                return uniform_matrix<T>(rows, cols, -limit, limit, device_id);
            }

            template<typename T>
            Matrix<T> he_normal(core::usize rows, core::usize cols, core::device_id_t device_id = 0) {
                static_assert(std::is_floating_point_v<T>, "He initialization requires floating point type");
                T stddev = static_cast<T>(std::sqrt(2.0 / rows));
                return normal_matrix<T>(rows, cols, T{ 0 }, stddev, device_id);
            }

            // Utility methods
            template<typename Container>
            void shuffle(Container& container) {
                for (auto i = container.size() - 1; i > 0; --i) {
                    auto j = uniform_int(0, static_cast<core::i32>(i));
                    std::swap(container[i], container[j]);
                }
            }

            template<typename T>
            T choice(const std::vector<T>& options) {
                PSI_ASSERT(!options.empty(), "Cannot choose from empty container");
                auto index = uniform_int(0, static_cast<core::i32>(options.size() - 1));
                return options[index];
            }

            std::vector<core::usize> permutation(core::usize n) {
                std::vector<core::usize> result(n);
                std::iota(result.begin(), result.end(), 0);
                shuffle(result);
                return result;
            }

            // Generator information
            PSI_NODISCARD GeneratorType generator_type() const {
                return generator_->type();
            }

            PSI_NODISCARD std::string generator_name() const {
                return generator_->name();
            }

        private:
            std::unique_ptr<RandomGenerator> generator_;

            std::unique_ptr<RandomGenerator> create_generator(GeneratorType type, core::u64 seed) {
                switch (type) {
                case GeneratorType::MersenneTwister:
                    return std::make_unique<MersenneTwisterGenerator>(seed);
                case GeneratorType::XORShift:
                    return std::make_unique<XORShiftGenerator>(seed);
                default:
                    return std::make_unique<MersenneTwisterGenerator>(seed);
                }
            }
        };

        // Global random instance
        inline Random& get_global_random() {
            static Random global_random;
            return global_random;
        }

        // Convenience functions using global random instance
        inline void set_seed(core::u64 seed) {
            get_global_random().seed(seed);
        }

        inline void random_seed() {
            get_global_random().random_seed();
        }

        template<typename T>
        PSI_NODISCARD Vector<T> randn(core::usize size, T mean = T{ 0 }, T stddev = T{ 1 },
            core::device_id_t device_id = 0) {
            return get_global_random().normal_vector<T>(size, mean, stddev, device_id);
        }

        template<typename T>
        PSI_NODISCARD Matrix<T> randn(core::usize rows, core::usize cols,
            T mean = T{ 0 }, T stddev = T{ 1 },
            core::device_id_t device_id = 0) {
            return get_global_random().normal_matrix<T>(rows, cols, mean, stddev, device_id);
        }

        template<typename T>
        PSI_NODISCARD Vector<T> rand(core::usize size, T min_val = T{ 0 }, T max_val = T{ 1 },
            core::device_id_t device_id = 0) {
            return get_global_random().uniform_vector<T>(size, min_val, max_val, device_id);
        }

        template<typename T>
        PSI_NODISCARD Matrix<T> rand(core::usize rows, core::usize cols,
            T min_val = T{ 0 }, T max_val = T{ 1 },
            core::device_id_t device_id = 0) {
            return get_global_random().uniform_matrix<T>(rows, cols, min_val, max_val, device_id);
        }

    } // namespace math
} // namespace psi