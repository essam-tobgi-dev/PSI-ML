#pragma once

#include "../core/types.h"
#include <chrono>
#include <string>

namespace psi {
    namespace utils {

        // High-resolution timer for performance measurement
        class Timer {
        public:
            using Clock = std::chrono::high_resolution_clock;
            using TimePoint = Clock::time_point;
            using Duration = std::chrono::duration<double>;

            Timer() : running_(false), elapsed_(0.0) {}

            // Start the timer
            void start() {
                if (!running_) {
                    start_time_ = Clock::now();
                    running_ = true;
                }
            }

            // Stop the timer and accumulate elapsed time
            void stop() {
                if (running_) {
                    auto end_time = Clock::now();
                    elapsed_ += std::chrono::duration_cast<Duration>(end_time - start_time_).count();
                    running_ = false;
                }
            }

            // Reset the timer
            void reset() {
                running_ = false;
                elapsed_ = 0.0;
            }

            // Restart the timer (reset and start)
            void restart() {
                reset();
                start();
            }

            // Get elapsed time in seconds
            double elapsed_seconds() const {
                double total = elapsed_;
                if (running_) {
                    auto now = Clock::now();
                    total += std::chrono::duration_cast<Duration>(now - start_time_).count();
                }
                return total;
            }

            // Get elapsed time in milliseconds
            double elapsed_ms() const {
                return elapsed_seconds() * 1000.0;
            }

            // Get elapsed time in microseconds
            double elapsed_us() const {
                return elapsed_seconds() * 1000000.0;
            }

            // Get elapsed time in nanoseconds
            double elapsed_ns() const {
                return elapsed_seconds() * 1000000000.0;
            }

            // Check if timer is running
            bool is_running() const {
                return running_;
            }

        private:
            TimePoint start_time_;
            bool running_;
            double elapsed_;  // Accumulated time in seconds
        };

        // RAII-style scoped timer
        class ScopedTimer {
        public:
            explicit ScopedTimer(Timer& timer) : timer_(timer) {
                timer_.start();
            }

            ~ScopedTimer() {
                timer_.stop();
            }

            // Non-copyable
            ScopedTimer(const ScopedTimer&) = delete;
            ScopedTimer& operator=(const ScopedTimer&) = delete;

        private:
            Timer& timer_;
        };

    } // namespace utils
} // namespace psi
