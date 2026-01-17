#pragma once

#include "../core/types.h"
#include "timer.h"
#include "string_utils.h"
#include <string>
#include <map>
#include <vector>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <stdexcept>
#include <cstring>

namespace psi {
    namespace utils {

        // Statistics for a single profiled section
        struct ProfileEntry {
            std::string name;
            core::u64 call_count = 0;
            double total_time = 0.0;      // Total time in seconds
            double min_time = 0.0;        // Minimum call time
            double max_time = 0.0;        // Maximum call time

            double average_time() const {
                return call_count > 0 ? total_time / static_cast<double>(call_count) : 0.0;
            }
        };

        // Profiler for tracking multiple timed sections with serialization support
        class Profiler {
        public:
            // Magic number for binary format identification
            static constexpr core::u32 MAGIC_NUMBER = 0x50524F46;  // "PROF"
            static constexpr core::u32 FORMAT_VERSION = 1;

            Profiler() = default;

            // Start timing a section
            void start(const std::string& name) {
                auto& timer = timers_[name];
                timer.start();
            }

            // Stop timing a section and record statistics
            void stop(const std::string& name) {
                auto it = timers_.find(name);
                if (it == timers_.end()) {
                    return;
                }

                auto& timer = it->second;
                double elapsed = timer.elapsed_seconds();
                timer.stop();
                double call_time = timer.elapsed_seconds() - (elapsed - timer.elapsed_seconds());

                // For proper call time measurement, we need to track the time at stop
                timer.stop();
                call_time = timer.elapsed_seconds();

                // Reset for next call measurement
                auto& entry = entries_[name];
                entry.name = name;

                // First call
                if (entry.call_count == 0) {
                    entry.min_time = call_time;
                    entry.max_time = call_time;
                } else {
                    // Calculate incremental time for this call
                    double prev_total = entry.total_time;
                    double this_call_time = call_time - prev_total;
                    if (this_call_time < entry.min_time) entry.min_time = this_call_time;
                    if (this_call_time > entry.max_time) entry.max_time = this_call_time;
                }

                entry.total_time = call_time;
                entry.call_count++;
            }

            // Start timing with automatic stop (returns RAII guard)
            class ScopedProfile {
            public:
                ScopedProfile(Profiler& profiler, const std::string& name)
                    : profiler_(profiler), name_(name) {
                    profiler_.start(name_);
                }
                ~ScopedProfile() {
                    profiler_.stop(name_);
                }
                ScopedProfile(const ScopedProfile&) = delete;
                ScopedProfile& operator=(const ScopedProfile&) = delete;
            private:
                Profiler& profiler_;
                std::string name_;
            };

            // Record a single timing manually (useful when you already have the time)
            void record(const std::string& name, double time_seconds) {
                auto& entry = entries_[name];
                entry.name = name;

                if (entry.call_count == 0) {
                    entry.min_time = time_seconds;
                    entry.max_time = time_seconds;
                } else {
                    if (time_seconds < entry.min_time) entry.min_time = time_seconds;
                    if (time_seconds > entry.max_time) entry.max_time = time_seconds;
                }

                entry.total_time += time_seconds;
                entry.call_count++;
            }

            // Get entry for a section
            const ProfileEntry* get_entry(const std::string& name) const {
                auto it = entries_.find(name);
                return it != entries_.end() ? &it->second : nullptr;
            }

            // Get all entries
            std::vector<ProfileEntry> get_all_entries() const {
                std::vector<ProfileEntry> result;
                result.reserve(entries_.size());
                for (const auto& [name, entry] : entries_) {
                    result.push_back(entry);
                }
                return result;
            }

            // Get entries sorted by total time (descending)
            std::vector<ProfileEntry> get_entries_sorted_by_time() const {
                auto result = get_all_entries();
                std::sort(result.begin(), result.end(),
                    [](const ProfileEntry& a, const ProfileEntry& b) {
                        return a.total_time > b.total_time;
                    });
                return result;
            }

            // Get total profiled time
            double get_total_time() const {
                double total = 0.0;
                for (const auto& [name, entry] : entries_) {
                    total += entry.total_time;
                }
                return total;
            }

            // Clear all profiling data
            void clear() {
                timers_.clear();
                entries_.clear();
            }

            // Get number of profiled sections
            core::usize size() const {
                return entries_.size();
            }

            // Check if empty
            bool empty() const {
                return entries_.empty();
            }

            // =========================================================================
            // Text format serialization (human-readable)
            // =========================================================================

            // Save profiling data to text file
            void save_text(const std::string& filename) const {
                std::ofstream file(filename);
                if (!file.is_open()) {
                    throw std::runtime_error("Cannot open file for writing: " + filename);
                }

                file << "# PsiML++ Profiler Data (Text Format)\n";
                file << "# Version: " << FORMAT_VERSION << "\n";
                file << "\n";

                file << "[summary]\n";
                file << "total_sections: " << entries_.size() << "\n";
                file << "total_time: " << std::fixed << std::setprecision(9) << get_total_time() << "\n";
                file << "\n";

                file << "[entries]\n";
                for (const auto& [name, entry] : entries_) {
                    file << "name: " << name << "\n";
                    file << "call_count: " << entry.call_count << "\n";
                    file << "total_time: " << std::fixed << std::setprecision(9) << entry.total_time << "\n";
                    file << "min_time: " << std::fixed << std::setprecision(9) << entry.min_time << "\n";
                    file << "max_time: " << std::fixed << std::setprecision(9) << entry.max_time << "\n";
                    file << "---\n";
                }

                file.close();
            }

            // Load profiling data from text file
            void load_text(const std::string& filename) {
                std::ifstream file(filename);
                if (!file.is_open()) {
                    throw std::runtime_error("Cannot open file: " + filename);
                }

                clear();

                std::string line;
                bool in_entries_section = false;
                ProfileEntry current_entry;
                bool has_entry = false;

                while (std::getline(file, line)) {
                    line = StringUtils::trim(line);

                    if (line.empty() || line[0] == '#') continue;

                    if (line == "[entries]") {
                        in_entries_section = true;
                        continue;
                    }

                    if (line == "[summary]") {
                        in_entries_section = false;
                        continue;
                    }

                    if (in_entries_section) {
                        if (line == "---") {
                            if (has_entry && !current_entry.name.empty()) {
                                entries_[current_entry.name] = current_entry;
                            }
                            current_entry = ProfileEntry();
                            has_entry = false;
                            continue;
                        }

                        if (StringUtils::starts_with(line, "name:")) {
                            current_entry.name = StringUtils::trim(line.substr(5));
                            has_entry = true;
                        } else if (StringUtils::starts_with(line, "call_count:")) {
                            current_entry.call_count = static_cast<core::u64>(
                                std::stoull(StringUtils::trim(line.substr(11))));
                        } else if (StringUtils::starts_with(line, "total_time:")) {
                            current_entry.total_time = std::stod(StringUtils::trim(line.substr(11)));
                        } else if (StringUtils::starts_with(line, "min_time:")) {
                            current_entry.min_time = std::stod(StringUtils::trim(line.substr(9)));
                        } else if (StringUtils::starts_with(line, "max_time:")) {
                            current_entry.max_time = std::stod(StringUtils::trim(line.substr(9)));
                        }
                    }
                }

                // Handle last entry if file doesn't end with ---
                if (has_entry && !current_entry.name.empty()) {
                    entries_[current_entry.name] = current_entry;
                }

                file.close();
            }

            // =========================================================================
            // Binary format serialization (efficient)
            // =========================================================================

            // Save profiling data to binary file
            void save_binary(const std::string& filename) const {
                std::ofstream file(filename, std::ios::binary);
                if (!file.is_open()) {
                    throw std::runtime_error("Cannot open file for writing: " + filename);
                }

                // Write header
                write_binary(file, MAGIC_NUMBER);
                write_binary(file, FORMAT_VERSION);

                // Write number of entries
                core::u64 num_entries = entries_.size();
                write_binary(file, num_entries);

                // Write each entry
                for (const auto& [name, entry] : entries_) {
                    write_string_binary(file, entry.name);
                    write_binary(file, entry.call_count);
                    write_binary(file, entry.total_time);
                    write_binary(file, entry.min_time);
                    write_binary(file, entry.max_time);
                }

                file.close();
            }

            // Load profiling data from binary file
            void load_binary(const std::string& filename) {
                std::ifstream file(filename, std::ios::binary);
                if (!file.is_open()) {
                    throw std::runtime_error("Cannot open file: " + filename);
                }

                // Read and verify header
                core::u32 magic = read_binary<core::u32>(file);
                if (magic != MAGIC_NUMBER) {
                    throw std::runtime_error("Invalid file format: wrong magic number");
                }

                core::u32 version = read_binary<core::u32>(file);
                if (version != FORMAT_VERSION) {
                    throw std::runtime_error("Unsupported format version: " + std::to_string(version));
                }

                clear();

                // Read number of entries
                core::u64 num_entries = read_binary<core::u64>(file);

                // Read each entry
                for (core::u64 i = 0; i < num_entries; ++i) {
                    ProfileEntry entry;
                    entry.name = read_string_binary(file);
                    entry.call_count = read_binary<core::u64>(file);
                    entry.total_time = read_binary<double>(file);
                    entry.min_time = read_binary<double>(file);
                    entry.max_time = read_binary<double>(file);
                    entries_[entry.name] = entry;
                }

                file.close();
            }

            // =========================================================================
            // Report generation
            // =========================================================================

            // Generate a human-readable report string
            std::string generate_report() const {
                std::ostringstream ss;

                ss << "========================================\n";
                ss << "         Profiler Report\n";
                ss << "========================================\n\n";

                if (entries_.empty()) {
                    ss << "No profiling data recorded.\n";
                    return ss.str();
                }

                auto sorted = get_entries_sorted_by_time();
                double total = get_total_time();

                ss << std::left << std::setw(30) << "Section"
                   << std::right << std::setw(12) << "Calls"
                   << std::setw(15) << "Total (ms)"
                   << std::setw(15) << "Avg (ms)"
                   << std::setw(10) << "%" << "\n";
                ss << std::string(82, '-') << "\n";

                for (const auto& entry : sorted) {
                    double percent = total > 0 ? (entry.total_time / total) * 100.0 : 0.0;
                    ss << std::left << std::setw(30) << entry.name
                       << std::right << std::setw(12) << entry.call_count
                       << std::setw(15) << std::fixed << std::setprecision(3) << (entry.total_time * 1000.0)
                       << std::setw(15) << std::fixed << std::setprecision(3) << (entry.average_time() * 1000.0)
                       << std::setw(9) << std::fixed << std::setprecision(1) << percent << "%\n";
                }

                ss << std::string(82, '-') << "\n";
                ss << std::left << std::setw(30) << "TOTAL"
                   << std::right << std::setw(12) << ""
                   << std::setw(15) << std::fixed << std::setprecision(3) << (total * 1000.0)
                   << std::setw(15) << ""
                   << std::setw(9) << "100.0%\n";

                return ss.str();
            }

        private:
            std::map<std::string, Timer> timers_;
            std::map<std::string, ProfileEntry> entries_;

            // Binary write helpers
            template<typename U>
            static void write_binary(std::ofstream& file, const U& value) {
                file.write(reinterpret_cast<const char*>(&value), sizeof(U));
            }

            static void write_string_binary(std::ofstream& file, const std::string& str) {
                core::u32 len = static_cast<core::u32>(str.size());
                write_binary(file, len);
                file.write(str.c_str(), len);
            }

            // Binary read helpers
            template<typename U>
            static U read_binary(std::ifstream& file) {
                U value;
                file.read(reinterpret_cast<char*>(&value), sizeof(U));
                return value;
            }

            static std::string read_string_binary(std::ifstream& file) {
                core::u32 len = read_binary<core::u32>(file);
                std::string str(len, '\0');
                file.read(&str[0], len);
                return str;
            }
        };

        // Macro for easy profiling
        #define PSI_PROFILE_SCOPE(profiler, name) \
            psi::utils::Profiler::ScopedProfile _psi_profile_##__LINE__(profiler, name)

    } // namespace utils
} // namespace psi
