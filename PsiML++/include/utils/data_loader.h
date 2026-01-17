#pragma once

#include "../core/types.h"
#include "../core/config.h"
#include "../core/exception.h"
#include "../math/vector.h"
#include "../math/matrix.h"
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <stdexcept>

namespace psi {
    namespace utils {

        // Result structure for loaded data
        template<typename T>
        struct LoadedData {
            math::Matrix<T> X;              // Feature matrix
            math::Vector<T> y;              // Target vector
            std::vector<std::string> feature_names;
            std::string target_name;
            core::usize n_samples;
            core::usize n_features;
        };

        // CSV parsing options
        struct CSVOptions {
            char delimiter = ',';
            bool has_header = true;
            core::usize target_column = static_cast<core::usize>(-1);  // Last column by default
            bool skip_empty_lines = true;
            char comment_char = '#';
            std::vector<core::usize> columns_to_skip;
        };

        // Data loader class for loading datasets from various formats
        template<typename T>
        class DataLoader {
        public:
            DataLoader() = default;

            // Load CSV file into a Matrix
            static math::Matrix<T> load_csv(
                const std::string& filename,
                const CSVOptions& options = CSVOptions()) {

                std::ifstream file(filename);
                if (!file.is_open()) {
                    throw std::runtime_error("Cannot open file: " + filename);
                }

                std::vector<std::vector<T>> data;
                std::string line;
                core::usize line_number = 0;
                core::usize expected_cols = 0;

                while (std::getline(file, line)) {
                    line_number++;

                    // Skip empty lines
                    if (options.skip_empty_lines && is_empty_or_whitespace(line)) {
                        continue;
                    }

                    // Skip comments
                    if (!line.empty() && line[0] == options.comment_char) {
                        continue;
                    }

                    // Skip header
                    if (options.has_header && data.empty()) {
                        if (expected_cols == 0) {
                            expected_cols = count_columns(line, options.delimiter);
                        }
                        continue;
                    }

                    std::vector<T> row = parse_row(line, options.delimiter);

                    if (expected_cols == 0) {
                        expected_cols = row.size();
                    } else if (row.size() != expected_cols) {
                        throw std::runtime_error(
                            "Line " + std::to_string(line_number) +
                            ": expected " + std::to_string(expected_cols) +
                            " columns, got " + std::to_string(row.size()));
                    }

                    data.push_back(row);
                }

                file.close();

                if (data.empty()) {
                    return math::Matrix<T>();
                }

                // Convert to Matrix
                core::usize n_rows = data.size();
                core::usize n_cols = data[0].size();
                math::Matrix<T> result(n_rows, n_cols);

                for (core::usize i = 0; i < n_rows; ++i) {
                    for (core::usize j = 0; j < n_cols; ++j) {
                        result(i, j) = data[i][j];
                    }
                }

                return result;
            }

            // Load CSV file with separate features (X) and target (y)
            static LoadedData<T> load_csv_with_target(
                const std::string& filename,
                const CSVOptions& options = CSVOptions()) {

                std::ifstream file(filename);
                if (!file.is_open()) {
                    throw std::runtime_error("Cannot open file: " + filename);
                }

                LoadedData<T> result;
                std::vector<std::vector<T>> feature_data;
                std::vector<T> target_data;
                std::string line;
                core::usize line_number = 0;
                core::usize expected_cols = 0;
                bool header_read = false;

                while (std::getline(file, line)) {
                    line_number++;

                    if (options.skip_empty_lines && is_empty_or_whitespace(line)) {
                        continue;
                    }

                    if (!line.empty() && line[0] == options.comment_char) {
                        continue;
                    }

                    // Read header
                    if (options.has_header && !header_read) {
                        result.feature_names = parse_header(line, options.delimiter);
                        expected_cols = result.feature_names.size();

                        core::usize target_col = (options.target_column == static_cast<core::usize>(-1))
                            ? expected_cols - 1 : options.target_column;

                        if (target_col < result.feature_names.size()) {
                            result.target_name = result.feature_names[target_col];
                            result.feature_names.erase(result.feature_names.begin() + target_col);
                        }

                        header_read = true;
                        continue;
                    }

                    std::vector<T> row = parse_row(line, options.delimiter);

                    if (expected_cols == 0) {
                        expected_cols = row.size();
                    }

                    core::usize target_col = (options.target_column == static_cast<core::usize>(-1))
                        ? row.size() - 1 : options.target_column;

                    // Extract target
                    T target_value = row[target_col];
                    target_data.push_back(target_value);

                    // Extract features (excluding target column)
                    std::vector<T> features;
                    for (core::usize j = 0; j < row.size(); ++j) {
                        if (j != target_col) {
                            features.push_back(row[j]);
                        }
                    }
                    feature_data.push_back(features);
                }

                file.close();

                if (feature_data.empty()) {
                    result.n_samples = 0;
                    result.n_features = 0;
                    return result;
                }

                // Convert to Matrix and Vector
                result.n_samples = feature_data.size();
                result.n_features = feature_data[0].size();
                result.X = math::Matrix<T>(result.n_samples, result.n_features);
                result.y = math::Vector<T>(result.n_samples);

                for (core::usize i = 0; i < result.n_samples; ++i) {
                    for (core::usize j = 0; j < result.n_features; ++j) {
                        result.X(i, j) = feature_data[i][j];
                    }
                    result.y[i] = target_data[i];
                }

                return result;
            }

            // Save Matrix to CSV file
            static void save_csv(
                const std::string& filename,
                const math::Matrix<T>& data,
                const std::vector<std::string>& column_names = {},
                char delimiter = ',') {

                std::ofstream file(filename);
                if (!file.is_open()) {
                    throw std::runtime_error("Cannot open file for writing: " + filename);
                }

                // Write header if column names provided
                if (!column_names.empty()) {
                    for (core::usize j = 0; j < column_names.size(); ++j) {
                        if (j > 0) file << delimiter;
                        file << column_names[j];
                    }
                    file << "\n";
                }

                // Write data
                for (core::usize i = 0; i < data.rows(); ++i) {
                    for (core::usize j = 0; j < data.cols(); ++j) {
                        if (j > 0) file << delimiter;
                        file << data(i, j);
                    }
                    file << "\n";
                }

                file.close();
            }

            // Save Matrix and target to CSV file
            static void save_csv_with_target(
                const std::string& filename,
                const math::Matrix<T>& X,
                const math::Vector<T>& y,
                const std::vector<std::string>& feature_names = {},
                const std::string& target_name = "target",
                char delimiter = ',') {

                PSI_CHECK_DIMENSIONS("save_csv_with_target", X.rows(), y.size());

                std::ofstream file(filename);
                if (!file.is_open()) {
                    throw std::runtime_error("Cannot open file for writing: " + filename);
                }

                // Write header
                if (!feature_names.empty()) {
                    for (core::usize j = 0; j < feature_names.size(); ++j) {
                        file << feature_names[j] << delimiter;
                    }
                } else {
                    for (core::usize j = 0; j < X.cols(); ++j) {
                        file << "feature_" << j << delimiter;
                    }
                }
                file << target_name << "\n";

                // Write data
                for (core::usize i = 0; i < X.rows(); ++i) {
                    for (core::usize j = 0; j < X.cols(); ++j) {
                        file << X(i, j) << delimiter;
                    }
                    file << y[i] << "\n";
                }

                file.close();
            }

            // Load Vector from text file (one value per line)
            static math::Vector<T> load_vector(const std::string& filename) {
                std::ifstream file(filename);
                if (!file.is_open()) {
                    throw std::runtime_error("Cannot open file: " + filename);
                }

                std::vector<T> data;
                T value;
                while (file >> value) {
                    data.push_back(value);
                }
                file.close();

                math::Vector<T> result(data.size());
                for (core::usize i = 0; i < data.size(); ++i) {
                    result[i] = data[i];
                }
                return result;
            }

            // Save Vector to text file
            static void save_vector(
                const std::string& filename,
                const math::Vector<T>& data) {

                std::ofstream file(filename);
                if (!file.is_open()) {
                    throw std::runtime_error("Cannot open file for writing: " + filename);
                }

                for (core::usize i = 0; i < data.size(); ++i) {
                    file << data[i] << "\n";
                }
                file.close();
            }

            // Create synthetic regression dataset
            static LoadedData<T> make_regression(
                core::usize n_samples = 100,
                core::usize n_features = 1,
                T noise = T{0.1},
                core::u64 seed = 42) {

                LoadedData<T> result;
                result.n_samples = n_samples;
                result.n_features = n_features;
                result.X = math::Matrix<T>(n_samples, n_features);
                result.y = math::Vector<T>(n_samples);

                // Simple linear random number generator
                core::u64 state = seed;
                auto next_random = [&state]() -> T {
                    state = state * 6364136223846793005ULL + 1442695040888963407ULL;
                    return static_cast<T>((state >> 33) & 0x7FFFFFFF) / static_cast<T>(0x7FFFFFFF);
                };

                // Generate random coefficients
                std::vector<T> coefficients(n_features);
                for (core::usize j = 0; j < n_features; ++j) {
                    coefficients[j] = (next_random() - T{0.5}) * T{4};  // Range [-2, 2]
                }
                T intercept = (next_random() - T{0.5}) * T{10};  // Range [-5, 5]

                // Generate data
                for (core::usize i = 0; i < n_samples; ++i) {
                    T y_val = intercept;
                    for (core::usize j = 0; j < n_features; ++j) {
                        T x_val = next_random() * T{10};  // Range [0, 10]
                        result.X(i, j) = x_val;
                        y_val += coefficients[j] * x_val;
                    }
                    // Add noise
                    T noise_val = (next_random() - T{0.5}) * T{2} * noise;
                    result.y[i] = y_val + noise_val;
                }

                // Generate feature names
                for (core::usize j = 0; j < n_features; ++j) {
                    result.feature_names.push_back("feature_" + std::to_string(j));
                }
                result.target_name = "target";

                return result;
            }

            // Create synthetic classification dataset
            static LoadedData<T> make_classification(
                core::usize n_samples = 100,
                core::usize n_features = 2,
                core::usize n_classes = 2,
                T separation = T{2.0},
                core::u64 seed = 42) {

                LoadedData<T> result;
                result.n_samples = n_samples;
                result.n_features = n_features;
                result.X = math::Matrix<T>(n_samples, n_features);
                result.y = math::Vector<T>(n_samples);

                core::u64 state = seed;
                auto next_random = [&state]() -> T {
                    state = state * 6364136223846793005ULL + 1442695040888963407ULL;
                    return static_cast<T>((state >> 33) & 0x7FFFFFFF) / static_cast<T>(0x7FFFFFFF);
                };

                // Generate cluster centers
                std::vector<std::vector<T>> centers(n_classes);
                for (core::usize c = 0; c < n_classes; ++c) {
                    centers[c].resize(n_features);
                    for (core::usize j = 0; j < n_features; ++j) {
                        centers[c][j] = static_cast<T>(c) * separation + (next_random() - T{0.5});
                    }
                }

                // Generate samples
                core::usize samples_per_class = n_samples / n_classes;
                for (core::usize i = 0; i < n_samples; ++i) {
                    core::usize class_idx = i / samples_per_class;
                    if (class_idx >= n_classes) class_idx = n_classes - 1;

                    result.y[i] = static_cast<T>(class_idx);

                    for (core::usize j = 0; j < n_features; ++j) {
                        // Add Gaussian-like noise using Box-Muller approximation
                        T u1 = next_random();
                        T u2 = next_random();
                        T noise_val = std::sqrt(T{-2} * std::log(u1 + T{1e-10})) *
                                     std::cos(T{2} * T{3.14159265358979} * u2);
                        result.X(i, j) = centers[class_idx][j] + noise_val * T{0.5};
                    }
                }

                // Generate feature names
                for (core::usize j = 0; j < n_features; ++j) {
                    result.feature_names.push_back("feature_" + std::to_string(j));
                }
                result.target_name = "class";

                return result;
            }

            // Create synthetic clustering dataset (blobs)
            static LoadedData<T> make_blobs(
                core::usize n_samples = 100,
                core::usize n_features = 2,
                core::usize n_centers = 3,
                T cluster_std = T{1.0},
                core::u64 seed = 42) {

                return make_classification(n_samples, n_features, n_centers, T{5.0}, seed);
            }

        private:
            static bool is_empty_or_whitespace(const std::string& str) {
                return str.empty() ||
                       std::all_of(str.begin(), str.end(), [](char c) { return std::isspace(c); });
            }

            static core::usize count_columns(const std::string& line, char delimiter) {
                if (line.empty()) return 0;
                return static_cast<core::usize>(std::count(line.begin(), line.end(), delimiter)) + 1;
            }

            static std::vector<T> parse_row(const std::string& line, char delimiter) {
                std::vector<T> row;
                std::stringstream ss(line);
                std::string cell;

                while (std::getline(ss, cell, delimiter)) {
                    // Trim whitespace
                    cell.erase(0, cell.find_first_not_of(" \t\r\n"));
                    cell.erase(cell.find_last_not_of(" \t\r\n") + 1);

                    if (cell.empty()) {
                        row.push_back(T{0});
                    } else {
                        try {
                            if constexpr (std::is_same_v<T, float>) {
                                row.push_back(std::stof(cell));
                            } else if constexpr (std::is_same_v<T, double>) {
                                row.push_back(std::stod(cell));
                            } else if constexpr (std::is_integral_v<T>) {
                                row.push_back(static_cast<T>(std::stoll(cell)));
                            } else {
                                row.push_back(static_cast<T>(std::stod(cell)));
                            }
                        } catch (...) {
                            row.push_back(T{0});
                        }
                    }
                }

                return row;
            }

            static std::vector<std::string> parse_header(const std::string& line, char delimiter) {
                std::vector<std::string> headers;
                std::stringstream ss(line);
                std::string cell;

                while (std::getline(ss, cell, delimiter)) {
                    cell.erase(0, cell.find_first_not_of(" \t\r\n"));
                    cell.erase(cell.find_last_not_of(" \t\r\n") + 1);
                    headers.push_back(cell);
                }

                return headers;
            }
        };

    } // namespace utils
} // namespace psi
