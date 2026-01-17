#pragma once

#include "../core/types.h"
#include "../core/config.h"
#include "../core/exception.h"
#include "../math/vector.h"
#include "../math/matrix.h"
#include "string_utils.h"
#include <string>
#include <fstream>
#include <sstream>
#include <map>
#include <stdexcept>
#include <cstring>

namespace psi {
    namespace utils {

        // Model metadata structure
        struct ModelMetadata {
            std::string model_type;
            std::string version = "1.0";
            std::map<std::string, std::string> params;
            core::usize n_features = 0;
        };

        // Model I/O class for saving and loading models
        template<typename T>
        class ModelIO {
        public:
            // Magic number for binary format identification
            static constexpr core::u32 MAGIC_NUMBER = 0x50534D4C;  // "PSML"
            static constexpr core::u32 FORMAT_VERSION = 1;

            // =========================================================================
            // Text format (human-readable)
            // =========================================================================

            // Save model weights to text file
            static void save_weights_text(
                const std::string& filename,
                const math::Vector<T>& weights,
                const ModelMetadata& metadata = ModelMetadata()) {

                std::ofstream file(filename);
                if (!file.is_open()) {
                    throw std::runtime_error("Cannot open file for writing: " + filename);
                }

                // Write metadata
                file << "# PsiML++ Model File (Text Format)\n";
                file << "# Version: " << metadata.version << "\n";
                if (!metadata.model_type.empty()) {
                    file << "# Model: " << metadata.model_type << "\n";
                }
                for (const auto& [key, value] : metadata.params) {
                    file << "# " << key << ": " << value << "\n";
                }
                file << "\n";

                // Write weights
                file << "[weights]\n";
                file << "size: " << weights.size() << "\n";
                for (core::usize i = 0; i < weights.size(); ++i) {
                    file << weights[i] << "\n";
                }

                file.close();
            }

            // Load model weights from text file
            static math::Vector<T> load_weights_text(
                const std::string& filename,
                ModelMetadata* metadata = nullptr) {

                std::ifstream file(filename);
                if (!file.is_open()) {
                    throw std::runtime_error("Cannot open file: " + filename);
                }

                std::string line;
                bool in_weights_section = false;
                core::usize expected_size = 0;
                std::vector<T> weights_data;

                ModelMetadata meta;

                while (std::getline(file, line)) {
                    line = StringUtils::trim(line);

                    // Skip empty lines
                    if (line.empty()) continue;

                    // Parse comments/metadata
                    if (line[0] == '#') {
                        if (StringUtils::contains(line, "Model:")) {
                            core::usize pos = line.find(':');
                            if (pos != std::string::npos) {
                                meta.model_type = StringUtils::trim(line.substr(pos + 1));
                            }
                        } else if (StringUtils::contains(line, "Version:")) {
                            core::usize pos = line.find(':');
                            if (pos != std::string::npos) {
                                meta.version = StringUtils::trim(line.substr(pos + 1));
                            }
                        }
                        continue;
                    }

                    // Section header
                    if (line == "[weights]") {
                        in_weights_section = true;
                        continue;
                    }

                    // Parse size
                    if (in_weights_section && StringUtils::starts_with(line, "size:")) {
                        expected_size = static_cast<core::usize>(
                            std::stoul(StringUtils::trim(line.substr(5))));
                        continue;
                    }

                    // Parse weight values
                    if (in_weights_section && StringUtils::is_numeric(line)) {
                        if constexpr (std::is_same_v<T, float>) {
                            weights_data.push_back(std::stof(line));
                        } else {
                            weights_data.push_back(std::stod(line));
                        }
                    }
                }

                file.close();

                if (metadata) {
                    *metadata = meta;
                }

                // Convert to Vector
                math::Vector<T> result(weights_data.size());
                for (core::usize i = 0; i < weights_data.size(); ++i) {
                    result[i] = weights_data[i];
                }

                return result;
            }

            // Save matrix to text file
            static void save_matrix_text(
                const std::string& filename,
                const math::Matrix<T>& matrix,
                const std::string& name = "matrix") {

                std::ofstream file(filename);
                if (!file.is_open()) {
                    throw std::runtime_error("Cannot open file for writing: " + filename);
                }

                file << "# PsiML++ Matrix File\n";
                file << "[" << name << "]\n";
                file << "rows: " << matrix.rows() << "\n";
                file << "cols: " << matrix.cols() << "\n";
                file << "data:\n";

                for (core::usize i = 0; i < matrix.rows(); ++i) {
                    for (core::usize j = 0; j < matrix.cols(); ++j) {
                        if (j > 0) file << " ";
                        file << matrix(i, j);
                    }
                    file << "\n";
                }

                file.close();
            }

            // Load matrix from text file
            static math::Matrix<T> load_matrix_text(const std::string& filename) {
                std::ifstream file(filename);
                if (!file.is_open()) {
                    throw std::runtime_error("Cannot open file: " + filename);
                }

                std::string line;
                core::usize rows = 0, cols = 0;
                bool in_data_section = false;
                std::vector<std::vector<T>> data;

                while (std::getline(file, line)) {
                    line = StringUtils::trim(line);

                    if (line.empty() || line[0] == '#') continue;
                    if (line[0] == '[') continue;

                    if (StringUtils::starts_with(line, "rows:")) {
                        rows = static_cast<core::usize>(std::stoul(StringUtils::trim(line.substr(5))));
                        continue;
                    }
                    if (StringUtils::starts_with(line, "cols:")) {
                        cols = static_cast<core::usize>(std::stoul(StringUtils::trim(line.substr(5))));
                        continue;
                    }
                    if (line == "data:") {
                        in_data_section = true;
                        continue;
                    }

                    if (in_data_section) {
                        std::vector<std::string> parts = StringUtils::split(line, " \t");
                        std::vector<T> row;
                        for (const auto& part : parts) {
                            if (!part.empty()) {
                                if constexpr (std::is_same_v<T, float>) {
                                    row.push_back(std::stof(part));
                                } else {
                                    row.push_back(std::stod(part));
                                }
                            }
                        }
                        if (!row.empty()) {
                            data.push_back(row);
                        }
                    }
                }

                file.close();

                if (data.empty()) {
                    return math::Matrix<T>();
                }

                rows = data.size();
                cols = data[0].size();
                math::Matrix<T> result(rows, cols);

                for (core::usize i = 0; i < rows; ++i) {
                    for (core::usize j = 0; j < data[i].size() && j < cols; ++j) {
                        result(i, j) = data[i][j];
                    }
                }

                return result;
            }

            // =========================================================================
            // Binary format (efficient)
            // =========================================================================

            // Save model weights to binary file
            static void save_weights_binary(
                const std::string& filename,
                const math::Vector<T>& weights,
                const ModelMetadata& metadata = ModelMetadata()) {

                std::ofstream file(filename, std::ios::binary);
                if (!file.is_open()) {
                    throw std::runtime_error("Cannot open file for writing: " + filename);
                }

                // Write header
                write_binary(file, MAGIC_NUMBER);
                write_binary(file, FORMAT_VERSION);

                // Write metadata
                write_string_binary(file, metadata.model_type);
                write_string_binary(file, metadata.version);
                write_binary(file, static_cast<core::u32>(metadata.params.size()));
                for (const auto& [key, value] : metadata.params) {
                    write_string_binary(file, key);
                    write_string_binary(file, value);
                }

                // Write weights
                core::u64 size = weights.size();
                write_binary(file, size);
                for (core::usize i = 0; i < weights.size(); ++i) {
                    write_binary(file, weights[i]);
                }

                file.close();
            }

            // Load model weights from binary file
            static math::Vector<T> load_weights_binary(
                const std::string& filename,
                ModelMetadata* metadata = nullptr) {

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

                // Read metadata
                ModelMetadata meta;
                meta.model_type = read_string_binary(file);
                meta.version = read_string_binary(file);

                core::u32 n_params = read_binary<core::u32>(file);
                for (core::u32 i = 0; i < n_params; ++i) {
                    std::string key = read_string_binary(file);
                    std::string value = read_string_binary(file);
                    meta.params[key] = value;
                }

                // Read weights
                core::u64 size = read_binary<core::u64>(file);
                math::Vector<T> weights(static_cast<core::usize>(size));
                for (core::usize i = 0; i < size; ++i) {
                    weights[i] = read_binary<T>(file);
                }

                file.close();

                if (metadata) {
                    *metadata = meta;
                }

                return weights;
            }

            // Save matrix to binary file
            static void save_matrix_binary(
                const std::string& filename,
                const math::Matrix<T>& matrix) {

                std::ofstream file(filename, std::ios::binary);
                if (!file.is_open()) {
                    throw std::runtime_error("Cannot open file for writing: " + filename);
                }

                // Write header
                write_binary(file, MAGIC_NUMBER);
                write_binary(file, FORMAT_VERSION);

                // Write dimensions
                core::u64 rows = matrix.rows();
                core::u64 cols = matrix.cols();
                write_binary(file, rows);
                write_binary(file, cols);

                // Write data (row-major)
                for (core::usize i = 0; i < matrix.rows(); ++i) {
                    for (core::usize j = 0; j < matrix.cols(); ++j) {
                        write_binary(file, matrix(i, j));
                    }
                }

                file.close();
            }

            // Load matrix from binary file
            static math::Matrix<T> load_matrix_binary(const std::string& filename) {
                std::ifstream file(filename, std::ios::binary);
                if (!file.is_open()) {
                    throw std::runtime_error("Cannot open file: " + filename);
                }

                // Verify header
                core::u32 magic = read_binary<core::u32>(file);
                if (magic != MAGIC_NUMBER) {
                    throw std::runtime_error("Invalid file format");
                }

                core::u32 version = read_binary<core::u32>(file);
                if (version != FORMAT_VERSION) {
                    throw std::runtime_error("Unsupported format version");
                }

                // Read dimensions
                core::u64 rows = read_binary<core::u64>(file);
                core::u64 cols = read_binary<core::u64>(file);

                // Read data
                math::Matrix<T> matrix(static_cast<core::usize>(rows),
                                       static_cast<core::usize>(cols));
                for (core::usize i = 0; i < rows; ++i) {
                    for (core::usize j = 0; j < cols; ++j) {
                        matrix(i, j) = read_binary<T>(file);
                    }
                }

                file.close();
                return matrix;
            }

            // =========================================================================
            // Complete model save/load (weights + scaler parameters)
            // =========================================================================

            // Save complete model state
            static void save_model(
                const std::string& filename,
                const math::Vector<T>& weights,
                const math::Vector<T>& scaler_mean,
                const math::Vector<T>& scaler_std,
                const ModelMetadata& metadata) {

                std::ofstream file(filename, std::ios::binary);
                if (!file.is_open()) {
                    throw std::runtime_error("Cannot open file for writing: " + filename);
                }

                // Header
                write_binary(file, MAGIC_NUMBER);
                write_binary(file, FORMAT_VERSION);

                // Metadata
                write_string_binary(file, metadata.model_type);
                write_string_binary(file, metadata.version);
                write_binary(file, static_cast<core::u32>(metadata.params.size()));
                for (const auto& [key, value] : metadata.params) {
                    write_string_binary(file, key);
                    write_string_binary(file, value);
                }

                // Weights
                write_vector_binary(file, weights);

                // Scaler mean
                write_vector_binary(file, scaler_mean);

                // Scaler std
                write_vector_binary(file, scaler_std);

                file.close();
            }

            // Load complete model state
            static void load_model(
                const std::string& filename,
                math::Vector<T>& weights,
                math::Vector<T>& scaler_mean,
                math::Vector<T>& scaler_std,
                ModelMetadata& metadata) {

                std::ifstream file(filename, std::ios::binary);
                if (!file.is_open()) {
                    throw std::runtime_error("Cannot open file: " + filename);
                }

                // Verify header
                core::u32 magic = read_binary<core::u32>(file);
                if (magic != MAGIC_NUMBER) {
                    throw std::runtime_error("Invalid file format");
                }

                core::u32 version = read_binary<core::u32>(file);
                if (version != FORMAT_VERSION) {
                    throw std::runtime_error("Unsupported format version");
                }

                // Metadata
                metadata.model_type = read_string_binary(file);
                metadata.version = read_string_binary(file);
                core::u32 n_params = read_binary<core::u32>(file);
                for (core::u32 i = 0; i < n_params; ++i) {
                    std::string key = read_string_binary(file);
                    std::string value = read_string_binary(file);
                    metadata.params[key] = value;
                }

                // Weights
                weights = read_vector_binary(file);

                // Scaler mean
                scaler_mean = read_vector_binary(file);

                // Scaler std
                scaler_std = read_vector_binary(file);

                file.close();
            }

        private:
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

            static void write_vector_binary(std::ofstream& file, const math::Vector<T>& vec) {
                core::u64 size = vec.size();
                write_binary(file, size);
                for (core::usize i = 0; i < vec.size(); ++i) {
                    write_binary(file, vec[i]);
                }
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

            static math::Vector<T> read_vector_binary(std::ifstream& file) {
                core::u64 size = read_binary<core::u64>(file);
                math::Vector<T> vec(static_cast<core::usize>(size));
                for (core::usize i = 0; i < size; ++i) {
                    vec[i] = read_binary<T>(file);
                }
                return vec;
            }
        };

    } // namespace utils
} // namespace psi
