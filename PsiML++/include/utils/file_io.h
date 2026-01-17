#pragma once

#include "../core/types.h"
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <cstdio>
#include <ctime>
#include <sys/stat.h>

#ifdef _WIN32
#include <direct.h>
#define PSI_MKDIR(path) _mkdir(path)
#else
#include <unistd.h>
#define PSI_MKDIR(path) mkdir(path, 0755)
#endif

namespace psi {
    namespace utils {

        // File I/O utility class
        class FileIO {
        public:
            // =========================================================================
            // File existence and info
            // =========================================================================

            // Check if a file exists
            static bool exists(const std::string& path) {
                struct stat buffer;
                return (stat(path.c_str(), &buffer) == 0);
            }

            // Check if path is a file (not directory)
            static bool is_file(const std::string& path) {
                struct stat buffer;
                if (stat(path.c_str(), &buffer) != 0) return false;
                return (buffer.st_mode & S_IFREG) != 0;
            }

            // Check if path is a directory
            static bool is_directory(const std::string& path) {
                struct stat buffer;
                if (stat(path.c_str(), &buffer) != 0) return false;
                return (buffer.st_mode & S_IFDIR) != 0;
            }

            // Get file size in bytes
            static core::i64 file_size(const std::string& path) {
                struct stat buffer;
                if (stat(path.c_str(), &buffer) != 0) return -1;
                return static_cast<core::i64>(buffer.st_size);
            }

            // =========================================================================
            // Text file operations
            // =========================================================================

            // Read entire file as string
            static std::string read_text(const std::string& path) {
                std::ifstream file(path);
                if (!file.is_open()) {
                    throw std::runtime_error("Cannot open file for reading: " + path);
                }

                std::stringstream buffer;
                buffer << file.rdbuf();
                file.close();
                return buffer.str();
            }

            // Write string to file
            static void write_text(const std::string& path, const std::string& content) {
                std::ofstream file(path);
                if (!file.is_open()) {
                    throw std::runtime_error("Cannot open file for writing: " + path);
                }

                file << content;
                file.close();
            }

            // Append string to file
            static void append_text(const std::string& path, const std::string& content) {
                std::ofstream file(path, std::ios::app);
                if (!file.is_open()) {
                    throw std::runtime_error("Cannot open file for appending: " + path);
                }

                file << content;
                file.close();
            }

            // Read file as lines
            static std::vector<std::string> read_lines(const std::string& path) {
                std::ifstream file(path);
                if (!file.is_open()) {
                    throw std::runtime_error("Cannot open file for reading: " + path);
                }

                std::vector<std::string> lines;
                std::string line;
                while (std::getline(file, line)) {
                    lines.push_back(line);
                }

                file.close();
                return lines;
            }

            // Write lines to file
            static void write_lines(const std::string& path, const std::vector<std::string>& lines) {
                std::ofstream file(path);
                if (!file.is_open()) {
                    throw std::runtime_error("Cannot open file for writing: " + path);
                }

                for (const auto& line : lines) {
                    file << line << "\n";
                }

                file.close();
            }

            // =========================================================================
            // Binary file operations
            // =========================================================================

            // Read entire file as binary data
            static std::vector<core::u8> read_binary(const std::string& path) {
                std::ifstream file(path, std::ios::binary | std::ios::ate);
                if (!file.is_open()) {
                    throw std::runtime_error("Cannot open file for reading: " + path);
                }

                std::streamsize size = file.tellg();
                file.seekg(0, std::ios::beg);

                std::vector<core::u8> buffer(static_cast<size_t>(size));
                if (!file.read(reinterpret_cast<char*>(buffer.data()), size)) {
                    throw std::runtime_error("Failed to read file: " + path);
                }

                file.close();
                return buffer;
            }

            // Write binary data to file
            static void write_binary(const std::string& path, const std::vector<core::u8>& data) {
                std::ofstream file(path, std::ios::binary);
                if (!file.is_open()) {
                    throw std::runtime_error("Cannot open file for writing: " + path);
                }

                file.write(reinterpret_cast<const char*>(data.data()), data.size());
                file.close();
            }

            // Write binary data from raw pointer
            static void write_binary(const std::string& path, const void* data, core::usize size) {
                std::ofstream file(path, std::ios::binary);
                if (!file.is_open()) {
                    throw std::runtime_error("Cannot open file for writing: " + path);
                }

                file.write(reinterpret_cast<const char*>(data), size);
                file.close();
            }

            // =========================================================================
            // File management
            // =========================================================================

            // Delete a file
            static bool remove_file(const std::string& path) {
                return std::remove(path.c_str()) == 0;
            }

            // Rename/move a file
            static bool rename_file(const std::string& old_path, const std::string& new_path) {
                return std::rename(old_path.c_str(), new_path.c_str()) == 0;
            }

            // Copy a file
            static void copy_file(const std::string& src, const std::string& dst) {
                std::ifstream source(src, std::ios::binary);
                if (!source.is_open()) {
                    throw std::runtime_error("Cannot open source file: " + src);
                }

                std::ofstream dest(dst, std::ios::binary);
                if (!dest.is_open()) {
                    source.close();
                    throw std::runtime_error("Cannot open destination file: " + dst);
                }

                dest << source.rdbuf();
                source.close();
                dest.close();
            }

            // Create directory
            static bool create_directory(const std::string& path) {
                return PSI_MKDIR(path.c_str()) == 0;
            }

            // =========================================================================
            // Path utilities
            // =========================================================================

            // Get file extension (including dot)
            static std::string get_extension(const std::string& path) {
                size_t dot_pos = path.rfind('.');
                size_t sep_pos = path.find_last_of("/\\");

                if (dot_pos == std::string::npos) return "";
                if (sep_pos != std::string::npos && dot_pos < sep_pos) return "";

                return path.substr(dot_pos);
            }

            // Get filename from path (with extension)
            static std::string get_filename(const std::string& path) {
                size_t sep_pos = path.find_last_of("/\\");
                if (sep_pos == std::string::npos) return path;
                return path.substr(sep_pos + 1);
            }

            // Get filename without extension
            static std::string get_basename(const std::string& path) {
                std::string filename = get_filename(path);
                size_t dot_pos = filename.rfind('.');
                if (dot_pos == std::string::npos) return filename;
                return filename.substr(0, dot_pos);
            }

            // Get directory from path
            static std::string get_directory(const std::string& path) {
                size_t sep_pos = path.find_last_of("/\\");
                if (sep_pos == std::string::npos) return ".";
                return path.substr(0, sep_pos);
            }

            // Join path components
            static std::string join_path(const std::string& base, const std::string& component) {
                if (base.empty()) return component;
                if (component.empty()) return base;

                char last = base.back();
                if (last == '/' || last == '\\') {
                    return base + component;
                }
#ifdef _WIN32
                return base + "\\" + component;
#else
                return base + "/" + component;
#endif
            }

            // Normalize path separators
            static std::string normalize_path(const std::string& path) {
                std::string result = path;
#ifdef _WIN32
                for (char& c : result) {
                    if (c == '/') c = '\\';
                }
#else
                for (char& c : result) {
                    if (c == '\\') c = '/';
                }
#endif
                return result;
            }

            // =========================================================================
            // Temporary files
            // =========================================================================

            // Generate a temporary filename
            static std::string temp_filename(const std::string& prefix = "tmp",
                                            const std::string& extension = ".tmp") {
                static int counter = 0;
                std::stringstream ss;
                ss << prefix << "_" << counter++ << "_" << std::time(nullptr) << extension;
                return ss.str();
            }
        };

    } // namespace utils
} // namespace psi
