#pragma once

#include "../core/types.h"
#include <string>
#include <vector>
#include <sstream>
#include <algorithm>
#include <cctype>
#include <iomanip>

namespace psi {
    namespace utils {

        // String utility functions for common operations
        class StringUtils {
        public:
            // Trim whitespace from the beginning of a string
            static std::string trim_left(const std::string& str) {
                auto start = std::find_if_not(str.begin(), str.end(), [](unsigned char c) {
                    return std::isspace(c);
                });
                return std::string(start, str.end());
            }

            // Trim whitespace from the end of a string
            static std::string trim_right(const std::string& str) {
                auto end = std::find_if_not(str.rbegin(), str.rend(), [](unsigned char c) {
                    return std::isspace(c);
                });
                return std::string(str.begin(), end.base());
            }

            // Trim whitespace from both ends of a string
            static std::string trim(const std::string& str) {
                return trim_left(trim_right(str));
            }

            // Convert string to lowercase
            static std::string to_lower(const std::string& str) {
                std::string result = str;
                std::transform(result.begin(), result.end(), result.begin(), [](unsigned char c) {
                    return static_cast<char>(std::tolower(c));
                });
                return result;
            }

            // Convert string to uppercase
            static std::string to_upper(const std::string& str) {
                std::string result = str;
                std::transform(result.begin(), result.end(), result.begin(), [](unsigned char c) {
                    return static_cast<char>(std::toupper(c));
                });
                return result;
            }

            // Split string by delimiter
            static std::vector<std::string> split(const std::string& str, char delimiter) {
                std::vector<std::string> tokens;
                std::stringstream ss(str);
                std::string token;

                while (std::getline(ss, token, delimiter)) {
                    tokens.push_back(token);
                }

                return tokens;
            }

            // Split string by multiple delimiters
            static std::vector<std::string> split(const std::string& str, const std::string& delimiters) {
                std::vector<std::string> tokens;
                core::usize start = 0;

                while (start < str.size()) {
                    core::usize end = str.find_first_of(delimiters, start);
                    if (end == std::string::npos) {
                        tokens.push_back(str.substr(start));
                        break;
                    }
                    if (end > start) {
                        tokens.push_back(str.substr(start, end - start));
                    }
                    start = end + 1;
                }

                return tokens;
            }

            // Join strings with delimiter
            static std::string join(const std::vector<std::string>& strings, const std::string& delimiter) {
                if (strings.empty()) return "";

                std::ostringstream oss;
                oss << strings[0];
                for (core::usize i = 1; i < strings.size(); ++i) {
                    oss << delimiter << strings[i];
                }
                return oss.str();
            }

            // Check if string starts with prefix
            static bool starts_with(const std::string& str, const std::string& prefix) {
                if (prefix.size() > str.size()) return false;
                return str.compare(0, prefix.size(), prefix) == 0;
            }

            // Check if string ends with suffix
            static bool ends_with(const std::string& str, const std::string& suffix) {
                if (suffix.size() > str.size()) return false;
                return str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
            }

            // Check if string contains substring
            static bool contains(const std::string& str, const std::string& substring) {
                return str.find(substring) != std::string::npos;
            }

            // Replace all occurrences of a substring
            static std::string replace_all(const std::string& str,
                                          const std::string& from,
                                          const std::string& to) {
                if (from.empty()) return str;

                std::string result = str;
                core::usize start_pos = 0;

                while ((start_pos = result.find(from, start_pos)) != std::string::npos) {
                    result.replace(start_pos, from.length(), to);
                    start_pos += to.length();
                }

                return result;
            }

            // Replace first occurrence of a substring
            static std::string replace_first(const std::string& str,
                                            const std::string& from,
                                            const std::string& to) {
                std::string result = str;
                core::usize pos = result.find(from);
                if (pos != std::string::npos) {
                    result.replace(pos, from.length(), to);
                }
                return result;
            }

            // Check if string is empty or contains only whitespace
            static bool is_blank(const std::string& str) {
                return str.empty() || std::all_of(str.begin(), str.end(), [](unsigned char c) {
                    return std::isspace(c);
                });
            }

            // Check if string represents a number
            static bool is_numeric(const std::string& str) {
                if (str.empty()) return false;

                std::string s = trim(str);
                if (s.empty()) return false;

                core::usize start = 0;
                if (s[0] == '+' || s[0] == '-') start = 1;

                bool has_dot = false;
                bool has_digit = false;

                for (core::usize i = start; i < s.size(); ++i) {
                    if (s[i] == '.') {
                        if (has_dot) return false;
                        has_dot = true;
                    } else if (std::isdigit(static_cast<unsigned char>(s[i]))) {
                        has_digit = true;
                    } else {
                        return false;
                    }
                }

                return has_digit;
            }

            // Check if string represents an integer
            static bool is_integer(const std::string& str) {
                if (str.empty()) return false;

                std::string s = trim(str);
                if (s.empty()) return false;

                core::usize start = 0;
                if (s[0] == '+' || s[0] == '-') start = 1;

                if (start >= s.size()) return false;

                for (core::usize i = start; i < s.size(); ++i) {
                    if (!std::isdigit(static_cast<unsigned char>(s[i]))) {
                        return false;
                    }
                }

                return true;
            }

            // Pad string on the left
            static std::string pad_left(const std::string& str, core::usize width, char pad_char = ' ') {
                if (str.size() >= width) return str;
                return std::string(width - str.size(), pad_char) + str;
            }

            // Pad string on the right
            static std::string pad_right(const std::string& str, core::usize width, char pad_char = ' ') {
                if (str.size() >= width) return str;
                return str + std::string(width - str.size(), pad_char);
            }

            // Center string
            static std::string center(const std::string& str, core::usize width, char pad_char = ' ') {
                if (str.size() >= width) return str;
                core::usize padding = width - str.size();
                core::usize left_pad = padding / 2;
                core::usize right_pad = padding - left_pad;
                return std::string(left_pad, pad_char) + str + std::string(right_pad, pad_char);
            }

            // Repeat string n times
            static std::string repeat(const std::string& str, core::usize n) {
                std::string result;
                result.reserve(str.size() * n);
                for (core::usize i = 0; i < n; ++i) {
                    result += str;
                }
                return result;
            }

            // Reverse string
            static std::string reverse(const std::string& str) {
                return std::string(str.rbegin(), str.rend());
            }

            // Count occurrences of substring
            static core::usize count(const std::string& str, const std::string& substring) {
                if (substring.empty()) return 0;

                core::usize count = 0;
                core::usize pos = 0;

                while ((pos = str.find(substring, pos)) != std::string::npos) {
                    ++count;
                    pos += substring.length();
                }

                return count;
            }

            // Format a number with specified precision
            template<typename T>
            static std::string format_number(T value, int precision = 2) {
                std::ostringstream oss;
                oss << std::fixed << std::setprecision(precision) << value;
                return oss.str();
            }

            // Format a number as percentage
            template<typename T>
            static std::string format_percent(T value, int precision = 1) {
                std::ostringstream oss;
                oss << std::fixed << std::setprecision(precision) << (value * T{100}) << "%";
                return oss.str();
            }

            // Convert value to string (generic)
            template<typename T>
            static std::string to_string(const T& value) {
                std::ostringstream oss;
                oss << value;
                return oss.str();
            }

            // Parse string to value
            template<typename T>
            static T parse(const std::string& str) {
                T value;
                std::istringstream iss(str);
                iss >> value;
                return value;
            }

            // Generate progress bar string
            static std::string progress_bar(double progress, core::usize width = 50,
                                           char fill = '=', char empty = ' ') {
                core::usize filled = static_cast<core::usize>(progress * width);
                if (filled > width) filled = width;

                std::string bar = "[";
                bar += std::string(filled, fill);
                if (filled < width) {
                    bar += ">";
                    bar += std::string(width - filled - 1, empty);
                }
                bar += "]";

                std::ostringstream oss;
                oss << bar << " " << std::fixed << std::setprecision(1) << (progress * 100) << "%";
                return oss.str();
            }

            // Create table cell with fixed width
            static std::string table_cell(const std::string& content, core::usize width,
                                         char align = 'l') {
                std::string cell = content.substr(0, std::min(content.size(), width));
                if (align == 'l') {
                    return pad_right(cell, width);
                } else if (align == 'r') {
                    return pad_left(cell, width);
                } else {
                    return center(cell, width);
                }
            }

            // Escape special characters for display
            static std::string escape(const std::string& str) {
                std::string result;
                result.reserve(str.size());

                for (char c : str) {
                    switch (c) {
                        case '\n': result += "\\n"; break;
                        case '\t': result += "\\t"; break;
                        case '\r': result += "\\r"; break;
                        case '\\': result += "\\\\"; break;
                        case '"': result += "\\\""; break;
                        default: result += c; break;
                    }
                }

                return result;
            }

            // Unescape special characters
            static std::string unescape(const std::string& str) {
                std::string result;
                result.reserve(str.size());

                for (core::usize i = 0; i < str.size(); ++i) {
                    if (str[i] == '\\' && i + 1 < str.size()) {
                        switch (str[i + 1]) {
                            case 'n': result += '\n'; ++i; break;
                            case 't': result += '\t'; ++i; break;
                            case 'r': result += '\r'; ++i; break;
                            case '\\': result += '\\'; ++i; break;
                            case '"': result += '"'; ++i; break;
                            default: result += str[i]; break;
                        }
                    } else {
                        result += str[i];
                    }
                }

                return result;
            }
        };

    } // namespace utils
} // namespace psi
