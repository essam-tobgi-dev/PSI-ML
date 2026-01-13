#include "../../include/core/logging.h"
#include "../../include/core/config.h"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <sstream>
#include <filesystem>
#include <thread>

#ifdef _WIN32
#include <windows.h>
#include <io.h>
#define isatty _isatty
#define STDOUT_FILENO 1
#define STDERR_FILENO 2
#else
#include <unistd.h>
#endif

namespace psi {
    namespace core {

        // Utility function to get thread ID
        u32 LogRecord::get_thread_id() {
            return static_cast<u32>(std::hash<std::thread::id>{}(std::this_thread::get_id()));
        }

        // ANSI color codes for console output
        namespace {
            const char* RESET = "\033[0m";
            const char* RED = "\033[31m";
            const char* YELLOW = "\033[33m";
            const char* CYAN = "\033[36m";
            const char* WHITE = "\033[37m";
            const char* BRIGHT_RED = "\033[91m";

            bool supports_color() {
#ifdef _WIN32
                // Enable ANSI escape sequences on Windows 10
                HANDLE hOut = GetStdHandle(STD_OUTPUT_HANDLE);
                if (hOut == INVALID_HANDLE_VALUE) return false;

                DWORD dwMode = 0;
                if (!GetConsoleMode(hOut, &dwMode)) return false;

                dwMode |= ENABLE_VIRTUAL_TERMINAL_PROCESSING;
                return SetConsoleMode(hOut, dwMode);
#else
                return isatty(STDOUT_FILENO);
#endif
            }

            const char* get_level_color(LogLevel level) {
                switch (level) {
                case LogLevel::Debug: return CYAN;
                case LogLevel::Info: return WHITE;
                case LogLevel::Warning: return YELLOW;
                case LogLevel::Error: return RED;
                case LogLevel::Fatal: return BRIGHT_RED;
                default: return WHITE;
                }
            }

            std::string format_timestamp(const std::chrono::system_clock::time_point& tp) {
                auto time_t = std::chrono::system_clock::to_time_t(tp);
                auto duration = tp.time_since_epoch();
                auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(duration) % 1000;

                std::stringstream ss;
                ss << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S");
                ss << '.' << std::setfill('0') << std::setw(3) << millis.count();
                return ss.str();
            }
        }

        // SimpleFormatter implementation
        std::string SimpleFormatter::format(const LogRecord& record) {
            std::ostringstream oss;
            oss << "[" << log_level_to_string(record.level) << "] ";

            if (!record.logger_name.empty() && record.logger_name != "psi") {
                oss << record.logger_name << ": ";
            }

            oss << record.message;
            return oss.str();
        }

        // DetailedFormatter implementation
        std::string DetailedFormatter::format(const LogRecord& record) {
            std::ostringstream oss;

            // Timestamp
            oss << "[" << format_timestamp(record.timestamp) << "] ";

            // Thread ID
            oss << "[T:" << std::setw(6) << std::setfill('0') << record.thread_id << "] ";

            // Log level
            oss << "[" << std::setw(5) << log_level_to_string(record.level) << "] ";

            // Logger name
            if (!record.logger_name.empty()) {
                oss << "[" << record.logger_name << "] ";
            }

            // Message
            oss << record.message;

            // File location (if enabled and available)
            std::string file_name_str = record.file_name();
            if (include_location_ && !file_name_str.empty()) {
                std::filesystem::path file_path(file_name_str);
                oss << " (" << file_path.filename().string() << ":" << record.line_number();
                std::string func_name = record.function_name();
                if (!func_name.empty()) {
                    oss << " in " << func_name << "()";
                }
                oss << ")";
            }

            return oss.str();
        }

        // JsonFormatter implementation
        std::string JsonFormatter::format(const LogRecord& record) {
            std::ostringstream oss;
            oss << "{";
            oss << "\"timestamp\":\"" << format_timestamp(record.timestamp) << "\",";
            oss << "\"level\":\"" << log_level_to_string(record.level) << "\",";
            oss << "\"logger\":\"" << record.logger_name << "\",";
            oss << "\"thread_id\":" << record.thread_id << ",";
            oss << "\"message\":\"";

            // Escape special characters in message
            for (char c : record.message) {
                switch (c) {
                case '"': oss << "\\\""; break;
                case '\\': oss << "\\\\"; break;
                case '\b': oss << "\\b"; break;
                case '\f': oss << "\\f"; break;
                case '\n': oss << "\\n"; break;
                case '\r': oss << "\\r"; break;
                case '\t': oss << "\\t"; break;
                default: oss << c; break;
                }
            }

            oss << "\"";

            std::string file_name_str = record.file_name();
            if (!file_name_str.empty()) {
                oss << ",\"file\":\"" << file_name_str << "\"";
                oss << ",\"line\":" << record.line_number();
                std::string func_name = record.function_name();
                if (!func_name.empty()) {
                    oss << ",\"function\":\"" << func_name << "\"";
                }
            }

            oss << "}";
            return oss.str();
        }

        // ConsoleHandler implementation
        void ConsoleHandler::emit(const LogRecord& record) {
            if (!should_emit(record.level)) {
                return;
            }

            std::lock_guard<std::mutex> lock(mutex_);

            std::ostream& stream = use_stderr_ ? std::cerr : std::cout;
            std::string formatted = formatter_->format(record);

            // Add color if supported
            static bool color_supported = supports_color();
            if (color_supported && !use_stderr_) {
                const char* color = get_level_color(record.level);
                stream << color << formatted << RESET << std::endl;
            }
            else {
                stream << formatted << std::endl;
            }
        }

        void ConsoleHandler::flush() {
            std::lock_guard<std::mutex> lock(mutex_);
            if (use_stderr_) {
                std::cerr.flush();
            }
            else {
                std::cout.flush();
            }
        }

        // FileHandler implementation
        void FileHandler::emit(const LogRecord& record) {
            if (!should_emit(record.level)) {
                return;
            }

            std::lock_guard<std::mutex> lock(mutex_);

            if (file_.is_open()) {
                std::string formatted = formatter_->format(record);
                file_ << formatted << std::endl;
                file_.flush();  // Ensure immediate write
            }
        }

        void FileHandler::flush() {
            std::lock_guard<std::mutex> lock(mutex_);
            if (file_.is_open()) {
                file_.flush();
            }
        }

        // RotatingFileHandler implementation
        void RotatingFileHandler::emit(const LogRecord& record) {
            if (!should_emit(record.level)) {
                return;
            }

            std::lock_guard<std::mutex> lock(mutex_);

            std::string formatted = formatter_->format(record);
            memory_size_t message_size = formatted.length() + 1;  // +1 for newline

            // Check if we need to rotate
            if (current_size_ + message_size > max_size_) {
                rotate_files();
            }

            if (file_.is_open()) {
                file_ << formatted << std::endl;
                current_size_ += message_size;
                file_.flush();
            }
        }

        void RotatingFileHandler::flush() {
            std::lock_guard<std::mutex> lock(mutex_);
            if (file_.is_open()) {
                file_.flush();
            }
        }

        void RotatingFileHandler::open_current_file() {
            if (file_.is_open()) {
                file_.close();
            }

            file_.open(base_filename_, std::ios::out | std::ios::app);
            if (!file_.is_open()) {
                throw std::runtime_error("Failed to open log file: " + base_filename_);
            }

            // Get current file size
            file_.seekp(0, std::ios::end);
            current_size_ = static_cast<memory_size_t>(file_.tellp());
            file_.seekp(0, std::ios::beg);
        }

        void RotatingFileHandler::rotate_files() {
            file_.close();

            // Remove the oldest backup if it exists
            std::string oldest_backup = get_backup_filename(backup_count_);
            std::filesystem::remove(oldest_backup);

            // Rotate existing backups
            for (u32 i = backup_count_; i > 1; --i) {
                std::string src = get_backup_filename(i - 1);
                std::string dst = get_backup_filename(i);

                if (std::filesystem::exists(src)) {
                    std::filesystem::rename(src, dst);
                }
            }

            // Move current file to first backup
            if (std::filesystem::exists(base_filename_)) {
                std::string first_backup = get_backup_filename(1);
                std::filesystem::rename(base_filename_, first_backup);
            }

            // Open new current file
            current_size_ = 0;
            open_current_file();
        }

        std::string RotatingFileHandler::get_backup_filename(u32 index) {
            return base_filename_ + "." + std::to_string(index);
        }

        // Logger implementation
        void Logger::log(LogLevel level, const std::string& message, const std::source_location& location) {
            if (!should_log(level)) {
                return;
            }

            LogRecord record(level, message, name_, location);

            std::lock_guard<std::mutex> lock(mutex_);
            for (auto& handler : handlers_) {
                handler->emit(record);
            }
        }

        // LoggingManager implementation
        std::shared_ptr<Logger> LoggingManager::get_logger(const std::string& name) {
            std::lock_guard<std::mutex> lock(mutex_);

            auto it = loggers_.find(name);
            if (it != loggers_.end()) {
                return it->second;
            }

            // Create new logger
            auto logger = std::make_shared<Logger>(name);
            logger->set_level(global_level_);
            loggers_[name] = logger;

            return logger;
        }

        void LoggingManager::setup_console_logging(LogLevel level, bool detailed) {
            auto logger = get_root_logger();
            logger->clear_handlers();

            auto handler = std::make_shared<ConsoleHandler>(false, level);

            if (detailed) {
                handler->set_formatter(std::make_unique<DetailedFormatter>());
            }
            else {
                handler->set_formatter(std::make_unique<SimpleFormatter>());
            }

            logger->add_handler(handler);
            logger->set_level(level);
            set_global_level(level);
        }

        void LoggingManager::setup_file_logging(const std::string& filename, LogLevel level, bool detailed) {
            auto logger = get_root_logger();

            auto handler = std::make_shared<FileHandler>(filename, true, level);

            if (detailed) {
                handler->set_formatter(std::make_unique<DetailedFormatter>());
            }
            else {
                handler->set_formatter(std::make_unique<SimpleFormatter>());
            }

            logger->add_handler(handler);

            if (logger->get_level() > level) {
                logger->set_level(level);
            }
        }

        void LoggingManager::setup_rotating_file_logging(const std::string& filename,
            memory_size_t max_size,
            u32 backup_count,
            LogLevel level) {
            auto logger = get_root_logger();

            auto handler = std::make_shared<RotatingFileHandler>(filename, max_size, backup_count, level);
            handler->set_formatter(std::make_unique<DetailedFormatter>());

            logger->add_handler(handler);

            if (logger->get_level() > level) {
                logger->set_level(level);
            }
        }

    } // namespace core
} // namespace psi