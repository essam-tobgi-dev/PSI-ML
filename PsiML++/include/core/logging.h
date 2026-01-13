#pragma once

#include "types.h"
#include "config.h"
#include <string>
#include <ostream>
#include <fstream>
#include <memory>
#include <mutex>
#include <chrono>
#include <source_location>
#include <sstream>
#include <vector>
#include <map>
#include <type_traits>

namespace psi {
    namespace core {

        // Log levels
        enum class LogLevel : u8 {
            Debug = 0,
            Info = 1,
            Warning = 2,
            Error = 3,
            Fatal = 4,
            Off = 255
        };

        // Convert log level to string
        constexpr const char* log_level_to_string(LogLevel level) {
            switch (level) {
            case LogLevel::Debug: return "DEBUG";
            case LogLevel::Info: return "INFO";
            case LogLevel::Warning: return "WARN";
            case LogLevel::Error: return "ERROR";
            case LogLevel::Fatal: return "FATAL";
            default: return "UNKNOWN";
            }
        }

        // Log record structure
        struct LogRecord {
            LogLevel level;
            std::string message;
            std::string logger_name;
            std::chrono::system_clock::time_point timestamp;
            std::source_location location;
            u32 thread_id;

            LogRecord(LogLevel lvl, const std::string& msg, const std::string& name,
                const std::source_location& loc = std::source_location::current())
                : level(lvl)
                , message(msg)
                , logger_name(name)
                , timestamp(std::chrono::system_clock::now())
                , location(loc)
                , thread_id(get_thread_id()) {
            }

            // Helper methods to extract location information
            std::string file_name() const {
                return location.file_name() ? location.file_name() : "";
            }

            u32 line_number() const {
                return location.line();
            }

            std::string function_name() const {
                return location.function_name() ? location.function_name() : "";
            }

        private:
            static u32 get_thread_id();
        };

        // Log formatter interface
        class LogFormatter {
        public:
            virtual ~LogFormatter() = default;
            virtual std::string format(const LogRecord& record) = 0;
        };

        // Simple text formatter
        class SimpleFormatter : public LogFormatter {
        public:
            std::string format(const LogRecord& record) override;
        };

        // Detailed formatter with timestamp and location info
        class DetailedFormatter : public LogFormatter {
        public:
            explicit DetailedFormatter(bool include_location = true)
                : include_location_(include_location) {
            }

            std::string format(const LogRecord& record) override;

        private:
            bool include_location_;
        };

        // JSON formatter
        class JsonFormatter : public LogFormatter {
        public:
            std::string format(const LogRecord& record) override;
        };

        // Log handler interface
        class LogHandler {
        public:
            explicit LogHandler(LogLevel level = LogLevel::Info)
                : level_(level), formatter_(std::make_unique<SimpleFormatter>()) {
            }

            virtual ~LogHandler() = default;

            virtual void emit(const LogRecord& record) = 0;
            virtual void flush() {}

            void set_level(LogLevel level) { level_ = level; }
            LogLevel get_level() const { return level_; }

            void set_formatter(std::unique_ptr<LogFormatter> formatter) {
                formatter_ = std::move(formatter);
            }

            bool should_emit(LogLevel level) const {
                return level >= level_;
            }

        protected:
            LogLevel level_;
            std::unique_ptr<LogFormatter> formatter_;
        };

        // Console handler (stdout/stderr)
        class ConsoleHandler : public LogHandler {
        public:
            explicit ConsoleHandler(bool use_stderr = false, LogLevel level = LogLevel::Info)
                : LogHandler(level), use_stderr_(use_stderr) {
            }

            void emit(const LogRecord& record) override;
            void flush() override;

        private:
            bool use_stderr_;
            mutable std::mutex mutex_;
        };

        // File handler
        class FileHandler : public LogHandler {
        public:
            explicit FileHandler(const std::string& filename, bool append = true,
                LogLevel level = LogLevel::Info)
                : LogHandler(level), filename_(filename) {
                file_.open(filename, append ? std::ios::app : std::ios::out);
                if (!file_.is_open()) {
                    throw std::runtime_error("Failed to open log file: " + filename);
                }
            }

            ~FileHandler() override {
                if (file_.is_open()) {
                    file_.close();
                }
            }

            void emit(const LogRecord& record) override;
            void flush() override;

            const std::string& get_filename() const { return filename_; }

        private:
            std::string filename_;
            std::ofstream file_;
            mutable std::mutex mutex_;
        };

        // Rotating file handler (size-based rotation)
        class RotatingFileHandler : public LogHandler {
        public:
            explicit RotatingFileHandler(const std::string& filename,
                memory_size_t max_size = 10 * 1024 * 1024,  // 10MB
                u32 backup_count = 5,
                LogLevel level = LogLevel::Info)
                : LogHandler(level)
                , base_filename_(filename)
                , max_size_(max_size)
                , backup_count_(backup_count)
                , current_size_(0) {
                open_current_file();
            }

            ~RotatingFileHandler() override {
                if (file_.is_open()) {
                    file_.close();
                }
            }

            void emit(const LogRecord& record) override;
            void flush() override;

        private:
            std::string base_filename_;
            memory_size_t max_size_;
            u32 backup_count_;
            memory_size_t current_size_;
            std::ofstream file_;
            mutable std::mutex mutex_;

            void open_current_file();
            void rotate_files();
            std::string get_backup_filename(u32 index);
        };

        // Logger class
        class Logger {
        public:
            explicit Logger(const std::string& name)
                : name_(name), level_(LogLevel::Info), propagate_(true) {
            }

            ~Logger() = default;

            // Logging methods
            void log(LogLevel level, const std::string& message,
                const std::source_location& location = std::source_location::current());

            void debug(const std::string& message,
                const std::source_location& location = std::source_location::current()) {
                log(LogLevel::Debug, message, location);
            }

            void info(const std::string& message,
                const std::source_location& location = std::source_location::current()) {
                log(LogLevel::Info, message, location);
            }

            void warning(const std::string& message,
                const std::source_location& location = std::source_location::current()) {
                log(LogLevel::Warning, message, location);
            }

            void error(const std::string& message,
                const std::source_location& location = std::source_location::current()) {
                log(LogLevel::Error, message, location);
            }

            void fatal(const std::string& message,
                const std::source_location& location = std::source_location::current()) {
                log(LogLevel::Fatal, message, location);
            }

            // Template logging with formatting (constrained to exclude source_location)
            template<typename... Args>
            requires (sizeof...(Args) > 0) && (!std::is_same_v<std::remove_cvref_t<Args>, std::source_location> && ...)
            void log(LogLevel level, const std::string& format, Args&&... args) {
                if (should_log(level)) {
                    std::ostringstream oss;
                    format_message(oss, format, std::forward<Args>(args)...);
                    log(level, oss.str());
                }
            }

            template<typename... Args>
            requires (sizeof...(Args) > 0) && (!std::is_same_v<std::remove_cvref_t<Args>, std::source_location> && ...)
            void debug(const std::string& format, Args&&... args) {
                log(LogLevel::Debug, format, std::forward<Args>(args)...);
            }

            template<typename... Args>
            requires (sizeof...(Args) > 0) && (!std::is_same_v<std::remove_cvref_t<Args>, std::source_location> && ...)
            void info(const std::string& format, Args&&... args) {
                log(LogLevel::Info, format, std::forward<Args>(args)...);
            }

            template<typename... Args>
            requires (sizeof...(Args) > 0) && (!std::is_same_v<std::remove_cvref_t<Args>, std::source_location> && ...)
            void warning(const std::string& format, Args&&... args) {
                log(LogLevel::Warning, format, std::forward<Args>(args)...);
            }

            template<typename... Args>
            requires (sizeof...(Args) > 0) && (!std::is_same_v<std::remove_cvref_t<Args>, std::source_location> && ...)
            void error(const std::string& format, Args&&... args) {
                log(LogLevel::Error, format, std::forward<Args>(args)...);
            }

            template<typename... Args>
            requires (sizeof...(Args) > 0) && (!std::is_same_v<std::remove_cvref_t<Args>, std::source_location> && ...)
            void fatal(const std::string& format, Args&&... args) {
                log(LogLevel::Fatal, format, std::forward<Args>(args)...);
            }

            // Configuration
            void set_level(LogLevel level) { level_ = level; }
            LogLevel get_level() const { return level_; }

            void add_handler(std::shared_ptr<LogHandler> handler) {
                std::lock_guard<std::mutex> lock(mutex_);
                handlers_.push_back(handler);
            }

            void remove_handler(std::shared_ptr<LogHandler> handler) {
                std::lock_guard<std::mutex> lock(mutex_);
                handlers_.erase(
                    std::remove(handlers_.begin(), handlers_.end(), handler),
                    handlers_.end());
            }

            void clear_handlers() {
                std::lock_guard<std::mutex> lock(mutex_);
                handlers_.clear();
            }

            const std::string& get_name() const { return name_; }

            bool should_log(LogLevel level) const {
                return level >= level_;
            }

            void flush() {
                std::lock_guard<std::mutex> lock(mutex_);
                for (auto& handler : handlers_) {
                    handler->flush();
                }
            }

        private:
            std::string name_;
            LogLevel level_;
            bool propagate_;
            std::vector<std::shared_ptr<LogHandler>> handlers_;
            mutable std::mutex mutex_;

            // Simple string formatting helper
            template<typename T>
            void format_message(std::ostringstream& oss, const std::string& format, T&& value) {
                size_t pos = format.find("{}");
                if (pos != std::string::npos) {
                    oss << format.substr(0, pos) << value << format.substr(pos + 2);
                }
                else {
                    oss << format;
                }
            }

            template<typename T, typename... Args>
            void format_message(std::ostringstream& oss, const std::string& format,
                T&& value, Args&&... args) {
                size_t pos = format.find("{}");
                if (pos != std::string::npos) {
                    std::string remaining = format.substr(pos + 2);
                    oss << format.substr(0, pos) << value;
                    format_message(oss, remaining, std::forward<Args>(args)...);
                }
                else {
                    oss << format;
                }
            }
        };

        // Logging manager singleton
        class LoggingManager {
        public:
            static LoggingManager& instance() {
                static LoggingManager manager;
                return manager;
            }

            // Get or create logger
            std::shared_ptr<Logger> get_logger(const std::string& name = "psi");

            // Root logger access
            std::shared_ptr<Logger> get_root_logger() {
                return get_logger("psi");
            }

            // Global configuration
            void set_global_level(LogLevel level) {
                global_level_ = level;
                for (auto& [name, logger] : loggers_) {
                    logger->set_level(level);
                }
            }

            LogLevel get_global_level() const { return global_level_; }

            // Convenience methods for setting up common configurations
            void setup_console_logging(LogLevel level = LogLevel::Info, bool detailed = false);
            void setup_file_logging(const std::string& filename, LogLevel level = LogLevel::Info,
                bool detailed = true);
            void setup_rotating_file_logging(const std::string& filename,
                memory_size_t max_size = 10 * 1024 * 1024,
                u32 backup_count = 5,
                LogLevel level = LogLevel::Info);

            // Disable all logging
            void disable_logging() {
                set_global_level(LogLevel::Off);
            }

            // Flush all loggers
            void flush_all() {
                std::lock_guard<std::mutex> lock(mutex_);
                for (auto& [name, logger] : loggers_) {
                    logger->flush();
                }
            }

        private:
            LoggingManager() : global_level_(LogLevel::Info) {}
            ~LoggingManager() = default;
            LoggingManager(const LoggingManager&) = delete;
            LoggingManager& operator=(const LoggingManager&) = delete;

            std::map<std::string, std::shared_ptr<Logger>> loggers_;
            LogLevel global_level_;
            mutable std::mutex mutex_;
        };

        // Global convenience functions
        PSI_NODISCARD inline std::shared_ptr<Logger> get_logger(const std::string& name = "psi") {
            return LoggingManager::instance().get_logger(name);
        }

        inline void setup_console_logging(LogLevel level = LogLevel::Info, bool detailed = false) {
            LoggingManager::instance().setup_console_logging(level, detailed);
        }

        inline void setup_file_logging(const std::string& filename, LogLevel level = LogLevel::Info,
            bool detailed = true) {
            LoggingManager::instance().setup_file_logging(filename, level, detailed);
        }

        inline void set_log_level(LogLevel level) {
            LoggingManager::instance().set_global_level(level);
        }

        inline void flush_logs() {
            LoggingManager::instance().flush_all();
        }

        // Logging macros for convenience
#define PSI_LOG(level, message) \
    ::psi::core::get_logger()->log(level, message, std::source_location::current())

#define PSI_LOG_DEBUG(message) \
    ::psi::core::get_logger()->debug(message, std::source_location::current())

#define PSI_LOG_INFO(message) \
    ::psi::core::get_logger()->info(message, std::source_location::current())

#define PSI_LOG_WARNING(message) \
    ::psi::core::get_logger()->warning(message, std::source_location::current())

#define PSI_LOG_ERROR(message) \
    ::psi::core::get_logger()->error(message, std::source_location::current())

#define PSI_LOG_FATAL(message) \
    ::psi::core::get_logger()->fatal(message, std::source_location::current())

// Named logger macros
#define PSI_LOG_NAMED(name, level, message) \
    ::psi::core::get_logger(name)->log(level, message, std::source_location::current())

#define PSI_DEBUG_NAMED(name, message) \
    ::psi::core::get_logger(name)->debug(message, std::source_location::current())

#define PSI_INFO_NAMED(name, message) \
    ::psi::core::get_logger(name)->info(message, std::source_location::current())

#define PSI_WARNING_NAMED(name, message) \
    ::psi::core::get_logger(name)->warning(message, std::source_location::current())

#define PSI_ERROR_NAMED(name, message) \
    ::psi::core::get_logger(name)->error(message, std::source_location::current())

#define PSI_FATAL_NAMED(name, message) \
    ::psi::core::get_logger(name)->fatal(message, std::source_location::current())

    } // namespace core
} // namespace psi