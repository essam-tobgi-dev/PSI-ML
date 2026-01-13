#include "../include/core/logging.h"
#include <iostream>
#include <cassert>
#include <fstream>
#include <string>

using namespace psi::core;

void test_log_levels() {
    std::cout << "Testing LogLevel enum..." << std::endl;

    assert(std::string(log_level_to_string(LogLevel::Debug)) == "DEBUG");
    assert(std::string(log_level_to_string(LogLevel::Info)) == "INFO");
    assert(std::string(log_level_to_string(LogLevel::Warning)) == "WARN");
    assert(std::string(log_level_to_string(LogLevel::Error)) == "ERROR");
    assert(std::string(log_level_to_string(LogLevel::Fatal)) == "FATAL");

    std::cout << "  LogLevel conversion: PASSED" << std::endl;
}

void test_formatters() {
    std::cout << "Testing Log Formatters..." << std::endl;

    LogRecord record(LogLevel::Info, "Test message", "test_logger");

    // Test SimpleFormatter
    SimpleFormatter simple_formatter;
    std::string simple_output = simple_formatter.format(record);
    assert(!simple_output.empty());
    assert(simple_output.find("Test message") != std::string::npos);
    std::cout << "  SimpleFormatter: " << simple_output;

    // Test DetailedFormatter
    DetailedFormatter detailed_formatter(true);
    std::string detailed_output = detailed_formatter.format(record);
    assert(!detailed_output.empty());
    assert(detailed_output.find("Test message") != std::string::npos);
    std::cout << "  DetailedFormatter: " << detailed_output;

    // Test JsonFormatter
    JsonFormatter json_formatter;
    std::string json_output = json_formatter.format(record);
    assert(!json_output.empty());
    assert(json_output.find("Test message") != std::string::npos);
    std::cout << "  JsonFormatter: " << json_output;

    std::cout << "  Log formatters: PASSED" << std::endl;
}

void test_console_handler() {
    std::cout << "Testing ConsoleHandler..." << std::endl;

    ConsoleHandler handler(false, LogLevel::Info);

    // Test level setting
    handler.set_level(LogLevel::Warning);
    assert(handler.get_level() == LogLevel::Warning);

    // Test should_emit
    assert(!handler.should_emit(LogLevel::Info));
    assert(handler.should_emit(LogLevel::Warning));
    assert(handler.should_emit(LogLevel::Error));

    // Test emit
    LogRecord record(LogLevel::Warning, "Console test message", "test");
    handler.emit(record);

    handler.flush();

    std::cout << "  ConsoleHandler: PASSED" << std::endl;
}

void test_file_handler() {
    std::cout << "Testing FileHandler..." << std::endl;

    const std::string test_file = "test_log.txt";

    {
        FileHandler handler(test_file, false, LogLevel::Info);

        LogRecord record(LogLevel::Info, "File test message", "test");
        handler.emit(record);
        handler.flush();
    }

    // Verify file was created and contains message
    std::ifstream file(test_file);
    assert(file.is_open());

    std::string content((std::istreambuf_iterator<char>(file)),
        std::istreambuf_iterator<char>());
    file.close();

    assert(!content.empty());
    assert(content.find("File test message") != std::string::npos);

    // Clean up
    std::remove(test_file.c_str());

    std::cout << "  FileHandler: PASSED" << std::endl;
}

void test_logger() {
    std::cout << "Testing Logger..." << std::endl;

    Logger logger("test_logger");

    // Test level setting
    logger.set_level(LogLevel::Debug);
    assert(logger.get_level() == LogLevel::Debug);
    assert(logger.get_name() == "test_logger");

    // Add console handler
    auto handler = std::make_shared<ConsoleHandler>(false, LogLevel::Info);
    logger.add_handler(handler);

    // Test logging methods
    logger.debug("Debug message");
    logger.info("Info message");
    logger.warning("Warning message");
    logger.error("Error message");

    // Test should_log
    assert(logger.should_log(LogLevel::Debug));
    assert(logger.should_log(LogLevel::Info));

    // Test flush
    logger.flush();

    // Test clear handlers
    logger.clear_handlers();

    std::cout << "  Logger operations: PASSED" << std::endl;
}

void test_logging_manager() {
    std::cout << "Testing LoggingManager..." << std::endl;

    auto& manager = LoggingManager::instance();

    // Test getting logger
    auto logger = manager.get_logger("test_manager");
    assert(logger != nullptr);
    assert(logger->get_name() == "test_manager");

    // Test getting root logger
    auto root = manager.get_root_logger();
    assert(root != nullptr);

    // Test global level
    manager.set_global_level(LogLevel::Warning);
    assert(manager.get_global_level() == LogLevel::Warning);

    // Test setup methods
    manager.setup_console_logging(LogLevel::Info, false);

    // Test getting same logger returns same instance
    auto logger2 = manager.get_logger("test_manager");
    assert(logger == logger2);

    std::cout << "  LoggingManager: PASSED" << std::endl;
}

void test_global_functions() {
    std::cout << "Testing global logging functions..." << std::endl;

    // Setup console logging
    setup_console_logging(LogLevel::Info, false);

    // Get logger
    auto logger = get_logger("global_test");
    assert(logger != nullptr);

    // Test logging
    logger->info("Global function test");

    // Set log level
    set_log_level(LogLevel::Warning);

    // Flush logs
    flush_logs();

    std::cout << "  Global functions: PASSED" << std::endl;
}

void test_logging_macros() {
    std::cout << "Testing logging macros..." << std::endl;

    setup_console_logging(LogLevel::Debug, false);

    PSI_LOG_DEBUG("Debug macro message");
    PSI_LOG_INFO("Info macro message");
    PSI_LOG_WARNING("Warning macro message");
    PSI_LOG_ERROR("Error macro message");

    // Named logger macros
    PSI_INFO_NAMED("macro_test", "Named logger message");

    std::cout << "  Logging macros: PASSED" << std::endl;
}

int main() {
    std::cout << "\n=== Logging System Tests ===" << std::endl;
    std::cout << std::endl;

    try {
        test_log_levels();
        std::cout << std::endl;

        test_formatters();
        std::cout << std::endl;

        test_console_handler();
        std::cout << std::endl;

        test_file_handler();
        std::cout << std::endl;

        test_logger();
        std::cout << std::endl;

        test_logging_manager();
        std::cout << std::endl;

        test_global_functions();
        std::cout << std::endl;

        test_logging_macros();
        std::cout << std::endl;

        std::cout << "=== All Logging Tests PASSED ===" << std::endl;
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        return 1;
    }
}
