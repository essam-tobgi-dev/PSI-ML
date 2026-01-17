#include "../include/utils/file_io.h"
#include <iostream>
#include <cassert>
#include <cstring>

using namespace psi::utils;
using namespace psi::core;

// =============================================================================
// Test: File existence
// =============================================================================

void test_file_exists() {
    std::cout << "Testing file existence..." << std::endl;

    // Create a test file
    FileIO::write_text("temp_exists_test.txt", "test content");

    assert(FileIO::exists("temp_exists_test.txt"));
    assert(!FileIO::exists("nonexistent_file_xyz.txt"));

    // Cleanup
    FileIO::remove_file("temp_exists_test.txt");
    assert(!FileIO::exists("temp_exists_test.txt"));

    std::cout << "  File existence: PASSED" << std::endl;
}

// =============================================================================
// Test: File type checks
// =============================================================================

void test_file_type_checks() {
    std::cout << "Testing file type checks..." << std::endl;

    // Create a test file
    FileIO::write_text("temp_type_test.txt", "test");

    assert(FileIO::is_file("temp_type_test.txt"));
    assert(!FileIO::is_directory("temp_type_test.txt"));

    // Test directory (current directory should exist)
    assert(FileIO::is_directory("."));
    assert(!FileIO::is_file("."));

    // Cleanup
    FileIO::remove_file("temp_type_test.txt");

    std::cout << "  File type checks: PASSED" << std::endl;
}

// =============================================================================
// Test: File size
// =============================================================================

void test_file_size() {
    std::cout << "Testing file size..." << std::endl;

    std::string content = "Hello, World!";  // 13 bytes
    FileIO::write_text("temp_size_test.txt", content);

    i64 size = FileIO::file_size("temp_size_test.txt");
    assert(size == 13);

    // Nonexistent file should return -1
    assert(FileIO::file_size("nonexistent_file.txt") == -1);

    // Cleanup
    FileIO::remove_file("temp_size_test.txt");

    std::cout << "  File size: PASSED" << std::endl;
}

// =============================================================================
// Test: Read/write text
// =============================================================================

void test_text_read_write() {
    std::cout << "Testing text read/write..." << std::endl;

    std::string original = "Line 1\nLine 2\nLine 3";
    FileIO::write_text("temp_text_test.txt", original);

    std::string loaded = FileIO::read_text("temp_text_test.txt");
    assert(loaded == original);

    // Cleanup
    FileIO::remove_file("temp_text_test.txt");

    std::cout << "  Text read/write: PASSED" << std::endl;
}

// =============================================================================
// Test: Append text
// =============================================================================

void test_text_append() {
    std::cout << "Testing text append..." << std::endl;

    FileIO::write_text("temp_append_test.txt", "First");
    FileIO::append_text("temp_append_test.txt", "Second");
    FileIO::append_text("temp_append_test.txt", "Third");

    std::string content = FileIO::read_text("temp_append_test.txt");
    assert(content == "FirstSecondThird");

    // Cleanup
    FileIO::remove_file("temp_append_test.txt");

    std::cout << "  Text append: PASSED" << std::endl;
}

// =============================================================================
// Test: Read/write lines
// =============================================================================

void test_lines_read_write() {
    std::cout << "Testing lines read/write..." << std::endl;

    std::vector<std::string> original = {"Line 1", "Line 2", "Line 3", "Line 4"};
    FileIO::write_lines("temp_lines_test.txt", original);

    std::vector<std::string> loaded = FileIO::read_lines("temp_lines_test.txt");

    assert(loaded.size() == original.size());
    for (size_t i = 0; i < original.size(); ++i) {
        assert(loaded[i] == original[i]);
    }

    // Cleanup
    FileIO::remove_file("temp_lines_test.txt");

    std::cout << "  Lines read/write: PASSED" << std::endl;
}

// =============================================================================
// Test: Binary read/write
// =============================================================================

void test_binary_read_write() {
    std::cout << "Testing binary read/write..." << std::endl;

    std::vector<u8> original = {0x00, 0x01, 0x02, 0xFF, 0xFE, 0x80, 0x7F};
    FileIO::write_binary("temp_binary_test.bin", original);

    std::vector<u8> loaded = FileIO::read_binary("temp_binary_test.bin");

    assert(loaded.size() == original.size());
    for (size_t i = 0; i < original.size(); ++i) {
        assert(loaded[i] == original[i]);
    }

    // Cleanup
    FileIO::remove_file("temp_binary_test.bin");

    std::cout << "  Binary read/write: PASSED" << std::endl;
}

// =============================================================================
// Test: Binary write from pointer
// =============================================================================

void test_binary_write_pointer() {
    std::cout << "Testing binary write from pointer..." << std::endl;

    int data[] = {1, 2, 3, 4, 5};
    FileIO::write_binary("temp_binary_ptr.bin", data, sizeof(data));

    std::vector<u8> loaded = FileIO::read_binary("temp_binary_ptr.bin");
    assert(loaded.size() == sizeof(data));

    int* loaded_data = reinterpret_cast<int*>(loaded.data());
    for (int i = 0; i < 5; ++i) {
        assert(loaded_data[i] == data[i]);
    }

    // Cleanup
    FileIO::remove_file("temp_binary_ptr.bin");

    std::cout << "  Binary write from pointer: PASSED" << std::endl;
}

// =============================================================================
// Test: File rename
// =============================================================================

void test_file_rename() {
    std::cout << "Testing file rename..." << std::endl;

    FileIO::write_text("temp_rename_old.txt", "test content");
    assert(FileIO::exists("temp_rename_old.txt"));

    bool success = FileIO::rename_file("temp_rename_old.txt", "temp_rename_new.txt");
    assert(success);
    assert(!FileIO::exists("temp_rename_old.txt"));
    assert(FileIO::exists("temp_rename_new.txt"));

    std::string content = FileIO::read_text("temp_rename_new.txt");
    assert(content == "test content");

    // Cleanup
    FileIO::remove_file("temp_rename_new.txt");

    std::cout << "  File rename: PASSED" << std::endl;
}

// =============================================================================
// Test: File copy
// =============================================================================

void test_file_copy() {
    std::cout << "Testing file copy..." << std::endl;

    std::string original_content = "Content to be copied";
    FileIO::write_text("temp_copy_src.txt", original_content);

    FileIO::copy_file("temp_copy_src.txt", "temp_copy_dst.txt");

    assert(FileIO::exists("temp_copy_src.txt"));
    assert(FileIO::exists("temp_copy_dst.txt"));

    std::string copied_content = FileIO::read_text("temp_copy_dst.txt");
    assert(copied_content == original_content);

    // Cleanup
    FileIO::remove_file("temp_copy_src.txt");
    FileIO::remove_file("temp_copy_dst.txt");

    std::cout << "  File copy: PASSED" << std::endl;
}

// =============================================================================
// Test: Get extension
// =============================================================================

void test_get_extension() {
    std::cout << "Testing get extension..." << std::endl;

    assert(FileIO::get_extension("file.txt") == ".txt");
    assert(FileIO::get_extension("path/to/file.cpp") == ".cpp");
    assert(FileIO::get_extension("archive.tar.gz") == ".gz");
    assert(FileIO::get_extension("noextension") == "");
    assert(FileIO::get_extension("path.to/file") == "");
    assert(FileIO::get_extension(".hidden") == ".hidden");

    std::cout << "  Get extension: PASSED" << std::endl;
}

// =============================================================================
// Test: Get filename
// =============================================================================

void test_get_filename() {
    std::cout << "Testing get filename..." << std::endl;

    assert(FileIO::get_filename("file.txt") == "file.txt");
    assert(FileIO::get_filename("path/to/file.txt") == "file.txt");
    assert(FileIO::get_filename("C:\\path\\to\\file.txt") == "file.txt");
    assert(FileIO::get_filename("/absolute/path/file.txt") == "file.txt");

    std::cout << "  Get filename: PASSED" << std::endl;
}

// =============================================================================
// Test: Get basename
// =============================================================================

void test_get_basename() {
    std::cout << "Testing get basename..." << std::endl;

    assert(FileIO::get_basename("file.txt") == "file");
    assert(FileIO::get_basename("path/to/file.cpp") == "file");
    assert(FileIO::get_basename("archive.tar.gz") == "archive.tar");
    assert(FileIO::get_basename("noextension") == "noextension");

    std::cout << "  Get basename: PASSED" << std::endl;
}

// =============================================================================
// Test: Get directory
// =============================================================================

void test_get_directory() {
    std::cout << "Testing get directory..." << std::endl;

    assert(FileIO::get_directory("file.txt") == ".");
    assert(FileIO::get_directory("path/to/file.txt") == "path/to");
    assert(FileIO::get_directory("C:\\path\\to\\file.txt") == "C:\\path\\to");
    assert(FileIO::get_directory("/absolute/path/file.txt") == "/absolute/path");

    std::cout << "  Get directory: PASSED" << std::endl;
}

// =============================================================================
// Test: Join path
// =============================================================================

void test_join_path() {
    std::cout << "Testing join path..." << std::endl;

    std::string result = FileIO::join_path("path", "file.txt");
#ifdef _WIN32
    assert(result == "path\\file.txt");
#else
    assert(result == "path/file.txt");
#endif

    // With trailing separator
    result = FileIO::join_path("path/", "file.txt");
    assert(result == "path/file.txt");

    // Empty components
    assert(FileIO::join_path("", "file.txt") == "file.txt");
    assert(FileIO::join_path("path", "") == "path");

    std::cout << "  Join path: PASSED" << std::endl;
}

// =============================================================================
// Test: Normalize path
// =============================================================================

void test_normalize_path() {
    std::cout << "Testing normalize path..." << std::endl;

#ifdef _WIN32
    assert(FileIO::normalize_path("path/to/file") == "path\\to\\file");
    assert(FileIO::normalize_path("path\\to\\file") == "path\\to\\file");
#else
    assert(FileIO::normalize_path("path\\to\\file") == "path/to/file");
    assert(FileIO::normalize_path("path/to/file") == "path/to/file");
#endif

    std::cout << "  Normalize path: PASSED" << std::endl;
}

// =============================================================================
// Test: Temp filename
// =============================================================================

void test_temp_filename() {
    std::cout << "Testing temp filename generation..." << std::endl;

    std::string name1 = FileIO::temp_filename();
    std::string name2 = FileIO::temp_filename();

    // Names should be different
    assert(name1 != name2);

    // Should have default extension
    assert(FileIO::get_extension(name1) == ".tmp");

    // Custom prefix and extension
    std::string custom = FileIO::temp_filename("myprefix", ".dat");
    assert(custom.find("myprefix") == 0);
    assert(FileIO::get_extension(custom) == ".dat");

    std::cout << "  Temp filename generation: PASSED" << std::endl;
}

// =============================================================================
// Test: Empty file handling
// =============================================================================

void test_empty_file() {
    std::cout << "Testing empty file handling..." << std::endl;

    // Write empty text file
    FileIO::write_text("temp_empty.txt", "");
    assert(FileIO::exists("temp_empty.txt"));
    assert(FileIO::file_size("temp_empty.txt") == 0);

    std::string content = FileIO::read_text("temp_empty.txt");
    assert(content.empty());

    // Write empty binary file
    std::vector<u8> empty_data;
    FileIO::write_binary("temp_empty.bin", empty_data);
    assert(FileIO::file_size("temp_empty.bin") == 0);

    std::vector<u8> loaded = FileIO::read_binary("temp_empty.bin");
    assert(loaded.empty());

    // Cleanup
    FileIO::remove_file("temp_empty.txt");
    FileIO::remove_file("temp_empty.bin");

    std::cout << "  Empty file handling: PASSED" << std::endl;
}

// =============================================================================
// Test: Large file handling
// =============================================================================

void test_large_file() {
    std::cout << "Testing large file handling..." << std::endl;

    // Create a moderately large file (1MB)
    std::vector<u8> large_data(1024 * 1024);  // 1MB
    for (size_t i = 0; i < large_data.size(); ++i) {
        large_data[i] = static_cast<u8>(i % 256);
    }

    FileIO::write_binary("temp_large.bin", large_data);

    i64 size = FileIO::file_size("temp_large.bin");
    assert(size == 1024 * 1024);

    std::vector<u8> loaded = FileIO::read_binary("temp_large.bin");
    assert(loaded.size() == large_data.size());

    // Verify content
    for (size_t i = 0; i < 1000; ++i) {  // Check first 1000 bytes
        assert(loaded[i] == large_data[i]);
    }

    // Cleanup
    FileIO::remove_file("temp_large.bin");

    std::cout << "  Large file handling: PASSED" << std::endl;
}

// =============================================================================
// Test: Special characters in content
// =============================================================================

void test_special_characters() {
    std::cout << "Testing special characters in content..." << std::endl;

    std::string special = "Tab:\tNewline:\nCarriage return:\rNull in text";
    FileIO::write_text("temp_special.txt", special);

    std::string loaded = FileIO::read_text("temp_special.txt");
    assert(loaded == special);

    // Cleanup
    FileIO::remove_file("temp_special.txt");

    std::cout << "  Special characters: PASSED" << std::endl;
}

// =============================================================================
// Main
// =============================================================================

int main() {
    std::cout << "\n=== FileIO Tests ===" << std::endl;
    std::cout << std::endl;

    try {
        // File info tests
        test_file_exists();
        test_file_type_checks();
        test_file_size();

        std::cout << std::endl;

        // Text operations tests
        test_text_read_write();
        test_text_append();
        test_lines_read_write();

        std::cout << std::endl;

        // Binary operations tests
        test_binary_read_write();
        test_binary_write_pointer();

        std::cout << std::endl;

        // File management tests
        test_file_rename();
        test_file_copy();

        std::cout << std::endl;

        // Path utilities tests
        test_get_extension();
        test_get_filename();
        test_get_basename();
        test_get_directory();
        test_join_path();
        test_normalize_path();
        test_temp_filename();

        std::cout << std::endl;

        // Edge case tests
        test_empty_file();
        test_large_file();
        test_special_characters();

        std::cout << std::endl;
        std::cout << "=== All FileIO Tests PASSED ===" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        return 1;
    }
}
