#include "../include/utils/string_utils.h"
#include <iostream>
#include <cassert>

using namespace psi::utils;
using namespace psi::core;

// =============================================================================
// Trim Tests
// =============================================================================

void test_trim() {
    std::cout << "Testing trim functions..." << std::endl;

    assert(StringUtils::trim("  hello  ") == "hello");
    assert(StringUtils::trim("hello") == "hello");
    assert(StringUtils::trim("  ") == "");
    assert(StringUtils::trim("") == "");
    assert(StringUtils::trim("\t\nhello\r\n") == "hello");

    assert(StringUtils::trim_left("  hello") == "hello");
    assert(StringUtils::trim_right("hello  ") == "hello");

    std::cout << "  trim functions: PASSED" << std::endl;
}

// =============================================================================
// Case Conversion Tests
// =============================================================================

void test_case_conversion() {
    std::cout << "Testing case conversion..." << std::endl;

    assert(StringUtils::to_lower("HELLO WORLD") == "hello world");
    assert(StringUtils::to_lower("Hello123") == "hello123");
    assert(StringUtils::to_upper("hello world") == "HELLO WORLD");
    assert(StringUtils::to_upper("Hello123") == "HELLO123");

    std::cout << "  case conversion: PASSED" << std::endl;
}

// =============================================================================
// Split and Join Tests
// =============================================================================

void test_split_join() {
    std::cout << "Testing split and join..." << std::endl;

    // Split by single delimiter
    auto parts = StringUtils::split("a,b,c", ',');
    assert(parts.size() == 3);
    assert(parts[0] == "a");
    assert(parts[1] == "b");
    assert(parts[2] == "c");

    // Split by multiple delimiters
    auto parts2 = StringUtils::split("a,b;c:d", ",;:");
    assert(parts2.size() == 4);

    // Join
    std::vector<std::string> words = {"hello", "world", "test"};
    assert(StringUtils::join(words, " ") == "hello world test");
    assert(StringUtils::join(words, ",") == "hello,world,test");
    assert(StringUtils::join({}, ",") == "");

    std::cout << "  split and join: PASSED" << std::endl;
}

// =============================================================================
// String Predicates Tests
// =============================================================================

void test_predicates() {
    std::cout << "Testing string predicates..." << std::endl;

    // starts_with
    assert(StringUtils::starts_with("hello world", "hello"));
    assert(!StringUtils::starts_with("hello world", "world"));
    assert(StringUtils::starts_with("hello", "hello"));
    assert(!StringUtils::starts_with("hi", "hello"));

    // ends_with
    assert(StringUtils::ends_with("hello world", "world"));
    assert(!StringUtils::ends_with("hello world", "hello"));
    assert(StringUtils::ends_with("world", "world"));

    // contains
    assert(StringUtils::contains("hello world", "lo wo"));
    assert(!StringUtils::contains("hello world", "xyz"));
    assert(StringUtils::contains("hello", "hello"));

    // is_blank
    assert(StringUtils::is_blank(""));
    assert(StringUtils::is_blank("   "));
    assert(StringUtils::is_blank("\t\n"));
    assert(!StringUtils::is_blank("hello"));
    assert(!StringUtils::is_blank(" a "));

    std::cout << "  string predicates: PASSED" << std::endl;
}

// =============================================================================
// Numeric Checks Tests
// =============================================================================

void test_numeric_checks() {
    std::cout << "Testing numeric checks..." << std::endl;

    // is_numeric
    assert(StringUtils::is_numeric("123"));
    assert(StringUtils::is_numeric("123.456"));
    assert(StringUtils::is_numeric("-123"));
    assert(StringUtils::is_numeric("+123.456"));
    assert(StringUtils::is_numeric(" 123 "));
    assert(!StringUtils::is_numeric(""));
    assert(!StringUtils::is_numeric("abc"));
    assert(!StringUtils::is_numeric("12.34.56"));

    // is_integer
    assert(StringUtils::is_integer("123"));
    assert(StringUtils::is_integer("-456"));
    assert(StringUtils::is_integer("+789"));
    assert(!StringUtils::is_integer("123.456"));
    assert(!StringUtils::is_integer("abc"));

    std::cout << "  numeric checks: PASSED" << std::endl;
}

// =============================================================================
// Replace Tests
// =============================================================================

void test_replace() {
    std::cout << "Testing replace functions..." << std::endl;

    // replace_all
    assert(StringUtils::replace_all("hello world world", "world", "there") == "hello there there");
    assert(StringUtils::replace_all("aaa", "a", "bb") == "bbbbbb");
    assert(StringUtils::replace_all("hello", "xyz", "abc") == "hello");
    assert(StringUtils::replace_all("hello", "", "abc") == "hello");

    // replace_first
    assert(StringUtils::replace_first("hello world world", "world", "there") == "hello there world");
    assert(StringUtils::replace_first("aaa", "a", "bb") == "bba");

    std::cout << "  replace functions: PASSED" << std::endl;
}

// =============================================================================
// Padding Tests
// =============================================================================

void test_padding() {
    std::cout << "Testing padding functions..." << std::endl;

    assert(StringUtils::pad_left("hi", 5) == "   hi");
    assert(StringUtils::pad_left("hi", 5, '0') == "000hi");
    assert(StringUtils::pad_left("hello", 3) == "hello");

    assert(StringUtils::pad_right("hi", 5) == "hi   ");
    assert(StringUtils::pad_right("hi", 5, '-') == "hi---");

    assert(StringUtils::center("hi", 6) == "  hi  ");
    assert(StringUtils::center("hi", 7) == "  hi   ");
    assert(StringUtils::center("hello", 3) == "hello");

    std::cout << "  padding functions: PASSED" << std::endl;
}

// =============================================================================
// String Manipulation Tests
// =============================================================================

void test_manipulation() {
    std::cout << "Testing string manipulation..." << std::endl;

    // repeat
    assert(StringUtils::repeat("ab", 3) == "ababab");
    assert(StringUtils::repeat("x", 5) == "xxxxx");
    assert(StringUtils::repeat("ab", 0) == "");

    // reverse
    assert(StringUtils::reverse("hello") == "olleh");
    assert(StringUtils::reverse("a") == "a");
    assert(StringUtils::reverse("") == "");

    // count
    assert(StringUtils::count("hello", "l") == 2);
    assert(StringUtils::count("hello world", "o") == 2);
    assert(StringUtils::count("aaa", "aa") == 1);
    assert(StringUtils::count("hello", "xyz") == 0);

    std::cout << "  string manipulation: PASSED" << std::endl;
}

// =============================================================================
// Format Tests
// =============================================================================

void test_format() {
    std::cout << "Testing format functions..." << std::endl;

    // format_number
    assert(StringUtils::format_number(3.14159f, 2) == "3.14");
    assert(StringUtils::format_number(3.14159, 4) == "3.1416");
    assert(StringUtils::format_number(100.0, 0) == "100");

    // format_percent
    assert(StringUtils::format_percent(0.5f, 0) == "50%");
    assert(StringUtils::format_percent(0.123, 1) == "12.3%");
    assert(StringUtils::format_percent(1.0, 0) == "100%");

    // to_string
    assert(StringUtils::to_string(42) == "42");
    assert(StringUtils::to_string(3.14f).substr(0, 4) == "3.14");

    std::cout << "  format functions: PASSED" << std::endl;
}

// =============================================================================
// Parse Tests
// =============================================================================

void test_parse() {
    std::cout << "Testing parse functions..." << std::endl;

    assert(StringUtils::parse<int>("42") == 42);
    assert(StringUtils::parse<int>("-10") == -10);
    assert(std::abs(StringUtils::parse<float>("3.14") - 3.14f) < 0.01f);
    assert(std::abs(StringUtils::parse<double>("2.718") - 2.718) < 0.001);

    std::cout << "  parse functions: PASSED" << std::endl;
}

// =============================================================================
// Progress Bar Tests
// =============================================================================

void test_progress_bar() {
    std::cout << "Testing progress_bar..." << std::endl;

    std::string bar0 = StringUtils::progress_bar(0.0, 20);
    std::string bar50 = StringUtils::progress_bar(0.5, 20);
    std::string bar100 = StringUtils::progress_bar(1.0, 20);

    std::cout << "  0%:   " << bar0 << std::endl;
    std::cout << "  50%:  " << bar50 << std::endl;
    std::cout << "  100%: " << bar100 << std::endl;

    assert(StringUtils::contains(bar0, "0.0%"));
    assert(StringUtils::contains(bar50, "50.0%"));
    assert(StringUtils::contains(bar100, "100.0%"));

    std::cout << "  progress_bar: PASSED" << std::endl;
}

// =============================================================================
// Escape Tests
// =============================================================================

void test_escape() {
    std::cout << "Testing escape/unescape..." << std::endl;

    assert(StringUtils::escape("hello\nworld") == "hello\\nworld");
    assert(StringUtils::escape("tab\there") == "tab\\there");
    assert(StringUtils::escape("quote\"here") == "quote\\\"here");

    assert(StringUtils::unescape("hello\\nworld") == "hello\nworld");
    assert(StringUtils::unescape("tab\\there") == "tab\there");
    assert(StringUtils::unescape("quote\\\"here") == "quote\"here");

    // Round-trip
    std::string original = "line1\nline2\ttab";
    std::string escaped = StringUtils::escape(original);
    std::string unescaped = StringUtils::unescape(escaped);
    assert(unescaped == original);

    std::cout << "  escape/unescape: PASSED" << std::endl;
}

// =============================================================================
// Table Cell Tests
// =============================================================================

void test_table_cell() {
    std::cout << "Testing table_cell..." << std::endl;

    assert(StringUtils::table_cell("hi", 10, 'l') == "hi        ");
    assert(StringUtils::table_cell("hi", 10, 'r') == "        hi");
    assert(StringUtils::table_cell("hi", 10, 'c') == "    hi    ");
    assert(StringUtils::table_cell("hello world", 5, 'l') == "hello");

    std::cout << "  table_cell: PASSED" << std::endl;
}

// =============================================================================
// Main
// =============================================================================

int main() {
    std::cout << "\n=== StringUtils Tests ===" << std::endl;
    std::cout << std::endl;

    try {
        test_trim();
        test_case_conversion();
        test_split_join();
        test_predicates();
        test_numeric_checks();
        test_replace();
        test_padding();
        test_manipulation();
        test_format();
        test_parse();
        test_progress_bar();
        test_escape();
        test_table_cell();

        std::cout << std::endl;
        std::cout << "=== All StringUtils Tests PASSED ===" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        return 1;
    }
}
