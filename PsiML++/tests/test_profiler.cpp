#include "../include/utils/profiler.h"
#include "../include/utils/timer.h"
#include <iostream>
#include <cassert>
#include <cmath>
#include <thread>
#include <chrono>

using namespace psi::utils;
using namespace psi::core;

bool approx_equal(double a, double b, double epsilon = 1e-6) {
    return std::abs(a - b) < epsilon;
}

// =============================================================================
// Test: Timer basic functionality
// =============================================================================

void test_timer_basic() {
    std::cout << "Testing Timer basic functionality..." << std::endl;

    Timer timer;

    // Timer should not be running initially
    assert(!timer.is_running());
    assert(approx_equal(timer.elapsed_seconds(), 0.0));

    // Start timer
    timer.start();
    assert(timer.is_running());

    // Sleep for a bit
    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    // Stop timer
    timer.stop();
    assert(!timer.is_running());

    // Elapsed time should be at least 50ms
    double elapsed = timer.elapsed_ms();
    assert(elapsed >= 40.0);  // Allow some tolerance
    assert(elapsed < 200.0);  // But not too much

    std::cout << "  Elapsed time: " << elapsed << " ms" << std::endl;
    std::cout << "  Timer basic functionality: PASSED" << std::endl;
}

// =============================================================================
// Test: Timer accumulation
// =============================================================================

void test_timer_accumulation() {
    std::cout << "Testing Timer accumulation..." << std::endl;

    Timer timer;

    // Start and stop multiple times
    timer.start();
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    timer.stop();

    double first_elapsed = timer.elapsed_ms();

    timer.start();
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    timer.stop();

    double total_elapsed = timer.elapsed_ms();

    // Total should be roughly double the first measurement
    assert(total_elapsed >= first_elapsed);
    assert(total_elapsed >= 30.0);  // At least 30ms total

    std::cout << "  First segment: " << first_elapsed << " ms" << std::endl;
    std::cout << "  Total accumulated: " << total_elapsed << " ms" << std::endl;
    std::cout << "  Timer accumulation: PASSED" << std::endl;
}

// =============================================================================
// Test: Timer reset and restart
// =============================================================================

void test_timer_reset() {
    std::cout << "Testing Timer reset and restart..." << std::endl;

    Timer timer;

    // Accumulate some time
    timer.start();
    std::this_thread::sleep_for(std::chrono::milliseconds(30));
    timer.stop();

    assert(timer.elapsed_ms() >= 20.0);

    // Reset
    timer.reset();
    assert(!timer.is_running());
    assert(approx_equal(timer.elapsed_seconds(), 0.0));

    // Restart
    timer.restart();
    assert(timer.is_running());
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    timer.stop();

    // Should only have time since restart
    assert(timer.elapsed_ms() >= 10.0);
    assert(timer.elapsed_ms() < 100.0);

    std::cout << "  Timer reset and restart: PASSED" << std::endl;
}

// =============================================================================
// Test: ScopedTimer RAII
// =============================================================================

void test_scoped_timer() {
    std::cout << "Testing ScopedTimer RAII..." << std::endl;

    Timer timer;

    {
        ScopedTimer scoped(timer);
        assert(timer.is_running());
        std::this_thread::sleep_for(std::chrono::milliseconds(25));
    }

    // Timer should be stopped when ScopedTimer goes out of scope
    assert(!timer.is_running());
    assert(timer.elapsed_ms() >= 15.0);

    std::cout << "  Elapsed in scope: " << timer.elapsed_ms() << " ms" << std::endl;
    std::cout << "  ScopedTimer RAII: PASSED" << std::endl;
}

// =============================================================================
// Test: Timer unit conversions
// =============================================================================

void test_timer_units() {
    std::cout << "Testing Timer unit conversions..." << std::endl;

    Timer timer;
    timer.start();
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    timer.stop();

    double seconds = timer.elapsed_seconds();
    double ms = timer.elapsed_ms();
    double us = timer.elapsed_us();
    double ns = timer.elapsed_ns();

    // Verify conversions
    assert(approx_equal(ms, seconds * 1000.0, 0.001));
    assert(approx_equal(us, seconds * 1000000.0, 1.0));
    assert(approx_equal(ns, seconds * 1000000000.0, 1000.0));

    std::cout << "  Seconds: " << seconds << std::endl;
    std::cout << "  Milliseconds: " << ms << std::endl;
    std::cout << "  Microseconds: " << us << std::endl;
    std::cout << "  Timer unit conversions: PASSED" << std::endl;
}

// =============================================================================
// Test: Profiler manual recording
// =============================================================================

void test_profiler_record() {
    std::cout << "Testing Profiler manual recording..." << std::endl;

    Profiler profiler;

    // Record some timings manually
    profiler.record("operation_a", 0.1);
    profiler.record("operation_a", 0.2);
    profiler.record("operation_a", 0.15);
    profiler.record("operation_b", 0.5);

    assert(profiler.size() == 2);
    assert(!profiler.empty());

    const ProfileEntry* entry_a = profiler.get_entry("operation_a");
    assert(entry_a != nullptr);
    assert(entry_a->call_count == 3);
    assert(approx_equal(entry_a->total_time, 0.45, 0.001));
    assert(approx_equal(entry_a->min_time, 0.1, 0.001));
    assert(approx_equal(entry_a->max_time, 0.2, 0.001));
    assert(approx_equal(entry_a->average_time(), 0.15, 0.001));

    const ProfileEntry* entry_b = profiler.get_entry("operation_b");
    assert(entry_b != nullptr);
    assert(entry_b->call_count == 1);
    assert(approx_equal(entry_b->total_time, 0.5, 0.001));

    std::cout << "  Profiler manual recording: PASSED" << std::endl;
}

// =============================================================================
// Test: Profiler text serialization
// =============================================================================

void test_profiler_text_serialization() {
    std::cout << "Testing Profiler text serialization..." << std::endl;

    Profiler profiler;

    // Record some data
    profiler.record("training", 1.5);
    profiler.record("training", 1.6);
    profiler.record("inference", 0.05);
    profiler.record("inference", 0.04);
    profiler.record("inference", 0.06);
    profiler.record("data_loading", 0.3);

    // Save to text file
    profiler.save_text("temp_profiler.txt");

    // Load into a new profiler
    Profiler loaded;
    loaded.load_text("temp_profiler.txt");

    // Verify
    assert(loaded.size() == 3);

    const ProfileEntry* training = loaded.get_entry("training");
    assert(training != nullptr);
    assert(training->call_count == 2);
    assert(approx_equal(training->total_time, 3.1, 0.0001));
    assert(approx_equal(training->min_time, 1.5, 0.0001));
    assert(approx_equal(training->max_time, 1.6, 0.0001));

    const ProfileEntry* inference = loaded.get_entry("inference");
    assert(inference != nullptr);
    assert(inference->call_count == 3);
    assert(approx_equal(inference->total_time, 0.15, 0.0001));

    const ProfileEntry* data_loading = loaded.get_entry("data_loading");
    assert(data_loading != nullptr);
    assert(data_loading->call_count == 1);

    // Cleanup
    std::remove("temp_profiler.txt");

    std::cout << "  Profiler text serialization: PASSED" << std::endl;
}

// =============================================================================
// Test: Profiler binary serialization
// =============================================================================

void test_profiler_binary_serialization() {
    std::cout << "Testing Profiler binary serialization..." << std::endl;

    Profiler profiler;

    // Record detailed data
    profiler.record("matrix_multiply", 0.123456789);
    profiler.record("matrix_multiply", 0.234567890);
    profiler.record("vector_add", 0.000123456);
    profiler.record("gradient_descent", 5.5);
    profiler.record("gradient_descent", 5.6);
    profiler.record("gradient_descent", 5.4);

    // Save to binary file
    profiler.save_binary("temp_profiler.bin");

    // Load into a new profiler
    Profiler loaded;
    loaded.load_binary("temp_profiler.bin");

    // Verify exact values (binary should preserve precision)
    assert(loaded.size() == 3);

    const ProfileEntry* mm = loaded.get_entry("matrix_multiply");
    assert(mm != nullptr);
    assert(mm->call_count == 2);
    assert(mm->total_time == 0.123456789 + 0.234567890);
    assert(mm->min_time == 0.123456789);
    assert(mm->max_time == 0.234567890);

    const ProfileEntry* va = loaded.get_entry("vector_add");
    assert(va != nullptr);
    assert(va->call_count == 1);
    assert(va->total_time == 0.000123456);

    const ProfileEntry* gd = loaded.get_entry("gradient_descent");
    assert(gd != nullptr);
    assert(gd->call_count == 3);
    assert(approx_equal(gd->total_time, 16.5, 0.0001));

    // Cleanup
    std::remove("temp_profiler.bin");

    std::cout << "  Profiler binary serialization: PASSED" << std::endl;
}

// =============================================================================
// Test: Profiler sorted entries
// =============================================================================

void test_profiler_sorted_entries() {
    std::cout << "Testing Profiler sorted entries..." << std::endl;

    Profiler profiler;

    profiler.record("fast_op", 0.001);
    profiler.record("medium_op", 0.1);
    profiler.record("slow_op", 1.0);

    auto sorted = profiler.get_entries_sorted_by_time();

    assert(sorted.size() == 3);
    assert(sorted[0].name == "slow_op");
    assert(sorted[1].name == "medium_op");
    assert(sorted[2].name == "fast_op");

    std::cout << "  Profiler sorted entries: PASSED" << std::endl;
}

// =============================================================================
// Test: Profiler total time
// =============================================================================

void test_profiler_total_time() {
    std::cout << "Testing Profiler total time..." << std::endl;

    Profiler profiler;

    profiler.record("op1", 1.0);
    profiler.record("op2", 2.0);
    profiler.record("op3", 3.0);

    double total = profiler.get_total_time();
    assert(approx_equal(total, 6.0, 0.001));

    std::cout << "  Total time: " << total << " seconds" << std::endl;
    std::cout << "  Profiler total time: PASSED" << std::endl;
}

// =============================================================================
// Test: Profiler clear
// =============================================================================

void test_profiler_clear() {
    std::cout << "Testing Profiler clear..." << std::endl;

    Profiler profiler;

    profiler.record("op1", 1.0);
    profiler.record("op2", 2.0);

    assert(profiler.size() == 2);
    assert(!profiler.empty());

    profiler.clear();

    assert(profiler.size() == 0);
    assert(profiler.empty());
    assert(profiler.get_entry("op1") == nullptr);

    std::cout << "  Profiler clear: PASSED" << std::endl;
}

// =============================================================================
// Test: Profiler report generation
// =============================================================================

void test_profiler_report() {
    std::cout << "Testing Profiler report generation..." << std::endl;

    Profiler profiler;

    profiler.record("training_epoch", 10.0);
    profiler.record("training_epoch", 9.5);
    profiler.record("training_epoch", 10.2);
    profiler.record("validation", 2.0);
    profiler.record("checkpoint_save", 0.5);

    std::string report = profiler.generate_report();

    // Verify report contains expected sections
    assert(report.find("Profiler Report") != std::string::npos);
    assert(report.find("training_epoch") != std::string::npos);
    assert(report.find("validation") != std::string::npos);
    assert(report.find("checkpoint_save") != std::string::npos);
    assert(report.find("TOTAL") != std::string::npos);

    std::cout << "\n" << report << std::endl;
    std::cout << "  Profiler report generation: PASSED" << std::endl;
}

// =============================================================================
// Test: Profiler scoped profile
// =============================================================================

void test_profiler_scoped_profile() {
    std::cout << "Testing Profiler scoped profile..." << std::endl;

    Profiler profiler;

    {
        Profiler::ScopedProfile scope(profiler, "scoped_section");
        std::this_thread::sleep_for(std::chrono::milliseconds(25));
    }

    const ProfileEntry* entry = profiler.get_entry("scoped_section");
    assert(entry != nullptr);
    assert(entry->call_count == 1);
    assert(entry->total_time >= 0.015);  // At least 15ms

    std::cout << "  Scoped section time: " << (entry->total_time * 1000.0) << " ms" << std::endl;
    std::cout << "  Profiler scoped profile: PASSED" << std::endl;
}

// =============================================================================
// Test: Round-trip text serialization with special characters
// =============================================================================

void test_profiler_text_special_chars() {
    std::cout << "Testing Profiler text serialization with special names..." << std::endl;

    Profiler profiler;

    profiler.record("layer_1/conv2d", 0.1);
    profiler.record("layer_2/batch_norm", 0.05);
    profiler.record("optimizer::step", 0.2);

    profiler.save_text("temp_profiler_special.txt");

    Profiler loaded;
    loaded.load_text("temp_profiler_special.txt");

    assert(loaded.size() == 3);
    assert(loaded.get_entry("layer_1/conv2d") != nullptr);
    assert(loaded.get_entry("layer_2/batch_norm") != nullptr);
    assert(loaded.get_entry("optimizer::step") != nullptr);

    std::remove("temp_profiler_special.txt");

    std::cout << "  Profiler text special chars: PASSED" << std::endl;
}

// =============================================================================
// Test: Empty profiler serialization
// =============================================================================

void test_profiler_empty_serialization() {
    std::cout << "Testing empty Profiler serialization..." << std::endl;

    Profiler profiler;
    assert(profiler.empty());

    // Text format
    profiler.save_text("temp_empty_profiler.txt");
    Profiler loaded_text;
    loaded_text.load_text("temp_empty_profiler.txt");
    assert(loaded_text.empty());

    // Binary format
    profiler.save_binary("temp_empty_profiler.bin");
    Profiler loaded_binary;
    loaded_binary.load_binary("temp_empty_profiler.bin");
    assert(loaded_binary.empty());

    std::remove("temp_empty_profiler.txt");
    std::remove("temp_empty_profiler.bin");

    std::cout << "  Empty Profiler serialization: PASSED" << std::endl;
}

// =============================================================================
// Main
// =============================================================================

int main() {
    std::cout << "\n=== Profiler and Timer Tests ===" << std::endl;
    std::cout << std::endl;

    try {
        // Timer tests
        test_timer_basic();
        test_timer_accumulation();
        test_timer_reset();
        test_scoped_timer();
        test_timer_units();

        std::cout << std::endl;

        // Profiler tests
        test_profiler_record();
        test_profiler_text_serialization();
        test_profiler_binary_serialization();
        test_profiler_sorted_entries();
        test_profiler_total_time();
        test_profiler_clear();
        test_profiler_report();
        test_profiler_scoped_profile();
        test_profiler_text_special_chars();
        test_profiler_empty_serialization();

        std::cout << std::endl;
        std::cout << "=== All Profiler and Timer Tests PASSED ===" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        return 1;
    }
}
