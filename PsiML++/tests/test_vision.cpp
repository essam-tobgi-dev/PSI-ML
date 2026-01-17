#include "../include/vision/image.h"
#include "../include/vision/image_io.h"
#include "../include/vision/image_processing.h"
#include "../include/vision/drawing.h"
#include <iostream>
#include <cassert>
#include <cmath>

using namespace psi::vision;
using namespace psi::core;

// =============================================================================
// Test: Image creation and properties
// =============================================================================

void test_image_creation() {
    std::cout << "Testing Image creation..." << std::endl;

    // Create grayscale image
    Image gray_img(100, 80, 1, 128);
    assert(gray_img.width() == 100);
    assert(gray_img.height() == 80);
    assert(gray_img.channels() == 1);
    assert(!gray_img.empty());

    // Create color image
    Image color_img(200, 150, 3, 0);
    assert(color_img.width() == 200);
    assert(color_img.height() == 150);
    assert(color_img.channels() == 3);

    // Create RGBA image
    Image rgba_img(50, 50, 4, 255);
    assert(rgba_img.channels() == 4);

    std::cout << "  Image creation: PASSED" << std::endl;
}

// =============================================================================
// Test: Image pixel access
// =============================================================================

void test_image_pixel_access() {
    std::cout << "Testing Image pixel access..." << std::endl;

    Image img(10, 10, 3, 0);

    // Set pixel using vector
    img.set_pixel(5, 5, {255, 128, 64});
    auto pixel = img.pixel(5, 5);
    assert(pixel.size() == 3);
    assert(pixel[0] == 255);
    assert(pixel[1] == 128);
    assert(pixel[2] == 64);

    // Set pixel using single value
    img.set_pixel(3, 3, 100);
    pixel = img.pixel(3, 3);
    assert(pixel[0] == 100);
    assert(pixel[1] == 100);
    assert(pixel[2] == 100);

    std::cout << "  Image pixel access: PASSED" << std::endl;
}

// =============================================================================
// Test: Image fill
// =============================================================================

void test_image_fill() {
    std::cout << "Testing Image fill..." << std::endl;

    Image img(10, 10, 3);

    // Fill with single value
    img.fill(200);
    auto pixel = img.pixel(5, 5);
    assert(pixel[0] == 200);
    assert(pixel[1] == 200);
    assert(pixel[2] == 200);

    // Fill with color
    img.fill({100, 150, 200});
    pixel = img.pixel(5, 5);
    assert(pixel[0] == 100);
    assert(pixel[1] == 150);
    assert(pixel[2] == 200);

    std::cout << "  Image fill: PASSED" << std::endl;
}

// =============================================================================
// Test: Image clone
// =============================================================================

void test_image_clone() {
    std::cout << "Testing Image clone..." << std::endl;

    Image original(10, 10, 3, 128);
    original.set_pixel(5, 5, {255, 0, 0});

    Image cloned = original.clone();

    // Modify original
    original.set_pixel(5, 5, {0, 255, 0});

    // Clone should not be affected
    auto pixel = cloned.pixel(5, 5);
    assert(pixel[0] == 255);
    assert(pixel[1] == 0);
    assert(pixel[2] == 0);

    std::cout << "  Image clone: PASSED" << std::endl;
}

// =============================================================================
// Test: Image ROI
// =============================================================================

void test_image_roi() {
    std::cout << "Testing Image ROI..." << std::endl;

    Image img(100, 100, 3, 0);
    img.fill({255, 255, 255});

    // Create ROI
    Image roi = img.roi(10, 10, 20, 20);
    assert(roi.width() == 20);
    assert(roi.height() == 20);

    // Verify ROI content
    auto pixel = roi.pixel(5, 5);
    assert(pixel[0] == 255);

    std::cout << "  Image ROI: PASSED" << std::endl;
}

// =============================================================================
// Test: ImageF conversion
// =============================================================================

void test_imagef_conversion() {
    std::cout << "Testing ImageF conversion..." << std::endl;

    Image img(10, 10, 3, 255);
    ImageF imgf(img);

    assert(imgf.width() == 10);
    assert(imgf.height() == 10);
    assert(imgf.channels() == 3);

    // Convert back
    Image converted = imgf.to_image();
    assert(converted.width() == 10);
    auto pixel = converted.pixel(5, 5);
    assert(pixel[0] == 255);

    std::cout << "  ImageF conversion: PASSED" << std::endl;
}

// =============================================================================
// Test: Color conversion
// =============================================================================

void test_color_conversion() {
    std::cout << "Testing color conversion..." << std::endl;

    // Create a color image
    Image color_img(50, 50, 3, 128);

    // To grayscale
    Image gray = ImageProcessing::to_grayscale(color_img);
    assert(gray.channels() == 1);
    assert(gray.width() == 50);
    assert(gray.height() == 50);

    // Back to BGR
    Image bgr = ImageProcessing::to_bgr(gray);
    assert(bgr.channels() == 3);

    // To HSV
    Image hsv = ImageProcessing::to_hsv(color_img);
    assert(hsv.channels() == 3);

    std::cout << "  Color conversion: PASSED" << std::endl;
}

// =============================================================================
// Test: Resize
// =============================================================================

void test_resize() {
    std::cout << "Testing resize..." << std::endl;

    Image img(100, 80, 3, 128);

    // Resize to specific dimensions
    Image resized = ImageProcessing::resize(img, 200, 160);
    assert(resized.width() == 200);
    assert(resized.height() == 160);

    // Resize by scale
    Image scaled = ImageProcessing::resize_scale(img, 0.5, 0.5);
    assert(scaled.width() == 50);
    assert(scaled.height() == 40);

    std::cout << "  Resize: PASSED" << std::endl;
}

// =============================================================================
// Test: Flip
// =============================================================================

void test_flip() {
    std::cout << "Testing flip..." << std::endl;

    Image img(10, 10, 1, 0);
    img.set_pixel(0, 0, 255);  // Top-left white

    // Horizontal flip
    Image h_flip = ImageProcessing::flip_horizontal(img);
    assert(h_flip.at(9, 0) == 255);  // Should be top-right now

    // Vertical flip
    Image v_flip = ImageProcessing::flip_vertical(img);
    assert(v_flip.at(0, 9) == 255);  // Should be bottom-left now

    std::cout << "  Flip: PASSED" << std::endl;
}

// =============================================================================
// Test: Blur operations
// =============================================================================

void test_blur() {
    std::cout << "Testing blur operations..." << std::endl;

    Image img(50, 50, 3, 128);

    // Box blur
    Image blurred = ImageProcessing::blur(img, 5);
    assert(blurred.width() == img.width());
    assert(!blurred.empty());

    // Gaussian blur
    Image gaussian = ImageProcessing::gaussian_blur(img, 5, 1.0);
    assert(!gaussian.empty());

    // Median blur
    Image median = ImageProcessing::median_blur(img, 5);
    assert(!median.empty());

    std::cout << "  Blur operations: PASSED" << std::endl;
}

// =============================================================================
// Test: Edge detection
// =============================================================================

void test_edge_detection() {
    std::cout << "Testing edge detection..." << std::endl;

    // Create image with a sharp edge
    Image img(100, 100, 1, 0);
    for (usize y = 0; y < 100; ++y) {
        for (usize x = 50; x < 100; ++x) {
            img.set_pixel(x, y, 255);
        }
    }

    // Canny edge detection
    Image canny = ImageProcessing::canny(img, 50, 150);
    assert(canny.channels() == 1);
    assert(!canny.empty());

    // Sobel
    Image sobel = ImageProcessing::sobel(img, 1, 0);
    assert(!sobel.empty());

    // Laplacian
    Image laplacian = ImageProcessing::laplacian(img);
    assert(!laplacian.empty());

    std::cout << "  Edge detection: PASSED" << std::endl;
}

// =============================================================================
// Test: Threshold
// =============================================================================

void test_threshold() {
    std::cout << "Testing threshold..." << std::endl;

    Image img(50, 50, 1);
    // Create gradient
    for (usize y = 0; y < 50; ++y) {
        for (usize x = 0; x < 50; ++x) {
            img.set_pixel(x, y, static_cast<u8>(x * 5));
        }
    }

    // Binary threshold
    Image thresh = ImageProcessing::threshold(img, 128, 255, ThresholdType::Binary);
    assert(!thresh.empty());

    // Check values
    assert(thresh.at(10, 0) == 0);   // Below threshold
    assert(thresh.at(40, 0) == 255); // Above threshold

    std::cout << "  Threshold: PASSED" << std::endl;
}

// =============================================================================
// Test: Morphological operations
// =============================================================================

void test_morphology() {
    std::cout << "Testing morphological operations..." << std::endl;

    // Create binary image with a dot
    Image img(50, 50, 1, 0);
    for (usize y = 20; y < 30; ++y) {
        for (usize x = 20; x < 30; ++x) {
            img.set_pixel(x, y, 255);
        }
    }

    // Erode
    Image eroded = ImageProcessing::erode(img, 3);
    assert(!eroded.empty());

    // Dilate
    Image dilated = ImageProcessing::dilate(img, 3);
    assert(!dilated.empty());

    // Open
    Image opened = ImageProcessing::morphology(img, MorphOp::Open, 3);
    assert(!opened.empty());

    // Close
    Image closed = ImageProcessing::morphology(img, MorphOp::Close, 3);
    assert(!closed.empty());

    std::cout << "  Morphological operations: PASSED" << std::endl;
}

// =============================================================================
// Test: Histogram
// =============================================================================

void test_histogram() {
    std::cout << "Testing histogram..." << std::endl;

    // Create image with known values
    Image img(100, 100, 1, 128);

    auto hist = ImageProcessing::calc_histogram(img);
    assert(hist.size() == 256);
    assert(hist[128] == 10000);  // All pixels are 128

    // Equalize histogram
    Image equalized = ImageProcessing::equalize_histogram(img);
    assert(!equalized.empty());

    std::cout << "  Histogram: PASSED" << std::endl;
}

// =============================================================================
// Test: Arithmetic operations
// =============================================================================

void test_arithmetic() {
    std::cout << "Testing arithmetic operations..." << std::endl;

    Image img1(50, 50, 3, 100);
    Image img2(50, 50, 3, 50);

    // Add
    Image added = ImageProcessing::add(img1, img2);
    auto pixel = added.pixel(25, 25);
    assert(pixel[0] == 150);

    // Subtract
    Image subtracted = ImageProcessing::subtract(img1, img2);
    pixel = subtracted.pixel(25, 25);
    assert(pixel[0] == 50);

    // Blend
    Image blended = ImageProcessing::blend(img1, img2, 0.5);
    pixel = blended.pixel(25, 25);
    assert(pixel[0] == 75);

    std::cout << "  Arithmetic operations: PASSED" << std::endl;
}

// =============================================================================
// Test: Bitwise operations
// =============================================================================

void test_bitwise() {
    std::cout << "Testing bitwise operations..." << std::endl;

    Image img1(10, 10, 1, 0xFF);
    Image img2(10, 10, 1, 0x0F);

    // AND
    Image result = ImageProcessing::bitwise_and(img1, img2);
    assert(result.at(0, 0) == 0x0F);

    // OR
    result = ImageProcessing::bitwise_or(img1, img2);
    assert(result.at(0, 0) == 0xFF);

    // NOT
    result = ImageProcessing::bitwise_not(img2);
    assert(result.at(0, 0) == 0xF0);

    std::cout << "  Bitwise operations: PASSED" << std::endl;
}

// =============================================================================
// Test: Drawing - shapes
// =============================================================================

void test_drawing_shapes() {
    std::cout << "Testing drawing shapes..." << std::endl;

    Image img = Drawing::create_canvas(200, 200, Color::white());
    assert(img.width() == 200);
    assert(img.height() == 200);

    // Draw line
    Drawing::line(img, Point(10, 10), Point(100, 100), Color::red(), 2);

    // Draw rectangle
    Drawing::rectangle(img, Rect(50, 50, 80, 60), Color::blue(), 2);

    // Draw filled rectangle
    Drawing::filled_rectangle(img, Rect(150, 50, 30, 30), Color::green());

    // Draw circle
    Drawing::circle(img, Point(100, 150), 30, Color::magenta(), 2);

    // Draw filled circle
    Drawing::filled_circle(img, Point(50, 150), 20, Color::cyan());

    // Verify image is modified (not all white anymore)
    // Check center of filled green rectangle
    auto pixel = img.pixel(165, 65);
    assert(pixel[0] == 0);    // B
    assert(pixel[1] == 255);  // G
    assert(pixel[2] == 0);    // R

    std::cout << "  Drawing shapes: PASSED" << std::endl;
}

// =============================================================================
// Test: Drawing - text
// =============================================================================

void test_drawing_text() {
    std::cout << "Testing drawing text..." << std::endl;

    Image img = Drawing::create_canvas(300, 100, Color::white());

    // Draw text
    Drawing::text(img, "Hello PsiML++", Point(10, 50), Color::black(), 1.0, 2);

    // Get text size
    auto size = Drawing::get_text_size("Hello", 1.0, 1);
    assert(size.width > 0);
    assert(size.height > 0);

    // Draw text with background
    Drawing::text_with_background(img, "Label", Point(200, 80),
                                  Color::white(), Color::red(), 0.8, 1);

    std::cout << "  Drawing text: PASSED" << std::endl;
}

// =============================================================================
// Test: Drawing - markers
// =============================================================================

void test_drawing_markers() {
    std::cout << "Testing drawing markers..." << std::endl;

    Image img = Drawing::create_canvas(200, 200, Color::white());

    // Draw different markers
    Drawing::marker(img, Point(30, 30), Color::red(), Drawing::MarkerType::Cross, 20);
    Drawing::marker(img, Point(70, 30), Color::green(), Drawing::MarkerType::Star, 20);
    Drawing::marker(img, Point(110, 30), Color::blue(), Drawing::MarkerType::Diamond, 20);
    Drawing::marker(img, Point(150, 30), Color::magenta(), Drawing::MarkerType::Square, 20);

    std::cout << "  Drawing markers: PASSED" << std::endl;
}

// =============================================================================
// Test: Drawing - utility functions
// =============================================================================

void test_drawing_utility() {
    std::cout << "Testing drawing utility functions..." << std::endl;

    Image img = Drawing::create_canvas(200, 200, Color::white());

    // Draw grid
    Drawing::grid(img, 50, 50, Color::gray(), 1);

    // Draw crosshair
    Drawing::crosshair(img, Color::red(), 30, 2);

    std::cout << "  Drawing utility functions: PASSED" << std::endl;
}

// =============================================================================
// Test: Image encode/decode (memory)
// =============================================================================

void test_image_encode_decode() {
    std::cout << "Testing image encode/decode..." << std::endl;

    // Create test image
    Image original(50, 50, 3, 0);
    original.fill({128, 64, 255});

    // Encode to PNG
    auto buffer = ImageIO::encode(original, ".png");
    assert(!buffer.empty());

    // Decode back
    Image decoded = ImageIO::decode(buffer);
    assert(decoded.width() == 50);
    assert(decoded.height() == 50);
    assert(decoded.channels() == 3);

    // Verify pixel values are close (PNG is lossless)
    auto orig_pixel = original.pixel(25, 25);
    auto dec_pixel = decoded.pixel(25, 25);
    assert(orig_pixel[0] == dec_pixel[0]);
    assert(orig_pixel[1] == dec_pixel[1]);
    assert(orig_pixel[2] == dec_pixel[2]);

    std::cout << "  Image encode/decode: PASSED" << std::endl;
}

// =============================================================================
// Test: Contour detection
// =============================================================================

void test_contours() {
    std::cout << "Testing contour detection..." << std::endl;

    // Create binary image with a rectangle
    Image img(100, 100, 1, 0);
    for (usize y = 20; y < 80; ++y) {
        for (usize x = 20; x < 80; ++x) {
            img.set_pixel(x, y, 255);
        }
    }

    // Find contours
    auto contours = ImageProcessing::find_contours(img);
    assert(!contours.empty());
    assert(contours[0].area > 0);
    assert(contours[0].perimeter > 0);

    // Draw contours
    Image color_img = ImageProcessing::to_bgr(img);
    Image with_contours = ImageProcessing::draw_contours(color_img, contours, -1, 0, 255, 0, 2);
    assert(!with_contours.empty());

    std::cout << "  Contour detection: PASSED" << std::endl;
}

// =============================================================================
// Test: Pad operation
// =============================================================================

void test_pad() {
    std::cout << "Testing pad operation..." << std::endl;

    Image img(50, 50, 3, 128);

    // Pad with zeros
    Image padded = ImageProcessing::pad(img, 10, 10, 20, 20, BorderType::Constant, 0);
    assert(padded.width() == 90);   // 50 + 20 + 20
    assert(padded.height() == 70);  // 50 + 10 + 10

    // Check corner is padded value
    auto pixel = padded.pixel(0, 0);
    assert(pixel[0] == 0);

    // Check center is original value
    pixel = padded.pixel(40, 30);
    assert(pixel[0] == 128);

    std::cout << "  Pad operation: PASSED" << std::endl;
}

// =============================================================================
// Test: Supported extensions
// =============================================================================

void test_supported_extensions() {
    std::cout << "Testing supported extensions..." << std::endl;

    auto extensions = ImageIO::supported_extensions();
    assert(!extensions.empty());

    // Should include common formats
    bool has_png = false, has_jpg = false, has_bmp = false;
    for (const auto& ext : extensions) {
        if (ext == ".png") has_png = true;
        if (ext == ".jpg" || ext == ".jpeg") has_jpg = true;
        if (ext == ".bmp") has_bmp = true;
    }
    assert(has_png);
    assert(has_jpg);
    assert(has_bmp);

    std::cout << "  Supported extensions: PASSED" << std::endl;
}

// =============================================================================
// Main
// =============================================================================

int main() {
    std::cout << "\n=== Vision Module Tests ===" << std::endl;
    std::cout << std::endl;

    try {
        // Image tests
        test_image_creation();
        test_image_pixel_access();
        test_image_fill();
        test_image_clone();
        test_image_roi();
        test_imagef_conversion();

        std::cout << std::endl;

        // Color and transform tests
        test_color_conversion();
        test_resize();
        test_flip();
        test_pad();

        std::cout << std::endl;

        // Filter tests
        test_blur();
        test_edge_detection();
        test_threshold();
        test_morphology();

        std::cout << std::endl;

        // Operation tests
        test_histogram();
        test_arithmetic();
        test_bitwise();
        test_contours();

        std::cout << std::endl;

        // Drawing tests
        test_drawing_shapes();
        test_drawing_text();
        test_drawing_markers();
        test_drawing_utility();

        std::cout << std::endl;

        // I/O tests
        test_image_encode_decode();
        test_supported_extensions();

        std::cout << std::endl;
        std::cout << "=== All Vision Module Tests PASSED ===" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        return 1;
    }
}
