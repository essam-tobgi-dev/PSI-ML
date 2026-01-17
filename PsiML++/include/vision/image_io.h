#pragma once

#include "image.h"
#include <string>
#include <vector>

namespace psi {
    namespace vision {

        // Image I/O flags
        enum class ImageReadMode : core::i32 {
            Unchanged = -1,
            Grayscale = 0,
            Color = 1,
            AnyDepth = 2,
            AnyColor = 4
        };

        // Image compression parameters
        struct JpegParams {
            int quality = 95;  // 0-100
        };

        struct PngParams {
            int compression = 3;  // 0-9
        };

        // Image I/O class
        class ImageIO {
        public:
            // Load image from file
            static Image load(const std::string& path, ImageReadMode mode = ImageReadMode::Color);

            // Load image as grayscale
            static Image load_grayscale(const std::string& path);

            // Load image with alpha channel
            static Image load_unchanged(const std::string& path);

            // Save image to file (format determined by extension)
            static bool save(const std::string& path, const Image& image);

            // Save with JPEG parameters
            static bool save_jpeg(const std::string& path, const Image& image, const JpegParams& params = JpegParams());

            // Save with PNG parameters
            static bool save_png(const std::string& path, const Image& image, const PngParams& params = PngParams());

            // Encode image to memory buffer
            static std::vector<core::u8> encode(const Image& image, const std::string& ext = ".png");

            // Decode image from memory buffer
            static Image decode(const std::vector<core::u8>& buffer, ImageReadMode mode = ImageReadMode::Color);

            // Check if file is a valid image
            static bool is_valid_image(const std::string& path);

            // Get supported extensions
            static std::vector<std::string> supported_extensions();
        };

    } // namespace vision
} // namespace psi
