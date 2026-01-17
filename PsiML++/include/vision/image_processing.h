#pragma once

#include "image.h"

namespace psi {
    namespace vision {

        // Threshold types
        enum class ThresholdType : core::u8 {
            Binary = 0,
            BinaryInv = 1,
            Truncate = 2,
            ToZero = 3,
            ToZeroInv = 4,
            Otsu = 8
        };

        // Morphological operations
        enum class MorphOp : core::u8 {
            Erode = 0,
            Dilate = 1,
            Open = 2,
            Close = 3,
            Gradient = 4,
            TopHat = 5,
            BlackHat = 6
        };

        // Morphological shapes
        enum class MorphShape : core::u8 {
            Rect = 0,
            Cross = 1,
            Ellipse = 2
        };

        // Image processing functions
        class ImageProcessing {
        public:
            // =========================================================================
            // Color conversion
            // =========================================================================

            static Image to_grayscale(const Image& src);
            static Image to_bgr(const Image& src);
            static Image to_rgb(const Image& src);
            static Image to_hsv(const Image& src);
            static Image to_hls(const Image& src);
            static Image bgr_to_rgb(const Image& src);
            static Image rgb_to_bgr(const Image& src);

            // =========================================================================
            // Geometric transforms
            // =========================================================================

            static Image resize(const Image& src, core::usize width, core::usize height,
                               Interpolation interp = Interpolation::Linear);

            static Image resize_scale(const Image& src, double scale_x, double scale_y,
                                     Interpolation interp = Interpolation::Linear);

            static Image rotate(const Image& src, double angle, bool resize_to_fit = true);

            static Image flip_horizontal(const Image& src);
            static Image flip_vertical(const Image& src);

            static Image crop(const Image& src, core::usize x, core::usize y,
                             core::usize width, core::usize height);

            static Image pad(const Image& src, core::usize top, core::usize bottom,
                            core::usize left, core::usize right,
                            BorderType border = BorderType::Constant,
                            core::u8 value = 0);

            // =========================================================================
            // Filtering
            // =========================================================================

            static Image blur(const Image& src, core::usize ksize);
            static Image gaussian_blur(const Image& src, core::usize ksize, double sigma = 0);
            static Image median_blur(const Image& src, core::usize ksize);
            static Image bilateral_filter(const Image& src, core::i32 d, double sigma_color, double sigma_space);

            static Image sharpen(const Image& src);
            static Image emboss(const Image& src);

            // Custom kernel convolution
            static Image convolve(const Image& src, const std::vector<std::vector<float>>& kernel);

            // =========================================================================
            // Edge detection
            // =========================================================================

            static Image canny(const Image& src, double threshold1, double threshold2,
                              core::usize aperture = 3);

            static Image sobel(const Image& src, core::i32 dx, core::i32 dy, core::usize ksize = 3);
            static Image laplacian(const Image& src, core::usize ksize = 3);

            // =========================================================================
            // Thresholding
            // =========================================================================

            static Image threshold(const Image& src, double thresh, double maxval,
                                  ThresholdType type = ThresholdType::Binary);

            static Image adaptive_threshold(const Image& src, double maxval, core::usize block_size,
                                           double C, bool gaussian = true);

            // =========================================================================
            // Morphological operations
            // =========================================================================

            static Image morphology(const Image& src, MorphOp op, core::usize ksize,
                                   MorphShape shape = MorphShape::Rect);

            static Image erode(const Image& src, core::usize ksize,
                              MorphShape shape = MorphShape::Rect);

            static Image dilate(const Image& src, core::usize ksize,
                               MorphShape shape = MorphShape::Rect);

            // =========================================================================
            // Histogram operations
            // =========================================================================

            static Image equalize_histogram(const Image& src);
            static std::vector<core::i32> calc_histogram(const Image& src, core::usize channel = 0);

            // =========================================================================
            // Normalization
            // =========================================================================

            static Image normalize(const Image& src, double alpha = 0, double beta = 255);
            static ImageF normalize_float(const Image& src);

            // =========================================================================
            // Arithmetic operations
            // =========================================================================

            static Image add(const Image& src1, const Image& src2);
            static Image subtract(const Image& src1, const Image& src2);
            static Image multiply(const Image& src, double scale);
            static Image blend(const Image& src1, const Image& src2, double alpha);
            static Image abs_diff(const Image& src1, const Image& src2);

            // =========================================================================
            // Bitwise operations
            // =========================================================================

            static Image bitwise_and(const Image& src1, const Image& src2);
            static Image bitwise_or(const Image& src1, const Image& src2);
            static Image bitwise_xor(const Image& src1, const Image& src2);
            static Image bitwise_not(const Image& src);

            // Apply mask
            static Image apply_mask(const Image& src, const Image& mask);

            // =========================================================================
            // Contour operations
            // =========================================================================

            struct Contour {
                std::vector<std::pair<core::i32, core::i32>> points;
                double area;
                double perimeter;
            };

            static std::vector<Contour> find_contours(const Image& src);
            static Image draw_contours(const Image& src, const std::vector<Contour>& contours,
                                       core::i32 idx = -1, core::u8 r = 0, core::u8 g = 255, core::u8 b = 0,
                                       core::i32 thickness = 1);
        };

    } // namespace vision
} // namespace psi
