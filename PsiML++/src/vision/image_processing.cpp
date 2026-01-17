#include "../../include/vision/image_processing.h"
#include <opencv2/imgproc.hpp>

namespace psi {
    namespace vision {

        namespace {
            int to_cv_interpolation(Interpolation interp) {
                switch (interp) {
                    case Interpolation::Nearest: return cv::INTER_NEAREST;
                    case Interpolation::Linear: return cv::INTER_LINEAR;
                    case Interpolation::Cubic: return cv::INTER_CUBIC;
                    case Interpolation::Area: return cv::INTER_AREA;
                    case Interpolation::Lanczos: return cv::INTER_LANCZOS4;
                    default: return cv::INTER_LINEAR;
                }
            }

            int to_cv_border(BorderType border) {
                switch (border) {
                    case BorderType::Constant: return cv::BORDER_CONSTANT;
                    case BorderType::Replicate: return cv::BORDER_REPLICATE;
                    case BorderType::Reflect: return cv::BORDER_REFLECT;
                    case BorderType::Wrap: return cv::BORDER_WRAP;
                    case BorderType::Reflect101: return cv::BORDER_REFLECT101;
                    default: return cv::BORDER_CONSTANT;
                }
            }

            int to_cv_morph_shape(MorphShape shape) {
                switch (shape) {
                    case MorphShape::Rect: return cv::MORPH_RECT;
                    case MorphShape::Cross: return cv::MORPH_CROSS;
                    case MorphShape::Ellipse: return cv::MORPH_ELLIPSE;
                    default: return cv::MORPH_RECT;
                }
            }
        }

        // =========================================================================
        // Color conversion
        // =========================================================================

        Image ImageProcessing::to_grayscale(const Image& src) {
            if (src.channels() == 1) return src.clone();
            cv::Mat result;
            cv::cvtColor(src.mat(), result, cv::COLOR_BGR2GRAY);
            return Image(result);
        }

        Image ImageProcessing::to_bgr(const Image& src) {
            if (src.channels() == 3) return src.clone();
            cv::Mat result;
            cv::cvtColor(src.mat(), result, cv::COLOR_GRAY2BGR);
            return Image(result);
        }

        Image ImageProcessing::to_rgb(const Image& src) {
            cv::Mat result;
            if (src.channels() == 1) {
                cv::cvtColor(src.mat(), result, cv::COLOR_GRAY2RGB);
            } else {
                cv::cvtColor(src.mat(), result, cv::COLOR_BGR2RGB);
            }
            return Image(result);
        }

        Image ImageProcessing::to_hsv(const Image& src) {
            cv::Mat result;
            cv::cvtColor(src.mat(), result, cv::COLOR_BGR2HSV);
            return Image(result);
        }

        Image ImageProcessing::to_hls(const Image& src) {
            cv::Mat result;
            cv::cvtColor(src.mat(), result, cv::COLOR_BGR2HLS);
            return Image(result);
        }

        Image ImageProcessing::bgr_to_rgb(const Image& src) {
            cv::Mat result;
            cv::cvtColor(src.mat(), result, cv::COLOR_BGR2RGB);
            return Image(result);
        }

        Image ImageProcessing::rgb_to_bgr(const Image& src) {
            cv::Mat result;
            cv::cvtColor(src.mat(), result, cv::COLOR_RGB2BGR);
            return Image(result);
        }

        // =========================================================================
        // Geometric transforms
        // =========================================================================

        Image ImageProcessing::resize(const Image& src, core::usize width, core::usize height,
                                      Interpolation interp) {
            cv::Mat result;
            cv::resize(src.mat(), result, cv::Size(static_cast<int>(width), static_cast<int>(height)),
                       0, 0, to_cv_interpolation(interp));
            return Image(result);
        }

        Image ImageProcessing::resize_scale(const Image& src, double scale_x, double scale_y,
                                            Interpolation interp) {
            cv::Mat result;
            cv::resize(src.mat(), result, cv::Size(), scale_x, scale_y, to_cv_interpolation(interp));
            return Image(result);
        }

        Image ImageProcessing::rotate(const Image& src, double angle, bool resize_to_fit) {
            cv::Point2f center(src.width() / 2.0f, src.height() / 2.0f);
            cv::Mat rot_mat = cv::getRotationMatrix2D(center, angle, 1.0);

            cv::Size new_size(src.width(), src.height());
            if (resize_to_fit) {
                double abs_cos = std::abs(rot_mat.at<double>(0, 0));
                double abs_sin = std::abs(rot_mat.at<double>(0, 1));
                new_size.width = static_cast<int>(src.height() * abs_sin + src.width() * abs_cos);
                new_size.height = static_cast<int>(src.height() * abs_cos + src.width() * abs_sin);
                rot_mat.at<double>(0, 2) += (new_size.width - src.width()) / 2.0;
                rot_mat.at<double>(1, 2) += (new_size.height - src.height()) / 2.0;
            }

            cv::Mat result;
            cv::warpAffine(src.mat(), result, rot_mat, new_size);
            return Image(result);
        }

        Image ImageProcessing::flip_horizontal(const Image& src) {
            cv::Mat result;
            cv::flip(src.mat(), result, 1);
            return Image(result);
        }

        Image ImageProcessing::flip_vertical(const Image& src) {
            cv::Mat result;
            cv::flip(src.mat(), result, 0);
            return Image(result);
        }

        Image ImageProcessing::crop(const Image& src, core::usize x, core::usize y,
                                    core::usize width, core::usize height) {
            return src.roi(x, y, width, height);
        }

        Image ImageProcessing::pad(const Image& src, core::usize top, core::usize bottom,
                                   core::usize left, core::usize right,
                                   BorderType border, core::u8 value) {
            cv::Mat result;
            cv::copyMakeBorder(src.mat(), result,
                               static_cast<int>(top), static_cast<int>(bottom),
                               static_cast<int>(left), static_cast<int>(right),
                               to_cv_border(border), cv::Scalar::all(value));
            return Image(result);
        }

        // =========================================================================
        // Filtering
        // =========================================================================

        Image ImageProcessing::blur(const Image& src, core::usize ksize) {
            cv::Mat result;
            cv::blur(src.mat(), result, cv::Size(static_cast<int>(ksize), static_cast<int>(ksize)));
            return Image(result);
        }

        Image ImageProcessing::gaussian_blur(const Image& src, core::usize ksize, double sigma) {
            cv::Mat result;
            int k = static_cast<int>(ksize);
            if (k % 2 == 0) k++;  // Must be odd
            cv::GaussianBlur(src.mat(), result, cv::Size(k, k), sigma);
            return Image(result);
        }

        Image ImageProcessing::median_blur(const Image& src, core::usize ksize) {
            cv::Mat result;
            int k = static_cast<int>(ksize);
            if (k % 2 == 0) k++;  // Must be odd
            cv::medianBlur(src.mat(), result, k);
            return Image(result);
        }

        Image ImageProcessing::bilateral_filter(const Image& src, core::i32 d,
                                                 double sigma_color, double sigma_space) {
            cv::Mat result;
            cv::bilateralFilter(src.mat(), result, d, sigma_color, sigma_space);
            return Image(result);
        }

        Image ImageProcessing::sharpen(const Image& src) {
            std::vector<std::vector<float>> kernel = {
                {0, -1, 0},
                {-1, 5, -1},
                {0, -1, 0}
            };
            return convolve(src, kernel);
        }

        Image ImageProcessing::emboss(const Image& src) {
            std::vector<std::vector<float>> kernel = {
                {-2, -1, 0},
                {-1, 1, 1},
                {0, 1, 2}
            };
            return convolve(src, kernel);
        }

        Image ImageProcessing::convolve(const Image& src, const std::vector<std::vector<float>>& kernel) {
            int rows = static_cast<int>(kernel.size());
            int cols = static_cast<int>(kernel[0].size());
            cv::Mat cv_kernel(rows, cols, CV_32F);
            for (int i = 0; i < rows; ++i) {
                for (int j = 0; j < cols; ++j) {
                    cv_kernel.at<float>(i, j) = kernel[i][j];
                }
            }

            cv::Mat result;
            cv::filter2D(src.mat(), result, -1, cv_kernel);
            return Image(result);
        }

        // =========================================================================
        // Edge detection
        // =========================================================================

        Image ImageProcessing::canny(const Image& src, double threshold1, double threshold2,
                                     core::usize aperture) {
            cv::Mat gray;
            if (src.channels() > 1) {
                cv::cvtColor(src.mat(), gray, cv::COLOR_BGR2GRAY);
            } else {
                gray = src.mat();
            }

            cv::Mat result;
            cv::Canny(gray, result, threshold1, threshold2, static_cast<int>(aperture));
            return Image(result);
        }

        Image ImageProcessing::sobel(const Image& src, core::i32 dx, core::i32 dy, core::usize ksize) {
            cv::Mat gray;
            if (src.channels() > 1) {
                cv::cvtColor(src.mat(), gray, cv::COLOR_BGR2GRAY);
            } else {
                gray = src.mat();
            }

            cv::Mat result;
            cv::Sobel(gray, result, CV_8U, dx, dy, static_cast<int>(ksize));
            return Image(result);
        }

        Image ImageProcessing::laplacian(const Image& src, core::usize ksize) {
            cv::Mat gray;
            if (src.channels() > 1) {
                cv::cvtColor(src.mat(), gray, cv::COLOR_BGR2GRAY);
            } else {
                gray = src.mat();
            }

            cv::Mat result;
            cv::Laplacian(gray, result, CV_8U, static_cast<int>(ksize));
            return Image(result);
        }

        // =========================================================================
        // Thresholding
        // =========================================================================

        Image ImageProcessing::threshold(const Image& src, double thresh, double maxval,
                                         ThresholdType type) {
            cv::Mat gray;
            if (src.channels() > 1) {
                cv::cvtColor(src.mat(), gray, cv::COLOR_BGR2GRAY);
            } else {
                gray = src.mat();
            }

            cv::Mat result;
            cv::threshold(gray, result, thresh, maxval, static_cast<int>(type));
            return Image(result);
        }

        Image ImageProcessing::adaptive_threshold(const Image& src, double maxval,
                                                   core::usize block_size, double C, bool gaussian) {
            cv::Mat gray;
            if (src.channels() > 1) {
                cv::cvtColor(src.mat(), gray, cv::COLOR_BGR2GRAY);
            } else {
                gray = src.mat();
            }

            int block = static_cast<int>(block_size);
            if (block % 2 == 0) block++;

            cv::Mat result;
            int method = gaussian ? cv::ADAPTIVE_THRESH_GAUSSIAN_C : cv::ADAPTIVE_THRESH_MEAN_C;
            cv::adaptiveThreshold(gray, result, maxval, method, cv::THRESH_BINARY, block, C);
            return Image(result);
        }

        // =========================================================================
        // Morphological operations
        // =========================================================================

        Image ImageProcessing::morphology(const Image& src, MorphOp op, core::usize ksize,
                                          MorphShape shape) {
            cv::Mat kernel = cv::getStructuringElement(
                to_cv_morph_shape(shape),
                cv::Size(static_cast<int>(ksize), static_cast<int>(ksize)));

            cv::Mat result;
            cv::morphologyEx(src.mat(), result, static_cast<int>(op), kernel);
            return Image(result);
        }

        Image ImageProcessing::erode(const Image& src, core::usize ksize, MorphShape shape) {
            return morphology(src, MorphOp::Erode, ksize, shape);
        }

        Image ImageProcessing::dilate(const Image& src, core::usize ksize, MorphShape shape) {
            return morphology(src, MorphOp::Dilate, ksize, shape);
        }

        // =========================================================================
        // Histogram operations
        // =========================================================================

        Image ImageProcessing::equalize_histogram(const Image& src) {
            if (src.channels() == 1) {
                cv::Mat result;
                cv::equalizeHist(src.mat(), result);
                return Image(result);
            } else {
                // Convert to YCrCb and equalize Y channel
                cv::Mat ycrcb;
                cv::cvtColor(src.mat(), ycrcb, cv::COLOR_BGR2YCrCb);

                std::vector<cv::Mat> channels;
                cv::split(ycrcb, channels);
                cv::equalizeHist(channels[0], channels[0]);
                cv::merge(channels, ycrcb);

                cv::Mat result;
                cv::cvtColor(ycrcb, result, cv::COLOR_YCrCb2BGR);
                return Image(result);
            }
        }

        std::vector<core::i32> ImageProcessing::calc_histogram(const Image& src, core::usize channel) {
            cv::Mat gray;
            if (src.channels() > 1) {
                std::vector<cv::Mat> channels;
                cv::split(src.mat(), channels);
                gray = channels[channel];
            } else {
                gray = src.mat();
            }

            cv::Mat hist;
            int hist_size = 256;
            float range[] = {0, 256};
            const float* hist_range = {range};
            cv::calcHist(&gray, 1, nullptr, cv::Mat(), hist, 1, &hist_size, &hist_range);

            std::vector<core::i32> result(256);
            for (int i = 0; i < 256; ++i) {
                result[i] = static_cast<core::i32>(hist.at<float>(i));
            }
            return result;
        }

        // =========================================================================
        // Normalization
        // =========================================================================

        Image ImageProcessing::normalize(const Image& src, double alpha, double beta) {
            cv::Mat result;
            cv::normalize(src.mat(), result, alpha, beta, cv::NORM_MINMAX, CV_8U);
            return Image(result);
        }

        ImageF ImageProcessing::normalize_float(const Image& src) {
            return ImageF(src);
        }

        // =========================================================================
        // Arithmetic operations
        // =========================================================================

        Image ImageProcessing::add(const Image& src1, const Image& src2) {
            cv::Mat result;
            cv::add(src1.mat(), src2.mat(), result);
            return Image(result);
        }

        Image ImageProcessing::subtract(const Image& src1, const Image& src2) {
            cv::Mat result;
            cv::subtract(src1.mat(), src2.mat(), result);
            return Image(result);
        }

        Image ImageProcessing::multiply(const Image& src, double scale) {
            cv::Mat result;
            src.mat().convertTo(result, -1, scale);
            return Image(result);
        }

        Image ImageProcessing::blend(const Image& src1, const Image& src2, double alpha) {
            cv::Mat result;
            cv::addWeighted(src1.mat(), alpha, src2.mat(), 1.0 - alpha, 0, result);
            return Image(result);
        }

        Image ImageProcessing::abs_diff(const Image& src1, const Image& src2) {
            cv::Mat result;
            cv::absdiff(src1.mat(), src2.mat(), result);
            return Image(result);
        }

        // =========================================================================
        // Bitwise operations
        // =========================================================================

        Image ImageProcessing::bitwise_and(const Image& src1, const Image& src2) {
            cv::Mat result;
            cv::bitwise_and(src1.mat(), src2.mat(), result);
            return Image(result);
        }

        Image ImageProcessing::bitwise_or(const Image& src1, const Image& src2) {
            cv::Mat result;
            cv::bitwise_or(src1.mat(), src2.mat(), result);
            return Image(result);
        }

        Image ImageProcessing::bitwise_xor(const Image& src1, const Image& src2) {
            cv::Mat result;
            cv::bitwise_xor(src1.mat(), src2.mat(), result);
            return Image(result);
        }

        Image ImageProcessing::bitwise_not(const Image& src) {
            cv::Mat result;
            cv::bitwise_not(src.mat(), result);
            return Image(result);
        }

        Image ImageProcessing::apply_mask(const Image& src, const Image& mask) {
            cv::Mat result;
            src.mat().copyTo(result, mask.mat());
            return Image(result);
        }

        // =========================================================================
        // Contour operations
        // =========================================================================

        std::vector<ImageProcessing::Contour> ImageProcessing::find_contours(const Image& src) {
            cv::Mat gray;
            if (src.channels() > 1) {
                cv::cvtColor(src.mat(), gray, cv::COLOR_BGR2GRAY);
            } else {
                gray = src.mat();
            }

            std::vector<std::vector<cv::Point>> cv_contours;
            cv::findContours(gray, cv_contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

            std::vector<Contour> result;
            result.reserve(cv_contours.size());

            for (const auto& cv_contour : cv_contours) {
                Contour contour;
                contour.points.reserve(cv_contour.size());
                for (const auto& pt : cv_contour) {
                    contour.points.emplace_back(pt.x, pt.y);
                }
                contour.area = cv::contourArea(cv_contour);
                contour.perimeter = cv::arcLength(cv_contour, true);
                result.push_back(contour);
            }

            return result;
        }

        Image ImageProcessing::draw_contours(const Image& src, const std::vector<Contour>& contours,
                                              core::i32 idx, core::u8 r, core::u8 g, core::u8 b,
                                              core::i32 thickness) {
            // Convert back to cv::contours format
            std::vector<std::vector<cv::Point>> cv_contours;
            cv_contours.reserve(contours.size());
            for (const auto& contour : contours) {
                std::vector<cv::Point> cv_contour;
                cv_contour.reserve(contour.points.size());
                for (const auto& pt : contour.points) {
                    cv_contour.emplace_back(pt.first, pt.second);
                }
                cv_contours.push_back(cv_contour);
            }

            cv::Mat result = src.mat().clone();
            cv::drawContours(result, cv_contours, idx, cv::Scalar(b, g, r), thickness);
            return Image(result);
        }

    } // namespace vision
} // namespace psi
