#pragma once

#include "../core/types.h"
#include <opencv2/core.hpp>
#include <string>
#include <vector>
#include <stdexcept>

namespace psi {
    namespace vision {

        // Color space enumeration
        enum class ColorSpace : core::u8 {
            Grayscale = 0,
            BGR = 1,
            RGB = 2,
            HSV = 3,
            HLS = 4,
            Lab = 5,
            YCrCb = 6,
            Unknown = 255
        };

        // Interpolation methods
        enum class Interpolation : core::u8 {
            Nearest = 0,
            Linear = 1,
            Cubic = 2,
            Area = 3,
            Lanczos = 4
        };

        // Border types for padding
        enum class BorderType : core::u8 {
            Constant = 0,
            Replicate = 1,
            Reflect = 2,
            Wrap = 3,
            Reflect101 = 4
        };

        // Image class - wrapper around cv::Mat
        class Image {
        public:
            // Constructors
            Image() = default;

            Image(core::usize width, core::usize height, core::usize channels = 3, core::u8 fill = 0)
                : mat_(static_cast<int>(height), static_cast<int>(width),
                       channels == 1 ? CV_8UC1 : (channels == 3 ? CV_8UC3 : CV_8UC4),
                       cv::Scalar::all(fill)) {}

            Image(const cv::Mat& mat) : mat_(mat) {}

            Image(cv::Mat&& mat) : mat_(std::move(mat)) {}

            // Create from raw data (copies data)
            Image(const core::u8* data, core::usize width, core::usize height, core::usize channels = 3) {
                int type = channels == 1 ? CV_8UC1 : (channels == 3 ? CV_8UC3 : CV_8UC4);
                mat_ = cv::Mat(static_cast<int>(height), static_cast<int>(width), type);
                std::memcpy(mat_.data, data, width * height * channels);
            }

            // Copy and move
            Image(const Image& other) : mat_(other.mat_.clone()) {}
            Image(Image&& other) noexcept : mat_(std::move(other.mat_)) {}

            Image& operator=(const Image& other) {
                if (this != &other) {
                    mat_ = other.mat_.clone();
                }
                return *this;
            }

            Image& operator=(Image&& other) noexcept {
                if (this != &other) {
                    mat_ = std::move(other.mat_);
                }
                return *this;
            }

            // Properties
            core::usize width() const { return static_cast<core::usize>(mat_.cols); }
            core::usize height() const { return static_cast<core::usize>(mat_.rows); }
            core::usize channels() const { return static_cast<core::usize>(mat_.channels()); }
            core::usize size() const { return width() * height() * channels(); }
            bool empty() const { return mat_.empty(); }

            // Data access
            core::u8* data() { return mat_.data; }
            const core::u8* data() const { return mat_.data; }

            // Pixel access
            core::u8& at(core::usize x, core::usize y, core::usize c = 0) {
                return mat_.at<core::u8>(static_cast<int>(y), static_cast<int>(x) * channels() + c);
            }

            const core::u8& at(core::usize x, core::usize y, core::usize c = 0) const {
                return mat_.at<core::u8>(static_cast<int>(y), static_cast<int>(x) * channels() + c);
            }

            // Get pixel as vector (for multi-channel images)
            std::vector<core::u8> pixel(core::usize x, core::usize y) const {
                std::vector<core::u8> p(channels());
                for (core::usize c = 0; c < channels(); ++c) {
                    p[c] = at(x, y, c);
                }
                return p;
            }

            // Set pixel
            void set_pixel(core::usize x, core::usize y, core::u8 value) {
                for (core::usize c = 0; c < channels(); ++c) {
                    at(x, y, c) = value;
                }
            }

            void set_pixel(core::usize x, core::usize y, const std::vector<core::u8>& values) {
                for (core::usize c = 0; c < std::min(channels(), values.size()); ++c) {
                    at(x, y, c) = values[c];
                }
            }

            // Fill with value
            void fill(core::u8 value) {
                mat_ = cv::Scalar::all(value);
            }

            void fill(const std::vector<core::u8>& values) {
                if (channels() == 1 && !values.empty()) {
                    mat_ = cv::Scalar(values[0]);
                } else if (channels() == 3 && values.size() >= 3) {
                    mat_ = cv::Scalar(values[0], values[1], values[2]);
                } else if (channels() == 4 && values.size() >= 4) {
                    mat_ = cv::Scalar(values[0], values[1], values[2], values[3]);
                }
            }

            // Clone
            Image clone() const {
                return Image(mat_.clone());
            }

            // Access underlying cv::Mat
            cv::Mat& mat() { return mat_; }
            const cv::Mat& mat() const { return mat_; }

            // Region of interest (ROI)
            Image roi(core::usize x, core::usize y, core::usize w, core::usize h) const {
                cv::Rect rect(static_cast<int>(x), static_cast<int>(y),
                              static_cast<int>(w), static_cast<int>(h));
                return Image(mat_(rect).clone());
            }

            // Set ROI
            void set_roi(core::usize x, core::usize y, const Image& src) {
                cv::Rect rect(static_cast<int>(x), static_cast<int>(y),
                              static_cast<int>(src.width()), static_cast<int>(src.height()));
                src.mat_.copyTo(mat_(rect));
            }

        private:
            cv::Mat mat_;
        };

        // Floating-point image for computations
        class ImageF {
        public:
            ImageF() = default;

            ImageF(core::usize width, core::usize height, core::usize channels = 3, core::f32 fill = 0.0f)
                : mat_(static_cast<int>(height), static_cast<int>(width),
                       channels == 1 ? CV_32FC1 : (channels == 3 ? CV_32FC3 : CV_32FC4),
                       cv::Scalar::all(fill)) {}

            ImageF(const cv::Mat& mat) : mat_(mat) {}

            // Convert from Image (u8)
            explicit ImageF(const Image& img) {
                img.mat().convertTo(mat_, CV_32F, 1.0 / 255.0);
            }

            // Convert to Image (u8)
            Image to_image() const {
                cv::Mat result;
                mat_.convertTo(result, CV_8U, 255.0);
                return Image(result);
            }

            // Properties
            core::usize width() const { return static_cast<core::usize>(mat_.cols); }
            core::usize height() const { return static_cast<core::usize>(mat_.rows); }
            core::usize channels() const { return static_cast<core::usize>(mat_.channels()); }
            bool empty() const { return mat_.empty(); }

            // Data access
            core::f32* data() { return reinterpret_cast<core::f32*>(mat_.data); }
            const core::f32* data() const { return reinterpret_cast<const core::f32*>(mat_.data); }

            cv::Mat& mat() { return mat_; }
            const cv::Mat& mat() const { return mat_; }

        private:
            cv::Mat mat_;
        };

    } // namespace vision
} // namespace psi
