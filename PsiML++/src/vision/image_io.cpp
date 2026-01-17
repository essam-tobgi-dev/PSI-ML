#include "../../include/vision/image_io.h"
#include <opencv2/imgcodecs.hpp>
#include <fstream>

namespace psi {
    namespace vision {

        Image ImageIO::load(const std::string& path, ImageReadMode mode) {
            cv::Mat mat = cv::imread(path, static_cast<int>(mode));
            if (mat.empty()) {
                throw std::runtime_error("Failed to load image: " + path);
            }
            return Image(mat);
        }

        Image ImageIO::load_grayscale(const std::string& path) {
            return load(path, ImageReadMode::Grayscale);
        }

        Image ImageIO::load_unchanged(const std::string& path) {
            return load(path, ImageReadMode::Unchanged);
        }

        bool ImageIO::save(const std::string& path, const Image& image) {
            if (image.empty()) {
                return false;
            }
            return cv::imwrite(path, image.mat());
        }

        bool ImageIO::save_jpeg(const std::string& path, const Image& image, const JpegParams& params) {
            if (image.empty()) {
                return false;
            }
            std::vector<int> compression_params = {
                cv::IMWRITE_JPEG_QUALITY, params.quality
            };
            return cv::imwrite(path, image.mat(), compression_params);
        }

        bool ImageIO::save_png(const std::string& path, const Image& image, const PngParams& params) {
            if (image.empty()) {
                return false;
            }
            std::vector<int> compression_params = {
                cv::IMWRITE_PNG_COMPRESSION, params.compression
            };
            return cv::imwrite(path, image.mat(), compression_params);
        }

        std::vector<core::u8> ImageIO::encode(const Image& image, const std::string& ext) {
            std::vector<core::u8> buffer;
            if (!image.empty()) {
                cv::imencode(ext, image.mat(), buffer);
            }
            return buffer;
        }

        Image ImageIO::decode(const std::vector<core::u8>& buffer, ImageReadMode mode) {
            cv::Mat mat = cv::imdecode(buffer, static_cast<int>(mode));
            if (mat.empty()) {
                throw std::runtime_error("Failed to decode image from buffer");
            }
            return Image(mat);
        }

        bool ImageIO::is_valid_image(const std::string& path) {
            std::ifstream file(path, std::ios::binary);
            if (!file.is_open()) {
                return false;
            }

            // Try to load the image
            cv::Mat mat = cv::imread(path, cv::IMREAD_UNCHANGED);
            return !mat.empty();
        }

        std::vector<std::string> ImageIO::supported_extensions() {
            return {
                ".bmp", ".dib",
                ".jpeg", ".jpg", ".jpe",
                ".jp2",
                ".png",
                ".webp",
                ".pbm", ".pgm", ".ppm", ".pxm", ".pnm",
                ".sr", ".ras",
                ".tiff", ".tif",
                ".exr",
                ".hdr", ".pic"
            };
        }

    } // namespace vision
} // namespace psi
