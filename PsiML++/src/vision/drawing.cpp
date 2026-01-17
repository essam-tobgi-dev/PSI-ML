#include "../../include/vision/drawing.h"
#include <opencv2/imgproc.hpp>

namespace psi {
    namespace vision {

        namespace {
            cv::Scalar to_cv_color(const Color& color) {
                return cv::Scalar(color.b, color.g, color.r);  // OpenCV uses BGR
            }

            cv::Point to_cv_point(const Point& pt) {
                return cv::Point(pt.x, pt.y);
            }

            cv::Rect to_cv_rect(const Rect& rect) {
                return cv::Rect(rect.x, rect.y, rect.width, rect.height);
            }
        }

        // =========================================================================
        // Basic shapes
        // =========================================================================

        void Drawing::line(Image& img, Point pt1, Point pt2, const Color& color,
                          core::i32 thickness, LineType line_type) {
            cv::line(img.mat(), to_cv_point(pt1), to_cv_point(pt2),
                     to_cv_color(color), thickness, static_cast<int>(line_type));
        }

        void Drawing::arrow(Image& img, Point pt1, Point pt2, const Color& color,
                           core::i32 thickness, core::f64 tip_length, LineType line_type) {
            cv::arrowedLine(img.mat(), to_cv_point(pt1), to_cv_point(pt2),
                            to_cv_color(color), thickness, static_cast<int>(line_type), 0, tip_length);
        }

        void Drawing::rectangle(Image& img, Rect rect, const Color& color,
                               core::i32 thickness, LineType line_type) {
            cv::rectangle(img.mat(), to_cv_rect(rect), to_cv_color(color),
                          thickness, static_cast<int>(line_type));
        }

        void Drawing::rectangle(Image& img, Point pt1, Point pt2, const Color& color,
                               core::i32 thickness, LineType line_type) {
            cv::rectangle(img.mat(), to_cv_point(pt1), to_cv_point(pt2),
                          to_cv_color(color), thickness, static_cast<int>(line_type));
        }

        void Drawing::filled_rectangle(Image& img, Rect rect, const Color& color) {
            cv::rectangle(img.mat(), to_cv_rect(rect), to_cv_color(color), cv::FILLED);
        }

        void Drawing::circle(Image& img, Point center, core::i32 radius, const Color& color,
                            core::i32 thickness, LineType line_type) {
            cv::circle(img.mat(), to_cv_point(center), radius, to_cv_color(color),
                       thickness, static_cast<int>(line_type));
        }

        void Drawing::filled_circle(Image& img, Point center, core::i32 radius, const Color& color) {
            cv::circle(img.mat(), to_cv_point(center), radius, to_cv_color(color), cv::FILLED);
        }

        void Drawing::ellipse(Image& img, Point center, core::i32 axis_x, core::i32 axis_y,
                             core::f64 angle, core::f64 start_angle, core::f64 end_angle,
                             const Color& color, core::i32 thickness, LineType line_type) {
            cv::ellipse(img.mat(), to_cv_point(center), cv::Size(axis_x, axis_y),
                        angle, start_angle, end_angle, to_cv_color(color),
                        thickness, static_cast<int>(line_type));
        }

        void Drawing::filled_ellipse(Image& img, Point center, core::i32 axis_x, core::i32 axis_y,
                                     core::f64 angle, const Color& color) {
            cv::ellipse(img.mat(), to_cv_point(center), cv::Size(axis_x, axis_y),
                        angle, 0, 360, to_cv_color(color), cv::FILLED);
        }

        void Drawing::polyline(Image& img, const std::vector<Point>& points, bool is_closed,
                              const Color& color, core::i32 thickness, LineType line_type) {
            std::vector<cv::Point> cv_points;
            cv_points.reserve(points.size());
            for (const auto& pt : points) {
                cv_points.push_back(to_cv_point(pt));
            }
            cv::polylines(img.mat(), cv_points, is_closed, to_cv_color(color),
                          thickness, static_cast<int>(line_type));
        }

        void Drawing::filled_polygon(Image& img, const std::vector<Point>& points, const Color& color) {
            std::vector<cv::Point> cv_points;
            cv_points.reserve(points.size());
            for (const auto& pt : points) {
                cv_points.push_back(to_cv_point(pt));
            }
            cv::fillPoly(img.mat(), cv_points, to_cv_color(color));
        }

        // =========================================================================
        // Text
        // =========================================================================

        void Drawing::text(Image& img, const std::string& text, Point origin,
                          const Color& color, core::f64 font_scale,
                          core::i32 thickness, FontFace font, LineType line_type) {
            cv::putText(img.mat(), text, to_cv_point(origin), static_cast<int>(font),
                        font_scale, to_cv_color(color), thickness, static_cast<int>(line_type));
        }

        Drawing::TextSize Drawing::get_text_size(const std::string& text, core::f64 font_scale,
                                                  core::i32 thickness, FontFace font) {
            int baseline = 0;
            cv::Size size = cv::getTextSize(text, static_cast<int>(font), font_scale, thickness, &baseline);
            return TextSize{size.width, size.height, baseline};
        }

        void Drawing::text_with_background(Image& img, const std::string& text, Point origin,
                                           const Color& text_color, const Color& bg_color,
                                           core::f64 font_scale, core::i32 thickness, FontFace font) {
            TextSize size = get_text_size(text, font_scale, thickness, font);

            // Draw background rectangle
            Rect bg_rect(origin.x, origin.y - size.height - size.baseline,
                         size.width, size.height + size.baseline + 4);
            filled_rectangle(img, bg_rect, bg_color);

            // Draw text
            Drawing::text(img, text, origin, text_color, font_scale, thickness, font);
        }

        // =========================================================================
        // Markers
        // =========================================================================

        void Drawing::marker(Image& img, Point position, const Color& color,
                            MarkerType type, core::i32 size, core::i32 thickness, LineType line_type) {
            cv::drawMarker(img.mat(), to_cv_point(position), to_cv_color(color),
                           static_cast<int>(type), size, thickness, static_cast<int>(line_type));
        }

        // =========================================================================
        // Utility
        // =========================================================================

        Image Drawing::create_canvas(core::usize width, core::usize height,
                                     const Color& bg_color, core::usize channels) {
            Image img(width, height, channels);
            if (channels == 1) {
                img.fill(bg_color.r);  // Use red channel for grayscale
            } else {
                img.fill({bg_color.b, bg_color.g, bg_color.r});  // BGR order
            }
            return img;
        }

        void Drawing::grid(Image& img, core::i32 step_x, core::i32 step_y,
                          const Color& color, core::i32 thickness) {
            int w = static_cast<int>(img.width());
            int h = static_cast<int>(img.height());

            // Vertical lines
            for (int x = step_x; x < w; x += step_x) {
                line(img, Point(x, 0), Point(x, h - 1), color, thickness);
            }

            // Horizontal lines
            for (int y = step_y; y < h; y += step_y) {
                line(img, Point(0, y), Point(w - 1, y), color, thickness);
            }
        }

        void Drawing::crosshair(Image& img, const Color& color, core::i32 size, core::i32 thickness) {
            int cx = static_cast<int>(img.width()) / 2;
            int cy = static_cast<int>(img.height()) / 2;

            line(img, Point(cx - size, cy), Point(cx + size, cy), color, thickness);
            line(img, Point(cx, cy - size), Point(cx, cy + size), color, thickness);
        }

    } // namespace vision
} // namespace psi
