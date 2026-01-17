#pragma once

#include "image.h"
#include <string>

namespace psi {
    namespace vision {

        // Line types
        enum class LineType : core::i32 {
            Filled = -1,
            Line4 = 4,
            Line8 = 8,
            LineAA = 16
        };

        // Font faces
        enum class FontFace : core::i32 {
            SimplexSmall = 0,
            ComplexSmall = 1,
            Simplex = 2,
            Duplex = 3,
            Complex = 4,
            Triplex = 5,
            ScriptSimplex = 6,
            ScriptComplex = 7
        };

        // Color struct for convenience
        struct Color {
            core::u8 r, g, b, a;

            Color(core::u8 r = 0, core::u8 g = 0, core::u8 b = 0, core::u8 a = 255)
                : r(r), g(g), b(b), a(a) {}

            // Predefined colors
            static Color black() { return Color(0, 0, 0); }
            static Color white() { return Color(255, 255, 255); }
            static Color red() { return Color(255, 0, 0); }
            static Color green() { return Color(0, 255, 0); }
            static Color blue() { return Color(0, 0, 255); }
            static Color yellow() { return Color(255, 255, 0); }
            static Color cyan() { return Color(0, 255, 255); }
            static Color magenta() { return Color(255, 0, 255); }
            static Color gray() { return Color(128, 128, 128); }
            static Color orange() { return Color(255, 165, 0); }
        };

        // Point struct
        struct Point {
            core::i32 x, y;
            Point(core::i32 x = 0, core::i32 y = 0) : x(x), y(y) {}
        };

        // Rectangle struct
        struct Rect {
            core::i32 x, y, width, height;
            Rect(core::i32 x = 0, core::i32 y = 0, core::i32 w = 0, core::i32 h = 0)
                : x(x), y(y), width(w), height(h) {}
        };

        // Drawing class
        class Drawing {
        public:
            // =========================================================================
            // Basic shapes
            // =========================================================================

            // Draw a line
            static void line(Image& img, Point pt1, Point pt2, const Color& color,
                            core::i32 thickness = 1, LineType line_type = LineType::Line8);

            // Draw an arrow
            static void arrow(Image& img, Point pt1, Point pt2, const Color& color,
                             core::i32 thickness = 1, core::f64 tip_length = 0.1,
                             LineType line_type = LineType::Line8);

            // Draw a rectangle
            static void rectangle(Image& img, Rect rect, const Color& color,
                                 core::i32 thickness = 1, LineType line_type = LineType::Line8);

            static void rectangle(Image& img, Point pt1, Point pt2, const Color& color,
                                 core::i32 thickness = 1, LineType line_type = LineType::Line8);

            // Draw a filled rectangle
            static void filled_rectangle(Image& img, Rect rect, const Color& color);

            // Draw a circle
            static void circle(Image& img, Point center, core::i32 radius, const Color& color,
                              core::i32 thickness = 1, LineType line_type = LineType::Line8);

            // Draw a filled circle
            static void filled_circle(Image& img, Point center, core::i32 radius, const Color& color);

            // Draw an ellipse
            static void ellipse(Image& img, Point center, core::i32 axis_x, core::i32 axis_y,
                               core::f64 angle, core::f64 start_angle, core::f64 end_angle,
                               const Color& color, core::i32 thickness = 1,
                               LineType line_type = LineType::Line8);

            // Draw a filled ellipse
            static void filled_ellipse(Image& img, Point center, core::i32 axis_x, core::i32 axis_y,
                                       core::f64 angle, const Color& color);

            // Draw a polyline
            static void polyline(Image& img, const std::vector<Point>& points, bool is_closed,
                                const Color& color, core::i32 thickness = 1,
                                LineType line_type = LineType::Line8);

            // Draw a filled polygon
            static void filled_polygon(Image& img, const std::vector<Point>& points, const Color& color);

            // =========================================================================
            // Text
            // =========================================================================

            // Draw text
            static void text(Image& img, const std::string& text, Point origin,
                            const Color& color, core::f64 font_scale = 1.0,
                            core::i32 thickness = 1, FontFace font = FontFace::Simplex,
                            LineType line_type = LineType::Line8);

            // Get text size
            struct TextSize {
                core::i32 width;
                core::i32 height;
                core::i32 baseline;
            };

            static TextSize get_text_size(const std::string& text, core::f64 font_scale = 1.0,
                                          core::i32 thickness = 1, FontFace font = FontFace::Simplex);

            // Draw text with background
            static void text_with_background(Image& img, const std::string& text, Point origin,
                                            const Color& text_color, const Color& bg_color,
                                            core::f64 font_scale = 1.0, core::i32 thickness = 1,
                                            FontFace font = FontFace::Simplex);

            // =========================================================================
            // Markers
            // =========================================================================

            enum class MarkerType : core::i32 {
                Cross = 0,
                TiltedCross = 1,
                Star = 2,
                Diamond = 3,
                Square = 4,
                TriangleUp = 5,
                TriangleDown = 6
            };

            static void marker(Image& img, Point position, const Color& color,
                              MarkerType type = MarkerType::Cross, core::i32 size = 20,
                              core::i32 thickness = 1, LineType line_type = LineType::Line8);

            // =========================================================================
            // Utility
            // =========================================================================

            // Create a blank canvas
            static Image create_canvas(core::usize width, core::usize height,
                                       const Color& bg_color = Color::white(),
                                       core::usize channels = 3);

            // Draw grid lines
            static void grid(Image& img, core::i32 step_x, core::i32 step_y,
                            const Color& color, core::i32 thickness = 1);

            // Draw crosshair at center
            static void crosshair(Image& img, const Color& color, core::i32 size = 20,
                                 core::i32 thickness = 1);
        };

    } // namespace vision
} // namespace psi
