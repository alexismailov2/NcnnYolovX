#pragma once

#include <vector>
#include <string>
#include <algorithm>

#ifdef OIYolo_OpenCV
#include <opencv2/core/types.hpp>
#include <opencv2/core/mat.hpp>//opencv.hpp>
#endif

namespace OIYolo {

struct Rect
{
  Rect() noexcept = default;

  Rect(float x,
       float y,
       float width,
       float height)
     : x{x}
     , y{y}
     , width{width}
     , height{height}
  {
  }
#ifdef OIYolo_OpenCV
  Rect(cv::Rect2d rect)
      : x{(float)rect.x}
      , y{(float)rect.y}
      , width{(float)rect.width}
      , height{(float)rect.height}
  {
  }

  Rect(cv::Rect2f rect)
    : x{(float)rect.x}
    , y{(float)rect.y}
    , width{(float)rect.width}
    , height{(float)rect.height}
  {
  }

  Rect(cv::Rect2i rect)
    : x{(float)rect.x}
    , y{(float)rect.y}
    , width{(float)rect.width}
    , height{(float)rect.height}
  {
  }
#endif
  Rect(Rect const&) noexcept = default;
  Rect& operator=(Rect const&) noexcept = default;
  Rect(Rect&&) noexcept = default;
  Rect& operator=(Rect&&) noexcept = default;

  float area() const noexcept
  {
    return width * height;
  }

  bool empty() const noexcept
  {
    return width <= 0 || height <= 0;
  }

  //https://quick-bench.com/q/1IlacaP7wUP0c-581LKbG2agjNg
  Rect operator&(Rect b) const noexcept
  {
    if (empty() || b.empty())
    {
      return {};
    }
    Rect const& Rx_min = (x < b.x) ? *this : b;
    Rect const& Rx_max = (x < b.x) ? b : *this;
    Rect const& Ry_min = (y < b.y) ? *this : b;
    Rect const& Ry_max = (y < b.y) ? b : *this;

    if ((Rx_min.x < 0 && Rx_min.x + Rx_min.width < Rx_max.x) ||
        (Ry_min.y < 0 && Ry_min.y + Ry_min.height < Ry_max.y))
    {
      return {};
    }

    Rect a{Rx_max.x,
           Ry_max.y,
           std::min(Rx_min.width - (Rx_max.x - Rx_min.x), Rx_max.width),
           std::min(Ry_min.height - (Ry_max.y - Ry_min.y), Ry_max.height)};
    if (a.empty())
    {
      return {};
    }
    return a;
  }

  bool operator>(Rect const& b) const noexcept
  {
    return area() > b.area();
  }

#ifdef OIYolo_OpenCV
  operator cv::Rect2d() const
  {
    return cv::Rect2d{x, y, width, height};
  }

  operator cv::Rect2f() const
  {
    return cv::Rect2f{x, y, width, height};
  }

  operator cv::Rect2i() const
  {
    return cv::Rect2i{(int)x, (int)y, (int)width, (int)height};
  }
#endif

  float x;
  float y;
  float width;
  float height;
};

struct Size
{
  int width;
  int height;
#ifdef OIYolo_OpenCV
  operator cv::Size() const
  {
    return cv::Size{width, height};
  }
#endif
};

struct Item
{
  using List = std::vector<Item>;

  std::string const* className;
  int id;
  float confidence;
  Rect boundingBox;
  cv::Mat mask;
  std::vector<float> mask_feat;
};

} // OIYolo
