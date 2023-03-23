#pragma once

#include <vector>
#include <string>
#include <algorithm>

namespace OIYolo {

struct Rect
{
  Rect() noexcept = default;
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
  float x;
  float y;
  float width;
  float height;
};

// TODO: Copied from OpenCV to ret rid of dependency
// TODO: Looks like not efficient implementation
static inline Rect& operator &= (Rect& a, const Rect& b)
{
  if (a.empty() || b.empty())
  {
    a = Rect{};
    return a;
  }
  const Rect& Rx_min = (a.x < b.x) ? a : b;
  const Rect& Rx_max = (a.x < b.x) ? b : a;
  const Rect& Ry_min = (a.y < b.y) ? a : b;
  const Rect& Ry_max = (a.y < b.y) ? b : a;

  if ((Rx_min.x < 0 && Rx_min.x + Rx_min.width < Rx_max.x) ||
      (Ry_min.y < 0 && Ry_min.y + Ry_min.height < Ry_max.y))
  {
    a = Rect{};
    return a;
  }

  a.width = std::min(Rx_min.width - (Rx_max.x - Rx_min.x), Rx_max.width);
  a.height = std::min(Ry_min.height - (Ry_max.y - Ry_min.y), Ry_max.height);
  a.x = Rx_max.x;
  a.y = Ry_max.y;
  if (a.empty())
  {
    a = Rect{};
  }
  return a;
}

static inline Rect operator&(Rect const& a, const Rect& b)
{
  Rect c = a;
  return c &= b;
}

struct Size
{
  int width;
  int height;
};

struct Item
{
  using List = std::vector<Item>;

  std::string const* className;
  float confidence;
  Rect boundingBox;
};

} // OIYolo
