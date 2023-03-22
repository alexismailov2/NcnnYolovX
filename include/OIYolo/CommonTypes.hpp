#pragma once

#include <vector>

namespace OIYolo {

struct Rect
{
  Rect() noexcept = default;
  Rect(Rect const&) noexcept = default;
  Rect& operator=(Rect const&) noexcept = default;
  Rect(Rect&&) noexcept = default;
  Rect& operator=(Rect&&) noexcept = default;
  float area() const noexcept {
    return width * height;
  }
  float x;
  float y;
  float width;
  float height;
};

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
