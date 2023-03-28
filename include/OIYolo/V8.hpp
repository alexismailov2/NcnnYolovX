#pragma once

#include <OIYolo/CommonTypes.hpp>

#ifdef OIYolo_OpenCV
#include <opencv2/core/mat.hpp>
#endif

#include <vector>
#include <memory>
#include <string>
#include <functional>

namespace OIYolo {

class V8
{
public:
  V8(std::string const& modelFile,
     std::string const& weightsFile,
     std::string const& classesFile,
     Size inputSize,
     bool isSegmentationEnabled = false,
     float confidenceThreshold = 0.25f,
     float nmsThreshold = 0.25f,
     float maskThreshold = 0.25f,
     std::function<bool(std::string const&)> filter = [](std::string const&) { return true; });

  void setFilter(std::function<bool(std::string const&)>&& filter);
  auto getFilter() -> std::function<bool(std::string const&)>;

#ifdef OIYolo_OpenCV
  auto performPrediction(::cv::Mat const& frame,
                         bool isNeededToBeSwappedRAndB = true,
                         bool isAlpha = false) -> Item::List;
#endif
  auto performPrediction(char const* frameData,
                         size_t frameWidth,
                         size_t frameHeight,
                         bool isNeededToBeSwappedRAndB = true,
                         bool isAlpha = false) -> Item::List;

private:
  class Impl;
  std::shared_ptr<Impl> _impl;
};

} // OIYolo
