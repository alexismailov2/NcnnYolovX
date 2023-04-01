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

class V8Cls
{
public:
  V8Cls(std::string const& modelFile,
        std::string const& weightsFile,
        std::string const& classesFile,
        Size inputSize);

#ifdef OIYolo_OpenCV
  auto performPrediction(::cv::Mat const& frame) -> Item::List;
#endif
  auto performPrediction(char const* frameData,
                         size_t frameWidth,
                         size_t frameHeight) -> Item::List;

private:
  class Impl;
  std::shared_ptr<Impl> _impl;
};

} // OIYolo
