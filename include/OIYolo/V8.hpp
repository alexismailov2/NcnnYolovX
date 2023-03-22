#pragma once

#include <OIYolo/CommonTypes.hpp>

#include <vector>
#include <memory>
#include <string>
#include <functional>

// TODO: Very temporary
#include <opencv2/core/mat.hpp>

namespace OIYolo {

class V8
{
public:
  V8(std::string const& modelFile,
     std::string const& weightsFile,
     std::string const& classesFile,
     Size inputSize,
     float confidenceThreshold = 0.25f,
     float nmsThreshold = 0.25f);

//  V8(std::string const& modelFile,
//     std::string const& classesFile,
//     Size inputSize,
//  float confidenceThreshold = 0.25f,
//  float nmsThreshold = 0.25f);

  auto performPrediction(::cv::Mat const& frame,
                         std::function<bool(std::string const&)>&& filter = [](std::string const&) { return true; },
                         bool isNeededToBeSwappedRAndB = true) -> Item::List;

private:
  auto frameExtract(std::vector<::cv::Mat> const& outs,
                    Size const& frameSize,
                    std::function<bool(std::string const&)>&& filter) const -> Item::List;

private:
  class Impl;
  std::shared_ptr<Impl> _impl;
};

} // OIYolo
