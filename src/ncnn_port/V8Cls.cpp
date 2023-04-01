#include <OIYolo/V8Cls.hpp>

#ifdef OIYolo_NCNN
#include <ncnn/net.h>
#include <ncnn/layer.h>
#endif

#ifdef OIYolo_OpenCV
#include <stdio.h>
#include <math.h>
#include <opencv2/opencv.hpp>
#endif

#include <utility>
#include <fstream>
#include <cfloat>
#include <random>

namespace OIYolo {

namespace {

auto readClasses(std::string const& filename) -> std::vector<std::string>
{
  std::vector<std::string> classes;
  std::string className;
  auto fileWithClasses{std::ifstream(filename)};
  while (std::getline(fileWithClasses, className))
  {
    if (!className.empty())
    {
      classes.push_back(className);
    }
  }
  return classes;
}

} // anonymous namespace

class V8Cls::Impl
{
public:
  Impl(std::string const& modelFile,
       std::string const& weightsFile,
       std::string const& classesFile,
       Size inputSize)
    : _classes{readClasses(classesFile)}
    , _inputSize{inputSize}
  {
#ifdef OIYolo_OpenCV_DNN
    bool cudaEnabled = false; // TODO: Temporary always false
    _net = cv::dnn::readNetFromONNX(modelFile);
    _net.setPreferableBackend(cudaEnabled ? cv::dnn::DNN_BACKEND_CUDA : cv::dnn::DNN_BACKEND_OPENCV);
    _net.setPreferableTarget(cudaEnabled ? cv::dnn::DNN_TARGET_CUDA : cv::dnn::DNN_TARGET_CPU);
#endif

#ifdef OIYolo_NCNN
    //_net.opt = ncnn::Option();
    //_net.opt.num_threads = 4;

    _net.load_param(modelFile.c_str());
    _net.load_model(weightsFile.c_str());
#endif
  }

#ifdef OIYolo_OpenCV_DNN
public:
  auto performPrediction(char const* frameData,
                         size_t frameWidth,
                         size_t frameHeight) -> Item::List
  {
    auto input = cv::Mat{{1, 3, (int)frameWidth, (int)frameHeight}, CV_8UC1};
    std::memcpy(input.data, frameData, frameWidth * frameHeight * 3);
    input.convertTo(input, CV_32FC1);
    input /= 255.0;
    _net.setInput(input);

    std::vector<cv::Mat> outputs;
    _net.forward(outputs, _net.getUnconnectedOutLayersNames());

    Item::List detections{};
    auto it = std::max_element((float const*)outputs[0].data, (float const*)outputs[0].data + outputs[0].cols);
    auto idx = it - (float const*)outputs[0].data;
    detections.emplace_back(Item{&_classes[idx], (int)idx, *it});
    return detections;
  }
#endif

#ifdef OIYolo_NCNN
public:

  auto performPrediction(const char* frameData,
                         size_t frameWidth,
                         size_t frameHeight) -> Item::List
  {
      ncnn::Mat in = ncnn::Mat::from_pixels((const uint8_t*)frameData,
                                            ncnn::Mat::PIXEL_GRAY,
                                            (int)frameWidth,
                                            (int)frameHeight);
      ncnn::Extractor ex = _net.create_extractor();
      ex.input("images", in);

      ncnn::Mat outputs;
      ex.extract("output0", outputs);

      Item::List detections{};
      auto it = std::max_element((float const*)outputs.data, (float const*)outputs.data + outputs.w);
      auto idx = it - (float const*)outputs.data;
      detections.emplace_back(Item{&_classes[idx], (int)idx, *it});
      return detections;
  }
#endif

private:
  std::vector<std::string> _classes;
  Size _inputSize;

#ifdef OIYolo_OpenCV_DNN
  cv::dnn::Net _net;
#endif
#ifdef OIYolo_NCNN
  ncnn::Net _net;
#endif
};

V8Cls::V8Cls(std::string const& modelFile,
             std::string const& weightsFile,
             std::string const& classesFile,
             Size inputSize)
  : _impl{std::make_shared<Impl>(modelFile,
                                 weightsFile,
                                 classesFile,
                                 inputSize)}
{
}

#ifdef OIYolo_OpenCV
auto V8Cls::performPrediction(cv::Mat const& frame) -> Item::List
{
  return _impl->performPrediction((const char*)frame.data,
                                  frame.cols,
                                  frame.rows);
}
#endif

auto V8Cls::performPrediction(char const* frameData,
                              size_t frameWidth,
                              size_t frameHeight) -> Item::List
{
  return _impl->performPrediction(frameData,
                                  frameWidth,
                                  frameHeight);
}

}// OIYolo

