#include <OIYolo/V8.hpp>

#include "common.hpp"

#ifdef OIYolo_NCNN
#include <ncnn/net.h>
#include <ncnn/layer.h>
#endif

#ifdef OIYolo_OpenCV_DNN
#include <stdio.h>
#include <math.h>
//#include <opencv2/dnn/dnn.hpp>
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

class V8::Impl
{
public:
  Impl(std::string const& modelFile,
       std::string const& weightsFile,
       std::string const& classesFile,
       Size inputSize,
       bool isSegmentationEnabled,
       float confidenceThreshold,
       float nmsThreshold,
       float maskThreshold,
       std::function<bool(std::string const&)> filter)
    : _classes{readClasses(classesFile)}
    , _inputSize{inputSize}
    , _isSegmentationEnabled{isSegmentationEnabled}
    , _confThreshold{confidenceThreshold}
    , _nmsThreshold{nmsThreshold}
    , _maskThreshold{maskThreshold}
    , _filter{std::move(filter)}
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

    _mean_vals[0] = 103.53f;
    _mean_vals[1] = 116.28f;
    _mean_vals[2] = 123.675f;
#endif
  }

  void setFilter(std::function<bool(std::string const&)> filter)
  {
    filter = std::move(filter);
  }

  auto getFilter() -> std::function<bool(std::string const&)>
  {
    return _filter;
  }

private:
  template<typename Rect>
  void extractDataFromOutput(std::vector<int>& class_ids,
                             std::vector<float>& confidences,
                             std::vector<Rect>& boxes,
                             float const* classes_scores,
                             float const* data,
                             float xFactor,
                             float yFactor,
                             std::vector<std::vector<float>>& picked_proposals)
  {
    auto it = std::max_element(classes_scores, classes_scores + _classes.size());
    double maxClassScore = *it;

    if (maxClassScore >= _confThreshold)
    {
      if (_isSegmentationEnabled)
      {
        int net_width = _classes.size() + 4 + _segChannels;
        std::vector<float> temp_proto(data + 4 + _classes.size(), data + net_width);
        picked_proposals.push_back(temp_proto);
      }

      class_ids.push_back(std::distance(classes_scores, it));
      confidences.push_back(maxClassScore);

      float x = data[0];
      float y = data[1];
      float w = data[2];
      float h = data[3];

      int left = int((x - 0.5 * w) * xFactor);
      int top = int((y - 0.5 * h) * yFactor);
      int width = int(w * xFactor);
      int height = int(h * yFactor);

      boxes.emplace_back(Rect(left, top, width, height));
    }
  }

  static void nms_sorted_bboxes(std::vector<Rect> const& boxes,
                                std::vector<int> &picked,
                                float nms_threshold)
  {
    picked.clear();

    const int n = boxes.size();
    std::vector<float> areas(n);
    for (int i = 0; i < n; i++)
    {
      areas[i] = boxes[i].area();
    }

    for (int i = 0; i < n; i++)
    {
      Rect const &a = boxes[i];

      int keep = 1;
      for (int j = 0; j < (int) picked.size(); j++)
      {
        Rect const &b = boxes[picked[j]];

        // intersection over union
        float inter_area = (a & b).area();
        float union_area = areas[i] + areas[picked[j]] - inter_area;
        // float IoU = inter_area / union_area
        if (inter_area / union_area > nms_threshold)
        {
          keep = 0;
        }
      }

      if (keep)
      {
        picked.push_back(i);
      }
    }
  }

#ifdef OIYolo_OpenCV_DNN
public:
  cv::Mat formatToSquare(const cv::Mat &source)
  {
    int col = source.cols;
    int row = source.rows;
    int _max = MAX(col, row);
    cv::Mat result = cv::Mat::zeros(_max, _max, CV_8UC3);
    source.copyTo(result(cv::Rect(0, 0, col, row)));
    return result;
  }

  void GetMask(cv::Mat const& maskProposals,
               cv::Mat const& mask_protos,
               OIYolo::Item& output,
               cv::Vec4f params)
  {
    cv::Rect temp_rect = (cv::Rect)output.boundingBox;
    //crop from mask_protos
    int rang_x = floor((temp_rect.x * params[0] + params[2]) / _inputSize.width * _segSize.width);
    int rang_y = floor((temp_rect.y * params[1] + params[3]) / _inputSize.height * _segSize.height);
    int rang_w = ceil(((temp_rect.x + temp_rect.width) * params[0] + params[2]) / _inputSize.width * _segSize.width) - rang_x;
    int rang_h = ceil(((temp_rect.y + temp_rect.height) * params[1] + params[3]) / _inputSize.height * _segSize.height) - rang_y;

    rang_w = MAX(rang_w, 1);
    rang_h = MAX(rang_h, 1);
    if (rang_x + rang_w > _segSize.width)
    {
      if (_segSize.width - rang_x > 0)
      {
        rang_w = _segSize.width - rang_x;
      }
      else
      {
        rang_x -= 1;
      }
    }
    if (rang_y + rang_h > _segSize.height)
    {
      if (_segSize.height - rang_y > 0)
      {
        rang_h = _segSize.height - rang_y;
      }
      else
      {
        rang_y -= 1;
      }
    }

    std::vector<cv::Range> roi_rangs{cv::Range(0, 1),
                                     cv::Range::all(),
                                     cv::Range(rang_y, rang_h + rang_y),
                                     cv::Range(rang_x, rang_w + rang_x)};

    //crop
    cv::Mat temp_mask_protos = mask_protos(roi_rangs).clone();
    cv::Mat protos = temp_mask_protos.reshape(0, { _segChannels,rang_w * rang_h });
    cv::Mat matmul_res = (maskProposals * protos).t();
    cv::Mat masks_feature = matmul_res.reshape(1, { rang_h,rang_w });
    cv::Mat dest;
    cv::Mat mask;

    //sigmoid
    cv::exp(-masks_feature, dest);
    dest = 1.0 / (1.0 + dest);

    int left = floor((_inputSize.width / _segSize.width * rang_x - params[2]) / params[0]);
    int top = floor((_inputSize.height / _segSize.height * rang_y - params[3]) / params[1]);
    int width = ceil(_inputSize.width / _segSize.width * rang_w / params[0]);
    int height = ceil(_inputSize.height / _segSize.height * rang_h / params[1]);

    resize(dest, mask, cv::Size(width, height), cv::INTER_NEAREST);
    output.mask = mask(temp_rect - cv::Point(left, top)) > _maskThreshold;
  }

  void LetterBox(cv::Mat const& image,
                 cv::Mat& outImage,
                 cv::Vec4d& params,
                 const cv::Size& newShape = cv::Size(640, 640),
                 bool autoShape = false,
                 bool scaleFill = false,
                 bool scaleUp = true,
                 int stride = 32,
                 const cv::Scalar& color = cv::Scalar(114, 114, 114))
  {
    if (false)
    {
      int maxLen = MAX(image.rows, image.cols);
      outImage = cv::Mat::zeros(cv::Size(maxLen, maxLen), CV_8UC3);
      image.copyTo(outImage(cv::Rect(0, 0, image.cols, image.rows)));
      params[0] = 1;
      params[1] = 1;
      params[3] = 0;
      params[2] = 0;
    }

    cv::Size shape = image.size();
    float r = std::min((float)newShape.height / (float)shape.height,
                       (float)newShape.width / (float)shape.width);
    if (!scaleUp)
    {
      r = std::min(r, 1.0f);
    }

    float ratio[2]{ r, r };
    int new_un_pad[2] = { (int)std::round((float)shape.width * r),
                          (int)std::round((float)shape.height * r) };

    auto dw = (float)(newShape.width - new_un_pad[0]);
    auto dh = (float)(newShape.height - new_un_pad[1]);

    if (autoShape)
    {
      dw = (float)((int)dw % stride);
      dh = (float)((int)dh % stride);
    }
    else if (scaleFill)
    {
      dw = 0.0f;
      dh = 0.0f;
      new_un_pad[0] = newShape.width;
      new_un_pad[1] = newShape.height;
      ratio[0] = (float)newShape.width / (float)shape.width;
      ratio[1] = (float)newShape.height / (float)shape.height;
    }

    dw /= 2.0f;
    dh /= 2.0f;

    if ((shape.width != new_un_pad[0]) &&
        (shape.height != new_un_pad[1]))
    {
      cv::resize(image, outImage, cv::Size(new_un_pad[0], new_un_pad[1]));
    }
    else
    {
      outImage = image.clone();
    }

    int top = int(std::round(dh - 0.1f));
    int bottom = int(std::round(dh + 0.1f));
    int left = int(std::round(dw - 0.1f));
    int right = int(std::round(dw + 0.1f));
    params[0] = ratio[0];
    params[1] = ratio[1];
    params[2] = left;
    params[3] = top;
    cv::copyMakeBorder(outImage, outImage, top, bottom, left, right, cv::BORDER_CONSTANT, color);
  }

  auto performPrediction(char const* frameData,
                         size_t frameWidth,
                         size_t frameHeight,
                         bool isNeededToBeSwappedRAndB,
                         bool isAlpha) -> Item::List
  {
    cv::Mat modelInputAligned;
    cv::Mat modelInput = formatToSquare(cv::Mat((int) frameHeight, (int) frameWidth, CV_8UC3, (void *) frameData));
#if 1
    cv::Vec4d params;
    LetterBox(modelInput, modelInputAligned, params, _inputSize);
#endif
    _net.setInput(cv::dnn::blobFromImage(modelInputAligned,
                                         1.0 / 255.0,
                                         _inputSize,
                                         cv::Scalar(/*_mean_vals[0], _mean_vals[1], _mean_vals[2]*/),
                                         isNeededToBeSwappedRAndB,
                                         false));

    std::vector<cv::Mat> outputs;
    _net.forward(outputs, _net.getUnconnectedOutLayersNames());

    if (outputs.size() < 2)
    {
      _isSegmentationEnabled = false;
    }

    int rows = outputs[0].size[1];
    int dimensions = outputs[0].size[2];

    bool yolov8 = false;
    // yolov5 has an output of shape (batchSize, 25200, 85) (Num classes + box[x,y,w,h] + confidence[c])
    // yolov8 has an output of shape (batchSize, 84,  8400) (Num classes + box[x,y,w,h])
    if (dimensions > rows) // Check if the shape[2] is more than shape[1] (yolov8)
    {
      yolov8 = true;
      rows = outputs[0].size[2];
      dimensions = outputs[0].size[1];

      outputs[0] = outputs[0].reshape(1, dimensions);
      cv::transpose(outputs[0], outputs[0]);
    }

    float x_factor = (float) modelInput.cols / (float) _inputSize.width;
    float y_factor = (float) modelInput.rows / (float) _inputSize.height;

    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<Rect> boxes;
    std::vector<std::vector<float>> picked_proposals;

    float *data = (float *)outputs[0].data;
    for (int i = 0; i < rows; ++i)
    {
      if (yolov8)
      {
          float* classes_scores = data + 4;
          extractDataFromOutput(class_ids, confidences, boxes, classes_scores, data, x_factor, y_factor, picked_proposals);
      }
      else
      { // yolov5
        float confidence = data[4];
        if (confidence >= _confThreshold)
        {
          float* classes_scores = data + 5;
          extractDataFromOutput(class_ids, confidences, boxes, classes_scores, data, x_factor, y_factor, picked_proposals);
        }
      }
      data += dimensions;
    }

    std::vector<int> nms_result;
    //cv::dnn::NMSBoxes(boxes, confidences, _confThreshold, _nmsThreshold, nms_result);
    nms_sorted_bboxes(boxes, nms_result, _nmsThreshold);
    std::vector<std::vector<float>> temp_mask_proposals;
    Rect holeImgRect(0, 0, frameWidth, frameHeight);

    Item::List detections{};
    for (auto const& idx : nms_result)
    {
      if (_isSegmentationEnabled)
      {
        temp_mask_proposals.push_back(picked_proposals[idx]);
      }
      detections.emplace_back(Item{&_classes[class_ids[idx]],
                                   class_ids[idx],
                                   confidences[idx],
                                   (OIYolo::Rect)(boxes[idx] & holeImgRect)});
    }
    if (_isSegmentationEnabled)
    {
      for (int i = 0; i < temp_mask_proposals.size(); ++i)
      {
        GetMask(cv::Mat(temp_mask_proposals[i]).t(), outputs[1], detections[i], params);
      }
    }
    return detections;
  }
#endif

#ifdef OIYolo_NCNN
public:
  auto performPrediction(const char* frameData,
                         size_t frameWidth,
                         size_t frameHeight,
                         bool isNeededToBeSwappedRAndB,
                         bool isAlpha) -> Item::List
  {
#if 0
    int width = (int)frameWidth;
    int height = (int)frameHeight;

    // pad to multiple of 32
    int w = width;
    int h = height;
    float scale = 1.f;
    if (w > h)
    {
      scale = (float)_inputSize.width / (float)w;
      w = _inputSize.width;
      h = h * scale;
    }
    else
    {
      scale = (float)_inputSize.width / (float)h;
      h = _inputSize.width;
      w = w * scale;
    }
#endif
    ncnn::Mat in = ncnn::Mat::from_pixels_resize((const uint8_t*)frameData,
                                                 isNeededToBeSwappedRAndB
                                                     ? isAlpha ? ncnn::Mat::PIXEL_RGBA2BGR : ncnn::Mat::PIXEL_RGB2BGR
                                                     : isAlpha ? ncnn::Mat::PIXEL_RGBA : ncnn::Mat::PIXEL_RGB,
                                                 (int)frameWidth,
                                                 (int)frameHeight,
                                                 _inputSize.width,
                                                 _inputSize.height);
#if 0
    // pad to target_size rectangle
    int wpad = (w + 31) / 32 * 32 - w;
    int hpad = (h + 31) / 32 * 32 - h;

    ncnn::Mat in_pad;
    ncnn::copy_make_border(in, in_pad, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, ncnn::BORDER_CONSTANT, 0.f);
#else
    ncnn::Mat in_pad = in;
#endif

    const float norm_vals[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};
    in_pad.substract_mean_normalize(nullptr, norm_vals);

    ncnn::Extractor ex = _net.create_extractor();
    ex.input("images", in_pad);

    ncnn::Mat out;
    ncnn::Mat output;
    ex.extract("output0", out);
    ncnn::Mat mask_proto;
    if (_isSegmentationEnabled)
    {
      ex.extract("output1", mask_proto);
      if (mask_proto.empty())
      {
        _isSegmentationEnabled = false;
      }
    }

    int rows = out.h;
    int dimensions = out.w;

    bool yolov8 = false;
    // yolov5 has an output of shape (batchSize, 25200, 85) (Num classes + box[x,y,w,h] + confidence[c])
    // yolov8 has an output of shape (batchSize, 84,  8400) (Num classes + box[x,y,w,h])
    if (dimensions > rows) // Check if the shape[2] is more than shape[1] (yolov8)
    {
      yolov8 = true;
      rows = out.w;
      dimensions = out.h;

      //out = out.reshape(dimensions, /*1*/rows);
      transpose(out, output);
    }

    float x_factor = (float) frameWidth / (float) _inputSize.width;
    float y_factor = (float) frameHeight / (float) _inputSize.height;

    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<Rect> boxes;
    std::vector<std::vector<float>> picked_proposals;

    float *data = (float *)output.data;
    for (int i = 0; i < rows; ++i)
    {
      if (yolov8)
      {
        float* classes_scores = data + 4;
        extractDataFromOutput(class_ids, confidences, boxes, classes_scores, data, x_factor, y_factor, picked_proposals);
      }
      else
      { // yolov5
        float confidence = data[4];
        if (confidence >= _confThreshold)
        {
          float* classes_scores = data + 5;
          extractDataFromOutput(class_ids, confidences, boxes, classes_scores, data, x_factor, y_factor, picked_proposals);
        }
      }
      data += dimensions;
    }

    std::vector<int> nms_result;
    //cv::dnn::NMSBoxes(boxes, confidences, _confThreshold, _nmsThreshold, nms_result);
    nms_sorted_bboxes(boxes, nms_result, _nmsThreshold);
    std::vector<std::vector<float>> temp_mask_proposals;
    Rect holeImgRect(0, 0, frameWidth, frameHeight);

    Item::List detections{};
    for (auto const& idx : nms_result)
    {
      if (_isSegmentationEnabled)
      {
        temp_mask_proposals.push_back(picked_proposals[idx]);
      }
      detections.emplace_back(Item{&_classes[class_ids[idx]],
                                   class_ids[idx],
                                   confidences[idx],
                                   (OIYolo::Rect)(boxes[idx] & holeImgRect)});
    }
    if (_isSegmentationEnabled)
    {
      for (int i = 0; i < temp_mask_proposals.size(); ++i)
      {
        //GetMask(cv::Mat(temp_mask_proposals[i]).t(), outputs[1], detections[i], params);
      }
    }

#if 0
    std::vector<Item> proposals;

    std::vector<int> strides = {8, 16, 32}; // might have stride=64
    std::vector<GridAndStride> grid_strides;
    generate_grids_and_stride(in_pad.w, in_pad.h, strides, grid_strides);

    //TODO: Here should be cut off classes by filter
    //cv::Mat output0 = cv::Mat(cv::Size(net_output_img[0].size[2], net_output_img[0].size[1]), CV_32F, (float*)net_output_img[0].data).t();
    generate_proposals(grid_strides, out, _confThreshold, _isSegmentationEnabled, _classes, proposals);

    // sort all proposals by score from highest to lowest
    qsort_descent_inplace(proposals);

    // apply nms with nms_threshold
    std::vector<int> picked;
    nms_sorted_bboxes(proposals, picked, _nmsThreshold);

    int count = picked.size();

    ncnn::Mat mask_pred_result;
    if (_isSegmentationEnabled)
    {
      ncnn::Mat mask_feat = ncnn::Mat(32, count, sizeof(float));
      for (int i = 0; i < count; i++)
      {
        float *mask_feat_ptr = mask_feat.row(i);
        std::memcpy(mask_feat_ptr, proposals[picked[i]].mask_feat.data(),
                    sizeof(float) * proposals[picked[i]].mask_feat.size());
      }

      decode_mask(mask_feat, width, height, mask_proto, in_pad, wpad, hpad, mask_pred_result);
    }

    Item::List objects(count);
    for (int i = 0; i < count; i++)
    {
      objects[i] = proposals[picked[i]];

      // adjust offset to original unpadded
      float x0 = (objects[i].boundingBox.x - ((float)wpad / 2)) / scale;
      float y0 = (objects[i].boundingBox.y - ((float)hpad / 2)) / scale;
      float x1 = (objects[i].boundingBox.x + objects[i].boundingBox.width - ((float)wpad / 2)) / scale;
      float y1 = (objects[i].boundingBox.y + objects[i].boundingBox.height - ((float)hpad / 2)) / scale;

      // clip
      x0 = std::max(std::min(x0, (float)(width - 1)), 0.f);
      y0 = std::max(std::min(y0, (float)(height - 1)), 0.f);
      x1 = std::max(std::min(x1, (float)(width - 1)), 0.f);
      y1 = std::max(std::min(y1, (float)(height - 1)), 0.f);

      objects[i].boundingBox.x = x0;
      objects[i].boundingBox.y = y0;
      objects[i].boundingBox.width = x1 - x0;
      objects[i].boundingBox.height = y1 - y0;

      if (_isSegmentationEnabled)
      {
        objects[i].mask = cv::Mat::zeros(height, width, CV_32FC1);
        cv::Mat mask = cv::Mat(height, width, CV_32FC1, (float *) mask_pred_result.channel(i));
        mask(cv::Rect2d{objects[i].boundingBox.x,
                        objects[i].boundingBox.y,
                        objects[i].boundingBox.width,
                        objects[i].boundingBox.height}).copyTo(objects[i].mask(cv::Rect2d{objects[i].boundingBox.x,
                                                                                          objects[i].boundingBox.y,
                                                                                          objects[i].boundingBox.width,
                                                                                          objects[i].boundingBox.height}));
      }
    }
#endif
    return detections;
  }
#endif

private:
  std::vector<std::string> _classes;
  Size _inputSize;
  bool _isSegmentationEnabled;
  // TODO: Should be added to constructor for an ability to configure
  Size _segSize{160, 160};
  int _segChannels{32};
  float _confThreshold{};
  float _nmsThreshold{};
  float _maskThreshold{};
  // TODO: Should be added for configuring
  float _mean_vals[3];
  std::function<bool(std::string const&)> _filter;
#ifdef OIYolo_OpenCV_DNN
  cv::dnn::Net _net;
#endif
#ifdef OIYolo_NCNN
  ncnn::Net _net;
#endif
};

V8::V8(std::string const& modelFile,
       std::string const& weightsFile,
       std::string const& classesFile,
       Size inputSize,
       bool isSegmentationEnabled,
       float confidenceThreshold,
       float nmsThreshold,
       float maskThreshold,
       std::function<bool(std::string const&)> filter)
   : _impl{std::make_shared<Impl>(modelFile,
                                  weightsFile,
                                  classesFile,
                                  inputSize,
                                  isSegmentationEnabled,
                                  confidenceThreshold,
                                  nmsThreshold,
                                  maskThreshold,
                                  std::move(filter))}
{
}

void V8::setFilter(std::function<bool(std::string const&)>&& filter)
{
  return _impl->setFilter(std::move(filter));
}

auto V8::getFilter() -> std::function<bool(std::string const&)>
{
  return _impl->getFilter();
}

#ifdef OIYolo_OpenCV
auto V8::performPrediction(cv::Mat const& frame,
                           bool isNeededToBeSwappedRAndB,
                           bool isAlpha) -> Item::List
{
    return _impl->performPrediction((const char*)frame.data,
                                    frame.cols,
                                    frame.rows,
                                    isNeededToBeSwappedRAndB,
                                    isAlpha);
}
#endif

auto V8::performPrediction(char const* frameData,
                           size_t frameWidth,
                           size_t frameHeight,
                           bool isNeededToBeSwappedRAndB,
                           bool isAlpha) -> Item::List
{
    return _impl->performPrediction(frameData,
                                    frameWidth,
                                    frameHeight,
                                    isNeededToBeSwappedRAndB,
                                    isAlpha);
}

}// OIYolo

