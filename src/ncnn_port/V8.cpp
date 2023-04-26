#include <OIYolo/V8.hpp>

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
       std::vector<float> meanVals,
       std::function<bool(std::string const&)> filter)
    : _classes{readClasses(classesFile)}
    , _inputSize{inputSize}
    , _isSegmentationEnabled{isSegmentationEnabled}
    , _confThreshold{confidenceThreshold}
    , _nmsThreshold{nmsThreshold}
    , _maskThreshold{maskThreshold}
    , _mean_vals{meanVals}
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
  struct Obj
  {
    cv::Rect2f r;
    float c;
    uint32_t i;
    std::vector<float> f;
  };

  // TODO: Not used can be deleted
  void qsortDescentInplace(std::vector<Obj>& objects, int left, int right)
  {
    int i = left;
    int j = right;
    float p = objects[(left + right) / 2].c;

    while (i <= j)
    {
      while (objects[i].c > p)
      {
        i++;
      }

      while (objects[j].c < p)
      {
        j--;
      }

      if (i <= j)
      {
        std::swap(objects[i], objects[j]);
        i++;
        j--;
      }
    }

#pragma omp parallel sections
    {
#pragma omp section
      {
        if (left < j)
        {
          qsortDescentInplace(objects, left, j);
        }
      }
#pragma omp section
      {
        if (i < right)
        {
          qsortDescentInplace(objects, i, right);
        }
      }
    }
  }

  void nmsSortedBboxes(std::vector<Obj> const& boxes,
                       std::vector<int> &picked,
                       float nms_threshold)
  {
    picked.clear();

    const int n = boxes.size();
    std::vector<float> areas(n);
    for (int i = 0; i < n; i++)
    {
      areas[i] = boxes[i].r.area();
    }

    for (int i = 0; i < n; i++)
    {
      auto const& a = boxes[i].r;

      int keep = 1;
      for (int j = 0; j < (int) picked.size(); j++)
      {
        auto const& b = boxes[picked[j]].r;

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
  };

  auto findMaxIndexVertical(float const* currentData,
                            uint32_t rows,
                            uint32_t cols,
                            uint32_t classesCount,
                            std::vector<float>& features) -> float const*
  {
    float const* end = &currentData[(rows - 4)* cols];
    features.reserve(rows - 4 - classesCount);
    float const* cur = currentData;
    float const* max = cur;
    float const* fbegin = &currentData[classesCount * cols];
    while(cur < fbegin)
    {
      if (*cur > *max)
      {
        max = cur;
      }
      cur += cols;
    }
    while(cur < end)
    {
      features.push_back(*cur);
      cur += cols;
    }
    return max;
  };

  cv::Mat formatToSquare(const cv::Mat &source)
  {
    int col = source.cols;
    int row = source.rows;
    int _max = MAX(col, row);
    cv::Mat result = cv::Mat::zeros(_max, _max, CV_8UC3);
    source.copyTo(result(cv::Rect(0, 0, col, row)));
    return result;
  }

  void LetterBox(cv::Mat const& image,
                 cv::Mat& outImage,
                 std::vector<double>& params,
                 const cv::Size& newShape = cv::Size(640, 640),
                 bool autoShape = false,
                 bool scaleFill = false,
                 bool scaleUp = true,
                 int stride = 32,
                 const cv::Scalar& color = cv::Scalar(114, 114, 114))
  {
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
    params = std::vector<double>{ratio[0], ratio[1], (double)left, (double)top};
    cv::copyMakeBorder(outImage, outImage, top, bottom, left, right, cv::BORDER_CONSTANT, color);
  }

#ifdef OIYolo_OpenCV_DNN
public:
  void GetMask(std::vector<float> const& f,
               cv::Mat const& mask_protos,
               OIYolo::Item& output,
               std::vector<double> params)//cv::Vec4f params)
  {
    cv::Mat maskProposals = cv::Mat(f); // rows: 32, cols: 1, dimension: 2
    maskProposals = maskProposals.t();  // rows: 1, cols: 32, dimension: 2
    auto sz0 = mask_protos.size[0];//1, dimension: 4
    auto sz1 = mask_protos.size[1];//32
    auto sz2 = mask_protos.size[2];//160
    auto sz3 = mask_protos.size[3];//160
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

    std::vector<cv::Range> roi_rangs{cv::Range::all(),
                                     cv::Range(rang_y, rang_h + rang_y), //12, 22
                                     cv::Range(rang_x, rang_w + rang_x)}; //128, 132

    //crop
    cv::Mat temp_mask_protos = mask_protos(roi_rangs).clone();
    auto ssz1 = temp_mask_protos.size[0];//32 (all) _segChannels
    auto ssz2 = temp_mask_protos.size[1];//10 (12..22) rang_w
    auto ssz3 = temp_mask_protos.size[2];//4 (128..132) rang_h
    cv::Mat protos = temp_mask_protos.reshape(0, { _segChannels, rang_w * rang_h }); // rows: 32, cols: 10 * 4
    cv::Mat matmul_res = (maskProposals * protos);
    matmul_res = matmul_res.t();
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

  auto performPrediction(char const* frameData,
                         size_t frameWidth,
                         size_t frameHeight,
                         bool isNeededToBeSwappedRAndB,
                         bool isAlpha) -> Item::List
  {
    cv::Mat modelInputAligned;
    cv::Mat modelInput = formatToSquare(cv::Mat((int) frameHeight, (int) frameWidth, CV_8UC3, (void *) frameData));
    std::vector<double> params;
    LetterBox(modelInput, modelInputAligned, params, _inputSize);

    _net.setInput(cv::dnn::blobFromImage(modelInput,//Aligned,
                                         1.0 / 255.0,
                                         _inputSize,
                                         cv::Scalar(_mean_vals[0], _mean_vals[1], _mean_vals[2]),
                                         isNeededToBeSwappedRAndB,
                                         false));

    std::vector<cv::Mat> outputs;
    _net.forward(outputs, _net.getUnconnectedOutLayersNames());

    outputs[0] = outputs[0].reshape(1, outputs[0].size[1]);

    float xFactor = (float) modelInput.cols / (float) _inputSize.width;
    float yFactor = (float) modelInput.rows / (float) _inputSize.height;

    std::vector<std::vector<float>> picked_proposals;
    std::vector<Obj> foundList;
    float const* data = (float *)outputs[0].ptr<float>();
    auto cols = outputs[0].cols;
    auto rows = outputs[0].rows;
    for (int i = 0; i < cols; ++i)
    {
      float const* begin = &data[4 * cols];
      std::vector<float> f;
      auto it = findMaxIndexVertical(begin, rows, cols, _classes.size(), f);
      auto maxConf = *it;
      if (maxConf > _confThreshold)
      {
        float x = data[0 * cols];
        float y = data[1 * cols];
        float w = data[2 * cols];
        float h = data[3 * cols];
        auto sz = it - begin;
        auto index = (uint32_t)sz/cols;
        foundList.emplace_back(Obj{cv::Rect2f{(x - 0.5f * w) * xFactor, (y - 0.5f * h) * yFactor, w * xFactor, h * yFactor}, maxConf, index, std::move(f)});
      }
      data++;
    }

    if (foundList.empty()) {
      return {};
    }

    qsortDescentInplace(foundList, 0, foundList.size() - 1);

    std::vector<int> nmsIndexes(foundList.size());

    nmsSortedBboxes(foundList, nmsIndexes, _nmsThreshold);

    auto sliceMat = [](cv::Mat L, int dim, std::vector<int> _sz) -> cv::Mat {
      cv::Mat M(L.dims - 1, std::vector<int>(_sz.begin() + 1, _sz.end()).data(), CV_32FC1, L.data + L.step[0] * dim);
      return M;
    };

    _isSegmentationEnabled = outputs.size() > 1 && !outputs[1].empty();
    cv::Mat maskProtos;
    if (_isSegmentationEnabled)
    {
      maskProtos = sliceMat(outputs[1], 0, {1, 32, 160, 160});
    }
    Item::List detections{};
    cv::Rect2f holeImgRect(0, 0, frameWidth, frameHeight);
    for (auto const& idx : nmsIndexes)
    {
      detections.emplace_back(Item{&_classes[foundList[idx].i],
                                   (int)foundList[idx].i,
                                   foundList[idx].c,
                                   (OIYolo::Rect)(foundList[idx].r & holeImgRect)});
      if (_isSegmentationEnabled)
      {
        GetMask(foundList[idx].f, maskProtos, detections.back(), params);
      }
    }
    return detections;
  }
#endif

#ifdef OIYolo_NCNN
public:
  void sigmoid(ncnn::Mat& bottom)
  {
    auto op = std::unique_ptr<ncnn::Layer>(ncnn::create_layer("Sigmoid"));

    ncnn::Option opt;
    opt.num_threads = 4;
    opt.use_fp16_storage = false;
    opt.use_packing_layout = false;

    op->create_pipeline(opt);
    op->forward_inplace(bottom, opt);
    op->destroy_pipeline(opt);
  }

  void matmul(std::vector<ncnn::Mat> const &bottom_blobs, ncnn::Mat &top_blob)
  {
    auto op = std::unique_ptr<ncnn::Layer>(ncnn::create_layer("MatMul"));

    ncnn::ParamDict pd;
    pd.set(0, 0);// axis

    op->load_param(pd);

    ncnn::Option opt;
    opt.num_threads = 2;
    opt.use_fp16_storage = false;
    opt.use_packing_layout = false;

    op->create_pipeline(opt);
    std::vector<ncnn::Mat> top_blobs(1);
    op->forward(bottom_blobs, top_blobs, opt);
    top_blob = top_blobs[0];

    op->destroy_pipeline(opt);
  }

//  void decode_mask(ncnn::Mat const &mask_feat,
//                   int const &img_w,
//                   int const &img_h,
//                   ncnn::Mat const &mask_proto,
//                   ncnn::Mat const &in_pad,
//                   int const &wpad,
//                   int const &hpad,
//                   ncnn::Mat &mask_pred_result)
//  {
//    ncnn::Mat masks;
//    matmul(std::vector<ncnn::Mat>{mask_feat, mask_proto}, masks);
//    sigmoid(masks);
//    reshape(masks, masks.h, in_pad.h / 4, in_pad.w / 4, 0, masks);
//    slice(masks, (wpad / 2) / 4, (in_pad.w - wpad / 2) / 4, 2, mask_pred_result);
//    slice(mask_pred_result, (hpad / 2) / 4, (in_pad.h - hpad / 2) / 4, 1, mask_pred_result);
//    interp(mask_pred_result, 4.0, img_w, img_h, mask_pred_result);
//  }

  void GetMask(std::vector<float> const& f,
               cv::Mat const& mask_protos,
               OIYolo::Item& output,
               std::vector<double> params)
  {
    cv::Mat maskProposals = cv::Mat(f); // rows: 32, cols: 1, dimension: 2
    maskProposals = maskProposals.t();  // rows: 1, cols: 32, dimension: 2

    //w: 160
    //h: 160
    //d: 1
    //c: 32
    Rect temp_rect = output.boundingBox;
    //crop from mask_protos
    int rang_x = floor((temp_rect.x * params[0] + params[2]) / _inputSize.width * _segSize.width);
    int rang_y = floor((temp_rect.y * params[1] + params[3]) / _inputSize.height * _segSize.height);
    int rang_w = ceil(((temp_rect.x + temp_rect.width) * params[0] + params[2]) / _inputSize.width * _segSize.width) - rang_x;
    int rang_h = ceil(((temp_rect.y + temp_rect.height) * params[1] + params[3]) / _inputSize.height * _segSize.height) - rang_y;

    rang_w = std::max(rang_w, 1);
    rang_h = std::max(rang_h, 1);
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


    std::vector<cv::Range> roi_rangs{cv::Range::all(),
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
    output.mask = mask((cv::Rect)temp_rect - cv::Point(left, top)) > _maskThreshold;
  }

  static float fast_exp(float x) noexcept
  {
    union
    {
      uint32_t i;
      float f;
    } v{};
    v.i = (1 << 23) * (1.4426950409 * x + 126.93490512f);
    return v.f;
  }

  static float sigmoid(float x) noexcept
  {
    return 1.0f / (1.0f + fast_exp(-x));
  }

  auto performPrediction(const char* frameData,
                         size_t frameWidth,
                         size_t frameHeight,
                         bool isNeededToBeSwappedRAndB,
                         bool isAlpha) -> Item::List
  {
#if 0
    auto const MAX_STRIDE = 32;
    auto target_size = _inputSize.width;
    // pad to multiple of 32
    int w = frameWidth;
    int h = frameHeight;
    float scale = 1.f;
    if (w > h)
    {
      scale = (float)target_size / w;
      w = target_size;
      h = h * scale;
    }
    else
    {
      scale = (float)target_size / h;
      h = target_size;
      w = w * scale;
    }

    ncnn::Mat in = ncnn::Mat::from_pixels_resize((uint8_t const*)frameData,
                                                 isNeededToBeSwappedRAndB
                                                     ? isAlpha
                                                         ? ncnn::Mat::PIXEL_RGBA2BGR
                                                         : ncnn::Mat::PIXEL_RGB2BGR
                                                     : isAlpha
                                                         ? ncnn::Mat::PIXEL_RGBA
                                                         : ncnn::Mat::PIXEL_RGB,
                                                 frameWidth,
                                                 frameHeight,
                                                 w,
                                                 h);

    // pad to target_size rectangle
    int wpad = (w + MAX_STRIDE-1) / MAX_STRIDE * MAX_STRIDE - w;
    int hpad = (h + MAX_STRIDE-1) / MAX_STRIDE * MAX_STRIDE - h;
    ncnn::Mat in_pad;
    ncnn::copy_make_border(in, in_pad, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, ncnn::BORDER_CONSTANT, 0.f);
    const float norm_vals[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};
    in_pad.substract_mean_normalize(_mean_vals, norm_vals);
#endif
#if 1 ////
    cv::Mat modelInputAligned;
    cv::Mat modelInput = formatToSquare(cv::Mat((int) frameHeight, (int) frameWidth, CV_8UC3, (void *) frameData));
#if 1
    std::vector<double> params;
    LetterBox(modelInput, modelInputAligned, params, _inputSize);
#endif
    ncnn::Mat in = ncnn::Mat::from_pixels_resize((const uint8_t*)modelInput/*Aligned*/.data,
                                                 isNeededToBeSwappedRAndB
                                                     ? isAlpha ? ncnn::Mat::PIXEL_RGBA2BGR : ncnn::Mat::PIXEL_RGB2BGR
                                                     : isAlpha ? ncnn::Mat::PIXEL_RGBA : ncnn::Mat::PIXEL_RGB,
                                                 (int)modelInput/*Aligned*/.cols,
                                                 (int)modelInput/*Aligned*/.rows,//);
                                                 _inputSize.width,
                                                 _inputSize.height);
    const float norm_vals[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};
    in.substract_mean_normalize(_mean_vals.data(), norm_vals);
#if 0
    // pad to target_size rectangle
    int wpad = (w + 31) / 32 * 32 - w;
    int hpad = (h + 31) / 32 * 32 - h;

    ncnn::Mat in_pad;
    ncnn::copy_make_border(in, in_pad, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, ncnn::BORDER_CONSTANT, 0.f);
#else
    ncnn::Mat in_pad = in;
#endif
#endif ////

    ncnn::Extractor ex = _net.create_extractor();
    ex.input("images", in_pad);

    std::vector<ncnn::Mat> outputs(2);
    ex.extract("output0", outputs[0]);
    if (_isSegmentationEnabled)
    {
      ex.extract("output1", outputs[1]);
      if (outputs[1].empty())
      {
        _isSegmentationEnabled = false;
      }
    }

    float xFactor = (float) modelInput.cols / (float) _inputSize.width;
    float yFactor = (float) modelInput.rows / (float) _inputSize.height;

    std::vector<Obj> foundList;
    float const* data = (float *)outputs[0].data;
    auto cols = outputs[0].w;
    auto rows = outputs[0].h;
    for (int i = 0; i < cols; ++i)
    {
      float const* begin = &data[4 * cols];
      std::vector<float> f;
      auto it = findMaxIndexVertical(begin, rows, cols, _classes.size(), f);
      auto maxConf = *it;
      if (/*maxConf < 1 && */maxConf > _confThreshold)
      {
        float x = data[0 * cols];
        float y = data[1 * cols];
        float w = data[2 * cols];
        float h = data[3 * cols];
        if (i >= 6400) {
          x *= 2;
          y *= 2;
          w *= 2;
          h *= 2;
        }
        if (i >= 8000) {
          x *= 2;
          y *= 2;
          w *= 2;
          h *= 2;
        }
        auto const sz = it - begin;
        auto const index = sz/cols;
        foundList.emplace_back(Obj{cv::Rect2f{(x - 0.5f*w) * xFactor,
                                              (y - 0.5f*h) * yFactor,
                                              w * xFactor,
                                              h * yFactor}, maxConf, (uint32_t)index, std::move(f)});
      }
      data++;
    }

    if (foundList.empty()) {
      return {};
    }

    qsortDescentInplace(foundList, 0, foundList.size() - 1);

    std::vector<int> nmsIndexes{};
    nmsSortedBboxes(foundList, nmsIndexes, _nmsThreshold);

    std::vector<std::vector<float>> temp_mask_proposals;

    Item::List detections{};
    cv::Rect2f holeImgRect(0, 0, frameWidth, frameHeight);

    // (ncnn)outputs[1] -> (opencv)mask_protos
    cv::Mat maskProtos({32, outputs[1].h, outputs[1].w}, CV_32FC1);
    memcpy((uchar*)maskProtos.data, outputs[1].data, outputs[1].w * outputs[1].h * 32 * sizeof(float));

//    auto sliceMat = [](cv::Mat L, int dim, std::vector<int> _sz) -> cv::Mat {
//      cv::Mat M(L.dims - 1, std::vector<int>(_sz.begin() + 1, _sz.end()).data(), CV_32FC1, L.data + L.step[0] * dim);
//      return M;
//    };

    for (auto const& idx : nmsIndexes)
    {
      detections.emplace_back(Item{&_classes[foundList[idx].i],
                                   (int)foundList[idx].i,
                                   foundList[idx].c,
                                   (OIYolo::Rect)(foundList[idx].r & holeImgRect)});
      if (_isSegmentationEnabled)
      {
        GetMask(foundList[idx].f, maskProtos, detections.back(), params);
      }
    }
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
  std::vector<float> _mean_vals;
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
       std::vector<float> meanVals,
       std::function<bool(std::string const&)> filter)
   : _impl{std::make_shared<Impl>(modelFile,
                                  weightsFile,
                                  classesFile,
                                  inputSize,
                                  isSegmentationEnabled,
                                  confidenceThreshold,
                                  nmsThreshold,
                                  maskThreshold,
                                  meanVals,
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

