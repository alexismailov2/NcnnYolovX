#include <OIYolo/V8.hpp>

#include <opencv2/dnn.hpp>

#include <ncnn/net.h>
#include <ncnn/layer.h>

#include <utility>
#include <fstream>

namespace OIYolo {

namespace {
float fast_exp(float x)
{
  union
  {
    uint32_t i;
    float f;
  } v{};
  v.i = (1 << 23) * (1.4426950409 * x + 126.93490512f);
  return v.f;
}

float sigmoid(float x)
{
  return 1.0f / (1.0f + fast_exp(-x));
}

float intersection_area(const Item &a, const Item &b)
{
  cv::Rect2d inter = cv::Rect2d{a.boundingBox.x, a.boundingBox.y, a.boundingBox.width, a.boundingBox.height} &
                     cv::Rect2d{b.boundingBox.x, b.boundingBox.y, b.boundingBox.width, b.boundingBox.height};
  return inter.area();
}

void qsort_descent_inplace(std::vector<Item> &faceobjects, int left, int right)
{
  int i = left;
  int j = right;
  float p = faceobjects[(left + right) / 2].confidence;

  while (i <= j) {
    while (faceobjects[i].confidence > p) {
      i++;
    }

    while (faceobjects[j].confidence < p) {
      j--;
    }

    if (i <= j) {
      std::swap(faceobjects[i], faceobjects[j]);
      i++;
      j--;
    }
  }

  //     #pragma omp parallel sections
  {
    //         #pragma omp section
    {
      if (left < j) qsort_descent_inplace(faceobjects, left, j);
    }
    //         #pragma omp section
    {
      if (i < right) qsort_descent_inplace(faceobjects, i, right);
    }
  }
}

void qsort_descent_inplace(std::vector<Item> &faceobjects)
{
  if (faceobjects.empty()) {
    return;
  }

  qsort_descent_inplace(faceobjects, 0, faceobjects.size() - 1);
}

void nms_sorted_bboxes(std::vector<Item> const &faceobjects,
                       std::vector<int> &picked,
                       float nms_threshold)
{
  picked.clear();

  const int n = faceobjects.size();

  std::vector<float> areas(n);
  for (int i = 0; i < n; i++) {
    areas[i] = faceobjects[i].boundingBox.width * faceobjects[i].boundingBox.height;
  }

  for (int i = 0; i < n; i++) {
    Item const &a = faceobjects[i];

    int keep = 1;
    for (int j = 0; j < (int) picked.size(); j++) {
      Item const &b = faceobjects[picked[j]];

      // intersection over union
      float inter_area = intersection_area(a, b);
      float union_area = areas[i] + areas[picked[j]] - inter_area;
      // float IoU = inter_area / union_area
      if (inter_area / union_area > nms_threshold) {
        keep = 0;
      }
    }

    if (keep) {
      picked.push_back(i);
    }
  }
}

struct GridAndStride
{
  int grid0;
  int grid1;
  int stride;
};

void generate_grids_and_stride(const int target_w,
                               const int target_h,
                               std::vector<int> &strides,
                               std::vector<GridAndStride> &grid_strides)
{
  for (int i = 0; i < (int) strides.size(); i++) {
    int stride = strides[i];
    int num_grid_w = target_w / stride;
    int num_grid_h = target_h / stride;
    for (int g1 = 0; g1 < num_grid_h; g1++) {
      for (int g0 = 0; g0 < num_grid_w; g0++) {
        GridAndStride gs;
        gs.grid0 = g0;
        gs.grid1 = g1;
        gs.stride = stride;
        grid_strides.push_back(gs);
      }
    }
  }
}

void generate_proposals(std::vector<GridAndStride> grid_strides,
                        ncnn::Mat const& pred,
                        float prob_threshold,
                        std::vector<std::string> const& classes,
                        std::vector<Item>& objects)
{
  const int num_points = grid_strides.size();
  const int num_class = 80;
  const int reg_max_1 = 16;

  for (int i = 0; i < num_points; i++)
  {
    const float *scores = pred.row(i) + 4 * reg_max_1;

    // find label with max score
    int label = -1;
    float score = -FLT_MAX;
    for (int k = 0; k < num_class; k++)
    {
      float confidence = scores[k];
      if (confidence > score)
      {
        label = k;
        score = confidence;
      }
    }
    float box_prob = sigmoid(score);
    if (box_prob >= prob_threshold)
    {
      ncnn::Mat bbox_pred(reg_max_1, 4, (void *) pred.row(i));
      {
        ncnn::Layer *softmax = ncnn::create_layer("Softmax");

        ncnn::ParamDict pd;
        pd.set(0, 1); // axis
        pd.set(1, 1);
        softmax->load_param(pd);

        ncnn::Option opt;
        opt.num_threads = 1;
        opt.use_packing_layout = false;

        softmax->create_pipeline(opt);
        softmax->forward_inplace(bbox_pred, opt);
        softmax->destroy_pipeline(opt);
        delete softmax;
      }

      float pred_ltrb[4];
      for (int k = 0; k < 4; k++)
      {
        float dis = 0.f;
        const float *dis_after_sm = bbox_pred.row(k);
        for (int l = 0; l < reg_max_1; l++)
        {
          dis += l * dis_after_sm[l];
        }

        pred_ltrb[k] = dis * grid_strides[i].stride;
      }

      float pb_cx = (grid_strides[i].grid0 + 0.5f) * grid_strides[i].stride;
      float pb_cy = (grid_strides[i].grid1 + 0.5f) * grid_strides[i].stride;

      float x0 = pb_cx - pred_ltrb[0];
      float y0 = pb_cy - pred_ltrb[1];
      float x1 = pb_cx + pred_ltrb[2];
      float y1 = pb_cy + pred_ltrb[3];

      Item obj;
      obj.boundingBox.x = x0;
      obj.boundingBox.y = y0;
      obj.boundingBox.width = x1 - x0;
      obj.boundingBox.height = y1 - y0;
      obj.className = &classes[label];
      obj.confidence = box_prob;

      objects.push_back(obj);
    }
  }
}

} // anonymous namespace

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

class V8::Impl
{
public:
  Impl(std::string const& modelFile,
       std::string const& weightsFile,
       std::string const& classesFile,
       Size inputSize,
       float confidenceThreshold = 0.25f,
       float nmsThreshold = 0.25f)
    : _inputSize{inputSize}
    , _classes{readClasses(classesFile)}
    , _confThreshold{confidenceThreshold}
    , _nmsThreshold{nmsThreshold}
  {
    //_net.clear();

    //_net.opt = ncnn::Option();
    //_net.opt.num_threads = 4;

    _net.load_param(modelFile.c_str());
    _net.load_model(weightsFile.c_str());

    _mean_vals[0] = 103.53f;
    _mean_vals[1] = 116.28f;
    _mean_vals[2] = 123.675f;
    _norm_vals[0] = 1.0 / 255.0f;
    _norm_vals[1] = 1.0 / 255.0f;
    _norm_vals[2] = 1.0 / 255.0f;
  }

  auto performPrediction(cv::Mat const& frame,
                         std::function<bool(std::string const&)>&& filter = [](std::string const&) { return true; },
                         bool isNeededToBeSwappedRAndB = true) -> Item::List
  {
    int width = frame.cols;
    int height = frame.rows;

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

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(frame.data,
                                                 isNeededToBeSwappedRAndB
                                                     ? ncnn::Mat::PIXEL_RGB2BGR
                                                     : ncnn::Mat::PIXEL_RGB,
                                                 width,
                                                 height,
                                                 w,
                                                 h);

    // pad to target_size rectangle
    int wpad = (w + 31) / 32 * 32 - w;
    int hpad = (h + 31) / 32 * 32 - h;
    ncnn::Mat in_pad;
    ncnn::copy_make_border(in, in_pad, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, ncnn::BORDER_CONSTANT, 0.f);

    in_pad.substract_mean_normalize(nullptr, _norm_vals);

    ncnn::Extractor ex = _net.create_extractor();

    ex.input("images", in_pad);

    std::vector<Item> proposals;

    ncnn::Mat out;
    ex.extract("output", out);

    std::vector<int> strides = {8, 16, 32}; // might have stride=64
    std::vector<GridAndStride> grid_strides;
    generate_grids_and_stride(in_pad.w, in_pad.h, strides, grid_strides);
    generate_proposals(grid_strides, out, _confThreshold, _classes, proposals);

    // sort all proposals by score from highest to lowest
    qsort_descent_inplace(proposals);

    // apply nms with nms_threshold
    std::vector<int> picked;
    nms_sorted_bboxes(proposals, picked, _nmsThreshold);

    int count = picked.size();

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
    }

    // sort objects by area
    struct
    {
      bool operator()(const Item& a, const Item& b) const
      {
        return a.boundingBox.area() > b.boundingBox.area();
      }
    } objects_area_greater;
    std::sort(objects.begin(), objects.end(), objects_area_greater);
    return objects;
  }

//  auto frameExtract(std::vector<::cv::Mat> const& outs,
//                    cv::Size const& frameSize,
//                    std::function<bool(std::string const&)>&& filter) const -> Item::List
//  {
//  }

private:
  std::vector<std::string> _classes;
  Size _inputSize;
  float _confThreshold{};
  float _nmsThreshold{};
  float _mean_vals[3];
  float _norm_vals[3];
  ncnn::Net _net;
};

V8::V8(std::string const& modelFile,
       std::string const& weightsFile,
       std::string const& classesFile,
       Size inputSize,
       float confidenceThreshold,
       float nmsThreshold)
   : _impl{std::make_shared<Impl>(modelFile, weightsFile, classesFile, inputSize, confidenceThreshold, nmsThreshold)}
{
}

//V8::V8(std::string const& modelFile,
//       std::string const& classesFile,
//       Size inputSize,
//       float confidenceThreshold = 0.25f,
//       float nmsThreshold = 0.25f)
//   : Impl(modelFile, {}, inputSize, confidenceThreshold, nmsThreshold)
//{
//
//}

auto V8::performPrediction(cv::Mat const& frame,
                           std::function<bool(std::string const&)>&& filter,
                           bool isNeededToBeSwappedRAndB) -> Item::List
{
    return _impl->performPrediction(frame, std::move(filter), isNeededToBeSwappedRAndB);
}

//auto frameExtract(std::vector<::cv::Mat> const& outs,
//                  cv::Size const& frameSize,
//                  std::function<bool(std::string const&)>&& filter) const -> Item::List
//{
//}

}// OIYolo

