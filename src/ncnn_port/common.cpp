#include "common.hpp"

#ifdef OIYolo_NCNN
#include <ncnn/layer.h>

void slice(ncnn::Mat const &in, int start, int end, int axis, ncnn::Mat &out)
{
  auto op = std::unique_ptr<ncnn::Layer>(ncnn::create_layer("Crop"));

  ncnn::ParamDict pd;

  ncnn::Mat axes = ncnn::Mat(1);
  axes.fill(axis);
  ncnn::Mat ends = ncnn::Mat(1);
  ends.fill(end);
  ncnn::Mat starts = ncnn::Mat(1);
  starts.fill(start);
  pd.set(9, starts);// start
  pd.set(10, ends);// end
  pd.set(11, axes);//axes

  op->load_param(pd);

  ncnn::Option opt;
  opt.num_threads = 4;
  opt.use_fp16_storage = false;
  opt.use_packing_layout = false;

  op->create_pipeline(opt);
  op->forward(in, out, opt);
  op->destroy_pipeline(opt);
}

void interp(ncnn::Mat const &in, float scale, int outWidth, int outHeight, ncnn::Mat &out)
{
  auto op = std::unique_ptr<ncnn::Layer>(ncnn::create_layer("Interp"));

  ncnn::ParamDict pd;
  pd.set(0, 2);// resize type
  pd.set(1, scale);// height scale
  pd.set(2, scale);// width scale
  pd.set(3, outHeight);
  pd.set(4, outWidth);

  op->load_param(pd);

  ncnn::Option opt;
  opt.num_threads = 4;
  opt.use_fp16_storage = false;
  opt.use_packing_layout = false;

  op->create_pipeline(opt);
  op->forward(in, out, opt);
  op->destroy_pipeline(opt);
}

void reshape(ncnn::Mat const &in, int c, int h, int w, int d, ncnn::Mat &out)
{
  auto op = std::unique_ptr<ncnn::Layer>(ncnn::create_layer("Reshape"));

  ncnn::ParamDict pd;

  pd.set(0, w);// start
  pd.set(1, h);// end
  if (d > 0)
  {
    pd.set(11, d);//axes
  }
  pd.set(2, c);//axes
  op->load_param(pd);

  ncnn::Option opt;
  opt.num_threads = 4;
  opt.use_fp16_storage = false;
  opt.use_packing_layout = false;

  op->create_pipeline(opt);
  op->forward(in, out, opt);
  op->destroy_pipeline(opt);
}

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

void qsort_descent_inplace(std::vector<OIYolo::Item> &faceobjects, int left, int right)
{
  int i = left;
  int j = right;
  float p = faceobjects[(left + right) / 2].confidence;

  while (i <= j) {
    while (faceobjects[i].confidence > p)
      i++;

    while (faceobjects[j].confidence < p)
      j--;

    if (i <= j) {
      // swap
      std::swap(faceobjects[i], faceobjects[j]);

      i++;
      j--;
    }
  }

#pragma omp parallel sections
  {
#pragma omp section
    {
      if (left < j) qsort_descent_inplace(faceobjects, left, j);
    }
#pragma omp section
    {
      if (i < right) qsort_descent_inplace(faceobjects, i, right);
    }
  }
}

void qsort_descent_inplace(std::vector<OIYolo::Item> &faceobjects)
{
  if (faceobjects.empty()) {
    return;
  }

  qsort_descent_inplace(faceobjects, 0, faceobjects.size() - 1);
}

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

#if 0
void nms_sorted_bboxes(std::vector<OIYolo::Item> const &faceobjects,
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
    OIYolo::Item const &a = faceobjects[i];

    int keep = 1;
    for (int j = 0; j < (int) picked.size(); j++) {
      OIYolo::Item const &b = faceobjects[picked[j]];

      // intersection over union
      float inter_area = (a.boundingBox & b.boundingBox).area();
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
#endif

void generate_grids_and_stride(const int target_w,
                               const int target_h,
                               std::vector<int> &strides,
                               std::vector<GridAndStride> &grid_strides)
{
  for (auto& stride : strides) {
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
                        const ncnn::Mat &pred,
                        float prob_threshold,
                        bool isSeg,
                        std::vector<std::string> const& classes,
                        std::vector<OIYolo::Item> &objects)
{
  const int num_points = grid_strides.size();
  const int num_class = classes.size();
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

      OIYolo::Item obj;
      obj.boundingBox.x = x0;
      obj.boundingBox.y = y0;
      obj.boundingBox.width = x1 - x0;
      obj.boundingBox.height = y1 - y0;
      obj.className = &classes[label];
      obj.confidence = box_prob;
      if (isSeg)
      {
        obj.mask_feat.resize(32);
        std::copy(pred.row(i) + 64 + num_class, pred.row(i) + 64 + num_class + 32, obj.mask_feat.begin());
      }
      objects.push_back(obj);
    }
  }
}

void decode_mask(ncnn::Mat const &mask_feat,
                 int const &img_w,
                 int const &img_h,
                 ncnn::Mat const &mask_proto,
                 ncnn::Mat const &in_pad,
                 int const &wpad,
                 int const &hpad,
                 ncnn::Mat &mask_pred_result)
{
  ncnn::Mat masks;
  matmul(std::vector<ncnn::Mat>{mask_feat, mask_proto}, masks);
  sigmoid(masks);
  reshape(masks, masks.h, in_pad.h / 4, in_pad.w / 4, 0, masks);
  slice(masks, (wpad / 2) / 4, (in_pad.w - wpad / 2) / 4, 2, mask_pred_result);
  slice(mask_pred_result, (hpad / 2) / 4, (in_pad.h - hpad / 2) / 4, 1, mask_pred_result);
  interp(mask_pred_result, 4.0, img_w, img_h, mask_pred_result);
}

void transpose(const ncnn::Mat& in, ncnn::Mat& out)
{
  ncnn::Option opt;
  opt.num_threads = 2;
  opt.use_fp16_storage = false;
  opt.use_packing_layout = true;

  ncnn::Layer* op = ncnn::create_layer("Permute");

  // set param
  ncnn::ParamDict pd;
  pd.set(0, 1);// order_type

  op->load_param(pd);

  op->create_pipeline(opt);

  ncnn::Mat in_packed = in;
  {
    // resolve dst_elempack
    int dims = in.dims;
    int elemcount = 0;
    if (dims == 1) elemcount = in.elempack * in.w;
    if (dims == 2) elemcount = in.elempack * in.h;
    if (dims == 3) elemcount = in.elempack * in.c;

    int dst_elempack = 1;
    if (op->support_packing)
    {
      if (elemcount % 8 == 0 && /*(ncnn::cpu_support_x86_avx2() || ncnn::cpu_support_x86_avx())*/false)
        dst_elempack = 8;
      else if (elemcount % 4 == 0)
        dst_elempack = 4;
    }

    if (in.elempack != dst_elempack)
    {
      convert_packing(in, in_packed, dst_elempack, opt);
    }
  }

  // forward
  op->forward(in_packed, out, opt);

  op->destroy_pipeline(opt);

  delete op;
}
#endif