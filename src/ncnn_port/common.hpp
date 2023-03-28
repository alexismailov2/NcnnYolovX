#pragma once

#include <OIYolo/CommonTypes.hpp>

#ifdef OIYolo_NCNN

#include <ncnn/mat.h>

#include <vector>

struct GridAndStride
{
  int grid0;
  int grid1;
  int stride;
};

void slice(ncnn::Mat const &in, ncnn::Mat &out, int start, int end, int axis);
void interp(ncnn::Mat const &in, float scale, int outWidth, int outHeight, ncnn::Mat &out);
void reshape(ncnn::Mat const &in, int c, int h, int w, int d, ncnn::Mat &out);
void sigmoid(ncnn::Mat& bottom);
void matmul(std::vector<ncnn::Mat> const &bottom_blobs, ncnn::Mat &top_blob);
void qsort_descent_inplace(std::vector<OIYolo::Item> &faceobjects, int left, int right);
void qsort_descent_inplace(std::vector<OIYolo::Item> &faceobjects);
float fast_exp(float x);
float sigmoid(float x);
void nms_sorted_bboxes(std::vector<OIYolo::Item> const &faceobjects, std::vector<int> &picked, float nms_threshold);
void generate_grids_and_stride(const int target_w, const int target_h, std::vector<int> &strides, std::vector<GridAndStride> &grid_strides);
void generate_proposals(std::vector<GridAndStride> grid_strides,
                        ncnn::Mat const& pred,
                        float prob_threshold,
                        bool isSeg,
                        std::vector<std::string> const& classes,
                        std::vector<OIYolo::Item> &objects);
void decode_mask(ncnn::Mat const &mask_feat,
                 int const &img_w,
                 int const &img_h,
                 ncnn::Mat const &mask_proto,
                 ncnn::Mat const &in_pad,
                 int const &wpad,
                 int const &hpad,
                 ncnn::Mat &mask_pred_result);
void transpose(const ncnn::Mat& in, ncnn::Mat& out);
#endif