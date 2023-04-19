#include <OIYolo/V8.hpp>
#include <OIYolo/TimeMeasuring.hpp>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <iostream>
#include <vector>

namespace {

void draw_objects(cv::Mat const& bgr,
                  std::vector<OIYolo::Item> const& objects)
{
  static const unsigned char colors[81][3] = {
    {56,  0,   255},
    {226, 255, 0},
    {0,   94,  255},
    {0,   37,  255},
    {0,   255, 94},
    {255, 226, 0},
    {0,   18,  255},
    {255, 151, 0},
    {170, 0,   255},
    {0,   255, 56},
    {255, 0,   75},
    {0,   75,  255},
    {0,   255, 169},
    {255, 0,   207},
    {75,  255, 0},
    {207, 0,   255},
    {37,  0,   255},
    {0,   207, 255},
    {94,  0,   255},
    {0,   255, 113},
    {255, 18,  0},
    {255, 0,   56},
    {18,  0,   255},
    {0,   255, 226},
    {170, 255, 0},
    {255, 0,   245},
    {151, 255, 0},
    {132, 255, 0},
    {75,  0,   255},
    {151, 0,   255},
    {0,   151, 255},
    {132, 0,   255},
    {0,   255, 245},
    {255, 132, 0},
    {226, 0,   255},
    {255, 37,  0},
    {207, 255, 0},
    {0,   255, 207},
    {94,  255, 0},
    {0,   226, 255},
    {56,  255, 0},
    {255, 94,  0},
    {255, 113, 0},
    {0,   132, 255},
    {255, 0,   132},
    {255, 170, 0},
    {255, 0,   188},
    {113, 255, 0},
    {245, 0,   255},
    {113, 0,   255},
    {255, 188, 0},
    {0,   113, 255},
    {255, 0,   0},
    {0,   56,  255},
    {255, 0,   113},
    {0,   255, 188},
    {255, 0,   94},
    {255, 0,   18},
    {18,  255, 0},
    {0,   255, 132},
    {0,   188, 255},
    {0,   245, 255},
    {0,   169, 255},
    {37,  255, 0},
    {255, 0,   151},
    {188, 0,   255},
    {0,   255, 37},
    {0,   255, 0},
    {255, 0,   170},
    {255, 0,   37},
    {255, 75,  0},
    {0,   0,   255},
    {255, 207, 0},
    {255, 0,   226},
    {255, 245, 0},
    {188, 255, 0},
    {0,   255, 18},
    {0,   255, 75},
    {0,   255, 151},
    {255, 56,  0},
    {245, 255, 0}
  };
  cv::Mat image = bgr.clone();
  int color_index = 0;
  for (size_t i = 0; i < objects.size(); i++)
  {
    const OIYolo::Item& obj = objects[i];
    const unsigned char* color = colors[color_index % 80];
    color_index++;

    cv::Scalar cc(color[0], color[1], color[2]);

    fprintf(stderr, "%s = %.5f at %.2f %.2f %.2f x %.2f\n", obj.className->c_str(), obj.confidence,
            obj.boundingBox.x, obj.boundingBox.y, obj.boundingBox.width, obj.boundingBox.height);
    if (!obj.mask.empty()) {
      for (int y = 0; y < image.rows; y++) {
        uchar *image_ptr = image.ptr(y);
        const float *mask_ptr = obj.mask.ptr<float>(y);
        for (int x = 0; x < image.cols; x++) {
          if (mask_ptr[x] >= 0.5) {
            image_ptr[0] = cv::saturate_cast<uchar>(image_ptr[0] * 0.5 + color[2] * 0.5);
            image_ptr[1] = cv::saturate_cast<uchar>(image_ptr[1] * 0.5 + color[1] * 0.5);
            image_ptr[2] = cv::saturate_cast<uchar>(image_ptr[2] * 0.5 + color[0] * 0.5);
          }
          image_ptr += 3;
        }
      }
    }
    cv::rectangle(image, (cv::Rect2d)obj.boundingBox, cc, 2);

    char text[256];
    sprintf(text, "%s %.1f%%", obj.className->c_str(), obj.confidence * 100);

    int baseLine = 0;
    cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

    int x = (int)obj.boundingBox.x;
    int y = obj.boundingBox.y - label_size.height - baseLine;
    if (y < 0)
      y = 0;
    if (x + label_size.width > image.cols)
      x = image.cols - label_size.width;

    cv::rectangle(image, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                  cv::Scalar(255, 255, 255), -1);

    cv::putText(image, text, cv::Point(x, y + label_size.height),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
  }

  cv::imshow("image", image);
  cv::waitKey(0);
}

void draw(cv::Mat& frame,
          OIYolo::Item::List const& list,
          cv::Scalar color,
          int32_t thickness)
{
  TAKEN_TIME();
  for (int i = 0; i < list.size(); ++i)
  {
    cv::rectangle(frame,
                  cv::Rect2d{list[i].boundingBox.x,
                             list[i].boundingBox.y,
                             list[i].boundingBox.width,
                             list[i].boundingBox.height},
                  color,
                  thickness);
  }
}

void DrawPred(cv::Mat& img,
              std::vector<OIYolo::Item> result,
              std::vector<cv::Scalar> color)
{
  cv::Mat mask = img.clone();
  for (int i = 0; i < result.size(); i++)
  {
    printf("label: %s, confidence: %f, x: %f, y: %f, w: %f, h: %f\n",
           result[i].className->c_str(), result[i].confidence, result[i].boundingBox.x,
           result[i].boundingBox.y, result[i].boundingBox.width, result[i].boundingBox.height);
    int left = result[i].boundingBox.x;
    int top = result[i].boundingBox.y;
    int color_num = i;
    cv::rectangle(img, result[i].boundingBox, color[result[i].id], 2, 8);
    if (result[i].mask.rows &&
        result[i].mask.cols > 0)
    {
      mask(result[i].boundingBox).setTo(color[result[i].id], result[i].mask);
    }
    std::string label = *result[i].className + ":" + std::to_string(result[i].confidence);
    int baseLine;
    cv::Size labelSize = getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 1, 2, &baseLine);
    cv::rectangle(img, cv::Rect(cv::Point(left, top - baseLine*2), cv::Size(labelSize.width, labelSize.height)),
                  cv::Scalar(255, 255, 255), -1);
    top = cv::max(top, labelSize.height);
    putText(img, label, cv::Point(left, top), cv::FONT_HERSHEY_SIMPLEX, 1, color[result[i].id], 2);
  }
  addWeighted(img, 0.8, mask, 0.5, 0, img);
  imshow("1", img);
  cv::waitKey();
}
} /// end namespace anonymous

auto main(int argc, char** argv) -> int32_t
{
  if (argc != 6)
  {
    fprintf(stderr, "%s <image file> <model> <weights> <classes> size\n", argv[0]);
    return 0;
  }

  static const std::string kWinName = "NCNN YoloV8 Demo";

  cv::Mat input = cv::imread(argv[1], cv::IMREAD_COLOR);
  if (input.empty())
  {
    fprintf(stderr, "cv::imread %s failed\n", argv[1]);
    return -1;
  }

  {
    auto yolo = OIYolo::V8{argv[2],
                           argv[3],
                           argv[4],
                           OIYolo::Size{atoi(argv[5]), atoi(argv[5])},
                           false,
                           0.7f,
                           0.7f,
                           0.7f};

    OIYolo::Item::List predictions;
    {
      TAKEN_TIME();
      predictions = yolo.performPrediction(input);
    }

    std::vector<cv::Scalar> color;
    srand(time(nullptr));
    for (int i = 0; i < 80; i++)
    {
      color.push_back(cv::Scalar(0, 0, 0));
    }
    //draw_objects(input, predictions);
    //draw(input, predictions, cv::Scalar{0x00, 0xFF, 0x00}, 2);
    DrawPred(input, predictions, color);
    cv::imshow(kWinName, input);
    cv::waitKey(0);
  }

  cv::destroyAllWindows();
  return 0;
}
