#include <OIYolo/V8.hpp>
#include <OIYolo/TimeMeasuring.hpp>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <iostream>
#include <vector>

namespace {

#if 0 // TODO: Shoud be deleted
int draw(cv::Mat& rgb,
         std::vector<OIYolo::Item> const& objects)
{
  for (auto const& obj : objects)
  {
//         fprintf(stderr, "%d = %.5f at %.2f %.2f %.2f x %.2f\n", obj.label, obj.prob,
//                 obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height);

    cv::rectangle(rgb, cv::Rect2d{obj.boundingBox.x,
                                  obj.boundingBox.y,
                                  obj.boundingBox.width,
                                  obj.boundingBox.height}, cv::Scalar(255, 0, 0));

    char text[256];
    sprintf(text, "%s %.1f%%", obj.className->c_str(), obj.confidence * 100);

    int baseLine = 0;
    cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

    int x = obj.boundingBox.x;
    int y = obj.boundingBox.y - label_size.height - baseLine;
    if (y < 0)
      y = 0;
    if (x + label_size.width > rgb.cols)
      x = rgb.cols - label_size.width;

    cv::rectangle(rgb, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                  cv::Scalar(255, 255, 255), -1);

    cv::putText(rgb, text, cv::Point(x, y + label_size.height),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
  }

  return 0;
}
#endif

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
                           OIYolo::Size{atoi(argv[5]), atoi(argv[5])}, 0.3f, 0.3f};

    OIYolo::Item::List predictions;
    {
      TAKEN_TIME();
      predictions = yolo.performPrediction(input);
    }

    draw(input, predictions, cv::Scalar{0x00, 0xFF, 0x00}, 2);
    cv::imshow(kWinName, input);
    cv::waitKey(0);
  }

  cv::destroyAllWindows();
  return 0;
}