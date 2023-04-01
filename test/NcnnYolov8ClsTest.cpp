#include <OIYolo/V8Cls.hpp>
#include <OIYolo/TimeMeasuring.hpp>

#include <opencv2/imgcodecs.hpp>

#include <iostream>
#include <vector>

void print(OIYolo::Item::List const& list)
{
  TAKEN_TIME();
  for (auto const& item : list)
  {
    printf("label: %s, confidence: %f, x: %f, y: %f, w: %f, h: %f\n",
           item.className->c_str(), item.confidence, item.boundingBox.x,
           item.boundingBox.y, item.boundingBox.width, item.boundingBox.height);
  }
}

auto main(int argc, char** argv) -> int32_t
{
  if (argc != 6)
  {
    fprintf(stderr, "%s <image file> <model> <weights> <classes> size\n", argv[0]);
    return 0;
  }

  cv::Mat input = cv::imread(argv[1], cv::IMREAD_COLOR);
  if (input.empty())
  {
    fprintf(stderr, "cv::imread %s failed\n", argv[1]);
    return -1;
  }

  auto yolo = OIYolo::V8Cls{argv[2],
                            argv[3],
                            argv[4],
                            OIYolo::Size{atoi(argv[5]), atoi(argv[5])}};

  OIYolo::Item::List predictions;
  {
    TAKEN_TIME();
    predictions = yolo.performPrediction(input);
  }

  print(predictions);

  return 0;
}
