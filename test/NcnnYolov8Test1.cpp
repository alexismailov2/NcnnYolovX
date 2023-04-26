#include <OIYolo/V8.hpp>
#include <OIYolo/TimeMeasuring.hpp>

#include "jpeg_decoder.h"

#include <vector>
#include <cstring>
#include <stdio.h>

namespace {

bool load_jpg_data(std::string const& fpath,
                   std::vector<uint8_t>& data,
                   int& width, int& height)
{
  FILE *f = fopen(fpath.c_str(), "rb");
  if (!f) { return false; }
  fseek(f, 0, SEEK_END);
  size_t size = ftell(f);
  unsigned char *buf = (unsigned char*)malloc(size);
  fseek(f, 0, SEEK_SET);
  size_t read = fread(buf, 1, size, f);
  fclose(f);
  Jpeg::Decoder decoder(buf, size);
  if (decoder.GetResult() != Jpeg::Decoder::OK)
  {
    const char* error_msgs[] = { "OK", "NotAJpeg", "Unsupported", "OutOfMemory", "InternalError", "SyntaxError", "Internal_Finished" };
    printf("Error decoding the input file %s\n", error_msgs[decoder.GetResult()]);
    return false;
  }
  if (!decoder.IsColor())
  {
    printf("Need a color image for this demo");
    return false;
  }
  width = decoder.GetWidth();
  height = decoder.GetHeight();
  data.resize(width*height*3);
  std::memcpy(data.data(), decoder.GetImage(), data.size());
  return true;
}

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
} /// end namespace anonymous

auto main(int argc, char** argv) -> int32_t
{
  if (argc != 6)
  {
    fprintf(stderr, "%s <image file> <model> <weights> <classes> size\n", argv[0]);
    return 0;
  }

  auto const targetSize = atoi(argv[5]);

  std::vector<uint8_t> input;
  int width{};
  int height{};
  if (!load_jpg_data(argv[1], input, width, height))
  {
    fprintf(stderr, "Image file could not be read %s\n", argv[1]);
    return 1;
  }

  {
    auto yolo = OIYolo::V8{argv[2],
                           argv[3],
                           argv[4],
                           OIYolo::Size{targetSize, targetSize},
                           false,
                           0.25f,
                           0.7f,
                           0.3f};

    OIYolo::Item::List predictions;
    {
      TAKEN_TIME();
      predictions = yolo.performPrediction((const char*)input.data(), width, height, true);
    }

    print(predictions);
  }
  return 0;
}
