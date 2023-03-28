#if 1
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
    cv::Size labelSize = getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
    top = cv::max(top, labelSize.height);
    putText(img, label, cv::Point(left, top), cv::FONT_HERSHEY_SIMPLEX, 1, color[result[i].id], 2);
  }
  addWeighted(img, 0.5, mask, 0.5, 0, img);
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
                           false,//true,//false,
                           0.3f,
                           0.3f,
                           0.3f};

    OIYolo::Item::List predictions;
    {
      TAKEN_TIME();
      predictions = yolo.performPrediction(input);
    }

    std::vector<cv::Scalar> color;
    srand(time(nullptr));
    for (int i = 0; i < 80; i++)
    {
      color.push_back(cv::Scalar(rand() % 256, rand() % 256, rand() % 256));
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
#else
// Cpp native
#include <fstream>
#include <vector>
#include <string>
#include <random>

// OpenCV / DNN / Inference
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

struct Detection
{
  int class_id{0};
  std::string className{};
  float confidence{0.0};
  cv::Scalar color{};
  cv::Rect box{};
};

class Inference
{
public:
  Inference(const std::string &onnxModelPath,
            const cv::Size2f &modelInputShape,
            const std::string &classesTxtFile,
            const bool &runWithCuda = true)
            : modelPath{onnxModelPath}
            , modelShape{modelInputShape}
            , classesPath{classesTxtFile}
            , cudaEnabled{runWithCuda}
  {
    net = cv::dnn::readNetFromONNX(modelPath);
    net.setPreferableBackend(cudaEnabled ? cv::dnn::DNN_BACKEND_CUDA : cv::dnn::DNN_BACKEND_OPENCV);
    net.setPreferableTarget(cudaEnabled ? cv::dnn::DNN_TARGET_CUDA : cv::dnn::DNN_TARGET_CPU);
    loadClassesFromFile();
  }
  std::vector<Detection> runInference(void const* data, int width, int height/*const cv::Mat &input*/);

private:
  void loadClassesFromFile();
  cv::Mat formatToSquare(const cv::Mat &source);

  std::string modelPath{};
  std::string classesPath{};
  bool cudaEnabled{};

  std::vector<std::string> classes{};
  cv::Size2f modelShape{};

  float modelConfidenseThreshold{0.25};
  float modelScoreThreshold{0.45};
  float modelNMSThreshold{0.50};
  bool letterBoxForSquare{true};

  cv::dnn::Net net;
};

std::vector<Detection> Inference::runInference(void const* frameData, int width, int height/*const cv::Mat &input*//*const cv::Mat &input*/)
{
  cv::Mat modelInput = cv::Mat(cv::Mat((int)height, (int)width, CV_8UC3, (void*)frameData));//input;
  if (letterBoxForSquare && modelShape.width == modelShape.height)
  {
    modelInput = formatToSquare(modelInput);
  }

  cv::Mat blob;
  cv::dnn::blobFromImage(modelInput, blob, 1.0/255.0, modelShape, cv::Scalar(), true, false);
  net.setInput(blob);

  std::vector<cv::Mat> outputs;
  net.forward(outputs, net.getUnconnectedOutLayersNames());

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
  float *data = (float *)outputs[0].data;

  float x_factor = modelInput.cols / modelShape.width;
  float y_factor = modelInput.rows / modelShape.height;

  std::vector<int> class_ids;
  std::vector<float> confidences;
  std::vector<cv::Rect> boxes;

  for (int i = 0; i < rows; ++i)
  {
    if (yolov8)
    {
      float *classes_scores = data+4;

      cv::Mat scores(1, classes.size(), CV_32FC1, classes_scores);
      cv::Point class_id;
      double maxClassScore;

      minMaxLoc(scores, 0, &maxClassScore, 0, &class_id);

      if (maxClassScore > modelScoreThreshold)
      {
        confidences.push_back(maxClassScore);
        class_ids.push_back(class_id.x);

        float x = data[0];
        float y = data[1];
        float w = data[2];
        float h = data[3];

        int left = int((x - 0.5 * w) * x_factor);
        int top = int((y - 0.5 * h) * y_factor);

        int width = int(w * x_factor);
        int height = int(h * y_factor);

        boxes.push_back(cv::Rect(left, top, width, height));
      }
    }
    else // yolov5
    {
      float confidence = data[4];

      if (confidence >= modelConfidenseThreshold)
      {
        float *classes_scores = data+5;

        cv::Mat scores(1, classes.size(), CV_32FC1, classes_scores);
        cv::Point class_id;
        double max_class_score;

        minMaxLoc(scores, 0, &max_class_score, 0, &class_id);

        if (max_class_score > modelScoreThreshold)
        {
          confidences.push_back(confidence);
          class_ids.push_back(class_id.x);

          float x = data[0];
          float y = data[1];
          float w = data[2];
          float h = data[3];

          int left = int((x - 0.5 * w) * x_factor);
          int top = int((y - 0.5 * h) * y_factor);

          int width = int(w * x_factor);
          int height = int(h * y_factor);

          boxes.push_back(cv::Rect(left, top, width, height));
        }
      }
    }

    data += dimensions;
  }

  std::vector<int> nms_result;
  cv::dnn::NMSBoxes(boxes, confidences, modelScoreThreshold, modelNMSThreshold, nms_result);

  std::vector<Detection> detections{};
  for (unsigned long i = 0; i < nms_result.size(); ++i)
  {
    int idx = nms_result[i];

    Detection result;
    result.class_id = class_ids[idx];
    result.confidence = confidences[idx];

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dis(100, 255);
    result.color = cv::Scalar(dis(gen),
                              dis(gen),
                              dis(gen));

    result.className = classes[result.class_id];
    result.box = boxes[idx];

    detections.push_back(result);
  }

  return detections;
}

void Inference::loadClassesFromFile()
{
  std::ifstream inputFile(classesPath);
  if (inputFile.is_open())
  {
    std::string classLine;
    while (std::getline(inputFile, classLine))
      classes.push_back(classLine);
    inputFile.close();
  }
}

cv::Mat Inference::formatToSquare(const cv::Mat &source)
{
  int col = source.cols;
  int row = source.rows;
  int _max = MAX(col, row);
  cv::Mat result = cv::Mat::zeros(_max, _max, CV_8UC3);
  source.copyTo(result(cv::Rect(0, 0, col, row)));
  return result;
}

int main(int argc, char **argv)
{
  Inference inf("./OIYolo/assets/yolov8s-2.onnx", cv::Size(640, 640/*480*/),
                "./OIYolo/assets/yolov8s.classes", false);

  std::vector<std::string> imageNames;
  imageNames.push_back("./OIYolo/assets/parking.jpg");
  //imageNames.push_back(projectBasePath + "/source/data/zidane.jpg");

  for (int i = 0; i < imageNames.size(); ++i)
  {
    cv::Mat frame = cv::imread(imageNames[i], cv::IMREAD_COLOR);

    // Inference starts here...
    std::vector<Detection> output = inf.runInference(frame.data, frame.cols, frame.rows);

    int detections = output.size();
    std::cout << "Number of detections:" << detections << std::endl;

    for (int j = 0; j < detections; ++j)
    {
      Detection detection = output[j];

      cv::Rect box = detection.box;
      cv::Scalar color = detection.color;

      // Detection box
      cv::rectangle(frame, box, color, 2);

      // Detection box text
      std::string classString = detection.className + ' ' + std::to_string(detection.confidence).substr(0, 4);
      cv::Size textSize = cv::getTextSize(classString, cv::FONT_HERSHEY_DUPLEX, 1, 2, 0);
      cv::Rect textBox(box.x, box.y - 40, textSize.width + 10, textSize.height + 20);

      cv::rectangle(frame, textBox, color, cv::FILLED);
      cv::putText(frame, classString, cv::Point(box.x + 5, box.y - 10), cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0, 0, 0), 2, 0);
    }
    // Inference ends here...

    // This is only for preview purposes
    float scale = 0.8;
    cv::resize(frame, frame, cv::Size(frame.cols*scale, frame.rows*scale));
    cv::imshow("Inference", frame);

    cv::waitKey(-1);
  }
}
#endif