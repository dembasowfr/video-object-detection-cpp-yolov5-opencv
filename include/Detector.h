#ifndef DETECTOR_H
#define DETECTOR_H

#include <opencv2/opencv.hpp>
#include "Detection.h"

class Detector {
public:
    // this constructor will load the model from the given path and use CUDA if the useCUDA flag is set to true
    Detector(const std::string& modelPath, bool useCUDA);

    // this function is used to detect objects in the input image
    void detect(const cv::Mat& image, std::vector<Detection>& output, const std::vector<std::string>& className);

private:
    // this is the neural network model
    cv::dnn::Net net;
    // this is the list of classes that the model can detect(objects), will be loaded from a file
    std::vector<std::string> classList;

    // this function is used to format the input image to the required size for YOLOv5
    cv::Mat formatYOLOv5(const cv::Mat& source);
};

#endif // DETECTOR_H
