#include <iostream>
#include "Image.h"
#include "Detector.h"
#include "Utils.h"

int main(int argc, char** argv) {


    // Check if the user wants to use CUDA
    bool useCUDA = argc > 1 && strcmp(argv[1], "cuda") == 0;

    std::string imagePath = "../data/input/image.webp";
    Image image(imagePath);
    if (image.isEmpty()) {
        return -1;
    }

    std::string modelPath = "../models/yolov5s.onnx";
    Detector detector(modelPath, useCUDA);
    std::vector<Detection> detections;


    std::string classesPath = "../data/input/classes.txt";
    auto classList = loadClassList(classesPath);

    // This function is used to detect objects in the input image and assign the detections vector to a new value ✅
    detector.detect(image.getMat(), detections, classList);

    
    // Take detections and draw them on the image ✅
    image.drawDetections(detections, classList);

    // Save the image with the detections to a file ✅
    image.saveOutputImage("Detections", 800);

    return 0;
}
