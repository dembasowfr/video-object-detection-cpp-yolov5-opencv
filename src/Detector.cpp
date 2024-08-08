#include "Detector.h"
#include <iostream>

const float INPUT_WIDTH = 640.0;
const float INPUT_HEIGHT = 640.0;
const float SCORE_THRESHOLD = 0.2;
const float NMS_THRESHOLD = 0.4;
const float CONFIDENCE_THRESHOLD = 0.4;

Detector::Detector(const std::string& modelPath, bool useCUDA) {


    net = cv::dnn::readNet(modelPath);

    if (useCUDA) {
        std::cout << "Attempt to use CUDA\n";
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA_FP16);
    } else {
        std::cout << "Running on CPU\n";
        std::cout << "Program running ... \n";
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    }


}

// This function is used to format the input image to the required size for YOLOv5 ✅
cv::Mat Detector::formatYOLOv5(const cv::Mat& source) {

    int col = source.cols;
    int row = source.rows;
    int _max = std::max(col, row);
    cv::Mat result = cv::Mat::zeros(_max, _max, CV_8UC3);
    source.copyTo(result(cv::Rect(0, 0, col, row)));
    return result;
}


// This function is used to detect objects in the input image ✅
void Detector::detect(const cv::Mat& image, std::vector<Detection>& detections, const std::vector<std::string>& classList) {

    // blob is the input size for YOLOv5
    cv::Mat blob;

    // format the input image to the required size for YOLOv5
    auto inputImage = formatYOLOv5(image);

    // convert the input image to a blob
    cv::dnn::blobFromImage(inputImage, blob, 1. / 255.0, cv::Size(INPUT_WIDTH, INPUT_HEIGHT), cv::Scalar(), true, false);

    //Feed the input image to the network
    net.setInput(blob);

// Get the output from the network
    std::vector<cv::Mat> outputs;
    net.forward(outputs, net.getUnconnectedOutLayersNames());

    float xFactor = inputImage.cols / INPUT_WIDTH;
    float yFactor = inputImage.rows / INPUT_HEIGHT;

    // This is the output from the network
    float* data = (float*)outputs[0].data;

    // This is the number of classes in the model = 
    const int dimensions = 85;
    const int rows = 25200;

    std::vector<int> classIds;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    for (int i = 0; i < rows; ++i) {

        // The confidence of the detection is the forth element in the array of data - output from the network
        float confidence = data[4];

        if (confidence >= CONFIDENCE_THRESHOLD) {

            // The classes scores are from the 5th element to the 85th element in the array of data - output from the network
            float* classesScores = data + 5;

            cv::Mat scores(1, classList.size(), CV_32FC1, classesScores);
            cv::Point classId;
            double maxClassScore;

            cv::minMaxLoc(scores, 0, &maxClassScore, 0, &classId);


            if (maxClassScore > SCORE_THRESHOLD) {
                confidences.push_back(confidence);
                classIds.push_back(classId.x);


                // Width, height and x,y coordinates of bounding box

                float x = data[0];
                float y = data[1];
                float w = data[2];
                float h = data[3];

                int left = int((x - 0.5 * w) * xFactor);
                int top = int((y - 0.5 * h) * yFactor);
                int width = int(w * xFactor);
                int height = int(h * yFactor);
                boxes.push_back(cv::Rect(left, top, width, height));
            }
        }
        data += 85;
    }

    // Suppress overlapping boxes with non-maximum suppression
    std::vector<int> nmsResult;
    cv::dnn::NMSBoxes(boxes, confidences, SCORE_THRESHOLD, NMS_THRESHOLD, nmsResult);

    // Get the final results
    for (int idx : nmsResult) {
        Detection result;
        result.class_id = classIds[idx];
        result.confidence = confidences[idx];
        result.box = boxes[idx];
        detections.push_back(result);
    }

}
