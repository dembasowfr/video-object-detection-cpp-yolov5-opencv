#ifndef DETECTION_H
#define DETECTION_H

#include <opencv2/opencv.hpp>

struct Detection {
    int class_id;
    // This is the confidence of the detection - how sure the model is that it has detected the object
    float confidence;

    // This is the bounding box of the detected object
    cv::Rect box;
};

#endif // DETECTION_H
