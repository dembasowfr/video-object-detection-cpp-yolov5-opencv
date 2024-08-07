#ifndef IMAGE_H
#define IMAGE_H

#include <opencv2/opencv.hpp>
#include "Detection.h"

class Image {
public:
// this constructor will load the image from the given path
    Image(const std::string& imagePath);

    // this function will return true if the image is empty
    bool isEmpty() const;

    // this function will return the image as a cv::Mat
    cv::Mat& getMat();

    // this function will draw the detections on the image
    void drawDetections(const std::vector<Detection>& detections, const std::vector<std::string>& classNames);

    // this function will save the  the image in a window
    void saveOutputImage(const std::string& windowName, int width) const;

private:
    cv::Mat mat;
};

#endif // IMAGE_H
