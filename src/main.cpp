#include <iostream>
#include "Detector.h"
#include "Utils.h"


//Color definitons and Constant assigments

const std::vector<cv::Scalar> colors = { cv::Scalar(255, 255, 0), cv::Scalar(0, 255, 0), cv::Scalar(0, 255, 255), cv::Scalar(255, 0, 0) };


int main(int argc, char** argv) {


    // Check if the user wants to use CUDA
    bool useCUDA = argc > 1 && strcmp(argv[1], "cuda") == 0;

    std::string videoPath = "../data/input/video.mp4";
    

    cv::Mat frame;
    cv::VideoCapture capture(videoPath);

    if (!capture.isOpened())
    {
        std::cerr << "Error opening the video file!!!" << std::endl;
        return -1;
    }
    


    std::string modelPath = "../models/yolov5s.onnx";
    Detector detector(modelPath, useCUDA);
    std::vector<Detection> detections;


    std::string classesPath = "../data/input/classes.txt";
    auto classList = loadClassList(classesPath);


    // Check the frame count and the frame rate of the video
    // Start the timer
    auto start = std::chrono::high_resolution_clock::now();
    int frame_count = 0;
    float fps = -1;
    int total_frames = 0;

    // Calculate the new dimensions while preserving the aspect ratio
    int targetWidth = 800; // Adjust the desired width

    while (true)
    {
        capture.read(frame);
        if (frame.empty())
        {
            std::cout << "Media finished\n";
            break;
        }

        std::vector<Detection> output;
        detector.detect(frame, output, classList);

        frame_count++;
        total_frames++;

        int detections = output.size();

        for (int i = 0; i < detections; ++i)
        {
            auto detection = output[i];
            auto box = detection.box;
            auto classId = detection.class_id;
            const auto color = colors[classId % colors.size()];
            cv::rectangle(frame, box, color, 3);

            cv::rectangle(frame, cv::Point(box.x, box.y - 20), cv::Point(box.x + box.width, box.y), color, cv::FILLED);
            cv::putText(frame, classList[classId].c_str(), cv::Point(box.x, box.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
        }

        if (frame_count >= 30)
        {
            auto end = std::chrono::high_resolution_clock::now();
            fps = frame_count * 1000.0 / std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

            frame_count = 0;
            start = std::chrono::high_resolution_clock::now();
        }

        if (fps > 0)
        {
            std::ostringstream fps_label;
            fps_label << std::fixed << std::setprecision(2);
            fps_label << "FPS: " << fps;
            std::string fps_label_str = fps_label.str();

            cv::putText(frame, fps_label_str.c_str(), cv::Point(10, 25), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);
        }

        // Resize the frame with it's new shape as it's original ratio
        int targetHeight = static_cast<int>(frame.rows * static_cast<float>(targetWidth) / frame.cols);
        cv::Size newSize(targetWidth, targetHeight);
        cv::Mat resized_frame;
        cv::resize(frame, resized_frame, newSize);

        //cv::imshow("output", resized_frame);
        // Get the date and time of the frame to add it as the frame name
        std::time_t t = std::time(0);
        std::tm* now = std::localtime(&t);  
        std::ostringstream oss; 
        oss << "../data/output/" << now->tm_year + 1900 << "-" << now->tm_mon + 1 << "-" << now->tm_mday << "_" << now->tm_hour << "-" << now->tm_min << "-" << now->tm_sec << ".jpg";
        std::string frame_name = oss.str(); 
        // Save the output frame as a file  
        cv::imwrite(frame_name, resized_frame);

        // Save the output video as a file
        /*cv::VideoWriter video("../data/output/output.mp4", cv::VideoWriter::fourcc('m', 'p', '4', 'v'), 30, newSize);
        video.write(resized_frame);*/


        int key = cv::waitKey(1);
        if (key == 27) // ESC to exit
        {
            break;
        }
    }
    


    return 0;
}
