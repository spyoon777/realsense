// Copyright(c) 2022 KOREA ELECTRONICS TECHNOLOGY INSTITUTE(KETI), All Rights Reserved.
// Author: spyoon777

#include <iostream>
#include <winsock2.h>
#include <windows.h>

#include "librealsense2/rs.hpp"
// #include "./cv-helpers.hpp"

#include "opencv2/opencv.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/objdetect/objdetect.hpp"
 
#pragma comment (lib,"ws2_32.lib")
#define BUFFER_SIZE 1024

// cv::CascadeClassifier face_cascade;
// cv::CascadeClassifier eye_cascade;

/**
 * Function to detect human face and the eyes from an image.
 *
 * @param  im    The source image
 * @param  tpl   Will be filled with the eye template, if detection success.
 * @param  rect  Will be filled with the bounding box of the eye
 * @return zero=failed, nonzero=success
 */
// int detectEye(cv::Mat& im, cv::Mat& tpl, cv::Rect& rect)
// {
//     std::vector<cv::Rect> faces, eyes;
//     face_cascade.detectMultiScale(im, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, cv::Size(30,30));

//     for (int i = 0; i < faces.size(); i++)
//     {
//         cv::Mat face = im(faces[i]);
//         eye_cascade.detectMultiScale(face, eyes, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, cv::Size(20,20));

//         if (eyes.size())
//         {
//             rect = eyes[0] + cv::Point(faces[i].x, faces[i].y);
//             tpl  = im(rect);
//         }
//     }

//     return eyes.size();
// }

/**
 * Perform template matching to search the user's eye in the given image.
 *
 * @param   im    The source image
 * @param   tpl   The eye template
 * @param   rect  The eye bounding box, will be updated with the new location of the eye
 */
// void trackEye(cv::Mat& im, cv::Mat& tpl, cv::Rect& rect)
// {
//     cv::Size size(rect.width * 2, rect.height * 2);
//     cv::Rect window(rect + size - cv::Point(size.width/2, size.height/2));

//     window &= cv::Rect(0, 0, im.cols, im.rows);

//     cv::Mat dst(window.width - tpl.rows + 1, window.height - tpl.cols + 1, CV_32FC1);
//     cv::matchTemplate(im(window), tpl, dst, CV_TM_SQDIFF_NORMED);

//     double minval, maxval;
//     cv::Point minloc, maxloc;
//     cv::minMaxLoc(dst, &minval, &maxval, &minloc, &maxloc);

//     if (minval <= 0.2)
//     {
//         rect.x = window.x + minloc.x;
//         rect.y = window.y + minloc.y;
//     }
//     else
//         rect.x = rect.y = rect.width = rect.height = 0;
// }

// int main() {
    /*
    // Create a Pipeline - this serves as a top-level API for streaming and processing frames
    rs2::pipeline p;

    // Configure and start the pipeline
    p.start();

    while (true) {
        // Block program until frames arrive
        rs2::frameset frames = p.wait_for_frames();

        // Try to get a frame of a depth image
        rs2::depth_frame depth = frames.get_depth_frame();

        // Get the depth frame's dimensions
        auto width = depth.get_width();
        auto height = depth.get_height();

        // Query the distance from the camera to the object in the center of the image
        float dist_to_center = depth.get_distance(width / 2, height / 2);

        // Print the distance
        std::cout << "The camera is facing an object " << dist_to_center << " meters away \r";
    }
*/
/*
    // Declare depth colorizer for pretty visualization of depth data
    rs2::colorizer color_map;

    // Declare RealSense pipeline, encapsulating the actual device and sensors
    rs2::pipeline pipe;
    // Start streaming with default recommended configuration
    pipe.start();

    const auto window_name = "Display Image";
    cv::namedWindow(window_name, cv::WindowFlags::WINDOW_AUTOSIZE);

    while (cv::waitKey(1) < 0 && cv::getWindowProperty(window_name, cv::WindowPropertyFlags::WND_PROP_AUTOSIZE) >= 0) {
        rs2::frameset data = pipe.wait_for_frames();  // Wait for next set of frames from the camera
        rs2::frame depth = data.get_depth_frame().apply_filter(color_map);
        rs2::frame color = data.get_color_frame();

        // Query frame size (width and height)
        const int w = depth.as<rs2::video_frame>().get_width();
        const int h = depth.as<rs2::video_frame>().get_height();

        auto color_mat = frame_to_mat(color);

        // Create OpenCV matrix of size (w,h) from the colorized depth data
        cv::Mat image(cv::Size(w, h), CV_8UC3, (void*)depth.get_data(), cv::Mat::AUTO_STEP);

        // Update the window with new data
        // cv::imshow(window_name, image);
        cv::imshow(window_name, color_mat);
    }
*/
/*
    // Load the cascade classifiers
    // Make sure you point the XML files to the right path, or 
    // just copy the files from [OPENCV_DIR]/data/haarcascades directory
    face_cascade.load("haarcascade_frontalface_alt2.xml");
    eye_cascade.load("haarcascade_eye.xml");

    // Open webcam
    cv::VideoCapture cap(1);

    // Check if everything is ok
    if (face_cascade.empty() || eye_cascade.empty() || !cap.isOpened())
        return 1;

    // Set video to 320x240
    cap.set(CV_CAP_PROP_FRAME_WIDTH, 320);
    cap.set(CV_CAP_PROP_FRAME_HEIGHT, 240);

    cv::Mat frame, eye_tpl;
    cv::Rect eye_bb;

    while (cv::waitKey(15) != 'q')
    {
        cap >> frame;
        if (frame.empty())
            break;

        // Flip the frame horizontally, Windows users might need this
        cv::flip(frame, frame, 1);

        // Convert to grayscale and 
        // adjust the image contrast using histogram equalization
        cv::Mat gray;
        cv::cvtColor(frame, gray, CV_BGR2GRAY);

        if (eye_bb.width == 0 && eye_bb.height == 0)
        {
            // Detection stage
            // Try to detect the face and the eye of the user
            detectEye(gray, eye_tpl, eye_bb);
        }
        else
        {
            // Tracking stage with template matching
            trackEye(gray, eye_tpl, eye_bb);

            // Draw bounding rectangle for the eye
            cv::rectangle(frame, eye_bb, CV_RGB(0,255,0));
        }

        // Display video
        cv::imshow("video", frame);
    }
*/

//     return 0;
// }

int main() {
    printf("helloworld");

    std::string cmd = "start;C:/Users/keti-007/Documents/source/python/iris/rs2iris/dist/rs2iris.exe";
    system(cmd.c_str());
    system("PAUSE");

    WSADATA wsaData;
    SOCKET ClientSocket;
    SOCKADDR_IN ToServer;

    int Send_Size;
    ULONG ServerPort = 9000;

    char Buffer[BUFFER_SIZE] = {};
    sprintf_s(Buffer, "stop");
    if (WSAStartup(0x202, &wsaData) == SOCKET_ERROR) {
        std::cout << "WinSock error" << std::endl;
        WSACleanup();
        exit(0);
    }

    memset(&ToServer, 0, sizeof(ToServer));

    ToServer.sin_family = AF_INET;
    ToServer.sin_addr.s_addr = inet_addr("127.0.0.1");
    ToServer.sin_port = htons(ServerPort);

    ClientSocket = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);

    if (ClientSocket == INVALID_SOCKET) {
        std::cout << "error" << std::endl;
        closesocket(ClientSocket);
        WSACleanup();
        exit(0);
    }

    sendto(ClientSocket, Buffer, BUFFER_SIZE, 0, (struct sockaddr*) &ToServer, sizeof(ToServer));

    closesocket(ClientSocket);
    WSACleanup();

    return 0;
}
