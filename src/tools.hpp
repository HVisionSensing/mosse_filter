#ifndef TOOLS_HPP
#define TOOLS_HPP

#include <opencv2/opencv.hpp>

cv::Mat arange(int length);

cv::Mat multPW(cv::Mat& first, cv::Mat& second); // Point-wise multiply of two matrix

#endif // TOOLS_HPP
