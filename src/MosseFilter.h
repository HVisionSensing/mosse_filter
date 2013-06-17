#ifndef MOSSEFILTER_H
#define MOSSEFILTER_H

#include <cmath>

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "tools.hpp"

namespace ac {

typedef cv::Mat (*functionROI)(cv::Point p, cv::Size size, double sigma); // = 100.0);

class MosseFilter {
public:
    MosseFilter(cv::Size size, double reg = 100, double sigma = 2.0, functionROI* roi = NULL);

    void addTraining(cv::Mat image, cv::Point p);

    cv::Mat correlate(cv::Mat image);

private:
    cv::Mat N_;
    cv::Mat D_;

    cv::Mat filter_;

    cv::Size size_;

    double reg_;
    double sigma_;

    functionROI* Roi_;
};

cv::Mat createCosineWindow(cv::Size size);
cv::Mat preprocessImage(cv::Mat image);
cv::Mat applyFFT(cv::Mat image);
cv::Mat meanUnit(cv::Mat image);
cv::Mat resizeWin(cv::Mat image);

cv::Mat createPointTarget(cv::Point p,cv::Size s, double sigma = 100.0);

cv::Mat addComplexPlane(cv::Mat real);

cv::Mat mulC(cv::Mat& first, cv::Mat& second);
cv::Mat divC(cv::Mat& first, cv::Mat& second);

} // End of namespace

#endif // MOSSEFILTER_H
