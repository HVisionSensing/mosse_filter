#include <ctime>

#include <fstream>
#include <iostream>

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>

//#include "MosseFilter_old.h"
#include "MosseFilter.h"

#ifdef DATA_DIRECTORY
    const std::string dataDirectory = DATA_DIRECTORY;
#else
    #define DATA_DIRECTORY ""
#endif

const std::string testVideoFileName = DATA_DIRECTORY"short_md.mov";
const std::string reporterFileName = "reporter.txt";

int main() {

    #ifndef MOSSEFILTER_H
    std::ofstream reporter(reporterFileName.c_str(),std::fstream::app);

    CvCapture* capture = cvCaptureFromFile(testVideoFileName.c_str());//cvCaptureFromCAM(0); //

    double fps = cvGetCaptureProperty(capture,CV_CAP_PROP_FPS);
    if(fps == -1)
        fps = 30; // Default webcam FPS

    if(capture == NULL)
        throw "Can't load VideoCapture.\n";

    reporter << "=============================================\n";
    reporter << "\t Current video: " << testVideoFileName << "\n";

    cvNamedWindow("Result");
    cvNamedWindow("Origin");

    IplImage* train_img = cvQueryFrame(capture);
    IplImage* train_img_copy = cvCreateImage(cvSize(train_img->width,train_img->height),train_img->depth,train_img->nChannels);
    cvCopy(train_img,train_img_copy);
    cvShowImage("Result",train_img_copy);

    CvPoint point = cvPoint(train_img->width/2,train_img->height/2);
    MosseFilter mosse(cvSize(train_img->width,train_img->height),21,point);
    mosse.addTraining(train_img_copy,point); // Training of the filter
    mosse.create();

    clock_t now;
    clock_t nowafter;
    double elapsed_preprocess = 0;
    double elapsed_apply = 0;

    IplImage* res;
    IplImage* frame;
    IplImage* frame_copy = cvCreateImage(cvSize(train_img->width,train_img->height),train_img->depth,train_img->nChannels);
    int last_key = -1;

    while(last_key != 32) {
        last_key = cvWaitKey(1000/fps - elapsed_preprocess - elapsed_apply);

        frame = cvQueryFrame(capture);

        if(frame == NULL)
            break;

        cvCopy(frame,frame_copy);

        now = clock();
        fftw_complex* preprocessed = mosse.preprocessImage(frame_copy);
        nowafter = clock();
        elapsed_preprocess = double(nowafter - now) / CLOCKS_PER_SEC;

        now = clock();
        res = mosse.apply(preprocessed);
        nowafter = clock();
        elapsed_apply = double(nowafter - now) / CLOCKS_PER_SEC;

        reporter << "P: " << elapsed_preprocess << " A: " << elapsed_apply << std::endl;
        std::cout << "P: " << elapsed_preprocess << " A: " << elapsed_apply << std::endl;

        cvShowImage("Result",res);
        cvShowImage("Origin",frame);
    }

    reporter.close();

    //cvDestroyWindow("Result");
    //cvReleaseCapture(&capture);

    //delete train_img;
    //delete frame;
    //delete res;
#endif
    //createPointTarget(cv::Point(50,50),cv::Size(100,150));

    //std::cout << meanUnit(cv::imread(DATA_DIRECTORY"2.jpg",CV_LOAD_IMAGE_GRAYSCALE)).at<double>(100,100) << std::endl;
    //cv::imshow("Res",meanUnit(cv::imread(DATA_DIRECTORY"2.jpg",CV_LOAD_IMAGE_GRAYSCALE)));
    //cv::waitKey();

    //cv::imshow("Cosine window",createCosineWindow(cv::Size(100,100)));
    //cv::waitKey();

    //preprocessImage(cv::imread(DATA_DIRECTORY"2.jpg",CV_LOAD_IMAGE_GRAYSCALE));
    //#undef MOSSEFILTER_H

#ifdef MOSSEFILTER_H
    std::ofstream reporter(reporterFileName.c_str(),std::fstream::app);

    //cv::VideoCapture capture(0);
    cv::VideoCapture capture(testVideoFileName.c_str());

    double fps = capture.get(CV_CAP_PROP_FPS);
    if(fps == -1)
        fps = 30; // Default webcam FPS

    if(!capture.isOpened())
        throw "Can't load VideoCapture.\n";

    reporter << "=============================================\n";
    reporter << "\t Current video: " << testVideoFileName << "\n";

    cv::namedWindow("Result");
    cv::namedWindow("Origin");

    cv::Mat train_img;
    capture.read(train_img);

    cv::Point point(train_img.cols/2,train_img.rows/2);

    MosseFilter mosse(cv::Size(train_img.cols,train_img.rows));
    mosse.addTraining(train_img,point); // Training of the filter

    clock_t now;
    clock_t nowafter;
    double elapsed_preprocess = 0;
    double elapsed_apply = 0;

    cv::Mat res;
    cv::Mat frame;
    int last_key = -1;

    while(last_key != 32) {
        last_key = cv::waitKey(1000/fps - elapsed_preprocess - elapsed_apply);

        bool rres = capture.read(frame);

        if(!rres)
            break;

        now = clock();
        res = mosse.correlate(frame);
        nowafter = clock();
        elapsed_preprocess = double(nowafter - now) / CLOCKS_PER_SEC;

        reporter << "P: " << elapsed_preprocess << " A: " << elapsed_apply << std::endl;
        std::cout << "P: " << elapsed_preprocess << " A: " << elapsed_apply << std::endl;

        cv::imshow("Result",res);
        cv::imshow("Origin",frame);
    }

    reporter.close();

#endif
}
