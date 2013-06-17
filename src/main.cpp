#include <ctime>

#include <fstream>
#include <iostream>

#include <vector>

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "MosseFilter.h"

#ifdef DATA_DIRECTORY
    const std::string dataDirectory = DATA_DIRECTORY;
#else
    #define DATA_DIRECTORY ""
#endif

const std::string testVideoFileName = DATA_DIRECTORY"short_md.mov";
const std::string reporterFileName = "reporter.txt";

int main(const int argn,const char** args) {
    std::vector<std::string> arguments; for(int i = 1; i < argn; i++) arguments.push_back(std::string(args[i]));

    if(std::find(arguments.begin(),arguments.end(),std::string("test")) != arguments.end()) {
    //createPointTarget(cv::Point(50,50),cv::Size(100,150));

    //std::cout << meanUnit(cv::imread(DATA_DIRECTORY"2.jpg",CV_LOAD_IMAGE_GRAYSCALE)).at<double>(100,100) << std::endl;
    //cv::imshow("Res",meanUnit(cv::imread(DATA_DIRECTORY"2.jpg",CV_LOAD_IMAGE_GRAYSCALE)));
    //cv::waitKey();

    //cv::imshow("Cosine window",createCosineWindow(cv::Size(100,100)));
    //cv::waitKey();

    ac::preprocessImage(cv::imread(DATA_DIRECTORY"2.jpg",CV_LOAD_IMAGE_GRAYSCALE));

    } else if(std::find(arguments.begin(),arguments.end(),std::string("main")) != arguments.end() || argn == 1) {

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

    ac::MosseFilter mosse(cv::Size(train_img.cols,train_img.rows));
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
    } else if(std::find(arguments.begin(),arguments.end(),std::string("measure")) != arguments.end()) {

    const int measure_times = 10;

    clock_t now;
    clock_t nowafter;
    double elapsed_train = 0;
    double elapsed_apply = 0;

    cv::Mat image = cv::imread(DATA_DIRECTORY"3.jpg",CV_LOAD_IMAGE_GRAYSCALE);

    std::ofstream measures("work_time.txt");

    while(image.cols > 15 && image.rows > 15) {
        elapsed_train = 0;
        elapsed_apply = 0;

        measures << image.cols << "x" << image.rows << "\t";

        for(int i = 0; i < measure_times; i++) {
            ac::MosseFilter mosse(cv::Size(image.cols,image.rows));

            now = clock();
            mosse.addTraining(image,cv::Point(image.cols/2,image.rows/2));
            nowafter = clock();
            elapsed_train += double(nowafter - now) / CLOCKS_PER_SEC;

            now = clock();
            mosse.correlate(image);
            nowafter = clock();
            elapsed_apply += double(nowafter - now) / CLOCKS_PER_SEC;
        }

        cv::resize(image,image,cv::Size(image.cols - 10,image.rows - 10));

        measures << elapsed_train/measure_times << "\t" << elapsed_apply/measure_times << std::endl;
    }

    measures.close();
    }
}
