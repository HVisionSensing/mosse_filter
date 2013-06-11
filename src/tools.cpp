#import "tools.hpp"


cv::Mat arange(int length) {
    cv::Mat res(1,length,CV_64F);

    for(int x = 0; x <= length; x++)
        res.at<double>(x) = x;

    return res;
}
