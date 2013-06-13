#include "MosseFilter.h"


namespace ac {

/*
 * Service functions
 */
cv::Mat applyFFT(cv::Mat image) {
    cv::Mat planes[] = {cv::Mat_<float>(image), cv::Mat::zeros(cv::Size(image.cols,image.rows),cv::DataType<float>::type)};

    cv::Mat res;
    cv::merge(planes, 2, res);
    cv::dft(res, res);

    return res;
}

cv::Mat preprocessImage(cv::Mat image) {
    if(image.channels() > 1) // Convert to grayscale image
        cv::cvtColor(image, image, CV_RGB2GRAY);

    if(image.type() != cv::DataType<float>::type)
        image = cv::Mat_<float>(image);

    cv::Mat cosine_window = createCosineWindow(cv::Size(image.cols,image.rows));
    cv::Mat res = meanUnit(image).mul(cosine_window);

    return applyFFT(res);
}

cv::Mat meanUnit(cv::Mat image) {
    cv::Mat mimage = cv::Mat_<float>(image) - cv::mean(image)[0];

    long double sum = 0;
    for(int x = 0; x < image.cols; x++)
        for(int y = 0; y < image.rows; y++) {
            float elem = mimage.at<float>(y,x);
            sum += (elem*elem);
        }

    //cv::imshow("Origin",image);
    //cv::imshow("Mean substracted",mimage);
    //cv::imshow("Powered",tmp);

    //std::cout << (tmp*length).at<double>(100,100) << std::endl;
    //std::cout << tmp.at<double>(100,100) << std::endl;
    //std::cout << mimage.at<double>(100,100) << std::endl;
    //std::cout << "Length" << length << " Sum " << sum << std::endl;

    mimage *= 1/sqrt(sum);

    return mimage;
}

cv::Mat createPointTarget(cv::Point p, cv::Size s, double sigma) {
    cv::Mat xv = arange(s.width) - p.x;
    cv::Mat yv = arange(s.height) - p.y;

    double scale = 1.0/(sigma*sigma);

    cv::Mat nxv;
    cv::Mat nyv;

    cv::pow(xv,2,nxv);
    cv::pow(yv,2,nyv);

    cv::exp(-scale*nxv,nxv);
    cv::exp(-scale*nyv,nyv);

    nyv = nyv.reshape(0,1*s.height);

    //nxv = nxv.reshape(0,s.width);

    //cv::Mat result = nyv*nxv;

    //std::cout << "X " << nxv.cols << " " << nxv.rows << " " << nxv.channels() << std::endl;
    //std::cout << "Y " << nyv.cols << " " << nyv.rows << " " << nyv.channels() << std::endl;

    //cv::imshow("X",nxv);
    //cv::imshow("Y",nyv);
    //cv::imshow("Res",result);

    //cv::waitKey();

    return nyv*nxv; // Works a bit different than python implementation, but result enough similar
}

cv::Mat addComplexPlane(cv::Mat real) {
    cv::Mat planes[] = {cv::Mat_<float>(real),cv::Mat::zeros(cv::Size(real.cols,real.rows),cv::DataType<float>::type)};
    cv::Mat res;

    cv::merge(planes, 2, res);

    return res;
}

cv::Mat createCosineWindow(cv::Size size) {
    cv::Mat cosine_window(size,cv::DataType<float>::type);

    for(int x = 0; x < cosine_window.cols; x++)
        for(int y = 0; y < cosine_window.rows; y++)
            cosine_window.at<float>(y,x) = sin(M_PI*x/cosine_window.cols)*sin(M_PI*y/cosine_window.rows);

    return cosine_window;
}

/*
 * MOSSE Filter implementation
 */
MosseFilter::MosseFilter(cv::Size size, double reg, double sigma, functionROI* roi) {
    size_ = size;

    reg_ = reg;
    sigma_ = sigma;

    Roi_ = roi;

    N_ = addComplexPlane(cv::Mat::zeros(size,cv::DataType<float>::type));
    D_ = addComplexPlane(cv::Mat::zeros(size,cv::DataType<float>::type));
}


void MosseFilter::addTraining(cv::Mat image, cv::Point p) {
    cv::Mat g = createPointTarget(p,size_,sigma_);

    cv::Mat F = preprocessImage(image);

    cv::Mat G = applyFFT(g);

    if(filter_.empty())
        filter_ = addComplexPlane(cv::Mat::zeros(size_,cv::DataType<float>::type));

    cv::Mat conjF = F;

    //conjF[1] = -conjF[1];
    for(int x = 0; x < conjF.cols; x++)
        for(int y = 0; y < conjF.rows; y++)
            conjF.at<float[2]>(y,x)[1] = -conjF.at<float[2]>(y,x)[1];

    N_ += G.mul(conjF);
    D_ += F.mul(conjF) + 100;

    filter_ = N_/D_;
}

cv::Mat MosseFilter::correlate(cv::Mat image) {
    cv::Mat preprocessed = preprocessImage(image);

    //cv::Mat G = filter_.mul(preprocessed); // Doesn't works :(
    cv::Mat G = addComplexPlane(cv::Mat(filter_.rows,filter_.cols,cv::DataType<float>::type));
    for(int x = 0; x < filter_.cols; x++)
        for(int y = 0; y < filter_.rows; y++) {
            G.at<float[2]>(y,x)[0] = filter_.at<float[2]>(y,x)[0]*preprocessed.at<float[2]>(y,x)[0];
            G.at<float[2]>(y,x)[1] = filter_.at<float[2]>(y,x)[0]*preprocessed.at<float[2]>(y,x)[1];
        }

    cv::Mat g;
    cv::idft(G,g);

    cv::Mat splitted[2];
    cv::split(g,splitted);

    double min, max;
    cv::Point minp, maxp;
    cv::minMaxLoc(splitted[0],&min,&max,&minp,&maxp);

    cv::Mat res(splitted[0].rows,splitted[0].cols,cv::DataType<unsigned char>::type);

    for(int x = 0; x < splitted[0].cols; x++)
        for(int y = 0; y < splitted[0].rows; y++)
            res.at<unsigned char>(y,x) = (abs(min) + splitted[0].at<float>(y,x))*255/(max+abs(min));

    cv::circle(res,maxp,20,cv::Scalar(255,255,255));

    return res;
    //return splitted[0];
}

} // End of namespace
