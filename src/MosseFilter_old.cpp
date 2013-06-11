// Filter.cpp
// This file contains the implementation of class Filter. The Filter class is the
// workhorse class that does all of the operations that are common to the set of all
// filters we generate. For each location of interest we have a corresponding filter.
// Extensions of this class create filters for each LOI. 

#include "MosseFilter_old.h"

// static member initialization
int MosseFilter::nComplexVectors = 8;

// Class construction and destruction

MosseFilter::MosseFilter(CvSize size, double spread, CvPoint& center)//, roiFnT roiFn) :
  : gaussianSpread(spread) {
  // check if the output directory exists, or else bail

  imgSize.height = size.height;
  imgSize.width = size.width;
  filter = 0;
  doAffineTransforms = false;

  postFilterImg = cvCreateImage(imgSize, IPL_DEPTH_64F, 1);

  // allocate numerator and denominator arrays as we are in filter creation
  // mode and initialize arrays to zero
  length = imgSize.height * imgSize.width;
  int nElements = imgSize.height * ((imgSize.width / 2) + 1);
  mosseNum = (fftw_complex*)malloc(sizeof(fftw_complex) * nElements);
  mosseDen = (fftw_complex*)malloc(sizeof(fftw_complex) * nElements);
  for (int i = 0; i < nElements; i++) {
    mosseNum[i][0] = mosseNum[i][1] = mosseDen[i][0] = mosseDen[i][1] = 0;
  }

  // allocate real complex vectors for use during filter creation or update
  realImg = cvCreateImage(imgSize, IPL_DEPTH_64F, 1);
  tempImg = cvCreateImage(imgSize, IPL_DEPTH_64F, 1);
  imageBuffer = (double*)fftw_malloc(sizeof(double) * length);

  windowCenter.x = center.x;
  windowCenter.y = center.y;

  // Now compute a window around the center 
  double xSpread = imgSize.width * Globals::windowXScale;
  double ySpread = imgSize.height * Globals::windowYScale;
  window = createWindow(windowCenter, xSpread, ySpread);

  storageIndex = 0;
  for (int i = 0; i < nComplexVectors; i++) {
    fftw_complex* buffer = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * nElements);
    complexVectors.push_back(buffer);
  }

  // create a buffer to perform forward and backward FFT. We use openmp to 
  // optimize the use of these filters. While fftw_execute is thread 
  // safe, the plan functions are not. We therefore create a buffer oand
  // plans for both direction apriori and use them in all runtime 
  // fftw_execute calls

  fftBuffer = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * nElements);

  // compute plans
  planForward = fftw_plan_dft_r2c_2d(imgSize.height, imgSize.width,
				     imageBuffer, fftBuffer,
				     FFTW_ESTIMATE);
  if (!planForward) {
    string err = "Filter::Filter. Error computing forward plan.";
    throw (err);
  }
  planBackward = fftw_plan_dft_c2r_2d(imgSize.height, imgSize.width,
				      fftBuffer, imageBuffer,
				      FFTW_ESTIMATE);
  if (!planBackward) {
    string err = "Filter::Filter. Error computing backward plan.";
    throw (err);
  }
}

MosseFilter::~MosseFilter() {
  if (filter)
    fftw_free(filter);
  if (mosseNum)
    fftw_free(mosseNum);
  if (mosseDen)
    fftw_free(mosseDen);

  cvReleaseImage(&realImg);
  cvReleaseImage(&tempImg);
  cvReleaseImage(&postFilterImg);
  fftw_free(imageBuffer);
  fftw_free(fftBuffer);
  fftw_free(window);

  for (int i = 0; i < nComplexVectors; i++)
    fftw_free(complexVectors[i]);
  
  fftw_destroy_plan(planForward);
  fftw_destroy_plan(planBackward);  
}

// setWindowCenter
// Method used to set the window center. This will re-compute the window
// function for all subsequent windowing

void MosseFilter::setWindowCenter(CvPoint& center) {
  windowCenter.x = center.x;
  windowCenter.y = center.y;

  // free existing window. One always exists as it is created in the
  // constructor
  fftw_free(window);

  // Now compute a window around the center 
  double xSpread = imgSize.width * Globals::windowXScale;
  double ySpread = imgSize.height * Globals::windowYScale;
  window = createWindow(windowCenter, xSpread, ySpread);
}

// addTrainingSet
void MosseFilter::addTraining(IplImage* image, CvPoint location) {
    //TODO redefine this shit after BIG BANG
    //CvPoint location = cvPoint(10,10);
    // Shit ends here

    if (!image) {
      string err = "Filter::update. u ARE faggot";
      throw (err);
    }

    // generate affine transforms if requested
    vector<ImgLocPairT>& imgLocPairs = getAffineTransforms(image, location);

    for (unsigned int i = 0; i < imgLocPairs.size(); i++) {
      image = imgLocPairs[i].first;
      location = imgLocPairs[i].second;

      CvPoint offset;
      offset.x = offset.y = 0;
#if 0
      if (roiFunction) {
        IplImage* roi = roiFunction(image, *fa, offset, Annotations::Face);
        image = roi;
      }
#endif

      location.x -= offset.x;
      location.y -= offset.y;

      // compute size and length of the image data
      CvSize size = cvGetSize(image);

      // check consistency
      if (imgSize.height != size.height || imgSize.width != size.width) {
        char buffer[32];
        sprintf(buffer, "(%d, %d).", imgSize.height, imgSize.width);
        string err = "Filter::update. Inconsistent image sizes. Expecting" + string(buffer);
        throw (err);
      }

      // preprocess
      fftw_complex* preImage = preprocessImage(image);
      fftw_complex* fftImage = getBuffer();
      int nElements = imgSize.height * ((imgSize.width / 2) + 1);
      memcpy(fftImage, preImage, (sizeof(fftw_complex) * nElements));

      // create a gaussian around the location centered at the location
      CvSize sizeOfGaussian = {gaussianSpread, gaussianSpread}; // size and sd
      double sd = gaussianSpread / 2;
      fftw_complex* gaussian = createGaussian(location, sizeOfGaussian, sd);

      // find the complex conjugate of fftImage
      fftw_complex* fftImageC = getBuffer();
      for (int i = 0; i < nElements; i++) {
        fftImageC[i][0] = fftImage[i][0];
        fftImageC[i][1] = -fftImage[i][1];
      }

      // Compute numerator and denominator terms for the filter update
      fftw_complex* num = convolve(gaussian, fftImageC);
      fftw_complex* den = convolve(fftImage, fftImageC);

      // accumulate into filter wide numerator and denominator terms
      for(int i = 0; i < imgSize.height; i++) {
        for(int j = 0; j < imgSize.width / 2 + 1; j++) {
      int ij = i * (imgSize.width / 2 + 1) + j;
      mosseNum[ij][0] += num[ij][0];
      mosseNum[ij][1] += num[ij][1];

      mosseDen[ij][0] += den[ij][0];
      mosseDen[ij][1] += den[ij][1];
        }
      }

      //if (roiFunction)
      //  cvReleaseImage(&image);
    }

    destroyAffineTransforms(imgLocPairs);
}


// create
// Method that computes the filter using the numerator and denominator
// terms that get computed using test data. This method is expected to be
// called after all the test image directories have been added using
// addTrainingSet.

void MosseFilter::create() {
  // Allocate filter
  int nElements = imgSize.height * ((imgSize.width / 2) + 1);
  if (!filter)
    filter = (fftw_complex*)malloc(sizeof(fftw_complex) * nElements);

  for(int i = 0; i < imgSize.height; i++) {
    for(int j = 0; j < imgSize.width / 2 + 1; j++) {
      int ij = i * (imgSize.width / 2 + 1) + j;

      // compute denominator
      double c = mosseDen[ij][0];
      double d = mosseDen[ij][1];
      double denom = (c * c) + (d * d);

      // compute real part of numerator
      double a = mosseNum[ij][0];
      double b = mosseNum[ij][1];
      double re = a * c + b * d;
      double im = b * c - a * d;

      // now assign
      filter[ij][0] = re / denom;
      filter[ij][1] = im / denom;
    }
  }
}

// boostFilter
// Method used to apply a high-boost filter to the image during preprocessing

void MosseFilter::boostFilter(IplImage* src, IplImage* dest) {
}

// preprocessImage
// Method that preprocesses an image that has already been loaded for a
// subsequent application of the filter. The method returns a preprocessed
// image which can then be used on a subsequent call to apply or update.

fftw_complex* MosseFilter::preprocessImage(IplImage* inputImg) {
  // we take the complex image and preprocess it here
  if (!inputImg) {
    string err = "Filter::preprocessImage. Call setImage with a valid image.";
    throw (err);
  }

  bool releaseImage = false;
  IplImage* image = 0;

  // First check if the image is in grayscale. If not, we first convert it
  // into grayscale. The input image is replaced with its grayscale version
  if ((inputImg->nChannels != 1 && strcmp(inputImg->colorModel, "GRAY"))) {
    image = cvCreateImage(imgSize, IPL_DEPTH_8U, 1);
    cvCvtColor(inputImg, image, CV_BGR2GRAY);
    releaseImage = true;
  } else
    image = inputImg;


  // now do histogram equalization
  cvEqualizeHist(image, image);

  // edge detection
  //  cvCanny(image, image, 10, 100, 3);

  // We follow preprocessing steps here as outlined in Bolme 2009
  // First populate a real image from the grayscale image
  double scale = 1.0 / 255.0;
  cvConvertScale(image, realImg, scale, 0.0);


  int step = realImg->widthStep;
  double* imageData = (double*)realImg->imageData;
  for (int i = 0; i < imgSize.height; i++) {
    for (int j = 0; j < imgSize.width; j++) {
      *(imageData) = 1 - *(imageData);
      imageData++;
    }
    imageData += step / sizeof(double) - imgSize.width;
  }

  // suppress DC
  cvDCT(realImg, tempImg, CV_DXT_FORWARD);
  step = tempImg->widthStep;
  imageData = (double*)tempImg->imageData;
  for (int i = 0; i < imgSize.height; i++) {
    for (int j = 0; j < imgSize.width; j++) {
      double sigmoid = (1 / (1 + (exp(-(i * imgSize.width + j)))));
      *(imageData) = *(imageData) * sigmoid;
      imageData++;
    }
    imageData += step / sizeof(double) - imgSize.width;
  }
  cvSet2D(tempImg, 0, 0, cvScalar(0));
  cvDCT(tempImg, realImg, CV_DXT_INVERSE);

  // Add a scalar 1.0 to all elements of the image
  cvAddS(realImg, cvScalar(1.0), realImg, NULL);

  // Take log
  cvLog(realImg, realImg);

  // Compute the mean of all the elements in the image matrix
  CvScalar mean = cvAvg(realImg, NULL);

  // Subtract the mean from all elements of the image
  cvSubS(realImg, mean, realImg);

  // Compute the sum of the squares of all the elements in the image
  cvMul(realImg, realImg, tempImg);
  CvScalar sumOfSquares = cvSum(tempImg);


  // Now divide each element by the sum of squares
  scale = 1.0 / sumOfSquares.val[0];
  cvConvertScale(realImg, realImg, scale, 0.0);

  // scale image to be in the range [0, 1]
  double min, max;
  cvMinMaxLoc(realImg, &min, &max, NULL, NULL);
  if (min < 0)
    cvAddS(realImg, cvScalar(-min), realImg, NULL);
  else
    cvAddS(realImg, cvScalar(min), realImg, NULL);

  cvMinMaxLoc(realImg, &min, &max, NULL, NULL);
  scale = 1.0 / max;
  cvConvertScale(realImg, realImg, scale, 0);

  // Apply the window
  applyWindow(realImg, window, imageBuffer);

  //  showImage((const char*)(filterName(xmlTag) + "__").c_str(), realImg);
  //  showRealImage((const char*)filterName(xmlTag).c_str(), imageBuffer);

  // Now compute the fft of the image
  fftw_complex* fft = computeFFT();

  if (releaseImage)
    cvReleaseImage(&image);

  return fft;
}

// apply
// This method is used to apply a filter passed as input to an image
// The filter is a complex array. The image FFT is also a complex array. The
// method takes the filter and the image FFT as inputs and takes their element-wise
// product. The steps taken are,
// a. Take element-wise complex product of the fft and filter
// b. Compute inverse FFT to transform data from complex to real domain
// c. Compose an image object using the IFFT data and return it

IplImage* MosseFilter::apply(fftw_complex* fft) {
  // first check if we have done the preprocessing step
  if (!fft) {
    string err = "Filter::apply. fft object is NULL.";
    throw (err);
  }
  // check if we have a valid filter
  if (!filter) {
    string err = "Filter::apply. Invalid filter (NULL).";
    throw (err);
  }

  // now take product of the fft data and the filter data
  convolve(fft, filter, fftBuffer /* result will be stored here */);

  // now get the inverse FFT and return
  IplImage* postFilterImg = computeInvFFT();
  
  double min;
  double max;

  //cvMinMaxLoc(postFilterImg, &min, &max, NULL, NULL);
  //double scale = 1.0 / max;
  //cvConvertScale(postFilterImg, postFilterImg, scale, 0.0);
  
  return postFilterImg;
}

IplImage* MosseFilter::correlate(IplImage* image) {
    return correlate(preprocessImage(image));
}

IplImage* MosseFilter::correlate(fftw_complex* imageFFT) {

}

// computeInvFFT
// This method computes an inverse FFT. The input is always expected to
// be in imageBuffer (for performance using openmp). The result will be
// This method takes as input a complex array and performs an inverse FFT
// on it. The result is then returned as an IplImage object

IplImage* MosseFilter::computeInvFFT() {
  // compute inverse FFT
  fftw_execute(planBackward);
  
  // create a new image and copy inverse data as image data
  int step = postFilterImg->widthStep;
  double* imageData = (double*)postFilterImg->imageData;
  for (int i = 0; i < imgSize.height; i++) {
    for (int j = 0; j < imgSize.width; j++) {
      *(imageData++) = imageBuffer[i * imgSize.width + j];
    }
    imageData += step / sizeof(double) - imgSize.width;
  }
  
  return postFilterImg;    
}

// The following section has implementations of the helper methods

// convolve
// This method is used to convolve two arrays. It takes as input two arrays in the
// complex domain and returns their element-wise complex product

fftw_complex* MosseFilter::convolve(fftw_complex* one, fftw_complex* two, fftw_complex* outputBuffer) {
  fftw_complex* result = (outputBuffer)? outputBuffer : getBuffer();

  for(int i = 0; i < imgSize.height; i++)
    for(int j = 0; j < imgSize.width / 2 + 1; j++) {
      int ij = i * (imgSize.width / 2 + 1) + j;
      result[ij][0] = ((one[ij][0] * two[ij][0]) - (one[ij][1] * two[ij][1]));
      result[ij][1] = ((one[ij][0] * two[ij][1]) + (one[ij][1] * two[ij][0]));
    }     
  return result;
}

// computeFFT
// This method takes as input an image and computes the FFT of the image data
// and returns a pointer to a complex array with the results

fftw_complex* MosseFilter::computeFFT(IplImage* image) {
  // copy image data to the fft input, which is always expected to
  // be in imageBuffer
  int step = image->widthStep;
  double* imageData = (double*)image->imageData;
  for (int i = 0, k = 0; i < imgSize.height; i++) {
    for (int j = 0; j < imgSize.width; j++) {
      imageBuffer[k] = *(imageData++);
      k++;
    }
    imageData += step / sizeof(double) - imgSize.width;
  }

  // now apply FFT
  fftw_complex* fft = computeFFT();

  return fft;
}

// computeFFT
// This method computes the forward FFT. The method expects that the input data
// has already been stored in imageBuffer. The result will be placed in fftBuffer.
// Since we use openmp to parallelize the application of filters, we create
// plans for forward and backward FFT apriori and simply execute them at
// runtime for each filter application

fftw_complex* MosseFilter::computeFFT() {
  // now apply FFT
  fftw_execute(planForward);

  return fftBuffer;
}

/* Private methods */

// createGaussian
// Method to create a gaussian field of a given size at a given location on an
// image plane that is of the same size as the training images

fftw_complex* MosseFilter::createGaussian(CvPoint& location, CvSize& size, double sd) {
  // Linear space vector. Create a meshgrid with x and y axes values
  // that stradle the location

  double xspacer[imgSize.width];
  double yspacer[imgSize.height];

  int lx = location.x;
  double left = -lx;
  for (int i = 0; i < imgSize.width; i++) {
    xspacer[i] = left;
    left += 1.0;
  }
  int ly = location.y;
  double top = -ly;
  for (int i = 0; i < imgSize.height; i++) {
    yspacer[i] = top;
    top += 1.0;
  }

  // Mesh grid
  //double x[imgSize.height][imgSize.width];
  //double y[imgSize.height][imgSize.width];
  double** x = (double**) malloc(sizeof(double)*imgSize.height);
  double** y = (double**) malloc(sizeof(double)*imgSize.height);
  for(int i = 0; i < imgSize.height; i++) {
      x[i] = (double*) malloc(sizeof(double)*imgSize.width);
      y[i] = (double*) malloc(sizeof(double)*imgSize.width);
  }

  for (int i = 0; i < imgSize.height; i++) {
    for (int j = 0; j < imgSize.width; j++) {
      x[i][j] = xspacer[j];
      y[i][j] = yspacer[i];
    }
  }

  // create a gaussian as big as the image
  double gaussian[imgSize.height][imgSize.width];

  double det = sd * sd;
  for (int i = 0; i < imgSize.height; i++) {
    for (int j = 0; j < imgSize.width; j++) {
      // using just the gaussian kernel
      double X = x[i][j] * x[i][j];
      double Y = y[i][j] * y[i][j];
      gaussian[i][j] = exp(-((X * sd + Y * sd) / det));
    }
  }

  // now initialize a real array as large as the image array with the values
  // of the gaussian
  for (int i = 0; i < length; i++)
    imageBuffer[i] = 0;
  for (int i = 0; i < imgSize.height; i++) {
    for (int j = 0; j < imgSize.width; j++) {
      imageBuffer[i * imgSize.width + j] = gaussian[i][j];
    }
  }

  for(int i = 0; i < imgSize.height; i++) {
      delete x[i];
      delete y[i];
  }
  delete x;
  delete y;

  //  showRealImage("__gaussian", imageBuffer);

  fftw_complex* fft = computeFFT();
  return fft;
}

// applyWindow
// Method used to apply a window function to a real image. It takes as input
// the real image source and a window as a 2d real array. The result is stored in
// the third parameter, the source and the destination can be the same. The step
// is expected to match in the src and dest images

void MosseFilter::applyWindow(IplImage* src, double* window, double* dest) {
  int step = src->widthStep;

  double* destImageData = dest;
  double* srcImageData = (double*)src->imageData;
  for (int i = 0; i < imgSize.height; i++) {
    for (int j = 0; j < imgSize.width; j++) {
      (*destImageData) = (*srcImageData) * window[i * imgSize.width + j];
      srcImageData++; destImageData++;
    }
    srcImageData += step / sizeof(double) - imgSize.width;
  }
}

// applyWindow
// Method used to apply a window function to a real image. It takes as input
// the real image source and a window as a 2d real array. The result is stored in
// the third parameter, the source and the destination can be the same. The step
// is expected to match in the src and dest images

void MosseFilter::applyWindow(IplImage* src, double* window, IplImage* dest) {
  int step = src->widthStep;

  double* srcImageData = (double*)src->imageData;
  double* destImageData = (double*)dest->imageData;

  for (int i = 0; i < imgSize.height; i++) {
    for (int j = 0; j < imgSize.width; j++) {
      (*destImageData) = (*srcImageData) * window[i * imgSize.width + j];
      srcImageData++; destImageData++;
    }
    srcImageData += step / sizeof(double) - imgSize.width;
    destImageData += step / sizeof(double) - imgSize.width;
  }
}

// createCosine
// Method to create a cosine field around a given location to drop the value of 
// the pixels all around the area of interest in the image. The size of the 
// field is the same as the filter size. We create a cosine window of equal dimensions
// with a half width that min(x, imgSize.width -x, y, imgSize.height - y). All pixel
// values outside this window are uniformly zero.
// This way, the cosine window isolates the locations of interest and pixel values 
// drop to zero close to the edges. If the location is given to be the center of the
// image then this is an unbiased cosine window, otherwise the window is biased to
// the location of interest

double* MosseFilter::createCosine(CvPoint& location) {
  // Linear space vector. We expect that the size parameter has equal dimensions.
  // We are creating a square meshgrid

  // first compute the largest square around location to place the cosine
  int width = imgSize.width;
  double spacer[width];

  int l = width / 2;
  double left = -l;

  for (int i = 0; i < width; i++) {
    spacer[i] = left;
    left += 1.0;
  }

  // Mesh grid
  double x[width][width];
  double y[width][width];

  for (int i = 0; i < width; i++) {
    for (int j = 0; j < width; j++) {
      x[i][j] = spacer[j];
      y[i][j] = spacer[i];
    }
  }

  // Compute cosine using the distance to radius ratio of the
  // points in the mesh grid. What we compute here is similar to
  // the hanning function

  double cosine[width][width];
  double radius = width * width / 2.0;

  for (int i = 0; i < width; i++)
    for (int j = 0; j < width; j++) {
      double d = (x[i][j] * x[i][j] + y[i][j] * y[i][j]);
      cosine[i][j] = 1.0 - sin((M_PI / 2) * (d / radius));
    }

  double* window = (double*)fftw_malloc(sizeof(double) * length);
  // initialize image array with the cosine window and return
  for (int i = 0; i < length; i++)
    window[i] = 0;

  // now initialize a real array as large as the image array with the values
  // of the gaussian
  for (int i = 0; i < length; i++)
    window[i] = 0;

  int meshWidth = width / 2;
  for (int i = location.y - meshWidth, k = 0; i < location.y + meshWidth; i++) {
    for (int j = location.x - meshWidth, l = 0; j < location.x + meshWidth; j++) {
      window[i * imgSize.width + j] = cosine[k][l++];
    }
    k++;
  }

  //  showRealImage("__window", window);

  return window;
}

// createWindow
// Method to create a window around a given location to drop the value of 
// the pixels all around the area of interest in the image. The size of the 
// field is the same as the filter size. The spread parameters are used to 
// define the spread of the hot spot on the image plane. Pixels close to 
// location have the highest values and those beyond the spread rapidly go 
// to zero

double* MosseFilter::createWindow(CvPoint& location, double xSpread, double ySpread) {
  // Linear space vector. Create a meshgrid with x and y axes values
  // that stradle the location

  double xspacer[imgSize.width];
  double yspacer[imgSize.height];

  int lx = location.x;
  double left = -lx;
  for (int i = 0; i < imgSize.width; i++) {
    xspacer[i] = left;
    left += 1.0;
  }
  int ly = location.y;
  double top = -ly;
  for (int i = 0; i < imgSize.height; i++) {
    yspacer[i] = top;
    top += 1.0;
  }

  // Mesh grid
  //double x[imgSize.height][imgSize.width];
  //double y[imgSize.height][imgSize.width];
  double** x = (double**) malloc(sizeof(double)*imgSize.height);
  double** y = (double**) malloc(sizeof(double)*imgSize.height);
  for(int i = 0; i < imgSize.height; i++) {
      x[i] = (double*) malloc(sizeof(double)*imgSize.width);
      y[i] = (double*) malloc(sizeof(double)*imgSize.width);
  }

  for (int i = 0; i < imgSize.height; i++) {
    for (int j = 0; j < imgSize.width; j++) {
      x[i][j] = xspacer[j];
      y[i][j] = yspacer[i];
    }
  }

  // create a gaussian as big as the image
  double** gaussian = (double**) malloc(sizeof(double)*imgSize.height); // new double[nsize.height][nsize.width];
  for(int i = 0; i < imgSize.height; i++) {
      gaussian[i] = (double*) malloc(sizeof(double)*imgSize.width);
  }

  double det = xSpread * ySpread;

  for (int i = 0; i < imgSize.height; i++) {
    for (int j = 0; j < imgSize.width; j++) {
      // using just the gaussian kernel
      double X = x[i][j] * x[i][j];
      double Y = y[i][j] * y[i][j];
      gaussian[i][j] = exp(-((X * ySpread + Y * xSpread) / det));
    }
  }

  double* window = (double*)fftw_malloc(sizeof(double) * length);

  // now initialize a real array as large as the image array with the values
  // of the gaussian
  for (int i = 0; i < length; i++)
    window[i] = 0;
  for (int i = 0; i < imgSize.height; i++) {
    for (int j = 0; j < imgSize.width; j++) {
      window[i * imgSize.width + j] = gaussian[i][j];
    }
  }

  for(int i = 0; i < imgSize.height; i++) {
      delete gaussian[i];
      delete x[i];
      delete y[i];
  }
  delete gaussian;
  delete x;
  delete y;

  // showRealImage("__window", window);

  return window;
}

// getAffineTransforms
// Method to generate a set of affine transformations of a given image. The
// image is rotated and translated to perturb the LOI around the given location.
// The method returns a vector of images that have been perturbed with small
// affine transforms

vector<ImgLocPairT>& MosseFilter::getAffineTransforms(IplImage* image, CvPoint location) {
  // first check if affine transformations are needed, if not then simply
  // push the input images to the vector of transformed images and return
  transformedImages.push_back(make_pair(image, location));
  if (!doAffineTransforms)
    return transformedImages;

  // Setup unchanging data sets used for transformations
  // for rotation
  Mat imageMat(image);
  Point2f center(imageMat.cols / 2.0F, imageMat.rows / 2.0F);

  // for translation
  Mat translationMat = getRotationMatrix2D(center, 0, 1.0);
  translationMat.at<double>(0, 0) = 1;
  translationMat.at<double>(0, 1) = 0;
  translationMat.at<double>(1, 0) = 0;
  translationMat.at<double>(1, 1) = 1;

  // perform rotations. For each rotated image, do a set of translations
  // and push images into the vector of transformed images
  for (double angle = -2.0; angle < 2.0; angle += 0.25) {
    // skip when no rotation is required. This image is already part of
    // out training set
    if (angle == 0) continue;
 
    Mat rotationMat = getRotationMatrix2D(center, angle, 1.0);
    IplImage* transformedImage = cvCloneImage(image);
    Mat dest(transformedImage);
    warpAffine(imageMat, dest, rotationMat, imageMat.size());

    CvPoint rotatedLocation;
    rotatedLocation.x = rotationMat.at<double>(0, 0) * location.x +
      rotationMat.at<double>(0, 1) * location.y +
      rotationMat.at<double>(0, 2);
    rotatedLocation.y = rotationMat.at<double>(1, 0) * location.x +
      rotationMat.at<double>(1, 1) * location.y +
      rotationMat.at<double>(1, 2);

    // check if the rotated location is out of bounds with respect
    // to the image window. Do not add those images to the set
    if (rotatedLocation.x < 0 || rotatedLocation.x > imgSize.width ||
	rotatedLocation.y < 0 || rotatedLocation.y > imgSize.height) {
      cvReleaseImage(&transformedImage);
      continue;
    }

    pair<IplImage*, CvPoint> p = make_pair(transformedImage, rotatedLocation);
    transformedImages.push_back(p);

    // perform a set of translations of each rotated image
    Mat src(transformedImage);
    for (double xdist = -20; xdist <= 20; xdist += 10) 
      for (double ydist = -20; ydist <= 20; ydist += 10) {
	// skip when no translation is needed
	if (!xdist && !ydist) continue;
      
	translationMat.at<double>(0, 2) = xdist;
	translationMat.at<double>(1, 2) = ydist;
	IplImage* translatedImage = cvCloneImage(transformedImage);
	Mat dest(translatedImage);
	warpAffine(src, dest, translationMat, src.size());

	CvPoint translatedLocation;
	translatedLocation.x = rotatedLocation.x + xdist;
	translatedLocation.y = rotatedLocation.y + ydist;

	// check if the translated location is out of bounds with respect
	// to the image window. Do not add those images to the set
	if (translatedLocation.x < 0 || translatedLocation.x > imgSize.width ||
	    translatedLocation.y < 0 || translatedLocation.y > imgSize.height) {
	  cvReleaseImage(&translatedImage);
	  continue;
	}
      
	pair<IplImage*, CvPoint> p = make_pair(translatedImage, translatedLocation);
	transformedImages.push_back(p);
      }
  }

  return transformedImages;
}

// destroyAffineTransforms
// Method to destroy the images generated using affine transforms

void MosseFilter::destroyAffineTransforms(vector<ImgLocPairT>& imgLocPairs) {
  for (unsigned int i = 0; i < imgLocPairs.size(); i++)
    cvReleaseImage(&(imgLocPairs[i].first));
  imgLocPairs.clear();
}
