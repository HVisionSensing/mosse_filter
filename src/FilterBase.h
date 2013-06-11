#ifndef __FILTERBASE_H
#define __FILTERBASE_H

// FilterBase.h
// This file contains the definition of the abstract base filter class.
// Each image has a set of locations of interest. For example,
// a. The location of the center of the left eye
// b. The location of the center of the right eye
// c. The location of the left eye corner
// d. ...
// For each location, we have a filter that has been created using our
// training data.
// In extensions of this class we learn filters given a set of images and
// the co-ordinates of the locations of interest

#include <stdio.h>
#include <string.h>
#include <limits.h>
#include <float.h>
#include <iostream>
#include <algorithm>

// The standard OpenCV headers
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <fftw3.h>
#include <omp.h>

using namespace std;
using namespace cv;

class FilterBase {
 public:
 public:
  FilterBase() {}
  virtual ~FilterBase() {}

  virtual fftw_complex* preprocessImage(IplImage* image) = 0;
  virtual void addTraining(IplImage* image, CvPoint location) = 0;
  virtual void create() = 0;
  virtual void setAffineTransforms() = 0;
  virtual IplImage* apply(fftw_complex* imageFFT) = 0;
};

#endif // __FILTERBASE_H
