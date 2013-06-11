#ifndef __FILTER_H
#define __FILTER_H

// Filter.h
// This file contains the definition of the Filter class, which extends the
// FilterBase class.
// We implement methods that are common to all filter creation classes here.
// The workhorse methods belong here.
// Filter creation follows the following steps,
// ...

#include "FilterBase.h"
#include <iostream>
#include <fstream>
#include <math.h>
#include <sys/types.h>
#include <dirent.h>
#include <unistd.h>
#include <stdlib.h>


class Globals {
public:
    static const double windowXScale = 3;
    static const double windowYScale = 3;
};

#if 0
// the ROI extraction function pointer type
typedef IplImage* (*roiFnT)(IplImage*, FrameAnnotation&, 
			    CvPoint&, Annotations::Tag xmlTag);
#endif

// an image and location pair type
typedef pair<IplImage*, CvPoint> ImgLocPairT;

class MosseFilter : public FilterBase {
 protected:
  string outputDirectory;          // the name of the output directory
  CvSize imgSize;                  // the size of the input image
  double gaussianSpread;           // the spread of the gaussian we want to drop
  int length;                      // length of the image array
  fftw_complex* filter;            // the filter
  //roiFnT roiFunction;              // an optional function to get image ROI

 public:
  // Constructor used to construct a filter
  MosseFilter(CvSize size, double gaussianSpread, CvPoint& windowCente); //, roiFnT roiFunction = 0);

  virtual ~MosseFilter();

  // main methods
  virtual void addTraining(IplImage* image, CvPoint point);
  virtual void create();
  virtual void setAffineTransforms() {
    doAffineTransforms = true;
  }

  // method to preprocess images
  virtual fftw_complex* preprocessImage(IplImage* image);

  // method to apply a filter to an image array
  virtual IplImage* apply(fftw_complex* imageFFT);

  virtual IplImage* correlate(IplImage* image);
  virtual IplImage* correlate(fftw_complex* imageFFT);

  // public helper methods
  fftw_complex* convolve(fftw_complex* one, fftw_complex* two, fftw_complex* result = 0);
  IplImage* computeInvFFT();
  fftw_complex* computeFFT(IplImage* image);
  fftw_complex* computeFFT();

  // method to set window center
  void setWindowCenter(CvPoint& center);

  CvSize getSize() { return imgSize; }

 protected:
  fftw_complex* createGaussian(CvPoint& location, CvSize& size, double sd);

  double* createCosine(CvPoint& location);
  double* createWindow(CvPoint& location, double xSpread, double ySpread);

  void applyWindow(IplImage* src, double* window, double* dest);
  void applyWindow(IplImage* src, double* window, IplImage* dest);

  vector<ImgLocPairT>& getAffineTransforms(IplImage* image, CvPoint location);

  void destroyAffineTransforms(vector<ImgLocPairT>& imgLocPairs);
  void boostFilter(IplImage* src, IplImage* dest);

  // if the following is set then for each update operation during
  // filter generation, we take a set of affine transformations of
  // each image and update the filter using the original image and
  // the affine transformations of the image
  bool doAffineTransforms;

  // used to store a set of images after doing small perturbations of the
  // images using affine transformations for filter update
  vector<ImgLocPairT> transformedImages;

  // members used to accumulate numerator and denominator terms for 
  // each filter object
  fftw_complex* mosseNum;
  fftw_complex* mosseDen;

  // post filter image
  IplImage* postFilterImg;

  // buffer variables that are used to re-use image buffers for obviating
  // the need to allocate and free memory
  static int nComplexVectors;

  int storageIndex;
  IplImage* realImg;
  IplImage* tempImg;
  double* imageBuffer;

  // pre-created window buffer
  double* window;
  CvPoint windowCenter;

  // complex buffer for forward and backward FFT. We create plans
  // in both directions using this buffer
  fftw_complex* fftBuffer;

  // pre-created forward and backward FFT plans 
  fftw_plan planForward;
  fftw_plan planBackward;

  vector<fftw_complex*> complexVectors;

  // helper methods to get a free buffer
  inline fftw_complex* getBuffer() {
    fftw_complex* __result = complexVectors[storageIndex];
    storageIndex = (storageIndex + 1) % nComplexVectors;
    return __result;
  }
};

#endif // __FILTER_H
