#ifndef _FT_PATCH_MODEL_HPP_
#define _FT_PATCH_MODEL_HPP_
#include "opencv_hotshots/ft/ft_data.hpp"
#include <opencv2/core/core.hpp>
#include <vector>
#include "flags.hpp"

#ifdef WITH_CUDA
#include <opencv2/gpu/gpu.hpp>
#include <opencv2/gpu/gpumat.hpp>
#endif /* WITH_CUDA */

using namespace cv;
using namespace std;

class patch_model{
	#ifdef WITH_CUDA
    typedef gpu::GpuMat GpuMat;
	#endif
public:
    Mat P;                                          //normalised patch
	#ifdef WITH_CUDA
	GpuMat gpuP;
	GpuMat conv;
	GpuMat res;
	#endif
  
    Size                                            //size of patch model
    patch_size(){return P.size();}

    Mat                                             //response map (CV_32F)
    calc_response(const Mat &im);                   //image to compute response from
	
	#ifdef WITH_CUDA
	Mat calc_response(const GpuMat &im);
	void pre_calc_response(gpu::Stream &st);
	#endif

    void
    train(const vector<Mat> &images,                //feature centered training images
          const Size psize,                         //desired patch size
          const float var = 1.0,                    //variance of annotation error
          const float lambda = 1e-6,                //regularization weight
          const float mu_init = 1e-3,               //initial stoch-grad step size
          const int nsamples = 1000,                //number of stoch-grad samples
          const bool visi = false);                 //visualize intermediate results?

    void
    write(FileStorage &fs) const;                   //file storage object to write to

    void
    read(const FileNode& node);                     //file storage node to read from

protected:
    Mat                                             //single channel log-scale image
    convert_image(const Mat &im);                   //gray or rgb unsigned char image
    
	#ifdef WITH_CUDA
    GpuMat                                     //GPU Version.
    convert_image(const GpuMat &im);
	#endif
};
//==============================================================================
class patch_models{
	#ifdef WITH_CUDA
    typedef gpu::GpuMat GpuMat;
	#endif
public:
    Mat reference;                                  //reference shape
    vector<patch_model> patches;                    //patch models

    inline int                                      //number of patches
    n_patches(){return patches.size();}

    void
    train(ft_data &data,                            //training data
          const vector<Point2f> &ref,               //reference shape
          const Size psize,                         //desired patch size
          const Size ssize,                         //search window size
          const bool mirror = true,                 //use mirrored images?
          const float var = 1.0,                    //variance of annotation error
          const float lambda = 1e-6,                //regularization weight
          const float mu_init = 1e-3,               //initial stoch-grad step size
          const int nsamples = 1000,                //number of stoch-grad samples
          const bool visi = true);                  //visualize intermediate results?

    vector<Point2f>                                 //locations of peaks/feature
    calc_peaks(const Mat &im,                       //image to detect features in
               const vector<Point2f> &points,       //initial estimate of shape
               const Size ssize=Size(21,21));       //search window size
    
	#ifdef WITH_CUDA
    vector<Point2f>                                 //GPU Version.
    calc_peaks(const GpuMat &im,
               const vector<Point2f> &points,
               const Size ssize=Size(21,21));
	#endif
    
    void
    write(FileStorage &fs) const;                   //file storage object to write to

    void
    read(const FileNode& node);                     //file storage node to read from

protected:
    Mat                                             //inverted similarity transform
    inv_simil(const Mat &S);                        //similarity transform
	
	#ifdef WITH_CUDA
	GpuMat                                     //GPU version
    inv_simil(const GpuMat &S);   
	#endif

    Mat                                             //similarity tranform referece->pts
    calc_simil(const Mat &pts);                     //destination shape
    
	#ifdef WITH_CUDA
    GpuMat                                     //GPU Version.
    calc_simil(const GpuMat &pts);
	#endif

    vector<Point2f>                                 //similarity transformed shape
    apply_simil(const Mat &S,                       //similarity transform
                const vector<Point2f> &points);     //shape to transform
    
	#ifdef WITH_CUDA
    vector<Point2f>                                 //GPU Version.
    apply_simil(const GpuMat &S,
                const vector<Point2f> &pts);
    
    vector<Point2f>                                 //GPU Version 2.
    apply_simil2(const GpuMat &S,
                const vector<Point2f> &pts);
	#endif
};
//==============================================================================


#endif /* _FT_PATCH_MODEL_HPP_ */
