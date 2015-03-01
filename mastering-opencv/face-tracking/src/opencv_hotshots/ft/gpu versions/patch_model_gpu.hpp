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

class patch_model_gpu{                              //correlation-based patch expert
    typedef gpu::GpuMat GpuMat>;
public:
    GpuMat P;                                       //normalised patch
  
    Size                                            //size of patch model
    patch_size(){return P.size();}

    GpuMat                                          //response map (CV_32F)
    calc_response(const GpuMat &im);                //image to compute response from

    void
    write(FileStorage &fs) const;                   //file storage object to write to

    void
    read(const FileNode& node);                     //file storage node to read from

protected:
    GpuMat                                          //single channel log-scale image
    convert_image(const GpuMat &im);                //gray or rgb unsigned char image
};
//==============================================================================
class patch_models_gpu{                                 //collection of patch experts
    typedef gpu::GpuMat GpuMat>;
public:
    GpuMat reference;                               //reference shape
    vector<patch_model_gpu> patches;                //patch models

    inline int                                      //number of patches
    n_patches(){return patches.size();}

    vector<Point2f>                                 //locations of peaks/feature
    calc_peaks(const GpuMat &im,                    //image to detect features in
               const vector<Point2f> &points,       //initial estimate of shape
               const Size ssize=Size(21,21));       //search window size
    
    void
    write(FileStorage &fs) const;                   //file storage object to write to

    void
    read(const FileNode& node);                     //file storage node to read from

protected:
    GpuMat                                          //inverted similarity transform
    inv_simil(const GpuMat &S);                     //similarity transform

    GpuMat                                          //similarity tranform referece->pts
    calc_simil(const GpuMat &pts);                  //destination shape

    vector<Point2f>                                 //similarity transformed shape
    apply_simil(const GpuMat &S,                    //similarity transform
                const vector<Point2f> &points);     //shape to transform
};
//==============================================================================


#endif /* _FT_PATCH_MODEL_HPP_ */
