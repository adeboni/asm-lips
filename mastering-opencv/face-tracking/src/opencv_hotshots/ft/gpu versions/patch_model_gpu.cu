#include "opencv_hotshots/ft/patch_model.hpp"
#include "opencv_hotshots/ft/ft.hpp"
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include "stdio.h"      // For 'sprintf()'

#ifdef WITH_CUDA
#include "cuda_runtime.h"
#endif

#define fl at<float>
//==============================================================================
#ifdef WITH_CUDA
GpuMat patch_model_gpu::convert_image(const GpuMat &im)
{
    GpuMat I;
    if (im.channels() == 1) {
        if (im.type() != CV_32F) im.convertTo(I, CV_32F);
        else I = im;
    } else {
        if (im.channels() == 3) {
            GpuMat img;
            gpu::cvtColor(im, img, CV_RGB2GRAY);
            if (img.type() != CV_32F) img.convertTo(I, CV_32F);
            else I = img;
        } else {
            cout << "Unsupported image type!" << endl;
            abort();
        }
    }
    gpu::add(I, Scalar(1.0), I); // Used to be I += 1.0;
    log(I, I);
    return I;
}
#endif /* WITH_CUDA */

//==============================================================================
#ifdef WITH_CUDA
GpuMat patch_model_gpu::calc_response(const GpuMat &im) {
    GpuMat res;
    gpu::matchTemplate(this->convert_image(im), P, res, CV_TM_SQDIFF); //might want to change to CV_TM_CCORR
    gpu::normalize(res, res, 0, 1, NORM_MINMAX); 
	gpu::divide(res, gpu::sum(res)[0], res);
    return res;
}
#endif /* WITH_CUDA */
//==============================================================================
void patch_model_gpu::write(FileStorage &fs) const {
    assert(fs.isOpened());
    fs << "{" << "P" << P << "}";
}  
//==============================================================================
void patch_model_gpu::read(const FileNode& node) {
    assert(node.type() == FileNode::MAP);
    node["P"] >> P;
}
//==============================================================================
#ifdef WITH_CUDA
__global__ void calc_peaks_kernel(gpu::PtrStepSz<float> A, gpu::PtrStepSz<float> S, gpu::PtrStepSz<float> pt, int i, int w, int h) {
	A(0, 0) = S(0, 0);
	A(0, 1) = S(0, 1);
	A(1, 0) = S(1, 0);
	A(1, 1) = S(1, 1);
	A(2, 0) = pt(2 * i, 1) - A(0, 0) * (w - 1) / 2 + A(1, 0) * (h - 1) / 2;
	A(2, 1) = pt(2 * i + 1, 1) - A(0, 1) * (w - 1) / 2 + A(1, 1) * (h - 1) / 2;
}

vector<Point2f> patch_models_gpu::calc_peaks(const GpuMat &im, const vector<Point2f> &points, const Size ssize) {
    int n = points.size();
    assert(n == int(patches.size()));
    GpuMat pt = GpuMat(Mat(points)).reshape(1, 2*n);
    GpuMat S = this->calc_simil(pt);
    vector<Point2f> pts = this->apply_simil(this->inv_simil(S), points);
    for (int i = 0; i < n; i++) {
        Size wsize = ssize + patches[i].patch_size();
        GpuMat A(2, 3, CV_32F);
		calc_peaks_kernel<<<1, 1>>>(A, S, pt, i, wsize.width, wsize.height);
        GpuMat I;
        Mat Amat(A);
		gpu::warpAffine(im, I, Amat, wsize, INTER_LINEAR+WARP_INVERSE_MAP);
        GpuMat R = patches[i].calc_response(I);
        
        Point maxLoc; 
		gpu::minMaxLoc(R, 0, 0, 0, &maxLoc);
        pts[i] = Point2f(pts[i].x + maxLoc.x - 0.5*ssize.width, pts[i].y + maxLoc.y - 0.5*ssize.height);
    }
    return this->apply_simil(S, pts);
}
#endif /* WITH_CUDA */
//=============================================================================
vector<Point2f> patch_models_gpu::apply_simil(const Mat &S, const vector<Point2f> &points) {
    int n = points.size();
    vector<Point2f> p(n);
    for(int i = 0; i < n; i++) {
        p[i].x = S.fl(0,0)*points[i].x + S.fl(0,1)*points[i].y + S.fl(0,2);
        p[i].y = S.fl(1,0)*points[i].x + S.fl(1,1)*points[i].y + S.fl(1,2);
    }
    return p;
}

#ifdef WITH_CUDA

__global__ void apply_simil_kernel(const gpu::PtrStepSz<float> S, float *points, int numPoints, float *output)
{
    /*for(int i = 0; i < numPoints; i++) {
        float points_x = *(points + i*2);
        float points_y = *(points + i*2 + 1);
        
        // Original CPU code for reference.
//        p[i].x = S.fl(0,0)*points[i].x + S.fl(0,1)*points[i].y + S.fl(0,2);
//        p[i].y = S.fl(1,0)*points[i].x + S.fl(1,1)*points[i].y + S.fl(1,2);
        
        *(output + i*2) = S(0,0) * points_x + S(1,0) * points_y + S(2,0);
        *(output + i*2 + 1) = S(0,1) * points_x + S(1,1) * points_y + S(2,1);
    }*/
    
    int i = blockIdx.x;
    float points_x = *(points + i*2);
    float points_y = *(points + i*2 + 1);
    
    /* Original CPU code for reference. */
    //        p[i].x = S.fl(0,0)*points[i].x + S.fl(0,1)*points[i].y + S.fl(0,2);
    //        p[i].y = S.fl(1,0)*points[i].x + S.fl(1,1)*points[i].y + S.fl(1,2);
    
    *(output + i*2) = S(0,0) * points_x + S(1,0) * points_y + S(2,0);
    *(output + i*2 + 1) = S(0,1) * points_x + S(1,1) * points_y + S(2,1);
}

vector<Point2f> patch_models_gpu::apply_simil(const GpuMat &S, const vector<Point2f> &points) {
    int n = points.size();
    int num_bytes = n*2*sizeof(float);
    vector<Point2f> p(n);
    
    float *funcInput = (float *) &(points[0].x);
    float *funcOutput = (float *) &(p[0].x);
    float *deviceFuncInput, *deviceFuncOutput;
    
    //cudaMalloc((void**)&device_array, num_bytes);
    cudaMalloc((void**)&deviceFuncInput, num_bytes);
    cudaMalloc((void**)&deviceFuncOutput, num_bytes);
    
    cudaMemcpy(deviceFuncInput, funcInput, num_bytes, cudaMemcpyHostToDevice);
    
    apply_simil_kernel<<<n, 1>>>(S, funcInput, n, funcOutput);
    
    cudaMemcpy(funcOutput, deviceFuncOutput, num_bytes, cudaMemcpyDeviceToHost);
    
    return p;
}

#endif /* WITH_CUDA */

//=============================================================================
Mat patch_models_gpu::inv_simil(const Mat &S) {
    Mat Si(2,3,CV_32F);
    float d = S.fl(0,0)*S.fl(1,1) - S.fl(1,0)*S.fl(0,1);
    Si.fl(0,0) = S.fl(1,1)/d; 
	Si.fl(0,1) = -S.fl(0,1)/d;
    Si.fl(1,1) = S.fl(0,0)/d; 
	Si.fl(1,0) = -S.fl(1,0)/d;
    Mat Ri = Si(Rect(0,0,2,2));
    Ri = -Ri*S.col(2); 
	Mat St = Si.col(2); 
	Ri.copyTo(St); 
	return Si;
}

#ifdef WITH_CUDA
__global__ void inv_simil_kernel(gpu::PtrStepSz<float> S, gpu::PtrStepSz<float> Si) {
	float d = S(0, 0)*S(1, 1) - S(0, 1)*S(1, 0);
    Si(0,0) = S(1,1)/d; 
	Si(1,0) = -S(1,0)/d;
    Si(1,1) = S(0,0)/d; 
	Si(0,1) = -S(0,1)/d;
}

GpuMat patch_models_gpu::inv_simil(const GpuMat &S) {
    GpuMat Si(2,3,CV_32F);
	inv_simil_kernel<<<1,1>>>(S, Si);
    GpuMat Ri = Si(Rect(0,0,2,2));
    
    gpu::multiply(Ri, Scalar(-1), Ri);  // Originally Ri = -Ri*S.col(2);
    gpu::multiply(Ri, S.col(2), Ri);
    
	GpuMat St = Si.col(2);
	Ri.copyTo(St); 
	return Si;
}
#endif /* WITH_CUDA */
//=============================================================================
#ifdef WITH_CUDA

__global__ void calc_simil_kernel1(gpu::PtrStepSz<float> pts, float *mx, float *my, int n) {
	for (int i = 0; i < n; i++) {
        *mx += pts(2*i, 1);         // mx += pts.fl(2*i);
        *my += pts(2*i+1, 1);       // my += pts.fl(2*i+1);
    }
}


__global__ void calc_simil_kernel2(gpu::PtrStepSz<float> pts, gpu::PtrStepSz<float> ref, float *p, float mx, float my, float *a, float *b, float *c, int n) {
    *a = 0;
    *b = 0;
    *c = 0;
    
    for (int i = 0; i < n; i++) {
		p[2*i] = pts(2*i, 1) - mx;
		p[2*i+1] = pts(2*i+1, 1) - my;
	}
	
	for (int i = 0; i < n; i++) {
        *a += ref(2*i, 1) * ref(2*i, 1) + ref(2*i+1, 1) * ref(2*i+1, 1);
        *b += ref(2*i, 1) * p[2*i] + ref(2*i+1, 1) * p[2*i+1];
        *c += ref(2*i, 1) * p[2*i+1] - ref(2*i+1, 1) * p[2*i];
    }
	
    *b /= *a;
    *c /= *a;
}


__global__ void calc_simil_kernel3(gpu::PtrStepSz<float> ret, float sc, float ss, float mx, float my) {
	ret(0, 0) = sc;
	ret(1, 0) = -ss;
	ret(0, 1) = mx;
	ret(1, 1) = ss;
	ret(0, 2) = sc;
	ret(1, 2) = my;
}


GpuMat patch_models_gpu::calc_simil(const GpuMat &pts) {
    //compute translation
    int n = pts.rows/2;
    float mx = 0, my = 0;
	calc_simil_kernel1<<<1, 1>>>(pts, &mx, &my, n);
    mx /= n;
    my /= n;
	
    vector<float> p(2*n);
	int num_bytes = 2*n*sizeof(float);
    float *funcInput = (float *) &(p[0]);
    float *deviceFuncInput;
    cudaMalloc((void**)&deviceFuncInput, num_bytes);
   
	float a=0, b=0, c=0;
    float *dev_a, *dev_b, *dev_c;
    cudaMalloc((void**)&dev_a, sizeof(float));
    cudaMalloc((void**)&dev_b, sizeof(float));
    cudaMalloc((void**)&dev_c, sizeof(float));
    
    cudaMemcpy(deviceFuncInput, funcInput, num_bytes, cudaMemcpyHostToDevice);
    calc_simil_kernel2<<<1, 1>>>(pts, reference, deviceFuncInput, mx, my, dev_a, dev_b, dev_c, n);
    
    cudaMemcpy(&a, dev_a, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&b, dev_b, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&c, dev_c, sizeof(float), cudaMemcpyDeviceToHost);

    float scale = sqrt(b*b+c*c), theta = atan2(c,b);
    float sc = scale*cos(theta), ss = scale*sin(theta);
	GpuMat ret(2,3,CV_32F);
	calc_simil_kernel3<<<1, 1>>>(ret, sc, ss, mx, my);
    
	return ret;
}

#endif /* WITH_CUDA */
//==============================================================================
void patch_models_gpu::write(FileStorage &fs) const {
    assert(fs.isOpened());
    fs << "{" << "reference" << reference;
    fs << "n_patches" << (int)patches.size();
    for(int i = 0; i < int(patches.size()); i++){
        char str[256]; const char* ss;
        sprintf(str,"patch %d",i); ss = str; fs << ss << patches[i];
    }
    fs << "}";
}
//==============================================================================
void patch_models_gpu::read(const FileNode& node) {
    assert(node.type() == FileNode::MAP);
    node["reference"] >> reference;
    int n; node["n_patches"] >> n; patches.resize(n);
    for(int i = 0; i < n; i++){
        char str[256]; const char* ss;
        sprintf(str,"patch %d",i); ss = str; node[ss] >> patches[i];
    }
}
//==============================================================================
