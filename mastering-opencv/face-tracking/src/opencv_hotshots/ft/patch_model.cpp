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
Mat patch_model::convert_image(const Mat &im) {
    Mat I;
    if (im.channels() == 1) {
        if (im.type() != CV_32F) im.convertTo(I, CV_32F);
        else I = im;
    } else {
        if (im.channels() == 3) {
            Mat img;
            cvtColor(im, img, CV_RGB2GRAY);
            if (img.type() != CV_32F) img.convertTo(I, CV_32F);
            else I = img;
        } else {
			cout << "Unsupported image type!" << endl; 
			abort();
		}
    }
    I += 1.0;
    log(I, I);
    return I;
}

#ifdef WITH_CUDA
gpu::GpuMat patch_model::convert_image(const gpu::GpuMat &im)
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
Mat patch_model::calc_response(const Mat &im) {
    Mat res;
    matchTemplate(this->convert_image(im), P, res, CV_TM_CCOEFF_NORMED);
    normalize(res, res, 0, 1, NORM_MINMAX); 
	res /= sum(res)[0];
    return res;
}

#ifdef WITH_CUDA
gpu::GpuMat patch_model::calc_response(const gpu::GpuMat &im) {
    GpuMat res;
    gpu::matchTemplate(this->convert_image(im), GpuMat(P), res, CV_TM_SQDIFF); //might want to change to CV_TM_CCORR
    gpu::normalize(res, res, 0, 1, NORM_MINMAX); 
	gpu::divide(res, gpu::sum(res)[0], res);
    return res;
}
#endif /* WITH_CUDA */
//==============================================================================
void patch_model::train(const vector<Mat> &images, const Size psize, const float var, const float lambda, const float mu_init, const int nsamples, const bool visi) {
    int N = images.size(), n = psize.width*psize.height;

    //compute desired response map
    Size wsize = images[0].size();
    if((wsize.width < psize.width) || (wsize.height < psize.height)){
        cerr << "Invalid image size < patch size!" << endl; throw std::exception();
    }
    int dx = wsize.width-psize.width, dy = wsize.height-psize.height;
    Mat F(dy,dx,CV_32F);
    for(int y = 0; y < dy; y++){   float vy = (dy-1)/2 - y;
        for(int x = 0; x < dx; x++){ float vx = (dx-1)/2 - x;
            F.fl(y,x) = exp(-0.5*(vx*vx+vy*vy)/var);
        }
    }
    normalize(F,F,0,1,NORM_MINMAX);

    //allocate memory
    Mat I(wsize.height,wsize.width,CV_32F);
    Mat dP(psize.height,psize.width,CV_32F);
    Mat O = Mat::ones(psize.height,psize.width,CV_32F)/n;
    P = Mat::zeros(psize.height,psize.width,CV_32F);

    //optimise using stochastic gradient descent
    RNG rn(getTickCount()); double mu=mu_init,step=pow(1e-8/mu_init,1.0/nsamples);
    for(int sample = 0; sample < nsamples; sample++){ int i = rn.uniform(0,N);
        I = this->convert_image(images[i]); dP = 0.0;
        for(int y = 0; y < dy; y++){
            for(int x = 0; x < dx; x++){
                Mat Wi = I(Rect(x,y,psize.width,psize.height)).clone();
                Wi -= Wi.dot(O); normalize(Wi,Wi);
                dP += (F.fl(y,x) - P.dot(Wi))*Wi;
            }
        }
        P += mu*(dP - lambda*P); mu *= step;
        if(visi){
            Mat R; matchTemplate(I,P,R,CV_TM_CCOEFF_NORMED);
            Mat PP; normalize(P,PP,0,1,NORM_MINMAX);
            normalize(dP,dP,0,1,NORM_MINMAX);
            normalize(R,R,0,1,NORM_MINMAX);
            imshow("P",PP); imshow("dP",dP); imshow("R",R);
            if(waitKey(10) == 27)break;
        }
    }
    return;
}
//==============================================================================
void patch_model::write(FileStorage &fs) const {
    assert(fs.isOpened());
    fs << "{" << "P" << P << "}";
}  
//==============================================================================
void patch_model::read(const FileNode& node) {
    assert(node.type() == FileNode::MAP);
    node["P"] >> P;
}
//==============================================================================
void patch_models::train(ft_data &data, const vector<Point2f> &ref, const Size psize, const Size ssize, const bool mirror, const float var, const float lambda, const float mu_init, const int nsamples, const bool visi) {
  //set reference shape
    int n = ref.size(); 
	reference = Mat(ref).reshape(1,2*n);
    Size wsize = psize + ssize;

    //train each patch model in turn
    patches.resize(n);
    for(int i = 0; i < n; i++){
        if (visi) cout << "training patch " << i << "..." << endl;
        vector<Mat> images(0);
        for (int j = 0; j < data.n_images(); j++) {
            Mat im = data.get_image(j, 0);
            vector<Point2f> p = data.get_points(j,false);
            Mat pt = Mat(p).reshape(1, 2*n);
            Mat S = this->calc_simil(pt), A(2, 3, CV_32F);
            A.fl(0,0) = S.fl(0,0); 
			A.fl(0,1) = S.fl(0,1);
            A.fl(1,0) = S.fl(1,0); 
			A.fl(1,1) = S.fl(1,1);
            A.fl(0,2) = pt.fl(2*i  ) - (A.fl(0,0) * (wsize.width-1)/2 + A.fl(0,1)*(wsize.height-1)/2);
            A.fl(1,2) = pt.fl(2*i+1) - (A.fl(1,0) * (wsize.width-1)/2 + A.fl(1,1)*(wsize.height-1)/2);
            Mat I;
			warpAffine(im,I,A,wsize,INTER_LINEAR+WARP_INVERSE_MAP);
            images.push_back(I);
            if (mirror) {
                im = data.get_image(j, 1);
                p = data.get_points(j, true);
                pt = Mat(p).reshape(1, 2*n);
                S = this->calc_simil(pt);
                A.fl(0,0) = S.fl(0,0); 
				A.fl(0,1) = S.fl(0,1);
                A.fl(1,0) = S.fl(1,0); 
				A.fl(1,1) = S.fl(1,1);
                A.fl(0,2) = pt.fl(2*i  ) - (A.fl(0,0) * (wsize.width-1)/2 + A.fl(0,1)*(wsize.height-1)/2);
                A.fl(1,2) = pt.fl(2*i+1) - (A.fl(1,0) * (wsize.width-1)/2 + A.fl(1,1)*(wsize.height-1)/2);
                warpAffine(im,I,A,wsize,INTER_LINEAR+WARP_INVERSE_MAP);
                images.push_back(I);
            }
        }
        patches[i].train(images,psize,var,lambda,mu_init,nsamples,visi);
    }
}
//==============================================================================
vector<Point2f> patch_models::calc_peaks(const Mat &im, const vector<Point2f> &points, const Size ssize) {
    int n = points.size();
    assert(n == int(patches.size()));
    Mat pt = Mat(points).reshape(1,2*n);
    Mat S = this->calc_simil(pt);
    vector<Point2f> pts = Mat(this->apply_simil(GpuMat(this->inv_simil(S)), points));
	//vector<Point2f> pts = this->apply_simil(this->inv_simil(S), points);
	//vector<Point2f> pts = this->apply_simil(Mat(this->inv_simil(GpuMat(S))), points);
    for (int i = 0; i < n; i++) {
        Size wsize = ssize + patches[i].patch_size();
        Mat A(2, 3, CV_32F);
        A.fl(0, 0) = S.fl(0, 0); 
		A.fl(0, 1) = S.fl(0, 1);
        A.fl(1, 0) = S.fl(1, 0); 
		A.fl(1, 1) = S.fl(1, 1);
        A.fl(0, 2) = pt.fl(2*i  ) - (A.fl(0,0) * (wsize.width-1)/2 + A.fl(0,1)*(wsize.height-1)/2);
        A.fl(1, 2) = pt.fl(2*i+1) - (A.fl(1,0) * (wsize.width-1)/2 + A.fl(1,1)*(wsize.height-1)/2);
        Mat I; 
		warpAffine(im, I, A, wsize, INTER_LINEAR+WARP_INVERSE_MAP);
        
        Mat R = patches[i].calc_response(I);
        
        Point maxLoc; 
		minMaxLoc(R, 0, 0, 0, &maxLoc);
        pts[i] = Point2f(pts[i].x + maxLoc.x - 0.5*ssize.width, pts[i].y + maxLoc.y - 0.5*ssize.height);
    }
    return this->apply_simil(S,pts);
}

#ifdef WITH_CUDA
__global__ void calc_peaks_kernel(gpu::PtrStepSz<float> A, gpu::PtrStepSz<float> S, gpu::PtrStepSz<float> pt, int i, int w, int h) {
	A(0, 0) = S(0, 0);
	A(0, 1) = S(0, 1);
	A(1, 0) = S(1, 0);
	A(1, 1) = S(1, 1);
	A(2, 0) = pt(2 * i, 1) - A(0, 0) * (w - 1) / 2 + A(1, 0) * (h - 1) / 2;
	A(2, 1) = pt(2 * i + 1, 1) - A(0, 1) * (w - 1) / 2 + A(1, 1) * (h - 1) / 2;
}

vector<Point2f> patch_models::calc_peaks(const GpuMat &im, const vector<Point2f> &points, const Size ssize) {
    int n = points.size();
    assert(n == int(patches.size()));
    GpuMat pt = GpuMat(Mat(points).reshape(1, 2*n));
    GpuMat S = this->calc_simil(pt);
    vector<Point2f> pts = this->apply_simil(this->inv_simil(S), points);
    for (int i = 0; i < n; i++) {
		cerr << "In loop, i = " << i << endl;
        Size wsize = ssize + patches[i].patch_size();
        GpuMat A(2, 3, CV_32F);
        cerr << "Starting calc_peaks_kernel" << endl;
		calc_peaks_kernel<<<1, 1>>>(A, S, pt, i, wsize.width, wsize.height);
        cerr << "Exiting calc_peaks_kernel" << endl;
        GpuMat I;
		gpu::warpAffine(im, I, Mat(A), wsize, INTER_LINEAR+WARP_INVERSE_MAP);
        GpuMat R = patches[i].calc_response(I);
        
        Point maxLoc; 
		gpu::minMaxLoc(R, 0, 0, 0, &maxLoc);
        pts[i] = Point2f(pts[i].x + maxLoc.x - 0.5*ssize.width, pts[i].y + maxLoc.y - 0.5*ssize.height);
    }
    return this->apply_simil(S, pts);
}
#endif /* WITH_CUDA */
//=============================================================================
vector<Point2f> patch_models::apply_simil(const Mat &S, const vector<Point2f> &points) {
    int n = points.size();
    vector<Point2f> p(n);
    for(int i = 0; i < n; i++) {
        p[i].x = S.fl(0,0)*points[i].x + S.fl(0,1)*points[i].y + S.fl(0,2);
        p[i].y = S.fl(1,0)*points[i].x + S.fl(1,1)*points[i].y + S.fl(1,2);
    }
    return p;
}

#ifdef WITH_CUDA

__global__ void apply_simil_kernel(const gpu::PtrStepSz<float> S, const float *points, float *output, int n) {

    /* Original CPU code for reference. */
    //        p[i].x = S.fl(0,0)*points[i].x + S.fl(0,1)*points[i].y + S.fl(0,2);
    //        p[i].y = S.fl(1,0)*points[i].x + S.fl(1,1)*points[i].y + S.fl(1,2);
    
	int i = threadIdx.x;
    printf("Thread %d --> %d\n", i, (int)points);
	/*if (i < n) {
        printf(" --> Point: (%f, %f)\n", points[i*2], points[i*2 + 1]);
		output[i*2] = S(0,0) * points[i*2] + S(1,0) * points[i*2 + 1] + S(2,0);
		output[i*2 + 1] = S(0,1) * points[i*2] + S(1,1) * points[i*2 + 1] + S(2,1);
	}*/
}

void printPoint(const Point2f &pnt)
{
    cerr << "(" << pnt.x << ", " << pnt.y << ")";
}

vector<Point2f> patch_models::apply_simil(const gpu::GpuMat &S, const vector<Point2f> &points) {
    int n = points.size();
    int num_bytes = n*2*sizeof(float);
    vector<Point2f> p(n);
    
    const float *input = &(points[0].x);
    float *output = &(p[0].x);
    float *dev_input, *dev_output;
    
    cout << "--- Printing Points ---" << endl;
    for (int i = 0; i < points.size(); i++) {
        cout << "Point " << i << " by vector: ";
        printPoint(points[i]);
        cout << endl;
    }
    cout << "--- Done Printing Points ---" << endl;
    
    cudaMalloc((void**)&dev_input, num_bytes);
    cudaMalloc((void**)&dev_output, num_bytes);
    
    cudaMemcpy(dev_input, input, num_bytes, cudaMemcpyHostToDevice);
	
    cerr << "Starting apply_simil_kernel" << endl;
    apply_simil_kernel<<<1, n>>>(S, input, output, n);
    cerr << "Exiting apply_simil_kernel" << endl;
    
    cudaMemcpy(output, dev_output, num_bytes, cudaMemcpyDeviceToHost);
    
    cudaFree(dev_input);
    cudaFree(dev_output);
    
    return p;
}

#endif /* WITH_CUDA */

//=============================================================================
Mat patch_models::inv_simil(const Mat &S) {
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
__global__ void inv_simil_kernel1(gpu::PtrStepSz<float> S, gpu::PtrStepSz<float> Si) {
	float d = S(0, 0)*S(1, 1) - S(0, 1)*S(1, 0);
    Si(0,0) = S(1,1)/d;
	Si(1,0) = -S(1,0)/d;
    Si(1,1) = S(0,0)/d;
	Si(0,1) = -S(0,1)/d;
}

// Used to do matrix multiplication.
__global__ void inv_simil_kernel2(gpu::PtrStepSz<float> src1, gpu::PtrStepSz<float> src2, gpu::PtrStepSz<float> dest) {
    dest(0,0) = src1(0,0)*src2(0,0) + src1(0,1)*src2(1,0);
    dest(1,0) = src1(1,0)*src2(0,0) + src1(1,1)*src2(1,0);
    
//    dest(0,0) = src1(0,0)*src2(0,0) + src1(1,0)*src2(0,1);
//    dest(0,1) = src1(0,1)*src2(0,0) + src1(1,1)*src2(0,1);
}

__global__ void print_mat(gpu::PtrStepSz<float> Ri, int width, int height)
{
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < height; j++)
        {
            printf("(%d, %d) = %f\n", j, i, Ri(j,i));
        }
    }
    printf("\n");
}

gpu::GpuMat patch_models::inv_simil(const gpu::GpuMat &S) {
    GpuMat Si(2,3,CV_32F);
//    cerr << "Starting inv_simil_kernel" << endl;
	inv_simil_kernel1<<<1,1>>>(S, Si);
//    cerr << "Exiting inv_simil_kernel" << endl;
    GpuMat Ri = Si(Rect(0,0,2,2));
    //cerr << "Initially:" << endl;
	//cout << S.size().height << " " << S.size().width << endl;
    //print_mat<<<1,1>>>(Ri, Ri.size().width, Ri.size().height);
    
	//cerr << "Starting first multiply" << endl;
    gpu::multiply(Ri, Scalar(-1.0), Ri);  // Originally Ri = -Ri*S.col(2);
    //cerr << "After first multiply:" << endl;
	//cerr << "Exiting first multiply and starting second multiply" << endl;
    GpuMat T(2,1,CV_32F);
    inv_simil_kernel2<<<1,1>>>(Ri, S.col(2), T);
    //cerr << "After second multiply:" << endl;
	//cerr << "Exiting second multiply" << endl;
    
	GpuMat St = Si.col(2);
	T.copyTo(St);
    //cerr << "About to return from inv_simil." << endl;
	return Si;
}
#endif /* WITH_CUDA */
//=============================================================================
Mat patch_models::calc_simil(const Mat &pts) {
    //compute translation
    int n = pts.rows/2; 
	float mx = 0, my = 0;
    for (int i = 0; i < n; i++) {
        mx += pts.fl(2*i); 
		my += pts.fl(2*i+1);
    }
	mx /= n; 
	my /= n;
	vector<float> p(2*n);
    for (int i = 0; i < n; i++) {
		p[2*i] = pts.fl(2*i) - mx; 
		p[2*i+1] = pts.fl(2*i+1) - my;
    }
    //compute rotation and scale
    float a=0, b=0, c=0;
    for (int i = 0; i < n; i++) {
        a += reference.fl(2*i) * reference.fl(2*i) + reference.fl(2*i+1) * reference.fl(2*i+1);
		b += reference.fl(2*i) * p[2*i] + reference.fl(2*i+1) * p[2*i+1];
        c += reference.fl(2*i) * p[2*i+1] - reference.fl(2*i+1) * p[2*i];
    }
    b /= a; 
	c /= a;
    float scale = sqrt(b*b+c*c), theta = atan2(c,b);
    float sc = scale*cos(theta), ss = scale*sin(theta);
    return (Mat_<float>(2,3) << sc,-ss,mx,ss,sc,my);
}

#ifdef WITH_CUDA

__global__ void calc_simil_kernel1(gpu::PtrStepSz<float> pts, float *mx, float *my, int n) {
    *mx = 0;
    *my = 0;
    
    for (int i = 0; i < n; i++) {
        *mx += pts(2*i, 0);         // mx += pts.fl(2*i);
        *my += pts(2*i+1, 0);       // my += pts.fl(2*i+1);
    }
}


__global__ void calc_simil_kernel2(gpu::PtrStepSz<float> pts, gpu::PtrStepSz<float> ref, float *p, float mx, float my, float *a, float *b, float *c, int n) {
    *a = 0;
    *b = 0;
    *c = 0;
    
    for (int i = 0; i < n; i++) {
		p[2*i] = pts(2*i, 0) - mx;
		p[2*i+1] = pts(2*i+1, 0) - my;
	}
	
	for (int i = 0; i < n; i++) {
        *a += ref(2*i, 0) * ref(2*i, 0) + ref(2*i+1, 0) * ref(2*i+1, 0);
        *b += ref(2*i, 0) * p[2*i] + ref(2*i+1, 0) * p[2*i+1];
        *c += ref(2*i, 0) * p[2*i+1] - ref(2*i+1, 0) * p[2*i];
    }
	
    *b /= *a;
    *c /= *a;
}


__global__ void calc_simil_kernel3(gpu::PtrStepSz<float> ret, float sc, float ss, float mx, float my) {
    ret(0, 0) = sc;
    ret(0, 1) = -ss;
    ret(0, 2) = mx;
    ret(1, 0) = ss;
    ret(1, 1) = sc;
    ret(1, 2) = my;
}


gpu::GpuMat patch_models::calc_simil(const gpu::GpuMat &pts) {
	GpuMat ref;
	ref.upload(reference);
	
    //compute translation
    int n = pts.rows/2;
    float mx = 0, my = 0;
    float *dev_mx, *dev_my;
    
    cudaMalloc((void**)&dev_mx, sizeof(float));
    cudaMalloc((void**)&dev_my, sizeof(float));
    
    //cerr << "Starting calc_simil_kernel1" << endl;
	calc_simil_kernel1<<<1, 1>>>(pts, dev_mx, dev_my, n);
    //cerr << "Exiting calc_simil_kernel1" << endl;
    
    cudaMemcpy(&mx, dev_mx, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&my, dev_my, sizeof(float), cudaMemcpyDeviceToHost);
    
    mx /= n;
    my /= n;
	
    vector<float> p(2*n);
	int num_bytes = 2*n*sizeof(float);
    float *funcInput = (float *) &(p[0]);
    float *deviceFuncInput;
	float a=0, b=0, c=0;
    float *dev_a, *dev_b, *dev_c;
    
    cudaMalloc((void**)&deviceFuncInput, num_bytes);
    cudaMalloc((void**)&dev_a, sizeof(float));
    cudaMalloc((void**)&dev_b, sizeof(float));
    cudaMalloc((void**)&dev_c, sizeof(float));
    
    cudaMemcpy(deviceFuncInput, funcInput, num_bytes, cudaMemcpyHostToDevice);
    //cerr << "Starting calc_simil_kernel2" << endl;
    calc_simil_kernel2<<<1, 1>>>(pts, ref, deviceFuncInput, mx, my, dev_a, dev_b, dev_c, n);
    //cerr << "Exiting calc_simil_kernel2" << endl;
    
    cudaMemcpy(&a, dev_a, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&b, dev_b, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&c, dev_c, sizeof(float), cudaMemcpyDeviceToHost);

    float scale = sqrt(b*b+c*c), theta = atan2(c,b);
    float sc = scale*cos(theta), ss = scale*sin(theta);
	GpuMat ret(2,3,CV_32F);
    //cerr << "Starting calc_simil_kernel3" << endl;
	calc_simil_kernel3<<<1, 1>>>(ret, sc, ss, mx, my);
    //cerr << "Exiting calc_simil_kernel3" << endl;
    
	return ret;
}

#endif /* WITH_CUDA */
//==============================================================================
void patch_models::write(FileStorage &fs) const {
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
void patch_models::read(const FileNode& node) {
    assert(node.type() == FileNode::MAP);
    node["reference"] >> reference;
    int n; node["n_patches"] >> n; patches.resize(n);
    for(int i = 0; i < n; i++){
        char str[256]; const char* ss;
        sprintf(str,"patch %d",i); ss = str; node[ss] >> patches[i];
    }
}
//==============================================================================
