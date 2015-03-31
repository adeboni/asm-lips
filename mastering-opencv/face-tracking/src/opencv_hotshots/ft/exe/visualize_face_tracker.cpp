#include "opencv_hotshots/ft/ft.hpp"
#include <opencv2/highgui/highgui.hpp>
#ifdef WITH_CUDA
#include <opencv2/gpu/gpu.hpp>
#endif
#include <iostream>
#define fl at<float>
void draw_string(Mat img, const string text, const int height = 0) {
	Size size = getTextSize(text,FONT_HERSHEY_COMPLEX,0.6f,1,NULL);
	putText(img,text,Point(0,size.height + height),FONT_HERSHEY_COMPLEX,0.6f,Scalar::all(0),1,CV_AA);
	putText(img,text,Point(1,size.height+1 + height),FONT_HERSHEY_COMPLEX,0.6f,Scalar::all(255),1,CV_AA);
}
//==============================================================================
bool paused = false;

int main(int argc,char** argv) {
	if (argc < 2) { 
		cout << "usage: ./visualise_face_tracker tracker [video_file]" << endl; 
		return 0;
	}
	
	//load detector model
	face_tracker tracker = load_ft<face_tracker>(argv[1]);

	//create tracker parameters
	face_tracker_params p; 
	p.ssize.resize(3);
	p.ssize[0] = Size(21,21);
	p.ssize[1] = Size(11,11);
	p.ssize[2] = Size(5,5);
	
#ifdef WITH_CUDA
	cout << gpu::getCudaEnabledDeviceCount() << " devices found" << endl;
	gpu::printShortCudaDeviceInfo(cv::gpu::getDevice());
	gpu::setDevice(0);
#endif

	//open video stream
	VideoCapture cam; 
	if (argc > 2) cam.open(argv[2]); 
	else cam.open(0);
	if (!cam.isOpened()) {
		cout << "Failed opening video stream." << endl; 
		return 0;
	}
	//detect until user quits
	namedWindow("lip tracker");
	while (cam.isOpened()) {
		if (!paused) {
			Mat im; cam >> im;
			if (tracker.track(im,p)) tracker.draw(im);
			draw_string(im,"d - redetection");
			draw_string(im,"p - pause", 20);
			tracker.timer.display_fps(im, Point(1,im.rows-1));
			imshow("lip tracker", im);
		}
		
		int c = waitKey(10);
		if (c == 'q') break;
		else if (c == 'd') {
			tracker.reset();
			paused = false;	
		}
		else if (c == 'p') {
			paused = !paused;
			if (!paused) tracker.reset();
		}
	}
	destroyWindow("lip tracker"); 
	cam.release(); 
	return 0;
}
//==============================================================================
