#include "liptracker.h"
#include "../ft.hpp"
#include <opencv2/highgui/highgui.hpp>

#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT extern "C"
#endif

face_tracker *tracker = NULL;
face_tracker_params p; 
Mat im;

EXPORT int getLipContour(char *filepath, float contour[]) {
	im = imread(filepath, CV_LOAD_IMAGE_GRAYSCALE); 
	while (!tracker->track(im, p));
	for (int i = 0; i < tracker->points.size(); i++) {
		contour[i*2] = tracker->points[i].x;
		contour[i*2+1] = tracker->points[i].y;
	}
}

EXPORT int getNumberOfContourPoints() {
	return tracker->pmodel.n_patches();
}

EXPORT void setAcceptanceThreshold(int threshold) {
	p.itol = threshold;
}

EXPORT int getAcceptanceThreshold() {
	return p.itol;
}

EXPORT void resetTracker() {
	if (tracker != NULL) {
		tracker->reset;
	}
}

EXPORT void initializeTracker(char *inifile) {
	tracker = new load_ft<face_tracker>(inifile);

	//create tracker parameters
	p.ssize.resize(3);
	p.ssize[0] = Size(21,21);
	p.ssize[1] = Size(11,11);
	p.ssize[2] = Size(5,5);
}