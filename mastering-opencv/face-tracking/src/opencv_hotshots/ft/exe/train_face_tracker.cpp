#include "opencv_hotshots/ft/ft.hpp"
#include <iostream>
#define fl at<float>

int main(int argc, char** argv) {
	if (argc < 5) { 
		cout << "usage: ./train_face_tracker shape_model_file patch_models_file face_detector_file face_tracker_file" << endl; 
		return 0;
	}

	//create face tracker model
	face_tracker tracker;
	tracker.smodel = load_ft<shape_model>(argv[1]);
	tracker.pmodel = load_ft<patch_models>(argv[2]);
	tracker.detector = load_ft<face_detector>(argv[3]);

	//save face tracker
	save_ft<face_tracker>(argv[4], tracker); 
	return 0;
}
