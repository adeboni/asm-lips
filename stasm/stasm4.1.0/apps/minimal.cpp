// minimal.cpp: Display the landmarks of a face in an image.
//              This demonstrates stasm_search_single.

#include <stdio.h>
#include <stdlib.h>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "stasm_lib.h"

using namespace cv;

int main()
{
	int foundface;
	float landmarks[2 * stasm_NLANDMARKS];
	Mat_<unsigned char> img, temp;

	VideoCapture cam(0);
	if (!cam.isOpened())
		return -1;

	namedWindow("face tracker");

	while (cam.get(CV_CAP_PROP_POS_AVI_RATIO) < 0.999999) {
		cam >> temp;
		cvtColor(temp, img, CV_RGB2GRAY);
	
		if (!stasm_search_single(&foundface, landmarks,
			(const char*)img.data, img.cols, img.rows, "No path", "../data"))
		{
			printf("Error in stasm_search_single: %s\n", stasm_lasterr());
			break;
		}

		if (foundface) {
			stasm_force_points_into_image(landmarks, img.cols, img.rows);
			for (int i = 0; i < stasm_NLANDMARKS-1; i++) {
				img(cvRound(landmarks[i * 2 + 1]), cvRound(landmarks[i * 2])) = 255;
				line(img, 
					Point(cvRound(landmarks[i * 2]), cvRound(landmarks[i * 2 + 1])),
					Point(cvRound(landmarks[(i + 1) * 2]), cvRound(landmarks[(i + 1) * 2 + 1])),
					Scalar(255, 255, 255), 2, 8
					);
			}
		}
		else {
			printf("No face found\n");
		}

		imshow("face tracker", img);
		int c = waitKey(10);
		if (c == 'q') break;
	}

	destroyWindow("face tracker"); 
	cam.release(); 
	return 0;
}
