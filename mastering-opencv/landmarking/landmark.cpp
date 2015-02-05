#include <iostream>
#include <fstream>
#include <limits>
#include <string>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#define NUM_POINTS 76

void draw();
void onMouse(int event, int x, int y, int, void*);
using namespace cv;
using namespace std;

static Mat img;
static int points[] = { 201, 348, 201, 381, 202, 408, 209, 435, 224, 461, 241, 483, 264, 498, 292, 501, 319, 493, 338, 470, 353, 448, 363, 423, 367, 395, 366, 371, 357, 344, 355, 316, 340, 311, 325, 318, 309, 328, 327, 324, 342, 317, 217, 328, 231, 323, 250, 327, 269, 333, 251, 334, 233, 331, 229, 345, 240, 337, 262, 349, 242, 352, 241, 344, 346, 337, 330, 330, 318, 341, 334, 344, 330, 336, 280, 344, 278, 381, 264, 399, 273, 406, 293, 409, 316, 399, 321, 392, 304, 376, 296, 342, 279, 402, 310, 399, 251, 431, 268, 427, 284, 425, 293, 425, 302, 423, 316, 425, 329, 426, 320, 442, 309, 451, 295, 454, 278, 452, 263, 442, 277, 440, 293, 442, 313, 437, 313, 429, 293, 432, 277, 431, 293, 436, 295, 395, 234, 341, 251, 343, 252, 350, 235, 348, 338, 333, 324, 335, 326, 342, 340, 340 };

int main(int argc, char **argv)
{

	img = cv::imread(argv[1]);

	namedWindow("Landmarks");
	setMouseCallback("Landmarks", onMouse, 0);

	while (1) {
		draw();

		int c = waitKey(10);
		if (c == 'q') break;
	}

	for (int i = 0; i < NUM_POINTS * 2 - 1; i++) {
		std::cout << points[i] << ",";
	}

	std::cout << points[NUM_POINTS * 2 - 1] << std::endl;

    if (argc < 1){
        cerr << "Incorrect number of arguements." << endl;
        return 1;
    }
    
    img = imread(argv[1]);
    
    namedWindow("Landmarks");
    setMouseCallback("Landmarks", onMouse, 0 );
    
    while (1) {
        draw();
        
        int c = waitKey(10);
        if (c == 'q') break;
    }
    
    string fileName = argv[1];
    fileName = fileName.substr(0, fileName.length()-3) + "txt";
    
    ofstream myfile;
    myfile.open(fileName);
    
    // Outputting data.
    for (int i = 0; i < NUM_POINTS*2-1; i++) {
        myfile << points[i] << ",";
    }
    
    myfile << points[NUM_POINTS*2-1] << endl;
>>>>>>> 39b620770492441fdfee61fc80b313311141ec47
}

int getNearestPoint(int x, int y)
{
	int best;
	int bestDist = INT_MAX;

	for (int i = 0; i < NUM_POINTS; i++) {
		int nextX = points[2 * i];
		int nextY = points[2 * i + 1];

		int dist = (x - nextX)*(x - nextX) + (y - nextY)*(y - nextY);

		if (dist < bestDist)
		{
			best = i;
			bestDist = dist;
		}
	}

	return best;
}

void draw()
{
<<<<<<< HEAD
	Mat newImg = img.clone();

	for (int i = 0; i < NUM_POINTS; i++) {
		int x = points[2 * i];
		int y = points[2 * i + 1];

		// Draws circle on image.
		circle(newImg, Point(x, y), 3, Scalar(0, 0, 255), -1, 8);
	}

	imshow("Landmarks", newImg);
=======
    Mat newImg = img.clone();
    
    for (int i = 0; i < NUM_POINTS; i++) {
        int x = points[2*i];
        int y = points[2*i + 1];
        
        // Draws circle on image.
        circle(newImg, Point(x,y), 3, Scalar(0, 0, 255), -1, 8);
    }
    
    imshow("Landmarks", newImg); 
>>>>>>> b4e34c17344c58641844fec44608a2a85f7c8d22
}

void onMouse(int event, int x, int y, int, void*)
{
	static int nearestPointIndex;
	static bool mouseDown = false;

	if (event == CV_EVENT_LBUTTONDOWN)
	{
		nearestPointIndex = getNearestPoint(x, y);
		mouseDown = true;
	}
	else if (event == CV_EVENT_LBUTTONUP)
	{

		mouseDown = false;
	}
	else if (mouseDown && event == CV_EVENT_MOUSEMOVE)
	{
		points[2 * nearestPointIndex] = x;
		points[2 * nearestPointIndex + 1] = y;
	}
}
