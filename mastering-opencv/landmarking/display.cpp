/*
 * display.cpp
 *
 * This program takes two arguements when run:
 * The first is the image file to be displayed.
 * The second is the file containing the point data for that image file.
 *
 */

#include <iostream>
#include <fstream>
#include <string>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#define NUM_POINTS 76
#define ARRAY_SIZE (NUM_POINTS*2)

void copyPointsArray(const int *src, int* dest);
void draw();
void onMouse(int event, int x, int y, int, void*);
using namespace cv;
using namespace std;

static Mat img;
static int points[ARRAY_SIZE];

int main(int argc, char **argv)
{
    if (argc < 2){
        cerr << "Incorrect number of arguements." << endl;
        return 1;
    }
    
    img = imread(argv[1]);
    
    namedWindow("Display");
    
    // Reading in data.
    ifstream infile;
    infile.open(argv[2]);
    string tempString;
    
    for (int i = -2; i < ARRAY_SIZE; i++){
        getline(infile, tempString, ';');
        if (i >= 0)
        {
            points[i] = stoi(tempString);
        }
    }
    
    // Drawing
    draw();
    while (1) {
        int c = waitKey(10);
        if (c == 'q') break;
    }
}

void copyPointsArray(const int *src, int* dest)
{
    for (int i = 0; i < ARRAY_SIZE; i++) {
        dest[i] = src[i];
    }
}

void draw()
{
    Mat newImg = img.clone();
    
    for (int i = 0; i < NUM_POINTS; i++) {
        int x = points[2*i];
        int y = points[2*i + 1];
        
        // Draws circle on image.
        circle(newImg, Point(x,y), 3, Scalar(0, 0, 255), -1, 8);
		putText(newImg, to_string(i), Point(x-3, y), FONT_HERSHEY_COMPLEX_SMALL, 0.6, Scalar(0, 0, 0));
		putText(newImg, to_string(i), Point(x-2, y), FONT_HERSHEY_COMPLEX_SMALL, 0.6, Scalar(255, 255, 255));
    }
    
    imshow("Display", newImg);
}
