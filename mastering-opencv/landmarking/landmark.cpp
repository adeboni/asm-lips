#include <iostream>
#include <stack>
#include <fstream>
#include <limits>
#include <string>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#define NUM_POINTS 76
#define ARRAY_SIZE (NUM_POINTS*2)

void copyPointsArray(const int *src, int* dest);
void undo();
void redo();
void draw();
void onMouse(int event, int x, int y, int, void*);
using namespace cv;
using namespace std;

static Mat img;
static int points[] = {201,348,201,381,202,408,209,435,224,461,241,483,264,498,292,501,319,493,338,470,353,448,363,423,367,395,366,371,357,344,355,316,340,311,325,318,309,328,327,324,342,317,217,328,231,323,250,327,269,333,251,334,233,331,229,345,240,337,262,349,242,352,241,344,346,337,330,330,318,341,334,344,330,336,280,344,278,381,264,399,273,406,293,409,316,399,321,392,304,376,296,342,279,402,310,399,251,431,268,427,284,425,293,425,302,423,316,425,329,426,320,442,309,451,295,454,278,452,263,442,277,440,293,442,313,437,313,429,293,432,277,431,293,436,295,395,234,341,251,343,252,350,235,348,338,333,324,335,326,342,340,340};
stack<int *> changesStack;
stack<int *> undoneChanges;

int main(int argc, char **argv)
{
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
        else if (c == 'k') return 0;
        else if (c == 'z') undo();
        else if (c == 'y') redo();
    }
    
    string inFileName = argv[1];
    string outFileName;
    if (argc >= 2)
        outFileName = argv[2];
    else
    {
        outFileName = inFileName;
        outFileName = outFileName.substr(0, outFileName.length()-3) + "txt";
    }
    
    ofstream myfile;
    myfile.open(outFileName);
    
    // Outputting data.
    int lastSlashIndex = inFileName.find_last_of("/");
    string pureFileName = inFileName.substr(lastSlashIndex+1);
    pureFileName = pureFileName.substr(0, pureFileName.length()-4);
    myfile << "\"" << pureFileName << "\";\n";
    myfile << "0;";
    for (int i = 0; i < ARRAY_SIZE-1; i++) {
        myfile << points[i] << ";";
    }
    myfile << points[NUM_POINTS*2-1] << endl;
}

void copyPointsArray(const int *src, int* dest)
{
    for (int i = 0; i < ARRAY_SIZE; i++) {
        dest[i] = src[i];
    }
}

void undo()
{
    if (!changesStack.empty())
    {
        // Getting previous state.
        int *lastState = changesStack.top();
        changesStack.pop();
        
        // Making copy of the current state.
        int currentStateTemp[NUM_POINTS*2];
        copyPointsArray(points, currentStateTemp);
        
        // Copying last state into the points array.
        copyPointsArray(lastState, points);
        
        // Adding what the current state was into undone stack.
        copyPointsArray(currentStateTemp, lastState);
        undoneChanges.push(lastState);
    }
}

void redo()
{
    if (!undoneChanges.empty())
    {
        // Getting next state.
        int *nextState = undoneChanges.top();
        undoneChanges.pop();
        
        // Making copy of the current state.
        int currentStateTemp[NUM_POINTS*2];
        copyPointsArray(points, currentStateTemp);
        
        // Copying last state into the points array.
        copyPointsArray(nextState, points);
        
        // Adding what the current state was into undone stack.
        copyPointsArray(currentStateTemp, nextState);
        changesStack.push(nextState);
    }
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
    Mat newImg = img.clone();
    
    for (int i = 0; i < NUM_POINTS; i++) {
        int x = points[2*i];
        int y = points[2*i + 1];
        
        // Draws circle on image.
        circle(newImg, Point(x,y), 3, Scalar(0, 0, 255), -1, 8);
		putText(newImg, to_string(i), Point(x-3, y), FONT_HERSHEY_COMPLEX_SMALL, 0.6, Scalar(0, 0, 0));
		putText(newImg, to_string(i), Point(x-2, y), FONT_HERSHEY_COMPLEX_SMALL, 0.6, Scalar(255, 255, 255));
    }
    
    imshow("Landmarks", newImg);
}

void onMouse(int event, int x, int y, int, void*)
{
    static int nearestPointIndex;
    static bool mouseDown = false;
    static bool hasMoved = false;
    
    if (event == CV_EVENT_LBUTTONDOWN)
    {
        mouseDown = true;
        nearestPointIndex = getNearestPoint(x,y);
        
        // Making stack record.
        int *oldPoints = new int [ARRAY_SIZE];
        copyPointsArray(points, oldPoints);
        changesStack.push(oldPoints);
    }
    else if(event == CV_EVENT_LBUTTONUP)
    {
        mouseDown = false;
        
        if (hasMoved)
            hasMoved = false;
        else
        {
            // If curser didn't move then the latest stack add was not needed.
            int *unNeededSave = changesStack.top();
            changesStack.pop();
            delete[] unNeededSave;
        }
    }
    else if (mouseDown && event == CV_EVENT_MOUSEMOVE)
    {
        hasMoved = true;
        points[2*nearestPointIndex] = x;
        points[2*nearestPointIndex + 1] = y;
    }
}
