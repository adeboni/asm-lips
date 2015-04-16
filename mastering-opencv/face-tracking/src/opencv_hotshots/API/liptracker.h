#ifndef __LIPTRACKER_H__
#define __LIPTRACKER_H__

int getLipContour(char *filepath, float contour[]);

int getNumberOfContourPoints();

void setAcceptanceThreshold(int threshold);

int getAcceptanceThreshold();

void resetTracker();

void initializeTracker(char *inifile);

#endif