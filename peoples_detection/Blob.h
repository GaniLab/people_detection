// Blob.h

#ifndef BLOB_H
#define BLOB_H

#include<iostream>
#include<conio.h>

#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include <opencv2/videoio/videoio.hpp>
#include <opencv2/video/video.hpp>

using namespace std;
using namespace cv;


class Blob {
public:
	
	vector<Point> currentContour;

	Rect currentBoundingRect;

	vector<Point> centerPositions;

	double dblCurrentDiagonalSize;
	double dblCurrentAspectRatio;

	bool blnCurrentMatchFoundOrNewBlob;

	bool blnStillBeingTracked;

	int intNumOfConsecutiveFramesWithoutAMatch;

	Point predictedNextPosition;

	// function prototypes 
	Blob(vector<Point> _contour);
	void predictNextPosition(void);

};

#endif    




