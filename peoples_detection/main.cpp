// main.cpp

#include<iostream>
#include<conio.h>           

#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include <opencv2/videoio/videoio.hpp>
#include <opencv2/video/video.hpp>

#include "Blob.h"

#define SHOW_STEPS            // un-comment or comment this line to show steps or not

using namespace std;
using namespace cv;

// global variables ///////////////////////////////////////////////////////////////////////////////
const Scalar SCALAR_BLACK = Scalar(0.0, 0.0, 0.0);
const Scalar SCALAR_WHITE = Scalar(255.0, 255.0, 255.0);
const Scalar SCALAR_YELLOW = Scalar(0.0, 255.0, 255.0);
const Scalar SCALAR_GREEN = Scalar(0.0, 200.0, 0.0);
const Scalar SCALAR_RED = Scalar(0.0, 0.0, 255.0);

// function prototypes ////////////////////////////////////////////////////////////////////////////
void matchCurrentFrameBlobsToExistingBlobs(vector<Blob> &existingBlobs, vector<Blob> &currentFrameBlobs);
void addBlobToExistingBlobs(Blob &currentFrameBlob, vector<Blob> &existingBlobs, int &intIndex);
void addNewBlob(Blob &currentFrameBlob, vector<Blob> &existingBlobs);
double distanceBetweenPoints(Point point1, Point point2);
void drawAndShowContours(Size imageSize, vector<vector<Point> > contours, string strImageName);
void drawAndShowContours(Size imageSize, vector<Blob> blobs, string strImageName);
void drawBlobInfoOnImage(vector<Blob> &blobs, Mat &imgFrame2Copy);

///////////////////////////////////////////////////////////////////////////////////////////////////
int main(void) {

	VideoCapture capVideo;

	Mat imgFrame1;
	Mat imgFrame2;

	vector<Blob> blobs;

	capVideo.open("peoples.avi");

	// check the video
	if (!capVideo.isOpened()) {                                                 
		cout << "error reading video file" << std::endl << std::endl;      // show error message
		_getch();                    
		return(0);                                                             
	}

	if (capVideo.get(CAP_PROP_FRAME_COUNT) < 2) {
		cout << "error: video file must have at least two frames";
		_getch();
		return(0);
	}
	
	capVideo.read(imgFrame1);
	capVideo.read(imgFrame2);

	char chCheckForEscKey = 0;

	bool blnFirstFrame = true;

	int frameCount = 2;

	while (capVideo.isOpened() && chCheckForEscKey != 27) {

		vector<Blob> currentFrameBlobs;

		Mat imgFrame1Copy = imgFrame1.clone();
		Mat imgFrame2Copy = imgFrame2.clone();

		Mat imgDifference;
		Mat imgThresh;

		cvtColor(imgFrame1Copy, imgFrame1Copy, COLOR_BGR2GRAY);
		cvtColor(imgFrame2Copy, imgFrame2Copy, COLOR_BGR2GRAY);

		GaussianBlur(imgFrame1Copy, imgFrame1Copy, Size(5, 5), 0);
		GaussianBlur(imgFrame2Copy, imgFrame2Copy, Size(5, 5), 0);

		absdiff(imgFrame1Copy, imgFrame2Copy, imgDifference);
		
		threshold(imgDifference, imgThresh, 30, 255.0, THRESH_BINARY);

		imshow("imgThresh", imgThresh);

		Mat structuringElement3x3 = getStructuringElement(MORPH_RECT, Size(3, 3));
		Mat structuringElement5x5 = getStructuringElement(MORPH_RECT, Size(5, 5));
		Mat structuringElement7x7 = getStructuringElement(MORPH_RECT, Size(7, 7));
		Mat structuringElement9x9 = getStructuringElement(MORPH_RECT, Size(9, 9));

		/*
		cv::dilate(imgThresh, imgThresh, structuringElement7x7);
		cv::erode(imgThresh, imgThresh, structuringElement3x3);
		*/

		dilate(imgThresh, imgThresh, structuringElement5x5);
		dilate(imgThresh, imgThresh, structuringElement5x5);
		erode(imgThresh, imgThresh, structuringElement5x5);


		Mat imgThreshCopy = imgThresh.clone();

		vector<vector<Point> > contours;

		findContours(imgThreshCopy, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

		drawAndShowContours(imgThresh.size(), contours, "imgContours");

		vector<vector<Point> > convexHulls(contours.size());

		for (unsigned int i = 0; i < contours.size(); i++) {
			convexHull(contours[i], convexHulls[i]);
		}

		drawAndShowContours(imgThresh.size(), convexHulls, "imgConvexHulls");

		for (auto &convexHull : convexHulls) {
			Blob possibleBlob(convexHull);

			if (possibleBlob.currentBoundingRect.area() > 100 &&
				possibleBlob.dblCurrentAspectRatio >= 0.2 &&
				possibleBlob.dblCurrentAspectRatio <= 1.25 &&
				possibleBlob.currentBoundingRect.width > 20 &&
				possibleBlob.currentBoundingRect.height > 20 &&
				possibleBlob.dblCurrentDiagonalSize > 30.0 &&
				(contourArea(possibleBlob.currentContour) / (double)possibleBlob.currentBoundingRect.area()) > 0.40) {
				currentFrameBlobs.push_back(possibleBlob);
			}
		}

		drawAndShowContours(imgThresh.size(), currentFrameBlobs, "imgCurrentFrameBlobs");

		if (blnFirstFrame == true) {
			for (auto &currentFrameBlob : currentFrameBlobs) {
				blobs.push_back(currentFrameBlob);
			}
		}
		else {
			matchCurrentFrameBlobsToExistingBlobs(blobs, currentFrameBlobs);
		}

		drawAndShowContours(imgThresh.size(), blobs, "imgBlobs");

		imgFrame2Copy = imgFrame2.clone();          // get another copy of frame 2 since we changed the previous frame 2 copy in the processing above

		drawBlobInfoOnImage(blobs, imgFrame2Copy);

		imshow("imgFrame2Copy", imgFrame2Copy);

		//cv::waitKey(0);                 // uncomment this line to go frame by frame for debugging

				// now we prepare for the next iteration

		currentFrameBlobs.clear();

		imgFrame1 = imgFrame2.clone();           // move frame 1 up to where frame 2 is

		if ((capVideo.get(CAP_PROP_POS_FRAMES) + 1) < capVideo.get(CAP_PROP_FRAME_COUNT)) {
			capVideo.read(imgFrame2);
		}
		else {
			cout << "end of video\n";
			break;
		}

		blnFirstFrame = false;
		frameCount++;
		chCheckForEscKey = waitKey(1);
	}

	if (chCheckForEscKey != 27) {               // if the user did not press esc (i.e. we reached the end of the video)
		waitKey(0);                         // hold the windows open to allow the "end of video" message to show
	}
	// note that if the user did press esc, we don't need to hold the windows open, we can simply let the program end which will close the windows

	return(0);
}

///////////////////////////////////////////////////////////////////////////////////////////////////
void matchCurrentFrameBlobsToExistingBlobs(vector<Blob> &existingBlobs, vector<Blob> &currentFrameBlobs) 
{

	for (auto &existingBlob : existingBlobs) {

		existingBlob.blnCurrentMatchFoundOrNewBlob = false;

		existingBlob.predictNextPosition();
	}

	for (auto &currentFrameBlob : currentFrameBlobs) {

		int intIndexOfLeastDistance = 0;
		double dblLeastDistance = 100000.0;

		for (unsigned int i = 0; i < existingBlobs.size(); i++) {
			if (existingBlobs[i].blnStillBeingTracked == true) {
				double dblDistance = distanceBetweenPoints(currentFrameBlob.centerPositions.back(), existingBlobs[i].predictedNextPosition);

				if (dblDistance < dblLeastDistance) {
					dblLeastDistance = dblDistance;
					intIndexOfLeastDistance = i;
				}
			}
		}

		if (dblLeastDistance < currentFrameBlob.dblCurrentDiagonalSize * 1.15) {
			addBlobToExistingBlobs(currentFrameBlob, existingBlobs, intIndexOfLeastDistance);
		}
		else {
			addNewBlob(currentFrameBlob, existingBlobs);
		}

	}

	for (auto &existingBlob : existingBlobs) {

		if (existingBlob.blnCurrentMatchFoundOrNewBlob == false) {
			existingBlob.intNumOfConsecutiveFramesWithoutAMatch++;
		}

		if (existingBlob.intNumOfConsecutiveFramesWithoutAMatch >= 5) {
			existingBlob.blnStillBeingTracked = false;
		}

	}

}

///////////////////////////////////////////////////////////////////////////////////////////////////
void addBlobToExistingBlobs(Blob &currentFrameBlob, vector<Blob> &existingBlobs, int &intIndex) {

	existingBlobs[intIndex].currentContour = currentFrameBlob.currentContour;
	existingBlobs[intIndex].currentBoundingRect = currentFrameBlob.currentBoundingRect;

	existingBlobs[intIndex].centerPositions.push_back(currentFrameBlob.centerPositions.back());

	existingBlobs[intIndex].dblCurrentDiagonalSize = currentFrameBlob.dblCurrentDiagonalSize;
	existingBlobs[intIndex].dblCurrentAspectRatio = currentFrameBlob.dblCurrentAspectRatio;

	existingBlobs[intIndex].blnStillBeingTracked = true;
	existingBlobs[intIndex].blnCurrentMatchFoundOrNewBlob = true;
}


void addNewBlob(Blob &currentFrameBlob, vector<Blob> &existingBlobs) 
{

	currentFrameBlob.blnCurrentMatchFoundOrNewBlob = true;

	existingBlobs.push_back(currentFrameBlob);
}


double distanceBetweenPoints(Point point1, Point point2) 
{

	int intX = abs(point1.x - point2.x);
	int intY = abs(point1.y - point2.y);

	return(sqrt(pow(intX, 2) + pow(intY, 2)));
}


void drawAndShowContours(Size imageSize, vector<vector<Point> > contours, string strImageName) 
{
	Mat image(imageSize, CV_8UC3, SCALAR_BLACK);

	drawContours(image, contours, -1, SCALAR_WHITE, -1);

	imshow(strImageName, image);
}


void drawAndShowContours(Size imageSize, vector<Blob> blobs, string strImageName) 
{

	Mat image(imageSize, CV_8UC3, SCALAR_BLACK);

	vector<vector<Point> > contours;

	for (auto &blob : blobs) {
		if (blob.blnStillBeingTracked == true) {
			contours.push_back(blob.currentContour);
		}
	}

	drawContours(image, contours, -1, SCALAR_WHITE, -1);

	imshow(strImageName, image);
}


void drawBlobInfoOnImage(vector<Blob> &blobs, Mat &imgFrame2Copy) 
{

	for (unsigned int i = 0; i < blobs.size(); i++) {

		if (blobs[i].blnStillBeingTracked == true) {
			rectangle(imgFrame2Copy, blobs[i].currentBoundingRect, SCALAR_RED, 2);

			int intFontFace = FONT_HERSHEY_SIMPLEX;
			double dblFontScale = blobs[i].dblCurrentDiagonalSize / 60.0;
			int intFontThickness = (int)round(dblFontScale * 1.0);

			putText(imgFrame2Copy, std::to_string(i), blobs[i].centerPositions.back(), intFontFace, dblFontScale, SCALAR_GREEN, intFontThickness);
		}
	}
}








