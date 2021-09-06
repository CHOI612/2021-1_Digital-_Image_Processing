#include <iostream>
#include <algorithm>
#include <vector>
#include <ctime>
#include "opencv2/core/core.hpp" 
#include "opencv2/highgui/highgui.hpp" 
#include "opencv2/imgproc/imgproc.hpp"  
#include "opencv2/imgcodecs/imgcodecs.hpp"
#include "opencv2/photo.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/features2d.hpp"

using namespace cv;
using namespace std;
void cvBlobDetection();

void cvBlobDetection() {

	Mat img = imread("./11w/coin.png", 1);
	if (img.empty()) {
		cout << "Empty image!\n";
		exit(-1);
	}
	//<set Params>
	SimpleBlobDetector::Params params;
	params.minThreshold = 10;
	params.maxThreshold = 300;
	params.filterByArea = true;
	params.minArea = 100;
	params.maxArea = 7000;
	params.filterByCircularity = true;
	params.minCircularity = 0.75;
	params.filterByConvexity = true;
	params.minConvexity = 0.8;
	params.filterByInertia = true;
	params.minInertiaRatio = 0.01;

	cout << "Blob detecting start" << endl;
	//<set blob detector>
	Ptr<SimpleBlobDetector> detector = SimpleBlobDetector::create(params);

	//<Detect blobs> -> detect coin
	std::vector<KeyPoint> keypoints;
	detector->detect(img, keypoints);
	
	//Count total number of coins
	int count = keypoints.size();
	cout << "Total count of coins : " << count << endl;

	//<Draw blobs>
	Mat result;
	drawKeypoints(img, keypoints, result, Scalar(255, 0, 255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

	imshow("keypoints", result);
	waitKey(0);
	destroyAllWindows();

}

int main() {

	cvBlobDetection();

	return 0;
}