#include <iostream>
#include <algorithm>
#include <vector>
#include <ctime>
#include "opencv2/highgui/highgui.hpp" 
#include "opencv2/imgproc/imgproc.hpp"  
#include "opencv2/imgcodecs/imgcodecs.hpp"
#include "opencv2/photo.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/features2d.hpp"

using namespace cv;
using namespace std;

Mat cvHarrisCorner(Mat img);
int cvBlobDetection(Mat img);
void FindNPolyGon(Mat img);

Mat cvHarrisCorner(Mat img) {
	
	resize(img, img, Size(500, 500), 0, 0, INTER_CUBIC);

	Mat gray;
	cvtColor(img, gray, CV_BGR2GRAY);

	//<Do harris corner detection>
	Mat harr;
	cornerHarris(gray, harr, 3, 3, 0.05, BORDER_DEFAULT);
	normalize(harr, harr, 0, 255, NORM_MINMAX, CV_32FC1, Mat());

	//<Get abs for Harris visualization>
	Mat harr_abs;
	convertScaleAbs(harr, harr_abs);

	//<print corners>
	int thresh =125;
	Mat result = img.clone();
	for (int y = 0; y < harr.rows; y += 1) {
		for (int x = 0; x < harr.cols; x += 1) {
			if ((int)harr.at<float>(y, x) > thresh) {
				circle(result, Point(x, y), 7, Scalar(255, 0, 255), 0, 4, 0);
			}
		}
	}

	return result;

}

int cvBlobDetection(Mat img) {

	//<set Params>
	SimpleBlobDetector::Params params;
	params.minThreshold = 10;
	params.maxThreshold = 150;
	params.filterByArea = true;
	params.minArea = 10;
	params.maxArea =100;
	params.filterByCircularity = true;
	params.minCircularity = 0.7;
	params.filterByConvexity = true;
	params.minConvexity = 0.7;
	params.filterByInertia = true;
	params.minInertiaRatio = 0.01;


	//<set blob detector>
	Ptr<SimpleBlobDetector> detector = SimpleBlobDetector::create(params);

	//<Detect blobs>
	std::vector<KeyPoint> keypoints;
	detector->detect(img, keypoints);

	//<Draw blobs>
	Mat result;
	drawKeypoints(img, keypoints, result, Scalar(0, 255,0), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

	imshow("detection_img", result);
	waitKey(0);
	int count = keypoints.size();

	return count;
}

void FindNPolyGon(Mat img) {
	Mat con = cvHarrisCorner(img);
	int cnt = cvBlobDetection(con);
	string polygon = "n";
	if (cnt == 3) polygon = "Triangle";
	else if (cnt == 4) polygon = "Square";
	else if (cnt == 5) polygon = "Pentagon";
	else if (cnt == 6) polygon = "Hexagon";

	cout << "This polygon is " << polygon << endl;
}


int main() {

	
	vector<Mat> images;
	int numImages = 4;
	static const char* filenames[] = { "./11w/obj_4.png", "./11w/obj_3.png", "./11w/obj_2.png", "./11w/obj_1.png" };
	for (int i = 0; i < numImages; i++) {
		Mat im = imread(filenames[i]);
		images.push_back(im);
	}

	for (int i = 0; i < numImages; i++) {
		if (images[i].empty()) {
			cout << "Empty image!\n";
			exit(-1);
		}
		cout <<"[ "<< filenames[i] << " ] image decteting..." << endl;
		FindNPolyGon(images[i]);
	}
	
	/*
	Mat img = imread("./12w/card_per.png", 1);
	if (img.empty()) {
		cout << "Empty image!\n";
		exit(-1);
	}
	FindNPolyGon(img);
	*/

	return 0;
}