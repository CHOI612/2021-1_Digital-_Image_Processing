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

void cvCardPerspective();
Mat cvHarrisCorner(Mat img);
vector<Point2f> cvBlobDetection(Mat img);

void cvCardPerspective() {
	Mat src = imread("./12w/card_per.png", 1);
	Mat dst, matrix;

	//사진에서 입력으로 들어갈 네 꼭지점 자동 탐색
	Mat con = cvHarrisCorner(src);
	vector<Point2f> points = cvBlobDetection(con);

	//변환할 point 저장
	float width = src.cols;
	float height = src.rows;

	Point2f srcTri[4], dstTri[4];
	for (int i = 0; i < 4; i++) {
		cout <<points[i]<<endl;
	}
	srcTri[0] = points[3];
	srcTri[1] = points[2];
	srcTri[2] = points[0];
	srcTri[3] = points[1];
	dstTri[0] = Point2f(width * 0.1f, height * 0.25f);
	dstTri[1] = Point2f(width * 0.9f, height * 0.25f);
	dstTri[2] = Point2f(width * 0.1f, height * 0.75f);
	dstTri[3] = Point2f(width * 0.9f, height * 0.75f);

	matrix = getPerspectiveTransform(srcTri, dstTri);
	warpPerspective(src, dst, matrix, src.size());

	imshow("nonper", src);
	imshow("per", dst);

	waitKey(0);

	destroyAllWindows();
}

Mat cvHarrisCorner(Mat img) {

	resize(img, img, Size(500, 500), 0, 0, INTER_CUBIC);

	Mat gray;
	cvtColor(img, gray, CV_BGR2GRAY);
	
	//배경과 카드 분리후 corner search
	double thresh_gray = 0;
	double maxValue = 255;
	threshold(gray, gray, thresh_gray, maxValue,THRESH_BINARY);


	//<Do harris corner detection>
	Mat harr;
	cornerHarris(gray, harr, 3, 3, 0.05, BORDER_DEFAULT);
	normalize(harr, harr, 0, 255, NORM_MINMAX, CV_32FC1, Mat());

	//<Get abs for Harris visualization>
	Mat harr_abs;
	convertScaleAbs(harr, harr_abs);

	//<print corners>
	int thresh = 125;
	Mat result = img.clone();
	for (int y = 0; y < harr.rows; y += 1) {
		for (int x = 0; x < harr.cols; x += 1) {
			if ((int)harr.at<float>(y, x) > thresh) {
				circle(result, Point(x, y), 7, Scalar(255, 0, 255), 0, 4, 0);
			}
		}
	}
	imshow("result_window", result);
	return result;
}

vector<Point2f>cvBlobDetection(Mat img) {

	//<set Params>
	SimpleBlobDetector::Params params;
	params.minThreshold = 50;
	params.maxThreshold = 100;
	params.filterByArea = true;
	params.minArea = 10;
	params.maxArea = 120;
	params.filterByCircularity = true;
	params.minCircularity = 0.2;
	params.filterByConvexity = true;
	params.minConvexity = 0;


	//<set blob detector>
	Ptr<SimpleBlobDetector> detector = SimpleBlobDetector::create(params);

	//<Detect blobs>
	std::vector<KeyPoint> keypoints;
	detector->detect(img, keypoints);

	//<Draw blobs>
	Mat result;
	drawKeypoints(img, keypoints, result, Scalar(0, 255, 0), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

	imshow("detection_img", result);

	//vector 자료형 변환 keyPoint -> Point2f
	vector<Point2f> corner;
	KeyPoint::convert(keypoints, corner, vector<int>());


	return corner;
}


int main() {

	cvCardPerspective();

	return 0;
}