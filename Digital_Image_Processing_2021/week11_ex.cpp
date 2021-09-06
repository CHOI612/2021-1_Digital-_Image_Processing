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

void cvHarrisCorner();
void myHarrisCorner();
void myBlobDetection();
void cvBlobDetection();

void cvHarrisCorner() {
	Mat img = imread("./11w/coin.png");
	if (img.empty()) {
		cout << "Empty image!\n";
		exit(-1);
	}

	resize(img, img, Size(500, 500), 0, 0, INTER_CUBIC);

	Mat gray;
	cvtColor(img, gray, CV_BGR2GRAY);

	//<Do harris corner detection>
	Mat harr;
	cornerHarris(gray, harr, 2, 3, 0.05, BORDER_DEFAULT);
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

	imshow("Source_img_cv", img);
	imshow("Harris_img_cv", harr_abs);
	imshow("Target_img_cv", result);
	waitKey(0);
	destroyAllWindows();

}

void myHarrisCorner() {

	Mat img = imread("./11w/ship.png");
	if (img.empty()) {
		cout << "Empty image!\n";
		exit(-1);
	}

	resize(img, img, Size(500, 500), 0, 0, INTER_CUBIC);

	Mat gray;
	cvtColor(img, gray, CV_BGR2GRAY);

	int height = gray.rows;
	int width = gray.cols;

	//<Get gradient>
	Mat blur;
	GaussianBlur(gray, blur, Size(3, 3), 1);

	Mat gx, gy;
	cv::Sobel(blur, gx, CV_64FC1, 1, 0, 3, 0.4, 128);
	cv::Sobel(blur, gy, CV_64FC1, 0, 1, 3, 0.4, 128);
	double* gx_data = (double*)(gx.data);
	double* gy_data = (double*)(gy.data);

	//<get score>

	Mat harr = Mat(height, width, CV_64FC1, Scalar(0));
	double* harr_data = (double*)(harr.data);
	double k = 0.02;

	for (int y = 1; y < height - 1; y++) {
		for (int x = 1; x < width - 1; x++) {
			int center = y * width + x;

			double dx = 0, dy = 0, dxdy = 0;
			for (int u = -1; u <= 1; u++) {
				for (int v = -1; v <= 1; v++) {
					int cur = center + u * width + v;

					double ix = *(gx_data + cur);
					double iy = *(gy_data + cur);
					dx += ix * ix;
					dy += iy * iy;
					dxdy += ix * iy;
				}
			}
			*(harr_data + center) = dx * dy - dxdy * dxdy - k * (dx + dy)*(dx + dy);
		}
	}

	//<detect corner by score>
	for (int y = 1; y < height - 1; y++) {
		for (int x = 1; x < width - 1; x++) {
			int center = y * width + x;
			double value = *(harr_data + center);

			bool isMaximum = true, isMinimum = true;
			for (int u = -1; u <= 1; u++) {
				for (int v = -1; v <= 1; v++) {
					if (u != 0 || v != 0) {
						int cur = center + u * width + v;

						double neighbor = *(harr_data + cur);
						if (value < neighbor) isMaximum = false;
						else if (value > neighbor) isMinimum = false;
					}
				}
			}
			if (isMaximum == false && isMinimum == false) *(harr_data + center) = 0;
			else *(harr_data + center) = value;
		}
	}

	//<Print corners>
	Mat result = img.clone();
	double thresh = 0.1;
	for (int y = 1; y < height - 1; y++) {
		for (int x = 1; x < width - 1; x++) {
			int center = y * width + x;

			if (*(harr_data + center) > thresh) {
				circle(result, Point(x, y), 7, Scalar(255, 0, 255), 0, 4, 0);
			}

		}
	}

	imshow("Source_img_my", img);
	imshow("Target_img_my", result);
	waitKey(0);
	destroyAllWindows();
}
void myBlobDetection() {
	Mat src, src_gray, dst;
	src = imread("./11w/butt.jpg",1);
	if (src.empty()) {
		cout << "Empty image!\n";
		exit(-1);
	}
	cv::cvtColor(src, src_gray, COLOR_RGB2GRAY);

	int gau_ksize = 11;
	int lap_ksize = 3;
	int lap_scale = 1;
	int lap_delta = 1;

	GaussianBlur(src_gray, src_gray, Size(gau_ksize, gau_ksize), 3, 3, BORDER_DEFAULT);
	Laplacian(src_gray, dst, CV_64F, lap_ksize, lap_scale, lap_delta, BORDER_DEFAULT);

	//Gaussian +Laplacian -> Log

	normalize(-dst, dst, 0, 255, NORM_MINMAX, CV_8U, Mat());

	imwrite("my_log_dst.png", dst);
	imshow("src_img", src);
	imshow("Laplacian img", dst);
	waitKey(0);
	destroyAllWindows();

}

void cvBlobDetection() {

	Mat img = imread("./11w/circle.jpg", 1);
	if (img.empty()) {
		cout << "Empty image!\n";
		exit(-1);
	}
	//<set Params>
	SimpleBlobDetector::Params params;
	params.minThreshold = 10;
	params.maxThreshold = 300;
	params.filterByArea = true;
	params.minArea = 10;
	params.filterByCircularity = true;
	params.minCircularity = 0.895;
	params.filterByConvexity = true;
	params.minConvexity = 0.9;
	params.filterByInertia = true;
	params.minInertiaRatio = 0.01;


	//<set blob detector>
	Ptr<SimpleBlobDetector> detector = SimpleBlobDetector::create(params);

	//<Detect blobs>
	std::vector<KeyPoint> keypoints;
	detector->detect(img, keypoints);

	//<Draw blobs>
	Mat result;
	drawKeypoints(img, keypoints, result, Scalar(255, 0, 255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

	imshow("keypoints", result);
	waitKey(0);
	destroyAllWindows();

}
int main() {

	cvHarrisCorner();
	//myHarrisCorner();

	return 0;
}