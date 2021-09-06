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

void cvFeatureSIFT(Mat img, string naming);
Mat warpPers(Mat src);
Mat change_img(Mat img, int b);


void cvFeatureSIFT(Mat img, string naming) {

	cout <<naming<< " SIFT starting..." << endl;
	Mat gray;
	cvtColor(img, gray, CV_BGR2GRAY);

	//Ư¡�� ����
	Ptr<cv::SiftFeatureDetector> detector = SiftFeatureDetector::create();
	std::vector<KeyPoint>keyPoints;
	detector->detect(gray, keyPoints);

	//Ư¡�� Ž�� ��� ǥ�� �� ����
	
	Mat result;
	drawKeypoints(img, keyPoints, result);

	string window = "./11w/church_SIFT_" + naming + ".jpg";
	imwrite(window, result);

	cout <<naming << " SIFT finish..." << endl;
}

Mat warpPers(Mat src) {

	cout << "WarpPers starting..." << endl;
	Mat dst;
	float width = src.cols;
	float height = src.rows;
	
	//���� ��ȯ ��ǥ ����
	Point2f src_p[4], dst_p[4];
	src_p[0] = Point2f(0, 0);
	src_p[1] = Point2f(height, 0);
	src_p[2] = Point2f(0, width);
	src_p[3] = Point2f(height, width);

	dst_p[0] = Point2f(0, 0);
	dst_p[1] = Point2f(height, 0);
	dst_p[2] = Point2f(0, width);
	dst_p[3] = Point2f(height*(0.75), width*(0.75));

	//���� ��ȯ �� ����
	Mat pers_mat = getPerspectiveTransform(src_p, dst_p);
	warpPerspective(src, dst, pers_mat, src.size());
	imwrite("./11w/church_w.jpg", dst);

	cout << "WarpPers img save..." << endl;
	return dst;
}

Mat change_img(Mat img, int b) {
	cout << "Change brightness starting..." << endl;
	Mat result = Mat::zeros(img.size(), img.type());

	// ��� ���� 
	//saturate_cast 0~255 ������ ������ �Ǵ��Ͽ� ������������ ����
	for (int y = 0; y < img.rows; y++) {
		for (int x = 0; x < img.cols; x++) {
			for (int c = 0; c < 3; c++) {
				result.at<Vec3b>(y, x)[c] = saturate_cast<uchar>((img.at<Vec3b>(y, x)[c]) + b );
			}
		}
	}

	//����
	imwrite("./11w/chruch_b.jpg", result);
	cout << "Change img save..." << endl;

	return result;
}

int main() {
	Mat img = imread("./11w/church.jpg", 1);
	if (img.empty()) {
		cout << "Empty image!\n";
		exit(-1);
	}
	string name = "original";
	cvFeatureSIFT(img,name);

	int brightness = 40;
	
	Mat img_new = warpPers(change_img(img,brightness));
	name = "new";
	cvFeatureSIFT(img_new, name);

	return 0;
}