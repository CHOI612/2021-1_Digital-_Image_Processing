#include <iostream>
#include <algorithm>
#include <vector>
#include "opencv2/core/core.hpp" 
#include "opencv2/highgui/highgui.hpp" 
#include "opencv2/imgproc/imgproc.hpp"  
#include "opencv2/imgcodecs/imgcodecs.hpp"

using namespace cv;
using namespace std;


void myMedian(const Mat& src_img, Mat& dst_img, const Size& kn_size);
void doMedianEx();

void myBilateral(const Mat& src_img, Mat& dst_img, int diameter, double sig_r, double sig_s);
void doBilateralEx();
void bilateral(const Mat& src_img, Mat& dst_img, int c, int r, int diameter, double sig_r, double sig_s);
double guassian(float x, double sigma);
double guassian2D(float c, float r, double sigma);
float distance(int x, int y, int i, int j);
void doCannyEx();

void doMedianEx() {
	cout << " ---- do MdedianEx() ------\n" << endl;

	//입력
	Mat src_img = imread("./image/salt_pepper.png", 0);
	if (!src_img.data) printf("No image data \n");

	// Median 필터링 수행
	Mat dst_img;

	myMedian(src_img, dst_img, Size(3, 3));

	Mat result_img;
	hconcat(src_img, dst_img, result_img);
	imshow("test_window_domedian", result_img);
	waitKey(0);
}



void myMedian(const Mat& src_img, Mat& dst_img, const Size& kn_size) {
	dst_img = Mat::zeros(src_img.size(), CV_8UC1);

	int wd = src_img.cols;
	int hg = src_img.rows;
	int kwd = kn_size.width;
	int khg = kn_size.height;
	int rad_w = kwd / 2;
	int rad_h = khg / 2;

	uchar* src_data = (uchar*)src_img.data;
	uchar* dst_data = (uchar*)dst_img.data;

	//kernel table 동적 할당
	float* table = new float[kwd*khg]();
	float tmp;

	//pixel indexing (가장자리 제외)
	for (int c = rad_w + 1; c < wd - rad_w; c++) {
		for (int r = rad_h + 1; r < hg - rad_h; r++) {
			tmp = 0.f;
			for (int kc = -rad_w; kc <= rad_w; kc++) {
				for (int kr = -rad_h; kr <= rad_h; kr++) {
					tmp = (float)src_data[(r + kr) * wd + (c + kc)];
					table[(kr + rad_h)* kwd + (kc + rad_w)] = tmp;
				}
			}
			sort(table, table + kwd * khg);  //kernel table 정렬
			dst_data[r*wd + c] = (uchar)table[(kwd * khg) / 2]; // 중간값 선택
		}
	}

	delete table; //동적 할당 해제
}

void doBilateralEx() {
	cout << " ---- do BilateralEx() ------\n" << endl;

	//입력
	Mat src_img = imread("./image/rock.png", 0);
	if (!src_img.data) printf("No image data \n");

	// bilateral 필터링 수행
	Mat dst_img;

	myBilateral(src_img, dst_img, 5, 25.0, 50.0);

	//출력
	Mat result_img;
	hconcat(src_img, dst_img, result_img);
	imshow("test_window_doBilateralEx", result_img);
	waitKey(0);
}


void myBilateral(const Mat& src_img, Mat& dst_img, int diameter, double sig_r, double sig_s) {

	dst_img = Mat::zeros(src_img.size(), CV_8UC1);

	Mat guide_img = Mat::zeros(src_img.size(), CV_64F);
	int wd = src_img.cols;
	int hg = src_img.rows;
	int radius = diameter / 2;

	// 픽셀 인덱싱 (가장자리 제외)
	for (int c = radius + 1; c < hg - radius; c++) {
		for (int r = radius + 1; r < wd - radius; r++) {
			//화소별 bilateral 계산 수행
			bilateral(src_img, guide_img, c, r, diameter, sig_r, sig_s); 
		}
	}

	guide_img.convertTo(dst_img, CV_8UC1); // mat type 전환
}


void bilateral(const Mat& src_img, Mat& dst_img, int c, int r, int diameter, double sig_r, double sig_s) {
	
	int radius = diameter / 2;
	
	double gr, gs, wei;
	double tmp = 0.;
	double sum = 0.;


	// 픽셀 인덱싱 
	for (int kc = -radius; kc <= radius; kc++) {
		for (int kr = -radius; kr <= - radius; kr++) {
			//range calc
			gr = guassian((float)src_img.at<uchar>(c + kc, r + kr) - (float)src_img.at<uchar>(c, r), sig_r);
			
			//spatial calc
			gs = guassian(distance(c, r, c + kc, r + kr), sig_s);

			wei = gr * gs;

			tmp += src_img.at<uchar>(c + kc, r + kr) * wei;

			sum += wei;
		}
	}

	dst_img.at<double>(c, r) = tmp / sum; // 정규화



}


double guassian(float x, double sigma) {
	return exp(-(pow(x, 2)) / (2 * pow(sigma, 2))) / (2 * CV_PI * pow(sigma, 2));
}

double guassian2D(float c, float r, double sigma) {
	return exp(-(pow(c, 2) + pow(r,2))/ (2 * pow(sigma, 2))) / (2 * CV_PI * pow(sigma, 2));
}

float distance(int x, int y, int i, int j) {
	return float(sqrt(pow(x - i, 2) + pow(y - j, 2)));
}

void doCannyEx() {
	cout << "-----doCannyEx() ----\n" << endl;
	//입력
	Mat src_img = imread("./image/rock.png", 0);
	if (!src_img.data) printf("No image data \n");

	// bilateral 필터링 수행
	Mat dst_img;

	Canny(src_img, dst_img, 180, 240);

	//출력
	Mat result_img;
	hconcat(src_img, dst_img, result_img);
	imshow("test_window_doCannyEx", result_img);
	waitKey(0);

}


int main() {

	
	doCannyEx();

	return 0;
}


