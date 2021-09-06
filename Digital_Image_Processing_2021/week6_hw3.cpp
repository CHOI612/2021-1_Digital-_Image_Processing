#include <iostream>
#include <algorithm>
#include <vector>
#include "opencv2/core/core.hpp" 
#include "opencv2/highgui/highgui.hpp" 
#include "opencv2/imgproc/imgproc.hpp"  
#include "opencv2/imgcodecs/imgcodecs.hpp"

using namespace cv;
using namespace std;

void doBilateralEx(double r_sig, double s_sig, int num);
void myBilateral(const Mat& src_img, Mat& dst_img, int diameter, double sig_r, double sig_s);
void bilateral(const Mat& src_img, Mat& dst_img, int c, int r, int diameter, double sig_r, double sig_s);
double guassian(float x, double sigma);
float distance(int x, int y, int i, int j);




void doBilateralEx(double r_sig, double s_sig,int num) {
	cout << " ---- do BilateralEx() ------\n" << num<<endl;
	cout << "parameter : r_sigma " << r_sig << " s_sigma " << s_sig << endl;

	//입력
	Mat src_img = imread("./image/rock.png", 0);
	if (!src_img.data) printf("No image data \n");

	// bilateral 필터링 수행
	Mat dst_img;

	myBilateral(src_img, dst_img, 7, r_sig, s_sig);

	String fname = "test_window_doBilateralEx" + to_string(num);
	//출력

	imshow(fname, dst_img);
	
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

	// kernel 인덱싱 
	for (int kc = -radius; kc <= radius; kc++) {
		for (int kr = -radius; kr <= radius; kr++) {
			
			//range calc
			gr = guassian((float)src_img.at<uchar>(c + kc, r+ kr) - (float)src_img.at<uchar>(c , r), sig_r);

			//spatial calc
			gs = guassian(distance(c, r, c + kc, r + kr), sig_s);
			
			wei = gr * gs; // 
			tmp += src_img.at<uchar>(c + kc, r + kr) * wei;
			sum += wei;
		}
	}

	dst_img.at<double>(c, r) = tmp / sum; // 정규화
}


double guassian(float x, double sigma) {
	return exp(-(pow(x, 2)) / (2 * pow(sigma, 2))) / (2 * CV_PI * pow(sigma, 2));
}


float distance(int x, int y, int i, int j) {
	return float(sqrt(pow(x - i, 2) + pow(y - j, 2)));
}




int main() {

	double r_sig = 20;
	double s_sig = 10;
	double tmp = 0;
	int num = 1;

	for (int i = 0; i < 3; i++) {
		tmp = r_sig;
		for (int j = 0; j < 3; j++) {
			doBilateralEx(tmp, s_sig, num);
			tmp *= 2;
			num++;
		}
		s_sig *= 10;
	}
	
	waitKey(0);
	destroyAllWindows();

	return 0;
}


