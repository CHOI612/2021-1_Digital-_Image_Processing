//과제 3
//주어진 영상을 이용해 (img2.jpg) 다음과 같은 두 영상을 생성하는 프로그램을 작성하고
//픽셀 값 접근을 이용 히스토그램 일치 여부를 확인 및 그러한 결과가 나온 이유를 분석할 것


#include <iostream>
#include "opencv2/core/core.hpp" 
#include "opencv2/highgui/highgui.hpp" 
#include "opencv2/imgproc/imgproc.hpp"  

using namespace cv;
using namespace std;

void GettingDark_up(Mat img);
void GettingDark_down(Mat img);
Mat getHistogram(Mat& src);

int main() {
	Mat main_img = imread("./image/img2.jpg", 0); // 이미지 읽기
	Mat up_img = imread("./image/img2.jpg", 0); // 이미지 읽기
	Mat down_img = imread("./image/img2.jpg", 0); // 이미지 읽기

	//각각의 영상 생성
	GettingDark_up(up_img);
	GettingDark_down(down_img);

	
	imshow("Test window_up", up_img);// 이미지 출력
	imshow("Test window_down", down_img);// 이미지 출력
	imshow("histogram", getHistogram(main_img)); //히스토그램 출력
	imshow("histogram_up", getHistogram(up_img)); //히스토그램 출력
	imshow("histogram_down", getHistogram(down_img)); //히스토그램 출력
	waitKey(0); // 키 입력 대기 (0: 키가 입력될 때 까지 프로그램 멈춤
	destroyAllWindows;
	imwrite("result_hw3_1.png", up_img); // 이미지 쓰기
	imwrite("result_hw3_2.png", down_img); // 이미지 쓰기
	
	return 0;
}

//위로 갈수록 어두워 지도록 
void GettingDark_up(Mat img) {

	//이미지 크기
	int x = img.cols;
	int y = img.rows;
	//기본 어두워지는 밝기값
	int Darkness = 180;

	//이미지 채널 수에 따라
	if (img.channels() == 1) {
		for (int i = 0; i < y; i++) {
			for (int j = 0; j < x; j++) {

				//overflow 주의
				if ((int)img.at<uchar>(i, j) < Darkness) img.at<uchar>(i, j) = 0;
				else img.at<uchar>(i, j) = img.at<uchar>(i, j) - Darkness;
			}
			if (i % 3 == 0) Darkness--;
			if (Darkness < 0) Darkness = 0;
		}
	}
	else {
		for (int i = 0; i < y; i++) {
			for (int j = 0; j < x; j++) {
			
				//overflow 주의 -> 넘어가면 0->255로 값이 매우 튄다 
				if ((int)img.at<Vec3b>(i, j)[0] < Darkness)img.at<Vec3b>(i, j)[0] = 0;
				else img.at<Vec3b>(i, j)[0] = img.at<Vec3b>(i, j)[0]  - Darkness;

				if ((int)img.at<Vec3b>(i, j)[1] < Darkness)img.at<Vec3b>(i, j)[1] = 0;
				else img.at<Vec3b>(i, j)[1] = img.at<Vec3b>(i, j) [1] - Darkness;

				if ((int)img.at<Vec3b>(i, j)[2] < Darkness)img.at<Vec3b>(i, j)[2] = 0;
				else img.at<Vec3b>(i, j)[2] = img.at<Vec3b>(i, j)[2] - Darkness;
	
			}
			//아래로 갈수록 덜 어두워 지도록
			if (i % 3 == 0) Darkness--;
			if (Darkness < 0) Darkness = 0;
		}
	}

}

//아래로 갈수록 어두워 지도록 
void GettingDark_down(Mat img) {
	//이미지 크기
	int x = img.cols;
	int y = img.rows;

	//기본 어두워지는 밝기값
	int Darkness = 50;

	//이미지 채널 수에 따라
	if (img.channels() == 1) {
		for (int i = 0; i < y; i++) {
			for (int j = 0; j < x; j++) {
				//overflow 주의 -> 넘어가면 0->255로 값이 매우 튄다 
				if ((int)img.at<uchar>(i, j) < Darkness) img.at<uchar>(i, j) = 0;
				else img.at<uchar>(i, j) = img.at<uchar>(i, j) - Darkness;
			}
			if (i % 3 == 0) Darkness++;
			if (Darkness >255) Darkness = 255;
		}
	}
	else {
		for (int i = 0; i < y; i++) {
			for (int j = 0; j < x; j++) {

				//overflow 주의 
				if ((int)img.at<Vec3b>(i, j)[0] < Darkness)img.at<Vec3b>(i, j)[0] = 0;
				else img.at<Vec3b>(i, j)[0] = img.at<Vec3b>(i, j)[0] - Darkness;

				if ((int)img.at<Vec3b>(i, j)[1] < Darkness)img.at<Vec3b>(i, j)[1] = 0;
				else img.at<Vec3b>(i, j)[1] = img.at<Vec3b>(i, j)[1] - Darkness;

				if ((int)img.at<Vec3b>(i, j)[2] < Darkness)img.at<Vec3b>(i, j)[2] = 0;
				else img.at<Vec3b>(i, j)[2] = img.at<Vec3b>(i, j)[2] - Darkness;

			}
			//아래로 갈수록 점차 어두워지도록
			if (i % 3 == 0) Darkness++;
			if (Darkness > 255) Darkness = 255;
		}
	}

}

//히스토그램 그래프 이미지 생성 함수
Mat getHistogram(Mat& src) {
	Mat histogram;
	// gray_scale 에서만 작동 
	const int* channel_numberes = { 0 };
	float channel_range[] = { 0.0, 255.0 };
	const float* channel_ranges = channel_range;
	int number_bins = 255; // 최대값 

	//픽셀값 분포 히스토그램 계산
	calcHist(&src, 1, channel_numberes, Mat(), histogram, 1, &number_bins, &channel_ranges);

	//히스토그램 그래프 이미지 크기
	int hist_w = 512;
	int hist_h = 400;
	int bin_w = cvRound((double)hist_w / number_bins);

	// histogram  정규화
	normalize(histogram, histogram, 0, histogram.rows, NORM_MINMAX, -1, Mat());

	//값과 값을 잇는 선과 그리는 방식의 plot 그래프 이미지
	Mat histImage(hist_h, hist_w, CV_8UC1, Scalar(0, 0, 0));
	for (int i = 1; i < number_bins; i++) {
		line(histImage, Point(bin_w*(i - 1), hist_h - cvRound(histogram.at<float>(i - 1))),
			Point(bin_w*(i), hist_h - cvRound(histogram.at<float>(i))),
			Scalar(255, 0, 0), 2, 8, 0); 
	}

	return histImage;
}