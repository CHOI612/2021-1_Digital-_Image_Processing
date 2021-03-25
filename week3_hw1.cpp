//과제 1 
//주어진 영상 (img1. 에 빨강 , 파랑 , 초록 색의 점을 각각 설정한 개수만큼 무작위
//로 생성하는 프로그램을 작성할 것

#include <iostream>
#include "opencv2/core/core.hpp" 
#include "opencv2/highgui/highgui.hpp" 
#include "opencv2/imgproc/imgproc.hpp"  

using namespace cv;
using namespace std;

void ChangeToPoint_B(Mat img, int num);
void ChangeToPoint_G(Mat img, int num);
void ChangeToPoint_R(Mat img, int num);

int main() {
	Mat src_img = imread("./image/img1.jpg", 1); // 이미지 읽기
	int R, G, B;
	cout << "enter num for RGB point" << endl;
	cout << "R : ";
	cin >> R;
	cout << "G : ";
	cin >> G;
	cout << "B : ";
	cin >> B;

	//전체 픽셀수 오류 
	if ((R + G + B) > (src_img.cols * src_img.rows)) {
		cout << "total number error" << endl;
		return 0;
	}

	//함수를 사용하여 각 색의 점 생성
	ChangeToPoint_B(src_img, B);
	ChangeToPoint_G(src_img, G);
	ChangeToPoint_R(src_img, R);
	
	imshow("Test window", src_img);// 이미지 출력
	waitKey(0); // 키 입력 대기 (0: 키가 입력될 때 까지 프로그램 멈춤
	destroyWindow("Test window"); // 이미지 출력창 종료
	imwrite("result_hw1.png", src_img); // 이미지 쓰기
	return 0;
}

//blue점 생성
void ChangeToPoint_B(Mat img, int num) {
	int count = 0;
	for (int n = 0; n < num; n++) {
		//random한 위치 설정
		int x = rand() % img.cols;
		int y = rand() % img.rows;

		//이미지의 채널의 수에 따라
		if (img.channels() == 1) {
			img.at<uchar>(y, x) = 255;  //단일 채널
		}
		else {
			//cout << y << " " << x << endl;
			if (img.at<Vec3b>(y, x)[0] == 255) n--; //blue 중복확인
			else {  
				img.at<Vec3b>(y, x)[0] = 255; 
				img.at<Vec3b>(y, x)[1] = 0;
				img.at<Vec3b>(y, x)[2] = 0;
			} //Blue
		}
	}
}

//green점 생성
void ChangeToPoint_G(Mat img, int num) {
	for (int n = 0; n < num; n++) {
		//random한 위치 설정
		int x = rand() % img.cols;
		int y = rand() % img.rows;

		//이미지의 채널의 수에 따라
		if (img.channels() == 1) {
			img.at<uchar>(y, x) = 255;  //단일 채널
		}
		else {
			if (img.at<Vec3b>(y, x)[0] == 255) n--; //blue 중복확인
			else if (img.at<Vec3b>(y, x)[1] == 255) n--;  //Green 중복확인
			else{
				img.at<Vec3b>(y, x)[0] = 0;
				img.at<Vec3b>(y, x)[1] = 255;
				img.at<Vec3b>(y, x)[2] = 0;
			} //Green
		
		}
	}
}


//red점 생성
void ChangeToPoint_R(Mat img, int num) {
	for (int n = 0; n < num; n++) {
		//random한 위치 설정
		int x = rand() % img.cols;
		int y = rand() % img.rows;

		//이미지의 채널의 수에 따라
		if (img.channels() == 1) {
			img.at<uchar>(y, x) = 255;  //단일 채널
		}
		else {
			if (img.at<Vec3b>(y, x)[0] == 255) n--; //blue 중복확인
			else if(img.at<Vec3b>(y, x)[1] == 255) n--;  //Green 중복확인
			else if( img.at<Vec3b>(y, x)[2] == 255) n--; //Red 중복확인
			else {

				img.at<Vec3b>(y, x)[0] = 0;
				img.at<Vec3b>(y, x)[1] = 0;
				img.at<Vec3b>(y, x)[2] = 255;
			} //Red
		}
	}
}
