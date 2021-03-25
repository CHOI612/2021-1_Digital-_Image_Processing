//과제 2
//앞서 생성한 영상에서 빨강 , 파랑 , 초록 색의 점을 각각 카운트하는 프로그램을 작성하고
//카운트 결과가 실제와 일치하는지 검증할 것


#include <iostream>
#include "opencv2/core/core.hpp" 
#include "opencv2/highgui/highgui.hpp" 
#include "opencv2/imgproc/imgproc.hpp"  

using namespace cv;
using namespace std;

int CountPoint(Mat img, char color);

int main() {
	Mat src_img = imread("result_hw1.png", 1); // 이미지 읽기
	int R, G, B = 0;

	//각 색에서의 점의 갯수
	B = CountPoint(src_img, 'B');
	G = CountPoint(src_img, 'G');
	R = CountPoint(src_img, 'R');

	//결과 출력
	cout << "Counting RGB point " << endl;
	cout << "R : " << R << endl;
	cout << "G : " << G << endl;
	cout << "B : " << B << endl;
	return 0;
}

//원하는 색의 점 갯수 카운트
int CountPoint(Mat img, char color){
	
	int b = 0;
	int g = 0;
	int r = 0;
	int count = 0;

	//검색을 원하는 색
	if (color == 'B') b = 255;
	else if (color == 'G') g = 255;	
	else if (color == 'R') r = 255;

	//이미지 크기
	int x = img.cols;
	int y = img.rows;

	//이미지 채널 수에 따라
	if (img.channels() == 1) {
		for (int i = 0; i < y; i++) {
			for (int j = 0; j < x; j++) {
				if (img.at<uchar>(i, j) == 255) count++;
			}
		}
	}
	else {
		for (int i = 0; i < y; i++) {
			for (int j = 0; j < x; j++) {
				//각각 3채널 모두의 값의 확인 
				if ((img.at<Vec3b>(i, j)[0] == b) &&
					(img.at<Vec3b>(i, j)[1] == g) &&
					(img.at<Vec3b>(i, j)[2] == r)) count++;
			}
		}
	}

	return count;
}



