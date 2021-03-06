//I/O and mat example

#include <iostream>
#include "opencv2/core/core.hpp" // Mat class와 각종 data structure 및 산술 루틴을 포함하는 헤더
#include "opencv2/highgui/highgui.hpp" // GUI 와 관련된 요소를 포함하는 헤더 imshow 등
#include "opencv2/imgproc/imgproc.hpp"  // 각종 이미지 처리 함수를 포함하는 헤더
//위 해더는 기본적으로 include가 필요
using namespace cv;
using namespace std;

int main() {
		Mat src_img = imread("./image/landing.jpg", 0); // 이미지 읽기

		// int flags = IMREAD_COLOR 또는 int flags = 1 -- > 컬러 영상으로 읽음
		// int flags = IMREAD_GRAYSCALE 또는 int flags = 0 -- > 흑백 영상으로 읽음
		// int flags = IMREAD_UNCHANGED 또는 int flags = 1 -- > 원본 영상의 형식대로 읽음

		imshow("Test window", src_img);// 이미지 출력
		waitKey(0); // 키 입력 대기 (0: 키가 입력될 때 까지 프로그램 멈춤
		destroyWindow("Test window"); // 이미지 출력창 종료
		imwrite("langding_gray.png", src_img); // 이미지 쓰기
		return 0;
}