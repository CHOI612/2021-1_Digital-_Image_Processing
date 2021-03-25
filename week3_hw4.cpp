//과제 4
//주어진 영상 img3. jpg, img4. jpg, img5. jpg) 을 이용해 다음의 영상을 완성할 것

#include <iostream>
#include "opencv2/core/core.hpp" 
#include "opencv2/highgui/highgui.hpp" 
#include "opencv2/imgproc/imgproc.hpp"  

using namespace cv;
using namespace std;

int main() {
	Mat rocket_img = imread("./image/img3.jpg", 1); // 이미지 읽기
	Mat effect_img = imread("./image/img4.jpg", 1); // 이미지 읽기
	Mat text_img = imread("./image/img5.jpg", 1); // 이미지 읽기

	//사이즈 확인
	cout << "image size : " << rocket_img.rows << "*" << rocket_img.cols << endl;
	cout << " text size : " << text_img.rows << "*" << text_img.cols << endl;
	
	//두 사진 크기 맞추기
	resize(effect_img, effect_img, Size(rocket_img.cols, rocket_img.rows));

	// 비네팅 이미지 합성
	Mat result_img;

	// 기존 비네팅 사진더 어둡게 하여 합성영상에서는 더 밝기가 밝도록
	subtract(effect_img, Scalar(30, 30, 30), effect_img);
	subtract(rocket_img, effect_img, result_img);
	

	//로고 배경 지우고 삽입 
	
	//로고 위치 잡기 
	Mat img_ROI(result_img, Rect(330, 350, text_img.cols, text_img.rows));
	
	//이진화를 위해 color 에서 grayscale 영상으로 변환
	Mat gray_img;
	cvtColor(text_img, gray_img, CV_BGR2GRAY); 
	
	Mat binary_img;
	//임계값 지정 이진화
	threshold(gray_img, binary_img, 190, 255, THRESH_BINARY);


	//원하는 위치에 로고 이미지 자리 따오기 
	Mat logo_space;
	bitwise_and(img_ROI, img_ROI, logo_space, binary_img);


	//배경값을 흑백전환
	bitwise_not(binary_img, binary_img);
	
	//로고 누끼 저장 장하기
	Mat logo_img;
	bitwise_and(text_img, text_img, logo_img, binary_img);

	//원하는 위치에 이미지 자리와 로고누끼로 합쳐 결과이미지에 저장하기
	//add(logo_space, logo_img, img_ROI); 
	bitwise_or(logo_space, logo_img, img_ROI); // or 연산도 가능

	imshow("Test window_result", result_img);// 이미지 출력

	waitKey(0); // 키 입력 대기 (0: 키가 입력될 때 까지 프로그램 멈춤
	destroyAllWindows;
	imwrite("result_hw4.png", result_img); // 이미지 쓰기


	return 0;
}