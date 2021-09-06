#include <iostream>
#include <algorithm>
#include <vector>
#include "opencv2/core/core.hpp" 
#include "opencv2/highgui/highgui.hpp" 
#include "opencv2/imgproc/imgproc.hpp"  
#include "opencv2/imgcodecs/imgcodecs.hpp"
using namespace cv;
using namespace std;


Mat padding(Mat img);
Mat doDft(Mat srcImg);
Mat getMagnitude(Mat complexImg);
Mat myNormalize(Mat src);
Mat getPhase(Mat complexImg);
Mat centralize(Mat complex);
Mat setComplex(Mat magImg, Mat phaImg);
Mat doIdft(Mat complexImg);

Mat doSBF(Mat srcImg);

Mat MyCopy(Mat srcImg);
int  MyKernelConv3x3(uchar* arr, int kernel[][3], int x, int y, int width, int height);
Mat mySobelFilter(Mat srcImg);



//2d discrete fourier transform
Mat doDft(Mat srcImg) {

	Mat floatImg;
	srcImg.convertTo(floatImg, CV_32F);
	// 행렬 타입 변경 8비트 unsigned int -> float (계산이 복잡)

	Mat complexImg;
	dft(floatImg, complexImg, DFT_COMPLEX_OUTPUT); //CCS format으로 저장

	return complexImg;

}

Mat padding(Mat img) {
	Mat padded;                            //expand input image to optimal size
	int m = getOptimalDFTSize(img.rows);
	int n = getOptimalDFTSize(img.cols); // on the border add zero values
	// DFT연산에 최적화된 가장 가까운 사이즈 취득
	copyMakeBorder(img, padded, 0, m - img.rows, 0, n - img.cols, BORDER_CONSTANT, Scalar::all(0));	// DFT연산에 최적화된 사이즈가 되도록 padding
	return padded;
}



//Magnitude 영상 취득 
Mat getMagnitude(Mat complexImg) {
	Mat planes[2];
	split(complexImg, planes); // 실수부 허수부 분리

	Mat magImg;
	magnitude(planes[0], planes[1], magImg);
	magImg += Scalar::all(1);
	log(magImg, magImg);
	//시각화를 하기 위해 +1 -> log
	//magnitude 취득
	//log(1 + sqrt(re(DFT(I))^2 + Im(DFR(I)^2))


	//영상 분포를 맞춰주기 위함 -> nomalize
	return magImg;

}

//Phase 영상 취득
Mat getPhase(Mat complexImg) {
	Mat planes[2];
	split(complexImg, planes); // 실수부 허수부 분리

	Mat PhaImg;
	phase(planes[0], planes[1], PhaImg); //phase 취득

	return PhaImg;
}


//정규화 
Mat myNormalize(Mat src) {
	Mat dst;
	src.copyTo(dst);
	normalize(dst, dst, 0, 255, NORM_MINMAX);
	//각 min/max 대응되도록 
	dst.convertTo(dst, CV_8UC1);

	return dst;
}

//centralize 좌표계 중앙 이동
//저주파가 중앙에 오고 고주파가 모서리로 가도록
Mat centralize(Mat complex) {
	Mat planes[2];
	split(complex, planes);
	//중심
	int cx = planes[0].cols / 2;
	int cy = planes[0].rows / 2;

	//각 4개로 사각형분리 -> planes 0 
	Mat q0Re(planes[0], Rect(0, 0, cx, cy));
	Mat q1Re(planes[0], Rect(cx, 0, cx, cy));
	Mat q2Re(planes[0], Rect(0, cy, cx, cy));
	Mat q3Re(planes[0], Rect(cx, cy, cx, cy));

	//위치 재조정
	Mat tmp;
	q0Re.copyTo(tmp);
	q3Re.copyTo(q0Re);
	tmp.copyTo(q3Re);
	q1Re.copyTo(tmp);
	q2Re.copyTo(q1Re);
	tmp.copyTo(q2Re);

	//각 4개로 사각형분리 -> planes 1
	Mat q0Im(planes[1], Rect(0, 0, cx, cy));
	Mat q1Im(planes[1], Rect(cx, 0, cx, cy));
	Mat q2Im(planes[1], Rect(0, cy, cx, cy));
	Mat q3Im(planes[1], Rect(cx, cy, cx, cy));

	//위치 재조정
	q0Im.copyTo(tmp);
	q3Im.copyTo(q0Im);
	tmp.copyTo(q3Im);
	q1Im.copyTo(tmp);
	q2Im.copyTo(q1Im);
	tmp.copyTo(q2Im);

	//다시 하나로 합침 
	Mat centerComplex;
	merge(planes, 2, centerComplex);

	return centerComplex;
}

//허수부, 실수부 합치기
Mat setComplex(Mat magImg, Mat phaImg) {
	exp(magImg, magImg);
	magImg -= Scalar::all(1);
	//magnitude 계산을 반대로 수행

	Mat planes[2];
	polarToCart(magImg, phaImg, planes[0], planes[1]);
	// 극 좌표계 -> 직교 좌표계 (각도와 크기로부터 2차원 좌표)

	Mat complexImg;
	merge(planes, 2, complexImg);
	//실수부, 허수부 합체

	return complexImg;
}

//inverse discrete fourier Transform
Mat doIdft(Mat complexImg) {
	Mat idftcvt;
	idft(complexImg, idftcvt);
	//IDFT를 이용하여 원봅 영상 취득 
	Mat planes[2];
	split(idftcvt, planes);

	Mat dstImg;
	magnitude(planes[0], planes[1], dstImg);
	normalize(dstImg, dstImg, 255, 0, NORM_MINMAX);
	dstImg.convertTo(dstImg, CV_8UC1);
	//일반 영상의 type 과 표현범위로 변환

	return dstImg;
}



//sobel filtering
Mat doSBF(Mat srcImg) {
	// DFT
	Mat padImg = padding(srcImg);
	Mat complexImg = doDft(srcImg);
	Mat centerComplexImg = centralize(complexImg);
	Mat magImg = getMagnitude(centerComplexImg);
	Mat phaImg = getPhase(centerComplexImg);


	imshow("Test window_ mag", myNormalize(magImg));


	// Filter 
	double minVal, maxVal;
	Point minLoc, maxLoc;
	minMaxLoc(magImg, &minVal, &maxVal, &minLoc, &maxLoc);
	normalize(magImg, magImg, 0, 1, NORM_MINMAX);

	// 십자가 두께
	int thick_w = magImg.cols/3;
	int thick_h = magImg.rows / 3;
	int w = magImg.cols;
	int h = magImg.rows;
	
	//y방향
	Mat maskImg_X = Mat::zeros(magImg.size(), CV_32F);
	maskImg_X(Rect(0, 0, w, h)) = 0.5; // 대각선 방향 wight 0.5
	maskImg_X(Rect(0, thick_h, w, thick_h)) = 1;  //수평 방향 wight 1
	
	//x방향 
	Mat maskImg_Y = Mat::zeros(magImg.size(), CV_32F);
	maskImg_Y(Rect(thick_w, 0, thick_w, h)) = 1; // 수평방향 wight 1

	//Box x,y masking 합치기 
	Mat Mask_img_B = maskImg_X  | maskImg_Y;

	//Center 중심 저주파 wight 0
	Mat maskImg_C= Mat::ones(magImg.size(), CV_32F);
	int hight_l = h / 2 - thick_h/ 2;
	int width_l = w / 2 - thick_w / 2;
	maskImg_C(Rect(width_l, hight_l, thick_w, thick_h)) = 0;

	//전체 마스크 합치기
	Mat Mask_img = Mask_img_B & maskImg_C; 

	imshow("Test window_ Mask", Mask_img);


	//생성한 mask로 magnitude filtering 
	Mat magImg2;
	multiply(magImg, Mask_img, magImg2);

	imshow("Test window_ filter", myNormalize(magImg2));

	//IDFT
	normalize(magImg2, magImg2, (float)minVal, (float)maxVal, NORM_MINMAX);
	Mat complexImg2 = setComplex(magImg2, phaImg);
	Mat dstImg = doIdft(complexImg2);

	return myNormalize(dstImg);

}



int main() {

	Mat src_img = imread("./image/img2.jpg", 0); // 이미지 읽기

	if (!src_img.data)
	{
		cout << "Image not loaded";
		return -1;
	}

	Mat result_img_2 = mySobelFilter(src_img);
	imshow("Test window_result", result_img_2);// 이미지 출력

	Mat bandImg = doSBF(src_img);
	imshow("Test window_ original", src_img);
	imshow("Test window_ Band", bandImg);


	waitKey(0); // 키 입력 대기 (0: 키가 입력될 때 까지 프로그램 멈춤
	destroyAllWindows;

	return 0;



}



//data 기반 화소 접근법
Mat MyCopy(Mat srcImg) {
	int width = srcImg.cols;
	int height = srcImg.rows;

	Mat dstImg(srcImg.size(), CV_8UC1); // 입력영상과 동일한 크기의 Mat 생성
	uchar* srcData = srcImg.data;  //Mat객체의 data를 가르키는 포인터
	uchar* dstData = dstImg.data;

	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			dstData[y*width + x] = srcData[y*width + x];
		} // 화소값을 일일이 읽어와 다른 배열에 저장
	}

	return dstImg;
}

//mask기반 kenrel 계산
int MyKernelConv3x3(uchar* arr, int kernel[][3], int x, int y, int width, int height) {

	int sum = 0;
	int sumkernel = 0;

	//Mask 특정 화소의 모든 이웃화소에 대해 계산 하도록 반복문 구성 
	for (int j = -1; j <= 1; j++) {
		for (int i = -1; i <= 1; i++) {
			//영상 가장자리에서 영상 밖의 화소를 읽지 않도록 하는 조건문
			if ((y + j) >= 0 && (y + j) < height && (x + i) >= 0 && (x + i) < width) {
				sum += arr[(y + j)*width + (x + i)] * kernel[i + 1][j + 1];
				sumkernel += kernel[i + 1][j + 1];
			}
		}
	}
	if (sumkernel != 0) { return sum / sumkernel; }
	else { return sum; }
}

//sobel 필터 구현
Mat mySobelFilter(Mat srcImg) {
	int width = srcImg.cols;
	int height = srcImg.rows;

	//x축 방향
	int kernelX[3][3] = { -1, 0, 1,
									   -2, 0, 2,
										-1, 0, 1 };

	//y축 방향
	int kernelY[3][3] = { -1, -2, -1,
									0,  0,  0,
									1,  2,  1 };

	//이미 mask의 합이 0이 되므로 1로 정규화하는 과정이 따로 필요 없다.

	Mat dstImg(srcImg.size(), CV_8UC1);
	uchar* srcData = srcImg.data;
	uchar* dstData = dstImg.data;

	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			dstData[y*width + x] = (abs(MyKernelConv3x3(srcData, kernelX, x, y, width, height)) +
				abs(MyKernelConv3x3(srcData, kernelY, x, y, width, height))) / 2;
		} // 각각의 엣지 결과의 절대값 합을 2로 나눠 평균화 하여 최종결과 도출
	}

	return dstImg;

}

