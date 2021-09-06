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
Mat doFRF(Mat srcImg);


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

//Flickering reducing filtering
Mat doFRF(Mat srcImg) {
	// DFT
	Mat padImg = padding(srcImg);
	Mat complexImg = doDft(srcImg);
	Mat centerComplexImg = centralize(complexImg);
	Mat magImg = getMagnitude(centerComplexImg);
	Mat phaImg = getPhase(centerComplexImg);
	imshow("Test window_ mag", myNormalize(magImg));


	// Filter
	//x축 방향으로 flickering 현상이 심함 -> 수직 방향인 y축 방향의 주파수 영역 지우기
	double minVal, maxVal;
	Point minLoc, maxLoc;
	minMaxLoc(magImg, &minVal, &maxVal, &minLoc, &maxLoc);
	normalize(magImg, magImg, 0, 1, NORM_MINMAX);
	
	//수직 방향으로 삼각형 모양을 만들어 그부분 빼고 filtering 되도록 
	Mat maskImg_tri = Mat::ones(magImg.size(), CV_32F);

	//삼각형 크기 설정 
	int thick = 70;
	int thick_l = 0;
	int pass_point = 5;
	int w = maskImg_tri.cols;
	int h = maskImg_tri.rows;

  // 삼각형 위 1번 
	Point tri_up[1][3]; 

	//setting
	tri_up[0][0] = Point(w / 2 - thick_l, h / 2 - pass_point); // Point(가로 픽셀 위치, 세로 픽셀 위치)

	tri_up[0][1] = Point(w / 2 - thick, 0);

	tri_up[0][2] = Point(w / 2 - thick_l, 0);

	const Point* ppt_up[1] = {tri_up[0] };

	int npt_up[] = { 3 };

	fillPoly(maskImg_tri, ppt_up, npt_up, 1, Scalar(0),8); // masking

	// 삼각형 위 2번 
	Point tri_up_2[1][3];
	//setting
	tri_up_2[0][0] = Point(w / 2 + thick_l, h / 2 - pass_point); // Point(가로 픽셀 위치, 세로 픽셀 위치)

	tri_up_2[0][1] = Point(w / 2 + thick, 0);

	tri_up_2[0][2] = Point(w / 2 + thick_l, 0);

	const Point* ppt_up_2[1] = { tri_up_2[0] };

	int npt_up_2[] = { 3 };

	fillPoly(maskImg_tri, ppt_up_2, npt_up_2, 1, Scalar(0), 8); // masking

	// 삼각형 아래 1번 
	Point tri_down[1][3];
	//setting
	tri_down[0][0] = Point(w / 2 - thick_l, h / 2 + pass_point); // Point(가로 픽셀 위치, 세로 픽셀 위치)

	tri_down[0][1] = Point(w / 2 - thick, h);

	tri_down[0][2] = Point(w / 2 - thick_l, h);

	const Point* ppt_down[1] = { tri_down[0] };

	int npt_down[] = { 3 };

	fillPoly(maskImg_tri, ppt_down, npt_down, 1, Scalar(0), 8); // masking

	// 삼각형 아래 2번 
	Point tri_down_2[1][3];
	//setting
	tri_down_2[0][0] = Point(w / 2 + thick_l, h / 2 + pass_point); // Point(가로 픽셀 위치, 세로 픽셀 위치)

	tri_down_2[0][1] = Point(w / 2 + thick, h);

	tri_down_2[0][2] = Point(w / 2 + thick_l, h);

	const Point* ppt_down_2[1] = { tri_down_2[0] };

	int npt_down_2[] = { 3 };

	fillPoly(maskImg_tri, ppt_down_2, npt_down_2, 1, Scalar(0), 8); // masking

	imshow("Test window_ Mask", maskImg_tri);
	Mat magImg2;
	multiply(magImg, maskImg_tri, magImg2);

	imshow("Test window_ After filtering", myNormalize(magImg2));

	//IDFT
	normalize(magImg2, magImg2, (float)minVal, (float)maxVal, NORM_MINMAX);
	Mat complexImg2 = setComplex(magImg2, phaImg);
	Mat dstImg = doIdft(complexImg2);

	return myNormalize(dstImg);

}



int main() {

	Mat src_img = imread("./image/img3.jpg", 0); // 이미지 읽기

	if (!src_img.data)
	{
		cout << "Image not loaded";
		return -1;
	}

	
	Mat filter_img = doFRF(src_img);
	imshow("Test window_ original", src_img);
	imshow("Test window_ filtering", filter_img);


	waitKey(0); // 키 입력 대기 (0: 키가 입력될 때 까지 프로그램 멈춤
	destroyAllWindows;

	return 0;
}


