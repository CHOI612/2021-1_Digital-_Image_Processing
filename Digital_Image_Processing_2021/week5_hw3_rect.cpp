#include <iostream>
#include <algorithm>
#include <vector>
#include "opencv2/core/core.hpp" 
#include "opencv2/highgui/highgui.hpp" 
#include "opencv2/imgproc/imgproc.hpp"  
#include "opencv2/imgcodecs/imgcodecs.hpp"
using namespace cv;
using namespace std;

/*
Mat padding(Mat img);
Mat doDft(Mat srcImg);
Mat getMagnitude(Mat complexImg);
Mat myNormalize(Mat src);
Mat getPhase(Mat complexImg);
Mat centralize(Mat complex);
Mat setComplex(Mat magImg, Mat phaImg);
Mat doIdft(Mat complexImg);
Mat doHPF(Mat srcImg);
Mat doLPF(Mat srcImg);
*/



//2d discrete fourier transform
Mat doDft(Mat srcImg) {

	Mat floatImg;
	srcImg.convertTo(floatImg, CV_32F);
	// ��� Ÿ�� ���� 8��Ʈ unsigned int -> float (����� ����)

	Mat complexImg;
	dft(floatImg, complexImg, DFT_COMPLEX_OUTPUT); //CCS format���� ����

	return complexImg;

}

Mat padding(Mat img) {
	Mat padded;                            //expand input image to optimal size
	int m = getOptimalDFTSize(img.rows);
	int n = getOptimalDFTSize(img.cols); // on the border add zero values
	// DFT���꿡 ����ȭ�� ���� ����� ������ ���
	copyMakeBorder(img, padded, 0, m - img.rows, 0, n - img.cols, BORDER_CONSTANT, Scalar::all(0));	// DFT���꿡 ����ȭ�� ����� �ǵ��� padding
	return padded;
}



//Magnitude ���� ��� 
Mat getMagnitude(Mat complexImg) {
	Mat planes[2];
	split(complexImg, planes); // �Ǽ��� ����� �и�

	Mat magImg;
	magnitude(planes[0], planes[1], magImg);
	magImg += Scalar::all(1);
	log(magImg, magImg);
	//�ð�ȭ�� �ϱ� ���� +1 -> log
	//magnitude ���
	//log(1 + sqrt(re(DFT(I))^2 + Im(DFR(I)^2))


	//���� ������ �����ֱ� ���� -> nomalize
	return magImg;

}

//Phase ���� ���
Mat getPhase(Mat complexImg) {
	Mat planes[2];
	split(complexImg, planes); // �Ǽ��� ����� �и�

	Mat PhaImg;
	phase(planes[0], planes[1], PhaImg); //phase ���

	return PhaImg;
}


//����ȭ 
Mat myNormalize(Mat src) {
	Mat dst;
	src.copyTo(dst);
	normalize(dst, dst, 0, 255, NORM_MINMAX);
	//�� min/max �����ǵ��� 
	dst.convertTo(dst, CV_8UC1);

	return dst;
}

//centralize ��ǥ�� �߾� �̵�
//�����İ� �߾ӿ� ���� �����İ� �𼭸��� ������
Mat centralize(Mat complex) {
	Mat planes[2];
	split(complex, planes);
	//�߽�
	int cx = planes[0].cols / 2;
	int cy = planes[0].rows / 2;

	//�� 4���� �簢���и� -> planes 0 
	Mat q0Re(planes[0], Rect(0, 0, cx, cy));
	Mat q1Re(planes[0], Rect(cx, 0, cx, cy));
	Mat q2Re(planes[0], Rect(0, cy, cx, cy));
	Mat q3Re(planes[0], Rect(cx, cy, cx, cy));

	//��ġ ������
	Mat tmp;
	q0Re.copyTo(tmp);
	q3Re.copyTo(q0Re);
	tmp.copyTo(q3Re);
	q1Re.copyTo(tmp);
	q2Re.copyTo(q1Re);
	tmp.copyTo(q2Re);

	//�� 4���� �簢���и� -> planes 1
	Mat q0Im(planes[1], Rect(0, 0, cx, cy));
	Mat q1Im(planes[1], Rect(cx, 0, cx, cy));
	Mat q2Im(planes[1], Rect(0, cy, cx, cy));
	Mat q3Im(planes[1], Rect(cx, cy, cx, cy));

	//��ġ ������
	q0Im.copyTo(tmp);
	q3Im.copyTo(q0Im);
	tmp.copyTo(q3Im);
	q1Im.copyTo(tmp);
	q2Im.copyTo(q1Im);
	tmp.copyTo(q2Im);

	//�ٽ� �ϳ��� ��ħ 
	Mat centerComplex;
	merge(planes, 2, centerComplex);

	return centerComplex;
}

//�����, �Ǽ��� ��ġ��
Mat setComplex(Mat magImg, Mat phaImg) {
	exp(magImg, magImg);
	magImg -= Scalar::all(1);
	//magnitude ����� �ݴ�� ����

	Mat planes[2];
	polarToCart(magImg, phaImg, planes[0], planes[1]);
	// �� ��ǥ�� -> ���� ��ǥ�� (������ ũ��κ��� 2���� ��ǥ)

	Mat complexImg;
	merge(planes, 2, complexImg);
	//�Ǽ���, ����� ��ü

	return complexImg;
}

//inverse discrete fourier Transform
Mat doIdft(Mat complexImg) {
	Mat idftcvt;
	idft(complexImg, idftcvt);
	//IDFT�� �̿��Ͽ� ���� ���� ��� 
	Mat planes[2];
	split(idftcvt, planes);

	Mat dstImg;
	magnitude(planes[0], planes[1], dstImg);
	normalize(dstImg, dstImg, 255, 0, NORM_MINMAX);
	dstImg.convertTo(dstImg, CV_8UC1);
	//�Ϲ� ������ type �� ǥ�������� ��ȯ

	return dstImg;
}

Mat doLPF(Mat srcImg) {
	// DFT
	Mat padImg = padding(srcImg);
	Mat complexImg = doDft(srcImg);
	Mat centerComplexImg = centralize(complexImg);
	Mat magImg = getMagnitude(centerComplexImg);
	Mat phaImg = getPhase(centerComplexImg);

	// LPF
	double minVal, maxVal;
	Point minLoc, maxLoc;
	minMaxLoc(magImg, &minVal, &maxVal, &minLoc, &maxLoc);
	normalize(magImg, magImg, 0, 1, NORM_MINMAX);

	Mat maskImg = Mat::zeros(magImg.size(), CV_32F);
	circle(maskImg, Point(maskImg.cols / 2, maskImg.rows / 2), 20, Scalar::all(1), -1, -1, 0);

	Mat magImg2;
	multiply(magImg, maskImg, magImg2);

	//IDFT
	normalize(magImg2, magImg2, (float)minVal, (float)maxVal, NORM_MINMAX);
	Mat complexImg2 = setComplex(magImg2, phaImg);
	Mat dstImg = doIdft(complexImg2);

	return myNormalize(dstImg);

}


//High pass filtering
Mat doHPF(Mat srcImg) {
	// DFT
	Mat padImg = padding(srcImg);
	Mat complexImg = doDft(srcImg);
	Mat centerComplexImg = centralize(complexImg);
	Mat magImg = getMagnitude(centerComplexImg);
	Mat phaImg = getPhase(centerComplexImg);

	// HPF
	double minVal, maxVal;
	Point minLoc, maxLoc;
	minMaxLoc(magImg, &minVal, &maxVal, &minLoc, &maxLoc);
	normalize(magImg, magImg, 0, 1, NORM_MINMAX);

	Mat maskImg = Mat::ones(magImg.size(), CV_32F);
	circle(maskImg, Point(maskImg.cols / 2, maskImg.rows / 2), 50, Scalar::all(0), -1, -1, 0);

	Mat magImg2;
	multiply(magImg, maskImg, magImg2);

	//IDFT
	normalize(magImg2, magImg2, (float)minVal, (float)maxVal, NORM_MINMAX);
	Mat complexImg2 = setComplex(magImg2, phaImg);
	Mat dstImg = doIdft(complexImg2);

	return myNormalize(dstImg);

}

//Flickering reducimg filtering
Mat doFRF(Mat srcImg) {
	// DFT
	Mat padImg = padding(srcImg);
	Mat complexImg = doDft(srcImg);
	Mat centerComplexImg = centralize(complexImg);
	Mat magImg = getMagnitude(centerComplexImg);
	Mat phaImg = getPhase(centerComplexImg);


	imshow("Test window_ mag", myNormalize(magImg));


	// Filter 
	//x�� �������� flickering ������ ���� -> ���� ������ y�� ������ ���ļ� ���� �����
	double minVal, maxVal;
	Point minLoc, maxLoc;
	minMaxLoc(magImg, &minVal, &maxVal, &minLoc, &maxLoc);
	normalize(magImg, magImg, 0, 1, NORM_MINMAX);

	// ���ڰ� �β�
	int thick = 30;
	int thick_l = 0;
	/*
	//X���� 
	Mat maskImg_X = Mat::ones(magImg.size(), CV_32F);
	int start_y = maskImg_X.rows / 2 - thick / 2;
	int width = maskImg_X.cols;
	maskImg_X(Rect(0, start_y, width, thick)) = 0;
	*/
	
	//y���� 
	Mat maskImg_Y = Mat::ones(magImg.size(), CV_32F);
	int start_x = maskImg_Y.cols / 2 - thick / 2;
	int hight = maskImg_Y.rows ;
	maskImg_Y(Rect(start_x, 0, thick, hight)) = 0;
	
	//masking ��ġ�� 
	//Mat Mask_img_B = maskImg_X  & maskImg_Y;
	
	//x���� ����
	Mat maskImg_LX = Mat::zeros(magImg.size(), CV_32F);
	int start_y_l = maskImg_LX.rows / 2 - thick / 2;
	int width_l = maskImg_LX.cols/2  -thick/2;
	maskImg_LX(Rect(width_l, start_y_l, thick, thick)) = 1;
	
	/*
	//y���� ����
	Mat maskImg_LY = Mat::zeros(magImg.size(), CV_32F);
	int start_x_l = maskImg_LY.cols / 2 - thick_l / 2;
	int hight_l = maskImg_LY.rows;
	maskImg_LY(Rect(start_x_l, 0, thick_l, hight_l)) =1;

	Mat Mask_img_S = maskImg_LX | maskImg_LY;
	*/

	Mat Mask_img;
	add(maskImg_Y, maskImg_LX, Mask_img);
	imshow("Test window_ Mask", Mask_img);


	//������ mask�� magnitude filtering 
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

	Mat src_img = imread("./image/img3.jpg", 0); // �̹��� �б�

	if (!src_img.data)
	{
		cout << "Image not loaded";
		return -1;
	}
	/*
	Mat change= doDft(src_img);
	Mat centerImg = centralize(change);
	Mat mag_img = myNormalize(getMagnitude(centerImg));
	Mat pha_img = myNormalize(getPhase(centerImg));
	imshow("Test window_ magnitude",mag_img);
	imshow("Test window_ phase", pha_img);
	imshow("Test_window_original", src_img);
	*/

	Mat bandImg = doFRF(src_img);
	imshow("Test window_ original", src_img);
	imshow("Test window_ Band", bandImg);
	

	waitKey(0); // Ű �Է� ��� (0: Ű�� �Էµ� �� ���� ���α׷� ����
	destroyAllWindows;

	return 0;



}

