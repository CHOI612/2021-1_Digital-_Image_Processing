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
	//x�� �������� flickering ������ ���� -> ���� ������ y�� ������ ���ļ� ���� �����
	double minVal, maxVal;
	Point minLoc, maxLoc;
	minMaxLoc(magImg, &minVal, &maxVal, &minLoc, &maxLoc);
	normalize(magImg, magImg, 0, 1, NORM_MINMAX);
	
	//���� �������� �ﰢ�� ����� ����� �׺κ� ���� filtering �ǵ��� 
	Mat maskImg_tri = Mat::ones(magImg.size(), CV_32F);

	//�ﰢ�� ũ�� ���� 
	int thick = 70;
	int thick_l = 0;
	int pass_point = 5;
	int w = maskImg_tri.cols;
	int h = maskImg_tri.rows;

  // �ﰢ�� �� 1�� 
	Point tri_up[1][3]; 

	//setting
	tri_up[0][0] = Point(w / 2 - thick_l, h / 2 - pass_point); // Point(���� �ȼ� ��ġ, ���� �ȼ� ��ġ)

	tri_up[0][1] = Point(w / 2 - thick, 0);

	tri_up[0][2] = Point(w / 2 - thick_l, 0);

	const Point* ppt_up[1] = {tri_up[0] };

	int npt_up[] = { 3 };

	fillPoly(maskImg_tri, ppt_up, npt_up, 1, Scalar(0),8); // masking

	// �ﰢ�� �� 2�� 
	Point tri_up_2[1][3];
	//setting
	tri_up_2[0][0] = Point(w / 2 + thick_l, h / 2 - pass_point); // Point(���� �ȼ� ��ġ, ���� �ȼ� ��ġ)

	tri_up_2[0][1] = Point(w / 2 + thick, 0);

	tri_up_2[0][2] = Point(w / 2 + thick_l, 0);

	const Point* ppt_up_2[1] = { tri_up_2[0] };

	int npt_up_2[] = { 3 };

	fillPoly(maskImg_tri, ppt_up_2, npt_up_2, 1, Scalar(0), 8); // masking

	// �ﰢ�� �Ʒ� 1�� 
	Point tri_down[1][3];
	//setting
	tri_down[0][0] = Point(w / 2 - thick_l, h / 2 + pass_point); // Point(���� �ȼ� ��ġ, ���� �ȼ� ��ġ)

	tri_down[0][1] = Point(w / 2 - thick, h);

	tri_down[0][2] = Point(w / 2 - thick_l, h);

	const Point* ppt_down[1] = { tri_down[0] };

	int npt_down[] = { 3 };

	fillPoly(maskImg_tri, ppt_down, npt_down, 1, Scalar(0), 8); // masking

	// �ﰢ�� �Ʒ� 2�� 
	Point tri_down_2[1][3];
	//setting
	tri_down_2[0][0] = Point(w / 2 + thick_l, h / 2 + pass_point); // Point(���� �ȼ� ��ġ, ���� �ȼ� ��ġ)

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

	Mat src_img = imread("./image/img3.jpg", 0); // �̹��� �б�

	if (!src_img.data)
	{
		cout << "Image not loaded";
		return -1;
	}

	
	Mat filter_img = doFRF(src_img);
	imshow("Test window_ original", src_img);
	imshow("Test window_ filtering", filter_img);


	waitKey(0); // Ű �Է� ��� (0: Ű�� �Էµ� �� ���� ���α׷� ����
	destroyAllWindows;

	return 0;
}


