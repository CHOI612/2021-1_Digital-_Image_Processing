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
Mat Sobel_mag(float kernel[][3], int w, int h);

Mat MyCopy(Mat srcImg);
int  MyKernelConv3x3(uchar* arr, int kernel[][3], int x, int y, int width, int height);
Mat mySobelFilter(Mat srcImg);



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



//sobel filtering
// F(image *  sobel filter) = F(image)F(sobel)
Mat doSBF(Mat srcImg) {
	// DFT
	Mat padImg = padding(srcImg);
	Mat complexImg = doDft(srcImg);
	Mat centerComplexImg = centralize(complexImg);
	Mat magImg = getMagnitude(centerComplexImg);
	Mat phaImg = getPhase(centerComplexImg);


	// Filter 
	double minVal, maxVal;
	Point minLoc, maxLoc;
	minMaxLoc(magImg, &minVal, &maxVal, &minLoc, &maxLoc);
	normalize(magImg, magImg, 0, 1, NORM_MINMAX);

	// ���� ���� Sobel ����ũ
	float kernelX[3][3] = { -1., 0., 1.,
									 -2., 0., 2.,
									 -1., 0., 1. };

	// ���� ���� Sobel ����ũ
	float kernelY[3][3] = { -1., -2., -1.,
									 0.,  0.,  0.,
									 1.,  2.,  1. };

	//sobel filter�� Fourier Trasform�� magnitude ���
	Mat kernelX_mag = Sobel_mag(kernelX, magImg.cols, magImg.rows);
	Mat kernelY_mag = Sobel_mag(kernelY, magImg.cols, magImg.rows);


	// (Spatial) Convolution  -> (Freq) Multiply 
	Mat kernelX_result;
	Mat kernelY_result;

	multiply(magImg, kernelX_mag, kernelX_result);
	multiply(magImg, kernelY_mag, kernelY_result);

	// IDFT
	normalize(kernelX_result, kernelX_result, (float)minVal, (float)maxVal, NORM_MINMAX);
	normalize(kernelY_result, kernelY_result, (float)minVal, (float)maxVal, NORM_MINMAX);

	Mat kernelX_complexImg = setComplex(kernelX_result, phaImg);
	Mat kernelY_complexImg = setComplex(kernelY_result, phaImg);

	Mat kernelX_dstImg = doIdft(kernelX_complexImg);
	Mat kernelY_dstImg = doIdft(kernelY_complexImg);

	//imshow("Test window__x", myNormalize(kernelX_dstImg));
	//imshow("Test window_ y", myNormalize(kernelY_dstImg));


	// ������ X, Y ���� sobel filter ��� �����ֱ� 
	Mat final_result = kernelX_dstImg + kernelY_dstImg;

	return myNormalize(final_result);

}

//3*3 kernel sobel filter  -> 2d discrete fourier transform
//get magnitude spectrum
Mat Sobel_mag(float kernel[][3],int w,int h){
	
	// Mat Sobel Filter
	Mat kernel_mat(3, 3, CV_32F);
	
	// Generate Sobel Filter(float array -> Mat)
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			kernel_mat.at<float>(i, j) = kernel[i][j];
		}
	}


	// �Է� �̹����� ������ ũ�⿡ ���߾�  padding
	Mat pad_kernel_mat;
	copyMakeBorder(kernel_mat, pad_kernel_mat, 0, h - kernel_mat.rows, 0, w - kernel_mat.cols, BORDER_CONSTANT, Scalar::all(0));


	// Sobel Filter Fourier Transform
	Mat kernelX_complex = doDft(pad_kernel_mat);
	Mat centerComplexKernelX = centralize(kernelX_complex);
	Mat magKernelX = getMagnitude(centerComplexKernelX);

	return magKernelX;
}

int main() {

	Mat src_img = imread("./image/img2.jpg", 0); // �̹��� �б�

	if (!src_img.data)
	{
		cout << "Image not loaded";
		return -1;
	}

	Mat result_img_2 = mySobelFilter(src_img);
	imshow("Test window_sobel_spatial", result_img_2);// �̹��� ���

	Mat filter_img = doSBF(src_img);
	imshow("Test window_ original", src_img);
	imshow("Test window_ sobel_freq", filter_img);


	waitKey(0); // Ű �Է� ��� (0: Ű�� �Էµ� �� ���� ���α׷� ����
	destroyAllWindows;

	return 0;

}



//data ��� ȭ�� ���ٹ�
Mat MyCopy(Mat srcImg) {
	int width = srcImg.cols;
	int height = srcImg.rows;

	Mat dstImg(srcImg.size(), CV_8UC1); // �Է¿���� ������ ũ���� Mat ����
	uchar* srcData = srcImg.data;  //Mat��ü�� data�� ����Ű�� ������
	uchar* dstData = dstImg.data;

	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			dstData[y*width + x] = srcData[y*width + x];
		} // ȭ�Ұ��� ������ �о�� �ٸ� �迭�� ����
	}

	return dstImg;
}

//mask��� kenrel ���
int MyKernelConv3x3(uchar* arr, int kernel[][3], int x, int y, int width, int height) {

	int sum = 0;
	int sumkernel = 0;

	//Mask Ư�� ȭ���� ��� �̿�ȭ�ҿ� ���� ��� �ϵ��� �ݺ��� ���� 
	for (int j = -1; j <= 1; j++) {
		for (int i = -1; i <= 1; i++) {
			//���� �����ڸ����� ���� ���� ȭ�Ҹ� ���� �ʵ��� �ϴ� ���ǹ�
			if ((y + j) >= 0 && (y + j) < height && (x + i) >= 0 && (x + i) < width) {
				sum += arr[(y + j)*width + (x + i)] * kernel[i + 1][j + 1];
				sumkernel += kernel[i + 1][j + 1];
			}
		}
	}
	if (sumkernel != 0) { return sum / sumkernel; }
	else { return sum; }
}

//sobel ���� ����
Mat mySobelFilter(Mat srcImg) {
	int width = srcImg.cols;
	int height = srcImg.rows;

	//x�� ����
	int kernelX[3][3] = { -1, 0, 1,
									   -2, 0, 2,
										-1, 0, 1 };

	//y�� ����
	int kernelY[3][3] = { -1, -2, -1,
									0,  0,  0,
									1,  2,  1 };

	//�̹� mask�� ���� 0�� �ǹǷ� 1�� ����ȭ�ϴ� ������ ���� �ʿ� ����.

	Mat dstImg(srcImg.size(), CV_8UC1);
	uchar* srcData = srcImg.data;
	uchar* dstData = dstImg.data;

	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			dstData[y*width + x] = (abs(MyKernelConv3x3(srcData, kernelX, x, y, width, height)) +
				abs(MyKernelConv3x3(srcData, kernelY, x, y, width, height))) / 2;
		} // ������ ���� ����� ���밪 ���� 2�� ���� ���ȭ �Ͽ� ������� ����
	}

	return dstImg;

}




