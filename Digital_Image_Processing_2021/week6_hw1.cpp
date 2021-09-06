#include <iostream>
#include <algorithm>
#include <vector>
#include "opencv2/core/core.hpp" 
#include "opencv2/highgui/highgui.hpp" 
#include "opencv2/imgproc/imgproc.hpp"  
#include "opencv2/imgcodecs/imgcodecs.hpp"

using namespace cv;
using namespace std;


void myMedian(const Mat& src_img, Mat& dst_img, const Size& kn_size);
void doMedianEx(int f_size);


void doMedianEx(int f_size) {
	cout << " ---- do MdedianEx() ------\n" << endl;
	cout << "filter size : " << f_size << endl;

	//�Է�
	Mat src_img = imread("./image/salt_pepper2.png", 0);
	if (!src_img.data) printf("No image data \n");

	Mat dst_img;

	//¦���� �߰����� 2���� �Ǿ������. 
	if (f_size % 2 != 1) printf("size error\n"); 
	else {
		// Median ���͸� ����
		//parameter�� filter�� size�� �޾ƿ� Median filter ����
		myMedian(src_img, dst_img, Size(f_size, f_size));
	}

	//���ϱ� ���� ���� ��ġ��
	Mat result_img;
	hconcat(src_img, dst_img, result_img);
	imshow("test_window_domedian", result_img);
	waitKey(0);
}



void myMedian(const Mat& src_img, Mat& dst_img, const Size& kn_size) {
	dst_img = Mat::zeros(src_img.size(), CV_8UC1);

	int wd = src_img.cols;
	int hg = src_img.rows;
	int kwd = kn_size.width;
	int khg = kn_size.height;
	int rad_w = kwd / 2;
	int rad_h = khg / 2;

	//data ����
	uchar* src_data = (uchar*)src_img.data;
	uchar* dst_data = (uchar*)dst_img.data;

	//kernel table ���� �Ҵ�
	float* table = new float[kwd*khg]();
	float tmp;

	//pixel indexing (�����ڸ� ����)
	for (int c = rad_w + 1; c < wd - rad_w; c++) {
		for (int r = rad_h + 1; r < hg - rad_h; r++) {
			tmp = 0;
			//���� ������ 
			//�簢�� height/width ����� �������� ������ ��
			//���� ������ kernel table �迭�� �־��ش�.
			for (int kc = -rad_w; kc <= rad_w; kc++){
				for (int kr = -rad_h; kr <= rad_h; kr++){
				{
					tmp = (float)src_data[(r + kr) * wd + (c + kc)];
					table[(kr + rad_h)* kwd + (kc + rad_w)] = tmp;
				}
			}
			//kernel table ������ �߰��� ã�� 
			sort(table, table + kwd * khg);  //kernel table ����
			dst_data[r*wd + c] = (uchar)table[(kwd * khg) / 2]; // �߰��� ����
		}
	}
	delete table; //���� �Ҵ� ����
}




int main() {

	int f_size = 5;
	doMedianEx(f_size);

	return 0;
}


