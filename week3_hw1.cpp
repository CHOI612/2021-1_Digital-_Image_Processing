//���� 1 
//�־��� ���� (img1. �� ���� , �Ķ� , �ʷ� ���� ���� ���� ������ ������ŭ ������
//�� �����ϴ� ���α׷��� �ۼ��� ��

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
	Mat src_img = imread("./image/img1.jpg", 1); // �̹��� �б�
	int R, G, B;
	cout << "enter num for RGB point" << endl;
	cout << "R : ";
	cin >> R;
	cout << "G : ";
	cin >> G;
	cout << "B : ";
	cin >> B;

	//��ü �ȼ��� ���� 
	if ((R + G + B) > (src_img.cols * src_img.rows)) {
		cout << "total number error" << endl;
		return 0;
	}

	//�Լ��� ����Ͽ� �� ���� �� ����
	ChangeToPoint_B(src_img, B);
	ChangeToPoint_G(src_img, G);
	ChangeToPoint_R(src_img, R);
	
	imshow("Test window", src_img);// �̹��� ���
	waitKey(0); // Ű �Է� ��� (0: Ű�� �Էµ� �� ���� ���α׷� ����
	destroyWindow("Test window"); // �̹��� ���â ����
	imwrite("result_hw1.png", src_img); // �̹��� ����
	return 0;
}

//blue�� ����
void ChangeToPoint_B(Mat img, int num) {
	int count = 0;
	for (int n = 0; n < num; n++) {
		//random�� ��ġ ����
		int x = rand() % img.cols;
		int y = rand() % img.rows;

		//�̹����� ä���� ���� ����
		if (img.channels() == 1) {
			img.at<uchar>(y, x) = 255;  //���� ä��
		}
		else {
			//cout << y << " " << x << endl;
			if (img.at<Vec3b>(y, x)[0] == 255) n--; //blue �ߺ�Ȯ��
			else {  
				img.at<Vec3b>(y, x)[0] = 255; 
				img.at<Vec3b>(y, x)[1] = 0;
				img.at<Vec3b>(y, x)[2] = 0;
			} //Blue
		}
	}
}

//green�� ����
void ChangeToPoint_G(Mat img, int num) {
	for (int n = 0; n < num; n++) {
		//random�� ��ġ ����
		int x = rand() % img.cols;
		int y = rand() % img.rows;

		//�̹����� ä���� ���� ����
		if (img.channels() == 1) {
			img.at<uchar>(y, x) = 255;  //���� ä��
		}
		else {
			if (img.at<Vec3b>(y, x)[0] == 255) n--; //blue �ߺ�Ȯ��
			else if (img.at<Vec3b>(y, x)[1] == 255) n--;  //Green �ߺ�Ȯ��
			else{
				img.at<Vec3b>(y, x)[0] = 0;
				img.at<Vec3b>(y, x)[1] = 255;
				img.at<Vec3b>(y, x)[2] = 0;
			} //Green
		
		}
	}
}


//red�� ����
void ChangeToPoint_R(Mat img, int num) {
	for (int n = 0; n < num; n++) {
		//random�� ��ġ ����
		int x = rand() % img.cols;
		int y = rand() % img.rows;

		//�̹����� ä���� ���� ����
		if (img.channels() == 1) {
			img.at<uchar>(y, x) = 255;  //���� ä��
		}
		else {
			if (img.at<Vec3b>(y, x)[0] == 255) n--; //blue �ߺ�Ȯ��
			else if(img.at<Vec3b>(y, x)[1] == 255) n--;  //Green �ߺ�Ȯ��
			else if( img.at<Vec3b>(y, x)[2] == 255) n--; //Red �ߺ�Ȯ��
			else {

				img.at<Vec3b>(y, x)[0] = 0;
				img.at<Vec3b>(y, x)[1] = 0;
				img.at<Vec3b>(y, x)[2] = 255;
			} //Red
		}
	}
}
