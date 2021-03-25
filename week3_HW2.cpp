//���� 2
//�ռ� ������ ���󿡼� ���� , �Ķ� , �ʷ� ���� ���� ���� ī��Ʈ�ϴ� ���α׷��� �ۼ��ϰ�
//ī��Ʈ ����� ������ ��ġ�ϴ��� ������ ��


#include <iostream>
#include "opencv2/core/core.hpp" 
#include "opencv2/highgui/highgui.hpp" 
#include "opencv2/imgproc/imgproc.hpp"  

using namespace cv;
using namespace std;

int CountPoint(Mat img, char color);

int main() {
	Mat src_img = imread("result_hw1.png", 1); // �̹��� �б�
	int R, G, B = 0;

	//�� �������� ���� ����
	B = CountPoint(src_img, 'B');
	G = CountPoint(src_img, 'G');
	R = CountPoint(src_img, 'R');

	//��� ���
	cout << "Counting RGB point " << endl;
	cout << "R : " << R << endl;
	cout << "G : " << G << endl;
	cout << "B : " << B << endl;
	return 0;
}

//���ϴ� ���� �� ���� ī��Ʈ
int CountPoint(Mat img, char color){
	
	int b = 0;
	int g = 0;
	int r = 0;
	int count = 0;

	//�˻��� ���ϴ� ��
	if (color == 'B') b = 255;
	else if (color == 'G') g = 255;	
	else if (color == 'R') r = 255;

	//�̹��� ũ��
	int x = img.cols;
	int y = img.rows;

	//�̹��� ä�� ���� ����
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
				//���� 3ä�� ����� ���� Ȯ�� 
				if ((img.at<Vec3b>(i, j)[0] == b) &&
					(img.at<Vec3b>(i, j)[1] == g) &&
					(img.at<Vec3b>(i, j)[2] == r)) count++;
			}
		}
	}

	return count;
}



