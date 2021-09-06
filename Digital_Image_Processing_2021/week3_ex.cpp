//I/O and mat example

#include <iostream>
#include "opencv2/core/core.hpp" // Mat class�� ���� data structure �� ��� ��ƾ�� �����ϴ� ���
#include "opencv2/highgui/highgui.hpp" // GUI �� ���õ� ��Ҹ� �����ϴ� ��� imshow ��
#include "opencv2/imgproc/imgproc.hpp"  // ���� �̹��� ó�� �Լ��� �����ϴ� ���
//�� �ش��� �⺻������ include�� �ʿ�
using namespace cv;
using namespace std;

int main() {
	Mat src_img = imread("./image/landing.jpg", 0); // �̹��� �б�

	// int flags = IMREAD_COLOR �Ǵ� int flags = 1 -- > �÷� �������� ����
	// int flags = IMREAD_GRAYSCALE �Ǵ� int flags = 0 -- > ��� �������� ����
	// int flags = IMREAD_UNCHANGED �Ǵ� int flags = 1 -- > ���� ������ ���Ĵ�� ����

	imshow("Test window", src_img);// �̹��� ���
	waitKey(0); // Ű �Է� ��� (0: Ű�� �Էµ� �� ���� ���α׷� ����
	destroyWindow("Test window"); // �̹��� ���â ����
	imwrite("langding_gray.png", src_img); // �̹��� ����
	return 0;
}