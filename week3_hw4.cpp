//���� 4
//�־��� ���� img3. jpg, img4. jpg, img5. jpg) �� �̿��� ������ ������ �ϼ��� ��

#include <iostream>
#include "opencv2/core/core.hpp" 
#include "opencv2/highgui/highgui.hpp" 
#include "opencv2/imgproc/imgproc.hpp"  

using namespace cv;
using namespace std;

int main() {
	Mat rocket_img = imread("./image/img3.jpg", 1); // �̹��� �б�
	Mat effect_img = imread("./image/img4.jpg", 1); // �̹��� �б�
	Mat text_img = imread("./image/img5.jpg", 1); // �̹��� �б�

	//������ Ȯ��
	cout << "image size : " << rocket_img.rows << "*" << rocket_img.cols << endl;
	cout << " text size : " << text_img.rows << "*" << text_img.cols << endl;
	
	//�� ���� ũ�� ���߱�
	resize(effect_img, effect_img, Size(rocket_img.cols, rocket_img.rows));

	// ����� �̹��� �ռ�
	Mat result_img;

	// ���� ����� ������ ��Ӱ� �Ͽ� �ռ����󿡼��� �� ��Ⱑ �൵��
	subtract(effect_img, Scalar(30, 30, 30), effect_img);
	subtract(rocket_img, effect_img, result_img);
	

	//�ΰ� ��� ����� ���� 
	
	//�ΰ� ��ġ ��� 
	Mat img_ROI(result_img, Rect(330, 350, text_img.cols, text_img.rows));
	
	//����ȭ�� ���� color ���� grayscale �������� ��ȯ
	Mat gray_img;
	cvtColor(text_img, gray_img, CV_BGR2GRAY); 
	
	Mat binary_img;
	//�Ӱ谪 ���� ����ȭ
	threshold(gray_img, binary_img, 190, 255, THRESH_BINARY);


	//���ϴ� ��ġ�� �ΰ� �̹��� �ڸ� ������ 
	Mat logo_space;
	bitwise_and(img_ROI, img_ROI, logo_space, binary_img);


	//��氪�� �����ȯ
	bitwise_not(binary_img, binary_img);
	
	//�ΰ� ���� ���� ���ϱ�
	Mat logo_img;
	bitwise_and(text_img, text_img, logo_img, binary_img);

	//���ϴ� ��ġ�� �̹��� �ڸ��� �ΰ����� ���� ����̹����� �����ϱ�
	//add(logo_space, logo_img, img_ROI); 
	bitwise_or(logo_space, logo_img, img_ROI); // or ���굵 ����

	imshow("Test window_result", result_img);// �̹��� ���

	waitKey(0); // Ű �Է� ��� (0: Ű�� �Էµ� �� ���� ���α׷� ����
	destroyAllWindows;
	imwrite("result_hw4.png", result_img); // �̹��� ����


	return 0;
}