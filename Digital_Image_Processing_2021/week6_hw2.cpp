#include <iostream>
#include <algorithm>
#include <vector>
#include <stdio.h>
#include <time.h>
#include "opencv2/core/core.hpp" 
#include "opencv2/highgui/highgui.hpp" 
#include "opencv2/imgproc/imgproc.hpp"  
#include "opencv2/imgcodecs/imgcodecs.hpp"

using namespace cv;
using namespace std;

void doCannyEx();

void doCannyEx(int low, int high, int task, int num) {
	cout << "-----doCannyEx() ----\n" << endl;
	//�Է�
	Mat src_img = imread("./image/edge_test.jpg", 0);
	if (!src_img.data) printf("No image data \n");
	//��� ������ Mat
	Mat dst_img;
	Mat result_img;
	//�ð� ���� ����
	clock_t start, end;
	double result;
	//���� threshold��
	int low_threshold = low;
	int high_threshold = high;

	//for�� ����Ͽ� threshold�� �ð� ��ȭ ���� 
	for (int i = 0; i < num; i++) {

		//�Լ� �ð� ���� �� �ð� ���� 
		start = clock();
		Canny(src_img, dst_img, low_threshold, high_threshold);
		//�Լ� ���� �� �ð� ���� 
		end = clock();
		//�� �ð� ���� ���Ͽ� �ɸ� �ð� ���� 
		result = (double)(end - start) / CLOCKS_PER_SEC;

		//���谪 ��� 
		cout << "==================" << endl;
		cout << "threshold : " << low_threshold << ", " << high_threshold << endl;
		cout << "time result :" << result << endl;

		//threshold �ٲ㰡�� ���� 
		if(task == 1) high_threshold -= 10;
		else if (task == 2) low_threshold += 10;
	}
	String fname = "test_window_doCannylEx" + to_string(low);
	//���
	hconcat(src_img, dst_img, result_img);
	imshow("test_window_doCannyEx", result_img);
	waitKey(0);
	

}


int main() {

	doCannyEx(180,240,1,5);
	
	return 0;
}


