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
	//입력
	Mat src_img = imread("./image/edge_test.jpg", 0);
	if (!src_img.data) printf("No image data \n");
	//결과 저장할 Mat
	Mat dst_img;
	Mat result_img;
	//시간 측정 변수
	clock_t start, end;
	double result;
	//비교할 threshold값
	int low_threshold = low;
	int high_threshold = high;

	//for문 사용하여 threshold별 시간 변화 측정 
	for (int i = 0; i < num; i++) {

		//함수 시간 측정 전 시간 저장 
		start = clock();
		Canny(src_img, dst_img, low_threshold, high_threshold);
		//함수 수행 후 시간 저장 
		end = clock();
		//두 시간 차를 통하여 걸린 시간 측정 
		result = (double)(end - start) / CLOCKS_PER_SEC;

		//실험값 출력 
		cout << "==================" << endl;
		cout << "threshold : " << low_threshold << ", " << high_threshold << endl;
		cout << "time result :" << result << endl;

		//threshold 바꿔가며 진행 
		if(task == 1) high_threshold -= 10;
		else if (task == 2) low_threshold += 10;
	}
	String fname = "test_window_doCannylEx" + to_string(low);
	//출력
	hconcat(src_img, dst_img, result_img);
	imshow("test_window_doCannyEx", result_img);
	waitKey(0);
	

}


int main() {

	doCannyEx(180,240,1,5);
	
	return 0;
}


