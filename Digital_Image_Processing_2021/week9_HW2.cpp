#include <iostream>
#include <algorithm>
#include <vector>
#include <ctime>
#include "opencv2/core/core.hpp" 
#include "opencv2/highgui/highgui.hpp" 
#include "opencv2/imgproc/imgproc.hpp"  
#include "opencv2/imgcodecs/imgcodecs.hpp"


using namespace cv;
using namespace std;


int main() {
	Mat img = imread("./image/test9_12.jpg", 1);
	if (img.empty()) exit(-1);

	/*
	//1
	int x1 = 90;
	int y1 = 103;
	int x2 = 514;
	int y2 = 367;

	//2
	int x1 = 80;
	int y1 = 60;
	int x2 = 463;
	int y2 = 333;
	
	//3
	int x1 = 10;
	int y1 = 13;
	int x2 = 590;
	int y2 = 402;
	
	//4
	int x1 = 225;
	int y1 = 75;
	int x2 = 858;
	int y2 = 574;
	
	//5
	int x1 =73;
	int y1 = 25;
	int x2 = 366;
	int y2 = 480;
	
	//7
	int x1 =141;
	int y1 = 34;
	int x2 = 626;
	int y2 = 431;
	
	//8
	int x1 = 151;
	int y1 = 10;
	int x2 = 441;
	int y2 = 196;

	//9
	int x1 = 53;
	int y1 = 60;
	int x2 = 448;
	int y2 = 355;


	//10
	int x1 = 153;
	int y1 = 36;
	int x2 = 370;
	int y2 = 257;
	
	//11
	int x1 = 182;
	int y1 = 48;
	int x2 = 761;
	int y2 = 450;
	*/
	//12
	int x1 = 83;
	int y1 = 79;
	int x2 = 413;
	int y2 = 260;

	Rect rect;
	rect = Rect(Point(x1, y1), Point(x2, y2));

	Mat result, bg_model, fg_model;
	grabCut(img, result, rect, bg_model, fg_model, 5, GC_INIT_WITH_RECT);

	compare(result, GC_PR_FGD, result, CMP_EQ);
	//GC_RP_FRD : GrabCut class foreground ÇÈ¼¿
	//CMP_EQ: compare ¿É¼Ç equal


	Mat mask(img.size(), CV_8UC3, cv::Scalar(255, 255, 255));
	img.copyTo(mask, result);
	
	Mat img_rect;
	img.copyTo(img_rect);
	rectangle(img_rect, rect, Scalar(0,0,255), 3, 4, 0);
	imshow("scr_img", img_rect);
	imshow("Dst_img", mask);
	waitKey(0);
	destroyAllWindows();
	
}