#include <iostream>
#include <algorithm>
#include <vector>
#include <ctime>
#include "opencv2/core/core.hpp" 
#include "opencv2/highgui/highgui.hpp" 
#include "opencv2/imgproc/imgproc.hpp"  
#include "opencv2/imgcodecs/imgcodecs.hpp"
#include "opencv2/photo.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/features2d.hpp"

using namespace cv;
using namespace std;

void mycvRotation();
Mat getMyRotationMatrix(Point2f,double,double);


void mycvRotation() {
	Mat src = imread("./12w/Lenna.png", 1);
	Mat dst, matrix;


	//getRotatiaonMatrix사용
	Point center = Point(src.cols / 2, src.rows / 2);
	matrix = getRotationMatrix2D(center, 45.0, 1.0);
	warpAffine(src, dst, matrix, src.size());


	//Matrix계산 함수를 만들어 사용
	Mat myDst,myMatrix;
	myMatrix = getMyRotationMatrix(center, 45.0, 1.0);
	warpAffine(src, myDst, myMatrix, src.size());


	imshow("nonrot", src);
	imshow("cvget_rot", dst);
	imshow("myget_rot", myDst);
	waitKey(0);

	destroyAllWindows();
}

Mat getMyRotationMatrix(Point2f center, double angle, double scale) {
	double a = scale * cos(angle *CV_PI / 180);
	double b = scale * sin(angle *CV_PI / 180);
	Mat matrix = (Mat_<double>(2, 3) <<
		a, b, ((1 - a)*center.x) - (b * center.y),
		-b, a, (b*center.x) + ((1 - a)* center.y));

	return matrix;
}

int main() {
	
	mycvRotation();
	return 0;
}