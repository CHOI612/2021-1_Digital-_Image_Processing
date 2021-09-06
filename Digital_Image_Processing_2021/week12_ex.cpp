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

void cvFlip();
void cvRotation();
void cvAffine();
void cvPerspective();
void mycvPerspective();
Mat myTransMat();

void cvFlip() {
	Mat src = imread("./12w/Lenna.png", 1);
	Mat dst_x, dst_y, dst_xy;

	flip(src, dst_x, 0);
	flip(src, dst_y, 1);
	flip(src, dst_xy, -1);

	imshow("nonflip", src);
	imshow("xflip", dst_x);
	imshow("yflip", dst_y);
	imshow("xyflip", dst_xy);
	waitKey(0);

	destroyAllWindows();

}

void cvRotation() {
	Mat src = imread("./12w/Lenna.png", 1);
	Mat dst, matrix;

	Point center = Point(src.cols / 2, src.rows / 2);
	matrix = getRotationMatrix2D(center, 45.0, 1.0);
	warpAffine(src, dst, matrix, src.size());

	imshow("nonrot", src);
	imshow("rot", dst);
	waitKey(0);

	destroyAllWindows();
}

void cvAffine() {
	Mat src = imread("./12w/Lenna.png", 1);
	Mat dst, matrix;

	float width = src.cols;
	float height = src.rows;

	Point2f srcTri[3], dstTri[3];
	srcTri[0] = Point2f(0.f, 0.f);
	srcTri[1] = Point2f(width - 1.f, 0.f);
	srcTri[2] = Point2f(0.f, height - 1.f);

	dstTri[0] = Point2f(0.f, height * 0.33f);
	dstTri[1] = Point2f(width * 0.85f, height * 0.25f);
	dstTri[2] = Point2f(width * 0.15f, height * 0.7f );
	
	matrix = getAffineTransform(srcTri, dstTri);
	warpAffine(src, dst, matrix, src.size());

	imshow("nonaff", src);
	imshow("aff", dst);

	waitKey(0);

	destroyAllWindows();

}

void cvPerspective() {
	Mat src = imread("./12w/Lenna.png", 1);
	Mat dst, matrix;

	float width = src.cols;
	float height = src.rows;

	Point2f srcTri[4], dstTri[4];
	srcTri[0] = Point2f(0.f, 0.f);
	srcTri[1] = Point2f(width - 1.f, 0.f);
	srcTri[2] = Point2f(0.f, height - 1.f);
	srcTri[3] = Point2f(width - 1.f, height - 1.f);

	dstTri[0] = Point2f(0.f, height * 0.33f);
	dstTri[1] = Point2f(width * 0.85f, height * 0.25f);
	dstTri[2] = Point2f(width * 0.15f, height * 0.7f);
	dstTri[3] = Point2f(width * 0.85f, height * 0.7f);

	matrix = getPerspectiveTransform(srcTri, dstTri);
	warpPerspective(src, dst, matrix, src.size());

	imshow("nonper", src);
	imshow("per", dst);

	waitKey(0);

	destroyAllWindows();
}

void mycvPerspective() {

	Mat src = imread("./12w/Lenna.png", 1);
	Mat dst, matrix;

	matrix = myTransMat();
	warpPerspective(src, dst, matrix, src.size());

	imshow("nonper", src);
	imshow("per", dst);

	waitKey(0);

	destroyAllWindows();
}

Mat myTransMat() {
	Mat matrix1 = (Mat_<double>(3, 3) <<
		1, tan(45 * CV_PI / 180), 0,
		0, 1, 0,
		0, 0, 1);
	Mat matrix2 = (Mat_<double>(3, 3) <<
		1, 0, -256,
		0, 1, 0,
		0, 0, 1);
	Mat matrix3 = (Mat_<double>(3, 3) <<
		0.5, 0, 0,
		0, 0.5, 0,
		0, 0, 1);
	return matrix3 * matrix2 * matrix1;
}

int main() {

	//cvFlip();
	//cvRotation();
	//cvAffine();
	//cvPerspective();
	mycvPerspective();
	return 0;
}