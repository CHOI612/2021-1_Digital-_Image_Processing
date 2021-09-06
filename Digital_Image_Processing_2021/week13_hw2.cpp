#include <iostream>
#include <algorithm>
#include <vector>
#include <ctime>
#include "opencv2/core/core.hpp" 
#include "opencv2/core/utility.hpp"
#include "opencv2/core/ocl.hpp"
#include "opencv2/highgui/highgui.hpp" 
#include "opencv2/imgproc/imgproc.hpp"  
#include "opencv2/imgcodecs/imgcodecs.hpp"
#include "opencv2/photo.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d/nonfree.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/stitching.hpp"

using namespace cv;
using namespace std;
using namespace cv::xfeatures2d;


void ex_matching();
Mat makematching(Mat img_l, Mat img_r, int thresh_dist, int min_matches);


Mat makematching(Mat img_l, Mat img_r, int thresh_dist, int min_matches) {
	
	cout << "Doing SIFT..." << endl;
	
	//<Grayscale�� ��ȯ>
	Mat img_gray_l, img_gray_r;
	cvtColor(img_l, img_gray_l, CV_BGR2GRAY);
	cvtColor(img_r, img_gray_r, CV_BGR2GRAY);

	//<Ư¡��(key point) ����>
	Ptr<cv::SiftFeatureDetector> Detector = SiftFeatureDetector::create();
	vector<KeyPoint> kpts_obj, kpts_scene;
	Detector->detect(img_gray_l, kpts_obj);
	Detector->detect(img_gray_r, kpts_scene);

	//<Ư¡�� �ð�ȭ>
	Mat img_kpts_l, img_kpts_r;
	drawKeypoints(img_gray_l, kpts_obj, img_kpts_l, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
	drawKeypoints(img_gray_r, kpts_scene, img_kpts_r, Scalar::all(-1), DrawMatchesFlags::DEFAULT);

	imwrite("./13w/img_kpts_l.png", img_kpts_l);
	imwrite("./13w/img_kpts_r.png", img_kpts_r);

	//<�����(descriptor) ����>
	Ptr<SiftDescriptorExtractor> Extractor = SIFT::create();

	Mat img_des_obj, img_des_scene;
	Extractor->compute(img_gray_l, kpts_obj, img_des_obj);
	Extractor->compute(img_gray_r, kpts_scene, img_des_scene);

	cout << "Doing keypoint matching..." << endl;

	//<����ڸ� �̿��� Ư¡�� ��Ī>
	BFMatcher matcher(NORM_L2);
	vector<DMatch> matches;
	matcher.match(img_des_obj, img_des_scene, matches);

	//<��Ī ��� �ð�ȭ>
	Mat img_matches;
	drawMatches(img_gray_l, kpts_obj, img_gray_r, kpts_scene,
		matches, img_matches, Scalar::all(-1), Scalar::all(-1),
		vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

	imwrite("./13w/img_matches.png", img_matches);

	//<��Ī ��� ����>
	//��Ī �Ÿ��� ���� ����� ��Ī ����� �����ϴ� ����
	//�ּ� ��Ī �Ÿ��� 3�� or ����� ��Ī ��� 60�̻� ���� ����
	double dist_max = matches[0].distance;
	double dist_min = matches[0].distance;
	double dist;
	for (int i = 0; i < img_des_obj.rows; i++) {
		dist = matches[i].distance;
		if (dist < dist_min) dist_min = dist;
		if (dist > dist_max) dist_max = dist;
	}
	printf("min_dist : %f \n", dist_min); //max�� ��ǻ� ���ʿ�
	printf("min_dist : %f \n", dist_min);


	vector<DMatch> matches_good;
	do {
		vector<DMatch> good_matches2;
		for (int i = 0; i < img_des_obj.rows; i++) {
			if (matches[i].distance < thresh_dist * dist_min)
				good_matches2.push_back(matches[i]);
		}
		matches_good = good_matches2;
		thresh_dist -= 1;
	} while (thresh_dist != 2 && matches_good.size() > min_matches);

	//< ����� ��Ī ��� �ð�ȭ>
	//color �̹����� Ȯ�� 
	Mat img_matches_good;
	drawMatches(img_l, kpts_obj, img_r, kpts_scene,
		matches_good, img_matches_good, Scalar::all(-1), Scalar::all(-1),
		vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
	imwrite("./13w/img_match_good.png", img_matches_good);


	cout << "Get matching point..." << endl;

	//<��Ī ��� ��ǥ ����>
	vector<Point2f> obj, scene;
	for (int i = 0; i < matches_good.size(); i++) {
		obj.push_back(kpts_obj[matches_good[i].queryIdx].pt); //img1
		scene.push_back(kpts_scene[matches_good[i].trainIdx].pt); //img2
	}

	//<��Ī ����� ���� homography ����� ����>
	//�̻�ġ ���Ÿ� ���� RANSAC �߰�
	Mat mat_homo = findHomography(obj, scene, CV_RANSAC);

	//ROI������ ���� point vector
	vector<Point2f> obj_corner(4);
	vector<Point2f> box_corner(4);
	
	//object ��ġ���� 
	obj_corner[0] = cvPoint(0, 0);
	obj_corner[1] = cvPoint(img_l.cols, 0);
	obj_corner[2] = cvPoint(img_l.cols, img_l.rows);
	obj_corner[3] = cvPoint(0, img_l.rows);

	cout << "Drawing box..." << endl;

	//<Homograpy ����� �̿���  object�� ROI ��ġ ��� >
	Mat img_result = img_matches_good.clone();
	perspectiveTransform(obj_corner, box_corner, mat_homo);
	
	//���ʿ� �ռ��� object�̹��� ũ�� ��ŭ x���� �߰��ؼ� ���
	Point2f p(img_l.cols, 0);
	cout << p << endl;
	cout << box_corner[0] + p << endl;

	// image�� �簢�� line ��Ÿ����
	line(img_result, box_corner[0] + p, box_corner[1] + p, Scalar(0,255,0), 2);
	line(img_result, box_corner[1] + p, box_corner[2] + p, Scalar(0, 255, 0), 2);
	line(img_result, box_corner[2] + p, box_corner[3] + p, Scalar(0, 255, 0), 2);
	line(img_result, box_corner[3] + p, box_corner[0] + p, Scalar(0, 255, 0), 2);
	
	cout << "Saving image..." << endl;
	imwrite("./13w/img_matching.png", img_result);
	
	return img_result;
}

void ex_matching() {
	Mat matImage1 = imread("./13w/Scene.jpg", IMREAD_COLOR);
	Mat matImage2 = imread("./13w/Book1.jpg", IMREAD_COLOR);
	Mat matImage3 = imread("./13w/Book2.jpg", IMREAD_COLOR);
	Mat matImage4 = imread("./13w/Book3.jpg", IMREAD_COLOR);
	if (matImage1.empty() || matImage2.empty() || matImage3.empty() || matImage4.empty()) exit(-1);

	Mat result;
	cout << " For first book..." << endl;
	result = makematching(matImage2, matImage1, 3, 60);
	cout << " For seconde book..." << endl;
	result = makematching(matImage3, result, 3, 60);
	cout << "For third book..." << endl;
	result = makematching(matImage4, result, 3, 60);

	imshow("matching_result", result);
	waitKey();
}



int main() {

	ex_matching();

	waitKey(0);
	destroyAllWindows();
	return 0;
}