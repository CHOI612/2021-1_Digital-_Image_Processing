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
	
	//<Grayscale로 변환>
	Mat img_gray_l, img_gray_r;
	cvtColor(img_l, img_gray_l, CV_BGR2GRAY);
	cvtColor(img_r, img_gray_r, CV_BGR2GRAY);

	//<특징점(key point) 추출>
	Ptr<cv::SiftFeatureDetector> Detector = SiftFeatureDetector::create();
	vector<KeyPoint> kpts_obj, kpts_scene;
	Detector->detect(img_gray_l, kpts_obj);
	Detector->detect(img_gray_r, kpts_scene);

	//<특징점 시각화>
	Mat img_kpts_l, img_kpts_r;
	drawKeypoints(img_gray_l, kpts_obj, img_kpts_l, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
	drawKeypoints(img_gray_r, kpts_scene, img_kpts_r, Scalar::all(-1), DrawMatchesFlags::DEFAULT);

	imwrite("./13w/img_kpts_l.png", img_kpts_l);
	imwrite("./13w/img_kpts_r.png", img_kpts_r);

	//<기술자(descriptor) 추출>
	Ptr<SiftDescriptorExtractor> Extractor = SIFT::create();

	Mat img_des_obj, img_des_scene;
	Extractor->compute(img_gray_l, kpts_obj, img_des_obj);
	Extractor->compute(img_gray_r, kpts_scene, img_des_scene);

	cout << "Doing keypoint matching..." << endl;

	//<기술자를 이용한 특징점 매칭>
	BFMatcher matcher(NORM_L2);
	vector<DMatch> matches;
	matcher.match(img_des_obj, img_des_scene, matches);

	//<매칭 결과 시각화>
	Mat img_matches;
	drawMatches(img_gray_l, kpts_obj, img_gray_r, kpts_scene,
		matches, img_matches, Scalar::all(-1), Scalar::all(-1),
		vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

	imwrite("./13w/img_matches.png", img_matches);

	//<매칭 결과 정제>
	//매칭 거리가 적은 우수한 매칭 결과를 정제하는 과정
	//최소 매칭 거리의 3배 or 우수한 매칭 결과 60이상 까지 정제
	double dist_max = matches[0].distance;
	double dist_min = matches[0].distance;
	double dist;
	for (int i = 0; i < img_des_obj.rows; i++) {
		dist = matches[i].distance;
		if (dist < dist_min) dist_min = dist;
		if (dist > dist_max) dist_max = dist;
	}
	printf("min_dist : %f \n", dist_min); //max는 사실상 불필요
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

	//< 우수한 매칭 결과 시각화>
	//color 이미지로 확인 
	Mat img_matches_good;
	drawMatches(img_l, kpts_obj, img_r, kpts_scene,
		matches_good, img_matches_good, Scalar::all(-1), Scalar::all(-1),
		vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
	imwrite("./13w/img_match_good.png", img_matches_good);


	cout << "Get matching point..." << endl;

	//<매칭 결과 좌표 추출>
	vector<Point2f> obj, scene;
	for (int i = 0; i < matches_good.size(); i++) {
		obj.push_back(kpts_obj[matches_good[i].queryIdx].pt); //img1
		scene.push_back(kpts_scene[matches_good[i].trainIdx].pt); //img2
	}

	//<매칭 결과로 부터 homography 행렬을 추출>
	//이상치 제거를 위해 RANSAC 추가
	Mat mat_homo = findHomography(obj, scene, CV_RANSAC);

	//ROI영역을 위한 point vector
	vector<Point2f> obj_corner(4);
	vector<Point2f> box_corner(4);
	
	//object 위치저장 
	obj_corner[0] = cvPoint(0, 0);
	obj_corner[1] = cvPoint(img_l.cols, 0);
	obj_corner[2] = cvPoint(img_l.cols, img_l.rows);
	obj_corner[3] = cvPoint(0, img_l.rows);

	cout << "Drawing box..." << endl;

	//<Homograpy 행렬을 이용해  object의 ROI 위치 계산 >
	Mat img_result = img_matches_good.clone();
	perspectiveTransform(obj_corner, box_corner, mat_homo);
	
	//왼쪽에 합서된 object이미지 크기 만큼 x값을 추가해서 계산
	Point2f p(img_l.cols, 0);
	cout << p << endl;
	cout << box_corner[0] + p << endl;

	// image에 사각형 line 나타내기
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