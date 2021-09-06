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

Mat MyKMeans(Mat src_img, int n_cluster);
void createClustersinfo(Mat imgInput, int n_cluster, vector <Scalar>& clustersCenters, vector <vector<Point>>& ptInClusters);
void findAssociatedCluster(Mat imgInput, int n_cluster, vector <Scalar> clustersCenters, vector <vector<Point>>& ptInClusters);
double adjustClusterCenters(Mat src_img, int n_cluster, vector <Scalar>& clustersCenters, vector <vector<Point>> ptInClusters, double& oldCenter, double newCenter);
Mat applyFinalClusterToImage(Mat src_img, int n_cluster, vector <Scalar> clustersCenters, vector <vector<Point>>ptInClusters);
double computeColorDistance(Scalar pixel, Scalar clusterPixel);


Mat MyKMeans(Mat src_img, int n_cluster) {
	vector <Scalar> clustersCenters;  //군집 중앙값 백터
	vector <vector<Point>> ptinClusters; //군집별 좌표 백터
	double threshold = 0.001;
	double oldCenter = INFINITY;
	double NewCenter = 0;
	double diffChange = oldCenter - NewCenter; //군집 조정의 변화량

	//<초기설정>
	//군집 중앙값을 무작위로 할당 및 군집별 좌표값을 저장할 벡터 할당
	createClustersinfo(src_img, n_cluster, clustersCenters, ptinClusters);

	//<중앙값 조정 및 화소별 군집 판별>
	//반복적인 방법으로 군집 중앙값 조정
	//설정한 입계값 보다 군집 조정의 변화가 작을 때 까지 반복 
	while (diffChange > threshold)
	{
		//<초기화>
		NewCenter = 0;
		for (int k = 0; k < n_cluster; k++) { ptinClusters[k].clear(); }

		//<현재의 중앙값을 기준으로 군집 탐색>
		findAssociatedCluster(src_img, n_cluster, clustersCenters, ptinClusters);

		//<군집 중앙값 조절>
		diffChange = adjustClusterCenters(src_img, n_cluster, clustersCenters, ptinClusters, oldCenter, NewCenter);
		cout << "diffChange" << diffChange << endl;
	}

	//<군집 중앙값으로만 이루어진 영상 생성>
	Mat dst_img = applyFinalClusterToImage(src_img, n_cluster, clustersCenters, ptinClusters);

	return dst_img;
}


void createClustersinfo(Mat imgInput, int n_cluster, vector <Scalar>& clustersCenters, vector <vector<Point>>& ptInClusters) {

	RNG random(cv::getTickCount()); //opencv에서 무작위 값을 설정하는 함수

	for (int k = 0; k < n_cluster; k++) { //군집별 계산

		//<무작위 좌표 획득>
		Point centerPoint;
		centerPoint.x = random.uniform(0, imgInput.cols);
		centerPoint.y = random.uniform(0, imgInput.rows);
		Scalar centerPixel = imgInput.at<Vec3b>(centerPoint.y, centerPoint.x);

		//< 무작위 좌표의 화소값으로 군집별 중앙값 설정>
		Scalar centerK(centerPixel.val[0], centerPixel.val[1], centerPixel.val[2]);
		clustersCenters.push_back(centerK);

		vector<Point> ptInClusterK;
		ptInClusters.push_back(ptInClusterK);
	}

}

void findAssociatedCluster(Mat imgInput, int n_cluster, vector <Scalar> clustersCenters, vector <vector<Point>>& ptInClusters) {

	for (int r = 0; r < imgInput.rows; r++) {
		for (int c = 0; c < imgInput.cols; c++) {
			double minDistance = INFINITY;
			int closestClusterIndex = 0;
			Scalar pixel = imgInput.at<Vec3b>(r, c);

			//군집별 계산
			for (int k = 0; k < n_cluster; k++) {
				// <각 군집 중앙값과의 차이를 계산>
				Scalar clusterPixel = clustersCenters[k];
				double distance = computeColorDistance(pixel, clusterPixel);


				//<차이가 가장 적은 군집으로 좌표의 군집을 판별>
				if (distance < minDistance) {
					minDistance = distance;
					closestClusterIndex = k;
				}
			}
			//< 좌표 저장>
			ptInClusters[closestClusterIndex].push_back(Point(c, r));
		}
	}
}

double computeColorDistance(Scalar pixel, Scalar clusterPixel) {
	double diffBlue = pixel.val[0] - clusterPixel.val[0];
	double diffGreen = pixel.val[1] - clusterPixel.val[1];
	double diffRed = pixel.val[2] - clusterPixel.val[2];

	double distance = sqrt(pow(diffBlue, 2) + pow(diffGreen, 2) + pow(diffRed, 2));
	//Euclidian distance

	return distance;
}

double adjustClusterCenters(Mat src_img, int n_cluster, vector <Scalar>& clustersCenters, vector <vector<Point>> ptInClusters, double& oldCenter, double newCenter) {

	double diffChange;

	//군집별 계산
	for (int k = 0; k < n_cluster; k++) {
		vector<Point> ptInCluster = ptInClusters[k];
		double newBlue = 0;
		double newGreen = 0;
		double newRed = 0;

		//< 평균값 계산 >
		for (int i = 0; i < ptInCluster.size(); i++) {
			Scalar pixel = src_img.at<Vec3b>(ptInCluster[i].y, ptInCluster[i].x);
			newBlue += pixel.val[0];
			newGreen += pixel.val[1];
			newRed += pixel.val[2];
		}
		newBlue /= ptInCluster.size();
		newGreen /= ptInCluster.size();
		newRed /= ptInCluster.size();

		// < 계산한 평균값으로 군집 중앙값 대체>
		Scalar newPixel(newBlue, newGreen, newRed);
		newCenter += computeColorDistance(newPixel, clustersCenters[k]);

		// 모든 군집에 대한 평균값 계산
		clustersCenters[k] = newPixel;
	}

	newCenter /= n_cluster;
	diffChange = abs(oldCenter - newCenter);
	//모든 군집에 대한 평균값 변화량 계산

	oldCenter = newCenter;
	return diffChange;
}


Mat applyFinalClusterToImage(Mat src_img, int n_cluster, vector <Scalar> clustersCenters, vector <vector<Point>> ptInClusters) {

	Mat dst_img(src_img.size(), src_img.type());
	//랜덤 색을 위한 time을 이용한 seed 생성 
	srand((unsigned int)time(NULL));

	for (int k = 0; k < n_cluster; k++) {
		vector<Point> ptInCluster = ptInClusters[k]; //군집별 좌표들
		//랜덤색 지정	
		Scalar randomColor(rand() % 255, rand() % 255, rand() % 255);

		//각 군집의 pixel 수와 색 확인
		cout << randomColor << endl;
		cout << ptInCluster.size() << endl;

		//각 군집 좌표 위치에 해당 군집 랜덤 색으로 pixel값 변경 
		for (int i = 0; i < ptInCluster.size(); i++) {
			dst_img.at<Vec3b>(ptInCluster[i])[0] = randomColor.val[0];
			dst_img.at<Vec3b>(ptInCluster[i])[1] = randomColor.val[1];
			dst_img.at<Vec3b>(ptInCluster[i])[2] = randomColor.val[2];
		}
	}

	return dst_img;
}

int main() {

	//이미지 가져오기
	Mat src_img = imread("./image/test_1.jpg", 1);
	if (!src_img.data) printf("No image data \n");

	//군집 수 지정 및 군집화
	int  n_cluster = 7;
	Mat result_img = MyKMeans(src_img, n_cluster);

	//결과 출력
	imshow("test_window_input", src_img);
	imshow("test_window", result_img);
	waitKey(0);
}