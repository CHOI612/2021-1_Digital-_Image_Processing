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
	vector <Scalar> clustersCenters;  //���� �߾Ӱ� ����
	vector <vector<Point>> ptinClusters; //������ ��ǥ ����
	double threshold = 0.001;
	double oldCenter = INFINITY;
	double NewCenter = 0;
	double diffChange = oldCenter - NewCenter; //���� ������ ��ȭ��

	//<�ʱ⼳��>
	//���� �߾Ӱ��� �������� �Ҵ� �� ������ ��ǥ���� ������ ���� �Ҵ�
	createClustersinfo(src_img, n_cluster, clustersCenters, ptinClusters);

	//<�߾Ӱ� ���� �� ȭ�Һ� ���� �Ǻ�>
	//�ݺ����� ������� ���� �߾Ӱ� ����
	//������ �԰谪 ���� ���� ������ ��ȭ�� ���� �� ���� �ݺ� 
	while (diffChange > threshold)
	{
		//<�ʱ�ȭ>
		NewCenter = 0;
		for (int k = 0; k < n_cluster; k++) { ptinClusters[k].clear(); }

		//<������ �߾Ӱ��� �������� ���� Ž��>
		findAssociatedCluster(src_img, n_cluster, clustersCenters, ptinClusters);

		//<���� �߾Ӱ� ����>
		diffChange = adjustClusterCenters(src_img, n_cluster, clustersCenters, ptinClusters, oldCenter, NewCenter);
		cout << "diffChange" << diffChange << endl;
	}

	//<���� �߾Ӱ����θ� �̷���� ���� ����>
	Mat dst_img = applyFinalClusterToImage(src_img, n_cluster, clustersCenters, ptinClusters);

	return dst_img;
}


void createClustersinfo(Mat imgInput, int n_cluster, vector <Scalar>& clustersCenters, vector <vector<Point>>& ptInClusters) {

	RNG random(cv::getTickCount()); //opencv���� ������ ���� �����ϴ� �Լ�

	for (int k = 0; k < n_cluster; k++) { //������ ���

		//<������ ��ǥ ȹ��>
		Point centerPoint;
		centerPoint.x = random.uniform(0, imgInput.cols);
		centerPoint.y = random.uniform(0, imgInput.rows);
		Scalar centerPixel = imgInput.at<Vec3b>(centerPoint.y, centerPoint.x);

		//< ������ ��ǥ�� ȭ�Ұ����� ������ �߾Ӱ� ����>
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

			//������ ���
			for (int k = 0; k < n_cluster; k++) {
				// <�� ���� �߾Ӱ����� ���̸� ���>
				Scalar clusterPixel = clustersCenters[k];
				double distance = computeColorDistance(pixel, clusterPixel);


				//<���̰� ���� ���� �������� ��ǥ�� ������ �Ǻ�>
				if (distance < minDistance) {
					minDistance = distance;
					closestClusterIndex = k;
				}
			}
			//< ��ǥ ����>
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

	//������ ���
	for (int k = 0; k < n_cluster; k++) {
		vector<Point> ptInCluster = ptInClusters[k];
		double newBlue = 0;
		double newGreen = 0;
		double newRed = 0;

		//< ��հ� ��� >
		for (int i = 0; i < ptInCluster.size(); i++) {
			Scalar pixel = src_img.at<Vec3b>(ptInCluster[i].y, ptInCluster[i].x);
			newBlue += pixel.val[0];
			newGreen += pixel.val[1];
			newRed += pixel.val[2];
		}
		newBlue /= ptInCluster.size();
		newGreen /= ptInCluster.size();
		newRed /= ptInCluster.size();

		// < ����� ��հ����� ���� �߾Ӱ� ��ü>
		Scalar newPixel(newBlue, newGreen, newRed);
		newCenter += computeColorDistance(newPixel, clustersCenters[k]);

		// ��� ������ ���� ��հ� ���
		clustersCenters[k] = newPixel;
	}

	newCenter /= n_cluster;
	diffChange = abs(oldCenter - newCenter);
	//��� ������ ���� ��հ� ��ȭ�� ���

	oldCenter = newCenter;
	return diffChange;
}


Mat applyFinalClusterToImage(Mat src_img, int n_cluster, vector <Scalar> clustersCenters, vector <vector<Point>> ptInClusters) {

	Mat dst_img(src_img.size(), src_img.type());
	//���� ���� ���� time�� �̿��� seed ���� 
	srand((unsigned int)time(NULL));

	for (int k = 0; k < n_cluster; k++) {
		vector<Point> ptInCluster = ptInClusters[k]; //������ ��ǥ��
		//������ ����	
		Scalar randomColor(rand() % 255, rand() % 255, rand() % 255);

		//�� ������ pixel ���� �� Ȯ��
		cout << randomColor << endl;
		cout << ptInCluster.size() << endl;

		//�� ���� ��ǥ ��ġ�� �ش� ���� ���� ������ pixel�� ���� 
		for (int i = 0; i < ptInCluster.size(); i++) {
			dst_img.at<Vec3b>(ptInCluster[i])[0] = randomColor.val[0];
			dst_img.at<Vec3b>(ptInCluster[i])[1] = randomColor.val[1];
			dst_img.at<Vec3b>(ptInCluster[i])[2] = randomColor.val[2];
		}
	}

	return dst_img;
}

int main() {

	//�̹��� ��������
	Mat src_img = imread("./image/test_1.jpg", 1);
	if (!src_img.data) printf("No image data \n");

	//���� �� ���� �� ����ȭ
	int  n_cluster = 7;
	Mat result_img = MyKMeans(src_img, n_cluster);

	//��� ���
	imshow("test_window_input", src_img);
	imshow("test_window", result_img);
	waitKey(0);
}