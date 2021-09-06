#include <iostream>
#include <algorithm>
#include <vector>
#include "opencv2/core/core.hpp" 
#include "opencv2/highgui/highgui.hpp" 
#include "opencv2/imgproc/imgproc.hpp"  
#include "opencv2/imgcodecs/imgcodecs.hpp"


using namespace cv;
using namespace std;

Mat myBGR2HSV(Mat src_img); // data RGB�� HSV�� ���� 
Mat myInRange(Mat hsv_img, Scalar min, Scalar max); //���ϴ� �� ���� masking
double* mycheckcolor(Mat hsv_img); // ����� �������� ������ main color ã��

Mat myBGR2HSV(Mat src_img) {
	double b, g, r, h, s, v;
	Mat dst_img(src_img.size(), src_img.type());

	//��ü �ȼ�
	for (int y = 0; y < src_img.rows; y++) {
		for (int x = 0; x < src_img.cols; x++) {
			//RGB�� ����
			b = (double)src_img.at<Vec3b>(y, x)[0];
			g = (double)src_img.at<Vec3b>(y, x)[1];
			r = (double)src_img.at<Vec3b>(y, x)[2];

			//BGR -> HSV�� ��ȯ
			vector<double> vec = { r , g, b};
			double min = *min_element(vec.begin(), vec.end());
			double max = *max_element(vec.begin(), vec.end());

			v = max; //���� 0~1 �������� 0~255���� ���
			if (v == 0) { s = 0; }
			else { s = (max - min) / max; }

			if (max == r) { h = 0 + (g - b) / (max - min); }
			else if (max == g) { h = 2 + (b - r) / (max - min); }
			else { h = 4 + (r - g) / (max - min); }

			h *= 60;
			if (h < 0) { h += 360; }

			//������ ����� ���Ͽ� 0~255 ���� ���� ���߾� HSV�� ��ȯ
			h /= 2;
			s *= 255;
			h = h > 255.0 ? 255.0 : h < 0 ? 0 : h;
			s = s > 255.0 ? 255.0 : s < 0 ? 0 : s;
			v = v > 255.0 ? 255.0 : v < 0 ? 0 : v;

			//HSV ���·� data ����
			dst_img.at<Vec3b>(y, x)[0] = (uchar)h;
			dst_img.at<Vec3b>(y, x)[1] = (uchar)s;
			dst_img.at<Vec3b>(y, x)[2] = (uchar)v;
		}
	}
	return dst_img;
}

Mat myInRange(Mat hsv_img, Scalar min, Scalar max) {

	//mask �̹���
	Mat mask_img = Mat::zeros(hsv_img.size(), CV_8UC1);
	
	double h, s, v;
	for (int y = 0; y < hsv_img.rows; y++) {
		for (int x = 0; x < hsv_img.cols; x++) {
			//�� �ȼ����� HSV�� �޾ƿ���
			h = (double)hsv_img.at<Vec3b>(y, x)[0];
			s = (double)hsv_img.at<Vec3b>(y, x)[1];
			v = (double)hsv_img.at<Vec3b>(y, x)[2];

			//�������϶� ������ �ٸ�
			if (max.val[0] == 12.5) {
				//���ϴ� ������ �ش�Ǵ� pixel�� ���
				if (h <= max.val[0] || h >= min.val[0]) {
					if (s<max.val[1] && s>min.val[1]) {
						if (v<max.val[2] && s>min.val[2]) {
							mask_img.at<uchar>(y, x) = 255;
						}
					}
				}
			}
			else {
				//���ϴ� ������ �ش�Ǵ� pixel�� ���
				if (h <= max.val[0] && h >= min.val[0]) {
					if (s<max.val[1] && s>min.val[1]) {
						if (v<max.val[2] && s>min.val[2]) {
							mask_img.at<uchar>(y, x) = 255;
						}
					}
				}
			}
		}
	}

	return mask_img;

}

double* mycheckcolor(Mat hsv_img) {

	double min_max[2]; //HSV���� hue�� min�� max�� return�� ����
	
	//pair�� �̿��Ͽ� �� �� ������ pixel ���� Ȯ��
	vector<pair<string, int>> color;
	color.push_back(make_pair("Red", 0));
	color.push_back(make_pair("Orange", 0));
	color.push_back(make_pair("Yellow", 0));
	color.push_back(make_pair("Green", 0));
	color.push_back(make_pair("Blue", 0));
	color.push_back(make_pair("Purple", 0));

	//�� ���� ���� H�� min/max��
	vector<pair<double, double>> term;
	term.push_back(make_pair(330, 25));
	term.push_back(make_pair(25, 35));
	term.push_back(make_pair(35, 75));
	term.push_back(make_pair(75,165));
	term.push_back(make_pair(165, 270));
	term.push_back(make_pair(270, 330));

	double h, s, v;
	for (int y = 0; y < hsv_img.rows; y++) {
		for (int x = 0; x < hsv_img.cols; x++) {

			//�� Ȯ���� ���� pixel data���� HSV ���� 
			h = (double)hsv_img.at<Vec3b>(y, x)[0];
			s = (double)hsv_img.at<Vec3b>(y, x)[1];
			v = (double)hsv_img.at<Vec3b>(y, x)[2];
			//RGB to HSV�Ҷ� �������� 1/2�� �ٿ� �ذ� ����
			h *= 2;

			//�������� ����� �ƴҶ� �� pixel �� ���� Ȯ��
			if (not (s == 0 || v == 0)) {
			    if (25 < h &&  h <= 35) { color[1].second +=1; }
				else if (35 < h  && h <= 75) { color[2].second +=1; }
				else if (75 < h &&  h <= 165) { color[3].second +=1; }
				else if (165 < h && h <= 270) { color[4].second +=1; }
				else if (270 < h && h < 330) { color[5].second +=1; }
				else if (330 <= h || 25 <= h) { color[0].second += 1; }
			}
		}
	}
	
	//���� ���� ������ �� �� max�� Ȯ��
	int max = color[0].second;
	int index = 0;
	for (int i = 0; i < color.size(); ++i)
	{
		cout << color[i].second << endl;
		if (max < color[i].second) {
			index = i;
			max = color[i].second;
		}
	}

	//������ ���� ���� main color ���
	cout << "image's main color is " << color[index].first << endl;

	//�ش� �� min/max�� return�� ���� ����
	min_max[0] = term[index].first/2;
	min_max[1] = term[index].second/2;

	return min_max;
}

int main() {

	//�̹��� ��������
	Mat src_img = imread("./image/bana.jpg", 1);
	if (!src_img.data) printf("No image data \n");

	//BGR to HSV�� ��ȯ
	Mat result_img = myBGR2HSV(src_img);
	
	//main color Ȯ�� �� �ش� �� ���� Ȯ��
	double* min_max = mycheckcolor(result_img);

	//�ش� �� �κ� ������ mask�̹��� ����
	Mat mask = myInRange(result_img, cv::Scalar(min_max[0], 20, 20), cv::Scalar(min_max[1], 230, 320));

	Mat dst_img;
	//mask�� and ������ ���Ͽ� ���ϴ� �κи� ����� ����
	bitwise_and(src_img, src_img, dst_img, mask);
	//��� ���
	imshow("test_window_input", src_img);
	imshow("test_window", result_img);
	imshow("test_window_mask", dst_img);
	waitKey(0);


	return 0;
}