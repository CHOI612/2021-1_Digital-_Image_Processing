//���� 3
//�־��� ������ �̿��� (img2.jpg) ������ ���� �� ������ �����ϴ� ���α׷��� �ۼ��ϰ�
//�ȼ� �� ������ �̿� ������׷� ��ġ ���θ� Ȯ�� �� �׷��� ����� ���� ������ �м��� ��


#include <iostream>
#include "opencv2/core/core.hpp" 
#include "opencv2/highgui/highgui.hpp" 
#include "opencv2/imgproc/imgproc.hpp"  

using namespace cv;
using namespace std;

void GettingDark_up(Mat img);
void GettingDark_down(Mat img);
Mat getHistogram(Mat& src);

int main() {
	Mat main_img = imread("./image/img2.jpg", 0); // �̹��� �б�
	Mat up_img = imread("./image/img2.jpg", 0); // �̹��� �б�
	Mat down_img = imread("./image/img2.jpg", 0); // �̹��� �б�

	//������ ���� ����
	GettingDark_up(up_img);
	GettingDark_down(down_img);

	
	imshow("Test window_up", up_img);// �̹��� ���
	imshow("Test window_down", down_img);// �̹��� ���
	imshow("histogram", getHistogram(main_img)); //������׷� ���
	imshow("histogram_up", getHistogram(up_img)); //������׷� ���
	imshow("histogram_down", getHistogram(down_img)); //������׷� ���
	waitKey(0); // Ű �Է� ��� (0: Ű�� �Էµ� �� ���� ���α׷� ����
	destroyAllWindows;
	imwrite("result_hw3_1.png", up_img); // �̹��� ����
	imwrite("result_hw3_2.png", down_img); // �̹��� ����
	
	return 0;
}

//���� ������ ��ο� ������ 
void GettingDark_up(Mat img) {

	//�̹��� ũ��
	int x = img.cols;
	int y = img.rows;
	//�⺻ ��ο����� ��Ⱚ
	int Darkness = 180;

	//�̹��� ä�� ���� ����
	if (img.channels() == 1) {
		for (int i = 0; i < y; i++) {
			for (int j = 0; j < x; j++) {

				//overflow ����
				if ((int)img.at<uchar>(i, j) < Darkness) img.at<uchar>(i, j) = 0;
				else img.at<uchar>(i, j) = img.at<uchar>(i, j) - Darkness;
			}
			if (i % 3 == 0) Darkness--;
			if (Darkness < 0) Darkness = 0;
		}
	}
	else {
		for (int i = 0; i < y; i++) {
			for (int j = 0; j < x; j++) {
			
				//overflow ���� -> �Ѿ�� 0->255�� ���� �ſ� Ƥ�� 
				if ((int)img.at<Vec3b>(i, j)[0] < Darkness)img.at<Vec3b>(i, j)[0] = 0;
				else img.at<Vec3b>(i, j)[0] = img.at<Vec3b>(i, j)[0]  - Darkness;

				if ((int)img.at<Vec3b>(i, j)[1] < Darkness)img.at<Vec3b>(i, j)[1] = 0;
				else img.at<Vec3b>(i, j)[1] = img.at<Vec3b>(i, j) [1] - Darkness;

				if ((int)img.at<Vec3b>(i, j)[2] < Darkness)img.at<Vec3b>(i, j)[2] = 0;
				else img.at<Vec3b>(i, j)[2] = img.at<Vec3b>(i, j)[2] - Darkness;
	
			}
			//�Ʒ��� ������ �� ��ο� ������
			if (i % 3 == 0) Darkness--;
			if (Darkness < 0) Darkness = 0;
		}
	}

}

//�Ʒ��� ������ ��ο� ������ 
void GettingDark_down(Mat img) {
	//�̹��� ũ��
	int x = img.cols;
	int y = img.rows;

	//�⺻ ��ο����� ��Ⱚ
	int Darkness = 50;

	//�̹��� ä�� ���� ����
	if (img.channels() == 1) {
		for (int i = 0; i < y; i++) {
			for (int j = 0; j < x; j++) {
				//overflow ���� -> �Ѿ�� 0->255�� ���� �ſ� Ƥ�� 
				if ((int)img.at<uchar>(i, j) < Darkness) img.at<uchar>(i, j) = 0;
				else img.at<uchar>(i, j) = img.at<uchar>(i, j) - Darkness;
			}
			if (i % 3 == 0) Darkness++;
			if (Darkness >255) Darkness = 255;
		}
	}
	else {
		for (int i = 0; i < y; i++) {
			for (int j = 0; j < x; j++) {

				//overflow ���� 
				if ((int)img.at<Vec3b>(i, j)[0] < Darkness)img.at<Vec3b>(i, j)[0] = 0;
				else img.at<Vec3b>(i, j)[0] = img.at<Vec3b>(i, j)[0] - Darkness;

				if ((int)img.at<Vec3b>(i, j)[1] < Darkness)img.at<Vec3b>(i, j)[1] = 0;
				else img.at<Vec3b>(i, j)[1] = img.at<Vec3b>(i, j)[1] - Darkness;

				if ((int)img.at<Vec3b>(i, j)[2] < Darkness)img.at<Vec3b>(i, j)[2] = 0;
				else img.at<Vec3b>(i, j)[2] = img.at<Vec3b>(i, j)[2] - Darkness;

			}
			//�Ʒ��� ������ ���� ��ο�������
			if (i % 3 == 0) Darkness++;
			if (Darkness > 255) Darkness = 255;
		}
	}

}

//������׷� �׷��� �̹��� ���� �Լ�
Mat getHistogram(Mat& src) {
	Mat histogram;
	// gray_scale ������ �۵� 
	const int* channel_numberes = { 0 };
	float channel_range[] = { 0.0, 255.0 };
	const float* channel_ranges = channel_range;
	int number_bins = 255; // �ִ밪 

	//�ȼ��� ���� ������׷� ���
	calcHist(&src, 1, channel_numberes, Mat(), histogram, 1, &number_bins, &channel_ranges);

	//������׷� �׷��� �̹��� ũ��
	int hist_w = 512;
	int hist_h = 400;
	int bin_w = cvRound((double)hist_w / number_bins);

	// histogram  ����ȭ
	normalize(histogram, histogram, 0, histogram.rows, NORM_MINMAX, -1, Mat());

	//���� ���� �մ� ���� �׸��� ����� plot �׷��� �̹���
	Mat histImage(hist_h, hist_w, CV_8UC1, Scalar(0, 0, 0));
	for (int i = 1; i < number_bins; i++) {
		line(histImage, Point(bin_w*(i - 1), hist_h - cvRound(histogram.at<float>(i - 1))),
			Point(bin_w*(i), hist_h - cvRound(histogram.at<float>(i))),
			Scalar(255, 0, 0), 2, 8, 0); 
	}

	return histImage;
}