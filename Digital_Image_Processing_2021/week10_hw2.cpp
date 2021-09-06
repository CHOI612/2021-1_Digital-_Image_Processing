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

using namespace cv;
using namespace std;

void readImagesAndTimes(vector<Mat>& images, vector<float>& times);




void readImagesAndTimes(vector<Mat>& images, vector<float>& times) {
	int numImages = 4;
	static const float timesArray[] = { 1 / 500.0f,1 / 125.0f,1 / 30.0f, 0.1f, 0.25f, 1 / 6.0f, 2.0f };
	times.assign(timesArray, timesArray + numImages);
	static const char* filenames[] = { "./Gmail/img_0.002.jpg", "./Gmail/img_0.008.jpg","./Gmail/img_0.033.jpg", "./Gmail/img_0.1.jpg","./Gmail/img_0.25.jpg","./Gmail/img_0.17.jpg","./Gmail/img_0.2.jpg" };
	for (int i = 0; i < numImages; i++) {
		Mat im = imread(filenames[i]);
		images.push_back(im);

	}

}

Mat getHistogram(Mat& src) {
	Mat histogram;
	// gray_scale 에서만 작동 
	const int* channel_numberes = { 0 };
	float channel_range[] = { 0.0, 255.0 };
	const float* channel_ranges = channel_range;
	int number_bins = 255; // 최대값 

	//픽셀값 분포 히스토그램 계산
	calcHist(&src, 1, channel_numberes, Mat(), histogram, 1, &number_bins, &channel_ranges);

	//히스토그램 그래프 이미지 크기
	int hist_w = 512;
	int hist_h = 400;
	int bin_w = cvRound((double)hist_w / number_bins);

	// histogram  정규화
	normalize(histogram, histogram, 0, histogram.rows, NORM_MINMAX, -1, Mat());

	//값과 값을 잇는 선과 그리는 방식의 plot 그래프 이미지
	Mat histImage(hist_h, hist_w, CV_8UC1, Scalar(0, 0, 0));
	for (int i = 1; i < number_bins; i++) {
		line(histImage, Point(bin_w*(i - 1), hist_h - cvRound(histogram.at<float>(i - 1))),
			Point(bin_w*(i), hist_h - cvRound(histogram.at<float>(i))),
			Scalar(255, 0, 0), 2, 8, 0);
	}

	return histImage;
}

void main() {
	cout << "Reading images and exposure times ..." << endl;
	vector<Mat> images;
	vector<float> times;
	readImagesAndTimes(images, times);

	cout << "finished" << endl;

	//<영상 정렬>
	cout << "Aligning images ..." << endl;
	Ptr<AlignMTB> alignMTB = createAlignMTB();
	alignMTB->process(images, images);


	//<Camera response function(CRF) 복원>
	cout << "Calculating Camera Response function..." << endl;
	Mat responseDebevec;
	Ptr<CalibrateDebevec> calibrateDebevec = createCalibrateDebevec();
	calibrateDebevec->process(images, responseDebevec, times);
	cout << "-----CRF-----" << endl;
	cout << responseDebevec << endl;

	//<24 bit 표현 범위로 이미지 병합>
	cout << " Merging images into one HDR image ..." << endl;
	Mat hdrDebevec;
	Ptr<MergeDebevec> mergeDebevec = createMergeDebevec();
	mergeDebevec->process(images, hdrDebevec, times, responseDebevec);
	imwrite("hdrDebevec.hdr", hdrDebevec);
	cout << " saved hdrDebevec.hdr" << endl;


	//<<Drago 톤맵>>
	cout << "Tonemaping using Drago's method..." << endl;
	Mat IdrDrago;
	Ptr<TonemapDrago> tonemapDrago = createTonemapDrago(1.0f, 0.7f, 0.85f);
	tonemapDrago->process(hdrDebevec, IdrDrago);
	IdrDrago = 3 * IdrDrago;
	imwrite("Idr-Drago_2.jpg", IdrDrago * 255);
	cout << "save idr-Drago.jpg" << endl;

	//<<ReinHard 톤맵>>
	cout << "Tonemaping using Reinhard's method..." << endl;
	Mat IdrReinhard;
	Ptr<TonemapReinhard> tonemapReinhard = createTonemapReinhard(1.5f, 0,0,0);
	tonemapReinhard->process(hdrDebevec, IdrReinhard);
	imwrite("Idr-Reinhard_2.jpg", IdrReinhard * 255);
	cout << "save idr-Reinhard.jpg" << endl;

	//<<Mantiuk 톤맵>>
	cout << "Tonemaping using Mantiuk's method..." << endl;
	Mat IdrMantiuk;
	Ptr<TonemapMantiuk> tonemapMantiuk = createTonemapMantiuk(2.2f, 0.85f, 1.2f);
	tonemapMantiuk->process(hdrDebevec, IdrMantiuk);
	IdrMantiuk = 3 * IdrMantiuk;
	imwrite("Idr-Mantiuk_2.jpg", IdrMantiuk * 255);
	cout << "save idr-Mantiuk.jpg" << endl;
	
	cout << "finish" << endl;

}