#pragma once

#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

class STCTracker
{
public:
	STCTracker();
	~STCTracker();
	void init(const Mat frame, const Rect box,Rect &boxRegion);	
	void tracking(const Mat frame, Rect &trackBox,Rect &boxRegion,int FrameNum);

private:
	void createHammingWin();
	void complexOperation(const Mat src1, const Mat src2, Mat &dst, int flag = 0);
	void getCxtPriorPosteriorModel(const Mat image);
	void learnSTCModel(const Mat image);

private:
	double sigma;			// scale parameter (variance)
	double alpha;			// scale parameter
	double beta;			// shape parameter
	double rho;				// learning parameter
	double scale;			//	scale ratio
	double lambda;		//	scale learning parameter
	int num;					//	the number of frames for updating the scale
	vector<double> maxValue;
	Point center;			//	the object position
	Rect cxtRegion;		// context region
	int padding;
	
	Mat cxtPriorPro;		// prior probability
	Mat cxtPosteriorPro;	// posterior probability
	Mat STModel;			// conditional probability
	Mat STCModel;			// spatio-temporal context model
	Mat hammingWin;			// Hamming window
};