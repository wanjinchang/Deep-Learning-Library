#ifndef MODEL_STRUCTURE
#define MODEL_STRUCTURE
#include <vector>
#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"

class RBMModel{
	cv::Mat weight;
	cv::Mat nBias;
	cv::Mat pBias;
	int nSize;
	int pSize;

  public:
  	RBMModel(int n, int p);
  	RBMModel(cv::Mat &wt);
  	RBMModel(cv::Mat &wt, cv::Mat bn, cv::Mat bp);
  	cv::Mat getWeight() const {return weight;}
  	cv::Mat getnBias() const {return nBias;}
  	cv::Mat getpBias() const {return pBias;}
  	void setWeight(cv::Mat &w){
  		weight = w;
  		nSize = w.rows;
  		pSize = w.rows;
  		nBias = cv::Mat::zeros(1, nSize, CV_64FC1);
		pBias = cv::Mat::zeros(1, pSize, CV_64FC1);
  	}
};

RBMModel::RBMModel(int n, int p): nSize(n), pSize(p)
{
	cv::Mat w;
	w.create(n, p, CV_64FC1);
    cv::Mat mean = cv::Mat::zeros(1,1,CV_64FC1);
    cv::Mat sigma = cv::Mat::ones(1,1,CV_64FC1);
    cv::randn(w, mean, sigma);
	weight = 0.1 * w;
	nBias = cv::Mat::zeros(1, nSize, CV_64FC1);
	pBias = cv::Mat::zeros(1, pSize, CV_64FC1);
}

RBMModel::RBMModel(cv::Mat &wt): weight(wt), nSize(weight.rows), pSize(weight.cols)
{
	nBias = cv::Mat::zeros(1, nSize, CV_64FC1);
	pBias = cv::Mat::zeros(1, pSize, CV_64FC1);
}

RBMModel::RBMModel(cv::Mat &wt, cv::Mat bv, cv::Mat bh): weight(wt), nBias(bv), pBias(bh)
{
	nSize = weight.rows;
	pSize = weight.cols;
}

#endif
