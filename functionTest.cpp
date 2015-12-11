#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <math.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
//#include "computation/activation.hpp"
using namespace std;
using namespace cv;
int main()
{	
	
	//std::vector<cv::Mat> imageData = imread_mnist("t10k-images-idx3-ubyte", 100);
	//imshow(imageData, 5, 6);
	/*
	cv::Mat B = (cv::Mat_<double>(4,3) << 0.2, 0.2, 1.2, 0.7, 0.2, 0.4, 0.2, 0.2, 0.3, 1.1, 0.9, 0.1);
	std::cout << B << std::endl;
	cv::Mat covAA = B(cv::Rect(1, 1, 1, 2));
	std::cout << covAA << std::endl;
	/*
	cv::Mat C = (cv::Mat_<double>(4,3) << 0.1, 0.2, 0.3, 0.4, 0.2, 0.2, 0.2, 0.2, 0.2, 0.1, 0.1, 0.1);
	cv::Mat D = (cv::Mat_<double>(3,2) << 0.5, 0.6, 0.2, 0.2, 0.2, 0.2);
	cv::Mat E = (cv::Mat_<double>(1,2) << 0.15, 0.16);
	cv::Mat F = logisticDIST(C, D, E);
	std::vector<cv::Mat> dataSet;
	dataSet.push_back(B);
	dataSet.push_back(C);
	std::cout<< F <<std::endl;
	*/
	/*
	cv::Mat B = (cv::Mat_<double>(4,3) << 0.2, 0.2, 1.2, 0.7, 0.2, 0.4, 0.2, 0.2, 0.3, 1.1, 0.9, 0.1);
	cv::Mat C = (cv::Mat_<double>(4,3) << 0.1, 0.2, 0.3, 0.4, 0.2, 0.2, 0.2, 0.2, 0.2, 0.1, 0.1, 0.1);
	cout << B << endl;
	cout << C << endl;
	cv::hconcat(B, C, B);
	cout << B << endl;
	cv::Rect roi = cv::Rect(0, 1, 5, 1);
	cout<< B(roi);
	*/
	cv::Mat A = (cv::Mat_<double>(3,3) <<
                   1, 2, 3,
                   2, 4, 6,
                   3, 6, 9);
 
	cv::Mat B = (cv::Mat_<double>(3,1) <<
                   15, 
                   30,
                   45);
	cv::Mat x;
	cv::solve(A, B, x, DECOMP_SVD);
 
// printout the result
	cout << "Coefficients: " << endl << " " << x << endl << endl;
	
	/*
	vishid = [0.5 0.6; 0.2 0.2; 0.2 0.2];
	numcases = 4;
	hidbiases = [0.15 0.16];
	poshidprobs = 1./(1 + exp(-data*vishid - repmat(hidbiases,numcases,1)))
	err= sum(sum( (data-data2).^2 ))
	*/
	/*
	Error<cv::Mat> errorSum("square");
	cv::Mat tempError = errorSum.compute(C, B);
	cv::Mat errorTotal(1, 1, CV_64FC1);
	cv::reduce(tempError, errorTotal, 0, CV_REDUCE_SUM, -1);
	double stepError = errorTotal.at<double>(0,0);
	std::cout<< stepError <<std::endl;

	ConProb<cv::Mat> p (C, D, E);
	cv::Mat kkk = p.getPdf(logisticDIST);
	std::cout<< kkk <<std::endl;

	RBM rbm(3, 2);
	rbm.train(dataSet, 2, "binary");
	cv::Mat w = rbm.getWeight();

	std::cout<< w <<std::endl;
	
	cv::Mat tmpCount = (cv::Mat_<double>(2,6) << 0.2, 0.2, 1.2, 0.7, 0.2, 0.4, 0.2, -0.2, 0.3, 1.1, 0.9, 0.1);
	cv::Mat tmpCount1 = (cv::Mat_<double>(2,6) << 0.75, 0.2, 1.2, 0.7, 0.2, 0.4, 0.2, -0.2, 0.3, 1.1, 0.9, 0.1);
	cv::Mat out = tmpCount.mul(tmpCount1);
	
	double tmpCountMinVal = 0, tmpCountMaxVal = 0; 
	double minVal = 0, maxVal = 0; 
	cv::Point minPoint, maxPoint, minCoor, maxCoor; 
	cv::minMaxLoc(tmpCount.row(0), &tmpCountMinVal, &tmpCountMaxVal, &minPoint, &maxPoint);
	cv::Mat weightsLabel = cv::Mat::zeros(1, 2000, CV_64FC1);
	auto t0 = std::chrono::high_resolution_clock::now();
	//cv::Mat weightsLabel_T;
	//cv::transpose(weightsLabel, weightsLabel_T);
	//cv::minMaxLoc(tmpCount.row(1), &minVal, NULL, &minCoor,NULL);
	cv::Mat M(1, 2000, CV_64FC1, cv::Scalar(25.7));
	auto t3 = std::chrono::high_resolution_clock::now();
	cv::Mat d = M - weightsLabel;
	auto t1 = std::chrono::high_resolution_clock::now();
	//cv::Mat k = 25.7 * cv::Mat::zeros(2000, 2000, CV_64FC1);
	//cv::Mat weightsLabel_T1;
	//weightsLabel_T1 = weightsLabel.t();
	//cv::minMaxLoc(tmpCount1.row(1), &minVal, NULL, NULL, NULL);
	cv::Mat O(1, 2000, CV_64FC1);
	double *PtrO;
	double *PtrB;
	PtrB = weightsLabel.ptr<double>(0);
	PtrO = O.ptr<double>(0);
	for (int i = 0; i < O.cols; i++){
		PtrO[i] = 25.7 - PtrB[i];
	}
	auto t2 = std::chrono::high_resolution_clock::now();
	auto dt = 1.e-9*std::chrono::duration_cast<std::chrono::nanoseconds>(t1-t0).count();
	auto dt1 = 1.e-9*std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count();
	auto dt2 = 1.e-9*std::chrono::duration_cast<std::chrono::nanoseconds>(t3-t0).count();
	std::cout<<dt<<std::endl;
	std::cout<<dt1<<std::endl; 
	std::cout<<dt2<<std::endl;
	cv::minMaxLoc(tmpCount.row(1), 0, &maxVal, 0,& maxCoor);
	std::cout<<maxVal<<std::endl;
	*/
	//std::cout << out << std::endl;



    return 0;
}
