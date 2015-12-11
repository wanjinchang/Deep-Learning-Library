#ifndef MATRIX
#define MATRIX
#include <vector>
#include <math.h>
#include <iostream>
#include <cmath>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "util.hpp"

indexLabel maxIdxInRow(data d){
	size_t dataRows = d.rows;
	size_t dataCols = d.cols;
	indexLabel output;
	output.resize(dataRows);
	for (size_t i = 0; i < dataRows; i++){
		cv::Mat piece = d.row(i);
		double maxVal;
		cv::Point maxLoc;
		minMaxLoc(piece, NULL, &maxVal, NULL, &maxLoc, cv::Mat());
		output[i] = maxLoc.x;
	}
	return output;
}

std::vector<double> maxValueInRow(data d){
	size_t dataRows = d.rows;
	size_t dataCols = d.cols;
	std::vector<double> output;
	output.resize(dataRows);
	for (size_t i = 0; i < dataRows; i++){
		cv::Mat piece = d.row(i);
		double maxVal;
		cv::Point maxLoc;
		minMaxLoc(piece, NULL, &maxVal, NULL, &maxLoc, cv::Mat());
		std::cout << maxVal << std::endl;
	}
	return output;
}

cv::Mat probNormalize(cv::Mat input){
	cv::Mat partition;
	cv::reduce(input, partition, 1, CV_REDUCE_SUM, -1);
	cv::Mat output = cv::Mat::zeros(input.rows, input.cols, CV_64FC1);

	double *PtrO;
	double *PtrI;
	double *PtrP;
	for (int i = 0; i < output.rows; i++){
		PtrO = output.ptr<double>(i);
		PtrI = input.ptr<double>(i);
		PtrP = partition.ptr<double>(i);
		for (int j = 0; j < output.cols; j++){
			PtrO[j] = PtrI[j] / PtrP[0];
		}
	}
	return output;
}

cv::Mat matrixElementSquare(cv::Mat m)
{
	cv::Mat output(m.rows, m.cols, CV_64FC1);
	double *PtrM;
	double *PtrO;
	for (int i = 0; i < output.rows; i++){
		PtrO = output.ptr<double>(i);
		PtrM = m.ptr<double>(i);
		for (int j = 0; j < output.cols; j++){
			PtrO[j] = PtrM[j] * PtrM[j];
		}
	}
	return output;
}

cv::Mat matrixElementRoot(cv::Mat m)
{
	cv::Mat output(m.rows, m.cols, CV_64FC1);
	double *PtrM;
	double *PtrO;
	for (int i = 0; i < output.rows; i++){
		PtrO = output.ptr<double>(i);
		PtrM = m.ptr<double>(i);
		for (int j = 0; j < output.cols; j++){
			PtrO[j] = pow(PtrM[j], 0.5);
		}
	}
	return output;
}

cv::Mat matrixElementAbs(cv::Mat m)
{
	cv::Mat output(m.rows, m.cols, CV_64FC1);
	double *PtrM;
	double *PtrO;
	for (int i = 0; i < output.rows; i++){
		PtrO = output.ptr<double>(i);
		PtrM = m.ptr<double>(i);
		for (int j = 0; j < output.cols; j++){
			PtrO[j] = std::abs(PtrM[j]);
		}
	}
	return output;
}
#endif

