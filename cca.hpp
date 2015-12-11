#ifndef CCA
#define CCA
#include <vector>
#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "util.hpp"
#include "decomposition.hpp"

using namespace std;
//Here one sample is a row in input matrix.
cv::Mat covMatrix(cv::Mat input){
	int n = input.rows;
	int k = input.cols;
	cv::Mat iMatrix = cv::Mat::ones(n, n, CV_64FC1);
	input = input - iMatrix * input / n;
	return input.t() * input / (n - 1);
}

std::vector<cv::Mat> cca(cv::Mat &a, cv::Mat &b, double &corr){
	cv::Mat x;
	cv::hconcat(a, b, x);
	cv::Mat covMat = covMatrix(x);
	int aSize = a.cols;
	int bSize = b.cols;
	cv::Mat covAA = covMat(cv::Rect(0, 0, aSize, aSize));
	cv::Mat covAB = covMat(cv::Rect(aSize, 0, bSize, aSize));
	cv::Mat covBA = covMat(cv::Rect(0, aSize, aSize, bSize));
	cv::Mat covBB = covMat(cv::Rect(aSize, aSize, bSize, bSize));
	cv::Mat E1 = covAA.inv() * covAB * covBB.inv() * covBA;
	cv::Mat E2 = covBB.inv() * covBA * covAA.inv() * covAB;
	cv::Mat eigenValueE1;
	cv::Mat eigenValueE2;
	cv::Mat eigenVectorE1;
	cv::Mat eigenVectorE2;

	EigenvalueDecomposition ed1(E1);
	eigenVectorE1 = ed1.eigenvectors();
	eigenValueE1 = ed1.eigenvalues();

	EigenvalueDecomposition ed2(E2);
	eigenVectorE2 = ed2.eigenvectors();
	eigenValueE2 = ed2.eigenvalues();

	double maxValue = 0;
	int index = 0;
	for (int i = 0; i < eigenValueE1.cols; i++){
		if (eigenValueE1.at<double>(0, i) > maxValue){
			maxValue = eigenValueE1.at<double>(0, i);
			index = i;
		}
	}
	corr = eigenValueE1.at<double>(0, index);
	std::vector<cv::Mat> output;
	output.push_back(eigenVectorE1.col(index));

	maxValue = 0;
	index = 0;
	for (int i = 0; i < eigenValueE2.cols; i++){
		if (eigenValueE2.at<double>(0, i) > maxValue){
			maxValue = eigenValueE2.at<double>(0, i);
			index = i;
		}
	}
	output.push_back(eigenVectorE2.col(index));
	return output;
}


std::vector<cv::Mat> kcca(cv::Mat a, cv::Mat b, kernel k, std::vector<double> parameter, double reg){
	cv::Mat x;
	cv::Mat kernelA = k(a, a, parameter);
	cv::Mat kernelB = k(b, b, parameter);
	cv::Mat iMatrixA = cv::Mat::ones(a.rows, a.rows, CV_64FC1);
	cv::Mat iMatrixB = cv::Mat::ones(b.rows, b.rows, CV_64FC1);
	kernelA = kernelA - iMatrixA * kernelA - kernelA * iMatrixA + iMatrixA * kernelA * iMatrixA;
	kernelB = kernelB - iMatrixB * kernelB - kernelB * iMatrixB + iMatrixB * kernelB * iMatrixB;

	cv::Mat E1 = (kernelA + reg * iMatrixA).inv() * kernelB * (kernelB + reg * iMatrixB).inv() * kernelA;
	cv::Mat eigenValue;
	cv::Mat eigenVectorE1;
	cv::eigen(E1, eigenValue, eigenVectorE1);
	cv::Mat temp = eigenValue.at<double>(0,0) * (kernelB + reg * iMatrixB).inv() * kernelA * eigenVectorE1.row(0);
	int bSize = temp.cols;
	cv::Mat eigenVectorE2 = cv::Mat::zeros(eigenVectorE1.rows, bSize, CV_64FC1);
	for (int i = 0; i < eigenVectorE1.rows; i++){
		temp = eigenValue.at<double>(i,0) * (kernelB + reg * iMatrixB).inv() * kernelA * eigenVectorE1.row(i);
		for (int j = 0; j < bSize; j++){
			eigenVectorE2.at<double>(i,j) = temp.at<double>(0,j);
		}
	}
	std::vector<cv::Mat> output;
	output.push_back(eigenVectorE1);
	output.push_back(eigenVectorE2);
	std::cout << eigenValue << std::endl;
	return output;
}

#endif
