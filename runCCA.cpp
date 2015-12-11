#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <vector>
#include "cca.hpp"
#include "kernel.hpp"

int main()
{	
	std::cout << "Tesing CCA" << std::endl;
	double temp = 0;
	cv::Mat testData = (cv::Mat_<double>(5,3) << 90, 60, 90, 90, 90, 30, 60 , 60, 60, 60, 60, 90, 30, 30, 30);
	cv::Mat covMat = covMatrix(testData);
	std::cout << "Input data is (each row is a sample):" << std::endl;
	std::cout << testData << std::endl;
	std::cout << "Covariance matrix is:" << std::endl;
	std::cout << covMat << std::endl;
	std::cout << '\n';
	cv::Mat Data0 = (cv::Mat_<double>(5,3) << 191, 36, 50, 189, 37, 52, 193, 38, 58, 162, 35, 62, 189,35, 46);
	cv::Mat Data1 = (cv::Mat_<double>(5,3) << 5, 162, 60, 2, 110, 60, 12, 101, 101, 12, 105, 37, 13, 155, 58);
	std::cout << "Data0 and Data1 are:" << std::endl;
	std::cout << Data0 << std::endl;
	std::cout << Data1 << std::endl;
	std::vector<cv::Mat> ccaOutput = cca(Data0, Data1, temp);
	/*
	std::cout << "Coefficients of CCA are:" << std::endl;
	std::cout << ccaOutput[0] << std::endl;
	std::cout << ccaOutput[1] << std::endl;
	std::cout << temp << std::endl;
	std::vector<double> kernelParameter;
	kernelParameter.push_back(0.1);
	*/
	//cv::Mat kccaOutput = kcca(Data0, Data1, linearKernel, kernelParameter, 0.5);
	//std::cout << "Coefficients of kCCA are:" << std::endl;
	//std::cout << kccaOutput << std::endl;

	return 0;
}
