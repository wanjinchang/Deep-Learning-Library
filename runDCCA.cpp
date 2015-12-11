#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgcodecs/imgcodecs.hpp"
#include "readCIFAR.hpp"
#include "loadData.hpp"
#include "visualization.hpp"
#include "rbm.hpp"
#include "testing.hpp"
#include "cca.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace cv;
int main()
{
	std::cout << "Read data" << std::endl;
    Mat trainX, testX;
    Mat trainY, testY;
    trainX = Mat::zeros(1024, 50000, CV_64FC1);  
    testX = Mat::zeros(1024, 10000, CV_64FC1);  
    trainY = Mat::zeros(1, 50000, CV_64FC1);  
    testY = Mat::zeros(1, 10000, CV_64FC1);
    imread_cifar(trainX, testX, trainY, testY);
    trainX = trainX.t();
    testX = testX.t();
    Mat testY1K(0, 10, CV_64FC1);
    Mat trainY1K(0, 10, CV_64FC1);
    double * rptr;
    rptr = trainY.ptr<double>(0);
    for (int i = 0; i < 50000; i++){
    	trainY1K.push_back(oneOfK(rptr[i], 10));
    }
    rptr = testY.ptr<double>(0);
    for (int i = 0; i < 10000; i++){
    	testY1K.push_back(oneOfK(rptr[i], 10));
    }
	trainY1K.convertTo(trainY1K, CV_64FC1);
	testY1K.convertTo(testY1K, CV_64FC1);

	cv::Mat leftHalfTrain = trainX(cv::Rect(0,0,512, 50000));
	cv::Mat rightHalfTrain = trainX(cv::Rect(512,0,512, 50000));
	cv::Mat leftHalfTest = testX(cv::Rect(0,0,512, 10000));
	cv::Mat rightHalfTest = testX(cv::Rect(512,0, 512, 10000));
	leftHalfTrain.convertTo(leftHalfTrain, CV_64FC1);
	rightHalfTrain.convertTo(rightHalfTrain, CV_64FC1);
	cout << leftHalfTrain.rows << " " << leftHalfTrain.cols << endl;
	cout << rightHalfTrain.rows << " " << rightHalfTrain.cols << endl;

    std::cout << "Make batches" << std::endl;
	dataInBatch training1 = imageBatches(leftHalfTrain, 100);
	dataInBatch training2 = imageBatches(rightHalfTrain, 100);
	dataInBatch test1 = imageBatches(leftHalfTest, 100);
	dataInBatch test2 = imageBatches(rightHalfTest, 100);

    std::cout << "Running of RBM begins" << std::endl;
	RBM rbm1(512, 20, 0);
	rbm1.dropout(0.2);
	rbm1.singleTrainBinary(training1, 8);
	dataInBatch modelOut1 = rbm1.g_activation(training1);

	RBM rbm2(512, 20, 0);
	rbm2.dropout(0.2);
	rbm2.singleTrainBinary(training2, 8);
	dataInBatch modelOut2 = rbm2.g_activation(training1);

	double corr = 0;
	dataInBatch k;
	k = cca(modelOut1[0], modelOut2[0], corr);
	cout << corr << endl;

    return 0;
}