#ifndef BACKP
#define BACKP

#include <vector>
#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"

#include "layer.hpp"
#include "model.hpp"
#include "activation.hpp"
#include "loss.hpp"
#include "gd.hpp"
#include "util.hpp"
#include "processData.hpp"

void backpropagation(std::vector<RBMlayer>& model, flowOverLayer& act, cv::Mat& predictions, cv::Mat& labels, cv::Mat& inputData){
	size_t numLayer = model.size();
	cv::Mat deltaM;
	if (model[numLayer - 1].g_ACT() == softmaxACT){
		std::cout << "Backpropagation: using softmax + cross entropy." << std::endl;
		deltaM = predictions - labels;
	}
	else if (model[numLayer - 1].g_ACT() == sigmoidACT){
		std::cout << "Backpropagation: using sigmoid + mean square error." << std::endl;
		deltaM = (predictions - labels).mul((1 - predictions).mul(predictions));		
	}

	dataCollection deltaW;
	deltaW.resize(numLayer);
	dataCollection deltab;
	deltab.resize(numLayer);
	for (size_t i = 0; i < numLayer; i++){
		size_t lSize = model[i].g_size();
		size_t iSize = model[i].g_weight().rows;
		data wMatrix = cv::Mat::zeros(iSize, lSize, CV_64FC1);
		data bMatrix = cv::Mat::zeros(1, lSize, CV_64FC1);
		deltaW[i] = wMatrix;
		deltab[i] = bMatrix;
	}

	size_t numData = predictions.rows;
	for (size_t p = 0; p < numData; p++){
		cv::Mat delta = deltaM.row(p);
		for (int i = numLayer - 1; i > 0; i--){
			cv::Mat w = model[i].g_weight();
			cv::Mat preACT = act[i - 1].row(p);
			deltaW[i] += preACT.t() * delta;
			deltab[i] += delta;
			cv::Mat d_out_in = (1 - preACT).mul(preACT);
			delta = (delta * w.t()).mul(d_out_in);
		}
		cv::Mat preACT = inputData.row(p);
		deltaW[0] += preACT.t() * delta;
		deltab[0] += delta;
	}

	for (size_t i = 0; i < numLayer; i++){
		cv::Mat w = model[i].g_weight();
		model[i].s_weight(w - 0.01 * (deltaW[i]  + 0.002 * w));
		cv::Mat b = model[i].g_bias();
		model[i].s_bias(b - 0.01 * (deltab[i] ));
	}
}

void backpropagationAE(std::vector<RBMlayer>& model, flowOverLayer& act, cv::Mat& predictions, cv::Mat& labels, cv::Mat& inputData){
	size_t numLayer = model.size();
	cv::Mat deltaM;
	if (model[numLayer - 1].g_ACT() == softmaxACT){
		std::cout << "Backpropagation: using softmax + cross entropy." << std::endl;
		deltaM = predictions - labels;
	}
	else if (model[numLayer - 1].g_ACT() == sigmoidACT){
		std::cout << "Backpropagation: using sigmoid + mean square error." << std::endl;
		deltaM = (predictions - labels).mul((1 - predictions).mul(predictions));		
	}

	dataCollection deltaW;
	deltaW.resize(numLayer);
	dataCollection deltab;
	deltab.resize(numLayer);
	for (size_t i = 0; i < numLayer; i++){
		size_t lSize = model[i].g_size();
		size_t iSize = model[i].g_weight().rows;
		data wMatrix = cv::Mat::zeros(iSize, lSize, CV_64FC1);
		data bMatrix = cv::Mat::zeros(1, lSize, CV_64FC1);
		deltaW[i] = wMatrix;
		deltab[i] = bMatrix;
	}

	size_t numData = predictions.rows;
	for (size_t p = 0; p < numData; p++){
		cv::Mat delta = deltaM.row(p);
		for (int i = numLayer - 1; i > 0; i--){
			cv::Mat w = model[i].g_weight();
			cv::Mat preACT = act[i - 1].row(p);
			deltaW[i] += preACT.t() * delta;
			deltab[i] += delta;
			cv::Mat d_out_in = (1 - preACT).mul(preACT);
			delta = (delta * w.t()).mul(d_out_in);
		}
		cv::Mat preACT = inputData.row(p);
		deltaW[0] += preACT.t() * delta;
		deltab[0] += delta;
	}

	for (size_t i = 0; i < numLayer; i++){
		cv::Mat w = model[i].g_weight();
		model[i].s_weight(w - 0.01 * (deltaW[i] / numData + 0.002 * w));
		cv::Mat b = model[i].g_bias();
		model[i].s_bias(b - 0.01 * (deltab[i] / numData));
	}
}

#endif
