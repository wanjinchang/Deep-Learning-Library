#ifndef INFERENCE
#define INFERENCE

#include <vector>
#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"

#include "layer.hpp"
#include "activation.hpp"
#include "loss.hpp"
#include "rbm.hpp"
#include "gd.hpp"
#include "processData.hpp"
using namespace std;
std::vector<cv::Mat> mfUpdate(std::vector<RBMlayer>& layers, cv::Mat &batchData, size_t updateEpoch);

void DBMinference(std::vector<RBMlayer>& layers, dataInBatch &data, size_t infEpoch){
	std::cout << "Begin DBM inference." << std::endl;
	double weightStep = 0.001;
	double biasStep = 0.001;
	double weightCost  = 0.002;

	size_t numLayer = layers.size();
	size_t numBatch = data.size();
	size_t batchSize = data[0].rows;
	size_t dimData = data[0].cols;
	actFunction activation = layers[0].g_ACT();
	//Update bias of each layer.
	for (size_t i = 0; i < numLayer - 1; i++){
		layers[i].s_bias(layers[i].g_bias() + layers[i + 1].g_biasInFlow());
	}
	//Randomly initialize M sample particles. Here M is the batchSize to facilitate computation.
	//Each particle is a collection of data in one negative and multiple positive layers.
	cv::Mat randomData;
	randomData.create(batchSize, dimData, CV_32SC1);
	cv::randu(randomData, cv::Scalar(0), cv::Scalar(2));
	randomData.convertTo(randomData, CV_64F);

	std::vector<cv::Mat> sampledParticle;
	sampledParticle.resize(layers.size() + 1);
	sampledParticle[0] = randomData;
	for (int i = 0; i < layers.size(); i++){
		RBMlayer currentLayer = layers[i];
		cv::Mat w = 2 * currentLayer.g_weight();
		cv::Mat b = currentLayer.g_bias();
		actFunction actLayer = currentLayer.g_ACT();
		cv::Mat input = sampledParticle[i];
		sampledParticle[i + 1] = actLayer(input, w, b);
	}
	//Begin inference
	for (size_t epoch = 0; epoch < infEpoch; epoch++){
		double epochError = 0;
		for (size_t batch = 0; batch < numBatch; batch++){
			cv::Mat batchData = data[batch];
			size_t batchSize = batchData.rows;
			//Variational Inference: Run mean-field updates to update its guess.
			std::vector<cv::Mat> mu = mfUpdate(layers, batchData, 1);
			std::vector<cv::Mat> hSample;
			hSample.resize(layers.size() + 1);
			cv::Mat input = batchData;
			// Stochastic Approximation
			actFunction actLayer = sigmoidACT;
			for (int i = 0; i < layers.size(); i++){
				RBMlayer currentLayer = layers[i];
				cv::Mat w = currentLayer.g_weight();
				cv::Mat b = currentLayer.g_bias();
				if (i == 0){
					cv::Mat hiddenState = sampledParticle[1];
					cv::Mat wn = w.t();
					cv::Mat bn = currentLayer.g_biasInFlow();
					cv::Mat visibleState = actLayer(hiddenState, wn, bn);
					hSample[0] = visibleState;
					w = 2 * w;
					hiddenState = actLayer(visibleState, w, b);
					hSample[1] = hiddenState;
				}
				else if (i < layers.size() - 1){
					w = 2 * w;
					hSample[i + 1] = actLayer(hSample[i], w, b);
				}
				else if (i == layers.size() - 1){
					hSample[i + 1] = actLayer(hSample[i], w, b);
				}
			}
			//Parameter Update:
			mu.insert(mu.begin(), batchData);

			for (int i = 0; i < numLayer; i++){
				cv::Mat weightGradient;
				weightGradient = (hSample[i].t() * hSample[i + 1] - mu[i].t() * mu[i + 1]) / batchSize;
				cv::Mat bStart;
				cv::Mat bEnd;
				cv::reduce(mu[i + 1], bStart, 0, CV_REDUCE_AVG, -1);
				cv::reduce(hSample[i + 1], bEnd, 0, CV_REDUCE_AVG, -1);
				cv::Mat biasGradient = bEnd - bStart;
				weightStep = annealRate(weightStep, epoch, step, 0.5);
				biasStep = annealRate(biasStep, epoch, step, 0.5);
				
				cv::Mat w = layers[i].g_weight();
				layers[i].s_weight(w - weightStep *  weightGradient);
				cv::Mat b = layers[i].g_bias();
				layers[i].s_bias(b - biasStep *  biasGradient);
			}
		}
	}
	
}

std::vector<cv::Mat> mfUpdate(std::vector<RBMlayer>& layers, cv::Mat &batchData, size_t updateEpoch){
	//mu[1] is the approximate probability q(positive layer 1 = 1)
	f activation = sigmoidF;
	size_t numLayer = layers.size();
	std::vector<cv::Mat> nu;
	nu.resize(numLayer);
	//get an initial guess of mu
	cv::Mat input = batchData;
	for (int i = 0; i < numLayer; i++){
		RBMlayer currentLayer = layers[i]; 
		cv::Mat w = 2 * currentLayer.g_weight();
		if (i == layers.size() - 1){
			w = currentLayer.g_weight();
		}
		cv::Mat b = currentLayer.g_bias();
		actFunction actLayer = currentLayer.g_ACT();
		input = actLayer(input, w, b);
		nu[i] = input;
	}

	//In mean-filed inference, update mu several times.
	std::vector<cv::Mat> mu = nu;
	for (size_t epo = 0; epo < updateEpoch; epo++){
		input = batchData;
		for (size_t i = 0; i < numLayer - 1; i++){
			cv::Mat product = input * layers[i].g_weight() + mu[i + 1] * layers[i + 1].g_weight().t();
			cv::Mat netIn = product + repeat(layers[i].g_bias(), product.rows, 1);
			mu[i] = activation(netIn);
			input = mu[i];
		}
		cv::Mat product = input * layers[numLayer - 1].g_weight();
		cv::Mat netIn = product + repeat(layers[numLayer - 1].g_bias(), product.rows, 1);
		activation = layers[numLayer - 1].g_ACTF();
		mu[numLayer - 1] = activation(netIn);
		//Check update step, break if it is trivial.
		int breakMark = 0;
		for (size_t j = 0; j < mu.size(); j++){
			cv::Mat difference = abs(nu[j] - mu[j]);
			double s = cv::sum(difference)[0];
			if (s < 0.0000001){
				std::cout << "Mean field update " << epo + 1 << " times." << std::endl;
				breakMark = 1;
				break;
			}
		}
		if (!breakMark){
			mu = nu;
		}
		else {
			break;
		}
	}
	return mu;
}
#endif


