/* Author:Jian Jin
RBM class has method for training autoencoder and DBM.
To indicate direction in stacked RBMS, the hidden layer of RBM is denoted as positive,
the visible layer of RBM is denoted as negative, which shows that from visible layer to hidden 
layer is a positive direction.
*/
#ifndef RMB
#define RMB

#include <vector>
#include <iostream>
#include <chrono>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"

#include "layer.hpp"
#include "model.hpp"
#include "activation.hpp"
#include "loss.hpp"
#include "gd.hpp"
#include "util.hpp"
#include "matrix.hpp"
#include "processData.hpp"

class RBM {
	cv::Mat weight;
	cv::Mat weightLabel;
	cv::Mat biasNegative;
	cv::Mat biasPositive;
	cv::Mat biasLabel;
	dataInBatch emission;
	dataInBatch reconstruction; 
	size_t numNegativeNeuro;
	size_t numPositiveNeuro;
	size_t batchSize;
	size_t classDimension;
	double dropoutRate = 0;
	actFunction activation;
	actFunction_multi activation_multi;
	LossType errorType;
	annealType annealT = step;
	std::vector<double> error;
	int layerIndex;

  public:
  	explicit RBM(size_t nSize, size_t pSize, int idx);
  	explicit RBM(cv::Mat wt, int idx);
  	explicit RBM(cv::Mat wt, cv::Mat nBias, cv::Mat pBias, int idx);
  	explicit RBM(RBMModel m, int idx);
  	cv::Mat g_weight() const {return weight;}
  	cv::Mat g_biasNegative() const {return biasNegative;}
  	cv::Mat g_biasPositive() const {return biasPositive;}
  	cv::Mat g_labelWeight() const {return weightLabel;}
  	cv::Mat g_biasLabel() const {return biasLabel;}
  	dataInBatch g_emission() const {return emission;}
  	dataInBatch g_reconstruction() const {return reconstruction;}
  	dataInBatch g_activation(dataInBatch& input, ActivationType at = sigmoid_t);
  	void s_errorType(LossType l){errorType = l;}
  	void s_annealType(annealType i){annealT = i;}
  	void dropout(double i) {dropoutRate = i;}
  	void doubleTrain(dataInBatch &trainingSet, int numEpoch, int inLayer, ActivationType at = sigmoid_t, LossType lt = MSE, GDType gd = SGD, int numGibbs = 1);
  	void doubleClassifier(dataInBatch &trainingSet, dataInBatch &labelSet, int numEpoch, ActivationType ac = sigmoid_t, LossType lt = MSE, GDType gd = SGD, int numGibbs = 1);
  	void singleTrainBinary(dataInBatch &trainingSet, int numEpoch, ActivationType at = sigmoid_t, LossType lt = MSE, GDType gd = SGD, int numGibbs = 1);
  	void singleTrainLinear(dataInBatch &trainingSet, int numEpoch, ActivationType at= sigmoid_t, LossType lt = MSE, GDType gd = SGD, int numGibbs = 1);
  	void singleClassifier(dataInBatch &modelOut, dataInBatch &labelSet, int numEpoch, GDType gd = SGD);
};

//RBM had four type of constructor: declare size of both layers, declare its weight, declare weight, bias of both layers, and use RBMModel as argument 
RBM::RBM(size_t nSize, size_t pSize, int idx): numNegativeNeuro(nSize), numPositiveNeuro(pSize), layerIndex(idx) {
	cv::Mat w(numNegativeNeuro, numPositiveNeuro, CV_64FC1);
    cv::Mat mean = cv::Mat::zeros(1,1,CV_64FC1);
    cv::Mat sigma = cv::Mat::ones(1,1,CV_64FC1);
    cv::randn(w, mean, sigma);
	weight = 0.01 * w;
	biasNegative = cv::Mat::zeros(1, numNegativeNeuro, CV_64FC1);
	biasPositive = cv::Mat::zeros(1, numPositiveNeuro, CV_64FC1);
	error.clear();
}

RBM::RBM(cv::Mat wt, int idx): weight(wt), numNegativeNeuro(wt.rows), numPositiveNeuro(wt.cols), layerIndex(idx) {
	biasNegative = cv::Mat::zeros(1, numNegativeNeuro, CV_64FC1);
	biasPositive = cv::Mat::zeros(1, numPositiveNeuro, CV_64FC1);
	layerIndex = idx;
	error.clear();
}

RBM::RBM(cv::Mat wt, cv::Mat nBias, cv::Mat pBias, int idx): 
weight(wt), biasNegative(nBias), biasPositive(pBias), numNegativeNeuro(wt.rows), numPositiveNeuro(wt.cols), layerIndex(idx) {
	error.clear();
}

RBM::RBM(RBMModel m, int idx): layerIndex(idx) {
	weight = m.getWeight();
	biasNegative = m.getnBias();
	biasPositive = m.getpBias();
	numNegativeNeuro = weight.rows;
	numPositiveNeuro = weight.cols;
	error.clear();
}

dataInBatch RBM::g_activation(dataInBatch& input, ActivationType actType){

	switch (actType)
	{
		case sigmoid_t: activation = sigmoidACT; break;
		case tanh_t: activation = tanhACT; break;
		case relu_t: activation = reluACT; break;
		case leaky_relu_t: activation = leaky_reluACT; break;
		case softmax_t: activation = softmaxACT; break;
		default: activation = identityACT;
	}
	dataInBatch output;
	output.resize(input.size());
	for (int i = 0; i < input.size(); i++){
		cv::Mat batchIn = input[i];
		cv::Mat batchOut = activation(batchIn, weight, biasPositive);
		output[i] = batchOut;
	}
	return output;
}

cv::Mat sample(cv::Mat distribution);
cv::Mat gaussian(cv::Mat &distribution);
cv::Mat dropoutMask(size_t rowSize, size_t colSize, double rate);

//train an RBM for autoencoder with binary units in hidden layer.
void RBM::singleTrainBinary(dataInBatch &trainingSet, int numEpoch, ActivationType act, LossType loss, GDType gd, int numGibbs)
{	
	error.clear(); emission.clear();
	cv::Mat weightUpdate = cv::Mat::zeros(numNegativeNeuro, numPositiveNeuro, CV_64FC1);
	cv::Mat biasPositiveUpdate = cv::Mat::zeros(1, numPositiveNeuro, CV_64FC1);
	cv::Mat biasNegativeUpdate = cv::Mat::zeros(1, numNegativeNeuro, CV_64FC1);
	double weightStep = 0.1;
	double biasPositiveStep = 0.1;
	double biasNegativeStep = 0.1;
	double weightCost  = 0.02;

//set activation function in both layers
	switch (act)
	{
		case sigmoid_t: activation = sigmoidACT; break;
		case tanh_t: activation = tanhACT; break;
		case relu_t: activation = reluACT; break;
		case leaky_relu_t: activation = leaky_reluACT; break;
		case softmax_t: activation = softmaxACT; break;
		default: activation = identityACT;
	}

	size_t numBatch = trainingSet.size();
	emission.resize(numBatch);
	reconstruction.resize(numBatch);
	std::vector<cv::Mat> updateStorage;
	updateStorage.resize(3);

/*
updateStorage vector stores in-process data while traing.
For SGD, in momentum method it will stores previous update when computing next update.
For AdaGrad, it will store the square sum of previous gradients, thus for update value of size k,
only space to store k values is required.
*/
	updateStorage[0] = cv::Mat::zeros(numNegativeNeuro, numPositiveNeuro, CV_64FC1);
	updateStorage[1] = cv::Mat::zeros(1, numNegativeNeuro, CV_64FC1);
	updateStorage[2] = cv::Mat::zeros(1, numPositiveNeuro, CV_64FC1);
	
	auto t01 = std::chrono::high_resolution_clock::now();
	for (int epoch = 0; epoch < numEpoch; epoch++){
		double epochError = 0;
		for (int batch = 0; batch < numBatch; batch++){
			data image = trainingSet[batch];
			batchSize = image.rows;
			cv::Mat mask;
			if (dropoutRate > 0){
				mask = dropoutMask(batchSize, numPositiveNeuro, dropoutRate);
			}

			cv::Mat weight_T;
			cv::transpose(weight, weight_T);

			cv::Mat input = image;
			cv::Mat prob_p = activation(input, weight, biasPositive);
			cv::Mat activation_p = sample(prob_p);

			if (epoch == numEpoch - 1) {
				emission[batch] = prob_p;
			}

			if (dropoutRate > 0){
				activation_p = activation_p.mul(mask);
			}

			cv::Mat prob_n = activation(activation_p, weight_T, biasNegative);
			cv::Mat activation_n = sample(prob_n);

			for (int i = 1; i < numGibbs; i++){
				prob_p = activation(activation_n, weight, biasPositive);
				activation_p = sample(prob_p);
				if (dropoutRate > 0){
					activation_p = activation_p.mul(mask);
				}

				prob_n = activation(activation_p, weight_T, biasNegative);
				activation_n = sample(prob_n);
			}
			cv::Mat prob_rec = activation(activation_n, weight, biasPositive);

			if (epoch == numEpoch - 1) {
				reconstruction[batch] = prob_n;
			}

			/*
			There are multiple error funtion type available:
			"MSE": mean square error
			"0-1": binaty loss
			"ABS": absolute loss
			if not indicates: cross entropy loss
			If not indicated in RBM train method, default type of training a autoencoder is mse.			
			*/

			Error<cv::Mat> error(errorType);
			cv::Mat temp = error.value(input, activation_n);
			cv::Mat total(1, 1, CV_64FC1);
			cv::reduce(temp, total, 0, CV_REDUCE_SUM, -1);
			double stepError = total.at<double>(0,0);
			epochError += stepError;
			//check stepError
			//std::cout << stepError << std::endl;

			cv::Mat nStart;
			cv::Mat nEnd;
			cv::Mat pStart;
			cv::Mat pEnd;
			cv::reduce(input, nStart, 0, CV_REDUCE_AVG, -1);
			cv::reduce(activation_n, nEnd, 0, CV_REDUCE_AVG, -1);
			cv::reduce(prob_p, pStart, 0, CV_REDUCE_AVG, -1);
			cv::reduce(prob_rec, pEnd, 0, CV_REDUCE_AVG, -1);

			//compute gradient
			double momentum =  epoch > 5 ? 0.9: 0.5;
			cv::Mat weightGradient = (activation_n.t() * prob_rec - input.t() * prob_p) / batchSize;
			cv::Mat biasPositiveGradient = pEnd - pStart;
			cv::Mat biasNegativeGradient = nEnd - nStart;

			//anneal learning rate
			weightStep = annealRate(weightStep, epoch, annealT, 0.5);
			biasNegativeStep = annealRate(biasNegativeStep, epoch, annealT, 0.5);
			biasPositiveStep = annealRate(biasPositiveStep, epoch, annealT, 0.5);

			cv::Mat weightUpdate;
			cv::Mat biasNegativeUpdate;

			/*
			Two update methos:
			"AdaGrad": adaptive gradient
			Not indicated: default is SGD
			In SGD, there are two type of regularized term for weight decay: "L1", "L2".
			*/
			cv::Mat biasPositiveUpdate;
			if (gd == AdaGrad){
			weightUpdate = updateAdaGrad(weightGradient, updateStorage[0], weightStep);
			biasNegativeUpdate = updateAdaGrad(biasNegativeGradient, updateStorage[1], biasNegativeStep);
			biasPositiveUpdate = updateAdaGrad(biasPositiveGradient, updateStorage[2], biasPositiveStep);
			}
			else {
			WeightDecayPenalty p(L2, 0.1, weight);
			weightUpdate = updateSGD(updateStorage[0], weightGradient, momentum, weightStep, p);
			biasNegativeUpdate = updateSGD(updateStorage[1], biasNegativeGradient, momentum, biasNegativeStep);
			biasPositiveUpdate = updateSGD(updateStorage[2], biasPositiveGradient, momentum, biasPositiveStep);
			updateStorage[0] = weightUpdate;
			updateStorage[1] = biasNegativeUpdate;
			updateStorage[2] = biasPositiveUpdate;
			}

			weight += weightUpdate;
			biasNegative += biasNegativeUpdate;
			biasPositive += biasPositiveUpdate;
		}
		error.push_back(epochError);
		std::cout << "Train RBM layer " << layerIndex << ", epoch " << epoch << ", epoch error: " << epochError << std::endl;
	}
	auto t02 = std::chrono::high_resolution_clock::now();
	auto dk = 1.e-9*std::chrono::duration_cast<std::chrono::nanoseconds>(t02-t01).count();
	std::cout << "total time: " << dk << std::endl;
}

//train an RBM for autoencoder with linear units in hidden layer.
void RBM::singleTrainLinear(dataInBatch &trainingSet, int numEpoch, ActivationType act, LossType loss, GDType gd, int numGibbs)
{	
	error.clear(); emission.clear();
	cv::Mat weightUpdate = cv::Mat::zeros(numNegativeNeuro, numPositiveNeuro, CV_64FC1);
	cv::Mat biasPositiveUpdate = cv::Mat::zeros(1, numPositiveNeuro, CV_64FC1);
	cv::Mat biasNegativeUpdate = cv::Mat::zeros(1, numNegativeNeuro, CV_64FC1);
	double weightStep = 0.1;
	double biasPositiveStep = 0.1;
	double biasNegativeStep = 0.1;
	double weightCost  = 0.02;

//set activation function in both layers
	switch (act)
	{
		case sigmoid_t: activation = sigmoidACT; break;
		case tanh_t: activation = tanhACT; break;
		case relu_t: activation = reluACT; break;
		case leaky_relu_t: activation = leaky_reluACT; break;
		case softmax_t: activation = softmaxACT; break;
		default: activation = identityACT;
	}

	size_t numBatch = trainingSet.size();
	emission.resize(numBatch);
	reconstruction.resize(numBatch);
	std::vector<cv::Mat> updateStorage;
	updateStorage.resize(3);

/*
updateStorage vector stores in-process data while traing.
For SGD, in momentum method it will stores previous update when computing next update.
For AdaGrad, it will store the square sum of previous gradients, thus for update value of size k,
only space to store k values is required.
*/
	updateStorage[0] = cv::Mat::zeros(numNegativeNeuro, numPositiveNeuro, CV_64FC1);
	updateStorage[1] = cv::Mat::zeros(1, numNegativeNeuro, CV_64FC1);
	updateStorage[2] = cv::Mat::zeros(1, numPositiveNeuro, CV_64FC1);
	
	for (int epoch = 0; epoch < numEpoch; epoch++){
		double epochError = 0;
		for (int batch = 0; batch < numBatch; batch++){
			data image = trainingSet[batch];
			batchSize = image.rows;
			cv::Mat mask;
			if (dropoutRate > 0){
				mask = dropoutMask(batchSize, numPositiveNeuro, dropoutRate);
			}

			cv::Mat weight_T;
			cv::transpose(weight, weight_T);

			cv::Mat input = image;
			cv::Mat prob_p = identityACT(input, weight, biasPositive);
			cv::Mat activation_p = gaussian(prob_p);

			if (epoch == numEpoch - 1) {
				emission[batch] = prob_p;
			}

			if (dropoutRate > 0){
				activation_p = activation_p.mul(mask);
			}

			cv::Mat activation_n = activation(activation_p, weight_T, biasNegative);
			//cv::Mat activation_n = sample(prob_n);

			for (int i = 1; i < numGibbs; i++){
				cv::Mat prob_p = identityACT(activation_n, weight, biasPositive);
				cv::Mat activation_p = gaussian(prob_p);
				if (dropoutRate > 0){
					activation_p = activation_p.mul(mask);
				}

				activation_n = activation(activation_p, weight_T, biasNegative);
				//activation_n = sample(prob_n);
			}
			cv::Mat prob_rec = identityACT(activation_n, weight, biasPositive);

			cv::Mat input_T;
			cv::transpose(input, input_T);
			cv::Mat activation_n_T;
			cv::transpose(activation_n, activation_n_T);

			Error<cv::Mat> error(errorType);
			cv::Mat temp = error.value(input, activation_n);
			cv::Mat total(1, 1, CV_64FC1);
			cv::reduce(temp, total, 0, CV_REDUCE_SUM, -1);
			double stepError = total.at<double>(0,0);
			epochError += stepError;
			//check stepError
			//std::cout << stepError << std::endl;

			cv::Mat nStart;
			cv::Mat nEnd;
			cv::Mat pStart;
			cv::Mat pEnd;
			cv::reduce(input, nStart, 0, CV_REDUCE_AVG, -1);
			cv::reduce(activation_n, nEnd, 0, CV_REDUCE_AVG, -1);
			cv::reduce(prob_p, pStart, 0, CV_REDUCE_AVG, -1);
			cv::reduce(prob_rec, pEnd, 0, CV_REDUCE_AVG, -1);

			//compute gradient
			double momentum =  epoch > 5 ? 0.9: 0.5;
			cv::Mat weightGradient = (activation_n_T * prob_rec - input_T * prob_p) / batchSize;
			cv::Mat biasPositiveGradient = pEnd - pStart;
			cv::Mat biasNegativeGradient = nEnd - nStart;

			//anneal learning rate
			weightStep = annealRate(weightStep, epoch, annealT, 0.5);
			biasNegativeStep = annealRate(biasNegativeStep, epoch, annealT, 0.5);
			biasPositiveStep = annealRate(biasPositiveStep, epoch, annealT, 0.5);

			cv::Mat weightUpdate;
			cv::Mat biasNegativeUpdate;

			/*
			Two update methos:
			"AdaGrad": adaptive gradient
			Not indicated: default is SGD
			In SGD, there are two type of regularized term for weight decay: "L1", "L2".
			*/
			cv::Mat biasPositiveUpdate;
			if (gd == AdaGrad){
			weightUpdate = updateAdaGrad(weightGradient, updateStorage[0], weightStep);
			biasNegativeUpdate = updateAdaGrad(biasNegativeGradient, updateStorage[1], biasNegativeStep);
			biasPositiveUpdate = updateAdaGrad(biasPositiveGradient, updateStorage[2], biasPositiveStep);
			}
			else {
			WeightDecayPenalty p(L2, 0.1, weight);
			weightUpdate = updateSGD(updateStorage[0], weightGradient, momentum, weightStep, p);
			biasNegativeUpdate = updateSGD(updateStorage[1], biasNegativeGradient, momentum, biasNegativeStep);
			biasPositiveUpdate = updateSGD(updateStorage[2], biasPositiveGradient, momentum, biasPositiveStep);
			updateStorage[0] = weightUpdate;
			updateStorage[1] = biasNegativeUpdate;
			updateStorage[2] = biasPositiveUpdate;
			}

			weight += weightUpdate;
			biasNegative += biasNegativeUpdate;
			biasPositive += biasPositiveUpdate;
		}
		error.push_back(epochError);
		std::cout << "Train RBM layer " << layerIndex << ", epoch " << epoch << ", epoch error: " << epochError << std::endl;
	}
}

void RBM::doubleTrain(dataInBatch &trainingSet, int numEpoch, int inLayer, ActivationType act, LossType loss, GDType gd, int numGibbs)
{	
	error.clear(); emission.clear();
	cv::Mat weightUpdate = cv::Mat::zeros(numNegativeNeuro, numPositiveNeuro, CV_64FC1);
	cv::Mat biasPositiveUpdate = cv::Mat::zeros(1, numPositiveNeuro, CV_64FC1);
	cv::Mat biasNegativeUpdate = cv::Mat::zeros(1, numNegativeNeuro, CV_64FC1);
	double weightStep = 0.05;
	double biasPositiveStep = 0.05;
	double biasNegativeStep = 0.05;
	double weightCost  = 0.0002;

	switch (act)
	{
		case sigmoid_t: activation = sigmoidACT; break;
		case tanh_t: activation = tanhACT; break;
		case relu_t: activation = reluACT; break;
		case leaky_relu_t: activation = leaky_reluACT; break;
		case softmax_t: activation = softmaxACT; break;
		default: activation = identityACT;
	}

	size_t numBatch = trainingSet.size();
	emission.resize(numBatch);
	std::vector<cv::Mat> updateStorage;
	updateStorage.resize(3);

	updateStorage[0] = cv::Mat::zeros(numNegativeNeuro, numPositiveNeuro, CV_64FC1);
	updateStorage[1] = cv::Mat::zeros(1, numNegativeNeuro, CV_64FC1);
	updateStorage[2] = cv::Mat::zeros(1, numPositiveNeuro, CV_64FC1);	
	
	std::cout << "Pretrain RBM layer." << std::endl;
	for (int epoch = 0; epoch < numEpoch; epoch++){
		double epochError = 0;
		for (int batch = 0; batch < numBatch; batch++){
			cv::Mat data = trainingSet[batch];
			batchSize = data.rows;

			cv::Mat toPositiveW;
			cv::Mat toNegativeW;
			cv::Mat biasP;
			cv::Mat biasN;
			if (inLayer > 0){
				toPositiveW = weight;
				toNegativeW = 2 * weight.t();
				biasP = biasPositive;
				biasN = 2 * biasNegative;
			}
			else if (inLayer < 0){
				toPositiveW = 2 * weight;
				toNegativeW = weight.t();
				biasP = 2 * biasPositive;
				biasN = biasNegative;
			}
			else {
				toPositiveW = 2 * weight;
				toNegativeW = 2 * weight.t();
				biasP = 2 * biasPositive;
				biasN = 2 * biasNegative;
			}

			cv::Mat input = sample(data);
			cv::Mat prob_p = activation(input, toPositiveW, biasP);
			cv::Mat activation_p = sample(prob_p);

			if (epoch == numEpoch - 1) {
				emission[batch] = prob_p;
			}

			cv::Mat prob_n = activation(activation_p, toNegativeW, biasN);
			cv::Mat activation_n = sample(prob_n);

			for (int i = 1; i < numGibbs; i++){
				prob_p = activation(input, toPositiveW, biasP);
				activation_p = sample(prob_p);

				prob_n = activation(activation_p, toNegativeW, biasN);
				activation_n = sample(prob_n);
			}

			cv::Mat prob_rec = activation(activation_n, toPositiveW, biasP);

			cv::Mat input_T;
			cv::transpose(input, input_T);
			cv::Mat activation_n_T;
			cv::transpose(activation_n, activation_n_T);

			Error<cv::Mat> error(loss);
			cv::Mat temp = error.value(input, activation_n);
			cv::Mat total(1, 1, CV_64FC1);
			cv::reduce(temp, total, 0, CV_REDUCE_SUM, -1);
			double stepError = total.at<double>(0,0);
			epochError += stepError;

			cv::Mat nStart;
			cv::Mat nEnd;
			cv::Mat pStart;
			cv::Mat pEnd;
			cv::reduce(input, nStart, 0, CV_REDUCE_AVG, -1);
			cv::reduce(activation_n, nEnd, 0, CV_REDUCE_AVG, -1);
			cv::reduce(prob_p, pStart, 0, CV_REDUCE_AVG, -1);
			cv::reduce(prob_rec, pEnd, 0, CV_REDUCE_AVG, -1);

			double momentum =  epoch > 5 ? 0.9: 0.5;
			cv::Mat weightGradient = (activation_n_T * prob_rec - input_T * prob_p) / batchSize;
			cv::Mat biasPositiveGradient = pEnd - pStart;
			cv::Mat biasNegativeGradient = nEnd - nStart;

			weightStep = annealRate(weightStep, epoch, annealT, 0.5);
			biasNegativeStep = annealRate(biasNegativeStep, epoch, annealT, 0.5);
			biasPositiveStep = annealRate(biasPositiveStep, epoch, annealT, 0.5);

			cv::Mat weightUpdate;
			cv::Mat biasNegativeUpdate;
			cv::Mat biasPositiveUpdate;

			if (gd == AdaGrad){
			weightUpdate = updateAdaGrad(weightGradient, updateStorage[0], weightStep);
			biasNegativeUpdate = updateAdaGrad(biasNegativeGradient, updateStorage[1], biasNegativeStep);
			biasPositiveUpdate = updateAdaGrad(biasPositiveGradient, updateStorage[2], biasPositiveStep);
			}
			else {
			WeightDecayPenalty p(L2, 0.1, weight);
			weightUpdate = updateSGD(updateStorage[0], weightGradient, momentum, weightStep, p);
			biasNegativeUpdate = updateSGD(updateStorage[1], biasNegativeGradient, momentum, biasNegativeStep);
			biasPositiveUpdate = updateSGD(updateStorage[2], biasPositiveGradient, momentum, biasPositiveStep);
			updateStorage[0] = weightUpdate;
			updateStorage[1] = biasNegativeUpdate;
			updateStorage[2] = biasPositiveUpdate;
			}

			weight += weightUpdate;
			biasNegative += biasNegativeUpdate;
			biasPositive += biasPositiveUpdate;
		}
		error.push_back(epochError);
		std::cout << "Trained RBM layer " << layerIndex << ", epoch " << epoch << ", epoch error: " << epochError << std::endl;
	}
}

void RBM::doubleClassifier(dataInBatch &trainingSet, dataInBatch &labelSet, int numEpoch, ActivationType act, LossType loss, GDType gd, int numGibbs)
{	
	error.clear(); emission.clear();
	classDimension = labelSet[0].cols;
	cv::Mat weightUpdate = cv::Mat::zeros(numNegativeNeuro, numPositiveNeuro, CV_64FC1);
	cv::Mat weightLabelUpdate = cv::Mat::zeros(classDimension, numPositiveNeuro, CV_64FC1);
	cv::Mat biasLabelUpdate = cv::Mat::zeros(1, classDimension, CV_64FC1);
	cv::Mat biasPositiveUpdate = cv::Mat::zeros(1, numPositiveNeuro, CV_64FC1);
	cv::Mat biasNegativeUpdate = cv::Mat::zeros(1, numNegativeNeuro, CV_64FC1);
	double weightStep = 0.05;
	double weightLabelStep = 0.05;
	double biasPositiveStep = 0.05;
	double biasNegativeStep = 0.05;
	double biasLabelStep = 0.05;
	double weightCost  = 0.0002;

	biasLabel = cv::Mat::zeros(1, classDimension, CV_64FC1);
	weightLabel.create(classDimension, numPositiveNeuro, CV_64FC1);
    cv::Mat mean = cv::Mat::zeros(1,1,CV_64FC1);
    cv::Mat sigma = cv::Mat::ones(1,1,CV_64FC1);
    cv::randn(weightLabel, mean, sigma);
	weightLabel = 0.01 * weightLabel;

	f activationF;
	switch (act)
	{
		case sigmoid_t: 
			activation = sigmoidACT;
			activation_multi = sigmoid_multi; 
			activationF = sigmoidF;
			break;
		case tanh_t:
			activation = tanhACT;
			activation_multi = tanh_multi;
			activationF = tanhF;
			break;
		case relu_t: 
			activation = reluACT;
			activation_multi = relu_multi;
			activationF = reluF;
			break;
		case leaky_relu_t: 
			activation = leaky_reluACT;
			activation_multi = leaky_relu_multi;
			activationF = leaky_reluF;
			break;
		case softmax_t: 
			activation = softmaxACT;
			activation_multi = softmax_multi;
			activationF = softmaxF;
			break;
		default: 
			activation = identityACT;
			activation_multi = identity_multi;
			activationF = identityF;
	}

	size_t numBatch = trainingSet.size();
	emission.resize(numBatch);
	std::vector<cv::Mat> updateStorage;
	updateStorage.resize(5);

	updateStorage[0] = cv::Mat::zeros(numNegativeNeuro, numPositiveNeuro, CV_64FC1);
	updateStorage[1] = cv::Mat::zeros(1, numNegativeNeuro, CV_64FC1);
	updateStorage[2] = cv::Mat::zeros(1, numPositiveNeuro, CV_64FC1);
	updateStorage[3] = cv::Mat::zeros(classDimension, numPositiveNeuro, CV_64FC1);
	updateStorage[4] = cv::Mat::zeros(1, classDimension, CV_64FC1);
	
	std::cout << "Pretrain RBM layer." << std::endl;
	for (int epoch = 0; epoch < numEpoch; epoch++){
		double epochError = 0;
		for (int batch = 0; batch < numBatch; batch++){
			cv::Mat data = trainingSet[batch];
			cv::Mat label = labelSet[batch];
			batchSize = data.rows;

			cv::Mat	toPositiveW = weight;
			cv::Mat	toNegativeW = 2 * weight.t();
			cv::Mat	biasP = biasPositive;
			cv::Mat	biasN = 2 * biasNegative;
			
			cv::Mat input = sample(data);
			cv::Mat prob_p = activation_multi(input * toPositiveW + label * weightLabel, biasP);
			cv::Mat activation_p = sample(prob_p);

			if (epoch == numEpoch - 1) {
				emission[batch] = prob_p;
			}

			cv::Mat prob_n = activation(activation_p, toNegativeW, biasN);
			cv::Mat activation_n = sample(prob_n);
			cv::Mat probPrediction = activation_multi(activation_p * weightLabel.t(), biasLabel);
			cv::Mat prediction = sample(probPrediction);

			for (int i = 1; i < numGibbs; i++){
				cv::Mat prob_p = activation_multi(activation_n * toPositiveW + label * weightLabel, biasP);
				activation_p = sample(prob_p);

				cv::Mat probPrediction = activation_multi(activation_p * weightLabel.t(), biasLabel);
				cv::Mat prediction = sample(probPrediction);

				prob_n = activation(activation_p, toNegativeW, biasN);
				activation_n = sample(prob_n);
			}

			cv::Mat prob_rec = activation_multi(activation_n * toPositiveW + label * weightLabel, biasP);

			cv::Mat input_T;
			cv::transpose(input, input_T);
			cv::Mat activation_n_T;
			cv::transpose(activation_n, activation_n_T);

			Error<cv::Mat> error(loss);
			cv::Mat temp = error.value(input, activation_n);
			cv::Mat total(1, 1, CV_64FC1);
			cv::reduce(temp, total, 0, CV_REDUCE_SUM, -1);
			double stepError = total.at<double>(0,0);
			epochError += stepError;

			cv::Mat nStart;
			cv::Mat nEnd;
			cv::Mat pStart;
			cv::Mat pEnd;
			cv::Mat lStart;
			cv::Mat lEnd;
			cv::reduce(input, nStart, 0, CV_REDUCE_AVG, -1);
			cv::reduce(activation_n, nEnd, 0, CV_REDUCE_AVG, -1);
			cv::reduce(prob_p, pStart, 0, CV_REDUCE_AVG, -1);
			cv::reduce(prob_rec, pEnd, 0, CV_REDUCE_AVG, -1);
			cv::reduce(label, lStart, 0, CV_REDUCE_AVG, -1);
			cv::reduce(prediction, lEnd, 0, CV_REDUCE_AVG, -1);

			double momentum =  epoch > 5 ? 0.9: 0.5;
			cv::Mat weightGradient = (activation_n_T * prob_rec - input_T * prob_p) / batchSize;
			cv::Mat weightLabelGradient = (prediction.t() * prob_rec - label.t() * prob_p) / batchSize;
			cv::Mat biasPositiveGradient = pEnd - pStart;
			cv::Mat biasNegativeGradient = nEnd - nStart;
			cv::Mat biasLabelGradient = lEnd - lStart;

			weightStep = annealRate(weightStep, epoch, annealT, 0.5);
			biasNegativeStep = annealRate(biasNegativeStep, epoch, annealT, 0.5);
			biasPositiveStep = annealRate(biasPositiveStep, epoch, annealT, 0.5);
			weightLabelStep = annealRate(weightLabelStep, epoch, annealT, 0.5);
			biasLabelStep = annealRate(biasLabelStep, epoch, annealT, 0.5);

			cv::Mat weightUpdate;
			cv::Mat biasNegativeUpdate;
			cv::Mat biasPositiveUpdate;
			cv::Mat biasLabelUpdate;
			cv::Mat weightLabelUpdate;

			if (gd == AdaGrad){
			weightUpdate = updateAdaGrad(weightGradient, updateStorage[0], weightStep);
			biasNegativeUpdate = updateAdaGrad(biasNegativeGradient, updateStorage[1], biasNegativeStep);
			biasPositiveUpdate = updateAdaGrad(biasPositiveGradient, updateStorage[2], biasPositiveStep);
			weightLabelUpdate = updateAdaGrad(weightLabelGradient, updateStorage[3], weightLabelStep);
			biasLabelUpdate = updateAdaGrad(biasLabelGradient, updateStorage[4], biasLabelStep);
			}
			else {
			WeightDecayPenalty p(L2, 0.1, weight);
			weightUpdate = updateSGD(updateStorage[0], weightGradient, momentum, weightStep, p);
			biasNegativeUpdate = updateSGD(updateStorage[1], biasNegativeGradient, momentum, biasNegativeStep);
			biasPositiveUpdate = updateSGD(updateStorage[2], biasPositiveGradient, momentum, biasPositiveStep);
			weightLabelUpdate = updateSGD(updateStorage[3], weightLabelGradient, momentum, weightLabelStep);
			biasLabelUpdate = updateSGD(updateStorage[4], biasLabelGradient, momentum, biasLabelStep);	
			updateStorage[0] = weightUpdate;
			updateStorage[1] = biasNegativeUpdate;
			updateStorage[2] = biasPositiveUpdate;
			updateStorage[3] = weightLabelUpdate;
			updateStorage[4] = biasLabelUpdate;
			}

			weight += weightUpdate;
			biasNegative += biasNegativeUpdate;
			biasPositive += biasPositiveUpdate;
			weightLabel += weightLabelUpdate;
			biasLabel += biasLabelUpdate;
		}
		error.push_back(epochError);
		std::cout << "Trained RBM layer " << layerIndex << ", epoch " << epoch << ", epoch error: " << epochError << std::endl;
	}
}

//train an RBM for DBM with lables available.
cv::Mat sampleLabel(cv::Mat prob);

void RBM::singleClassifier(dataInBatch &modelOut, dataInBatch &labelSet, int numEpoch, GDType gd)
{	
	error.clear();
	classDimension = labelSet[0].cols;
	cv::Mat weightUpdate = cv::Mat::zeros(weight.rows, weight.cols, CV_64FC1);
	cv::Mat biasUpdate = cv::Mat::zeros(1, weight.cols, CV_64FC1);
	double weightStep = 0.1;
	double biasStep = 0.1;
	double weightCost  = 0.002;

	size_t numBatch = modelOut.size();
	emission.resize(numBatch);
	reconstruction.resize(numBatch);
	std::vector<cv::Mat> updateStorage;
	updateStorage.resize(2);
	updateStorage[0] = cv::Mat::zeros(weight.rows, weight.cols, CV_64FC1);
	updateStorage[1] = cv::Mat::zeros(1, weight.cols, CV_64FC1);
	std::cout << "Train Classifier:" << std::endl;
	
	for (int epoch = 0; epoch < numEpoch; epoch++){
		double epochError = 0;
		for (int batch = 0; batch < numBatch; batch++){
			data batchIn = modelOut[batch];
			data batchLabel = labelSet[batch];
			batchSize = batchIn.rows;

			Error<cv::Mat> errorC(CrossEntropy);
			cv::Mat labelOut = softmaxACT(batchIn, weight, biasPositive);

			batchLabel.convertTo(batchLabel, CV_64FC1);
			cv::Mat weightGradient = batchIn.t() * (labelOut - batchLabel) / batchSize;

			Error<cv::Mat> error(CrossEntropy);
			cv::Mat temp = error.value(labelOut, batchLabel);
			cv::Mat total(1, 1, CV_64FC1);
			cv::reduce(temp, total, 0, CV_REDUCE_SUM, -1);
			double stepError = total.at<double>(0,0);
			epochError += stepError;

			cv::Mat biasGradient;
			cv::reduce(labelOut - batchLabel, biasGradient, 0, CV_REDUCE_AVG, -1);
			double momentum =  epoch > 5 ? 0.9: 0.5;

			weightStep = annealRate(weightStep, epoch, annealT, 0.5);
			biasStep = annealRate(biasStep, epoch, annealT, 0.5);
			if (gd == AdaGrad){
			weightUpdate = updateAdaGrad(weightGradient, updateStorage[0], weightStep);
			biasUpdate = updateAdaGrad(biasGradient, updateStorage[1], biasStep);
			}
			else {
			WeightDecayPenalty p(L2, 0.1, weight);
			weightUpdate = updateSGD(updateStorage[0], weightGradient, momentum, weightStep, p);
			biasUpdate = updateSGD(updateStorage[1], biasGradient, momentum, biasStep);		
			updateStorage[0] = weightUpdate;
			updateStorage[1] = biasUpdate;
			}
			weight += weightUpdate;
			biasPositive += biasUpdate;
		}
		error.push_back(epochError);
		std::cout << "Train Classifier" << layerIndex << ", epoch " << epoch << ", epoch error: " << epochError << std::endl;
	}
	std::cout << "Training of classifier is finished." <<std::endl;
}

classificationError classifyRBM(RBM preModel, RBM classiferModel, dataInBatch &testingSet, dataInBatch &testinglabel, ActivationType act){
	size_t batchNum = testingSet.size();
	if (!batchNum || !testinglabel.size()){
		throw errorReport("classify cannot load testing data or label set. Check input.");
	}

	size_t batchSize = testingSet[0].rows;
	if (!batchSize || !testinglabel[0].rows){
		throw errorReport("classify cannot load testing data or label set. Check input.");
	}
	
	actFunction activation;
	switch (act)
	{
		case sigmoid_t: activation = sigmoidACT; break;
		case tanh_t: activation = tanhACT; break;
		case relu_t: activation = reluACT; break;
		case leaky_relu_t: activation = leaky_reluACT; break;
		case softmax_t: activation = softmaxACT; break;
		default: activation = identityACT;
	}

	std::cout << "Classification begins." <<std::endl;
	int labelDim = testinglabel[0].cols;

	size_t totalError = 0;
	for (size_t i = 0; i < batchNum; i++){
		size_t batchError = 0;
		data batchData = testingSet[i];
		data batchLabel = testinglabel[i];
		cv::Mat preWeight = preModel.g_weight();
		cv::Mat preBias = preModel.g_biasPositive();
		data layerActivation = activation(batchData, preWeight, preBias);
		cv::Mat classWeight = classiferModel.g_weight();
		cv::Mat classBias = classiferModel.g_biasPositive();
		data predictionDIST = softmaxACT(layerActivation, classWeight, classBias);

		indexLabel il = maxIdxInRow(predictionDIST);
		data labelMatrix = oneOfK(il, labelDim);
		batchError = sum(abs(labelMatrix - batchLabel))[0] / 2;
		totalError += batchError;
	}
	cv::Mat m;
	double errorRate = (double)totalError / (batchSize * batchNum);
	std::cout << "Error rate is "<< errorRate << std::endl;
	return classificationError(m, errorRate);
}

cv::Mat sample(cv::Mat distribution){
	cv::Mat state;
	state.create(distribution.rows, distribution.cols, CV_64FC1);
	cv::Mat uniform;
	uniform.create(distribution.rows, distribution.cols, CV_64FC1);
    cv::randu(uniform, cv::Scalar(0), cv::Scalar(1));
    double *rowPtr_st;
    double *rowPtr_dt;
    double *rowPtr_uni;
    for (int i = 0; i < distribution.rows; i++){
    	rowPtr_st = state.ptr<double>(i);
    	rowPtr_dt = distribution.ptr<double>(i);
    	rowPtr_uni = uniform.ptr<double>(i);
    	for (int j = 0; j < distribution.cols; j++){
    		rowPtr_st[j] = rowPtr_dt[j] > rowPtr_uni[j];
   		}
   	}
	return state;
}

cv::Mat gaussian(cv::Mat &distribution){
	cv::Mat state;
	state.create(distribution.rows, distribution.cols, CV_64FC1);
	cv::Mat normal;
	normal.create(distribution.rows, distribution.cols, CV_64FC1);
	cv::Mat meanN = cv::Mat::zeros(1,1,CV_64FC1);
    cv::Mat sigmaN = cv::Mat::ones(1,1,CV_64FC1);
    cv::randn(normal, meanN, sigmaN);
    double *rowPtr_st;
    double *rowPtr_dt;
    double *rowPtr_norm;
    for (int i = 0; i < distribution.rows; i++){
    	rowPtr_st = state.ptr<double>(i);
    	rowPtr_dt = distribution.ptr<double>(i);
    	rowPtr_norm = normal.ptr<double>(i);
    	for (int j = 0; j < distribution.cols; j++){
    		rowPtr_st[j] = rowPtr_dt[j] + rowPtr_norm[j];
   		}
   	}
	return state;
}

cv::Mat dropoutMask(size_t rowSize, size_t colSize, double rate){
	cv::Mat mask;
	mask.create(rowSize, colSize, CV_64FC1);
    double *rowPtr_mask;
    srand(time(NULL));
    for (int i = 0; i < rowSize; i++){
    	rowPtr_mask = mask.ptr<double>(i);
    	for (int j = 0; j < colSize; j++){
    		rowPtr_mask[j] = ((double) rand() / RAND_MAX) > rate;
   		}
   	}
	return mask;
}

cv::Mat sampleLabel(cv::Mat prob){
	cv::Mat predictions = cv::Mat(prob.rows, prob.cols, CV_64FC1, 0.0);
	std::vector<double> randArray;
	randArray.resize(prob.rows);
	srand (time(NULL));
	for (size_t i = 0; i < prob.rows; i++){
		randArray[i] = (double) rand() / RAND_MAX;
	}
    double *rowPtr_pred;
   	double *rowPtr_p;
   	for (int i = 0; i < prob.rows; i++){
   		double rowSum = 0;
   		rowPtr_pred = predictions.ptr<double>(i);
   		rowPtr_p = prob.ptr<double>(i);
   		for (int j = 0; j < prob.cols; j++){
   			rowSum += rowPtr_p[j];
   			if (rowSum >= randArray[i]){
   				rowPtr_pred[j] = 1.0;
   				break;
   			}
   		}
	}
	return predictions;
}

classificationError classify(RBM classiferModel, dataInBatch &testingSet, dataInBatch &testinglabel){
	size_t batchNum = testingSet.size();
	if (!batchNum || !testinglabel.size()){
		throw errorReport("classify cannot load testing data or label set. Check input.");
	}

	size_t batchSize = testingSet[0].rows;
	if (!batchSize || !testinglabel[0].rows){
		throw errorReport("classify cannot load testing data or label set. Check input.");
	}
	actFunction activation;

	std::cout << "Classification begins." <<std::endl;
	int labelDim = testinglabel[0].cols;
	size_t totalError = 0;
	for (size_t i = 0; i < batchNum; i++){
		size_t batchError = 0;
		data batchData = testingSet[i];
		data batchLabel = testinglabel[i];
		cv::Mat classWeight = classiferModel.g_weight();
		cv::Mat classBias = classiferModel.g_biasPositive();
		data predictionDIST = softmaxACT(batchData, classWeight, classBias);
		indexLabel il = maxIdxInRow(predictionDIST);
		data labelMatrix = oneOfK(il, labelDim);
		batchError = sum(abs(labelMatrix - batchLabel))[0] / 2;
		totalError += batchError;
	}
	cv::Mat m;
	double errorRate = (double)totalError / (batchSize * batchNum);
	std::cout << "Error rate is "<< errorRate << std::endl;
	return classificationError(m, errorRate);
}

#endif
