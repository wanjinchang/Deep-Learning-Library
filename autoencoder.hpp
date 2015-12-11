#ifndef AUTOENCODER
#define AUTOENCODER

#include "util.hpp"
#include "rbm.hpp"
#include "backpropagation.hpp"
#include "layer.hpp"

cv::Mat denoiseMask(size_t rowSize, size_t colSize, double rate);

class AutoEncoder{
	size_t numlayer;
	size_t numData;
	size_t numEpoch;
	double denoiseRate = 0;
	std::vector<RBMlayer> layers;
	ActivationType act;
	LossType loss;
	f layerACT;
	flowOverLayer layerIn;
	flowOverLayer layerOut;
	std::vector<double> error;
	dataInBatch reconstruction;

  public:
  	explicit AutoEncoder(size_t ne): numEpoch(ne) { numlayer = 0;}
  	explicit AutoEncoder(std::vector<size_t> rbmSize, ActivationType a, LossType l, size_t ne): act(a), loss(l), numEpoch(ne)
  	{
  		setLayer(rbmSize);
  	}
  	void s_numEpoch(size_t i) {numEpoch = i;}
  	dataInBatch g_reconstruction() {return reconstruction;}
  	void addLayer(RBMlayer &l) {layers.push_back(l); numlayer++;}
  	void setLayer(std::vector<size_t> rbmSize);
  	void train(dataInBatch &trainingData, size_t rbmEpoch, LossType l = MSE, ActivationType a = sigmoid_t);
  	void reconstruct(dataInBatch &testingData);
  	void denoise(double dr) {denoiseRate = dr;}
  	void fineTuning(dataInBatch &originSet, dataInBatch &inputSet);
};

void AutoEncoder::setLayer(std::vector<size_t> rbmSize){
	for (size_t i = 0; i < rbmSize.size(); i++){
		RBMlayer temp(rbmSize[i]);
		temp.s_index(i);
		addLayer(temp);
	}
}

void AutoEncoder::train(dataInBatch &trainingData, size_t rbmEpoch, LossType l, ActivationType a){
	loss = l;
	act = a;
	size_t numBatch = trainingData.size();
	size_t numData = trainingData[0].rows;
	size_t dataDimension = trainingData[0].cols;
	size_t inputDimension = dataDimension;
	//dataInBatch input = trainingData;
	dataInBatch corruptedData;
	corruptedData.resize(numBatch);
	
	if (denoiseRate > 0){
		for (int i = 0; i < numBatch; i++) {
			cv::Mat mask = denoiseMask(numData, dataDimension, denoiseRate);
			corruptedData[i] = trainingData[i].mul(mask);
		}
	}
	else {
		corruptedData = trainingData;
	}

	dataInBatch input = corruptedData;
	std::cout << "Pretrain " << numlayer << " layers." << std::endl;
	size_t numPretrain = numlayer;
	
	//Pretrain RBMs
	for (int i = 0; i < numPretrain; i++){
		std::cout << "Pretrain layer " << i << ": binary" << std::endl; 
		RBM rbm(inputDimension, layers[i].g_size(), i);
		rbm.singleTrainBinary(input, rbmEpoch, act, loss);
		layers[i].s_weight(rbm.g_weight());
		layers[i].s_bias(rbm.g_biasPositive());
		layers[i].s_biasInFlow(rbm.g_biasNegative());
		layers[i].s_ACT(act);
		input = rbm.g_emission();
		inputDimension = layers[i].g_size();
	}
	
	//Unroll RBMs
	for (int i = numPretrain; i < 2 * numPretrain; i++){
		cv::Mat unrollWeight = layers[2 * numPretrain - i - 1].g_weight().t();
		RBMlayer rbm(unrollWeight);
		rbm.s_bias(layers[2 * numPretrain - i - 1].g_biasInFlow());
		rbm.s_ACT(act);
		addLayer(rbm);
	}
	layerIn.resize(numlayer);
	layerOut.resize(numlayer);

	//Reconstruction
	std::vector<double> trainingError;
	trainingError.resize(numBatch);
	reconstruction.resize(numBatch);

	switch (act)
	{
		case sigmoid_t: layerACT = sigmoidF; break;
		case tanh_t: layerACT = tanhF; break;
		case relu_t: layerACT = reluF; break;
		case leaky_relu_t: layerACT = leaky_reluF; break;
		case softmax_t: layerACT = softmaxF; break;
		default: layerACT = identityF;
	}

	double batcherrorSum_ = 0;
	for (int i = 0; i < numBatch; i++){
		cv::Mat target = trainingData[i];
		cv::Mat inFlow = corruptedData[i];
		for (int k = 0; k < numlayer; k++){
			cv::Mat w = layers[k].g_weight();
			cv::Mat b = layers[k].g_bias();
			cv::Mat temp = identityACT(inFlow, w, b);
			inFlow = layerACT(temp);
		}
		reconstruction[i] = inFlow;
		Error<cv::Mat> outputError(loss);
		cv::Mat difference = outputError.value(inFlow, target);
		double batchError = cv::sum(difference)[0];
		batcherrorSum_ += batchError;
	}
	std::cout << "Average error without fine tuning " << batcherrorSum_ / numBatch << std::endl;
}

void AutoEncoder::fineTuning(dataInBatch &originSet, dataInBatch &inputSet){
	size_t numBatch = originSet.size();
	size_t numData = originSet[0].rows;
	size_t dataDimension = originSet[0].cols;
	size_t inputDimension = dataDimension;

	for (int i = 0; i < numBatch; i++){
		for (int j = 0; j < 10; j++){
		cv::Mat batchData = inputSet[i];
		cv::Mat target = originSet[i];
		cv::Mat inFlow = batchData;
		for (int k = 0; k < numlayer; k++){
			cv::Mat w = layers[k].g_weight();
			cv::Mat b = layers[k].g_bias();
			layerIn[k] = identityACT(inFlow, w, b);
			layerOut[k] = layerACT(layerIn[k]);
			inFlow = layerOut[k];
		}
		
		Error<cv::Mat> bpError(loss);
		backpropagation(layers, layerOut, inFlow, target, batchData);
	}
	}

	double batcherrorSum = 0;
	reconstruction.resize(numBatch);
	for (int i = 0; i < numBatch; i++){
		cv::Mat inFlow = inputSet[i];
		cv::Mat target = originSet[i];
		for (int k = 0; k < numlayer; k++){
			cv::Mat w = layers[k].g_weight();
			cv::Mat b = layers[k].g_bias();
			cv::Mat temp = identityACT(inFlow, w, b);
			inFlow = layerACT(temp);
		}
		reconstruction[i] = inFlow;
		Error<cv::Mat> outputError(loss);
		cv::Mat difference = outputError.value(inFlow, target);
		double batchError = cv::sum(difference)[0];
		batcherrorSum += batchError;
	}
	std::cout << "Average error after fine tuning " << batcherrorSum / numBatch << std::endl;
}

void AutoEncoder::reconstruct(dataInBatch &testingData){
	size_t numBatch = testingData.size();
	size_t numData = testingData[0].rows;
	size_t dataDimension = testingData[0].cols;
	size_t inputDimension = dataDimension;
	dataInBatch input = testingData;

	//Reconstruction
	std::vector<double> testingError;
	testingError.resize(numBatch);
	reconstruction.resize(numBatch);

	switch (act)
	{
		case sigmoid_t: layerACT = sigmoidF; break;
		case tanh_t: layerACT = tanhF; break;
		case relu_t: layerACT = reluF; break;
		case leaky_relu_t: layerACT = leaky_reluF; break;
		case softmax_t: layerACT = softmaxF; break;
		default: layerACT = identityF;
	}

	double batcherrorSum_ = 0;
	for (int i = 0; i < numBatch; i++){
		cv::Mat batchData = testingData[i];
		cv::Mat inFlow = batchData;
		for (int k = 0; k < numlayer; k++){
			cv::Mat w = layers[k].g_weight();
			cv::Mat b = layers[k].g_bias();
			cv::Mat temp = identityACT(inFlow, w, b);
			inFlow = layerACT(temp);
		}
		reconstruction[i] = inFlow;
		Error<cv::Mat> outputError(loss);
		cv::Mat difference = outputError.value(inFlow, batchData);
		double batchError = cv::sum(difference)[0];
		batcherrorSum_ += batchError;
	}
	std::cout << "Average error is " << batcherrorSum_ / numBatch << std::endl;
}


#endif
