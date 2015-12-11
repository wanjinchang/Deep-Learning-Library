#ifndef DNNC
#define DNNC

#include "util.hpp"
#include "rbm.hpp"
#include "backpropagation.hpp"
#include "layer.hpp"

class DNN {
	size_t numlayer;
	size_t numData;
	size_t numEpoch;
	std::vector<RBMlayer> layers;
	ActivationType act;
	LossType loss;
	actFunction activation;
	flowOverLayer layerIn;
	flowOverLayer layerOut;
	std::vector<double> error;

  public:
  	DNN (LossType l): loss(l) {}
  	DNN (std::vector<size_t> rbmSize, ActivationType a, LossType l, size_t ne): act(a), loss(l), numEpoch(ne)
  	{
  		setLayer(rbmSize);
  	}
  	void s_numEpoch(size_t i) {numEpoch = i;}
  	void addLayer(RBMlayer &l){layers.push_back(l); numlayer++;}
  	void setLayer(std::vector<size_t> rbmSize);
  	void train(dataInBatch &trainingData, size_t rbmEpoch, LossType l = MSE, ActivationType a = sigmoid_t);
  	void classifier(dataInBatch &trainingData, dataInBatch &trainingLabel, size_t rbmEpoch, int preTrainOpt, LossType l = MSE, ActivationType a = sigmoid_t);
  	void fineTuning(dataInBatch &label, dataInBatch &inputSet);
  	dataInBatch getEmission(dataInBatch &data, int isClassifier);
  	classificationError classify(dataInBatch &testingSet, dataInBatch &testinglabel);
};

void DNN::setLayer(std::vector<size_t> rbmSize){
	numlayer = 0;
	layers.clear();
	for (size_t i = 0; i < rbmSize.size(); i++){
		RBMlayer temp(rbmSize[i]);
		temp.s_index(i);
		addLayer(temp);
	}
}

void DNN::train(dataInBatch &trainingData, size_t rbmEpoch, LossType l, ActivationType a){
	std::cout << "Layers" << numlayer << std::endl;
	loss = l;
	act = a;
	size_t numBatch = trainingData.size();
	size_t numData = trainingData[0].rows;
	size_t dataDimension = trainingData[0].cols;
	size_t inputDimension = dataDimension;
	dataInBatch input = trainingData;
	std::cout << "Pretrain " << numlayer << " layers." << std::endl;
	
	//Pretrain RBMs
	for (int i = 0; i < numlayer; i++){
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
}

void DNN::classifier(dataInBatch &trainingData, dataInBatch &trainingLabel, size_t rbmEpoch, int preTrainOpt, LossType l, ActivationType a){
	loss = l;
	act = a;
	size_t numBatch = trainingData.size();
	size_t numData = trainingData[0].rows;
	size_t dataDimension = trainingData[0].cols;
	size_t inputDimension = dataDimension;
	size_t classDimension = trainingLabel[0].cols;
	dataInBatch input = trainingData;
	std::cout << "Pretrain " << numlayer + 1 << " layers." << std::endl;
	
	//Pretrain RBMs
	if (preTrainOpt == 1){
	for (int i = 0; i < numlayer; i++){
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
	std::cout << "Pretrain layer " << numlayer << ": classifier" << std::endl; 
	}
	else{
	for (int i = 0; i < numlayer; i++){
		std::cout << "Random initialize layer " << i << ": binary" << std::endl; 
		RBM rbm(inputDimension, layers[i].g_size(), i);
		layers[i].s_weight(rbm.g_weight());
		layers[i].s_bias(rbm.g_biasPositive());
		layers[i].s_biasInFlow(rbm.g_biasNegative());
		layers[i].s_ACT(act);
		input = rbm.g_emission();
		inputDimension = layers[i].g_size();
	}
	std::cout << "initialize layer " << numlayer << ": classifier" << std::endl; 
	}

	dataInBatch modelOut = trainingData;
	for (int i = 0; i < numlayer; i++){
		modelOut = layers[i].g_activation(modelOut);
	}
	RBM classifier(layers[numlayer - 1].g_size(), classDimension, 0);
	if (preTrainOpt == 1){
		std::cout << "Train classifier" << std::endl;
		classifier.singleClassifier(modelOut, trainingLabel, rbmEpoch);
	}

	RBMlayer classifierLayer(classDimension);
	classifierLayer.s_weight(classifier.g_weight());
	classifierLayer.s_bias(classifier.g_biasPositive());
	classifierLayer.s_ACT(softmax_t);
	layers.push_back(classifierLayer);
	numlayer++;
	std::cout << "DBN classifier training is finished." << std::endl;
	std::cout << numlayer << " layers are trained(Include a classifier). " << std::endl;
}

void DNN::fineTuning(dataInBatch &label, dataInBatch &inputSet){
	size_t numBatch = inputSet.size();
	size_t numData = inputSet[0].rows;
	size_t dataDimension = inputSet[0].cols;
	size_t inputDimension = dataDimension;
	layerOut.resize(numlayer);

	for (int i = 0; i < numBatch; i++){
		for (int j = 0; j < 10; j++){
		cv::Mat batchData = inputSet[i];
		data batchLabel = label[i];
		cv::Mat inFlow = batchData;
		for (int k = 0; k < numlayer; k++){
			cv::Mat w = layers[k].g_weight();
			cv::Mat b = layers[k].g_bias();
			actFunction layerACT = layers[k].g_ACT();
			layerOut[k] = layerACT(inFlow, w, b);
			inFlow = layerOut[k];
		}
		backpropagation(layers, layerOut, inFlow, batchLabel, batchData);
	}
	}

}

classificationError DNN::classify(dataInBatch &testingSet, dataInBatch &testinglabel){
	size_t classDimension = testinglabel[0].cols;
	size_t numBatch = testingSet.size();
	if (!numBatch || !testinglabel.size()){
		throw errorReport("classify cannot load testing data or label set. Check input.");
	}

	size_t batchSize = testingSet[0].rows;
	if (!batchSize || !testinglabel[0].rows){
		throw errorReport("classify cannot load testing data or label set. Check input.");
	}

	size_t totalError = 0;
	for (int i = 0; i < numBatch; i++){
		double batchError = 0;
		data batchData = testingSet[i];
		data inFlow = batchData;
		for (int k = 0; k < numlayer; k++){
			cv::Mat w = layers[k].g_weight();
			cv::Mat b = layers[k].g_bias();
			actFunction layerACT = layers[k].g_ACT();
			inFlow = layerACT(inFlow, w, b);
		}

		data pDIST = inFlow;
		data batchLabel = testinglabel[i];
		indexLabel predictionDIST = maxIdxInRow(pDIST);
		data labelMatrix = oneOfK(predictionDIST, classDimension);
		batchError = sum(abs(labelMatrix - batchLabel))[0] / 2;
		totalError += batchError;
	}
	cv::Mat m;
	double errorRate = (double)totalError / (batchSize * numBatch);
	std::cout << "Error rate is " << errorRate << std::endl;
	return classificationError(m, errorRate);
}

dataInBatch DNN::getEmission(dataInBatch &data, int isClassifier){
	int numBatch = data.size();
	dataInBatch output;
	output.resize(numBatch);
	int topLayer = 0;
	if (isClassifier == 1){
		topLayer = numlayer - 1;
	}
	else{
		topLayer = numlayer;
	}
	for (int i = 0; i < numBatch; i++){
		cv::Mat batchData = data[i];
		cv::Mat inFlow = batchData;
		for (int k = 0; k < topLayer; k++){
			cv::Mat w = layers[k].g_weight();
			cv::Mat b = layers[k].g_bias();
			actFunction layerACT = sigmoidACT;
			inFlow = layerACT(inFlow, w, b);
		}
		output[i] = inFlow;
	}
	return output;
}

#endif
