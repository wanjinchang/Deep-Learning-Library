#ifndef DBNC
#define DBNC

#include "util.hpp"
#include "rbm.hpp"
#include "layer.hpp"
#include "upDown.hpp"
using namespace std;
class DBN {
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
  	DBN (LossType l): loss(l) {}
  	DBN (std::vector<size_t> rbmSize, ActivationType a, LossType l, size_t ne): act(a), loss(l), numEpoch(ne)
  	{
  		setLayer(rbmSize);
  	}
  	void s_numEpoch(size_t i) {numEpoch = i;}
  	void addLayer(RBMlayer &l){layers.push_back(l); numlayer++;}
  	void setLayer(std::vector<size_t> rbmSize);
  	void train(dataInBatch &trainingData, size_t rbmEpoch, LossType l = MSE, ActivationType a = sigmoid_t);
  	void classifier(dataInBatch &trainingData, dataInBatch &trainingLabel, size_t rbmEpoch, LossType l = MSE, ActivationType a = sigmoid_t);
  	void fineTuning(dataInBatch &inputSet, dataInBatch &label, int epoch);
  	classificationError classify(dataInBatch &testingSet, dataInBatch &testinglabel);
};

void DBN::setLayer(std::vector<size_t> rbmSize){
	numlayer = 0;
	layers.clear();
	for (size_t i = 0; i < rbmSize.size(); i++){
		RBMlayer temp(rbmSize[i]);
		temp.s_index(i);
		addLayer(temp);
	}
}

void DBN::train(dataInBatch &trainingData, size_t rbmEpoch, LossType l, ActivationType a){
	loss = l;
	act = a;
	size_t numBatch = trainingData.size();
	size_t numData = trainingData[0].rows;
	size_t dataDimension = trainingData[0].cols;
	size_t inputDimension = dataDimension;
	dataInBatch input = trainingData;
	std::cout << "Pretrain " << numlayer << " layers." << std::endl;
	size_t numlayer = numlayer;
	
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

void DBN::classifier(dataInBatch &trainingData, dataInBatch &trainingLabel, size_t rbmEpoch, LossType l, ActivationType a){
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
	for (int i = 0; i < numlayer - 1; i++){
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

	dataInBatch jointPattern;
	jointPattern.resize(numBatch);
	for (int i = 0; i < numBatch; i++){
		cv::hconcat(trainingLabel[i], input[i], jointPattern[i]);
	}
	inputDimension += classDimension;

	std::cout << "Pretrain layer" << numlayer << " with classifier: binary" << std::endl; 
	RBM rbm(inputDimension, layers[numlayer - 1].g_size(), numlayer - 1);
	rbm.dropout(std::max(0.1, (double)classDimension / inputDimension));
	rbm.singleTrainBinary(jointPattern, rbmEpoch, act, loss);
	layers[numlayer - 1].s_weight(rbm.g_weight());
	layers[numlayer - 1].s_bias(rbm.g_biasPositive());
	layers[numlayer - 1].s_biasInFlow(rbm.g_biasNegative());
	layers[numlayer - 1].s_ACT(act);

	std::cout << "DBN classifier training is finished." << std::endl;
	std::cout << numlayer << " layers are trained (As a classifier). " << std::endl;
}

void DBN::fineTuning(dataInBatch &inputSet, dataInBatch &label, int epoch){
	upDown(layers, inputSet, label, epoch);
}

classificationError DBN::classify(dataInBatch &testingSet, dataInBatch &testinglabel){
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
		data batchLabel = testinglabel[i];
		data inFlow = batchData;
		for (int k = 0; k < numlayer; k++){
			cv::Mat w = layers[k].g_weight();
			cv::Mat b = layers[k].g_bias();
			actFunction layerACT = layers[k].g_ACT();
			if (k < numlayer - 1){
				inFlow = layerACT(inFlow, w, b);
			}
			else {
				cv::Mat temp = cv::Mat::zeros(batchSize, classDimension, CV_64FC1);
				cv::hconcat(temp, inFlow, inFlow);
				inFlow = layerACT(inFlow, w, b);
			}
		}
		cv::Mat w = layers[numlayer - 1].g_weight().t();
		cv::Mat bv = layers[numlayer - 1].g_biasInFlow();
		actFunction layerACT = layers[numlayer - 1].g_ACT();
		inFlow = layerACT(inFlow, w, bv);
		cv::Rect roi = cv::Rect(0, 0, classDimension, batchSize);
		cv::Mat predictionMat = inFlow(roi);
		indexLabel predictionDIST = maxIdxInRow(predictionMat);
		data labelMatrix = oneOfK(predictionDIST, classDimension);
		batchError = sum(abs(labelMatrix - batchLabel))[0] / 2;
		totalError += batchError;	
	}
	cv::Mat m;
	double errorRate = (double)totalError / (batchSize * numBatch);
	std::cout << "Error rate is "<< errorRate << std::endl;
	return classificationError(m, errorRate);
}

#endif
