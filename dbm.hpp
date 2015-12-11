#ifndef DBMC
#define DBMC

#include "rbm.hpp"
#include "backpropagation.hpp"
#include "inference.hpp"

using namespace std;
class DBM{
	size_t numlayer;
	size_t numData;
	size_t numEpoch;
	std::vector<RBMlayer> layers;
	ActivationType act = sigmoid_t;
	LossType loss = MSE;
	f layerACT;
	std::vector<cv::Mat> flow;
	std::vector<cv::Mat> emissions;
	std::vector<double> error;

  public:
  	explicit DBM(LossType l): loss(l) {}
  	explicit DBM(std::vector<size_t> rbmSize, ActivationType a, LossType l, size_t ne): act(a), loss(l), numEpoch(ne)
  	{
  		setLayer(rbmSize);
  	}
  	void s_numEpoch(size_t i) {numEpoch = i;}
  	void addLayer(RBMlayer &l){layers.push_back(l); numlayer++;}
  	void setLayer(std::vector<size_t> rbmSize);
  	void train(dataInBatch &data, size_t rbmEpoch, LossType l = MSE, ActivationType a = sigmoid_t, size_t infEpoch = 3);
  	void classify(dataInBatch &data, dataInBatch &label);
  	dataInBatch getEmission(dataInBatch &testingData);
};

void DBM::setLayer(std::vector<size_t> rbmSize){
	for (size_t i = 0; i < rbmSize.size(); i++){
		RBMlayer temp(rbmSize[i]);
		temp.s_index(i);
		addLayer(temp);
	}
}

void DBM::train(dataInBatch &data, size_t rbmEpoch, LossType l, ActivationType a, size_t infEpoch){
	loss = l;
	act = a;
	size_t numBatch = data.size();
	size_t dataSize = data[0].rows;
	size_t dataDim = data[0].cols;
	dataInBatch input = data;
	int inputDim = dataDim;
	//Pretrain RBMs
	for (int i = 0; i < numlayer; i++){
		RBM rbm(inputDim, layers[i].g_size(), i);
		int depth = 0;
		if (i == 0){
			depth = -1;
		}
		else if (i == numlayer - 1){
			depth = 1;
		}
		rbm.doubleTrain(input, rbmEpoch, depth, act, loss);
		layers[i].s_weight(rbm.g_weight());
		layers[i].s_bias(rbm.g_biasPositive());
		layers[i].s_biasInFlow(rbm.g_biasNegative());
		layers[i].s_ACT(act);
		input = rbm.g_emission();
		inputDim = layers[i].g_size();
	}
	
	//DBM Inference
	DBMinference(layers, data, infEpoch);
}

dataInBatch DBM::getEmission(dataInBatch &data){
	dataInBatch output;
	output.resize(data.size());
	int inputDim = data[0].cols;
	for (int e = 0; e < data.size(); e++){
		cv::Mat input = data[e];
		for (int i = 0; i < numlayer; i++){
			cv::Mat w = 2 * layers[i].g_weight();
			cv::Mat b = 2 * layers[i].g_bias();
			actFunction layerACT = layers[i].g_ACT();
			input = layerACT(input, w, b);
		}
		output[e] = input;
	}
	return output;
}


void DBM::classify(dataInBatch &data, dataInBatch &label){
	int inputDim = data[0].cols;
	int classDimension = label[0].cols;
	double totalError = 0;
	int numData = data.size() * data[0].rows;

	for (int e = 0; e < data.size(); e++){
		cv::Mat input = data[e];
		cv::Mat batchLabel = label[e];
		for (int i = 0; i < numlayer - 1; i++){
			cv::Mat w = 2 * layers[i].g_weight();
			cv::Mat b = 2 * layers[i].g_bias();
			actFunction layerACT = layers[i].g_ACT();
			input = layerACT(input, w, b);
			inputDim = layers[i].g_size();
		}
		cv::Mat w = layers[numlayer - 1].g_weight();
		cv::Mat b = layers[numlayer - 1].g_bias();
		actFunction layerACT = layers[numlayer - 1].g_ACT();
		input = layerACT(input, w, b);
		cv::Mat biasW = layers[numlayer - 1].g_labelWeight().t();
		cv::Mat biasB = layers[numlayer - 1].g_biasLabel();
		cv::Mat prediction = layerACT(input, biasW, biasB);

		indexLabel predictionDIST = maxIdxInRow(prediction);
		cv::Mat labelMatrix = oneOfK(predictionDIST, classDimension);
		double batchError = sum(abs(labelMatrix - batchLabel))[0] / 2;
		totalError += batchError;
	}
	cout << totalError << endl;
	cout << "error rate is " << totalError / numData << endl;

}

#endif
