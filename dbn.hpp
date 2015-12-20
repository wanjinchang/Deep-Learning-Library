#ifndef DBNC
#define DBNC

#include "util.hpp"
#include "rbm.hpp"
using namespace std;

class DBN {
	size_t numLayer;
	size_t numData;
	size_t numEpoch;
	vector<size_t> arch;
	ActivationType act;
	LossType loss;
	actFunction activation;
	flowOverLayer layerIn;
	flowOverLayer layerOut;
	vector<double> error;
	vector<cv::Mat> generativeWeight;
	vector<cv::Mat> generativeBias;
	vector<cv::Mat> recognitionWeight;
	vector<cv::Mat> recognitionBias;

  public:
  	DBN (){}
  	DBN (vector<size_t> rbmSize): arch(rbmSize) {}
  	void setLayer(vector<size_t> rbmSize){arch = rbmSize;}
  	void train(dataInBatch &trainingData, size_t rbmEpoch, LossType l = MSE, ActivationType a = sigmoid_t);
  	void classifier(dataInBatch &trainingData, dataInBatch &trainingLabel, size_t rbmEpoch, LossType l = MSE, ActivationType a = sigmoid_t);
  	void fineTuning(dataInBatch &inputSet, dataInBatch &label, int epoch);
  	classificationError classify(dataInBatch &testingSet, dataInBatch &testinglabel);
};

void DBN::train(dataInBatch &trainingData, size_t rbmEpoch, LossType l, ActivationType a){
	loss = l;
	act = a;
	size_t numBatch = trainingData.size();
	size_t numData = trainingData[0].rows;
	size_t dataDimension = trainingData[0].cols;
	size_t inputDimension = dataDimension;
	dataInBatch input = trainingData;
	std::cout << "Pretrain " << numLayer << " layers." << std::endl;
	size_t numLayer = arch.size();
	
	//Pretrain RBMs
	generativeWeight.resize(numLayer);
	generativeBias.resize(numLayer);
	recognitionWeight.resize(numLayer);
	recognitionBias.resize(numLayer);
	for (int i = 0; i < numLayer; i++){
		std::cout << "Pretrain layer " << i << ": binary" << std::endl; 
		RBM rbm(inputDimension, arch[i], i);
		rbm.singleTrainBinary(input, rbmEpoch, act, loss);
		recognitionWeight[i] = rbm.g_weight();
		recognitionBias[i] = rbm.g_biasPositive();
		generativeWeight[i] = rbm.g_weight().t();
		generativeBias[i] = rbm.g_biasNegative();
		input = rbm.g_emission();
		inputDimension = arch[i];
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
	size_t numLayer = arch.size();
	std::cout << "Pretrain " << numLayer << " layers." << std::endl;
	
	//Pretrain RBMs
	generativeWeight.resize(numLayer);
	generativeBias.resize(numLayer);
	recognitionWeight.resize(numLayer);
	recognitionBias.resize(numLayer);
	for (int i = 0; i < numLayer - 1; i++){
		std::cout << "Pretrain layer " << i << ": binary" << std::endl; 
		RBM rbm(inputDimension, arch[i], i);
		rbm.singleTrainBinary(input, rbmEpoch, act, loss);
		recognitionWeight[i] = rbm.g_weight();
		recognitionBias[i] = rbm.g_biasPositive();
		generativeWeight[i] = rbm.g_weight().t();
		generativeBias[i] = rbm.g_biasNegative();
		input = rbm.g_emission();
		inputDimension = arch[i];
	}

	dataInBatch jointPattern;
	jointPattern.resize(numBatch);
	for (int i = 0; i < numBatch; i++){
		cv::hconcat(trainingLabel[i], input[i], jointPattern[i]);
	}
	inputDimension += classDimension;

	std::cout << "Pretrain layer" << numLayer << " with classifier: binary" << std::endl; 
	RBM rbm(inputDimension, arch[numLayer - 1], numLayer - 1);
	rbm.dropout(std::max(0.1, (double)classDimension / inputDimension));
	rbm.singleTrainBinary(jointPattern, rbmEpoch, act, loss);
	recognitionWeight[numLayer - 1] = rbm.g_weight();
	recognitionBias[numLayer - 1] = rbm.g_biasPositive();
	generativeWeight[numLayer - 1] = rbm.g_weight().t();
	generativeBias[numLayer - 1] = rbm.g_biasNegative();
	std::cout << "DBN classifier training is finished." << std::endl;
	std::cout << numLayer << " layers are trained (As a classifier). " << std::endl;
}

void DBN::fineTuning(dataInBatch &dataSet, dataInBatch &labelSet, int epoch){
	size_t numLayer = arch.size();
	size_t numBatch = dataSet.size();
	size_t batchSize = dataSet[0].rows;
	int numClass = labelSet[0].cols;
	size_t dataDim = dataSet[0].cols;
	for (int i = 0; i < epoch; i++){
		for (int k = 0; k < numBatch; k++){
			cv::Mat input = dataSet[k];
			cv::Mat label = labelSet[k];

			//Bottem-up Pass
			vector<cv::Mat> recognition;
			recognition.resize(numLayer + 1);
			recognition[0] = input;
			for (size_t j = 0; j < numLayer - 1; j++){
				recognition[j + 1] = sigmoidACT(recognition[j], recognitionWeight[j], recognitionBias[j]);
			}
			cv::Mat jointPattern;
			cv::hconcat(label, recognition[numLayer - 1], jointPattern);
			recognition[numLayer] = sigmoidACT(jointPattern, recognitionWeight[numLayer - 1], recognitionBias[numLayer - 1]);

			cv::Rect dataPart = cv::Rect(numClass, 0, recognition[numLayer - 1].cols, batchSize);
			cv::Rect labelPart = cv::Rect(0, 0, numClass, batchSize);
			
			//Gibbs Sampling on the top layer
			cv::Mat GBbottom = sigmoidACT(recognition[numLayer], generativeWeight[numLayer - 1], generativeBias[numLayer - 1]);
			cv::Mat GBtop = sigmoidACT(GBbottom, recognitionWeight[numLayer - 1], recognitionBias[numLayer - 1]);

			//Top-down Pass
			vector<cv::Mat> generation;
			generation.resize(numLayer + 1);
			generation[numLayer - 1] = GBbottom(dataPart);
			for (size_t j = numLayer - 1; j > 0; j--){
				generation[j - 1] = sigmoidACT(generation[j], generativeWeight[j - 1], generativeBias[j - 1]);
			}

			//Transmission in each layer
			vector<cv::Mat> transmissionDown;
			vector<cv::Mat>	transmissionUp;
			transmissionUp.resize(numLayer + 1);
			transmissionDown.resize(numLayer + 1);
			for (size_t j = 0; j < numLayer - 1; j++){
				transmissionDown[j] = sigmoidACT(recognition[j + 1], generativeWeight[j], generativeBias[j]);
			}
			for (size_t j = 0; j < numLayer - 1; j++){
				transmissionUp[j + 1] = sigmoidACT(generation[j], recognitionWeight[j], recognitionBias[j]);
			}
			double r = 0.01;
		
			//Update of generative parameters
			for (int j = 0; j < numLayer - 1; j++){
				generativeWeight[j] += r * (recognition[j + 1].t()) * (recognition[j] - transmissionDown[j]) / batchSize;
				cv::Mat generativeBiasUpdate = recognition[j] - transmissionDown[j];
				cv::reduce(generativeBiasUpdate, generativeBiasUpdate, 0, CV_REDUCE_AVG, -1);
				generativeBias[j] += r * generativeBiasUpdate;
			}
			
			//Updata of recognition parameters
			for (int j = 0; j < numLayer - 1; j++){
				recognitionWeight[j] += r * generation[j].t() * (generation[j + 1] - transmissionUp[j + 1]) / batchSize;
				cv::Mat recognitionBiasUpdate = generation[j + 1] - transmissionUp[j + 1];
				cv::reduce(recognitionBiasUpdate, recognitionBiasUpdate, 0, CV_REDUCE_AVG, -1);
				recognitionBias[j] += r * recognitionBiasUpdate;
			}
			
			cv::Mat nStart;
			cv::Mat nEnd;
			cv::Mat pStart;
			cv::Mat pEnd;
			cv::reduce(jointPattern, nStart, 0, CV_REDUCE_AVG, -1);
			cv::reduce(GBbottom, nEnd, 0, CV_REDUCE_AVG, -1);
			cv::reduce(recognition[numLayer], pStart, 0, CV_REDUCE_AVG, -1);
			cv::reduce(GBtop, pEnd, 0, CV_REDUCE_AVG, -1);
			//Update of top parameters
			recognitionWeight[numLayer - 1] += r * ((jointPattern.t()) * recognition[numLayer] - (GBbottom.t()) * GBtop) / batchSize;		
			recognitionBias[numLayer - 1] += r * (pStart - pEnd);	
			generativeBias[numLayer - 1] += r * (nStart - nEnd);
			generativeWeight[numLayer - 1] = recognitionWeight[numLayer - 1].t();
		}
	}
}

classificationError DBN::classify(dataInBatch &testingSet, dataInBatch &testinglabel){
	size_t classDimension = testinglabel[0].cols;
	size_t numBatch = testingSet.size();
	size_t totalError = 0;
	int batchSize = testingSet[0].rows;
	int numLayer = arch.size();
	for (int i = 0; i < numBatch; i++){
		double batchError = 0;
		data batchData = testingSet[i];
		data batchLabel = testinglabel[i];
		data inFlow = batchData;
		for (int k = 0; k < numLayer; k++){
			if (k < numLayer - 1){
				inFlow = sigmoidACT(inFlow, recognitionWeight[k], recognitionBias[k]);
			}
			else {
				cv::Mat temp = cv::Mat::zeros(batchSize, classDimension, CV_64FC1);
				cv::hconcat(temp, inFlow, inFlow);
				inFlow = sigmoidACT(inFlow, recognitionWeight[k], recognitionBias[k]);
			}
		}
		
		inFlow = sigmoidACT(inFlow, generativeWeight[numLayer - 1], generativeBias[numLayer - 1]);
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
