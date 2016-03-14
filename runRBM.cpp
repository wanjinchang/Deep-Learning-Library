#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <vector>
#include "loadData.hpp"
#include "visualization.hpp"
#include "rbm.hpp"
#include "testing.hpp"

int main()
{	
	std::cout << "Running of RBM begins" << std::endl;
	std::vector<dataInBatch> shuffled = loadData("data/train-images-idx3-ubyte", "data/train-labels-idx1-ubyte", 100);
	dataInBatch trainingData = shuffled[0];
	dataInBatch trainingLabel = shuffled[1];
	std::vector<dataInBatch> shuffledT = loadData("data/t10k-images-idx3-ubyte", "data/t10k-labels-idx1-ubyte", 100);
	dataInBatch testingData = shuffledT[0];
	dataInBatch testingLabel = shuffledT[1];	
	//testLoading();
	labelPixelToData(trainingLabel);
	labelPixelToData(testingLabel);

	RBM rbm(784, 500, 0);
	rbm.dropout(0.2);
	rbm.singleTrainBinary(trainingData, 8);
	dataInBatch modelOut = rbm.g_activation(trainingData);

	RBM classifier(500, 10, 0);
	classifier.singleClassifier(modelOut, trainingLabel, 10);
	//layerWeightTesting(rbm, trainingData);
	classificationError e = classifyRBM(rbm, classifier, testingData, testingLabel, sigmoid_t);

	return 0;
}
