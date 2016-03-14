#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <vector>

#include "util.hpp"
#include "loadData.hpp"
#include "visualization.hpp"
#include "rbm.hpp"
#include "testing.hpp"
#include "autoencoder.hpp"
#include "dnn.hpp"

int main()
{	
	std::cout << "Running of Deep Neural Networks begins." << std::endl;
	std::vector<dataInBatch> shuffled = loadData("data/train-images-idx3-ubyte", "data/train-labels-idx1-ubyte", 100);
	dataInBatch trainingData = shuffled[0];
	dataInBatch trainingLabel = shuffled[1];
	std::vector<dataInBatch> shuffledT = loadData("data/t10k-images-idx3-ubyte", "data/t10k-labels-idx1-ubyte", 100);
	dataInBatch testingData = shuffledT[0];
	dataInBatch testingLabel = shuffledT[1];
	//testLoading();
	labelPixelToData(trainingLabel);
	labelPixelToData(testingLabel);
	
	DNN dnn(MSE);
	std::vector<size_t> k;
	k.push_back(500);
	k.push_back(300);
	k.push_back(200);
	dnn.setLayer(k);
	//dnn.train(testingData, 5);
	dnn.classifier(trainingData, trainingLabel, 6, 1);
	dnn.fineTuning(trainingLabel, trainingData);
	dnn.classify(testingData, testingLabel);
	
	return 0;
}