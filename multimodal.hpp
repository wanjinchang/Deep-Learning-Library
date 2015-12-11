#ifndef MULTIMODAL
#define MULTIMODAL
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
#include "dbn.hpp"
#include "cca.hpp"

int main()
{	
	std::cout << "Multimodal learning with DBN and CCA" << std::endl;
	std::vector<dataInBatch> shuffled = loadData("train-images-idx3-ubyte", "train-labels-idx1-ubyte", 100);
	dataInBatch trainingData = shuffled[0];
	dataInBatch trainingLabel = shuffled[1];
	std::vector<dataInBatch> shuffledT = loadData("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte", 100);
	dataInBatch testingData = shuffledT[0];
	dataInBatch testingLabel = shuffledT[1];
	//testLoading();
	labelPixelToData(trainingLabel);
	labelPixelToData(testingLabel);
	
	DBN dbn(MSE);
	std::vector<size_t> k;
	k.push_back(500);
	k.push_back(300);
	k.push_back(200);
	dbn.setLayer(k);
	//dbn.train(testingData, 5);
	dbn.classifier(trainingData, trainingLabel, 8);
	dbn.fineTuning(trainingLabel, trainingData, CrossEntropy);
	dbn.classify(testingData, testingLabel);
	
	return 0;
}

#endif