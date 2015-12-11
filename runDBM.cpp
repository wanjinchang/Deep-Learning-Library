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
#include "dbm.hpp"

int main()
{	
	std::cout << "Running of Deep Boltzman Machine begins." << std::endl;
	std::vector<dataInBatch> shuffled = loadData("data/train-images-idx3-ubyte", "data/train-labels-idx1-ubyte", 100);
	dataInBatch trainingData = shuffled[0];
	dataInBatch trainingLabel = shuffled[1];
	std::vector<dataInBatch> shuffledT = loadData("data/t10k-images-idx3-ubyte", "data/t10k-labels-idx1-ubyte", 100);
	dataInBatch testingData = shuffledT[0];
	dataInBatch testingLabel = shuffledT[1];	
	//testLoading();
	labelPixelToData(trainingLabel);
	labelPixelToData(testingLabel);
	
	DBM dbm(MSE);
	std::vector<size_t> k;
	k.push_back(500);
	k.push_back(500);
	dbm.setLayer(k);
	dbm.train(testingData, 10);
	dataInBatch dbmOut = dbm.getEmission(testingData);
	RBM classifier(500, 10, 0);
	classifier.singleClassifier(dbmOut, testingLabel, 15);
	classificationError e = classify(classifier, dbmOut, testingLabel);
	
	return 0;
}