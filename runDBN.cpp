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

int main()
{	
	std::cout << "Running of Deep Belief Networks begins." << std::endl;
	std::vector<dataInBatch> shuffled = loadData("data/train-images-idx3-ubyte", "data/train-labels-idx1-ubyte", 100);
	dataInBatch trainingData = shuffled[0];
	dataInBatch trainingLabel = shuffled[1];
	std::vector<dataInBatch> shuffledT = loadData("data/t10k-images-idx3-ubyte", "data/t10k-labels-idx1-ubyte", 100);
	dataInBatch testingData = shuffledT[0];
	dataInBatch testingLabel = shuffledT[1];
	//testLoading();
	labelPixelToData(trainingLabel);
	labelPixelToData(testingLabel);
	
	std::vector<size_t> arch;
	arch.push_back(500);
	arch.push_back(300);
	DBN dbn(arch);
	//dbn.train(testingData, 5);
	dbn.classifier(trainingData, trainingLabel, 7);
	dbn.fineTuning(trainingData, trainingLabel, 8);
	dbn.classify(testingData, testingLabel);
	return 0;
}
