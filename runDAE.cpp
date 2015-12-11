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

int main()
{	
	std::cout << "Running of Denoising AutoEncoder begins." << std::endl;
	std::vector<dataInBatch> shuffled = loadData("data/train-images-idx3-ubyte", "data/train-labels-idx1-ubyte", 100);
	dataInBatch trainingData = shuffled[0];
	dataInBatch trainingLabel = shuffled[1];
	std::vector<dataInBatch> shuffledT = loadData("data/t10k-images-idx3-ubyte", "data/t10k-labels-idx1-ubyte", 100);
	dataInBatch testingData = shuffledT[0];
	dataInBatch testingLabel = shuffledT[1];
	
	AutoEncoder ae(6);
	std::vector<size_t> k;
	k.push_back(500);
	k.push_back(300);
	ae.setLayer(k);
	dataInBatch corruptedTrainingData = corruptImage(trainingData, 0.2);
	dataInBatch corruptedTestingData = corruptImage(testingData, 0.2);
	
	ae.denoise(0.2);
	ae.train(trainingData, 6);
	ae.fineTuning(trainingData, corruptedTrainingData);
	ae.reconstruct(testingData);
	dataInBatch reTest = ae.g_reconstruction();
	
	dimension boardDIM(6, 6, 1);
	dimension imgDIM(28, 28, 1);
	imageBoard board(boardDIM, imgDIM);

	std::vector<imageIndex> selectionIndex = generateSelections();
	
	std::vector<cv::Mat> samples = sampleSelection(corruptedTestingData, selectionIndex);
	dataToPixel(samples);
	board.s_referenceSet(samples);
	
	std::vector<cv::Mat> samplesN = sampleSelection(reTest, selectionIndex);
	dataToPixel(samplesN);
	board.s_imageSet(samplesN);

	showBoard(board, 0, "Reconstruction visualization");
	
	return 0;
}
