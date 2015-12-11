#ifndef LOADDATA
#define LOADDATA

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <vector>
#include "readMNIST.hpp"
#include "processData.hpp"
#include "visualization.hpp"

void testLoading(){
	data images = imread_mnist_image("t10k-images-idx3-ubyte");
	data labels = imread_mnist_label("t10k-labels-idx1-ubyte");
	dataCollection temp (2);
	temp[0] = images;
	temp[1] = labels;
	std::vector<dataInBatch> shuffled = dataProcess(temp, 100);
	dataInBatch trainingData = shuffled[0];
	dataInBatch trainingLabel = shuffled[1];

	dimension boardDIM(2, 4, 1);
	dimension imgDIM(28, 28, 1);
	imageBoard board(boardDIM, imgDIM);

	std::vector<imageIndex> selectionIndex (8);
	selectionIndex[0] = imageIndex(1,3);
	selectionIndex[1] = imageIndex(2,4);
	selectionIndex[2] = imageIndex(3,7);
	selectionIndex[3] = imageIndex(1,17);
	selectionIndex[4] = imageIndex(2,14);
	selectionIndex[5] = imageIndex(3,11);
	selectionIndex[6] = imageIndex(1,9);
	selectionIndex[7] = imageIndex(2,8);

	std::vector<cv::Mat> datasamples = sampleSelection(trainingData, selectionIndex);
	std::vector<cv::Mat> labelsamples = sampleSelection(trainingLabel, selectionIndex);
	dataToPixel(datasamples);
	board.s_imageSet(datasamples);
	for (cv::Mat k : labelsamples){
		std::cout << k << std::endl;
	}

	showBoard(board, 0, "testingLoad: image loading checking");
}

std::vector<dataInBatch> loadData(const char* imagePath, const char* labelPath, size_t numBatch){
	data images = imread_mnist_image(imagePath);
	data labels = imread_mnist_label(labelPath);
	dataCollection temp (2);
	temp[0] = images;
	temp[1] = labels;
	std::vector<dataInBatch> shuffled = dataProcess(temp, numBatch);
	return shuffled;
}

#endif
