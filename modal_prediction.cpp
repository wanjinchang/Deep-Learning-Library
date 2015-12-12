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
#include "cca.hpp"
#include "readDat.hpp"
#include "readAvLetters.hpp"
#include "autoencoder.hpp"
#include "processData.hpp"

int main()
{	
	std::cout << "Running of Deep Canonical Correlation begins." << std::endl;
	dataInBatch TrainingLetters;
	TrainingLetters.resize(7);
	dataInBatch TestingLetters;
	TestingLetters.resize(2);
	cv::Mat temp = readBinary("data/letters_lip/A1_Anya-lips.dat", 4800).t();
	TrainingLetters[0] = temp;
	temp = readBinary("data/letters_lip/B1_Anya-lips.dat", 4800).t();
	TrainingLetters[1] = temp;
	temp = readBinary("data/letters_lip/C1_Anya-lips.dat", 4800).t();
	TrainingLetters[2] = temp;
	temp = readBinary("data/letters_lip/D1_Anya-lips.dat", 4800).t();
	TrainingLetters[3] = temp;
	temp = readBinary("data/letters_lip/E1_Anya-lips.dat", 4800).t();
	TrainingLetters[4] = temp;
	temp = readBinary("data/letters_lip/F1_Anya-lips.dat", 4800).t();
	TrainingLetters[5] = temp;
	temp = readBinary("data/letters_lip/G1_Anya-lips.dat", 4800).t();
	TrainingLetters[6] = temp;
	temp = readBinary("data/letters_lip/H1_Anya-lips.dat", 4800).t();
	TestingLetters[0] = temp;
	temp = readBinary("data/letters_lip/I1_Anya-lips.dat", 4800).t();
	TestingLetters[1] = temp;
	std::cout << "Reading MFCC files" << std::endl;
	dataInBatch TrainingMFCC;
	TrainingMFCC.resize(7);
	dataInBatch TestingMFCC;
	TestingMFCC.resize(2);
	temp = imread_avletters_mfcc("data/avletters/Audio/A1_Anya.mfcc").t();
	TrainingMFCC[0] = temp;
	temp = imread_avletters_mfcc("data/avletters/Audio/B1_Anya.mfcc").t();
	TrainingMFCC[1] = temp;
	temp = imread_avletters_mfcc("data/avletters/Audio/C1_Anya.mfcc").t();
	TrainingMFCC[2] = temp;
	temp = imread_avletters_mfcc("data/avletters/Audio/D1_Anya.mfcc").t();
	TrainingMFCC[3] = temp;
	temp = imread_avletters_mfcc("data/avletters/Audio/E1_Anya.mfcc").t();
	TrainingMFCC[4] = temp;
	temp = imread_avletters_mfcc("data/avletters/Audio/F1_Anya.mfcc").t();
	TrainingMFCC[5] = temp;
	temp = imread_avletters_mfcc("data/avletters/Audio/G1_Anya.mfcc").t();
	TrainingMFCC[6] = temp;
	temp = imread_avletters_mfcc("data/avletters/Audio/H1_Anya.mfcc").t();
	TestingMFCC[0] = temp;
	temp = imread_avletters_mfcc("data/avletters/Audio/I1_Anya.mfcc").t();
	TestingMFCC[1] = temp;

	cv::Mat trainigData = cv::Mat(0, 4800, CV_64FC1);
	for (int i = 0; i < 7; i++){
		TrainingLetters[i] = TrainingLetters[i] / 255;
		trainigData.push_back(TrainingLetters[i]);
	}
	dataInBatch t;
	t.resize(1);
	t[0] = trainigData;

	cv::Mat testingData = cv::Mat(0, 4800, CV_64FC1);
	for (int j = 0; j < 2; j++){
		TestingLetters[j] = TestingLetters[j] / 255;
		testingData.push_back(TestingLetters[j]);
	}
	dataInBatch tt;
	tt.resize(1);
	tt[0] = testingData;

	cv::Mat trainigData2 = cv::Mat(0, 26, CV_64FC1);
	for (int i = 0; i < 7; i++){
		cv::Mat tempmfcc(0, 26, CV_64FC1);
		for (int j = 0; j < TrainingLetters[i].rows; j++){
			tempmfcc.push_back(TrainingMFCC[i].row(2 * j));
		}
		trainigData2.push_back(tempmfcc);
	}
	dataInBatch t2;
	t2.resize(1);
	t2[0] = trainigData2;

	cv::Mat testingData2 = cv::Mat(0, 26, CV_64FC1);
	for (int i = 0; i < 2; i++){
		cv::Mat tempmfcc(0, 26, CV_64FC1);
		for (int j = 0; j < TestingLetters[i].rows; j++){
			tempmfcc.push_back(TestingMFCC[i].row(2 * j));
		}
		testingData2.push_back(tempmfcc);
	}
	dataInBatch tt2;
	tt2.resize(1);
	tt2[0] = testingData2;

	dataInBatch combination;
	combination.resize(1);
	cv::hconcat(t[0], t2[0], combination[0]);
	cv::Mat empty2(t2[0].rows, t2[0].cols, CV_64FC1);
	dataInBatch ccombination;
	ccombination.resize(1);
	cv::hconcat(t[0], empty2, ccombination[0]);

	dataInBatch combination2;
	combination2.resize(1);
	cv::hconcat(tt[0], tt2[0], combination2[0]);
	cv::Mat empty22(tt2[0].rows, tt2[0].cols, CV_64FC1);
	dataInBatch ccombination2;
	ccombination2.resize(1);
	cv::hconcat(tt[0], empty22, ccombination2[0]);

	AutoEncoder ae(6);
	std::vector<size_t> k;
	k.push_back(1000);
	k.push_back(500);
	ae.setLayer(k);
	std::cout << k[0] << " " << k[1] << " " << k.size() << std::endl;
	std::cout << "Reading MFCC files1" << std::endl;
	dataInBatch corruptedTrainingData = corruptImage(t, 0.2);
	std::cout << "Reading MFCC files12" << std::endl;
	dataInBatch corruptedTestingData = corruptImage(tt, 0.2);
	std::cout << "Reading MFCC files2" << std::endl;
	
	ae.denoise(0.2);
	ae.train(ccombination, 6);
	ae.fineTuning(combination, ccombination);
	ae.reconstruct(ccombination2);
	dataInBatch reTest = ae.g_reconstruction();
	double rec1 = cv::sum(cv::abs(combination2[0]))[0];
	double ori1 = cv::sum(cv::abs(reTest[0] - combination2[0]))[0];
	std::cout << rec1 << std::endl;
	std::cout << ori1 << std::endl;
	
	return 0;
}
