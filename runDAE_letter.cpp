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
	ae.train(t, 6);
	std::cout << "Fdfadsfas" << std::endl;
	ae.fineTuning(t, corruptedTrainingData);
	ae.reconstruct(tt);
	dataInBatch reTest = ae.g_reconstruction();

	dataInBatch cor;
	dataInBatch rec;
	cv::Mat lip0 = reTest[0].row(0);
	lip0 = lip0.reshape(0, 80).t();
	cv::Mat lip1 = reTest[0].row(1);
	lip1 = lip1.reshape(0, 80).t();
	cv::Mat lip2 = reTest[0].row(2);
	lip2 = lip2.reshape(0, 80).t();
	cv::Mat lip3 = reTest[0].row(3);
	lip3 = lip3.reshape(0, 80).t();
	cv::Mat lip4 = reTest[0].row(4);
	lip4= lip4.reshape(0, 80).t();
	cv::Mat lip5 = reTest[0].row(10);
	lip5 = lip5.reshape(0, 80).t();
	cor.push_back(lip0);
	cor.push_back(lip1);
	cor.push_back(lip2);
	cor.push_back(lip3);
	cor.push_back(lip4);
	cor.push_back(lip5);
	cv::Mat lip6 = corruptedTestingData[0].row(0);
	lip6 = lip6.reshape(0, 80).t();
	cv::Mat lip7 = corruptedTestingData[0].row(1);
	lip7 = lip7.reshape(0, 80).t();
	cv::Mat lip8 = corruptedTestingData[0].row(2);
	lip8 = lip8.reshape(0, 80).t();
	cv::Mat lip9 = corruptedTestingData[0].row(3);
	lip9 = lip9.reshape(0, 80).t();
	cv::Mat lip10 = corruptedTestingData[0].row(4);
	lip10 = lip10.reshape(0, 80).t();
	cv::Mat lip11 = corruptedTestingData[0].row(10);
	lip11 = lip11.reshape(0, 80).t();
	rec.push_back(lip6);
	rec.push_back(lip7);
	rec.push_back(lip8);
	rec.push_back(lip9);
	rec.push_back(lip10);
	rec.push_back(lip11);
	cv::Mat board(120, 480, CV_64FC1);
	for (int i = 0; i < 60; i++){
		for (int j = 0; j < 6; j++){
			for (int k = 0; k < 80; k++){
				board.at<double>(i, j * 80 + k) = cor[j].at<double>(i, k);
			}
		}
	}
	for (int i = 60; i < 120; i++){
		for (int j = 0; j < 6; j++){
			for (int k = 0; k < 80; k++){
				board.at<double>(i, j * 80 + k) = rec[j].at<double>(i - 60, k);
			}
		}
	}
	cv::imshow( "avletters", board);        
    cv::waitKey(0);

	
	return 0;
}