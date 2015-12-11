#include "readDat.hpp"
#include "readAvLetters.hpp"
#include <iostream>
#include <fstream>
#include "opencv2/imgcodecs/imgcodecs.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
int main(){
	cv::Mat reading =  readBinary("A1_Anya-lips.dat", 4800);
	std::cout << "reading succeed" << std::endl;
	//cv::Mat reading = imread_avletters_mfcc("data/avletters/Audio/B3_Anya.mfcc");
	std::cout << reading.rows << std::endl;
	std::cout << reading.cols << std::endl;
	std::cout << reading << std::endl;
	return 0;
}
