#include "readMat.hpp"
#include <iostream>
#include <fstream>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgcodecs/imgcodecs.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
int main(){
	cv::Mat reading =  matRead("data/avletters/Lips/A1_Anya-lips.mat", "vid", "A1_Anya-lips.dat");
	std::cout << reading.rows << std::endl;
	std::cout << reading.cols << std::endl;
	reading =  matRead("data/avletters/Lips/B3_Anya-lips.mat", "vid", "B3_Anya-lips.dat");
	std::cout << reading.rows << std::endl;
	std::cout << reading.cols << std::endl;
	reading =  matRead("data/avletters/Lips/C1_Anya-lips.mat", "vid", "C1_Anya-lips.dat");
	std::cout << reading.rows << std::endl;
	std::cout << reading.cols << std::endl;
	reading =  matRead("data/avletters/Lips/D1_Anya-lips.mat", "vid", "D1_Anya-lips.dat");
	std::cout << reading.rows << std::endl;
	std::cout << reading.cols << std::endl;
	reading =  matRead("data/avletters/Lips/E1_Anya-lips.mat", "vid", "E1_Anya-lips.dat");
	std::cout << reading.rows << std::endl;
	std::cout << reading.cols << std::endl;
	reading =  matRead("data/avletters/Lips/F1_Anya-lips.mat", "vid", "F1_Anya-lips.dat");
	std::cout << reading.rows << std::endl;
	std::cout << reading.cols << std::endl;
	reading =  matRead("data/avletters/Lips/G1_Anya-lips.mat", "vid", "G1_Anya-lips.dat");
	std::cout << reading.rows << std::endl;
	std::cout << reading.cols << std::endl;
	reading =  matRead("data/avletters/Lips/H1_Anya-lips.mat", "vid", "H1_Anya-lips.dat");
	std::cout << reading.rows << std::endl;
	std::cout << reading.cols << std::endl;
	reading =  matRead("data/avletters/Lips/I1_Anya-lips.mat", "vid", "I1_Anya-lips.dat");
	std::cout << "reading succeed" << std::endl;
	std::cout << reading.rows << std::endl;
	std::cout << reading.cols << std::endl;
	return 0;
}
