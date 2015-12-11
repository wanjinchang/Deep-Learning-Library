#ifndef XRMB
#define XRMB

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <fstream>
#include <vector>
#include <assert.h>
#include <algorithm>

cv::Mat imread_XRMB_data(const char* path, int inputDim) {
	std::ifstream is(path, std::ios::in|std::ios::binary);
	if (is.is_open())
	{
		is.seekg(0, std::ios_base::end);
		int length = is.tellg();
		is.seekg(0, std::ios_base::beg);
		assert(length / sizeof(double) % inputDim == 0);
		int dataSize = (int)(length / sizeof(double) / inputDim);

		cv::Mat output(dataSize, inputDim, CV_64FC1);
    	double *rowPtr;
    	for (int r = 0; r < dataSize; r++)
    	{
    		rowPtr = output.ptr<double>(r);
        	for (int c = 0; c < inputDim; c++)
        	{
            	double temp = 0;
            	is.read((char*) &temp, sizeof(temp));
            	rowPtr[c] = temp;
        	}
    	}
    	std::cout << dataSize << " " << inputDim << std::endl;
		is.close();
		return output;
	}
	else{
		std::cout<<"Error: cannot find data file of XRMB:" << path << std::endl;
        exit(1);
	}
}

indexLabel imread_XRMB_label(const char* path) {
	std::ifstream is(path, std::ios::in|std::ios::binary);
	if (is.is_open())
	{
		is.seekg(0, std::ios_base::end);
		int length = is.tellg();
		is.seekg(0, std::ios_base::beg);
		int dataSize = (int)(length / sizeof(double));
		indexLabel output;
		std::cout << dataSize << std::endl;

    	for (int i = 0; i < dataSize; i++)
    	{
    		double temp = 0;
            is.read((char*) &temp, sizeof(temp));
            output.push_back(temp);
    	}
		is.close();
		return output;
	}
	else{
		std::cout<<"Error: cannot find data file of XRMB:" << path << std::endl;
        exit(1);
	}
}

#endif