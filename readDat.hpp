#ifndef DATAREAD
#define DATAREAD

#include <iostream>
#include <fstream>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <assert.h>
#include "util.hpp"

cv::Mat readBinary(const char* file, int rowSize){
    double *rowPtr;
    std::ifstream fileIn(file, std::ios::binary);
    fileIn.seekg(0, std::ios_base::end);
    int length = fileIn.tellg();
    fileIn.seekg(0, std::ios_base::beg);
    int colSize = (int)(length / sizeof(double) / rowSize);
    cv::Mat imageSet = cv::Mat::zeros(rowSize, colSize, CV_64FC1);
    for (int r = 0; r < rowSize; r++)
    {
        rowPtr = imageSet.ptr<double>(r);
        for (int c = 0; c < colSize; c++)
        {
            double temp = 0;
            fileIn.read((char*) &temp, sizeof(temp));
            rowPtr[c] = temp;
        }
    }
    return imageSet;
}



#endif
