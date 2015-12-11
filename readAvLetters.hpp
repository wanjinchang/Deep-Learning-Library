#ifndef MFCC_READ
#define MFCC_READ
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <math.h>
#include <iostream>
#include <fstream>
#include <vector>
#include "util.hpp"
#include <stdint.h>


cv::Mat imread_avletters_mfcc(const char* path)
{   
    std::ifstream file (path, std::ios::binary);
    if (file.is_open())
    {   
        unsigned char temp[sizeof(int)];
        file.read(reinterpret_cast<char*>(temp), sizeof(int));
        unsigned char t = temp[0];
        temp[0] = temp[3];
        temp[3] = t;
        t = temp[1];
        temp[1] = temp[2];
        temp[2] = t;
        int numSample = reinterpret_cast<int&>(temp);
        file.read(reinterpret_cast<char*>(temp), sizeof(int));
        std::cout << "Number of samples is: " << numSample << std::endl;

        unsigned char temp2[sizeof(int16_t)];
        file.read(reinterpret_cast<char*>(temp2), sizeof(int16_t));
        t = temp2[0];
        temp2[0] = temp2[1];
        temp2[1] = t;
        int16_t inputDim = reinterpret_cast<int16_t&>(temp2) / 4;
        file.read(reinterpret_cast<char*>(temp2), sizeof(int16_t));
        std::cout << "Input dimension is: " << inputDim << std::endl;

        cv::Mat mfccSet(numSample, inputDim, CV_32FC1);
        float *rowPtr;
        for(int i = 0; i < numSample; i++)
        {
            rowPtr = mfccSet.ptr<float>(i);
            for (int j = 0; j < inputDim; j++){ 
                unsigned char temp3[sizeof(float)];
                file.read(reinterpret_cast<char*>(temp3), sizeof(float));
                unsigned char t = temp3[0];
                temp3[0] = temp3[3];
                temp3[3] = t;
                t = temp3[1];
                temp3[1] = temp3[2];
                temp3[2] = t;
                float k = reinterpret_cast<float&>(temp3);
                rowPtr[j] = k;
            }
        }
        mfccSet.convertTo(mfccSet, CV_64FC1);
        /*
        for (int i =0 ; i < numSample; i++){
            for (int j = 0; j < 12; j++){
                std::cout << mfccSet.at<double>(i,j) << " ";
            }
            std::cout << '\n';
        }
        */
        return mfccSet.t();
    }
    else {
        throw errorReport("No file found.");
        exit(1);
    }
}

#endif