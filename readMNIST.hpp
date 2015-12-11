#ifndef IMAGE_MNIST
#define IMAGE_MNIST
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <math.h>
#include <iostream>
#include <fstream>
#include <vector>

void reverseEvery8bit (std::vector<int>& input)
{
    unsigned char part_1, part_2, part_3, part_4;
    int origin = 0;
    for (int i = 0; i < input.size(); i++){
        origin = input[i];
        part_1 = origin & 255;
        part_2 = (origin >> 8) & 255;
        part_3 = (origin >> 16) & 255;
        part_4 = (origin >> 24) & 255;
        input[i] = ((int) part_1 << 24) + ((int)part_2 << 16) + ((int)part_3 << 8) + part_4;
    }
}

cv::Mat imread_mnist_image(const char* path)
{
    //std::string path = "t10k-images-idx3-ubyte";
    std::ifstream file (path, std::ios::binary);
    cv::Mat imageSet;
    if (file.is_open())
    {
        std::vector<int> information;
        information.resize(4);
        file.read((char*) &information[0], sizeof(information[0]));
        file.read((char*) &information[1], sizeof(information[1]));
        file.read((char*) &information[2], sizeof(information[2]));
        file.read((char*) &information[3], sizeof(information[3]));
        reverseEvery8bit(information);

        //information[1] stores image amount, information[2] stores number of rows, information[3] sotres number of columns
        int numImage = information[1];
        int rowSize = information[2];
        int colSize = information[3];
 
        imageSet = cv::Mat::zeros(numImage, rowSize * colSize, CV_8UC1);
        uchar *rowPtr;
        for (int k = 0; k < numImage; k++)
        {
            rowPtr = imageSet.ptr<uchar>(k);
            for (int r = 0; r < rowSize; r++)
            {
                for (int c = 0; c < colSize; c++)
                {
                    unsigned char temp = 0;
                    file.read((char*) &temp, sizeof(temp));
                    rowPtr[r * rowSize + c] = temp;
                }
            }
        }
    }
    else {
        std::cout<<"Error: cannot find data file (from readMNIST):" << path << std::endl;
        cv::Mat batch = cv::Mat::zeros(1, 1, CV_8UC1);
        imageSet = batch;
        exit(1);
    }
    return imageSet;
}

cv::Mat imread_mnist_label(const char* path)
{
    //std::string path = "t10k-images-idx3-ubyte";
    std::ifstream file (path, std::ios::binary);
    cv::Mat labelSet;
    if (file.is_open())
    {
        std::vector<int> information;
        information.resize(4);
        file.read((char*) &information[0], sizeof(information[0]));
        file.read((char*) &information[1], sizeof(information[1]));
        reverseEvery8bit(information);

        //information[1] stores image amount
        int numImage = information[1];
 
        labelSet = cv::Mat::zeros(numImage, 10, CV_8UC1);
        std::vector<unsigned char> rawLabel;
        rawLabel.resize(numImage);
        
        for (int k = 0; k < numImage; k++)
        {
            unsigned char temp = 0;
            file.read((char*) &temp, sizeof(temp));        
            rawLabel[k] = temp;
        }

        uchar *rowPtr;
        for (int k = 0; k < numImage; k++)
        {
            rowPtr = labelSet.ptr<uchar>(k);
            int index = rawLabel[k];
            rowPtr[index] = 1;
        }
    }
    else {
        std::cout<<"Error: cannot find data file (from imread_MNIST)"<<std::endl;
        cv::Mat batch = cv::Mat::zeros(1, 1, CV_8UC1);
        labelSet = batch;
    }
    return labelSet;
}

#endif