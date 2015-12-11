#ifndef DATA_PROCESS
#define DATA_PROCESS
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <math.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include "util.hpp"

cv::Mat denoiseMask(size_t rowSize, size_t colSize, double rate){
    cv::Mat mask;
    mask.create(rowSize, colSize, CV_64FC1);
    double *rowPtr_mask;
    for (int i = 0; i < rowSize; i++){
        rowPtr_mask = mask.ptr<double>(i);
        for (int j = 0; j < colSize; j++){
            rowPtr_mask[j] = ((double) rand()/ RAND_MAX) > rate;
        }
    }
    return mask;
}

dataInBatch imageBatches(cv::Mat& m, int numBatch)
{
    dataInBatch batchSet;
    batchSet.resize(numBatch);

    int numImage = m.rows;
    int dim = m.cols;
    int batchSize = numImage / numBatch;

    for (int i = 0; i < numBatch; i++)
    {   
        cv::Mat batch = cv::Mat::zeros(batchSize, dim, CV_64FC1);
        double *rowPtr;
        double *rowPtrM;
        for (int k = 0; k < batchSize; k++)
        {
            rowPtr = batch.ptr<double>(k);
            rowPtrM = m.ptr<double>(i * batchSize + k);
            for (int j = 0; j < dim; j++)
            {
                rowPtr[j] = rowPtrM[j];
            }
        }
        batchSet[i] = batch;
    }
    return batchSet;
}

dataInBatch labelBatches(cv::Mat& m, int numBatch)
{
    dataInBatch batchSet;
    batchSet.resize(numBatch);

    int numImage = m.rows;
    int dim = m.cols;
    int batchSize = numImage / numBatch;

    for (int i = 0; i < numBatch; i++)
    {   
        cv::Mat batch = cv::Mat::zeros(batchSize, dim, CV_8UC1);
        uchar *rowPtr;
        uchar *rowPtrM;
        for (int k = 0; k < batchSize; k++)
        {
            rowPtr = batch.ptr<uchar>(k);
            rowPtrM = m.ptr<uchar>(i * batchSize + k);
            for (int j = 0; j < dim; j++)
            {
                rowPtr[j] = rowPtrM[j];
            }
        }
        batchSet[i] = batch;
    }
    return batchSet;
}

void labelPixelToData(dataInBatch& m)
{   
    for (size_t i = 0; i < m.size(); i++){
        m[i].convertTo(m[i], CV_64FC1);
    }
}

void pixelToData(cv::Mat& m)
{
    m.convertTo(m, CV_64FC1);
    m = m/255;
}

void dataToPixel(cv::Mat& m)
{   
    m = m * 255;
    m.convertTo(m, CV_8UC1);
}

void dataToPixel(dataInBatch& m)
{   
    for (int i = 0; i < m.size(); i++){
        m[i] = m[i] * 255;
        m[i].convertTo(m[i], CV_8UC1);
    }
}

dataCollection shuffleByRow(dataCollection& m)
{
if (!m.size()){
    throw errorReport("shuffleByRow has no target.");
}
dataCollection output;
std::vector<size_t> perm;
size_t dataSize = m[0].rows;
perm.resize(dataSize);
output.resize(m.size());

for (size_t i = 0; i < dataSize; i++){
    perm[i] = i;
}
std::random_shuffle(perm.begin(), perm.end());

for (size_t j = 0; j < dataSize; j++){
    size_t index = perm[j];
    for (int p = 0; p < m.size(); p++){
        output[p].push_back(m[p].row(index));
    }
}

return output;
}

std::vector<dataInBatch> dataProcess(dataCollection& reading, int numBatch){
    dataCollection shuffled = shuffleByRow(reading);
    pixelToData(shuffled[0]);
    std::vector<dataInBatch> output;
    output.resize(reading.size());
    output[0] = imageBatches(shuffled[0], numBatch);
    output[1] = labelBatches(shuffled[1], numBatch);
    return output;
}

data oneOfK(indexLabel l, int labelDim){
    size_t numLabel = l.size();
    data output = cv::Mat::zeros(numLabel, labelDim, CV_64FC1);
    double *PtrO;
    for (size_t i = 0; i < numLabel; i++){
        PtrO = output.ptr<double>(i);
        int index = l[i];
        PtrO[index] = 1;
    }
    return output;
}

cv::Mat oneOfK(int l, int labelDim){
    data output = cv::Mat::zeros(1, labelDim, CV_64FC1);
    output.at<double>(0, l) = 1.0;
    return output;
}

dataInBatch corruptImage(dataInBatch input, double denoiseRate){
    size_t numBatch = input.size();
    size_t numData = input[0].rows;
    size_t dataDimension = input[0].cols;
    dataInBatch output;
    output.resize(numBatch);
    if (denoiseRate > 0){
        for (int i = 0; i < numBatch; i++) {
            std::cout << "hjkhk" << std::endl;
            cv::Mat mask = denoiseMask(numData, dataDimension, denoiseRate);
            output[i] = input[i].mul(mask);
        }
    }
    return output;  
}

#endif
