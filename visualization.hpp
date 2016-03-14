#ifndef IMAGE_SHOW
#define IMAGE_SHOW
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "util.hpp"

#include <math.h>
#include <iostream>
#include <fstream>
#include <vector>

void showData(std::vector<cv::Mat> imData, imageIndex idx){
    cv::Mat temp = imData[idx.first];
    cv::Mat pieceData = temp.row(idx.second);
    pieceData = pieceData * 255;
    pieceData.convertTo(pieceData, CV_8UC1);

    uchar *piecePtr;
    piecePtr = pieceData.ptr<uchar>(0);
    cv::Mat showData(28, 28, CV_8UC1);
    uchar *showPtr;
    for (int j = 0; j < 28; j++){
        showPtr = showData.ptr<uchar>(j);
        for (int i = 0; i < 28; i++){
            showPtr[i] = piecePtr[28 * j + i];
        }
    }

    cv::imshow("Image show", showData);
    cv::waitKey(0);
}

void showPixel(std::vector<cv::Mat> imData, imageIndex idx){
    cv::Mat temp = imData[idx.first];
    cv::Mat pieceData = temp.row(idx.second);

    uchar *piecePtr;
    piecePtr = pieceData.ptr<uchar>(0);
    cv::Mat showData(28, 28, CV_8UC1);
    uchar *showPtr;
    for (int j = 0; j < 28; j++){
        showPtr = showData.ptr<uchar>(j);
        for (int i = 0; i < 28; i++){
            showPtr[i] = piecePtr[28 * j + i];
        }
    }

    cv::imshow("Image show", showData);
    cv::waitKey(0);
}

void showSample(cv::Mat imData){

    uchar *piecePtr;
    piecePtr = imData.ptr<uchar>(0);
    cv::Mat showData(28, 28, CV_8UC1);
    uchar *showPtr;
    for (int j = 0; j < 28; j++){
        showPtr = showData.ptr<uchar>(j);
        for (int i = 0; i < 28; i++){
            showPtr[i] = piecePtr[28 * j + i];
        }
    }

    cv::imshow("Image show", showData);
    cv::waitKey(0);
}

std::vector<cv::Mat> sampleSelection(dataInBatch& set, std::vector<imageIndex> indexes){
    std::vector<cv::Mat> samples;
    for (imageIndex i: indexes) {
        cv::Mat temp = set[i.first];
        cv::Mat img = temp.row(i.second);
        samples.push_back(img);
    }
    return samples;
}

void showBoard(imageBoard ib, size_t lineWidth, const char* title){
    cv::Mat board;
    size_t imageHeight = ib.imageStructure.height;
    size_t imageWidth = ib.imageStructure.width;
    size_t boardHeight = ib.boardStructure.height * (lineWidth + imageHeight);
    size_t boardWidth = ib.boardStructure.width * (lineWidth + imageWidth);

    if (!ib.referenceSet.size()){
        board.create(boardHeight, boardWidth, CV_8UC1);
    }
    else {
        board.create(2 * boardHeight + 10, boardWidth, CV_8UC1);
    }

    for (int i = 0; i < ib.imageSet.size(); i++){
        int currentRow = (i / ib.boardStructure.width) * (imageHeight + lineWidth);
        int currentCol = (i % ib.boardStructure.width) * (imageWidth + lineWidth);
        cv::Mat imData = ib.imageSet[i];
        uchar *piecePtr;
        uchar *boardPtr;
        piecePtr = imData.ptr<uchar>(0);
        for (int j = 0; j < imageHeight; j++){
            boardPtr = board.ptr<uchar>(j + currentRow);
            for (int i = 0; i < imageWidth; i++){
                boardPtr[i + currentCol] = piecePtr[imageWidth * j + i];
            }
        }
    }

    if (ib.referenceSet.size() > 0){
        uchar *tempPtr;
        for (int i = boardHeight; i < 10 + boardHeight; i++){
            tempPtr = board.ptr<uchar>(i);
            for (int j = 0; j < boardWidth; j++){
                tempPtr[j] = 50;
            }
        }

        for (int i = 0; i < ib.referenceSet.size(); i++){
            int currentRow = 10 + (i / ib.boardStructure.width + ib.boardStructure.height) * (imageHeight + lineWidth);
            int currentCol = (i % ib.boardStructure.width) * (imageWidth + lineWidth);
            cv::Mat imData = ib.referenceSet[i];
            uchar *piecePtr;
            uchar *boardPtr;
            piecePtr = imData.ptr<uchar>(0);
            for (int j = 0; j < imageHeight; j++){
                boardPtr = board.ptr<uchar>(j + currentRow);
                for (int i = 0; i < imageWidth; i++){
                    boardPtr[i + currentCol] = piecePtr[imageWidth * j + i];
                }
            }
        }
    }
    cv::imshow(title, board);
    cv::waitKey(0);
}
#endif
