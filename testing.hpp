#ifndef TESTING
#define TESTING
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "util.hpp"
#include "rbm.hpp"
#include "loadData.hpp"
#include "visualization.hpp"

#include <math.h>
#include <iostream>
#include <fstream>
#include <vector>

void layerWeightTesting(RBM rbm, dataInBatch image){
    dataInBatch re = rbm.g_reconstruction();
    
    dimension boardDIM(2, 4, 1);
    dimension imgDIM(28, 28, 1);
    imageBoard board(boardDIM, imgDIM);

    std::vector<imageIndex> selectionIndex (8);
    selectionIndex[0] = imageIndex(1,3);
    selectionIndex[1] = imageIndex(2,4);
    selectionIndex[2] = imageIndex(3,7);
    selectionIndex[3] = imageIndex(1,17);
    selectionIndex[4] = imageIndex(2,14);
    selectionIndex[5] = imageIndex(3,11);
    selectionIndex[6] = imageIndex(1,9);
    selectionIndex[7] = imageIndex(2,8);

    std::vector<cv::Mat> samplesIMG = sampleSelection(image, selectionIndex);
    dataToPixel(samplesIMG);
    board.s_referenceSet(samplesIMG);
    std::vector<cv::Mat> samples = sampleSelection(re, selectionIndex);
    dataToPixel(samples);
    board.s_imageSet(samples);

    showBoard(board, 0, "Weight Tesing: reconstruction similarity");
}

void layerWeightTesting(RBM rbm, int imageHeight, int imageWidth){
    dataInBatch re = rbm.g_emission();
    
    dimension boardDIM(2, 4, 1);
    dimension imgDIM(imageHeight, imageWidth, 1);
    imageBoard board(boardDIM, imgDIM);

    std::vector<imageIndex> selectionIndex (8);
    selectionIndex[0] = imageIndex(1,3);
    selectionIndex[1] = imageIndex(2,4);
    selectionIndex[2] = imageIndex(3,7);
    selectionIndex[3] = imageIndex(1,17);
    selectionIndex[4] = imageIndex(2,14);
    selectionIndex[5] = imageIndex(3,11);
    selectionIndex[6] = imageIndex(1,9);
    selectionIndex[7] = imageIndex(2,8);

    std::vector<cv::Mat> samples = sampleSelection(re, selectionIndex);
    dataToPixel(samples);
    board.s_imageSet(samples);

    showBoard(board, 0, "Weight Tesing: reconstruction similarity");
}

std::vector<imageIndex> generateSelections(){
    std::vector<imageIndex> selectionIndex (36);
    selectionIndex[0] = imageIndex(1,3);
    selectionIndex[1] = imageIndex(2,4);
    selectionIndex[2] = imageIndex(3,7);
    selectionIndex[3] = imageIndex(1,17);
    selectionIndex[4] = imageIndex(2,14);
    selectionIndex[5] = imageIndex(3,11);
    selectionIndex[6] = imageIndex(1,9);
    selectionIndex[7] = imageIndex(2,8);
    selectionIndex[8] = imageIndex(1,4);
    selectionIndex[9] = imageIndex(2,5);
    selectionIndex[10] = imageIndex(3,8);
    selectionIndex[11] = imageIndex(1,18);
    selectionIndex[12] = imageIndex(2,15);
    selectionIndex[13] = imageIndex(3,12);
    selectionIndex[14] = imageIndex(1,10);
    selectionIndex[15] = imageIndex(2,9);
    selectionIndex[16] = imageIndex(1,11);
    selectionIndex[17] = imageIndex(2,27);
    selectionIndex[18] = imageIndex(1,88);
    selectionIndex[19] = imageIndex(2,23);
    selectionIndex[20] = imageIndex(3,24);
    selectionIndex[21] = imageIndex(1,52);
    selectionIndex[22] = imageIndex(17,15);
    selectionIndex[23] = imageIndex(13,12);
    selectionIndex[24] = imageIndex(11,10);
    selectionIndex[25] = imageIndex(2,10);
    selectionIndex[26] = imageIndex(25,13);
    selectionIndex[27] = imageIndex(2,28);
    selectionIndex[28] = imageIndex(1,89);
    selectionIndex[29] = imageIndex(2,24);
    selectionIndex[30] = imageIndex(3,25);
    selectionIndex[31] = imageIndex(1,54);
    selectionIndex[32] = imageIndex(17,19);
    selectionIndex[33] = imageIndex(13,15);
    selectionIndex[34] = imageIndex(11,17);
    selectionIndex[35] = imageIndex(24,10);
    return selectionIndex;
}

std::vector<imageIndex> randomSelection(size_t n){
    std::vector<imageIndex> selectionIndex (n);
    for (int i = 0; i < n; i++){
        int fst = rand() % 20 + 1;
        int snd = rand() % 20 + 1;
        selectionIndex.push_back(imageIndex(fst, snd));
    }
    return selectionIndex;
}




#endif
