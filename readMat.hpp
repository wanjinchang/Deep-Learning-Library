#ifndef MATREAD
#define MATREAD

#include <iostream>
#include <fstream>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgcodecs/imgcodecs.hpp"
#include <mat.h>
#include <stdio.h>
#include <vector>
/*
First set environment:
export PATH='/Applications/MATLAB_R2014b.app/bin':$PATH
echo $PATH
export DYLD_LIBRARY_PATH='/Applications/MATLAB_R2014b.app/bin/maci64:/Applications/MATLAB_R2014b.app/sys/os/maci64':$DYLD_LIBRARY_PATH

Then use the following to compile:
g++ matRead.cpp -I/Applications/MATLAB_R2014b.app/extern/include/ -L/Applications/MATLAB_R2014b.app/bin/maci64 -leng -lmx -lm -lmat -lut -lstdc++ -L/usr/local/lib -lopencv_core -lopencv_imgcodecs -lopencv_highgui -lopencv_imgproc -I/use/local/include -std=c++11
*/

cv::Mat matRead(const char* fileName, const char* variableName, const char* saveName)
{
	MATFile *matFile = matOpen(fileName,"r");
	if (matFile == NULL) printf("Cannot open mat file.\n");
	mxArray *matArray = matGetVariable(matFile, variableName);
	if (matArray == NULL) printf("Cannot read variable from mat file. \n");
	double *dataArray = (double*) mxGetData(matArray);  
	int numRow = mxGetM(matArray);
	int numCol = mxGetN(matArray); 
	cv::Mat Reading(numRow,numCol,CV_64FC1);
	double *rowPtr;
	for (int i = 0; i < numRow; i++)  
	{
		rowPtr = Reading.ptr<double>(i);
		for (int j = 0; j < numCol; j++)
		{
			rowPtr[j] = (double) dataArray[j * numRow + i];
		}
	}

	if (strcmp(saveName, "") != 0){
		std::ofstream fileOut(saveName, std::ios::binary);
		for (int i = 0; i < numRow; i++)  
		{
			for (int j = 0; j < numCol; j++)
			{	
				double temp = (double) dataArray[j * numRow + i];
				fileOut.write((char *) (&temp), sizeof(temp));
			}
		}
		fileOut.close();
	}
	mxDestroyArray(matArray);
	return Reading;
}

#endif