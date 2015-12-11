#ifndef KERNEL
#define KERNEL
#include <vector>
#include <iostream>
#include <cmath>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "util.hpp"

//In kernels, for each input, number of rows is the sample size, number of cols is dimension of variable.
data linearKernel(data& x, data&  y, std::vector<double> c){
	return x * y.t() + c[0];
}

data polynomialKernel(data& x, data&  y, std::vector<double> c){
	data output;
	pow(c[0] * x * y.t() + c[1], c[2], output);
	return output;
}

data gaussianKernel(data& x, data&  y, std::vector<double> c){
	int xSize = x.rows;
	int ySize = y.rows;
	data output(xSize, ySize, CV_64FC1);
	double* oPtr;
	for (int i = 0; i < xSize; i++){
		oPtr = output.ptr<double>(i);
		for(int j = 0; j < ySize; j++){
			data xx = x.row(i);
			data yy = y.row(j);
			data temp;
			pow(xx - yy, 2, temp);
			double temp2 = sum(temp)[0];
			temp2 = sqrt(temp2);
			temp2 = exp(-temp2 / (2 * pow(c[0], 2)));
			oPtr[j] = temp2;
		}
	}
	return output;
}

data exponentialKernel(data& x, data&  y, std::vector<double> c){
	int xSize = x.rows;
	int ySize = y.rows;
	data output(xSize, ySize, CV_64FC1);
	double* oPtr;
	for (int i = 0; i < xSize; i++){
		oPtr = output.ptr<double>(i);
		for(int j = 0; j < ySize; j++){
			data xx = x.row(i);
			data yy = y.row(j);
			data temp = abs(xx - yy);
			double temp2 = sum(temp)[0];
			temp2 = exp(-c[0] * temp2);
			oPtr[j] = temp2;
		}
	}
	return output;
}

data sigmoidKernel(data& x, data&  y, std::vector<double> c){
	int xSize = x.rows;
	int ySize = y.rows;
	data output(xSize, ySize, CV_64FC1);
	double* oPtr;
	for (int i = 0; i < xSize; i++){
		oPtr = output.ptr<double>(i);
		for(int j = 0; j < ySize; j++){
			data xx = x.row(i);
			data yy = y.row(j);
			double temp2 = sum(xx.mul(yy))[0];
			temp2 = 1 / (1 + exp(-c[0] * temp2 - c[1]));
			oPtr[j] = temp2;
		}
	}
	return output;
}

data powerKernel(data& x, data&  y, std::vector<double> c){
	int xSize = x.rows;
	int ySize = y.rows;
	data output(xSize, ySize, CV_64FC1);
	double* oPtr;
	for (int i = 0; i < xSize; i++){
		oPtr = output.ptr<double>(i);
		for(int j = 0; j < ySize; j++){
			data xx = x.row(i);
			data yy = y.row(j);
			data temp = abs(xx - yy);
			pow(temp, c[0], temp);
			double temp2 = sum(temp)[0];
			temp2 = pow(temp2, 1 / c[0]);
			oPtr[j] = -temp2;
		}
	}
	return output;
}

data logKernel(data& x, data&  y, std::vector<double> c){
	int xSize = x.rows;
	int ySize = y.rows;
	data output(xSize, ySize, CV_64FC1);
	double* oPtr;
	for (int i = 0; i < xSize; i++){
		oPtr = output.ptr<double>(i);
		for(int j = 0; j < ySize; j++){
			data xx = x.row(i);
			data yy = y.row(j);
			data temp = abs(xx - yy);
			pow(temp, c[0], temp);
			double temp2 = sum(temp)[0];
			temp2 = pow(temp2, 1 / c[0]);
			oPtr[j] = -log(temp2 + 1);
		}
	}
	return output;
}

#endif
