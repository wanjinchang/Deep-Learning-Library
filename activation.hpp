#ifndef ACTIVATION
#define ACTIVATION
#include <vector>
#include <math.h>
#include <iostream>
#include <cmath>
#include <algorithm>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "util.hpp"

/*
cv::Mat sigmoidACT(cv::Mat &v, cv::Mat &p, cv::Mat &b)
{
	cv::Mat product = v * p;
	cv::Mat output;
	output.create(v.rows, p.cols, CV_64FC1);
	double *PtrP;
	double *PtrO;
	double *PtrB;
	PtrB = b.ptr<double>(0);
	for (int i = 0; i < output.rows; i++){
		PtrO = output.ptr<double>(i);
		PtrP = product.ptr<double>(i);
		for (int j = 0; j < output.cols; j++){
			PtrO[j] = (double) 1 / (1 + exp(-PtrP[j] - PtrB[j]));
		}
	}
	return output;
}
*/

cv::Mat sigmoidACT(cv::Mat &v, cv::Mat &p, cv::Mat &b)
{
	cv::Mat netIn = v * p + repeat(b, v.rows, 1);
	cv::Mat output;
	exp(-netIn, output);
	return 1.0 / (1.0 + output);
}

cv::Mat sigmoid_multi(cv::Mat input, cv::Mat b)
{
	cv::Mat output;
	output.create(input.rows, input.cols, CV_64FC1);
	double *PtrI;
	double *PtrO;
	double *PtrB;
	PtrB = b.ptr<double>(0);
	for (int i = 0; i < output.rows; i++){
		PtrO = output.ptr<double>(i);
		PtrI = input.ptr<double>(i);
		for (int j = 0; j < output.cols; j++){
			PtrO[j] = (double) 1 / (1 + exp(-PtrI[j] - PtrB[j]));
		}
	}
	return output;
}

cv::Mat sigmoidF(cv::Mat &input)
{
	cv::Mat output;
	output.create(input.rows, input.cols, CV_64FC1);
	double *PtrI;
	double *PtrO;
	for (int i = 0; i < output.rows; i++){
		PtrO = output.ptr<double>(i);
		PtrI = input.ptr<double>(i);
		for (int j = 0; j < output.cols; j++){
			PtrO[j] = (double) 1 / (1 + exp(-PtrI[j]));
		}
	}
	return output;
}

cv::Mat tanhACT(cv::Mat &v, cv::Mat &p, cv::Mat &b)
{
	cv::Mat product = v * p;
	cv::Mat output;
	output.create(v.rows, p.cols, CV_64FC1);
	double *PtrP;
	double *PtrO;
	double *PtrB;
	PtrB = b.ptr<double>(0);
	for (int i = 0; i < output.rows; i++){
		PtrO = output.ptr<double>(i);
		PtrP = product.ptr<double>(i);
		for (int j = 0; j < output.cols; j++){
			double exp_z = exp(PtrP[j] - PtrB[j]);
			double exp_mz = (double) 1 / exp_z;
			PtrO[j] = (exp_z - exp_mz) / (exp_z + exp_mz);
		}
	}
	return output;
}

cv::Mat tanh_multi(cv::Mat product, cv::Mat b)
{
	cv::Mat output;
	output.create(product.rows, product.cols, CV_64FC1);
	double *PtrP;
	double *PtrO;
	double *PtrB;
	PtrB = b.ptr<double>(0);
	for (int i = 0; i < output.rows; i++){
		PtrO = output.ptr<double>(i);
		PtrP = product.ptr<double>(i);
		for (int j = 0; j < output.cols; j++){
			double exp_z = exp(PtrP[j] - PtrB[j]);
			double exp_mz = (double) 1 / exp_z;
			PtrO[j] = (exp_z - exp_mz) / (exp_z + exp_mz);
		}
	}
	return output;
}

cv::Mat tanhF(cv::Mat &input)
{
	cv::Mat output;
	output.create(input.rows, input.cols, CV_64FC1);
	double *PtrI;
	double *PtrO;
	for (int i = 0; i < output.rows; i++){
		PtrO = output.ptr<double>(i);
		PtrI = input.ptr<double>(i);
		for (int j = 0; j < output.cols; j++){
			double exp_z = exp(PtrI[j]);
			double exp_mz = (double) 1 / exp_z;
			PtrO[j] = (exp_z - exp_mz) / (exp_z + exp_mz);
		}
	}
	return output;
}

cv::Mat reluACT(cv::Mat &v, cv::Mat &p, cv::Mat &b)
{
	cv::Mat product = v * p;
	cv::Mat output;
	output.create(v.rows, p.cols, CV_64FC1);
	double *PtrP;
	double *PtrO;
	double *PtrB;
	PtrB = b.ptr<double>(0);
	for (int i = 0; i < output.rows; i++){
		PtrO = output.ptr<double>(i);
		PtrP = product.ptr<double>(i);
		for (int j = 0; j < output.cols; j++){
			PtrO[j] = std::max(0.0, PtrP[j] + PtrB[j]);
		}
	}
	return output;
}

cv::Mat relu_multi(cv::Mat product, cv::Mat b)
{
	cv::Mat output;
	output.create(product.rows, product.cols, CV_64FC1);
	double *PtrP;
	double *PtrO;
	double *PtrB;
	PtrB = b.ptr<double>(0);
	for (int i = 0; i < output.rows; i++){
		PtrO = output.ptr<double>(i);
		PtrP = product.ptr<double>(i);
		for (int j = 0; j < output.cols; j++){
			PtrO[j] = std::max(0.0, PtrP[j] + PtrB[j]);
		}
	}
	return output;
}

cv::Mat reluF(cv::Mat &input)
{
	cv::Mat output;
	output.create(input.rows, input.cols, CV_64FC1);
	double *PtrI;
	double *PtrO;
	for (int i = 0; i < output.rows; i++){
		PtrO = output.ptr<double>(i);
		PtrI = input.ptr<double>(i);
		for (int j = 0; j < output.cols; j++){
			PtrO[j] = std::max(0.0, PtrI[j]);
		}
	}
	return output;
}

cv::Mat leaky_reluACT(cv::Mat &v, cv::Mat &p, cv::Mat &b)
{
	cv::Mat product = v * p;
	cv::Mat output;
	output.create(v.rows, p.cols, CV_64FC1);
	double *PtrP;
	double *PtrO;
	double *PtrB;
	PtrB = b.ptr<double>(0);
	for (int i = 0; i < output.rows; i++){
		PtrO = output.ptr<double>(i);
		PtrP = product.ptr<double>(i);
		for (int j = 0; j < output.cols; j++){
			double temp = PtrP[j] + PtrB[j];
			PtrO[j] = temp > 0 ? temp: 0.01 * temp;
		}
	}
	return output;
}

cv::Mat leaky_relu_multi(cv::Mat product, cv::Mat b)
{
	cv::Mat output;
	output.create(product.rows, product.cols, CV_64FC1);
	double *PtrP;
	double *PtrO;
	double *PtrB;
	PtrB = b.ptr<double>(0);
	for (int i = 0; i < output.rows; i++){
		PtrO = output.ptr<double>(i);
		PtrP = product.ptr<double>(i);
		for (int j = 0; j < output.cols; j++){
			double temp = PtrP[j] + PtrB[j];
			PtrO[j] = temp > 0 ? temp: 0.01 * temp;
		}
	}
	return output;
}

cv::Mat leaky_reluF(cv::Mat &input)
{
	cv::Mat output;
	output.create(input.rows, input.cols, CV_64FC1);
	double *PtrI;
	double *PtrO;
	for (int i = 0; i < output.rows; i++){
		PtrO = output.ptr<double>(i);
		PtrI = input.ptr<double>(i);
		for (int j = 0; j < output.cols; j++){
			PtrO[j] = PtrI[j] > 0 ? PtrI[j]: 0.01 * PtrI[j];
		}
	}
	return output;
}

//bias here is not used.
cv::Mat softmaxACT(cv::Mat &v, cv::Mat &p, cv::Mat &b)
{
	cv::Mat product = v * p;
	cv::Mat output;
	output.create(v.rows, p.cols, CV_64FC1);
	double *PtrO;
	double *PtrI;
	for (int i = 0; i < output.rows; i++){
		cv::Mat InRow = product.row(i) + b;
		PtrI = InRow.ptr<double>(0);
		PtrO = output.ptr<double>(i);
		double maxVal = 0;
		cv::minMaxLoc(InRow, NULL, &maxVal, NULL, NULL);

		double sumExp = 0; 
		for (int j = 0; j < output.cols; j++){
			double temp = exp(PtrI[j] - maxVal);
			sumExp += temp;
			PtrO[j] = temp;
		}

		for (int j = 0; j < output.cols; j++){
			PtrO[j] = (double) PtrO[j] / sumExp;
		}
	}
	return output;
}

cv::Mat softmax_multi(cv::Mat product, cv::Mat b)
{
	cv::Mat output;
	output.create(product.rows, product.cols, CV_64FC1);
	double *PtrO;
	double *PtrI;
	for (int i = 0; i < output.rows; i++){
		cv::Mat InRow = product.row(i) + b;
		PtrI = InRow.ptr<double>(0);
		PtrO = output.ptr<double>(i);
		double maxVal = 0;
		cv::minMaxLoc(InRow, NULL, &maxVal, NULL, NULL);

		double sumExp = 0; 
		for (int j = 0; j < output.cols; j++){
			double temp = exp(PtrI[j] - maxVal);
			sumExp += temp;
			PtrO[j] = temp;
		}

		for (int j = 0; j < output.cols; j++){
			PtrO[j] = (double) PtrO[j] / sumExp;
		}
	}
	return output;
}

cv::Mat softmaxF(cv::Mat &input)
{
	cv::Mat output;
	output.create(input.rows, input.cols, CV_64FC1);
	double *PtrO;
	double *PtrI;
	for (int i = 0; i < output.rows; i++){
		cv::Mat InRow = input.row(i);
		PtrI = InRow.ptr<double>(0);
		PtrO = output.ptr<double>(i);
		double maxVal = 0;
		cv::minMaxLoc(InRow, NULL, &maxVal, NULL, NULL);

		double sumExp = 0; 
		for (int j = 0; j < output.cols; j++){
			double temp = exp(PtrI[j] - maxVal);
			sumExp += temp;
			PtrO[j] = temp;
		}

		for (int j = 0; j < output.cols; j++){
			PtrO[j] = (double) PtrO[j] / sumExp;
		}
	}
	return output;
}

cv::Mat identityACT(cv::Mat &v, cv::Mat &p, cv::Mat &b)
{
	cv::Mat product = v * p;
	cv::Mat output;
	output.create(v.rows, p.cols, CV_64FC1);
	double *PtrP;
	double *PtrO;
	double *PtrB;
	PtrB = b.ptr<double>(0);
	for (int i = 0; i < output.rows; i++){
		PtrO = output.ptr<double>(i);
		PtrP = product.ptr<double>(i);
		for (int j = 0; j < output.cols; j++){
			PtrO[j] = PtrP[j] + PtrB[j];
		}
	}
	return output;
}

cv::Mat identity_multi(cv::Mat product, cv::Mat b)
{
	cv::Mat output;
	output.create(product.rows, product.cols, CV_64FC1);
	double *PtrP;
	double *PtrO;
	double *PtrB;
	PtrB = b.ptr<double>(0);
	for (int i = 0; i < output.rows; i++){
		PtrO = output.ptr<double>(i);
		PtrP = product.ptr<double>(i);
		for (int j = 0; j < output.cols; j++){
			PtrO[j] = PtrP[j] + PtrB[j];
		}
	}
	return output;
}

cv::Mat identityF(cv::Mat &input)
{
	return input;
}

cv::Mat exponentialACT(cv::Mat &v, cv::Mat &p, cv::Mat &b)
{
	cv::Mat product = v * p;
	cv::Mat output;
	output.create(v.rows, p.cols, CV_64FC1);
	double *PtrP;
	double *PtrO;
	double *PtrB;
	PtrB = b.ptr<double>(0);
	for (int i = 0; i < output.rows; i++){
		PtrO = output.ptr<double>(i);
		PtrP = product.ptr<double>(i);
		for (int j = 0; j < output.cols; j++){
			PtrO[j] = exp(PtrP[j] + PtrB[j]);
		}
	}
	return output;
}

cv::Mat exponential_multi(cv::Mat product, cv::Mat b)
{
	cv::Mat output;
	output.create(product.rows, product.cols, CV_64FC1);
	double *PtrP;
	double *PtrO;
	double *PtrB;
	PtrB = b.ptr<double>(0);
	for (int i = 0; i < output.rows; i++){
		PtrO = output.ptr<double>(i);
		PtrP = product.ptr<double>(i);
		for (int j = 0; j < output.cols; j++){
			PtrO[j] = exp(PtrP[j] + PtrB[j]);
		}
	}
	return output;
}

cv::Mat exponentialF(cv::Mat &input)
{
	cv::Mat output;
	output.create(input.rows,input.cols, CV_64FC1);
	double *PtrI;
	double *PtrO;
	for (int i = 0; i < input.rows; i++){
		PtrO = output.ptr<double>(i);
		PtrI = input.ptr<double>(i);
		for (int j = 0; j < input.cols; j++){
			PtrO[j] = exp(PtrI[j]);
		}
	}
	return output;
}

cv::Mat dsigmoidACT(cv::Mat &input)
{
	return (1 - input).mul(input);
}
//Notice: not finished
cv::Mat dtanhACT(cv::Mat &input)
{
	return (1 + input).mul(1 + input);
}

cv::Mat dreluACT(cv::Mat &input)
{
	cv::Mat output;
	output.create(input.rows, input.cols, CV_64FC1);
	double *PtrI;
	double *PtrO;
	for (int i = 0; i < output.rows; i++){
		PtrO = output.ptr<double>(i);
		PtrI = input.ptr<double>(i);
		for (int j = 0; j < output.cols; j++){
			PtrO[j] = std::max(0.0, PtrI[j]);
		}
	}
	return output;
}

cv::Mat dleaky_reluACT(cv::Mat &input)
{
	cv::Mat output;
	output.create(input.rows, input.cols, CV_64FC1);
	double *PtrI;
	double *PtrO;
	for (int i = 0; i < output.rows; i++){
		PtrO = output.ptr<double>(i);
		PtrI = input.ptr<double>(i);
		for (int j = 0; j < output.cols; j++){
			PtrO[j] = PtrI[j] > 0 ? PtrI[j]: 0.01 * PtrI[j];
		}
	}
	return output;
}

cv::Mat dsoftmaxACT(cv::Mat &input)
{
	return input.mul(1 - input);
}

cv::Mat didentityACT(cv::Mat &input)
{
	return input;
}

#endif
