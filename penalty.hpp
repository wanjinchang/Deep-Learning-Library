#ifndef PENALTY
#define PENALTY
#include <vector>
#include <iostream>
#include <cmath>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "matrix.hpp"
#include "util.hpp"

template <typename T>
class Penalty{
protected:
	T penaltyValue;
  
public:
  	T penaltyTerm(){return penaltyValue;}
};

class WeightDecayPenalty: public Penalty<cv::Mat>{
	double lambda;
	cv::Mat weight;
	RegType type;
public:
  	WeightDecayPenalty(RegType rt, double l, cv::Mat &w): type(rt), lambda(l), weight(w){}
  	double penaltyTerm();
};

double WeightDecayPenalty::penaltyTerm(){
	double output;
	if (type == L1){
		cv::Mat absMat = cv::abs(weight);
		double output = lambda * cv::sum(absMat)[0];
	}
	else if (type == L2){
		cv::Mat squareMat = matrixElementSquare(weight);
		double output = lambda * cv::sum(squareMat)[0];
	}
	else {
		output = lambda;
	}
	return output;
};

#endif
