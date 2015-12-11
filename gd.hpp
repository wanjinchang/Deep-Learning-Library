#ifndef GD
#define GD

#include <vector>
#include <iostream>
#include <cmath>
#include <math.h>
#include <algorithm>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "penalty.hpp"
#include "matrix.hpp"
#include "util.hpp"


double annealRate(double rate, int epoch, annealType type, double parameter){
  double output;
  switch (type)
  {
    case exponential: 
      output = std::max(rate * exp( - parameter * epoch), pow(0.1, 8));
      break;
    case divideBy:
    //parameter = 0.05;
      output = std::max(rate / (1 + parameter * epoch), pow(0.1, 8));
      break;
    case step:
      output = std::max(rate * pow(0.5, epoch/5), pow(0.1,8));
      break;
    default:
      output = std::max(rate, pow(0.1, 8));
  }
  return output;
}

cv::Mat updateSGD(cv::Mat &previousUpdata, cv::Mat &gradient, double momentum, double step, WeightDecayPenalty p){
  cv::Mat gap = gradient + p.penaltyTerm();
  return momentum * previousUpdata - step * gap;
}

cv::Mat updateSGD(cv::Mat &previousUpdata, cv::Mat &gradient, double momentum, double step){
  return momentum * previousUpdata - step * gradient;
}

cv::Mat updateAdaGrad(cv::Mat &gradient, cv::Mat &gradientSquareSum, double step){
  cv::Mat square = matrixElementSquare(gradient);
  gradientSquareSum += square;
  cv::Mat root = matrixElementRoot(gradientSquareSum);
  return - step * gradient / root;
}

#endif
