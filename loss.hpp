#ifndef LOSS
#define LOSS

#include <vector>
#include <iostream>
#include <cmath>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"

template<typename T> using errorFunction = T (*) (T&, T&);
template<typename T>
T squareLoss(T &est, T &lab);
template<typename T>
T absoluteLoss(T &est, T &lab);
template<typename T>
T binaryLoss(T &est, T &lab);
template<typename T>
T crossEntropyLoss(T &est, T &lab);
template<typename T>
T dSquare(T &est, T &lab);
template<typename T>
T dAbsolute(T &est, T &lab);
template<typename T>
T dBinary(T &est, T &lab);
template<typename T>
T dCrossEntropy(T &est, T &lab);

template <typename T>
class Error{
  LossType type;
  T estimation;
  T label;
  errorFunction<T> efunction;
  errorFunction<T> defunction;
  public:
    Error(LossType tp);
    Error(LossType tp, T &est, T &lab);
    bool sizeCheck();
    bool sizeCheck(T &est, T &lab);
    void fill(T &est, T &lab);
    std::string functionList();
    T value () const;
    T value(T &est, T &lab) const;
    T dvalue() const;
    T dvalue(T &est, T &lab) const;
};

template <typename T>
Error<T>::Error(LossType tp): type(tp)
{
    switch (tp)
    {
      case MSE: 
        efunction = squareLoss;
        defunction = dSquare;
        break;
      case _0_1:
        efunction = binaryLoss;
        defunction = dBinary;
        break;
      case Absolute:
        efunction = absoluteLoss;
        defunction = dAbsolute;
        break;
      case CrossEntropy:
        efunction = crossEntropyLoss;
        defunction = dCrossEntropy;
      default:
        efunction = squareLoss;
        defunction = dSquare;      
    }
}

template <typename T>
Error<T>::Error(LossType tp, T &est, T &lab): type(tp), estimation(est), label(lab)
{
    switch (tp)
    {
      case MSE: 
        efunction = squareLoss;
        defunction = dSquare;
        break;
      case _0_1:
        efunction = binaryLoss;
        defunction = dBinary;
        break;
      case Absolute:
        efunction = absoluteLoss;
        defunction = dAbsolute;
        break;
      default:
        efunction = crossEntropyLoss;
        defunction = dCrossEntropy;
    }
}

template <>
bool Error<cv::Mat>::sizeCheck(){
  return (estimation.rows == label.rows && estimation.cols == label.cols);
}

template <>
bool Error<cv::Mat>::sizeCheck(cv::Mat &est, cv::Mat &lab){
  if (est.rows == lab.rows && est.cols == lab.cols){
    estimation = est;
    label = lab;
    return true;
  }
  else{
    return false;
  }
}

template <typename T>
std::string Error<T>::functionList(){
  return "MSE, _0_1, Absolute, CrossEntropy(default)";
}

template <typename T>
T Error<T>::value() const {
  return efunction(estimation, label);
}

template <typename T>
T Error<T>::value(T &est, T &lab) const {
  return efunction(est, lab);
}

template <typename T>
void Error<T>::fill(T &est, T &lab){
  estimation = est;
  label = lab;
}

template <typename T>
T Error<T>::dvalue() const {
  return defunction(estimation, label);
}

template <typename T>
T Error<T>::dvalue(T &est, T &lab) const {
  return defunction(est, lab);
}

template<>
cv::Mat squareLoss(cv::Mat &lab, cv::Mat &est){
  size_t numRow = est.rows;
  size_t numCol = est.cols;
  cv::Mat output;
  output.create(numRow, 1, CV_64FC1);
  double *rowPtr_est;
  double *rowPtr_lab;
  for (int i = 0; i < numRow; i++){
    double temp = 0;
    rowPtr_est = est.ptr<double>(i);
    rowPtr_lab = lab.ptr<double>(i);
    for (int j = 0; j < numCol; j++){
      temp += pow(rowPtr_est[j] - rowPtr_lab[j], 2.0) / 2;
    }
    output.at<double>(i, 0) = temp;
  }
  return output;
}

template<>
cv::Mat absoluteLoss(cv::Mat &lab, cv::Mat &est){
  size_t numRow = est.rows;
  size_t numCol = est.cols;
  cv::Mat output;
  output.create(numRow, 1, CV_64FC1);
  double *rowPtr_est;
  double *rowPtr_lab;

  for (int i = 0; i < numRow; i++){
    double temp = 0;
    rowPtr_est = est.ptr<double>(i);
    rowPtr_lab = lab.ptr<double>(i);
    for (int j = 0; j < numCol; j++){
      temp += std::abs(rowPtr_est[j] - rowPtr_lab[j]);
    }
    output.at<double>(i, 0) = temp;
  }
  return output;
}

template<>
cv::Mat binaryLoss(cv::Mat &est, cv::Mat &lab){
  size_t numRow = est.rows;
  size_t numCol = est.cols;
  cv::Mat output;
  output.create(numRow, 1, CV_64FC1);
  double *rowPtr_est;
  double *rowPtr_lab;

  for (int i = 0; i < numRow; i++){
    double temp = 0;
    rowPtr_est = est.ptr<double>(i);
    rowPtr_lab = lab.ptr<double>(i);
    for (int j = 0; j < numCol; j++){
      if (rowPtr_est[j] != rowPtr_lab[j]){
        temp = 1;
        break;
      }
    }
    output.at<double>(i, 0) = (temp == 0) ? 0 : 1;
  }
  return output;
}

template<>
cv::Mat crossEntropyLoss(cv::Mat &est, cv::Mat &lab){
  size_t numRow = est.rows;
  size_t numCol = est.cols;
  cv::Mat output;
  output.create(numRow, 1, CV_64FC1);
  double *rowPtr_est;
  double *rowPtr_lab;

  for (int i = 0; i < numRow; i++){
    double temp = 0;
    rowPtr_est = est.ptr<double>(i);
    rowPtr_lab = lab.ptr<double>(i);
    for (int j = 0; j < numCol; j++){
      temp -= std::log(rowPtr_est[j]) * rowPtr_lab[j];
    }
    output.at<double>(i, 0) = temp;
  }
  return output;
}

template<>
cv::Mat dSquare(cv::Mat &est, cv::Mat &lab){
  return est - lab;
}

/*
template<>
cv::Mat dCrossEntropy(cv::Mat &est, cv::Mat &lab){
  size_t numRow = est.rows;
  size_t numCol = est.cols;
  cv::Mat output;
  output.create(numRow, numCol, CV_64FC1);
  double *rowPtr_est;
  double *rowPtr_lab;
  double *rowPtr_output;

  for (int i = 0; i < numRow; i++){
    rowPtr_est = est.ptr<double>(i);
    rowPtr_lab = lab.ptr<double>(i);
    rowPtr_output = output.ptr<double>(i);
    for (int j = 0; j < numCol; j++){
      rowPtr_output[j] = - rowPtr_lab[j] / rowPtr_est[j];
    }
  }
  return output;
}
*/
template<>
cv::Mat dCrossEntropy(cv::Mat &est, cv::Mat &lab){
  cv::Mat output;
  cv::divide(lab, est, output);
  return -output;
}

template<>
cv::Mat dAbsolute(cv::Mat &est, cv::Mat &lab){
  size_t numRow = est.rows;
  size_t numCol = est.cols;
  cv::Mat output;
  output.create(numRow, numCol, CV_64FC1);
  double *rowPtr_est;
  double *rowPtr_lab;
  double *rowPtr_output;

  for (int i = 0; i < numRow; i++){
    double temp = 0;
    rowPtr_est = est.ptr<double>(i);
    rowPtr_lab = lab.ptr<double>(i);
    rowPtr_output = output.ptr<double>(i);
    for (int j = 0; j < numCol; j++){
      rowPtr_output[j] = rowPtr_est[j] > rowPtr_lab[j] ? 1 : -1;
    }
  }
  return output;
}

template<>
cv::Mat dBinary(cv::Mat &est, cv::Mat &lab){
  size_t numRow = est.rows;
  size_t numCol = est.cols;
  cv::Mat output;
  output.create(numRow, 1, CV_64FC1);
  double *rowPtr_est;
  double *rowPtr_lab;

  for (int i = 0; i < numRow; i++){
    double temp = 0;
    rowPtr_est = est.ptr<double>(i);
    rowPtr_lab = lab.ptr<double>(i);
    for (int j = 0; j < numCol; j++){
      if (rowPtr_est[j] != rowPtr_lab[j]){
        temp = 1;
        break;
      }
    }
    output.at<double>(i, 0) = (temp == 0) ? 0 : 1;
  }
  return output;
}
#endif
