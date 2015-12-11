#ifndef LAYER
#define LAYER
#include <vector>
#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "util.hpp"
#include "activation.hpp"

class layer{
protected:
	  size_t index;
	  cv::Mat positiveConnection;
	  cv::Mat negativeConnection;
	  cv::Mat pIFlow;
	  cv::Mat pOFlow;
	  cv::Mat nOFlow;
	  cv::Mat nIFlow;
public:
  	layer(){}
  	cv::Mat emitP() const {return pOFlow;}
  	cv::Mat emitN() const {return nOFlow;}
  	void absorbP(cv::Mat flow){ pIFlow = flow;}
  	void absorbN(cv::Mat flow){ nIFlow = flow;}
};

class RBMlayer: public layer{
	cv::Mat bias;
  cv::Mat biasInFlow;
  cv::Mat weight;
  cv::Mat biasLabel;
  cv::Mat labelWeight;
	size_t size;
	size_t sizeInFlow;
  actFunction ACT;
  act_dFunction dACT;
  f fACT;
  cv::Mat storage;
  cv::Mat storage2;

public:
    RBMlayer(){}
	  RBMlayer(size_t s): size(s){}
    RBMlayer(cv::Mat& w): weight(w), size(w.cols), sizeInFlow(w.rows){}
  	RBMlayer(size_t s, size_t n): size(s), sizeInFlow(n){
  		weight = cv::Mat::zeros(sizeInFlow, size, CV_64FC1);
  	}
  	cv::Mat g_bias() const {return bias;}
    cv::Mat g_biasInFlow() const {return biasInFlow;}
  	cv::Mat g_weight() const {return weight;}
  	size_t g_index() const {return index;}
  	size_t g_size() const {return size;}
    cv::Mat g_labelWeight() const {return labelWeight;}
    cv::Mat g_biasLabel() const {return biasLabel;}
    actFunction g_ACT() const {return ACT;}
    act_dFunction g_dACT() const {return dACT;}
    f g_ACTF() const {return fACT;}
  	void s_size(size_t i) {size = i;}
  	void s_bias(cv::Mat b) {bias = b;}
  	void s_index(size_t i) {index = i;}
  	void s_weight(cv::Mat w) {weight = w;}
    void s_biasInFlow(cv::Mat b) {biasInFlow = b;}
    void s_labelWeight(cv::Mat w) {labelWeight = w;}
    void s_biasLabel(cv::Mat b) {biasLabel = b;}
    void s_ACT(actFunction i) {ACT = i;}
    void s_dACT(act_dFunction i) {dACT = i;}
    void s_ACT(ActivationType i);
    void s_storage(cv::Mat s) {storage = s;}
    cv::Mat g_storage() {return storage;}
    void s_storage2(cv::Mat s) {storage2 = s;}
    cv::Mat g_storage2() {return storage2;}
    dataInBatch g_activation(dataInBatch& input);
};

dataInBatch RBMlayer::g_activation(dataInBatch& input){
  dataInBatch output;
  output.resize(input.size());
  for (int i = 0; i < input.size(); i++){
    cv::Mat batchIn = input[i];
    cv::Mat batchOut = ACT(batchIn, weight, bias);
    output[i] = batchOut;
  }
  return output;
}

void RBMlayer::s_ACT(ActivationType i){
  switch (i)
  {
    case sigmoid_t: 
      ACT = sigmoidACT; 
      dACT = dsigmoidACT;
      fACT = sigmoidF;
      break;
    case tanh_t: 
      ACT = tanhACT; 
      dACT = dtanhACT;
      fACT = tanhF;
      break;
    case relu_t: 
      ACT = reluACT; 
      dACT = dreluACT;
      fACT = reluF;
      break;
    case leaky_relu_t: 
      ACT = leaky_reluACT; 
      dACT = dleaky_reluACT;
      fACT = leaky_reluF;
      break;
    case softmax_t: 
      ACT = softmaxACT; 
      dACT = dsoftmaxACT;
      fACT = softmaxF;
      break;
    default: 
      ACT = identityACT;
      dACT = didentityACT;
      fACT = identityF;
  }
}

#endif
