#ifndef UTIL
#define UTIL
#include <vector>
#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"

typedef cv::Mat (*actFunction) (cv::Mat&, cv::Mat&, cv::Mat&);
typedef cv::Mat (*actFunction_multi) (cv::Mat, cv::Mat);
typedef cv::Mat (*act_dFunction) (cv::Mat&);
typedef cv::Mat (*f) (cv::Mat&);
typedef cv::Mat (*kernel) (cv::Mat&, cv::Mat&, std::vector<double> parameter);

typedef std::pair<size_t, size_t> imageIndex;
typedef std::vector<cv::Mat> dataInBatch;
typedef std::vector<cv::Mat> dataCollection;
typedef std::vector<cv::Mat> flowOverLayer;
typedef cv::Mat data;
typedef std::pair<cv::Mat, double> classificationError;
typedef std::vector<int> indexLabel;

struct dimension{
	dimension(size_t h, size_t w, size_t d): width(w), height(h), depth(d) {}	
	size_t width;
	size_t height;
	size_t depth;
	size_t size(){return width * height * depth;}
};

struct imageBoard{
	std::vector<cv::Mat> imageSet;
	std::vector<cv::Mat> referenceSet;
	dimension boardStructure;
	dimension imageStructure;
	imageBoard(dimension bs, dimension is): boardStructure(bs), imageStructure(is) {
		imageSet.resize(boardStructure.size());
	}
	void rebuild(dimension s) {boardStructure = s;}
	void s_imageSet(std::vector<cv::Mat> i) {imageSet = i;}
	void s_referenceSet(std::vector<cv::Mat> i) {referenceSet = i;}
};

class errorReport : public std::exception{
	std::string msg;
public:
	explicit errorReport(const std::string& m): msg(m) {}
};

enum ActivationType{
	sigmoid_t, tanh_t, relu_t, leaky_relu_t, softmax_t, identity 
};

enum GDType{
	SGD, AdaGrad 
};

enum LossType{
	MSE, _0_1, Absolute, CrossEntropy
};

enum RegType{
	L1, L2
};

enum annealType{
	exponential, divideBy, step
};

#endif
