#ifndef UPDOWN
#define UPDOWN

#include <vector>
#include <iostream>
#include <chrono>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"

#include "layer.hpp"
#include "model.hpp"
#include "activation.hpp"
#include "loss.hpp"
#include "gd.hpp"
#include "util.hpp"
#include "matrix.hpp"
#include "processData.hpp"

using namespace std;

void upDown(std::vector<RBMlayer> model, dataInBatch &dataSet, dataInBatch &labelSet, int epoch){
	size_t numLayer = model.size();
	size_t numBatch = dataSet.size();
	size_t batchSize = dataSet[0].rows;
	int numClass = labelSet[0].cols;
	size_t dataDim = dataSet[0].cols;
	for (int i = 0; i < epoch; i++){
		for (int k = 0; k < numBatch; k++){
			cv::Mat input = dataSet[k];
			cv::Mat label = labelSet[k];
			vector<cv::Mat> flow;
			flow.resize(numLayer + 1);
			flow[0] = input;
			for (size_t j = 0; j < numLayer - 1; j++){
				cv::Mat w = model[j].g_weight();
				cv::Mat b = model[j].g_bias();
				actFunction activation = model[j].g_ACT();
				flow[j + 1] = activation(flow[j], w, b);
			}
			cv::hconcat(label, flow[numLayer - 1], flow[numLayer - 1]);
			cv::Mat w = model[numLayer - 1].g_weight();
			cv::Mat b = model[numLayer - 1].g_bias();
			cv::Mat bv = model[numLayer - 1].g_biasInFlow();
			actFunction activation = model[numLayer - 1].g_ACT();
			flow[numLayer] = activation(flow[numLayer - 1], w, b);
			cv::Rect roi = cv::Rect(numClass, 0, dataDim, batchSize);
			cv::Rect roiL = cv::Rect(0, 0, numClass, batchSize);
			cv::Mat states1 = label.t() * flow[numLayer];
			cv::Mat states2 = flow[numLayer - 1](roi).t() * flow[numLayer];
			cv::Mat GBw = w.t();
			cv::Mat GBstate = activation(flow[numLayer], GBw, bv);
			cv::Mat GBTop = activation(GBstate, w, b);
			cv::Mat state3 = GBstate(roi).t() * GBTop;
			cv::Mat state4 = GBstate(roiL).t() * GBTop;
			vector<cv::Mat> flowDown;
			flowDown.resize(numLayer + 1);
			flowDown[numLayer - 1] = GBstate(roi);
			for (size_t j = numLayer - 1; j > 0; j--){
				w = model[j - 1].g_weight().t();
				b = model[j - 1].g_biasInFlow();
				activation = model[j - 1].g_ACT();
				flowDown[j - 1] = activation(flow[j], w, b);
			}
			vector<cv::Mat> flowUp;
			flowUp.resize(numLayer + 1);
			for (size_t j = 0; j < numLayer - 1; j++){
				w = model[j + 1].g_weight();
				b = model[j + 1].g_bias();
				activation = model[j + 1].g_ACT();
				flowUp[j + 1] = activation(flowDown[j], w, b);
			}
			double r = 1;
			cv::Mat labelTopUpdate = r * (states1 - state4);
			cv::Mat LastTopUpdate = r * (states2 - state3);
			cv::Mat RBMWupdate;
			cv::vconcat(labelTopUpdate, LastTopUpdate, RBMWupdate);
			model[numLayer - 1].s_weight(model[numLayer - 1].g_weight() + RBMWupdate);
			cv::Mat bStateDifference = flow[numLayer] - GBTop;
			cv::Mat bUpdateAVG;
			cv::reduce(bStateDifference, bUpdateAVG, 0, CV_REDUCE_SUM, -1);
			model[numLayer - 1].s_bias(model[numLayer - 1].g_bias() + r * bUpdateAVG);
			for (int j = 0; j < numLayer - 1; j++){
				model[j].s_weight(model[j].g_weight() + r * flowDown[j].t() * (flowDown[j + 1] - flowUp[j + 1]));
				model[j].s_bias(model[j].g_bias() + r * (flowDown[j + 1] - flowUp[j + 1]));
			}
		}
	}

}
#endif