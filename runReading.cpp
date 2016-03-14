#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <vector>
#include "readAvletters.hpp"
#include "readXRMB.hpp"
#include "readAvLetters.hpp"

int main()
{	
	cv::Mat output = imread_avletters_mfcc("data/avletters/Audio/A1_Anya.mfcc");
	return 0;
}
