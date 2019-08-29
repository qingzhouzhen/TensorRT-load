#include <string>
#include <sstream>
#include <algorithm>
#include <memory>
#include <opencv2/opencv.hpp>
#include "TrtNet.h"
#include "argsParser.h"
#include "configs.h"
#include <chrono>
#include <iomanip>
#include <sstream>

using namespace std;
using namespace argsParser;
using namespace Tn;

std::vector<float> prepareImage(const string& fileName,int* width = nullptr,int* height = nullptr)
{
    using namespace cv;

    Mat img = imread(fileName);
    if(img.data== nullptr)
    {
        std::cout << "can not open image :" << fileName  << std::endl;
        return {};
    }

    if(width)
        *width = img.cols;

    if(height)
        *height = img.rows;

    int c = parser::getIntValue("C");
    int h = parser::getIntValue("H");
    int w = parser::getIntValue("W");

    auto scaleSize = cv::Size(96,96);

    cv::Mat rgb ;
    cv::cvtColor(img, rgb, CV_BGR2RGB);
    cv::Mat resized;
    cv::resize(rgb, resized,scaleSize,0,0,INTER_CUBIC);
    cv::Mat img_float;
    if (INPUT_CHANNEL == 3)
        resized.convertTo(img_float, CV_32FC3, 1);
    else
        resized.convertTo(img_float, CV_32FC1 ,1);

    //HWC TO CHW
    cv::Mat input_channels[INPUT_CHANNEL];
    cv::split(img_float, input_channels);

    std::vector<float> result(h*w*c);
    auto data = result.data();
    int channelLength = h * w;
    for (int i = 0; i < c; ++i) {
        memcpy(data,input_channels[i].data,channelLength*sizeof(float));
        data += channelLength;
    }

    return result;
}

int main( int argc, char* argv[] ) {
    parser::ADD_ARG_STRING("input", Desc("input image file"), DefaultValue(INPUT_IMAGE), ValueDesc("file"));
    parser::ADD_ARG_STRING("trt_path", Desc("tensorrt model path"), DefaultValue(RT_path));
    parser::ADD_ARG_INT("C", Desc("channel"), DefaultValue(to_string(INPUT_CHANNEL)));
    parser::ADD_ARG_INT("H", Desc("height"), DefaultValue(to_string(INPUT_HEIGHT)));
    parser::ADD_ARG_INT("W", Desc("width"), DefaultValue(to_string(INPUT_WIDTH)));
    parser::ADD_ARG_INT("class", Desc("num of classes"), DefaultValue(to_string(DETECT_CLASSES)));

    parser::parseArgs(argc, argv);

    string trt_path = parser::getStringValue("trt_path");
    string FileName = parser::getStringValue("input");

    trtNet net;
    net.Init(trt_path);

    ifstream txt_file = ifstream(FileName, ios::in);
    if(txt_file)
    {
        std::cout<<"open test.txt!"<<endl;
        string line;
        int count = 0;
        while (getline(txt_file, line))
        {
            if(!line.empty())
            {
                string inputFileName;
                std::stringstream aa;
                aa<<line;
                aa>>inputFileName;
                int width,height;
                std::cout<<"   iamge name:"<<inputFileName<<std::endl;
                auto inputData = prepareImage(inputFileName, &width, &height);
                count++;

                int outputCount = net.getOutputSize()/ sizeof(float);
                unique_ptr<float[]> outputData(new float[outputCount]);
                net.doInference(inputData.data(), outputData.get());
                int index = 0;
                float max = outputData[0];
                for(int i=1; i<3; i++)
                {
                    if(outputData[i] > max)
                    {
                        index = i;
                        max = outputData[i];
                    }
                }
                std::cout<<index<<", "<<outputData[0]<<", "<<outputData[1]<<", "<<outputData[2]<<endl;
            }
        }
    }
}
