#ifndef _CONFIGS_H_
#define _CONFIGS_H_

#include <string>
namespace Tn
{
    const int INPUT_CHANNEL = 3;
    const std::string INPUT_IMAGE = "/data0/hhq/project/TensorRT-classify/data/light_6_9.txt";
    const int INPUT_WIDTH = 96;
    const int INPUT_HEIGHT = 96;
    const int DETECT_CLASSES = 3;
    const std::string RT_path = "/data0/hhq/project/TensorRT-classify/data/model/model.rt";

}

#endif