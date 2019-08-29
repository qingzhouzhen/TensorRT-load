#ifndef __TRT_NET_H_
#define __TRT_NET_H_

#include <string>
#include <vector>
#include <algorithm>
#include <fstream>
#include "NvInferPlugin.h"
#include "NvCaffeParser.h"
#include "Utils.h"
#include <memory>

namespace Tn
{
    enum class RUN_MODE
    {
        FLOAT32 = 0,
        FLOAT16 = 1,    
        INT8 = 2
    };

    class trtNet 
    {
        public:

            trtNet()
            :mTrtContext(nullptr),mTrtEngine(nullptr),mTrtRunTime(nullptr),mTrtRunMode(RUN_MODE::FLOAT32),mTrtInputCount(0),mTrtIterationTime(0)
            {};

            //load RT model
            void Init(std::string RT_load_path);

            ~trtNet()
            {
                // Release the stream and the buffers
                cudaStreamSynchronize(mTrtCudaStream);
                cudaStreamDestroy(mTrtCudaStream);
                for(auto& item : mTrtCudaBuffer)
                    cudaFree(item);

                if(!mTrtRunTime)
                    mTrtRunTime->destroy();
                if(!mTrtContext)
                    mTrtContext->destroy();
                if(!mTrtEngine)
                    mTrtEngine->destroy();
            };

        void doInference(const void* inputData, void* outputData);

        inline size_t getInputSize() {
            return std::accumulate(mTrtBindBufferSize.begin(), mTrtBindBufferSize.begin() + mTrtInputCount,0);
        };

        inline size_t getOutputSize() {
            return std::accumulate(mTrtBindBufferSize.begin() + mTrtInputCount, mTrtBindBufferSize.end(),0);
        };

        void printTime()
        {
            mTrtProfiler.printLayerTimes(mTrtIterationTime);
        }

        private:

            void InitEngine();

            nvinfer1::IExecutionContext* mTrtContext;
            nvinfer1::ICudaEngine* mTrtEngine;
            nvinfer1::IRuntime* mTrtRunTime;
            cudaStream_t mTrtCudaStream;
            Profiler mTrtProfiler;
            RUN_MODE mTrtRunMode;

            std::vector<void*> mTrtCudaBuffer;
            std::vector<int64_t> mTrtBindBufferSize;
            int mTrtInputCount;
            int mTrtIterationTime;
    };
}

#endif //__TRT_NET_H_
