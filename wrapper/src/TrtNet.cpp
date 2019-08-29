#include "TrtNet.h"
#include <cassert>
#include <chrono>
#include <cublas_v2.h>
#include <cudnn.h>
#include <iostream>
#include <sstream>
#include <string.h>
#include <time.h>
#include <unordered_map>

using namespace nvinfer1;
using namespace nvcaffeparser1;
using namespace plugin;

static Tn::Logger gLogger;

inline void* safeCudaMalloc(size_t memSize)
{
    void* deviceMem;
    CUDA_CHECK(cudaMalloc(&deviceMem, memSize));
    if (deviceMem == nullptr)
    {
        std::cerr << "Out of memory" << std::endl;
        exit(1);
    }
    return deviceMem;
}


inline int64_t volume(const nvinfer1::Dims& d)
{
    return std::accumulate(d.d, d.d + d.nbDims, 1, std::multiplies<int64_t>());
}

inline unsigned int getElementSize(nvinfer1::DataType t)
{
    switch (t)
    {
        case nvinfer1::DataType::kINT32: return 4;
        case nvinfer1::DataType::kFLOAT: return 4;
        case nvinfer1::DataType::kHALF: return 2;
        case nvinfer1::DataType::kINT8: return 1;
    }
    throw std::runtime_error("Invalid DataType.");
    return 0;
}

namespace Tn
{
    //load RT model
    void trtNet::Init(std::string RT_load_path)
    {
        std::string model_rt_file = RT_load_path;
        std::ifstream in_file(model_rt_file, std::ios::in | std::ios::binary);
        if (!in_file.is_open())
        {
            std::cout<<"fail to open rt model file !"<<std::endl;
            exit(-1);
        }
        std::streampos begin, end;
        begin = in_file.tellg();
        in_file.seekg(0, std::ios::end);
        end = in_file.tellg();
        std::size_t size = end - begin;
        std::cout<<model_rt_file.c_str()<<" model file is used, model file size: "<<size<<" bytes"<<std::endl;
        in_file.seekg(0, std::ios::beg);
        std::unique_ptr<unsigned char[]> engine_data(new unsigned char[size]);
        in_file.read((char*)engine_data.get(), size);
        in_file.close();

        mTrtRunTime = createInferRuntime(gLogger);
        assert(mTrtRunTime != nullptr);
        mTrtEngine= mTrtRunTime->deserializeCudaEngine((const void*)engine_data.get(), size,  NULL);
        assert(mTrtEngine != nullptr);
        // Deserialize the engine.
//        trtModelStream->destroy();

        InitEngine();
    }

    void trtNet::InitEngine()
    {
        const int maxBatchSize = 1;
        mTrtContext = mTrtEngine->createExecutionContext();
        assert(mTrtContext != nullptr);
        mTrtContext->setProfiler(&mTrtProfiler);

        // Input and output buffer pointers that we pass to the engine - the engine requires exactly IEngine::getNbBindings()
        int nbBindings = mTrtEngine->getNbBindings();

        mTrtCudaBuffer.resize(nbBindings);
        mTrtBindBufferSize.resize(nbBindings);
        for (int i = 0; i < nbBindings; ++i)
        {
            Dims dims = mTrtEngine->getBindingDimensions(i);
            DataType dtype = mTrtEngine->getBindingDataType(i);
            int64_t totalSize = volume(dims) * maxBatchSize * getElementSize(dtype);
            mTrtBindBufferSize[i] = totalSize;
            mTrtCudaBuffer[i] = safeCudaMalloc(totalSize);
            if(mTrtEngine->bindingIsInput(i))
                mTrtInputCount++;
        }

        CUDA_CHECK(cudaStreamCreate(&mTrtCudaStream));
    }

    void trtNet::doInference(const void* inputData, void* outputData)
    {
        static const int batchSize = 1;
        assert(mTrtInputCount == 1);

        // DMA the input to the GPU,  execute the batch asynchronously, and DMA it back:
        int inputIndex = 0;
        CUDA_CHECK(cudaMemcpyAsync(mTrtCudaBuffer[inputIndex], inputData, mTrtBindBufferSize[inputIndex], cudaMemcpyHostToDevice, mTrtCudaStream));
        auto t_start = std::chrono::high_resolution_clock::now();
        mTrtContext->execute(batchSize, &mTrtCudaBuffer[inputIndex]);
//	    mTrtContext->enqueue(batchSize, &mTrtCudaBuffer[inputIndex],  mTrtCudaStream, nullptr);
        auto t_end = std::chrono::high_resolution_clock::now();
        float total = std::chrono::duration<float, std::milli>(t_end - t_start).count();

//        std::cout << "Time taken for inference is " << total << " ms." << std::endl;

        for (size_t bindingIdx = mTrtInputCount; bindingIdx < mTrtBindBufferSize.size(); ++bindingIdx)
        {
            auto size = mTrtBindBufferSize[bindingIdx];
            CUDA_CHECK(cudaMemcpyAsync(outputData, mTrtCudaBuffer[bindingIdx], size, cudaMemcpyDeviceToHost, mTrtCudaStream));
            outputData = (char *)outputData + size;
            //CUDA_CHECK(cudaMemcpyAsync(outputData, mTrtCudaBuffer[bindingIdx], size, cudaMemcpyDeviceToHost, mTrtCudaStream));
            //outputData +=size;
        }

        mTrtIterationTime ++ ;
    }
}
