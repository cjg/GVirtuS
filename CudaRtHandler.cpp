/* 
 * File:   CudaRtHandler.cpp
 * Author: cjg
 * 
 * Created on October 10, 2009, 10:51 PM
 */

#include <host_defines.h>
#include <builtin_types.h>
#include <driver_types.h>
#include <GL/gl.h>
#include <cuda_runtime_api.h>
#include "CudaRtHandler.h"

using namespace std;

CudaRtHandler::CudaRtHandler(istream& input, ostream& output)
    : mInput(input), mOutput(output) {
}

CudaRtHandler::~CudaRtHandler() {
}

void CudaRtHandler::GetDeviceCount() {
    /* cudaError_t cudaGetDeviceCount(int *count) */
    char *iobuffer = new char[sizeof(int)];
    size_t iobuffer_size = sizeof(int);
    
    mInput.read(iobuffer, sizeof(int));
    int *count = (int *) iobuffer;

    cudaError_t result = cudaGetDeviceCount(count);

    mOutput.write((char *) &result, sizeof(cudaError_t));
    if(result == cudaSuccess) {
        mOutput.write((char *) &iobuffer_size, sizeof(size_t));
        mOutput.write(iobuffer, sizeof(int));
    }

    delete[] iobuffer;
}

void CudaRtHandler::GetDeviceProperties() {
    /* cudaError_t cudaGetDeviceCount(struct cudaDeviceProp *prop,
       int device) */
    size_t iobuffer_size = sizeof(struct cudaDeviceProp) + sizeof(int);
    char *iobuffer = new char[iobuffer_size];

    mInput.read(iobuffer, iobuffer_size);

    struct cudaDeviceProp *prop = (struct cudaDeviceProp *) iobuffer;
    int *pDevice = (int *) (iobuffer + sizeof(struct cudaDeviceProp));

    cudaError_t result = cudaGetDeviceProperties(prop, *pDevice);

    mOutput.write((char *) &result, sizeof(cudaError_t));
    if(result == cudaSuccess) {
        iobuffer_size = sizeof(struct cudaDeviceProp);
        mOutput.write((char *) &iobuffer_size, sizeof(size_t));
        mOutput.write(iobuffer, sizeof(struct cudaDeviceProp));
    }

    delete[] iobuffer;
}