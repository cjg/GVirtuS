#include <string>
#include <iostream>
#include <cstdio>
#include <vector>
#include "cuda_runtime_api.h"
#include "CudaUtil.h"
#include "CudaRtHandler.h"

using namespace std;

extern "C" {
    extern void** __cudaRegisterFatBinary(void *fatCubin);
    extern void __cudaUnregisterFatBinary(void **fatCubinHandle);
    extern void __cudaRegisterFunction(void **fatCubinHandle,
            const char *hostFun, char *deviceFun, const char *deviceName,
            int thread_limit, uint3 *tid, uint3 *bid, dim3 *bDim, dim3 *gDim,
            int *wSize);
    extern void __cudaRegisterVar(void **fatCubinHandle, char *hostVar,
            char *deviceAddress, const char *deviceName, int ext, int size,
            int constant, int global);
};

CUDA_ROUTINE_HANDLER(RegisterFatBinary) {
    char * handler = input_buffer->AssignString();
    __cudaFatCudaBinary * fatBin = CudaUtil::UnmarshalFatCudaBinary(input_buffer);
    void **fatCubinHandler = __cudaRegisterFatBinary((void *) fatBin);
    pThis->RegisterFatBinary(handler, fatCubinHandler);
    return new Result(cudaSuccess);
}

CUDA_ROUTINE_HANDLER(UnregisterFatBinary) {
    char * handler = input_buffer->AssignString();
    void **fatCubinHandle = pThis->GetFatBinary(handler);

    __cudaUnregisterFatBinary(fatCubinHandle);

    pThis->UnregisterFatBinary(handler);

    return new Result(cudaSuccess);
}

CUDA_ROUTINE_HANDLER(RegisterFunction) {
    /* void __cudaRegisterFunction(void **fatCubinHandle, 
     * const char *hostFun,
     * char *deviceFun,
     * const char *deviceName,
     * int thread_limit,
     * uint3 *tid,
     * uint3 *bid,
     * dim3 *bDim,
     * dim3 *gDim,
     * int *wSize) */
    char * handler = input_buffer->AssignString();
    void **fatCubinHandle = pThis->GetFatBinary(handler);
    const char *hostFun = input_buffer->AssignString();
    char *deviceFun = input_buffer->AssignString();
    const char *deviceName = input_buffer->AssignString();
    int thread_limit = input_buffer->Get<int>();
    uint3 *tid = input_buffer->Assign<uint3 > ();
    uint3 *bid = input_buffer->Assign<uint3 > ();
    dim3 *bDim = input_buffer->Assign<dim3 > ();
    dim3 *gDim = input_buffer->Assign<dim3 > ();
    int *wSize = input_buffer->Assign<int>();


    __cudaRegisterFunction(fatCubinHandle, hostFun, deviceFun, deviceName,
            thread_limit, tid, bid, bDim, gDim, wSize);

    pThis->RegisterDeviceFunction(hostFun, deviceFun);

    Buffer * output_buffer = new Buffer();
    output_buffer->AddString(deviceFun);
    output_buffer->Add(tid);
    output_buffer->Add(bid);
    output_buffer->Add(bDim);
    output_buffer->Add(gDim);
    output_buffer->Add(wSize);

    return new Result(cudaSuccess, output_buffer);
}

CUDA_ROUTINE_HANDLER(RegisterVar) {
    char * handler = input_buffer->AssignString();
    void **fatCubinHandle = pThis->GetFatBinary(handler);
    handler = input_buffer->AssignString();
    char *hostVar;
    char *deviceAddress = input_buffer->AssignString();
    char *deviceName = input_buffer->AssignString();
    int ext = input_buffer->Get<int>();
    int size = input_buffer->Get<int>();
    int constant = input_buffer->Get<int>();
    int global = input_buffer->Get<int>();

    // FIXME: this shouldn't be lost as it is
    hostVar = (char *) malloc(size);

    __cudaRegisterVar(fatCubinHandle, hostVar, deviceAddress, deviceName, ext,
            size, constant, global);

    return new Result(cudaSuccess);
}

#if 0
CUDA_ROUTINE_HANDLER(RegisterShared) {
    char * handler = input_buffer->AssignString();
    void **fatCubinHandle = pThis->GetFatBinary(handler);
    char *dev_ptr_handler =
            input_buffer->Assign<char>(CudaUtil::MarshaledDevicePointerSize);
    void *devPtr = pThis->GetDevicePointer(dev_ptr_handler);
}
#endif