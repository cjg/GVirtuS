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
    extern void * __cudaRegisterVar(void **fatCubinHandle, char *hostVar,
            char *deviceAddress, const char *deviceName, int ext, int size,
            int constant, int global);
    extern void __cudaRegisterShared(void **fatCubinHandle, void **devicePtr);
    extern void __cudaRegisterTexture(void **fatCubinHandle,
            const textureReference *hostVar, void **deviceAddress, char *deviceName,
            int dim, int norm, int ext);
};

CUDA_ROUTINE_HANDLER(RegisterFatBinary) {
    char * handler = input_buffer->AssignString();
    __cudaFatCudaBinary * fatBin =
            CudaUtil::UnmarshalFatCudaBinary(input_buffer);
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
    char * handler = input_buffer->AssignString();
    void **fatCubinHandle = pThis->GetFatBinary(handler);
    const char *hostFun = strdup(input_buffer->AssignString());
    char *deviceFun = strdup(input_buffer->AssignString());
    const char *deviceName = strdup(input_buffer->AssignString());
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
    CudaUtil::CudaVar *var;
    vector<CudaUtil::CudaVar *> *vars = new vector<CudaUtil::CudaVar *>();
    while (true) {
        try {
            var = input_buffer->Assign<CudaUtil::CudaVar > ();
            vars->push_back(var);
        } catch (string e) {
            break;
        }
    }

    void **fatCubinHandle = pThis->GetFatBinary(var->fatCubinHandle);

    for (vector<CudaUtil::CudaVar *>::iterator it = vars->begin();
            it != vars->end(); it++) {
        var = *it;
        __cudaRegisterVar(fatCubinHandle,
                (char *) CudaUtil::UnmarshalPointer(var->hostVar),
                var->deviceAddress, var->deviceName, var->ext, var->size,
                var->constant, var->global);
    }
    for (vector<CudaUtil::CudaVar *>::iterator it = vars->begin();
            it != vars->end(); it++) {
        var = *it;
        void *devPtr;
        if(cudaGetSymbolAddress(&devPtr, var->deviceName) != cudaSuccess) {
            cerr << "Error while registering Var " << var->deviceName << endl;
            continue;
        }
        pThis->RegisterVar(var->hostVar, var->deviceName);
        pThis->RegisterDevicePointer(var->hostVar, devPtr, var->size);
    }

    return new Result(cudaSuccess);
}

CUDA_ROUTINE_HANDLER(RegisterShared) {
    char * handler = input_buffer->AssignString();
    void **fatCubinHandle = pThis->GetFatBinary(handler);
    char *devPtr = strdup(input_buffer->AssignString());
    __cudaRegisterShared(fatCubinHandle, (void **) devPtr);
    cout << "Registerd Shared: " << (char *) devPtr << " for " << fatCubinHandle << endl;
    return new Result(cudaSuccess);
}

CUDA_ROUTINE_HANDLER(RegisterTexture) {
    char * handler = input_buffer->AssignString();
    void **fatCubinHandle = pThis->GetFatBinary(handler);
    handler = input_buffer->AssignString();
    textureReference *hostVar = new textureReference;
    memmove(hostVar, input_buffer->Assign<textureReference > (),
            sizeof (textureReference));
    void **deviceAddress = (void **) input_buffer->AssignAll<char>();
    char *deviceName = strdup(input_buffer->AssignString());
    int dim = input_buffer->Get<int>();
    int norm = input_buffer->Get<int>();
    int ext = input_buffer->Get<int>();

    __cudaRegisterTexture(fatCubinHandle, hostVar, deviceAddress, deviceName,
            dim, norm, ext);

    pThis->RegisterTexture(handler, hostVar);

    return new Result(cudaSuccess);
}
