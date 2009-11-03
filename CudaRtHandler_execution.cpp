#include "cuda_runtime_api.h"
#include "CudaUtil.h"
#include "CudaRtHandler.h"

CUDA_ROUTINE_HANDLER(ConfigureCall) {
    /* cudaError_t cudaConfigureCall(dim3 gridDim, dim3 blockDim,
     * size_t sharedMem, cudaStream_t stream) */
    dim3 gridDim = input_buffer->Get<dim3>();
    dim3 blockDim = input_buffer->Get<dim3>();
    size_t sharedMem = input_buffer->Get<size_t>();
    cudaStream_t stream = input_buffer->Get<cudaStream_t>();

    cudaError_t exit_code = cudaConfigureCall(gridDim, blockDim, sharedMem,
            stream);

    return new Result(exit_code);
}

CUDA_ROUTINE_HANDLER(Launch) {
    /* cudaError_t cudaLaunch(const char * entry) */
    char *handler = input_buffer->AssignString();
    const char *entry = pThis->GetDeviceFunction(handler);
    cudaError_t exit_code = cudaLaunch(entry);

    return new Result(exit_code);
}

CUDA_ROUTINE_HANDLER(SetupArgument) {
    /* cudaError_t cudaSetupArgument(const void *arg, size_t size, size_t offset) */
    size_t offset = input_buffer->BackGet<size_t>();
    size_t size = input_buffer->BackGet<size_t>();
    void *arg = input_buffer->Assign<char>(size);

    // try to translate arg to a device pointer
    if(size == sizeof(void *)) {
        char *handler = CudaUtil::MarshalDevicePointer((void *) * ((int *) arg));
        try {
            void *devPtr = pThis->GetDevicePointer(handler);
            arg = (void *) ((char *) &devPtr);
        } catch(string e) {
        }
        delete handler;
    }

    cudaError_t exit_code = cudaSetupArgument(arg, size, offset);

    return new Result(exit_code);
}