#include "cuda_runtime_api.h"
#include "CudaRtHandler.h"

CUDA_ROUTINE_HANDLER(ThreadExit) {
    return new Result(cudaThreadExit());
}

CUDA_ROUTINE_HANDLER(ThreadSynchronize) {
    return new Result(cudaThreadSynchronize());
}