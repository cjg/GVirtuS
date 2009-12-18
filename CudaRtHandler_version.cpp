#include <cuda_runtime_api.h>
#include "CudaRtHandler.h"

#if 0
CUDA_ROUTINE_HANDLER(DriverGetVersion) {
    int *driverVersion = input_buffer->Assign<int>();

    cudaError_t exit_code = cudaDriverGetVersion(driverVersion);

    Buffer *out = new Buffer();
    out->Add(driverVersion);

    return new Result(exit_code, out);
}

CUDA_ROUTINE_HANDLER(RuntimeGetVersion) {
    int *runtimeVersion = input_buffer->Assign<int>();

    cudaError_t exit_code = cudaRuntimeGetVersion(runtimeVersion);

    Buffer *out = new Buffer();
    out->Add(runtimeVersion);

    return new Result(exit_code, out);
}
#endif
