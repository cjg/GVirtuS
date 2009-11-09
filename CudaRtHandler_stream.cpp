#include <cuda_runtime_api.h>
#include "CudaUtil.h"
#include "CudaRtHandler.h"

CUDA_ROUTINE_HANDLER(StreamCreate) {
    cudaStream_t *pStream = input_buffer->Assign<cudaStream_t>();

    cudaError_t exit_code = cudaStreamCreate(pStream);

    Buffer *out = new Buffer();
    out->Add(pStream);

    return new Result(exit_code, out);
}

CUDA_ROUTINE_HANDLER(StreamDestroy) {
    cudaStream_t stream = input_buffer->Get<cudaStream_t>();

    return new Result(cudaStreamDestroy(stream));
}

CUDA_ROUTINE_HANDLER(StreamQuery) {
    cudaStream_t stream = input_buffer->Get<cudaStream_t>();

    return new Result(cudaStreamQuery(stream));
}

CUDA_ROUTINE_HANDLER(StreamSynchronize) {
    cudaStream_t stream = input_buffer->Get<cudaStream_t>();

    return new Result(cudaStreamSynchronize(stream));
}
