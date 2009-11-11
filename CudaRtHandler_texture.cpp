#include <cuda_runtime_api.h>
#include "CudaUtil.h"
#include "CudaRtHandler.h"

CUDA_ROUTINE_HANDLER(BindTextureToArray) {
    char * texrefHandler = input_buffer->AssignString();
    textureReference *guestTexref = input_buffer->Assign<textureReference>();
    char * arrayHandler =
        input_buffer->Assign<char>(CudaUtil::MarshaledDevicePointerSize);
    cudaChannelFormatDesc *desc = input_buffer->Assign<cudaChannelFormatDesc>();

    textureReference *texref = pThis->GetTexture(texrefHandler);
    memmove(texref, guestTexref, sizeof(textureReference));

    cudaArray *array = (cudaArray *) pThis->GetDevicePointer(arrayHandler);

    cudaError_t exit_code = cudaBindTextureToArray(texref, array, desc);

    return new Result(exit_code);
}

CUDA_ROUTINE_HANDLER(GetChannelDesc) {
    cudaChannelFormatDesc *guestDesc =
            input_buffer->Assign<cudaChannelFormatDesc>();
    char * arrayHandler =
        input_buffer->Assign<char>(CudaUtil::MarshaledDevicePointerSize);
    cudaArray *array = (cudaArray *) pThis->GetDevicePointer(arrayHandler);
    Buffer *out = new Buffer();
    cudaChannelFormatDesc *desc = out->Delegate<cudaChannelFormatDesc>();
    memmove(desc, guestDesc, sizeof(cudaChannelFormatDesc));

    cudaError_t exit_code = cudaGetChannelDesc(desc, array);

    return new Result(exit_code, out);
}

CUDA_ROUTINE_HANDLER(UnbindTexture) {
    char * texrefHandler = input_buffer->AssignString();
    textureReference *texref = pThis->GetTexture(texrefHandler);

    cudaError_t exit_code = cudaUnbindTexture(texref);

    return new Result(exit_code);
}
