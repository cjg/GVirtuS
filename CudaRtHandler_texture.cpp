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
