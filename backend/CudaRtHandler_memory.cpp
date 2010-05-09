#include <cuda_runtime_api.h>
#include "CudaUtil.h"
#include "CudaRtHandler.h"

CUDA_ROUTINE_HANDLER(Free) {
    void *devPtr = input_buffer->GetFromMarshal<void *>();
    
    cudaError_t exit_code = cudaFree(devPtr);

    return new Result(exit_code);
}

CUDA_ROUTINE_HANDLER(FreeArray) {
    cudaArray *arrayPtr = input_buffer->GetFromMarshal<cudaArray *>();

    cudaError_t exit_code = cudaFreeArray(arrayPtr);

    return new Result(exit_code);
}

CUDA_ROUTINE_HANDLER(GetSymbolAddress) {
    void *devPtr;
    const char *symbol = pThis->GetSymbol(input_buffer);
    
    cudaError_t exit_code = cudaGetSymbolAddress(&devPtr, symbol);

    Buffer *out = new Buffer();
    if(exit_code == cudaSuccess)
        out->AddMarshal(devPtr);

    return new Result(exit_code, out);
}

CUDA_ROUTINE_HANDLER(GetSymbolSize) {
    Buffer *out = new Buffer();
    size_t *size = out->Delegate<size_t>();
    *size = *(input_buffer->Assign<size_t>());
    const char *symbol = pThis->GetSymbol(input_buffer);

    cudaError_t exit_code = cudaGetSymbolSize(size, symbol);

    return new Result(exit_code, out);
}

CUDA_ROUTINE_HANDLER(Malloc) {
    void *devPtr = NULL;
    size_t size = input_buffer->Get<size_t > ();

    cudaError_t exit_code = cudaMalloc(&devPtr, size);

    cout << "Allocated DevicePointer " << devPtr << " with a size of " << size
            << endl;

    Buffer *out = new Buffer();
    out->AddMarshal(devPtr);

    return new Result(exit_code, out);
}

CUDA_ROUTINE_HANDLER(MallocArray) {
    cudaArray *arrayPtr = NULL;
    cudaChannelFormatDesc *desc =
            input_buffer->Assign<cudaChannelFormatDesc > ();
    size_t width = input_buffer->Get<size_t > ();
    size_t height = input_buffer->Get<size_t > ();

    cudaError_t exit_code = cudaMallocArray(&arrayPtr, desc, width, height);

    Buffer *out = new Buffer();
    out->AddMarshal(arrayPtr);

    return new Result(exit_code, out);
}

CUDA_ROUTINE_HANDLER(MallocPitch) {
    void *devPtr = NULL;
    size_t *pitch = input_buffer->Assign<size_t>();
    size_t width = input_buffer->Get<size_t>();
    size_t height = input_buffer->Get<size_t>();

    cudaError_t exit_code = cudaMallocPitch(&devPtr, pitch, width, height);

    cout << "Allocated DevicePointer " << devPtr << " with a size of " << width * height
            << endl;

    Buffer *out = new Buffer();
    out->AddMarshal(devPtr);
    out->Add(pitch);

    return new Result(exit_code, out);
}


CUDA_ROUTINE_HANDLER(Memcpy) {
    /* cudaError_t cudaError_t cudaMemcpy(void *dst, const void *src,
        size_t count, cudaMemcpyKind kind) */
    void *dst = NULL;
    void *src = NULL;


    cudaMemcpyKind kind = input_buffer->BackGet<cudaMemcpyKind > ();
    size_t count = input_buffer->BackGet<size_t > ();
    cudaError_t exit_code;
    Result * result = NULL;
    Buffer *out;

    switch (kind) {
        case cudaMemcpyHostToHost:
            // This should never happen
            result = NULL;
            break;
        case cudaMemcpyHostToDevice:
            if(pThis->HasSharedMemory()) {
                dst = input_buffer->GetFromMarshal<void *>();
                src = pThis->GetSharedMemory();
                exit_code = cudaMemcpy(dst, src, count, kind);
                result = new Result(exit_code);
            } else {
                dst = input_buffer->GetFromMarshal<void *>();
                src = input_buffer->AssignAll<char>();
                exit_code = cudaMemcpy(dst, src, count, kind);
                result = new Result(exit_code);
            }
            break;
        case cudaMemcpyDeviceToHost:
            if(pThis->HasSharedMemory()) {
                dst = pThis->GetSharedMemory();
                src = input_buffer->GetFromMarshal<void *>();
                exit_code = cudaMemcpy(dst, src, count, kind);
                result = new Result(exit_code);
            } else {
                // FIXME: use buffer delegate
                dst = new char[count];
                /* skipping a char for fake host pointer */
                input_buffer->Assign<char>();
                src = input_buffer->GetFromMarshal<void *>();
                exit_code = cudaMemcpy(dst, src, count, kind);
                out = new Buffer();
                out->Add<char>((char *) dst, count);
                delete[] (char *) dst;
                result = new Result(exit_code, out);
            }
            break;
        case cudaMemcpyDeviceToDevice:
            dst = input_buffer->GetFromMarshal<void *>();
            src = input_buffer->GetFromMarshal<void *>();
            exit_code = cudaMemcpy(dst, src, count, kind);
            result = new Result(exit_code);
            break;
    }
    return result;
}

CUDA_ROUTINE_HANDLER(MemcpyAsync) {
    void *dst = NULL;
    void *src = NULL;

    cudaStream_t stream = input_buffer->BackGet<cudaStream_t > ();
    cudaMemcpyKind kind = input_buffer->BackGet<cudaMemcpyKind > ();
    size_t count = input_buffer->BackGet<size_t > ();
    cudaError_t exit_code;
    Buffer * out;
    Result * result = NULL;

    switch (kind) {
        case cudaMemcpyHostToHost:
            result = new Result(cudaSuccess);
            break;
        case cudaMemcpyHostToDevice:
            dst = input_buffer->GetFromMarshal<void *>();
            src = input_buffer->AssignAll<char>();
            exit_code = cudaMemcpy(dst, src, count, kind);
            result = new Result(exit_code);
            break;
        case cudaMemcpyDeviceToHost:
            // FIXME: use buffer delegate
            dst = new char[count];
            /* skipping a char for fake host pointer */
            input_buffer->Assign<char>();
            src = input_buffer->GetFromMarshal<void *>();
            exit_code = cudaMemcpy(dst, src, count, kind);
            out = new Buffer();
            out->Add<char>((char *) dst, count);
            delete[] (char *) dst;
            result = new Result(exit_code, out);
            break;
        case cudaMemcpyDeviceToDevice:
            dst = input_buffer->GetFromMarshal<void *>();
            src = input_buffer->GetFromMarshal<void *>();
            exit_code = cudaMemcpyAsync(dst, src, count, kind, stream);
            result = new Result(exit_code);
            break;
    }
    return result;
}

CUDA_ROUTINE_HANDLER(MemcpyFromSymbol) {
    void *dst = input_buffer->GetFromMarshal<void *>();
    char *handler = input_buffer->AssignString();
    char *symbol = input_buffer->AssignString();
    symbol = (char *) CudaUtil::UnmarshalPointer(handler);
    size_t count = input_buffer->Get<size_t > ();
    size_t offset = input_buffer->Get<size_t > ();
    cudaMemcpyKind kind = input_buffer->Get<cudaMemcpyKind > ();

    cudaError_t exit_code;
    Result * result = NULL;
    Buffer * out = NULL;

    switch (kind) {
        case cudaMemcpyHostToHost:
            // This should never happen
            result = new Result(cudaErrorInvalidMemcpyDirection);
            break;
        case cudaMemcpyHostToDevice:
            // This should never happen
            result = new Result(cudaErrorInvalidMemcpyDirection);
            break;
        case cudaMemcpyDeviceToHost:
            out = new Buffer(count);
            dst = out->Delegate<char>(count);
            exit_code = cudaMemcpyFromSymbol(dst, symbol, count, offset, kind);
            result = new Result(exit_code, out);
            break;
        case cudaMemcpyDeviceToDevice:
            exit_code = cudaMemcpyFromSymbol(dst, symbol, count, offset, kind);
            result = new Result(exit_code);
            break;
    }
    return result;
}

CUDA_ROUTINE_HANDLER(MemcpyToArray) {
    void *src = NULL;

    cudaArray *dst = input_buffer->GetFromMarshal<cudaArray *>();
    size_t wOffset = input_buffer->Get<size_t > ();
    size_t hOffset = input_buffer->Get<size_t > ();
    cudaMemcpyKind kind = input_buffer->BackGet<cudaMemcpyKind > ();
    size_t count = input_buffer->BackGet<size_t > ();
    cudaError_t exit_code;
    Result * result = NULL;

    switch (kind) {
        case cudaMemcpyHostToHost:
            // This should never happen
            result = new Result(cudaErrorInvalidMemcpyDirection);
            break;
        case cudaMemcpyHostToDevice:
            /* Achtung: this isn't strictly correct because here we assign just
             * a pointer to one character, any successive assign should
             * take inaxpectated result ... but it works here!
             */
            src = input_buffer->AssignAll<char>();
            exit_code = cudaMemcpyToArray(dst, wOffset, hOffset, src, count,
                    kind);
            result = new Result(exit_code);
            break;
        case cudaMemcpyDeviceToHost:
            // This should never happen
            result = new Result(cudaErrorInvalidMemcpyDirection);
            break;
        case cudaMemcpyDeviceToDevice:
            src = input_buffer->GetFromMarshal<void *>();
            exit_code = cudaMemcpyToArray(dst, wOffset, hOffset, src, count,
                    kind);
            result = new Result(exit_code);
            break;
    }
    return result;
}

CUDA_ROUTINE_HANDLER(MemcpyToSymbol) {
    void *src = NULL;

    cudaMemcpyKind kind = input_buffer->BackGet<cudaMemcpyKind > ();
    size_t offset = input_buffer->BackGet<size_t > ();
    size_t count = input_buffer->BackGet<size_t > ();
    char *handler = input_buffer->AssignString();
    char *symbol = input_buffer->AssignString();
    symbol = (char *) CudaUtil::UnmarshalPointer(handler);

    cudaError_t exit_code;
    Result * result = NULL;

    switch (kind) {
        case cudaMemcpyHostToHost:
            // This should never happen
            result = new Result(cudaErrorInvalidMemcpyDirection);
            break;
        case cudaMemcpyHostToDevice:
            src = input_buffer->AssignAll<char>();
            exit_code = cudaMemcpyToSymbol(symbol, src, count, offset, kind);
            result = new Result(exit_code);
            break;
        case cudaMemcpyDeviceToHost:
            // This should never happen
            result = new Result(cudaErrorInvalidMemcpyDirection);
            break;
        case cudaMemcpyDeviceToDevice:
            src = input_buffer->GetFromMarshal<void *>();
            exit_code = cudaMemcpyToSymbol(symbol, src, count, offset, kind);
            result = new Result(exit_code);
            break;
    }
    return result;
}

CUDA_ROUTINE_HANDLER(Memset) {
    void *devPtr = input_buffer->GetFromMarshal<void *>();
    int value = input_buffer->Get<int>();
    size_t count = input_buffer->Get<size_t > ();

    cudaError_t exit_code = cudaMemset(devPtr, value, count);

    return new Result(exit_code);
}