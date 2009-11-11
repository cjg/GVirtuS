#include <cstring>
#include "Frontend.h"
#include "CudaUtil.h"
#include "CudaRt.h"

using namespace std;

extern cudaError_t cudaFree(void *devPtr) {
    Frontend *f = Frontend::GetFrontend();
    f->AddDevicePointerForArguments(devPtr);
    f->Execute("cudaFree");
    return f->GetExitCode();
}

extern cudaError_t cudaFreeArray(cudaArray *array) {
    Frontend *f = Frontend::GetFrontend();
    f->AddDevicePointerForArguments((void *) array);
    f->Execute("cudaFreeArray");
    return f->GetExitCode();
}

extern cudaError_t cudaFreeHost(void *ptr) {
    free(ptr);
    return cudaSuccess;
}

extern cudaError_t cudaGetSymbolAddress(void **devPtr, const char *symbol) {
    // FIXME: implement
    cerr << "*** Error: cudaGetSymbolAddress() not yet implemented!" << endl;
    return cudaErrorUnknown;
}

extern cudaError_t cudaGetSymbolSize(size_t *size, const char *symbol) {
    // FIXME: implement
    cerr << "*** Error: cudaGetSymbolSize() not yet implemented!" << endl;
    return cudaErrorUnknown;
}

extern cudaError_t cudaHostAlloc(void **ptr, size_t size, unsigned int flags) {
    // FIXME: implement
    cerr << "*** Error: cudaHostAlloc() not yet implemented!" << endl;
    return cudaErrorUnknown;
}

extern cudaError_t cudaHostGetDevicePointer(void **pDevice, void *pHost,
        unsigned int flags) {
    // FIXME: implement
    cerr << "*** Error: cudaHostGetDevicePointer() not yet implemented!"
            << endl;
    return cudaErrorUnknown;
}

extern cudaError_t cudaHostGetFlags(unsigned int *pFlags, void *pHost) {
    // FIXME: implement
    cerr << "*** Error: cudaHostGetFlags() not yet implemented!" << endl;
    return cudaErrorUnknown;
}

extern cudaError_t cudaMalloc(void **devPtr, size_t size) {
    Frontend *f = Frontend::GetFrontend();

    /* Fake device pointer */
    *devPtr = new char[1];
    f->AddDevicePointerForArguments(*devPtr);
    f->AddVariableForArguments(size);
    f->Execute("cudaMalloc");
    return f->GetExitCode();
}

extern cudaError_t cudaMalloc3D(cudaPitchedPtr *pitchedDevPtr,
        cudaExtent extent) {
    // FIXME: implement
    cerr << "*** Error: cudaMalloc3D() not yet implemented!" << endl;
    return cudaErrorUnknown;
}

extern cudaError_t cudaMalloc3DArray(cudaArray **arrayPtr,
        const cudaChannelFormatDesc *desc, cudaExtent extent) {
    // FIXME: implement
    cerr << "*** Error: cudaMalloc3DArray() not yet implemented!" << endl;
    return cudaErrorUnknown;
}

extern cudaError_t cudaMallocArray(cudaArray **arrayPtr,
        const cudaChannelFormatDesc *desc, size_t width, size_t height) {
    Frontend *f = Frontend::GetFrontend();

    /* Fake device pointer */
    *arrayPtr = (cudaArray *) new char[1];
    f->AddDevicePointerForArguments(*arrayPtr);
    f->AddHostPointerForArguments(desc);
    f->AddVariableForArguments(width);
    f->AddVariableForArguments(height);
    f->Execute("cudaMallocArray");
    return f->GetExitCode();
}

extern cudaError_t cudaMallocHost(void **ptr, size_t size) {
    // Achtung: we can't use host page-locked memory, so we use simple pageable
    // memory here.
    if ((*ptr = malloc(size)) == NULL)
        return cudaErrorMemoryAllocation;
    return cudaSuccess;
}

extern cudaError_t cudaMallocPitch(void **devPtr, size_t *pitch, size_t width,
        size_t height) {
    // FIXME: implement
    cerr << "*** Error: cudaMallocPitch() not yet implemented!" << endl;
    return cudaErrorUnknown;
}

extern cudaError_t cudaMemcpy(void *dst, const void *src, size_t count,
        cudaMemcpyKind kind) {
    Frontend *f = Frontend::GetFrontend();
    switch (kind) {
        case cudaMemcpyHostToHost:
            /* NOTE: no communication is performed, because it's just overhead
             * here */
            if (memmove(dst, src, count) == NULL)
                return cudaErrorInvalidValue;
            return cudaSuccess;
            break;
        case cudaMemcpyHostToDevice:
            f->AddDevicePointerForArguments(dst);
            f->AddHostPointerForArguments<char>(static_cast<char *>
                    (const_cast<void *> (src)), count);
            f->AddVariableForArguments(count);
            f->AddVariableForArguments(kind);
            f->Execute("cudaMemcpy");
            break;
        case cudaMemcpyDeviceToHost:
            /* NOTE: adding a fake host pointer */
            f->AddHostPointerForArguments("");
            f->AddDevicePointerForArguments(src);
            f->AddVariableForArguments(count);
            f->AddVariableForArguments(kind);
            f->Execute("cudaMemcpy");
            if (f->Success())
                memmove(dst, f->GetOutputHostPointer<char>(count), count);
            break;
        case cudaMemcpyDeviceToDevice:
            f->AddDevicePointerForArguments(dst);
            f->AddDevicePointerForArguments(src);
            f->AddVariableForArguments(count);
            f->AddVariableForArguments(kind);
            f->Execute("cudaMemcpy");
            break;
    }

    return f->GetExitCode();
}

extern cudaError_t cudaMemcpy2D(void *dst, size_t dpitch, const void *src,
        size_t spitch, size_t width, size_t height, cudaMemcpyKind kind) {
    // FIXME: implement
    cerr << "*** Error: cudaMemcpy2D() not yet implemented!" << endl;
    return cudaErrorUnknown;
}

extern cudaError_t cudaMemcpy2DArrayToArray(cudaArray *dst, size_t wOffsetDst,
        size_t hOffsetDst, const cudaArray *src, size_t wOffsetSrc,
        size_t hOffsetSrc, size_t width, size_t height, cudaMemcpyKind kind) {
    // FIXME: implement
    cerr << "*** Error: cudaMemcpy2DArrayToArray() not yet implemented!"
            << endl;
    return cudaErrorUnknown;
}

extern cudaError_t cudaMemcpy2DAsync(void *dst, size_t dpitch, const void *src,
        size_t spitch, size_t width, size_t height, cudaMemcpyKind kind,
        cudaStream_t stream) {
    // FIXME: implement
    cerr << "*** Error: cudaMemcpy2DAsync() not yet implemented!" << endl;
    return cudaErrorUnknown;
}

extern cudaError_t cudaMemcpy2DFromArray(void *dst, size_t dpitch,
        const cudaArray *src, size_t wOffset, size_t hOffset, size_t width,
        size_t height, cudaMemcpyKind kind) {
    // FIXME: implement
    cerr << "*** Error: cudaMemcpy2DFromArray() not yet implemented!" << endl;
    return cudaErrorUnknown;
}

extern cudaError_t cudaMemcpy2DFromArrayAsync(void *dst, size_t dpitch,
        const cudaArray *src, size_t wOffset, size_t hOffset, size_t width,
        size_t height, cudaMemcpyKind kind, cudaStream_t stream) {
    // FIXME: implement
    cerr << "*** Error: cudaMemcpy2DFromArrayAsync() not yet implemented!" << endl;
    return cudaErrorUnknown;
}

extern cudaError_t cudaMemcpy2DToArray(cudaArray *dst, size_t wOffset,
        size_t hOffset, const void *src, size_t spitch, size_t width,
        size_t height, cudaMemcpyKind kind) {
    // FIXME: implement
    cerr << "*** Error: cudaMemcpy2DToArray() not yet implemented!" << endl;
    return cudaErrorUnknown;
}

extern cudaError_t cudaMemcpy2DToArrayAsync(cudaArray *dst, size_t wOffset,
        size_t hOffset, const void *src, size_t spitch, size_t width,
        size_t height, cudaMemcpyKind kind, cudaStream_t stream) {
    // FIXME: implement
    cerr << "*** Error: cudaMemcpy2DToArrayAsync() not yet implemented!" << endl;
    return cudaErrorUnknown;
}

extern cudaError_t cudaMemcpy3D(const cudaMemcpy3DParms *p) {
    // FIXME: implement
    cerr << "*** Error: cudaMemcpy3D() not yet implemented!" << endl;
    return cudaErrorUnknown;
}

extern cudaError_t cudaMemcpy3DAsync(const cudaMemcpy3DParms *p,
        cudaStream_t stream) {
    // FIXME: implement
    cerr << "*** Error: cudaMemcpy3DAsync() not yet implemented!" << endl;
    return cudaErrorUnknown;
}

extern cudaError_t cudaMemcpyArrayToArray(cudaArray *dst, size_t wOffsetDst,
        size_t hOffsetDst, const cudaArray *src, size_t wOffsetSrc,
        size_t hOffsetSrc, size_t count,
        cudaMemcpyKind kind __dv(cudaMemcpyDeviceToDevice)) {
    // FIXME: implement
    cerr << "*** Error: cudaMemcpyArrayToArray() not yet implemented!" << endl;
    return cudaErrorUnknown;
}

extern cudaError_t cudaMemcpyAsync(void *dst, const void *src, size_t count,
        cudaMemcpyKind kind, cudaStream_t stream) {
    Frontend *f = Frontend::GetFrontend();
    switch (kind) {
        case cudaMemcpyHostToHost:
            /* NOTE: no communication is performed, because it's just overhead
             * here */
            f->AddHostPointerForArguments("");
            f->AddHostPointerForArguments("");
            f->AddVariableForArguments(kind);
            f->AddVariableForArguments(stream);
            f->Execute("cudaMemcpyAsync");
            if (memmove(dst, src, count) == NULL)
                return cudaErrorInvalidValue;
            return cudaSuccess;
            break;
        case cudaMemcpyHostToDevice:
            f->AddDevicePointerForArguments(dst);
            f->AddHostPointerForArguments<char>(static_cast<char *>
                    (const_cast<void *> (src)), count);
            f->AddVariableForArguments(count);
            f->AddVariableForArguments(kind);
            f->AddVariableForArguments(stream);
            f->Execute("cudaMemcpyAsync");
            break;
        case cudaMemcpyDeviceToHost:
            /* NOTE: adding a fake host pointer */
            f->AddHostPointerForArguments("");
            f->AddDevicePointerForArguments(src);
            f->AddVariableForArguments(count);
            f->AddVariableForArguments(kind);
            f->AddVariableForArguments(stream);
            f->Execute("cudaMemcpyAsync");
            if (f->Success())
                memmove(dst, f->GetOutputHostPointer<char>(count), count);
            break;
        case cudaMemcpyDeviceToDevice:
            f->AddDevicePointerForArguments(dst);
            f->AddDevicePointerForArguments(src);
            f->AddVariableForArguments(count);
            f->AddVariableForArguments(kind);
            f->AddVariableForArguments(stream);
            f->Execute("cudaMemcpyAsync");
            break;
    }

    return f->GetExitCode();
}

extern cudaError_t cudaMemcpyFromArray(void *dst, const cudaArray *src,
        size_t wOffset, size_t hOffset, size_t count, cudaMemcpyKind kind) {
    // FIXME: implement
    cerr << "*** Error: cudaMemcpyFromArray() not yet implemented!" << endl;
    return cudaErrorUnknown;
}

extern cudaError_t cudaMemcpyFromArrayAsync(void *dst, const cudaArray *src,
        size_t wOffset, size_t hOffset, size_t count, cudaMemcpyKind kind,
        cudaStream_t stream) {
    // FIXME: implement
    cerr << "*** Error: cudaMemcpyFromArrayAsync() not yet implemented!"
            << endl;
    return cudaErrorUnknown;
}

extern cudaError_t cudaMemcpyFromSymbol(void *dst, const char *symbol,
        size_t count, size_t offset __dv(0),
        cudaMemcpyKind kind __dv(cudaMemcpyDeviceToHost)) {
    Frontend *f = Frontend::GetFrontend();
    switch (kind) {
        case cudaMemcpyHostToHost:
            /* This should never happen. */
            return cudaErrorInvalidMemcpyDirection;
            break;
        case cudaMemcpyHostToDevice:
            /* This should never happen. */
            return cudaErrorInvalidMemcpyDirection;
            break;
        case cudaMemcpyDeviceToHost:
            // Achtung: adding a fake host pointer 
            f->AddDevicePointerForArguments((void *) 0x666);
            // Achtung: passing the address and the content of symbol
            f->AddStringForArguments(CudaUtil::MarshalHostPointer((void *) symbol));
            f->AddStringForArguments(symbol);
            f->AddVariableForArguments(count);
            f->AddVariableForArguments(offset);
            f->AddVariableForArguments(kind);
            f->Execute("cudaMemcpyFromSymbol");
            if (f->Success())
                memmove(dst, f->GetOutputHostPointer<char>(count), count);
            break;
        case cudaMemcpyDeviceToDevice:
            f->AddDevicePointerForArguments(dst);
            // Achtung: passing the address and the content of symbol
            f->AddStringForArguments(CudaUtil::MarshalHostPointer((void *) symbol));
            f->AddStringForArguments(symbol);
            f->AddVariableForArguments(count);
            f->AddVariableForArguments(offset);
            f->AddVariableForArguments(kind);
            f->Execute("cudaMemcpyFromSymbol");
            break;
    }

    return f->GetExitCode();
}

extern cudaError_t cudaMemcpyFromSymbolAsync(void *dst, const char *symbol,
        size_t count, size_t offset, cudaMemcpyKind kind, cudaStream_t stream) {
    // FIXME: implement
    cerr << "*** Error: cudaMemcpyFromSymbolAsync() not yet implemented!"
            << endl;
    return cudaErrorUnknown;
}

extern cudaError_t cudaMemcpyToArray(cudaArray *dst, size_t wOffset,
        size_t hOffset, const void *src, size_t count, cudaMemcpyKind kind) {
    Frontend *f = Frontend::GetFrontend();
    switch (kind) {
        case cudaMemcpyHostToHost:
            /* This should never happen. */
            return cudaErrorInvalidMemcpyDirection;
            break;
        case cudaMemcpyHostToDevice:
            cout << "h2d " << count << endl;
            f->AddDevicePointerForArguments((void *) dst);
            f->AddVariableForArguments(wOffset);
            f->AddVariableForArguments(hOffset);
            f->AddHostPointerForArguments<char>(static_cast<char *>
                    (const_cast<void *> (src)), count);
            f->AddVariableForArguments(count);
            f->AddVariableForArguments(kind);
            f->Execute("cudaMemcpyToArray");
            break;
        case cudaMemcpyDeviceToHost:
            /* This should never happen. */
            return cudaErrorInvalidMemcpyDirection;
            break;
        case cudaMemcpyDeviceToDevice:
            f->AddDevicePointerForArguments((void *) dst);
            f->AddVariableForArguments(wOffset);
            f->AddVariableForArguments(hOffset);
            f->AddDevicePointerForArguments(src);
            f->AddVariableForArguments(count);
            f->AddVariableForArguments(kind);
            f->Execute("cudaMemcpyToArray");
            break;
    }

    return f->GetExitCode();
}

extern cudaError_t cudaMemcpyToArrayAsync(cudaArray *dst, size_t wOffset,
        size_t hOffset, const void *src, size_t count, cudaMemcpyKind kind,
        cudaStream_t stream) {
    // FIXME: implement
    cerr << "*** Error: cudaMemcpyToArrayAsync() not yet implemented!" << endl;
    return cudaErrorUnknown;
}

extern cudaError_t cudaMemcpyToSymbol(const char *symbol, const void *src,
        size_t count, size_t offset __dv(0),
        cudaMemcpyKind kind __dv(cudaMemcpyHostToDevice)) {
    Frontend *f = Frontend::GetFrontend();
    switch (kind) {
        case cudaMemcpyHostToHost:
            /* This should never happen. */
            return cudaErrorInvalidMemcpyDirection;
            break;
        case cudaMemcpyHostToDevice:
            // Achtung: passing the address and the content of symbol
            f->AddStringForArguments(CudaUtil::MarshalHostPointer((void *) symbol));
            f->AddStringForArguments(symbol);
            f->AddHostPointerForArguments<char>(static_cast<char *>
                    (const_cast<void *> (src)), count);
            f->AddVariableForArguments(count);
            f->AddVariableForArguments(offset);
            f->AddVariableForArguments(kind);
            f->Execute("cudaMemcpyToSymbol");
            break;
        case cudaMemcpyDeviceToHost:
            /* This should never happen. */
            return cudaErrorInvalidMemcpyDirection;
            break;
        case cudaMemcpyDeviceToDevice:
            // Achtung: passing the address and the content of symbol
            f->AddStringForArguments(CudaUtil::MarshalHostPointer((void *) symbol));
            f->AddStringForArguments(symbol);
            f->AddDevicePointerForArguments(src);
            f->AddVariableForArguments(count);
            f->AddVariableForArguments(kind);
            f->Execute("cudaMemcpyToSymbol");
            break;
    }

    return f->GetExitCode();
}

extern cudaError_t cudaMemcpyToSymbolAsync(const char *symbol, const void *src,
        size_t count, size_t offset, cudaMemcpyKind kind, cudaStream_t stream) {
    // FIXME: implement
    cerr << "*** Error: cudaMemcpyToSymbolAsync() not yet implemented!" << endl;
    return cudaErrorUnknown;
}

extern cudaError_t cudaMemset(void *devPtr, int c, size_t count) {
    Frontend *f = Frontend::GetFrontend();
    f->AddDevicePointerForArguments(devPtr);
    f->AddVariableForArguments(c);
    f->AddVariableForArguments(count);
    f->Execute("cudaMemset");
    return f->GetExitCode();
}

extern cudaError_t cudaMemset2D(void *mem, size_t pitch, int c, size_t width,
        size_t height) {
    // FIXME: implement
    cerr << "*** Error: cudaMemset2D() not yet implemented!" << endl;
    return cudaErrorUnknown;
}

extern cudaError_t cudaMemset3D(cudaPitchedPtr pitchDevPtr, int value,
        cudaExtent extent) {
    // FIXME: implement
    cerr << "*** Error: cudaMemset3D() not yet implemented!" << endl;
    return cudaErrorUnknown;
}