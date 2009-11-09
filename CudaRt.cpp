/* 
 * File:   CudaRt.cpp
 * Author: cjg
 * 
 * Created on October 9, 2009, 3:55 PM
 */

#include <cstdio>
#include <iostream>
#include <string.h>
#include <__cudaFatFormat.h>
#include <fstream>
#include "Frontend.h"
#include "CudaUtil.h"
#include "CudaRt.h"

using namespace std;

extern __host__ cudaError_t CUDARTAPI cudaMemcpy2DFromArray(void *dst, size_t dpitch, const struct cudaArray *src, size_t wOffset, size_t hOffset, size_t width, size_t height, enum cudaMemcpyKind kind) {
    /* FIXME: implement */
    return cudaErrorUnknown;
}

extern __host__ cudaError_t CUDARTAPI cudaMemcpy2DFromArrayAsync(void *dst, size_t dpitch, const struct cudaArray *src, size_t wOffset, size_t hOffset, size_t width, size_t height, enum cudaMemcpyKind kind, cudaStream_t stream) {
    /* FIXME: implement */
    return cudaErrorUnknown;
}

extern __host__ cudaError_t CUDARTAPI cudaMemcpy2DToArray(struct cudaArray *dst, size_t wOffset, size_t hOffset, const void *src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind) {
    /* FIXME: implement */
    return cudaErrorUnknown;
}

extern __host__ cudaError_t CUDARTAPI cudaMemcpy2DToArrayAsync(struct cudaArray *dst, size_t wOffset, size_t hOffset, const void *src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind, cudaStream_t stream) {
    /* FIXME: implement */
    return cudaErrorUnknown;
}

extern __host__ cudaError_t CUDARTAPI cudaMemcpy3D(const struct cudaMemcpy3DParms *p) {
    /* FIXME: implement */
    return cudaErrorUnknown;
}

extern __host__ cudaError_t CUDARTAPI cudaMemcpy3DAsync(const struct cudaMemcpy3DParms *p, cudaStream_t stream) {
    /* FIXME: implement */
    return cudaErrorUnknown;
}

extern __host__ cudaError_t CUDARTAPI cudaMemcpyArrayToArray(struct cudaArray *dst, size_t wOffsetDst, size_t hOffsetDst, const struct cudaArray *src, size_t wOffsetSrc, size_t hOffsetSrc, size_t count, enum cudaMemcpyKind kind __dv(cudaMemcpyDeviceToDevice)) {
    /* FIXME: implement */
    return cudaErrorUnknown;
}

extern __host__ cudaError_t CUDARTAPI cudaMemcpyAsync(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream) {
    /* FIXME: implement */
    return cudaErrorUnknown;
}

extern __host__ cudaError_t CUDARTAPI cudaMemcpyFromArray(void *dst, const struct cudaArray *src, size_t wOffset, size_t hOffset, size_t count, enum cudaMemcpyKind kind) {
    /* FIXME: implement */
    return cudaErrorUnknown;
}

extern __host__ cudaError_t CUDARTAPI cudaMemcpyFromArrayAsync(void *dst, const struct cudaArray *src, size_t wOffset, size_t hOffset, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream) {
    /* FIXME: implement */
    return cudaErrorUnknown;
}

extern __host__ cudaError_t CUDARTAPI cudaMemcpyFromSymbol(void *dst, const char *symbol, size_t count, size_t offset __dv(0), enum cudaMemcpyKind kind __dv(cudaMemcpyDeviceToHost)) {
    /* FIXME: implement */
    return cudaErrorUnknown;
}

extern __host__ cudaError_t CUDARTAPI cudaMemcpyFromSymbolAsync(void *dst, const char *symbol, size_t count, size_t offset, enum cudaMemcpyKind kind, cudaStream_t stream) {
    /* FIXME: implement */
    return cudaErrorUnknown;
}

extern __host__ cudaError_t CUDARTAPI cudaMemcpyToArray(struct cudaArray *dst, size_t wOffset, size_t hOffset, const void *src, size_t count, enum cudaMemcpyKind kind) {
    /* FIXME: implement */
    return cudaErrorUnknown;
}

extern __host__ cudaError_t CUDARTAPI cudaMemcpyToArrayAsync(struct cudaArray *dst, size_t wOffset, size_t hOffset, const void *src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream) {
    /* FIXME: implement */
    return cudaErrorUnknown;
}

extern __host__ cudaError_t CUDARTAPI cudaMemcpyToSymbol(const char *symbol, const void *src, size_t count, size_t offset __dv(0), enum cudaMemcpyKind kind __dv(cudaMemcpyHostToDevice)) {
    /* FIXME: implement */
    return cudaErrorUnknown;
}

extern __host__ cudaError_t CUDARTAPI cudaMemcpyToSymbolAsync(const char *symbol, const void *src, size_t count, size_t offset, enum cudaMemcpyKind kind, cudaStream_t stream) {
    /* FIXME: implement */
    return cudaErrorUnknown;
}

extern __host__ cudaError_t CUDARTAPI cudaMemset(void *mem, int c, size_t count) {
    /* FIXME: implement */
    return cudaErrorUnknown;
}

extern __host__ cudaError_t CUDARTAPI cudaMemset2D(void *mem, size_t pitch, int c, size_t width, size_t height) {
    /* FIXME: implement */
    return cudaErrorUnknown;
}

extern __host__ cudaError_t CUDARTAPI cudaMemset3D(struct cudaPitchedPtr pitchDevPtr, int value, struct cudaExtent extent) {
    /* FIXME: implement */
    return cudaErrorUnknown;
}
