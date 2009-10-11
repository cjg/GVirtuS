/* 
 * File:   CudaRt.cpp
 * Author: cjg
 * 
 * Created on October 9, 2009, 3:55 PM
 */

#include <cstdio>
#include <iostream>
#include <string.h>
#include "Frontend.h"
#include "CudaRt.h"

using namespace std;

/*
 Routines not found in the cuda's header files.
 KEEP THEM WITH CARE
 */

void** __cudaRegisterFatBinary(void *fatCubin) {
    /* FIXME: implement */
    return NULL;
}

void __cudaUnregisterFatBinary(void **fatCubinHandle) {
    /* FIXME: implement */
}

void __cudaRegisterFunction(void **fatCubinHandle, const char *hostFun, char *deviceFun,
        const char *deviceName, int thread_limit, uint3 *tid,
        uint3 *bid, dim3 *bDim, dim3 *gDim, int *wSize) {
    /* FIXME: implement */
}

void __cudaRegisterVar(void **fatCubinHandle, char *hostVar, char *deviceAddress,
        const char *deviceName, int ext, int size, int constant,
        int global) {
    /* FIXME: implement */
}

void __cudaRegisterShared(void **fatCubinHandle, void **devicePtr) {
    /* FIXME: implement */
}

void __cudaRegisterSharedVar(void **fatCubinHandle, void **devicePtr, size_t size,
        size_t alignment, int storage) {
    /* FIXME: implement */
}

/* */


extern int CUDARTAPI __cudaSynchronizeThreads(void** x, void* y) {
    /* FIXME: implement */
    return 0;
}

extern void CUDARTAPI __cudaTextureFetch(const void *tex, void *index, int integer, void *val) {
    /* FIXME: implement */
}

extern __host__ cudaError_t CUDARTAPI cudaBindTexture(size_t *offset, const struct textureReference *texref, const void *devPtr, const struct cudaChannelFormatDesc *desc, size_t size __dv(UINT_MAX)) {
    /* FIXME: implement */
    return cudaErrorUnknown;
}

extern __host__ cudaError_t CUDARTAPI cudaBindTextureToArray(const struct textureReference *texref, const struct cudaArray *array, const struct cudaChannelFormatDesc *desc) {
    /* FIXME: implement */
    return cudaErrorUnknown;
}

extern __host__ cudaError_t CUDARTAPI cudaChooseDevice(int *device, const struct cudaDeviceProp *prop) {
    /* FIXME: implement */
    return cudaErrorUnknown;
}

extern __host__ cudaError_t CUDARTAPI cudaConfigureCall(dim3 gridDim, dim3 blockDim, size_t sharedMem __dv(0), cudaStream_t stream __dv(0)) {
    /* FIXME: implement */
    return cudaErrorUnknown;
}

extern __host__ struct cudaChannelFormatDesc CUDARTAPI cudaCreateChannelDesc(int x, int y, int z, int w, enum cudaChannelFormatKind f) {
    /* FIXME: implement */
    struct cudaChannelFormatDesc result;
    return result;
}

extern __host__ cudaError_t CUDARTAPI cudaEventCreate(cudaEvent_t *event) {
    /* FIXME: implement */
    return cudaErrorUnknown;
}

extern __host__ cudaError_t CUDARTAPI cudaEventDestroy(cudaEvent_t event) {
    /* FIXME: implement */
    return cudaErrorUnknown;
}

extern __host__ cudaError_t CUDARTAPI cudaEventElapsedTime(float *ms, cudaEvent_t start, cudaEvent_t end) {
    /* FIXME: implement */
    return cudaErrorUnknown;
}

extern __host__ cudaError_t CUDARTAPI cudaEventQuery(cudaEvent_t event) {
    /* FIXME: implement */
    return cudaErrorUnknown;
}

extern __host__ cudaError_t CUDARTAPI cudaEventRecord(cudaEvent_t event, cudaStream_t stream) {
    /* FIXME: implement */
    return cudaErrorUnknown;
}

extern __host__ cudaError_t CUDARTAPI cudaEventSynchronize(cudaEvent_t event) {
    /* FIXME: implement */
    return cudaErrorUnknown;
}

extern __host__ cudaError_t CUDARTAPI cudaFree(void *devPtr) {
    /* FIXME: implement */
    return cudaErrorUnknown;
}

extern __host__ cudaError_t CUDARTAPI cudaFreeArray(struct cudaArray *array) {
    /* FIXME: implement */
    return cudaErrorUnknown;
}

extern __host__ cudaError_t CUDARTAPI cudaFreeHost(void *ptr) {
    /* FIXME: implement */
    return cudaErrorUnknown;
}

extern __host__ cudaError_t CUDARTAPI cudaGLMapBufferObject(void **devPtr, GLuint bufObj) {
    /* FIXME: implement */
    return cudaErrorUnknown;
}

extern __host__ cudaError_t CUDARTAPI cudaGLRegisterBufferObject(GLuint bufObj) {
    /* FIXME: implement */
    return cudaErrorUnknown;
}

extern __host__ cudaError_t CUDARTAPI cudaGLSetGLDevice(int device) {
    /* FIXME: implement */
    return cudaErrorUnknown;
}

extern __host__ cudaError_t CUDARTAPI cudaGLUnmapBufferObject(GLuint bufObj) {
    /* FIXME: implement */
    return cudaErrorUnknown;
}

extern __host__ cudaError_t CUDARTAPI cudaGLUnregisterBufferObject(GLuint bufObj) {
    /* FIXME: implement */
    return cudaErrorUnknown;
}

extern __host__ cudaError_t CUDARTAPI cudaGetChannelDesc(struct cudaChannelFormatDesc *desc, const struct cudaArray *array) {
    /* FIXME: implement */
    return cudaErrorUnknown;
}

extern __host__ cudaError_t CUDARTAPI cudaGetDevice(int *device) {
    /* FIXME: implement */
    return cudaErrorUnknown;
}

extern __host__ cudaError_t CUDARTAPI cudaGetDeviceCount(int *count) {
    /* FIXME: implement */
    char *in_buffer, *out_buffer;
    size_t in_buffer_size, out_buffer_size;
    cudaError_t result;

    in_buffer_size = sizeof (int);
    in_buffer = new char[in_buffer_size];

    memmove(in_buffer, count, sizeof (int));

    result = Frontend::GetFrontend().Execute("cudaGetDeviceCount",
            in_buffer, in_buffer_size, &out_buffer, &out_buffer_size);

    if (result == cudaSuccess) {
        memmove(count, out_buffer, sizeof (int));
        delete[] out_buffer;
    }

    delete[] in_buffer;

    return result;
}

extern __host__ cudaError_t CUDARTAPI cudaGetDeviceProperties(struct
        cudaDeviceProp
        *prop,
        int device) {
    /* FIXME: implement */
    char *in_buffer, *out_buffer;
    size_t in_buffer_size, out_buffer_size;
    cudaError_t result;

    in_buffer_size = sizeof (struct cudaDeviceProp) + sizeof (int);
    in_buffer = new char[in_buffer_size];

    memmove(in_buffer, prop, sizeof (struct cudaDeviceProp));
    memmove(in_buffer + sizeof (struct cudaDeviceProp), &device,
            sizeof (int));

    result = Frontend::GetFrontend().Execute("cudaGetDeviceProperties",
            in_buffer, in_buffer_size, &out_buffer, &out_buffer_size);

    if (result == cudaSuccess) {
        memmove(prop, out_buffer, sizeof (struct cudaDeviceProp));
        delete[] out_buffer;
    }

    delete[] in_buffer;

    return result;
}

extern __host__ const char* CUDARTAPI cudaGetErrorString(cudaError_t error) {
    /* FIXME: implement */
    return "";
}

extern __host__ cudaError_t CUDARTAPI cudaGetLastError(void) {
    /* FIXME: implement */
    return cudaErrorUnknown;
}

extern __host__ cudaError_t CUDARTAPI cudaGetSymbolAddress(void **devPtr, const char *symbol) {
    /* FIXME: implement */
    return cudaErrorUnknown;
}

extern __host__ cudaError_t CUDARTAPI cudaGetSymbolSize(size_t *size, const char *symbol) {
    /* FIXME: implement */
    return cudaErrorUnknown;
}

extern __host__ cudaError_t CUDARTAPI cudaGetTextureAlignmentOffset(size_t *offset, const struct textureReference *texref) {
    /* FIXME: implement */
    return cudaErrorUnknown;
}

extern __host__ cudaError_t CUDARTAPI cudaGetTextureReference(const struct textureReference **texref, const char *symbol) {
    /* FIXME: implement */
    return cudaErrorUnknown;
}

extern __host__ cudaError_t CUDARTAPI cudaLaunch(const char *symbol) {
    /* FIXME: implement */
    return cudaErrorUnknown;
}

extern __host__ cudaError_t CUDARTAPI cudaMalloc(void **devPtr, size_t size) {
    /* FIXME: implement */
    cout << "*** cudaMalloc(): devPtr=" << devPtr << " size=" << size << endl;
    return cudaErrorUnknown;
}

extern __host__ cudaError_t CUDARTAPI cudaMalloc3D(struct cudaPitchedPtr* pitchDevPtr, struct cudaExtent extent) {
    /* FIXME: implement */
    return cudaErrorUnknown;
}

extern __host__ cudaError_t CUDARTAPI cudaMalloc3DArray(struct cudaArray** arrayPtr, const struct cudaChannelFormatDesc* desc, struct cudaExtent extent) {
    /* FIXME: implement */
    return cudaErrorUnknown;
}

extern __host__ cudaError_t CUDARTAPI cudaMallocArray(struct cudaArray **array, const struct cudaChannelFormatDesc *desc, size_t width, size_t height __dv(1)) {
    /* FIXME: implement */
    return cudaErrorUnknown;
}

extern __host__ cudaError_t CUDARTAPI cudaMallocHost(void **ptr, size_t size) {
    /* FIXME: implement */
    return cudaErrorUnknown;
}

extern __host__ cudaError_t CUDARTAPI cudaMallocPitch(void **devPtr, size_t *pitch, size_t width, size_t height) {
    /* FIXME: implement */
    return cudaErrorUnknown;
}

extern __host__ cudaError_t CUDARTAPI cudaMemcpy(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind) {
    /* FIXME: implement */
    return cudaErrorUnknown;
}

extern __host__ cudaError_t CUDARTAPI cudaMemcpy2D(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind) {
    /* FIXME: implement */
    return cudaErrorUnknown;
}

extern __host__ cudaError_t CUDARTAPI cudaMemcpy2DArrayToArray(struct cudaArray *dst, size_t wOffsetDst, size_t hOffsetDst, const struct cudaArray *src, size_t wOffsetSrc, size_t hOffsetSrc, size_t width, size_t height, enum cudaMemcpyKind kind __dv(cudaMemcpyDeviceToDevice)) {
    /* FIXME: implement */
    return cudaErrorUnknown;
}

extern __host__ cudaError_t CUDARTAPI cudaMemcpy2DAsync(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind, cudaStream_t stream) {
    /* FIXME: implement */
    return cudaErrorUnknown;
}

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

extern __host__ cudaError_t CUDARTAPI cudaSetDevice(int device) {
    /* FIXME: implement */
    return cudaErrorUnknown;
}

extern __host__ cudaError_t CUDARTAPI cudaSetDoubleForDevice(double *d) {
    /* FIXME: implement */
    return cudaErrorUnknown;
}

extern __host__ cudaError_t CUDARTAPI cudaSetDoubleForHost(double *d) {
    /* FIXME: implement */
    return cudaErrorUnknown;
}

extern __host__ cudaError_t CUDARTAPI cudaSetupArgument(const void *arg, size_t size, size_t offset) {
    /* FIXME: implement */
    return cudaErrorUnknown;
}

extern __host__ cudaError_t CUDARTAPI cudaStreamCreate(cudaStream_t *stream) {
    /* FIXME: implement */
    return cudaErrorUnknown;
}

extern __host__ cudaError_t CUDARTAPI cudaStreamDestroy(cudaStream_t stream) {
    /* FIXME: implement */
    return cudaErrorUnknown;
}

extern __host__ cudaError_t CUDARTAPI cudaStreamQuery(cudaStream_t stream) {
    /* FIXME: implement */
    return cudaErrorUnknown;
}

extern __host__ cudaError_t CUDARTAPI cudaStreamSynchronize(cudaStream_t stream) {
    /* FIXME: implement */
    return cudaErrorUnknown;
}

extern __host__ cudaError_t CUDARTAPI cudaThreadExit(void) {
    /* FIXME: implement */
    return cudaErrorUnknown;
}

extern __host__ cudaError_t CUDARTAPI cudaThreadSynchronize(void) {
    /* FIXME: implement */
    return cudaErrorUnknown;
}

extern __host__ cudaError_t CUDARTAPI cudaUnbindTexture(const struct textureReference *texref) {
    /* FIXME: implement */
    return cudaErrorUnknown;
}
