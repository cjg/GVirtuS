/*
 * File:   CudaRt.h
 * Author: cjg
 *
 * Created on October 9, 2009, 3:55 PM
 */

#ifndef _CUDART_H
#define	_CUDART_H

#include <host_defines.h>
#include <builtin_types.h>
#include <driver_types.h>
#include <GL/gl.h>
#include "CudaUtil.h"
#include "Frontend.h"

#define __dv(v)

extern "C" {
    /* CudaRt_device */
    extern cudaError_t cudaChooseDevice(int *device,
            const cudaDeviceProp *prop);
    extern cudaError_t cudaGetDevice(int *device);
    extern cudaError_t cudaGetDeviceCount(int *count);
    extern cudaError_t cudaGetDeviceProperties(cudaDeviceProp *prop,
            int device);
    extern cudaError_t cudaSetDevice(int device);
    extern cudaError_t cudaSetDeviceFlags(int flags);
    extern cudaError_t cudaSetValidDevices(int *device_arr, int len);

    /* CudaRt_error */
    extern const char* cudaGetErrorString(cudaError_t error);
    extern cudaError_t cudaGetLastError(void);

    /* CudaRt_event */
    extern cudaError_t cudaEventCreate(cudaEvent_t *event);
    extern cudaError_t cudaEventCreateWithFlags(cudaEvent_t *event, int flags);
    extern cudaError_t cudaEventDestroy(cudaEvent_t event);
    extern cudaError_t cudaEventElapsedTime(float *ms, cudaEvent_t start,
            cudaEvent_t end);
    extern cudaError_t cudaEventQuery(cudaEvent_t event);
    extern cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream);
    extern cudaError_t cudaEventSynchronize(cudaEvent_t event);

    /* CudaRt_execution */
    extern cudaError_t cudaConfigureCall(dim3 gridDim, dim3 blockDim,
            size_t sharedMem, cudaStream_t stream);
    extern cudaError_t cudaFuncGetAttributes(struct cudaFuncAttributes *attr,
            const char *func);
    extern cudaError_t cudaLaunch(const char *entry);
    extern cudaError_t cudaSetDoubleForDevice(double *d);
    extern cudaError_t cudaSetDoubleForHost(double *d);
    extern cudaError_t cudaSetupArgument(const void *arg, size_t size,
            size_t offset);

    /* CudaRt_internal */
    extern void** __cudaRegisterFatBinary(void *fatCubin);
    extern void __cudaUnregisterFatBinary(void **fatCubinHandle);
    extern void __cudaRegisterFunction(void **fatCubinHandle,
            const char *hostFun, char *deviceFun, const char *deviceName,
            int thread_limit, uint3 *tid, uint3 *bid, dim3 *bDim, dim3 *gDim,
            int *wSize);
    extern void __cudaRegisterVar(void **fatCubinHandle, char *hostVar,
            char *deviceAddress, const char *deviceName, int ext, int size,
            int constant, int global);
    extern void __cudaRegisterShared(void **fatCubinHandle, void **devicePtr);
    extern void __cudaRegisterSharedVar(void **fatCubinHandle, void **devicePtr,
            size_t size, size_t alignment, int storage);
    extern int __cudaSynchronizeThreads(void** x, void* y);
    extern void __cudaTextureFetch(const void *tex, void *index, int integer,
            void *val);

    /* CudaRt_memory */
    extern cudaError_t cudaFree(void *devPtr);
    extern cudaError_t cudaFreeArray(struct cudaArray * array);
    extern cudaError_t cudaFreeHost(void *ptr);
    extern cudaError_t cudaGetSymbolAddress(void **devPtr, const char *symbol);
    extern cudaError_t cudaGetSymbolSize(size_t *size, const char *symbol);
    extern cudaError_t cudaHostAlloc(void **ptr, size_t size,
            unsigned int flags);
    extern cudaError_t cudaHostGetDevicePointer(void **pDevice, void *pHost,
            unsigned int flags);
    extern cudaError_t cudaHostGetFlags(unsigned int *pFlags, void *pHost);
    extern cudaError_t cudaMalloc(void **devPtr, size_t size);
    extern cudaError_t cudaMalloc3D(cudaPitchedPtr *pitchedDevPtr,
            cudaExtent extent);
    extern cudaError_t cudaMalloc3DArray(cudaArray **arrayPtr,
            const cudaChannelFormatDesc *desc, cudaExtent extent);
    extern cudaError_t cudaMallocArray(struct cudaArray **arrayPtr,
            const cudaChannelFormatDesc *desc, size_t width, size_t height);
    extern cudaError_t cudaMallocHost(void **ptr, size_t size);
    extern cudaError_t cudaMallocPitch(void **devPtr, size_t *pitch,
            size_t width, size_t height);
    extern cudaError_t cudaMemcpy(void *dst, const void *src, size_t count,
            cudaMemcpyKind kind);
    extern cudaError_t cudaMemcpy2D(void *dst, size_t dpitch, const void *src,
            size_t spitch, size_t width, size_t height, cudaMemcpyKind kind);
    extern cudaError_t cudaMemcpy2DArrayToArray(cudaArray *dst,
            size_t wOffsetDst, size_t hOffsetDst, const cudaArray *src,
            size_t wOffsetSrc, size_t hOffsetSrc, size_t width, size_t height,
            cudaMemcpyKind kind);
    extern cudaError_t cudaMemcpy2DAsync(void *dst, size_t dpitch,
            const void *src, size_t spitch, size_t width, size_t height,
            cudaMemcpyKind kind, cudaStream_t stream);
    extern cudaError_t cudaMemcpy2DFromArray(void *dst, size_t dpitch,
            const cudaArray *src, size_t wOffset, size_t hOffset, size_t width,
            size_t height, cudaMemcpyKind kind);
    extern cudaError_t cudaMemcpy2DFromArrayAsync(void *dst, size_t dpitch,
            const cudaArray *src, size_t wOffset, size_t hOffset, size_t width,
            size_t height, cudaMemcpyKind kind, cudaStream_t stream);
    extern cudaError_t cudaMemcpy2DToArray(cudaArray *dst, size_t wOffset,
            size_t hOffset, const void *src, size_t spitch, size_t width,
            size_t height, cudaMemcpyKind kind);
    extern cudaError_t cudaMemcpy2DToArrayAsync(cudaArray *dst, size_t wOffset,
            size_t hOffset, const void *src, size_t spitch, size_t width,
            size_t height, cudaMemcpyKind kind, cudaStream_t stream);
    extern cudaError_t cudaMemcpy3D(const cudaMemcpy3DParms * p);
    extern cudaError_t cudaMemcpy3DAsync(const cudaMemcpy3DParms *p,
            cudaStream_t stream);
    extern cudaError_t cudaMemcpyArrayToArray(cudaArray *dst, size_t wOffsetDst,
            size_t hOffsetDst, const cudaArray *src, size_t wOffsetSrc,
            size_t hOffsetSrc, size_t count,
            cudaMemcpyKind kind __dv(cudaMemcpyDeviceToDevice));
    extern cudaError_t cudaMemcpyAsync(void *dst, const void *src, size_t count,
            cudaMemcpyKind kind, cudaStream_t stream);
    extern cudaError_t cudaMemcpyFromArray(void *dst, const cudaArray *src,
            size_t wOffset, size_t hOffset, size_t count, cudaMemcpyKind kind);
    extern cudaError_t cudaMemcpyFromArrayAsync(void *dst, const cudaArray *src,
            size_t wOffset, size_t hOffset, size_t count, cudaMemcpyKind kind,
            cudaStream_t stream);
    extern cudaError_t cudaMemcpyFromSymbol(void *dst, const char *symbol,
            size_t count, size_t offset __dv(0),
            cudaMemcpyKind kind __dv(cudaMemcpyDeviceToHost));
    extern cudaError_t cudaMemcpyFromSymbolAsync(void *dst, const char *symbol,
            size_t count, size_t offset, cudaMemcpyKind kind,
            cudaStream_t stream);
    extern cudaError_t cudaMemcpyToArray(cudaArray *dst, size_t wOffset,
            size_t hOffset, const void *src, size_t count, cudaMemcpyKind kind);
    extern cudaError_t cudaMemcpyToArrayAsync(cudaArray *dst, size_t wOffset,
            size_t hOffset, const void *src, size_t count, cudaMemcpyKind kind,
            cudaStream_t stream);
    extern cudaError_t cudaMemcpyToSymbol(const char *symbol, const void *src,
            size_t count, size_t offset __dv(0),
            cudaMemcpyKind kind __dv(cudaMemcpyHostToDevice));
    extern cudaError_t cudaMemcpyToSymbolAsync(const char *symbol,
            const void *src, size_t count, size_t offset, cudaMemcpyKind kind,
            cudaStream_t stream);
    extern cudaError_t cudaMemset(void *mem, int c, size_t count);
    extern cudaError_t cudaMemset2D(void *mem, size_t pitch, int c,
            size_t width, size_t height);
    extern cudaError_t cudaMemset3D(cudaPitchedPtr pitchDevPtr, int value,
            cudaExtent extent);

    /* CudaRt_opengl */
    extern cudaError_t cudaGLMapBufferObject(void **devPtr, GLuint bufObj);
    extern cudaError_t cudaGLMapBufferObjectAsync(void **devPtr, GLuint bufObj,
            cudaStream_t stream);
    extern cudaError_t cudaGLRegisterBufferObject(GLuint bufObj);
    extern cudaError_t cudaGLSetBufferObjectMapFlags(GLuint bufObj,
            unsigned int flags);
    extern cudaError_t cudaGLSetGLDevice(int device);
    extern cudaError_t cudaGLUnmapBufferObject(GLuint bufObj);
    extern cudaError_t cudaGLUnmapBufferObjectAsync(GLuint bufObj,
            cudaStream_t stream);
    extern cudaError_t cudaGLUnregisterBufferObject(GLuint bufObj);

    /* CudaRt_stream */
    extern cudaError_t cudaStreamCreate(cudaStream_t * pStream);
    extern cudaError_t cudaStreamDestroy(cudaStream_t stream);
    extern cudaError_t cudaStreamQuery(cudaStream_t stream);
    extern cudaError_t cudaStreamSynchronize(cudaStream_t stream);

    /* CudaRt_texture */
    extern cudaError_t cudaBindTexture(size_t *offset,
            const textureReference *texref, const void *devPtr,
            const cudaChannelFormatDesc *desc, size_t size);
    extern cudaError_t cudaBindTexture2D(size_t *offset,
            const textureReference *texref, const void *devPtr,
            const cudaChannelFormatDesc *desc, size_t width, size_t height,
            size_t pitch);
    extern cudaError_t cudaBindTextureToArray(const textureReference *texref,
            const cudaArray *array, const cudaChannelFormatDesc *desc);
    extern cudaChannelFormatDesc cudaCreateChannelDesc(int x, int y, int z, int w,
            cudaChannelFormatKind f);
    extern cudaError_t cudaGetChannelDesc(cudaChannelFormatDesc *desc,
            const cudaArray *array);
    extern cudaError_t cudaGetTextureAlignmentOffset(size_t *offset,
            const textureReference *texref);
    extern cudaError_t cudaGetTextureReference(const textureReference **texref,
            const char *symbol);
    extern cudaError_t cudaUnbindTexture(const textureReference * texref);

    /* CudaRt_thread */
    extern cudaError_t cudaThreadSynchronize();
    extern cudaError_t cudaThreadExit();

    /* CudaRt_version */
    extern cudaError_t cudaDriverGetVersion(int *driverVersion);
    extern cudaError_t cudaRuntimeGetVersion(int *runtimeVersion);
}

#endif	/* _CUDART_H */

