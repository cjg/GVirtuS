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

#define __dv(v)

class CudaRt {
public:
    static char * MarshalDevicePointer(void *devPtr);
    static void MarshalDevicePointer(void *devPtr, char * marshal);
};

extern "C" {
    /*
     Routines not found in the cuda's header files.
     KEEP THEM WITH CARE
     */

    void** __cudaRegisterFatBinary(void *fatCubin);
    void __cudaUnregisterFatBinary(void **fatCubinHandle);
    void __cudaRegisterFunction(void **fatCubinHandle, const char *hostFun, char *deviceFun,
            const char *deviceName, int thread_limit, uint3 *tid,
            uint3 *bid, dim3 *bDim, dim3 *gDim, int *wSize);
    void __cudaRegisterVar(void **fatCubinHandle, char *hostVar, char *deviceAddress,
            const char *deviceName, int ext, int size, int constant,
            int global);

    void __cudaRegisterShared(void **fatCubinHandle, void **devicePtr);
    void __cudaRegisterSharedVar(void **fatCubinHandle, void **devicePtr, size_t size,
            size_t alignment, int storage);

    /* */

    extern int CUDARTAPI __cudaSynchronizeThreads(void**, void*);
    extern void CUDARTAPI __cudaTextureFetch(const void *tex, void *index, int integer, void *val);
    extern __host__ cudaError_t CUDARTAPI cudaBindTexture(size_t *offset, const struct textureReference *texref, const void *devPtr, const struct cudaChannelFormatDesc *desc, size_t size __dv(UINT_MAX));
    extern __host__ cudaError_t CUDARTAPI cudaBindTextureToArray(const struct textureReference *texref, const struct cudaArray *array, const struct cudaChannelFormatDesc *desc);
    extern __host__ cudaError_t CUDARTAPI cudaChooseDevice(int *device, const struct cudaDeviceProp *prop);
    extern __host__ cudaError_t CUDARTAPI cudaConfigureCall(dim3 gridDim, dim3 blockDim, size_t sharedMem __dv(0), cudaStream_t stream __dv(0));
    extern __host__ struct cudaChannelFormatDesc CUDARTAPI cudaCreateChannelDesc(int x, int y, int z, int w, enum cudaChannelFormatKind f);
    extern __host__ cudaError_t CUDARTAPI cudaEventCreate(cudaEvent_t *event);
    extern __host__ cudaError_t CUDARTAPI cudaEventDestroy(cudaEvent_t event);
    extern __host__ cudaError_t CUDARTAPI cudaEventElapsedTime(float *ms, cudaEvent_t start, cudaEvent_t end);
    extern __host__ cudaError_t CUDARTAPI cudaEventQuery(cudaEvent_t event);
    extern __host__ cudaError_t CUDARTAPI cudaEventRecord(cudaEvent_t event, cudaStream_t stream);
    extern __host__ cudaError_t CUDARTAPI cudaEventSynchronize(cudaEvent_t event);
    extern __host__ cudaError_t CUDARTAPI cudaFree(void *devPtr);
    extern __host__ cudaError_t CUDARTAPI cudaFreeArray(struct cudaArray *array);
    extern __host__ cudaError_t CUDARTAPI cudaFreeHost(void *ptr);
    extern __host__ cudaError_t CUDARTAPI cudaGLMapBufferObject(void **devPtr, GLuint bufObj);
    extern __host__ cudaError_t CUDARTAPI cudaGLRegisterBufferObject(GLuint bufObj);
    extern __host__ cudaError_t CUDARTAPI cudaGLSetGLDevice(int device);
    extern __host__ cudaError_t CUDARTAPI cudaGLUnmapBufferObject(GLuint bufObj);
    extern __host__ cudaError_t CUDARTAPI cudaGLUnregisterBufferObject(GLuint bufObj);
    extern __host__ cudaError_t CUDARTAPI cudaGetChannelDesc(struct cudaChannelFormatDesc *desc, const struct cudaArray *array);
    extern __host__ cudaError_t CUDARTAPI cudaGetDevice(int *device);
    extern __host__ cudaError_t CUDARTAPI cudaGetDeviceCount(int *count);
    extern __host__ cudaError_t CUDARTAPI cudaGetDeviceProperties(struct cudaDeviceProp *prop, int device);
    extern __host__ const char* CUDARTAPI cudaGetErrorString(cudaError_t error);
    extern __host__ cudaError_t CUDARTAPI cudaGetLastError(void);
    extern __host__ cudaError_t CUDARTAPI cudaGetSymbolAddress(void **devPtr, const char *symbol);
    extern __host__ cudaError_t CUDARTAPI cudaGetSymbolSize(size_t *size, const char *symbol);
    extern __host__ cudaError_t CUDARTAPI cudaGetTextureAlignmentOffset(size_t *offset, const struct textureReference *texref);
    extern __host__ cudaError_t CUDARTAPI cudaGetTextureReference(const struct textureReference **texref, const char *symbol);
    extern __host__ cudaError_t CUDARTAPI cudaLaunch(const char *symbol);
    extern __host__ cudaError_t CUDARTAPI cudaMalloc(void **devPtr, size_t size);
    extern __host__ cudaError_t CUDARTAPI cudaMalloc3D(struct cudaPitchedPtr* pitchDevPtr, struct cudaExtent extent);
    extern __host__ cudaError_t CUDARTAPI cudaMalloc3DArray(struct cudaArray** arrayPtr, const struct cudaChannelFormatDesc* desc, struct cudaExtent extent);
    extern __host__ cudaError_t CUDARTAPI cudaMallocArray(struct cudaArray **array, const struct cudaChannelFormatDesc *desc, size_t width, size_t height __dv(1));
    extern __host__ cudaError_t CUDARTAPI cudaMallocHost(void **ptr, size_t size);
    extern __host__ cudaError_t CUDARTAPI cudaMallocPitch(void **devPtr, size_t *pitch, size_t width, size_t height);
    extern __host__ cudaError_t CUDARTAPI cudaMemcpy(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind);
    extern __host__ cudaError_t CUDARTAPI cudaMemcpy2D(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind);
    extern __host__ cudaError_t CUDARTAPI cudaMemcpy2DArrayToArray(struct cudaArray *dst, size_t wOffsetDst, size_t hOffsetDst, const struct cudaArray *src, size_t wOffsetSrc, size_t hOffsetSrc, size_t width, size_t height, enum cudaMemcpyKind kind __dv(cudaMemcpyDeviceToDevice));
    extern __host__ cudaError_t CUDARTAPI cudaMemcpy2DAsync(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind, cudaStream_t stream);
    extern __host__ cudaError_t CUDARTAPI cudaMemcpy2DFromArray(void *dst, size_t dpitch, const struct cudaArray *src, size_t wOffset, size_t hOffset, size_t width, size_t height, enum cudaMemcpyKind kind);
    extern __host__ cudaError_t CUDARTAPI cudaMemcpy2DFromArrayAsync(void *dst, size_t dpitch, const struct cudaArray *src, size_t wOffset, size_t hOffset, size_t width, size_t height, enum cudaMemcpyKind kind, cudaStream_t stream);
    extern __host__ cudaError_t CUDARTAPI cudaMemcpy2DToArray(struct cudaArray *dst, size_t wOffset, size_t hOffset, const void *src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind);
    extern __host__ cudaError_t CUDARTAPI cudaMemcpy2DToArrayAsync(struct cudaArray *dst, size_t wOffset, size_t hOffset, const void *src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind, cudaStream_t stream);
    extern __host__ cudaError_t CUDARTAPI cudaMemcpy3D(const struct cudaMemcpy3DParms *p);
    extern __host__ cudaError_t CUDARTAPI cudaMemcpy3DAsync(const struct cudaMemcpy3DParms *p, cudaStream_t stream);
    extern __host__ cudaError_t CUDARTAPI cudaMemcpyArrayToArray(struct cudaArray *dst, size_t wOffsetDst, size_t hOffsetDst, const struct cudaArray *src, size_t wOffsetSrc, size_t hOffsetSrc, size_t count, enum cudaMemcpyKind kind __dv(cudaMemcpyDeviceToDevice));
    extern __host__ cudaError_t CUDARTAPI cudaMemcpyAsync(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream);
    extern __host__ cudaError_t CUDARTAPI cudaMemcpyFromArray(void *dst, const struct cudaArray *src, size_t wOffset, size_t hOffset, size_t count, enum cudaMemcpyKind kind);
    extern __host__ cudaError_t CUDARTAPI cudaMemcpyFromArrayAsync(void *dst, const struct cudaArray *src, size_t wOffset, size_t hOffset, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream);
    extern __host__ cudaError_t CUDARTAPI cudaMemcpyFromSymbol(void *dst, const char *symbol, size_t count, size_t offset __dv(0), enum cudaMemcpyKind kind __dv(cudaMemcpyDeviceToHost));
    extern __host__ cudaError_t CUDARTAPI cudaMemcpyFromSymbolAsync(void *dst, const char *symbol, size_t count, size_t offset, enum cudaMemcpyKind kind, cudaStream_t stream);
    extern __host__ cudaError_t CUDARTAPI cudaMemcpyToArray(struct cudaArray *dst, size_t wOffset, size_t hOffset, const void *src, size_t count, enum cudaMemcpyKind kind);
    extern __host__ cudaError_t CUDARTAPI cudaMemcpyToArrayAsync(struct cudaArray *dst, size_t wOffset, size_t hOffset, const void *src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream);
    extern __host__ cudaError_t CUDARTAPI cudaMemcpyToSymbol(const char *symbol, const void *src, size_t count, size_t offset __dv(0), enum cudaMemcpyKind kind __dv(cudaMemcpyHostToDevice));
    extern __host__ cudaError_t CUDARTAPI cudaMemcpyToSymbolAsync(const char *symbol, const void *src, size_t count, size_t offset, enum cudaMemcpyKind kind, cudaStream_t stream);
    extern __host__ cudaError_t CUDARTAPI cudaMemset(void *mem, int c, size_t count);
    extern __host__ cudaError_t CUDARTAPI cudaMemset2D(void *mem, size_t pitch, int c, size_t width, size_t height);
    extern __host__ cudaError_t CUDARTAPI cudaMemset3D(struct cudaPitchedPtr pitchDevPtr, int value, struct cudaExtent extent);
    extern __host__ cudaError_t CUDARTAPI cudaSetDevice(int device);
    extern __host__ cudaError_t CUDARTAPI cudaSetDoubleForDevice(double *d);
    extern __host__ cudaError_t CUDARTAPI cudaSetDoubleForHost(double *d);
    extern __host__ cudaError_t CUDARTAPI cudaSetupArgument(const void *arg, size_t size, size_t offset);
    extern __host__ cudaError_t CUDARTAPI cudaStreamCreate(cudaStream_t *stream);
    extern __host__ cudaError_t CUDARTAPI cudaStreamDestroy(cudaStream_t stream);
    extern __host__ cudaError_t CUDARTAPI cudaStreamQuery(cudaStream_t stream);
    extern __host__ cudaError_t CUDARTAPI cudaStreamSynchronize(cudaStream_t stream);
    extern __host__ cudaError_t CUDARTAPI cudaThreadExit(void);
    extern __host__ cudaError_t CUDARTAPI cudaThreadSynchronize(void);
    extern __host__ cudaError_t CUDARTAPI cudaUnbindTexture(const struct textureReference *texref);
}

#endif	/* _CUDART_H */

