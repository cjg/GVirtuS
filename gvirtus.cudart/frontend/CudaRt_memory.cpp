/*
 * gVirtuS -- A GPGPU transparent virtualization component.
 *
 * Copyright (C) 2009-2010  The University of Napoli Parthenope at Naples.
 *
 * This file is part of gVirtuS.
 *
 * gVirtuS is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * gVirtuS is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with gVirtuS; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
 *
 * Written by: Giuseppe Coviello <giuseppe.coviello@uniparthenope.it>,
 *             Department of Applied Science
 */

#include <cstdio>
#include <string.h>
#include <algorithm>
#include "CudaRt.h"

#ifndef CUDART_VERSION
#error CUDART_VERSION not defined
#endif

using namespace std;

extern "C" __host__ cudaError_t CUDARTAPI cudaFree(void *devPtr) {
    void* p = devPtr;

    if (CudaRtFrontend::isMappedMemory(devPtr)) {
#ifdef DEBUG
        cerr << "Mapped pointer detected" << endl;
#endif
        mappedPointer a = CudaRtFrontend::getMappedPointer(devPtr);
        p = a.pointer;
        free(devPtr);
    }

    CudaRtFrontend::Prepare();
    CudaRtFrontend::AddDevicePointerForArguments(p);
    CudaRtFrontend::Execute("cudaFree");
    return CudaRtFrontend::GetExitCode();
}

extern "C" __host__ cudaError_t CUDARTAPI cudaFreeArray(cudaArray *array) {
    CudaRtFrontend::Prepare();
    CudaRtFrontend::AddDevicePointerForArguments((void *) array);
    CudaRtFrontend::Execute("cudaFreeArray");
    return CudaRtFrontend::GetExitCode();
}

extern "C" __host__ cudaError_t CUDARTAPI cudaFreeHost(void *ptr) {
#ifdef DEBUG
    printf("Requesting cudaFreeHost\n");
#endif
    free(ptr);
    return cudaSuccess;
}

extern "C" __host__ cudaError_t CUDARTAPI cudaGetSymbolAddress(void **devPtr, const void *symbol) {
    CudaRtFrontend::Prepare();
    // Achtung: skip adding devPtr
    CudaRtFrontend::AddSymbolForArguments((char*) symbol);
    CudaRtFrontend::Execute("cudaGetSymbolAddress");
    if (CudaRtFrontend::Success())
        *devPtr = CudaUtil::UnmarshalPointer(CudaRtFrontend::GetOutputString());
    return CudaRtFrontend::GetExitCode();
}

extern "C" __host__ cudaError_t CUDARTAPI cudaGetSymbolSize(size_t *size, const void *symbol) {
    CudaRtFrontend::Prepare();
    CudaRtFrontend::AddHostPointerForArguments(size);
    CudaRtFrontend::AddSymbolForArguments((char*) symbol);
    CudaRtFrontend::Execute("cudaGetSymbolSize");
    if (CudaRtFrontend::Success())
        *size = *(CudaRtFrontend::GetOutputHostPointer<size_t > ());
    return CudaRtFrontend::GetExitCode();
}

extern "C" __host__ cudaError_t CUDARTAPI cudaHostAlloc(void **ptr, size_t size, unsigned int flags) {
#ifdef DEBUG
    printf("Requesting cudaHostAlloc\n");
#endif
    // Achtung: we can't use host page-locked memory, so we use simple pageable
    // memory here.
    if ((*ptr = malloc(size)) == NULL)
        return cudaErrorMemoryAllocation;
    return cudaSuccess;
}

extern "C" __host__ cudaError_t CUDARTAPI cudaHostGetDevicePointer(void **pDevice, void *pHost,
        unsigned int flags) {
#ifdef DEBUG
    printf("Requesting cudaHostGetDevicePointer\n");
#endif
    // Achtung: we can't use mapped memory
    return cudaErrorMemoryAllocation;
}

extern "C" __host__ cudaError_t CUDARTAPI cudaHostGetFlags(unsigned int *pFlags, void *pHost) {
#ifdef DEBUG
    printf("Requesting cudaHostGetFlags\n");
#endif
    // Achtung: falling back to the simplest method because we can't map memory
#ifndef CUDA_VERSION
#error CUDA_VERSION not defined
#endif
#if CUDA_VERSION >= 2030
    *pFlags = cudaHostAllocDefault;
#endif
    return cudaSuccess;
}

extern "C" __host__ cudaError_t CUDARTAPI cudaMalloc(void **devPtr, size_t size) {
    CudaRtFrontend::Prepare();
    CudaRtFrontend::AddVariableForArguments(size);
    CudaRtFrontend::Execute("cudaMalloc");

    if (CudaRtFrontend::Success())
        *devPtr = CudaRtFrontend::GetOutputDevicePointer();

    return CudaRtFrontend::GetExitCode();
}

extern "C" __host__ cudaError_t CUDARTAPI cudaMalloc3D(cudaPitchedPtr *pitchedDevPtr,
        cudaExtent extent) {
    // FIXME: implement
    cerr << "*** Error: cudaMalloc3D() not yet implemented!" << endl;
    return cudaErrorUnknown;
}

//#if CUDART_VERSION >= 3000
//extern "C" __host__ cudaError_t CUDARTAPI cudaMalloc3DArray(cudaArray **array,
//        const cudaChannelFormatDesc *desc,struct cudaExtent extent, unsigned int flags) {
//#else
//extern "C" __host__ cudaError_t CUDARTAPI cudaMalloc3DArray(cudaArray **array,
//       const cudaChannelFormatDesc *desc, cudaExtent extent, unsigned int flags) {
//#endif

extern "C" __host__ cudaError_t cudaMalloc3DArray(cudaArray_t* array, const cudaChannelFormatDesc* desc, cudaExtent extent, unsigned int flags) {
    CudaRtFrontend::Prepare();
    CudaRtFrontend::AddHostPointerForArguments(desc);
    CudaRtFrontend::AddVariableForArguments(extent);
    CudaRtFrontend::AddVariableForArguments(flags);

#ifdef DEBUG
    printf("x %d\n", desc->x);
    printf("y %d\n", desc->y);
    printf("z %d\n", desc->z);
    printf("w %d\n", desc->w);
    printf("f %d\n", desc->f);

    printf("width %d\n", extent.width);
    printf("height %d\n", extent.height);
    printf("depth %d\n", extent.depth);

    printf("flags %d\n", flags);
#endif 
    
    CudaRtFrontend::Execute("cudaMalloc3DArray");
    if (CudaRtFrontend::Success())
        *array = *(CudaRtFrontend::GetOutputHostPointer<cudaArray_t>());
    //*array = (cudaArray_t) CudaRtFrontend::GetOutputDevicePointer();

//    printf("%x\n", *array);
    return CudaRtFrontend::GetExitCode();
}
// FIXME: new mapping way

#if CUDART_VERSION >= 3000

extern "C" __host__ cudaError_t CUDARTAPI cudaMallocArray(cudaArray **arrayPtr,
        const cudaChannelFormatDesc *desc, size_t width, size_t height,
        unsigned int flags) {
#else

extern "C" __host__ cudaError_t CUDARTAPI cudaMallocArray(cudaArray **arrayPtr,
        const cudaChannelFormatDesc *desc, size_t width, size_t height) {
#endif
    CudaRtFrontend::Prepare();

    CudaRtFrontend::AddHostPointerForArguments(desc);
    CudaRtFrontend::AddVariableForArguments(width);
    CudaRtFrontend::AddVariableForArguments(height);

#ifdef DEBUG
    printf("x %d\n", desc->x);
    printf("y %d\n", desc->y);
    printf("z %d\n", desc->z);
    printf("w %d\n", desc->w);
    printf("f %d\n", desc->f);

    printf("width %d\n", width);
    printf("height %d\n", height);
#endif    
    
    CudaRtFrontend::Execute("cudaMallocArray");
    if (CudaRtFrontend::Success())
        *arrayPtr = (cudaArray *) CudaRtFrontend::GetOutputDevicePointer();

//    printf("%x\n", *arrayPtr);
    return CudaRtFrontend::GetExitCode();
}

extern "C" __host__ cudaError_t CUDARTAPI cudaMallocHost(void **ptr, size_t size) {
    // Achtung: we can't use host page-locked memory, so we use simple pageable
    // memory here.
    if ((*ptr = malloc(size)) == NULL)
        return cudaErrorMemoryAllocation;
    return cudaSuccess;
}

extern "C" __host__ cudaError_t CUDARTAPI cudaMallocPitch(void **devPtr,
        size_t *pitch, size_t width, size_t height) {
    CudaRtFrontend::Prepare();

    CudaRtFrontend::AddVariableForArguments(*pitch);
    CudaRtFrontend::AddVariableForArguments(width);
    CudaRtFrontend::AddVariableForArguments(height);
    CudaRtFrontend::Execute("cudaMallocPitch");

    if (CudaRtFrontend::Success()) {
        *devPtr = CudaRtFrontend::GetOutputDevicePointer();
        *pitch = CudaRtFrontend::GetOutputVariable<size_t>();
    }
    return CudaRtFrontend::GetExitCode();
}

extern "C" __host__ cudaError_t CUDARTAPI cudaMemcpyPeerAsync(void *dst,
        int dstDevice, const void * src, int srcDevice, size_t count,
        cudaStream_t stream) {
    CudaRtFrontend::Prepare();
    CudaRtFrontend::AddDevicePointerForArguments(dst);
    CudaRtFrontend::AddVariableForArguments(dstDevice);
    CudaRtFrontend::AddDevicePointerForArguments(src);
    CudaRtFrontend::AddVariableForArguments(srcDevice);
    CudaRtFrontend::AddVariableForArguments(count);
    CudaRtFrontend::AddDevicePointerForArguments(stream);

    CudaRtFrontend::Execute("cudaMemcpyPeerAsync");

//    if (CudaRtFrontend::Success()) {
//        // *dst = CudaRtFrontend::GetOutputDevicePointer();
//        // *src = CudaRtFrontend::GetOutputDevicePointer();
//    }
    
    return CudaRtFrontend::GetExitCode();
}

extern "C" __host__ CUDARTAPI cudaError_t cudaMallocManaged(void ** devPtr,
        size_t size, unsigned flags) {
    void* hp = NULL;
    *devPtr = malloc(size);
    CudaRtFrontend::Prepare();

    CudaRtFrontend::AddVariableForArguments(size);
    CudaRtFrontend::Execute("cudaMalloc");

    if (CudaRtFrontend::Success()) {
        hp = CudaRtFrontend::GetOutputDevicePointer();
    }

    mappedPointer host;
    host.pointer = hp;
    host.size = size;

#ifdef DEBUG
    cerr << "device: " << std::hex << hp << " host: " << *devPtr << endl;
#endif
    
    CudaRtFrontend::addMappedPointer(*devPtr, host);
    return CudaRtFrontend::GetExitCode();
}

extern "C" __host__ cudaError_t CUDARTAPI cudaMemcpy3D(const cudaMemcpy3DParms* p) {
    
    CudaRtFrontend::Prepare();
    CudaRtFrontend::AddHostPointerForArguments(p);

    unsigned int width = p->extent.width;
    unsigned int num_faces = p->extent.depth;
    unsigned int num_layers = 1;
    unsigned int cubemap_size = width * width * num_faces;
    unsigned int size = cubemap_size * num_layers * (p->srcPtr.pitch / p->extent.width);
    //    unsigned int size = cubemap_size * num_layers * sizeof(float);
    //float *h_data = (float *) malloc(size);

    CudaRtFrontend::AddHostPointerForArguments<char>(static_cast<char *>
            (const_cast<void *> (p->srcPtr.ptr)), size);
    CudaRtFrontend::Execute("cudaMemcpy3D");
    if (CudaRtFrontend::Success()) {
        // memmove(p, CudaRtFrontend::GetOutputHostPointer<cudaMemcpy3DParms>(),
        //         sizeof(cudaMemcpy3DParms));
        //*p = *(CudaRtFrontend::GetOutputHostPointer<cudaMemcpy3DParms>());
        return CudaRtFrontend::GetExitCode();
    }
    return CudaRtFrontend::GetExitCode();
}

extern "C" __host__ cudaError_t CUDARTAPI cudaMemcpy(void *dst,
        const void *src, size_t count, cudaMemcpyKind kind) {
    CudaRtFrontend::Prepare();
    switch (kind) {
        case cudaMemcpyDefault:
            cerr << "MemCpyDefault" << endl;
            if (CudaRtFrontend::isDevicePointer(dst) && CudaRtFrontend::isDevicePointer(src)) {
                cerr << "Device2Device" << endl;
                return cudaMemcpy(dst, src, count, cudaMemcpyDeviceToDevice);
            } else if (!CudaRtFrontend::isDevicePointer(dst) && !CudaRtFrontend::isDevicePointer(src))
                return cudaMemcpy(dst, src, count, cudaMemcpyHostToHost);
            else if (!CudaRtFrontend::isDevicePointer(dst) && CudaRtFrontend::isDevicePointer(src))
                return cudaMemcpy(dst, src, count, cudaMemcpyDeviceToHost);
            else if (CudaRtFrontend::isDevicePointer(dst) && !CudaRtFrontend::isDevicePointer(src))
                return cudaMemcpy(dst, src, count, cudaMemcpyHostToDevice);
        case cudaMemcpyHostToHost:
            /* NOTE: no communication is performed, because it's just overhead
             * here */
            if (memmove(dst, src, count) == NULL)
                return cudaErrorInvalidValue;
            return cudaSuccess;
            break;
        case cudaMemcpyHostToDevice:
            CudaRtFrontend::AddDevicePointerForArguments(dst);
            CudaRtFrontend::AddHostPointerForArguments<char>(static_cast<char *>
                    (const_cast<void *> (src)), count);
            CudaRtFrontend::AddVariableForArguments(count);
            CudaRtFrontend::AddVariableForArguments(kind);
            CudaRtFrontend::Execute("cudaMemcpy");
            break;
        case cudaMemcpyDeviceToHost:
            /* NOTE: adding a fake host pointer */
            CudaRtFrontend::AddHostPointerForArguments("");
            CudaRtFrontend::AddDevicePointerForArguments(src);
            CudaRtFrontend::AddVariableForArguments(count);
            CudaRtFrontend::AddVariableForArguments(kind);
            CudaRtFrontend::Execute("cudaMemcpy");
            if (CudaRtFrontend::Success())
                memmove(dst, CudaRtFrontend::GetOutputHostPointer<char>(count), count);
            break;
        case cudaMemcpyDeviceToDevice:
            CudaRtFrontend::AddDevicePointerForArguments(dst);
            CudaRtFrontend::AddDevicePointerForArguments(src);
            CudaRtFrontend::AddVariableForArguments(count);
            CudaRtFrontend::AddVariableForArguments(kind);
            CudaRtFrontend::Execute("cudaMemcpy");
            break;
    }

    return CudaRtFrontend::GetExitCode();
}

extern "C" __host__ cudaError_t CUDARTAPI cudaMemcpy2D(void *dst, size_t dpitch, const void *src,
        size_t spitch, size_t width, size_t height, cudaMemcpyKind kind) {
    CudaRtFrontend::Prepare();
    switch (kind) {
        case cudaMemcpyDefault:
        case cudaMemcpyHostToHost:
            /* NOTE: no communication is performed, because it's just overhead
             * here */
            if (dpitch < width)
                return cudaErrorInvalidValue;

            if (dpitch == spitch) {
                if (memcpy(dst, src, spitch * height) == NULL)
                    return cudaErrorInvalidValue;
            } else {
                for (int i = 0; i < height; i++)
                    if (memcpy(dst + (dpitch * i), src + (spitch * i), width)
                            == NULL)
                        return cudaErrorInvalidValue;
            }
            return cudaSuccess;
            break;
        case cudaMemcpyHostToDevice:
            CudaRtFrontend::AddDevicePointerForArguments(dst);
            CudaRtFrontend::AddHostPointerForArguments<char>(static_cast<char *>
                    (const_cast<void *> (src)), spitch * height);
            CudaRtFrontend::AddVariableForArguments(dpitch);
            CudaRtFrontend::AddVariableForArguments(spitch);
            CudaRtFrontend::AddVariableForArguments(width);
            CudaRtFrontend::AddVariableForArguments(height);
            CudaRtFrontend::AddVariableForArguments(kind);
            CudaRtFrontend::Execute("cudaMemcpy2D");
            break;
        case cudaMemcpyDeviceToHost:
            /* NOTE: adding a fake host pointer */
            CudaRtFrontend::AddHostPointerForArguments("");
            CudaRtFrontend::AddDevicePointerForArguments(src);
            CudaRtFrontend::AddVariableForArguments(dpitch);
            CudaRtFrontend::AddVariableForArguments(spitch);
            CudaRtFrontend::AddVariableForArguments(width);
            CudaRtFrontend::AddVariableForArguments(height);
            CudaRtFrontend::AddVariableForArguments(kind);
            CudaRtFrontend::Execute("cudaMemcpy2D");
            if (CudaRtFrontend::Success())
                memmove(dst, CudaRtFrontend::GetOutputHostPointer<char>(dpitch * height),
                    dpitch * height);
            break;
        case cudaMemcpyDeviceToDevice:
            CudaRtFrontend::AddDevicePointerForArguments(dst);
            CudaRtFrontend::AddDevicePointerForArguments(src);
            CudaRtFrontend::AddVariableForArguments(dpitch);
            CudaRtFrontend::AddVariableForArguments(spitch);
            CudaRtFrontend::AddVariableForArguments(width);
            CudaRtFrontend::AddVariableForArguments(height);
            CudaRtFrontend::AddVariableForArguments(kind);
            CudaRtFrontend::Execute("cudaMemcpy2D");
            break;
    }

    return CudaRtFrontend::GetExitCode();
}

extern "C" __host__ cudaError_t CUDARTAPI cudaMemcpy2DArrayToArray(cudaArray *dst, size_t wOffsetDst,
        size_t hOffsetDst, const cudaArray *src, size_t wOffsetSrc,
        size_t hOffsetSrc, size_t width, size_t height, cudaMemcpyKind kind) {
    // FIXME: implement
    cerr << "*** Error: cudaMemcpy2DArrayToArray() not yet implemented!"
            << endl;
    return cudaErrorUnknown;
}

extern "C" __host__ cudaError_t CUDARTAPI cudaMemcpy2DAsync(void *dst, size_t dpitch, const void *src,
        size_t spitch, size_t width, size_t height, cudaMemcpyKind kind,
        cudaStream_t stream) {
    // FIXME: implement
    cerr << "*** Error: cudaMemcpy2DAsync() not yet implemented!" << endl;
    return cudaErrorUnknown;
}

extern "C" __host__ cudaError_t CUDARTAPI cudaMemcpy2DFromArray(void *dst, size_t dpitch,
        const cudaArray *src, size_t wOffset, size_t hOffset, size_t width,
        size_t height, cudaMemcpyKind kind) {

    CudaRtFrontend::Prepare();

    switch (kind) {
        case cudaMemcpyDefault:
        case cudaMemcpyHostToHost:
        case cudaMemcpyHostToDevice:
            /* This should never happen. */
            return cudaErrorInvalidMemcpyDirection;
            break;

        case cudaMemcpyDeviceToHost:
            //pass contenuto source 
            CudaRtFrontend::AddHostPointerForArguments("");
            CudaRtFrontend::AddVariableForArguments(dpitch);
            CudaRtFrontend::AddDevicePointerForArguments(src);
            CudaRtFrontend::AddVariableForArguments(wOffset);
            CudaRtFrontend::AddVariableForArguments(hOffset);
            CudaRtFrontend::AddVariableForArguments(width);
            CudaRtFrontend::AddVariableForArguments(height);
            CudaRtFrontend::AddVariableForArguments(kind);
            CudaRtFrontend::Execute("cudaMemcpy2DFromArray");
            if (CudaRtFrontend::Success())
                memmove(dst, CudaRtFrontend::GetOutputHostPointer<char>(dpitch * height),
                    dpitch * height);
            break;
        case cudaMemcpyDeviceToDevice:
            CudaRtFrontend::AddDevicePointerForArguments(dst);
            CudaRtFrontend::AddVariableForArguments(dpitch);
            CudaRtFrontend::AddDevicePointerForArguments(src);
            CudaRtFrontend::AddVariableForArguments(wOffset);
            CudaRtFrontend::AddVariableForArguments(hOffset);
            CudaRtFrontend::AddVariableForArguments(width);
            CudaRtFrontend::AddVariableForArguments(height);
            CudaRtFrontend::AddVariableForArguments(kind);
            CudaRtFrontend::Execute("cudaMemcpy2DFromArray");
            break;
    }

    return CudaRtFrontend::GetExitCode();
}

extern "C" __host__ cudaError_t CUDARTAPI cudaMemcpy2DFromArrayAsync(void *dst, size_t dpitch,
        const cudaArray *src, size_t wOffset, size_t hOffset, size_t width,
        size_t height, cudaMemcpyKind kind, cudaStream_t stream) {
    // FIXME: implement
    cerr << "*** Error: cudaMemcpy2DFromArrayAsync() not yet implemented!" << endl;
    return cudaErrorUnknown;
}

extern "C" __host__ cudaError_t CUDARTAPI cudaMemcpy2DToArray(cudaArray *dst,
        size_t wOffset, size_t hOffset, const void *src, size_t spitch,
        size_t width, size_t height, cudaMemcpyKind kind) {
    CudaRtFrontend::Prepare();

    switch (kind) {
        case cudaMemcpyDefault:
        case cudaMemcpyHostToHost:
        case cudaMemcpyDeviceToHost:
            /* This should never happen. */
            return cudaErrorInvalidMemcpyDirection;

        case cudaMemcpyHostToDevice:
            //pass contenuto source 
            CudaRtFrontend::AddDevicePointerForArguments(dst);
            CudaRtFrontend::AddVariableForArguments(wOffset);
            CudaRtFrontend::AddVariableForArguments(hOffset);            
            CudaRtFrontend::AddHostPointerForArguments<char>(static_cast<char *>
                    (const_cast<void *> (src)), spitch * height);            
            CudaRtFrontend::AddVariableForArguments(spitch);
            CudaRtFrontend::AddVariableForArguments(width);
            CudaRtFrontend::AddVariableForArguments(height);
            CudaRtFrontend::AddVariableForArguments(kind);            
            break;
        case cudaMemcpyDeviceToDevice:
            CudaRtFrontend::AddDevicePointerForArguments(dst);
            CudaRtFrontend::AddVariableForArguments(wOffset);
            CudaRtFrontend::AddVariableForArguments(hOffset);
            CudaRtFrontend::AddDevicePointerForArguments(src);            
            CudaRtFrontend::AddVariableForArguments(spitch);
            CudaRtFrontend::AddVariableForArguments(width);
            CudaRtFrontend::AddVariableForArguments(height);
            CudaRtFrontend::AddVariableForArguments(kind);            
            break;
    }
    CudaRtFrontend::Execute("cudaMemcpy2DToArray");
    return CudaRtFrontend::GetExitCode();
}

extern "C" __host__ cudaError_t CUDARTAPI cudaMemcpy2DToArrayAsync(cudaArray *dst, size_t wOffset,
        size_t hOffset, const void *src, size_t spitch, size_t width,
        size_t height, cudaMemcpyKind kind, cudaStream_t stream) {
    // FIXME: implement
    cerr << "*** Error: cudaMemcpy2DToArrayAsync() not yet implemented!" << endl;
    return cudaErrorUnknown;
}

extern "C" __host__ cudaError_t CUDARTAPI cudaMemcpy3DAsync(const cudaMemcpy3DParms *p,
        cudaStream_t stream) {
    // FIXME: implement
    cerr << "*** Error: cudaMemcpy3DAsync() not yet implemented!" << endl;
    return cudaErrorUnknown;
}

extern "C" __host__ cudaError_t CUDARTAPI cudaMemcpyAsync(void *dst, const void *src, size_t count,
        cudaMemcpyKind kind, cudaStream_t stream) {
    CudaRtFrontend::Prepare();
    switch (kind) {
        case cudaMemcpyDefault:
        case cudaMemcpyHostToHost:
            /* NOTE: no communication is performed, because it's just overhead
             * here */
            CudaRtFrontend::AddHostPointerForArguments("");
            CudaRtFrontend::AddHostPointerForArguments("");
            CudaRtFrontend::AddVariableForArguments(kind);
#if CUDART_VERSION >= 3010
            CudaRtFrontend::AddDevicePointerForArguments(stream);
#else
            CudaRtFrontend::AddVariableForArguments(stream);
#endif
            CudaRtFrontend::Execute("cudaMemcpyAsync");
            if (memmove(dst, src, count) == NULL)
                return cudaErrorInvalidValue;
            return cudaSuccess;
            break;
        case cudaMemcpyHostToDevice:
            CudaRtFrontend::AddDevicePointerForArguments(dst);
            CudaRtFrontend::AddHostPointerForArguments<char>(static_cast<char *>
                    (const_cast<void *> (src)), count);
            CudaRtFrontend::AddVariableForArguments(count);
            CudaRtFrontend::AddVariableForArguments(kind);
#if CUDART_VERSION >= 3010
            CudaRtFrontend::AddDevicePointerForArguments(stream);
#else
            CudaRtFrontend::AddVariableForArguments(stream);
#endif
            CudaRtFrontend::Execute("cudaMemcpyAsync");
            break;
        case cudaMemcpyDeviceToHost:
            /* NOTE: adding a fake host pointer */
            CudaRtFrontend::AddHostPointerForArguments("");
            CudaRtFrontend::AddDevicePointerForArguments(src);
            CudaRtFrontend::AddVariableForArguments(count);
            CudaRtFrontend::AddVariableForArguments(kind);
#if CUDART_VERSION >= 3010
            CudaRtFrontend::AddDevicePointerForArguments(stream);
#else
            CudaRtFrontend::AddVariableForArguments(stream);
#endif
            CudaRtFrontend::Execute("cudaMemcpyAsync");
            if (CudaRtFrontend::Success())
                memmove(dst, CudaRtFrontend::GetOutputHostPointer<char>(count), count);
            break;
        case cudaMemcpyDeviceToDevice:
            CudaRtFrontend::AddDevicePointerForArguments(dst);
            CudaRtFrontend::AddDevicePointerForArguments(src);
            CudaRtFrontend::AddVariableForArguments(count);
            CudaRtFrontend::AddVariableForArguments(kind);
#if CUDART_VERSION >= 3010
            CudaRtFrontend::AddDevicePointerForArguments(stream);
#else
            CudaRtFrontend::AddVariableForArguments(stream);
#endif
            CudaRtFrontend::Execute("cudaMemcpyAsync");
            break;
    }

    return CudaRtFrontend::GetExitCode();
}

extern "C" __host__ cudaError_t CUDARTAPI cudaMemcpyFromArray(void *dst, const cudaArray *src,
        size_t wOffset, size_t hOffset, size_t count, cudaMemcpyKind kind) {
    CudaRtFrontend::Prepare();

    switch (kind) {
        case cudaMemcpyDefault:
        case cudaMemcpyHostToDevice:
        case cudaMemcpyHostToHost:
            /* This should never happen. cudaArray is only on device */
            return cudaErrorInvalidMemcpyDirection;
            break;
        case cudaMemcpyDeviceToHost:
            //pass contenuto source 
            CudaRtFrontend::AddHostPointerForArguments("");
            CudaRtFrontend::AddDevicePointerForArguments(src);
            CudaRtFrontend::AddVariableForArguments(wOffset);
            CudaRtFrontend::AddVariableForArguments(hOffset);
            CudaRtFrontend::AddVariableForArguments(count);
            CudaRtFrontend::AddVariableForArguments(kind);
            CudaRtFrontend::Execute("cudaMemcpyFromArray");
            if (CudaRtFrontend::Success())
                memmove(dst, CudaRtFrontend::GetOutputHostPointer<char>(count), count);
            break;
        case cudaMemcpyDeviceToDevice:
            CudaRtFrontend::AddDevicePointerForArguments(dst);
            CudaRtFrontend::AddDevicePointerForArguments(src);
            CudaRtFrontend::AddVariableForArguments(wOffset);
            CudaRtFrontend::AddVariableForArguments(hOffset);
            CudaRtFrontend::AddVariableForArguments(count);
            CudaRtFrontend::AddVariableForArguments(kind);
            CudaRtFrontend::Execute("cudaMemcpyFromArray");
            break;
    }

    return CudaRtFrontend::GetExitCode();
}

extern "C" __host__ cudaError_t CUDARTAPI cudaMemcpyArrayToArray(cudaArray *dst, size_t wOffsetDst, size_t hOffsetDst, const cudaArray *src,
        size_t wOffsetSrc, size_t hOffsetSrc, size_t count, cudaMemcpyKind kind) {
    CudaRtFrontend::Prepare();

    switch (kind) {
        case cudaMemcpyDefault:
        case cudaMemcpyHostToHost:
        case cudaMemcpyDeviceToHost:
        case cudaMemcpyHostToDevice:

            /* This should never happen. cudaArray is only on device */
            return cudaErrorInvalidMemcpyDirection;
            break;


        case cudaMemcpyDeviceToDevice:
            CudaRtFrontend::AddDevicePointerForArguments(dst);
            CudaRtFrontend::AddVariableForArguments(wOffsetDst);
            CudaRtFrontend::AddVariableForArguments(hOffsetDst);
            CudaRtFrontend::AddDevicePointerForArguments(src);
            CudaRtFrontend::AddVariableForArguments(wOffsetSrc);
            CudaRtFrontend::AddVariableForArguments(hOffsetSrc);
            CudaRtFrontend::AddVariableForArguments(count);
            CudaRtFrontend::AddVariableForArguments(kind);
            CudaRtFrontend::Execute("cudaMemcpyArrayToArray");
            break;
    }

    return CudaRtFrontend::GetExitCode();
}

extern "C" __host__ cudaError_t CUDARTAPI cudaMemcpyFromArrayAsync(void *dst, const cudaArray *src,
        size_t wOffset, size_t hOffset, size_t count, cudaMemcpyKind kind,
        cudaStream_t stream) {
    // FIXME: implement
    cerr << "*** Error: cudaMemcpyFromArrayAsync() not yet implemented!"
            << endl;
    return cudaErrorUnknown;
}

extern "C" __host__ cudaError_t CUDARTAPI cudaMemcpyFromSymbol(void *dst, const void *symbol,
        size_t count, size_t offset,
        cudaMemcpyKind kind) {
    CudaRtFrontend::Prepare();
    switch (kind) {
        case cudaMemcpyDefault:
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
            CudaRtFrontend::AddDevicePointerForArguments((void *) 0x666);
            // Achtung: passing the address and the content of symbol
            CudaRtFrontend::AddStringForArguments(CudaUtil::MarshalHostPointer(symbol));
            CudaRtFrontend::AddStringForArguments((char*) symbol);
            CudaRtFrontend::AddVariableForArguments(count);
            CudaRtFrontend::AddVariableForArguments(offset);
            CudaRtFrontend::AddVariableForArguments(kind);
            CudaRtFrontend::Execute("cudaMemcpyFromSymbol");
            if (CudaRtFrontend::Success())
                memmove(dst, CudaRtFrontend::GetOutputHostPointer<char>(count), count);
            break;
        case cudaMemcpyDeviceToDevice:
            CudaRtFrontend::AddDevicePointerForArguments(dst);
            // Achtung: passing the address and the content of symbol
            CudaRtFrontend::AddStringForArguments(CudaUtil::MarshalHostPointer(symbol));
            CudaRtFrontend::AddStringForArguments((char*) symbol);
            CudaRtFrontend::AddVariableForArguments(count);
            CudaRtFrontend::AddVariableForArguments(offset);
            CudaRtFrontend::AddVariableForArguments(kind);
            CudaRtFrontend::Execute("cudaMemcpyFromSymbol");
            break;
    }

    return CudaRtFrontend::GetExitCode();
}

extern "C" __host__ cudaError_t CUDARTAPI cudaMemcpyFromSymbolAsync(void *dst, const void *symbol,
        size_t count, size_t offset, cudaMemcpyKind kind, cudaStream_t stream) {
    // FIXME: implement
    cerr << "*** Error: cudaMemcpyFromSymbolAsync() not yet implemented!"
            << endl;
    return cudaErrorUnknown;
}

extern "C" __host__ cudaError_t CUDARTAPI cudaMemcpyToArray(cudaArray *dst, size_t wOffset,
        size_t hOffset, const void *src, size_t count, cudaMemcpyKind kind) {
    CudaRtFrontend::Prepare();
    switch (kind) {

        case cudaMemcpyDefault:
        case cudaMemcpyDeviceToHost:
        case cudaMemcpyHostToHost:
            /* This should never happen. */
            return cudaErrorInvalidMemcpyDirection;
            break;
        case cudaMemcpyHostToDevice:
            CudaRtFrontend::AddDevicePointerForArguments((void *) dst);
            CudaRtFrontend::AddVariableForArguments(wOffset);
            CudaRtFrontend::AddVariableForArguments(hOffset);
            CudaRtFrontend::AddHostPointerForArguments<char>(static_cast<char *>
                    (const_cast<void *> (src)), count);
            CudaRtFrontend::AddVariableForArguments(count);
            CudaRtFrontend::AddVariableForArguments(kind);
            CudaRtFrontend::Execute("cudaMemcpyToArray");
            break;
        case cudaMemcpyDeviceToDevice:
            CudaRtFrontend::AddDevicePointerForArguments((void *) dst);
            CudaRtFrontend::AddVariableForArguments(wOffset);
            CudaRtFrontend::AddVariableForArguments(hOffset);
            CudaRtFrontend::AddDevicePointerForArguments(src);
            CudaRtFrontend::AddVariableForArguments(count);
            CudaRtFrontend::AddVariableForArguments(kind);
            CudaRtFrontend::Execute("cudaMemcpyToArray");
            break;
    }

    return CudaRtFrontend::GetExitCode();
}

extern "C" __host__ cudaError_t CUDARTAPI cudaMemcpyToArrayAsync(cudaArray *dst, size_t wOffset,
        size_t hOffset, const void *src, size_t count, cudaMemcpyKind kind,
        cudaStream_t stream) {
    // FIXME: implement
    cerr << "*** Error: cudaMemcpyToArrayAsync() not yet implemented!" << endl;
    return cudaErrorUnknown;
}

extern "C" __host__ cudaError_t CUDARTAPI cudaMemcpyToSymbol(const void *symbol, const void *src,
        size_t count, size_t offset,
        cudaMemcpyKind kind) {
    CudaRtFrontend::Prepare();
    switch (kind) {
        case cudaMemcpyDefault:
        case cudaMemcpyHostToHost:
            /* This should never happen. */
            return cudaErrorInvalidMemcpyDirection;
            break;
        case cudaMemcpyHostToDevice:
            // Achtung: passing the address and the content of symbol
            CudaRtFrontend::AddStringForArguments(CudaUtil::MarshalHostPointer(symbol));
            CudaRtFrontend::AddStringForArguments((char*) symbol);
            CudaRtFrontend::AddHostPointerForArguments<char>(static_cast<char *>
                    (const_cast<void *> (src)), count);
            CudaRtFrontend::AddVariableForArguments(count);
            CudaRtFrontend::AddVariableForArguments(offset);
            CudaRtFrontend::AddVariableForArguments(kind);
            CudaRtFrontend::Execute("cudaMemcpyToSymbol");
            break;
        case cudaMemcpyDeviceToHost:
            /* This should never happen. */
            return cudaErrorInvalidMemcpyDirection;
            break;
        case cudaMemcpyDeviceToDevice:
            // Achtung: passing the address and the content of symbol
            CudaRtFrontend::AddStringForArguments(CudaUtil::MarshalHostPointer(symbol));
            CudaRtFrontend::AddStringForArguments((char*) symbol);
            CudaRtFrontend::AddDevicePointerForArguments(src);
            CudaRtFrontend::AddVariableForArguments(count);
            CudaRtFrontend::AddVariableForArguments(kind);
            CudaRtFrontend::Execute("cudaMemcpyToSymbol");
            break;
    }

    return CudaRtFrontend::GetExitCode();
}

extern "C" __host__ cudaError_t CUDARTAPI cudaMemcpyToSymbolAsync(const void *symbol, const void *src,
        size_t count, size_t offset, cudaMemcpyKind kind, cudaStream_t stream) {
    // FIXME: implement
    cerr << "*** Error: cudaMemcpyToSymbolAsync() not yet implemented!" << endl;
    return cudaErrorUnknown;
}

extern "C" __host__ cudaError_t CUDARTAPI cudaMemset(void *devPtr, int c, size_t count) {
    CudaRtFrontend::Prepare();
    CudaRtFrontend::AddDevicePointerForArguments(devPtr);
    CudaRtFrontend::AddVariableForArguments(c);
    CudaRtFrontend::AddVariableForArguments(count);
    CudaRtFrontend::Execute("cudaMemset");
    return CudaRtFrontend::GetExitCode();
}

extern "C" __host__ cudaError_t CUDARTAPI cudaMemsetAsync(void *devPtr, int c,
        size_t count, cudaStream_t stream) {
    CudaRtFrontend::Prepare();
    CudaRtFrontend::AddDevicePointerForArguments(devPtr);
    CudaRtFrontend::AddVariableForArguments(c);
    CudaRtFrontend::AddVariableForArguments(count);
    CudaRtFrontend::Execute("cudaMemset");
    return CudaRtFrontend::GetExitCode();
}

extern "C" __host__ cudaError_t CUDARTAPI cudaMemset2D(void *devPtr, size_t pitch, int value, size_t width,
        size_t height) {
    CudaRtFrontend::Prepare();
    CudaRtFrontend::AddDevicePointerForArguments(devPtr);
    CudaRtFrontend::AddVariableForArguments(pitch);
    CudaRtFrontend::AddVariableForArguments(value);
    CudaRtFrontend::AddVariableForArguments(width);
    CudaRtFrontend::AddVariableForArguments(height);
    CudaRtFrontend::Execute("cudaMemset2D");
    return CudaRtFrontend::GetExitCode();
}

extern "C" __host__ cudaError_t CUDARTAPI cudaMemset3D(cudaPitchedPtr pitchDevPtr, int value,
        cudaExtent extent) {
    // FIXME: implement
    cerr << "*** Error: cudaMemset3D() not yet implemented!" << endl;
    return cudaErrorUnknown;
}