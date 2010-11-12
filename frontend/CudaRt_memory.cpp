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

#include <cuda.h>
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
    Frontend *f = Frontend::GetFrontend();
    // Achtung: skip adding devPtr
    f->AddSymbolForArguments(symbol);
    f->Execute("cudaGetSymbolAddress");
    if(f->Success())
        *devPtr = CudaUtil::UnmarshalPointer(f->GetOutputString());
    return f->GetExitCode();
}

extern cudaError_t cudaGetSymbolSize(size_t *size, const char *symbol) {
    Frontend *f = Frontend::GetFrontend();
    f->AddHostPointerForArguments(size);
    f->AddSymbolForArguments(symbol);
    f->Execute("cudaGetSymbolSize");
    if(f->Success())
        *size = *(f->GetOutputHostPointer<size_t>());
    return f->GetExitCode();
}

extern cudaError_t cudaHostAlloc(void **ptr, size_t size, unsigned int flags) {
    // Achtung: we can't use host page-locked memory, so we use simple pageable
    // memory here.
    if ((*ptr = malloc(size)) == NULL)
        return cudaErrorMemoryAllocation;
    return cudaSuccess;
}

extern cudaError_t cudaHostGetDevicePointer(void **pDevice, void *pHost,
        unsigned int flags) {
    // Achtung: we can't use mapped memory
    return cudaErrorMemoryAllocation;
}

extern cudaError_t cudaHostGetFlags(unsigned int *pFlags, void *pHost) {
    // Achtung: falling back to the simplest method because we can't map memory
#ifndef CUDA_VERSION
#error CUDA_VERSION not defined
#endif
#if CUDA_VERSION >= 2030
    *pFlags = cudaHostAllocDefault;
#endif
    return cudaSuccess;
}

extern cudaError_t cudaMalloc(void **devPtr, size_t size) {
    Frontend *f = Frontend::GetFrontend();

    f->AddVariableForArguments(size);
    f->Execute("cudaMalloc");

    if(f->Success())
        *devPtr = f->GetOutputDevicePointer();

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

// FIXME: new mapping way
extern cudaError_t cudaMallocArray(cudaArray **arrayPtr,
        const cudaChannelFormatDesc *desc, size_t width, size_t height) {
    Frontend *f = Frontend::GetFrontend();

    f->AddHostPointerForArguments(desc);
    f->AddVariableForArguments(width);
    f->AddVariableForArguments(height);
    f->Execute("cudaMallocArray");
    if(f->Success())
        *arrayPtr = (cudaArray *) f->GetOutputDevicePointer();
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
    Frontend *f = Frontend::GetFrontend();

    f->AddHostPointerForArguments(pitch);
    f->AddVariableForArguments(width);
    f->AddVariableForArguments(height);
    f->Execute("cudaMallocPitch");

    if(f->Success()) {
        *devPtr = f->GetOutputDevicePointer();
        *pitch = *(f->GetOutputHostPointer<size_t>());
    }
    return f->GetExitCode();
}

extern cudaError_t cudaMemcpy(void *dst, const void *src, size_t count,
        cudaMemcpyKind kind) {
    Frontend *f = Frontend::GetFrontend();
    Communicator *c;
    switch (kind) {
        case cudaMemcpyHostToHost:
            /* NOTE: no communication is performed, because it's just overhead
             * here */
            if (memmove(dst, src, count) == NULL)
                return cudaErrorInvalidValue;
            return cudaSuccess;
            break;
        case cudaMemcpyHostToDevice:
            c = f->GetCommunicator();
            if(c->HasSharedMemory()) {
                size_t offset = 0;
                size_t copyied = 0;
                size_t chunk_size = c->GetSharedMemorySize();
                void *shm = c->GetSharedMemory();

                while(copyied < count) {
                    if(chunk_size > count - copyied)
                        chunk_size = count - copyied;
                    f->Prepare();
                    f->AddDevicePointerForArguments(static_cast<char *>(dst) + offset);
                    f->AddVariableForArguments(chunk_size);
                    f->AddVariableForArguments(kind);
                    memmove(shm, static_cast<char *>(const_cast<void *>(src)) + offset,
                            chunk_size);
                    f->Execute("cudaMemcpy");
                    offset += chunk_size;
                    copyied += chunk_size;
                }
            } else {
                f->AddDevicePointerForArguments(dst);
                f->AddHostPointerForArguments<char>(static_cast<char *>
                        (const_cast<void *> (src)), count);
                f->AddVariableForArguments(count);
                f->AddVariableForArguments(kind);
                f->Execute("cudaMemcpy");
            }
            break;
        case cudaMemcpyDeviceToHost:
            c = f->GetCommunicator();
            if(c->HasSharedMemory()) {
                size_t offset = 0;
                size_t copyied = 0;
                size_t chunk_size = c->GetSharedMemorySize();
                void *shm = c->GetSharedMemory();

                while(copyied < count) {
                    if(chunk_size > count - copyied)
                        chunk_size = count - copyied;
                    f->Prepare();
                    f->AddDevicePointerForArguments(static_cast<char *>(const_cast<void *>(src)) + offset);
                    f->AddVariableForArguments(chunk_size);
                    f->AddVariableForArguments(kind);
                    f->Execute("cudaMemcpy");
                    memmove(static_cast<char *>(dst) + offset, shm,
                            chunk_size);
                    offset += chunk_size;
                    copyied += chunk_size;
                }
            } else {
                /* NOTE: adding a fake host pointer */
                f->AddHostPointerForArguments("");
                f->AddDevicePointerForArguments(src);
                f->AddVariableForArguments(count);
                f->AddVariableForArguments(kind);
                f->Execute("cudaMemcpy");
                if (f->Success())
                    memmove(dst, f->GetOutputHostPointer<char>(count), count);
            }
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
            f->AddDevicePointerForArguments(stream);
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
            f->AddDevicePointerForArguments(stream);
            f->Execute("cudaMemcpyAsync");
            break;
        case cudaMemcpyDeviceToHost:
            /* NOTE: adding a fake host pointer */
            f->AddHostPointerForArguments("");
            f->AddDevicePointerForArguments(src);
            f->AddVariableForArguments(count);
            f->AddVariableForArguments(kind);
            f->AddDevicePointerForArguments(stream);
            f->Execute("cudaMemcpyAsync");
            if (f->Success())
                memmove(dst, f->GetOutputHostPointer<char>(count), count);
            break;
        case cudaMemcpyDeviceToDevice:
            f->AddDevicePointerForArguments(dst);
            f->AddDevicePointerForArguments(src);
            f->AddVariableForArguments(count);
            f->AddVariableForArguments(kind);
            f->AddDevicePointerForArguments(stream);
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
