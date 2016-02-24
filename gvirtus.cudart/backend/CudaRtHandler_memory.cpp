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

#include "CudaRtHandler.h"

#include "CudaUtil.h"

#include <iostream>
#include <string>
#include <string.h>

using namespace std;

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
    if (exit_code == cudaSuccess)
        out->AddMarshal(devPtr);

    return new Result(exit_code, out);
}

CUDA_ROUTINE_HANDLER(GetSymbolSize) {
    try {
        Buffer *out = new Buffer();
        size_t *size = out->Delegate<size_t > ();
        *size = *(input_buffer->Assign<size_t > ());
        const char *symbol = pThis->GetSymbol(input_buffer);
        cudaError_t exit_code = cudaGetSymbolSize(size, symbol);
        return new Result(exit_code, out);
    } catch (string e) {
        cerr << e << endl;
        return new Result(cudaErrorMemoryAllocation);
    }

}

//testing vpelliccia

CUDA_ROUTINE_HANDLER(MemcpyPeerAsync) {
    void *dst = NULL;
    void *src = NULL;
    try {

        dst = input_buffer->GetFromMarshal<void *>();
        int dstDevice = input_buffer->Get<int>();
        src = input_buffer->GetFromMarshal<void *>();
        int srcDevice = input_buffer->Get<int>();
        size_t count = input_buffer->Get<size_t>();
        cudaStream_t stream = input_buffer->Get<cudaStream_t>();


        cudaError_t exit_code = cudaMemcpyPeerAsync(dst, dstDevice, src, srcDevice, count, stream);
        return new Result(exit_code);
    } catch (string e) {
        cerr << e << endl;
        return new Result(cudaErrorMemoryAllocation);
    }
}

CUDA_ROUTINE_HANDLER(MallocManaged) {

    void *devPtr = NULL;
    try {
        devPtr = input_buffer->Get<void* > ();
        size_t size = input_buffer->Get<size_t > ();
        unsigned flags = input_buffer->Get<unsigned > ();
        cudaError_t exit_code = cudaMallocManaged(&devPtr, size, flags);
        std::cout << "Allocated DevicePointer " << devPtr << " with a size of " << size << std::endl;
        Buffer *out = new Buffer();
        out->AddMarshal(devPtr);
        return new Result(exit_code, out);
    } catch (string e) {
        cerr << e << endl;
        return new Result(cudaErrorMemoryAllocation);
    }
}

CUDA_ROUTINE_HANDLER(Malloc3DArray) {

    cudaArray *array = NULL;

    cudaChannelFormatDesc *desc = input_buffer->Assign<cudaChannelFormatDesc >();
    cudaExtent extent = input_buffer->Get<cudaExtent>();
    unsigned int flags = input_buffer->Get<unsigned int>();
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
    printf("array %x\n", array);
#endif
    cudaError_t exit_code = cudaMalloc3DArray(&array, desc, extent, flags);
//    printf("array %x\n", array);
    Buffer *out = new Buffer();
    try {
        out->Add(&array);
    } catch (string e) {
        cerr << e << endl;
        return new Result(cudaErrorMemoryAllocation);
    }

    return new Result(exit_code, out);

}

CUDA_ROUTINE_HANDLER(Malloc) {
    void *devPtr = NULL;
    try {
        size_t size = input_buffer->Get<size_t > ();
        cudaError_t exit_code = cudaMalloc(&devPtr, size);
        std::cout << "Allocated DevicePointer " << devPtr << " with a size of " << size << std::endl;
        Buffer *out = new Buffer();
        out->AddMarshal(devPtr);
        return new Result(exit_code, out);
    } catch (string e) {
        cerr << e << endl;
        return new Result(cudaErrorMemoryAllocation);
    }
}

CUDA_ROUTINE_HANDLER(MallocArray) {
    cudaArray *arrayPtr = NULL;
    try {
        cudaChannelFormatDesc *desc = input_buffer->Assign<cudaChannelFormatDesc > ();
        size_t width = input_buffer->Get<size_t > ();
        size_t height = input_buffer->Get<size_t > ();

#ifdef DEBUG
        printf("x %d\n", desc->x);
        printf("y %d\n", desc->y);
        printf("z %d\n", desc->z);
        printf("w %d\n", desc->w);
        printf("f %d\n", desc->f);

        printf("width %d\n", width);
        printf("height %d\n", height);
#endif
        
        cudaError_t exit_code = cudaMallocArray(&arrayPtr, desc, width, height);
        Buffer *out = new Buffer();
        out->AddMarshal(arrayPtr);
        cout << hex << arrayPtr << endl;
        return new Result(exit_code, out);
    } catch (string e) {
        cerr << e << endl;
        return new Result(cudaErrorMemoryAllocation);
    }
}

CUDA_ROUTINE_HANDLER(MallocPitch) {
    void *devPtr = NULL;
    try {
        size_t pitch = input_buffer->Get<size_t > ();
        size_t width = input_buffer->Get<size_t > ();
        size_t height = input_buffer->Get<size_t > ();
        cudaError_t exit_code = cudaMallocPitch(&devPtr, &pitch, width, height);
        std::cout << "Allocated DevicePointer " << devPtr << " with a size of " << width * height << std::endl;
        Buffer *out = new Buffer();
        out->AddMarshal(devPtr);
        out->Add(pitch);
        return new Result(exit_code, out);
    } catch (string e) {
        cerr << e << endl;
        return new Result(cudaErrorMemoryAllocation);
    }


}

CUDA_ROUTINE_HANDLER(Memcpy) {
    /* cudaError_t cudaError_t cudaMemcpy(void *dst, const void *src,
        size_t count, cudaMemcpyKind kind) */
    void *dst = NULL;
    void *src = NULL;

    try {
        cudaMemcpyKind kind = input_buffer->BackGet<cudaMemcpyKind > ();
        size_t count = input_buffer->BackGet<size_t > ();

        cudaError_t exit_code;
        Result * result = NULL;
        Buffer *out;

        switch (kind) {
            case cudaMemcpyDefault:
            case cudaMemcpyHostToHost:
                // This should never happen
                result = NULL;
                break;
            case cudaMemcpyHostToDevice:
                try {
                    dst = input_buffer->GetFromMarshal<void *>();
                    src = input_buffer->AssignAll<char>();
                } catch (string e) {
                    cerr << e << endl;
                    return new Result(cudaErrorMemoryAllocation);
                }
                exit_code = cudaMemcpy(dst, src, count, kind);
                result = new Result(exit_code);
                break;
            case cudaMemcpyDeviceToHost:
                // FIXME: use buffer delegate
                dst = new char[count];
                /* skipping a char for fake host pointer */
                try {
                    input_buffer->Assign<char>();
                    src = input_buffer->GetFromMarshal<void *>();
                } catch (string e) {
                    cerr << e << endl;
                    return new Result(cudaErrorMemoryAllocation);
                }
                exit_code = cudaMemcpy(dst, src, count, kind);
                try {
                    out = new Buffer();
                    out->Add<char>((char *) dst, count);
                } catch (string e) {
                    cerr << e << endl;
                    return new Result(cudaErrorMemoryAllocation);
                }
                delete[] (char *) dst;
                result = new Result(exit_code, out);
                break;
            case cudaMemcpyDeviceToDevice:
                try {
                    dst = input_buffer->GetFromMarshal<void *>();
                    src = input_buffer->GetFromMarshal<void *>();
                } catch (string e) {
                    cerr << e << endl;
                    return new Result(cudaErrorMemoryAllocation);
                }
                exit_code = cudaMemcpy(dst, src, count, kind);
                result = new Result(exit_code);
                break;
        }
        return result;
    } catch (string e) {
        cerr << e << endl;
        return new Result(cudaErrorMemoryAllocation);
    }
}
//

CUDA_ROUTINE_HANDLER(Memcpy2DFromArray) {

    void *dst = NULL;
    cudaArray *src = NULL;
    size_t dpitch;
    size_t height;
    size_t width;
    size_t wOffset, hOffset;

    try {
        cudaMemcpyKind kind = input_buffer->BackGet<cudaMemcpyKind > ();


        cudaError_t exit_code;
        Result * result = NULL;
        Buffer *out;

        switch (kind) {
            case cudaMemcpyDefault:
            case cudaMemcpyHostToHost:
            case cudaMemcpyHostToDevice:
                // This should never happen
                result = NULL;
                break;
            case cudaMemcpyDeviceToHost:
                // FIXME: use buffer delegate
                /* skipping a char for fake host pointer */
                try {
                    input_buffer->Assign<char>(); // fittizio
                    src = (cudaArray *) input_buffer->GetFromMarshal<void *>();
                    dpitch = input_buffer->Get<size_t>();
                    wOffset = input_buffer->Get<size_t>();
                    hOffset = input_buffer->Get<size_t>();
                    width = input_buffer->Get<size_t>();
                    height = input_buffer->Get<size_t>();
                } catch (string e) {
                    cerr << e << endl;
                    return new Result(cudaErrorMemoryAllocation);
                }
                dst = new char[dpitch * height];
                exit_code = cudaMemcpy2DFromArray(dst, dpitch, src, wOffset, hOffset, width, height, kind);
                try {
                    out = new Buffer();
                    out->Add<char>((char *) dst, dpitch * height);
                } catch (string e) {
                    cerr << e << endl;
                    return new Result(cudaErrorMemoryAllocation);
                }
                delete[] (char *) dst;
                result = new Result(exit_code, out);
                break;
            case cudaMemcpyDeviceToDevice:
                try {
                    dst = input_buffer->GetFromMarshal<void *>();
                    src = (cudaArray *) input_buffer->GetFromMarshal<void *>();
                    dpitch = input_buffer->Get<size_t>();
                    wOffset = input_buffer->Get<size_t>();
                    hOffset = input_buffer->Get<size_t>();
                    width = input_buffer->Get<size_t>();
                    height = input_buffer->Get<size_t>();
                } catch (string e) {
                    cerr << e << endl;
                    return new Result(cudaErrorMemoryAllocation);
                }
                exit_code = cudaMemcpy2DFromArray(dst, dpitch, src, wOffset, hOffset, width, height, kind);
                result = new Result(exit_code);
                break;
        }
        return result;
    } catch (string e) {
        cerr << e << endl;
        return new Result(cudaErrorMemoryAllocation);
    }
}

CUDA_ROUTINE_HANDLER(Memcpy2DToArray) {

    void *src = NULL;
    cudaArray *dst = NULL;
    size_t spitch = 0;
    size_t height = 0;
    size_t width = 0;
    size_t wOffset = 0;
    size_t hOffset = 0;

    try {
        cudaMemcpyKind kind = input_buffer->BackGet<cudaMemcpyKind > ();
        
        cudaError_t exit_code;
        Result * result = NULL;

        switch (kind) {
            case cudaMemcpyDefault:
            case cudaMemcpyHostToHost:
            case cudaMemcpyDeviceToHost:
                // This should never happen
                result = new Result(cudaErrorInvalidMemcpyDirection);
                break;
            case cudaMemcpyHostToDevice:
                // FIXME: use buffer delegate
                /* skipping a char for fake host pointer */
                try {
                    dst = (cudaArray *)input_buffer->GetFromMarshal<void *>();
                    wOffset = input_buffer->Get<size_t>();
                    hOffset = input_buffer->Get<size_t>();
                    height = input_buffer->BackGet<size_t>();                    
                    width = input_buffer->BackGet<size_t>();
                    spitch = input_buffer->BackGet<size_t>();
                    src = input_buffer->AssignAll<char>();
                } catch (string e) {
                    cerr << e << endl;
                    return new Result(cudaErrorMemoryAllocation);
                }                
                break;
            case cudaMemcpyDeviceToDevice:
                try {
                    dst = (cudaArray *)input_buffer->GetFromMarshal<void *>();
                    wOffset = input_buffer->Get<size_t>();
                    hOffset = input_buffer->Get<size_t>();
                    src = input_buffer->GetFromMarshal<void *>();
                    spitch = input_buffer->Get<size_t>();
                    width = input_buffer->Get<size_t>();
                    height = input_buffer->Get<size_t>();
                } catch (string e) {
                    cerr << e << endl;
                    return new Result(cudaErrorMemoryAllocation);
                }                
        }
        exit_code = cudaMemcpy2DToArray(dst, wOffset, hOffset, src,
                        spitch, width, height, kind);
        return new Result(exit_code);
    } catch (string e) {
        cerr << e << endl;
        return new Result(cudaErrorMemoryAllocation);
    }
}

CUDA_ROUTINE_HANDLER(Memcpy3D) {

    void *src = NULL;

    try {
        cudaMemcpy3DParms *p = input_buffer->Assign<cudaMemcpy3DParms>();
        src = input_buffer->AssignAll<char>();
        unsigned int width = p->extent.width;
        p->srcPtr.ptr = src;
        
#ifdef DEBUG
        printf("PARAMETRI BACKEND\n");
        printf("dstArray %x\n\n", p->dstArray);

        printf("dstPos x %d\n", p->dstPos.x);
        printf("dstPos y %d\n", p->dstPos.y);
        printf("dstPos z %d\n\n", p->dstPos.z);

        printf("dstPtr pitch %d\n", p->dstPtr.pitch);
        printf("dstPtr ptr %x\n", p->dstPtr.ptr);
        printf("dstPtr x %d\n", p->dstPtr.xsize);
        printf("dstPtr y %d\n\n", p->dstPtr.ysize);


        printf("extent depth %d\n", p->extent.depth);
        printf("extent height %d\n", p->extent.height);
        printf("extent width %d\n\n", p->extent.width);

        printf("kind %d\n\n", p->kind);

        printf("srcArray %x\n\n", p->srcArray);

        printf("srcPos x %d\n", p->srcPos.x);
        printf("srcPos y %d\n", p->srcPos.y);
        printf("srcPos z %d\n\n", p->srcPos.z);

        printf("srcPtr pitch %d\n", p->srcPtr.pitch);
        printf("srcPtr ptr %x\n", p->srcPtr.ptr);
        printf("srcPtr x %d\n", p->srcPtr.xsize);
        printf("srcPtr y  %d\n", p->srcPtr.ysize);
#endif


        cudaError_t exit_code = cudaMemcpy3D(p);
        Buffer *out = new Buffer();
        out->Add(p, 1);
        return new Result(exit_code, out);
    } catch (string e) {
        cerr << e << endl;
        return new Result(cudaErrorMemoryAllocation);
    }

}

CUDA_ROUTINE_HANDLER(Memcpy2D) {
    void *dst = NULL;
    void *src = NULL;
    size_t dpitch;
    size_t spitch;
    size_t height;
    size_t width;

    try {
        cudaMemcpyKind kind = input_buffer->BackGet<cudaMemcpyKind > ();


        cudaError_t exit_code;
        Result * result = NULL;
        Buffer *out;

        switch (kind) {
            case cudaMemcpyDefault:
            case cudaMemcpyHostToHost:
                // This should never happen
                result = NULL;
                break;
            case cudaMemcpyHostToDevice:
                try {
                    dst = input_buffer->GetFromMarshal<void *>();
                    src = input_buffer->AssignAll<char>();
                    dpitch = input_buffer->Get<size_t>();
                    spitch = input_buffer->Get<size_t>();
                    width = input_buffer->Get<size_t>();
                    height = input_buffer->Get<size_t>();
                } catch (string e) {
                    cerr << e << endl;
                    return new Result(cudaErrorMemoryAllocation);
                }
                exit_code = cudaMemcpy2D(dst, dpitch, src, spitch, width, height,
                        kind);
                result = new Result(exit_code);
                break;
            case cudaMemcpyDeviceToHost:
                // FIXME: use buffer delegate
                /* skipping a char for fake host pointer */
                try {
                    input_buffer->Assign<char>();
                    src = input_buffer->GetFromMarshal<void *>();
                    dpitch = input_buffer->Get<size_t>();
                    spitch = input_buffer->Get<size_t>();
                    width = input_buffer->Get<size_t>();
                    height = input_buffer->Get<size_t>();
                } catch (string e) {
                    cerr << e << endl;
                    return new Result(cudaErrorMemoryAllocation);
                }
                dst = new char[dpitch * height];
                exit_code = cudaMemcpy2D(dst, dpitch, src, spitch, width, height, kind);
                try {
                    out = new Buffer();
                    out->Add<char>((char *) dst, dpitch * height);
                } catch (string e) {
                    cerr << e << endl;
                    return new Result(cudaErrorMemoryAllocation);
                }
                delete[] (char *) dst;
                result = new Result(exit_code, out);
                break;
            case cudaMemcpyDeviceToDevice:
                try {
                    dst = input_buffer->GetFromMarshal<void *>();
                    src = input_buffer->GetFromMarshal<void *>();
                    dpitch = input_buffer->Get<size_t>();
                    spitch = input_buffer->Get<size_t>();
                    width = input_buffer->Get<size_t>();
                    height = input_buffer->Get<size_t>();
                } catch (string e) {
                    cerr << e << endl;
                    return new Result(cudaErrorMemoryAllocation);
                }
                exit_code = cudaMemcpy2D(dst, dpitch, src, spitch, width, height, kind);
                result = new Result(exit_code);
                break;
        }
        return result;
    } catch (string e) {
        cerr << e << endl;
        return new Result(cudaErrorMemoryAllocation);
    }
}

CUDA_ROUTINE_HANDLER(MemcpyAsync) {
    void *dst = NULL;
    void *src = NULL;

    try {
        cudaStream_t stream = input_buffer->BackGet<cudaStream_t > ();
        cudaMemcpyKind kind = input_buffer->BackGet<cudaMemcpyKind > ();
        size_t count = input_buffer->BackGet<size_t > ();

        cudaError_t exit_code;
        Buffer * out;
        Result * result = NULL;

        switch (kind) {
            case cudaMemcpyDefault:
            case cudaMemcpyHostToHost:
                result = new Result(cudaSuccess);
                break;
            case cudaMemcpyHostToDevice:
                try {
                    dst = input_buffer->GetFromMarshal<void *>();
                } catch (string e) {
                    cerr << e << endl;
                    return new Result(cudaErrorMemoryAllocation);
                }
                try {
                    src = input_buffer->AssignAll<char>();
                } catch (string e) {
                    cerr << e << endl;
                    return new Result(cudaErrorMemoryAllocation);
                }
                exit_code = cudaMemcpy(dst, src, count, kind);
                result = new Result(exit_code);
                break;
            case cudaMemcpyDeviceToHost:
                // FIXME: use buffer delegate
                dst = new char[count];
                /* skipping a char for fake host pointer */
                try {
                    input_buffer->Assign<char>();
                    src = input_buffer->GetFromMarshal<void *>();
                } catch (string e) {
                    cerr << e << endl;
                    return new Result(cudaErrorMemoryAllocation);
                }
                exit_code = cudaMemcpy(dst, src, count, kind);
                try {
                    out = new Buffer();
                    out->Add<char>((char *) dst, count);
                } catch (string e) {
                    cerr << e << endl;
                    return new Result(cudaErrorMemoryAllocation);
                }
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
    } catch (string e) {
        cerr << e << endl;
        return new Result(cudaErrorMemoryAllocation);
    }
}

CUDA_ROUTINE_HANDLER(MemcpyFromSymbol) {
    try {
        void *dst = input_buffer->GetFromMarshal<void *>();
        char *handler = input_buffer->AssignString();
        char *symbol = input_buffer->AssignString();
        handler = (char *) CudaUtil::UnmarshalPointer(handler);
        size_t count = input_buffer->Get<size_t > ();
        size_t offset = input_buffer->Get<size_t > ();
        cudaMemcpyKind kind = input_buffer->Get<cudaMemcpyKind > ();

        size_t size;

        if (cudaGetSymbolSize(&size, symbol) != cudaSuccess) {
            symbol = handler;
            cudaGetLastError();
        }

        cudaError_t exit_code;
        Result * result = NULL;
        Buffer * out = NULL;

        switch (kind) {
            case cudaMemcpyDefault:
            case cudaMemcpyHostToHost:
                // This should never happen
                result = new Result(cudaErrorInvalidMemcpyDirection);
                break;
            case cudaMemcpyHostToDevice:
                // This should never happen
                result = new Result(cudaErrorInvalidMemcpyDirection);
                break;
            case cudaMemcpyDeviceToHost:
                try {
                    out = new Buffer(count);
                    dst = out->Delegate<char>(count);
                } catch (string e) {
                    cerr << e << endl;
                    return new Result(cudaErrorMemoryAllocation);
                }
                exit_code = cudaMemcpyFromSymbol(dst, symbol, count, offset, kind);
                result = new Result(exit_code, out);
                break;
            case cudaMemcpyDeviceToDevice:
                exit_code = cudaMemcpyFromSymbol(dst, symbol, count, offset, kind);
                result = new Result(exit_code);
                break;
        }
        return result;
    } catch (string e) {
        cerr << e << endl;
        return new Result(cudaErrorMemoryAllocation);
    }
}

CUDA_ROUTINE_HANDLER(MemcpyToArray) {
    void *src = NULL;

    try {
        cudaArray *dst = input_buffer->GetFromMarshal<cudaArray *>();
        size_t wOffset = input_buffer->Get<size_t > ();
        size_t hOffset = input_buffer->Get<size_t > ();
        cudaMemcpyKind kind = input_buffer->BackGet<cudaMemcpyKind > ();
        size_t count = input_buffer->BackGet<size_t > ();

        cudaError_t exit_code;
        Result * result = NULL;

        switch (kind) {
            case cudaMemcpyDefault:
            case cudaMemcpyHostToHost:
                // This should never happen
                result = new Result(cudaErrorInvalidMemcpyDirection);
                break;
            case cudaMemcpyHostToDevice:
                /* Achtung: this isn't strictly correct because here we assign just
                 * a pointer to one character, any successive assign should
                 * take inaxpectated result ... but it works here!
                 */
                try {
                    src = input_buffer->AssignAll<char>();
                } catch (string e) {
                    cerr << e << endl;
                    return new Result(cudaErrorMemoryAllocation);
                }
                exit_code = cudaMemcpyToArray(dst, wOffset, hOffset, src, count, kind);
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
    } catch (string e) {
        cerr << e << endl;
        return new Result(cudaErrorMemoryAllocation);
    }
}

CUDA_ROUTINE_HANDLER(MemcpyFromArray) {
    /* cudaMemcpyFromArray(void *dst, const cudaArray *src,
        size_t wOffset, size_t hOffset, size_t count, cudaMemcpyKind kind) */

    void *dst = NULL;
    cudaArray *src = NULL;
    cudaMemcpyKind kind = input_buffer->BackGet<cudaMemcpyKind > ();
    size_t count = input_buffer->BackGet<size_t > ();
    size_t hOffset = input_buffer->BackGet<size_t > ();
    size_t wOffset = input_buffer->BackGet<size_t > ();

#ifdef DEBUG
    std::cout << "wOffset " << wOffset << " hOffset " << hOffset << std::endl;
#endif
    cudaError_t exit_code;
    Result * result = NULL;
    Buffer *out;

    switch (kind) {
        case cudaMemcpyDefault:
        case cudaMemcpyHostToHost:
        case cudaMemcpyHostToDevice:
            // This should never happen
            result = new Result(cudaErrorInvalidMemcpyDirection);
            break;

        case cudaMemcpyDeviceToHost:
            // FIXME: use buffer delegate
            dst = new char[count];
            /* skipping a char for fake host pointer */
            input_buffer->Assign<char>(); //???
            src = (cudaArray *) input_buffer->GetFromMarshal<void *>();

            exit_code = cudaMemcpyFromArray(dst, src, wOffset, hOffset, count, kind);
            out = new Buffer();
            out->Add<char>((char *) dst, count);
            delete[] (char *) dst;
            result = new Result(exit_code, out);
            break;

        case cudaMemcpyDeviceToDevice:
            dst = input_buffer->GetFromMarshal<void *>();
            src = (cudaArray *) input_buffer->GetFromMarshal<void *>();
            // src = input_buffer->GetFromMarshal<void *>();
            exit_code = cudaMemcpyFromArray(dst, src, wOffset, hOffset, count, kind);
            result = new Result(exit_code);
            break;
    }
    return result;
}

CUDA_ROUTINE_HANDLER(MemcpyArrayToArray) {
    cudaArray *src = NULL;
    cudaArray *dst = NULL;
    cudaMemcpyKind kind = input_buffer->BackGet<cudaMemcpyKind > ();
    size_t count;
    size_t hOffsetDst, hOffsetSrc;
    size_t wOffsetDst, wOffsetSrc;
    cudaError_t exit_code;
    Result * result = NULL;

    switch (kind) {
        case cudaMemcpyDefault:
        case cudaMemcpyHostToHost:
        case cudaMemcpyHostToDevice:
        case cudaMemcpyDeviceToHost:
            // This should never happen
            result = new Result(cudaErrorInvalidMemcpyDirection);
            break;

        case cudaMemcpyDeviceToDevice:
            dst = (cudaArray *) input_buffer->GetFromMarshal<void *>();
            wOffsetDst = input_buffer->Get<size_t>();
            hOffsetDst = input_buffer->Get<size_t>();
            src = (cudaArray *) input_buffer->GetFromMarshal<void *>();
            // src = input_buffer->GetFromMarshal<void *>();
            wOffsetSrc = input_buffer->Get<size_t>();
            hOffsetSrc = input_buffer->Get<size_t>();
            count = input_buffer->Get<size_t>();
            exit_code = cudaMemcpyArrayToArray(dst, wOffsetDst, hOffsetDst, src, wOffsetSrc, hOffsetSrc, count, kind);
            result = new Result(exit_code);
            break;
    }
    return result;
}

CUDA_ROUTINE_HANDLER(MemcpyToSymbol) {
    void *src = NULL;

    try {
        cudaMemcpyKind kind = input_buffer->BackGet<cudaMemcpyKind > ();
        size_t offset = input_buffer->BackGet<size_t > ();
        size_t count = input_buffer->BackGet<size_t > ();
        char *handler = input_buffer->AssignString();
        char *symbol = input_buffer->AssignString();

        handler = (char *) CudaUtil::UnmarshalPointer(handler);
        size_t size;

        if (cudaGetSymbolSize(&size, symbol) != cudaSuccess) {
            symbol = handler;
            cudaGetLastError();
        }

        cudaError_t exit_code;
        Result * result = NULL;

        switch (kind) {
            case cudaMemcpyDefault:
            case cudaMemcpyHostToHost:
                // This should never happen
                result = new Result(cudaErrorInvalidMemcpyDirection);
                break;
            case cudaMemcpyHostToDevice:
                try {
                    src = input_buffer->AssignAll<char>();
                } catch (string e) {
                    cerr << e << endl;
                    return new Result(cudaErrorMemoryAllocation);
                }
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
    } catch (string e) {
        cerr << e << endl;
        return new Result(cudaErrorMemoryAllocation);
    }
}

CUDA_ROUTINE_HANDLER(Memset) {
    try {
        void *devPtr = input_buffer->GetFromMarshal<void *>();
        int value = input_buffer->Get<int>();
        size_t count = input_buffer->Get<size_t > ();
        cudaError_t exit_code = cudaMemset(devPtr, value, count);
        return new Result(exit_code);
    } catch (string e) {
        cerr << e << endl;
        return new Result(cudaErrorMemoryAllocation);
    }


}

CUDA_ROUTINE_HANDLER(Memset2D) {
    try {
        void *devPtr = input_buffer->GetFromMarshal<void *>();
        size_t pitch = input_buffer->Get<size_t > ();
        int value = input_buffer->Get<int> ();
        size_t width = input_buffer->Get<size_t > ();
        size_t height = input_buffer->Get<size_t > ();
        cudaError_t exit_code = cudaMemset2D(devPtr, pitch, value, width, height);
        return new Result(exit_code);
    } catch (string e) {
        cerr << e << endl;
        return new Result(cudaErrorMemoryAllocation);
    }
}
