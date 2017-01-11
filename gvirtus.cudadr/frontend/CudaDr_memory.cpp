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
 * Written by: Flora Giannone <flora.giannone@studenti.uniparthenope.it>,
 *             Department of Applied Science
 */


#include <cstring>
#include "CudaDrFrontend.h"
#include "CudaUtil.h"
#include "CudaDr.h"
#include <cuda.h>
#include <stdio.h>
using namespace std;

/*Frees device memory.*/
extern CUresult cuMemFree(CUdeviceptr dptr) {
    CudaDrFrontend::Prepare();
    CudaDrFrontend::AddVariableForArguments(dptr);
    CudaDrFrontend::Execute("cuMemFree");
    return (CUresult) CudaDrFrontend::GetExitCode();
}

/*Allocates device memory.*/
extern CUresult cuMemAlloc(CUdeviceptr *dptr, size_t bytesize) {
    CudaDrFrontend::Prepare();
    CudaDrFrontend::AddVariableForArguments(bytesize);
    CudaDrFrontend::Execute("cuMemAlloc");
    if (CudaDrFrontend::Success())
        *dptr = (CUdeviceptr) (CudaDrFrontend::GetOutputDevicePointer());
    return (CUresult) CudaDrFrontend::GetExitCode();
}

/*Copies memory from Device to Host. */
extern CUresult cuMemcpyDtoH(void *dstHost, CUdeviceptr srcDevice, size_t ByteCount) {
    CudaDrFrontend::Prepare();
    CudaDrFrontend::AddVariableForArguments(srcDevice);
    CudaDrFrontend::AddVariableForArguments(ByteCount);
    CudaDrFrontend::Execute("cuMemcpyDtoH");
    if (CudaDrFrontend::Success())
        memmove(dstHost, CudaDrFrontend::GetOutputHostPointer<char>(ByteCount), ByteCount);
    return (CUresult) CudaDrFrontend::GetExitCode();
}

/*Copies memory from Host to Device.*/
extern CUresult cuMemcpyHtoD(CUdeviceptr dstDevice, const void *srcHost, size_t ByteCount) {
    CudaDrFrontend::Prepare();
    CudaDrFrontend::AddVariableForArguments(ByteCount);
    CudaDrFrontend::AddVariableForArguments(dstDevice);
    CudaDrFrontend::AddHostPointerForArguments<char>(static_cast<char *>(const_cast<void *> (srcHost)), ByteCount);
    CudaDrFrontend::Execute("cuMemcpyHtoD");
    return (CUresult) CudaDrFrontend::GetExitCode();

}

/*Creates a 1D or 2D CUDA array. */
extern CUresult cuArrayCreate(CUarray *pHandle, const CUDA_ARRAY_DESCRIPTOR *pAllocateArray) {
    CudaDrFrontend::Prepare();
    CudaDrFrontend::AddHostPointerForArguments(pAllocateArray);
    CudaDrFrontend::Execute("cuArrayCreate");
    if (CudaDrFrontend::Success())
        *pHandle = (CUarray) CudaDrFrontend::GetOutputDevicePointer();
    return (CUresult) CudaDrFrontend::GetExitCode();
}

/*Creates a 3D CUDA array.*/
extern CUresult cuArray3DCreate(CUarray *pHandle, const CUDA_ARRAY3D_DESCRIPTOR *pAllocateArray) {
    CudaDrFrontend::Prepare();
    CudaDrFrontend::AddHostPointerForArguments(pAllocateArray);
    CudaDrFrontend::Execute("cuArrayCreate");
    if (CudaDrFrontend::Success())
        *pHandle = (CUarray) CudaDrFrontend::GetOutputDevicePointer();
    return (CUresult) CudaDrFrontend::GetExitCode();
}

/*Copies memory for 2D arrays.*/
extern CUresult cuMemcpy2D(const CUDA_MEMCPY2D *pCopy) {
    CudaDrFrontend::Prepare();
    int flag = 0;
    CudaDrFrontend::AddHostPointerForArguments(pCopy);
    if (pCopy->srcHost != NULL) {
        flag = 1;
        CudaDrFrontend::AddVariableForArguments(flag);
        CudaDrFrontend::AddHostPointerForArguments((char*) pCopy->srcHost, (pCopy->WidthInBytes)*(pCopy->Height));
    } else {
        flag = 0;
        CudaDrFrontend::AddVariableForArguments(flag);
    }
    CudaDrFrontend::Execute("cuMemcpy2D");
    return (CUresult) CudaDrFrontend::GetExitCode();
}

/*Destroys a CUDA array.*/
extern CUresult cuArrayDestroy(CUarray hArray) {
    CudaDrFrontend::Prepare();
    CudaDrFrontend::AddDevicePointerForArguments(hArray);
    CudaDrFrontend::Execute("cuArrayDestroy");
    return (CUresult) CudaDrFrontend::GetExitCode();
}

/*Allocates pitched device memory.*/
extern CUresult cuMemAllocPitch(CUdeviceptr *dptr, size_t *pPitch, size_t WidthInBytes, size_t Height, unsigned int ElementSizeBytes) {
    CudaDrFrontend::Prepare();
    CudaDrFrontend::AddVariableForArguments(WidthInBytes);
    CudaDrFrontend::AddVariableForArguments(Height);
    CudaDrFrontend::AddVariableForArguments(ElementSizeBytes);
    CudaDrFrontend::Execute("cuMemAllocPitch");
    if (CudaDrFrontend::Success()) {
        *dptr = (CUdeviceptr) (CudaDrFrontend::GetOutputDevicePointer());
        *pPitch = *(CudaDrFrontend::GetOutputHostPointer<size_t > ());
    }
    return (CUresult) CudaDrFrontend::GetExitCode();
}

/*Get information on memory allocations.*/
extern CUresult cuMemGetAddressRange(CUdeviceptr *pbase, size_t *psize, CUdeviceptr dptr) {
    CudaDrFrontend::Prepare();
    CudaDrFrontend::AddVariableForArguments(dptr);
    CudaDrFrontend::Execute("cuMemGetAddressRange");
    if (CudaDrFrontend::Success()) {
        *pbase = (CUdeviceptr) (CudaDrFrontend::GetOutputDevicePointer());
        *psize = *(CudaDrFrontend::GetOutputHostPointer<size_t > ());
    }
    return (CUresult) CudaDrFrontend::GetExitCode();
}

/*Gets free and total memory.*/
extern CUresult cuMemGetInfo(size_t *free, size_t *total) {
    CudaDrFrontend::Prepare();
    CudaDrFrontend::Execute("cuMemGetInfo");
    if (CudaDrFrontend::Success()) {
        *free = *(CudaDrFrontend::GetOutputHostPointer<size_t > ());
        *total = *(CudaDrFrontend::GetOutputHostPointer<size_t > ());
    }
    return (CUresult) CudaDrFrontend::GetExitCode();
}

extern CUresult cuArray3DGetDescriptor(CUDA_ARRAY3D_DESCRIPTOR *pArrayDescriptor, CUarray hArray) {
    // FIXME: implement
    cerr << "*** Error: cuArray3DGetDescriptor() not yet implemented!" << endl;
    return (CUresult) 1;
}

extern CUresult cuArrayGetDescriptor(CUDA_ARRAY_DESCRIPTOR *pArrayDescriptor, CUarray hArray) {
    // FIXME: implement
    cerr << "*** Error: cuArrayGetDescriptor() not yet implemented!" << endl;
    return (CUresult) 1;
}

extern CUresult cuMemAllocHost(void **pp, size_t bytesize) {
    // FIXME: implement
    cerr << "*** Error: cuMemAllocHost() not yet implemented!" << endl;
    return (CUresult) 1;
}

extern CUresult cuMemcpy2DAsync(const CUDA_MEMCPY2D *pCopy, CUstream hStream) {
    // FIXME: implement
    cerr << "*** Error: cuMemcpy2Dasync() not yet implemented!" << endl;
    return (CUresult) 1;
}

extern CUresult cuMemcpy2DUnaligned(const CUDA_MEMCPY2D *pCopy) {
    // FIXME: implement
    cerr << "*** Error: cuMemcpy2DUnaligned() not yet implemented!" << endl;
    return (CUresult) 1;
}

extern CUresult cuMemcpy3D(const CUDA_MEMCPY3D *pCopy) {
    // FIXME: implement
    cerr << "*** Error: cuMemcpy3D() not yet implemented!" << endl;
    return (CUresult) 1;
}

extern CUresult cuMemcpy3DAsync(const CUDA_MEMCPY3D *pCopy, CUstream hStream) {
    // FIXME: implement
    cerr << "*** Error: cuMemcpy3DAsync() not yet implemented!" << endl;
    return (CUresult) 1;
}

extern CUresult cuMemcpyAtoA(CUarray dstArray, size_t dstIndex, CUarray srcArray, size_t srcIndex, size_t ByteCount) {
    // FIXME: implement
    cerr << "*** Error: cuMemcpyAtoA() not yet implemented!" << endl;
    return (CUresult) 1;
}

extern CUresult cuMemcpyAtoD(CUdeviceptr dstDevice, CUarray hSrc, size_t SrcIndex, size_t ByteCount) {
    // FIXME: implement
    cerr << "*** Error: cuMemcpyAtoD() not yet implemented!" << endl;
    return (CUresult) 1;
}

extern CUresult cuMemcpyAtoH(void *dstHost, CUarray srcArray, size_t srcIndex, size_t ByteCount) {
    // FIXME: implement
    cerr << "*** Error: cuMemcpyAtoH() not yet implemented!" << endl;
    return (CUresult) 1;
}

extern CUresult cuMemcpyAtoHAsync(void *dstHost, CUarray srcArray, size_t srcIndex, size_t ByteCount, CUstream hStream) {
    // FIXME: implement
    cerr << "*** Error: cuMemcpyAtoHAsync() not yet implemented!" << endl;
    return (CUresult) 1;
}

extern CUresult cuMemcpyDtoA(CUarray dstArray, size_t dstIndex, CUdeviceptr srcDevice, size_t ByteCount) {
    // FIXME: implement
    cerr << "*** Error: cuMemcpyDtoA() not yet implemented!" << endl;
    return (CUresult) 1;
}

extern CUresult cuMemcpyDtoD(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount) {
    // FIXME: implement
    cerr << "*** Error: cuMemcpyDtoD() not yet implemented!" << endl;
    return (CUresult) 1;
}

extern CUresult cuMemcpyDtoDAsync(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount, CUstream hStream) {
    // FIXME: implement
    cerr << "*** Error: cuMemcpyDtoDAsync() not yet implemented!" << endl;
    return (CUresult) 1;
}

extern CUresult cuMemcpyDtoHAsync(void *dstHost, CUdeviceptr srcDevice, size_t ByteCount, CUstream hStream) {
    // FIXME: implement
    cerr << "*** Error: cuMemcpyDtoHAsync() not yet implemented!" << endl;
    return (CUresult) 1;
}

extern CUresult cuMemcpyHtoA(CUarray dstArray, size_t dstIndex, const void *pSrc, size_t ByteCount) {
    // FIXME: implement
    cerr << "*** Error: cuMemcpyHtoA() not yet implemented!" << endl;
    return (CUresult) 1;
}

extern CUresult cuMemcpyHtoAAsync(CUarray dstArray, size_t dstIndex, const void *pSrc, size_t ByteCount, CUstream hStream) {
    // FIXME: implement
    cerr << "*** Error: cuMemcpyHtoAAsync() not yet implemented!" << endl;
    return (CUresult) 1;
}

extern CUresult cuMemcpyHtoDAsync(CUdeviceptr dstDevice, const void *srcHost, size_t ByteCount, CUstream hStream) {
    // FIXME: implement
    cerr << "*** Error: cuMemcpyHtoDAsync() not yet implemented!" << endl;
    return (CUresult) 1;
}

extern CUresult cuMemFreeHost(void *p) {
    // FIXME: implement
    cerr << "*** Error: cuMemFreeHost() not yet implemented!" << endl;
    return (CUresult) 1;
}

extern CUresult cuMemHostAlloc(void **pp, size_t bytesize, unsigned int Flags) {
    // FIXME: implement
    cerr << "*** Error: cuMemHostAlloc() not yet implemented!" << endl;
    return (CUresult) 1;
}

extern CUresult cuMemHostGetDevicePointer(CUdeviceptr *pdptr, void *p, unsigned int Flags) {
    // FIXME: implement
    cerr << "*** Error: cuMemHostGetDevicePointer() not yet implemented!" << endl;
    return (CUresult) 1;
}

extern CUresult cuMemHostGetFlags(unsigned int *pFlags, void *p) {
    // FIXME: implement
    cerr << "*** Error: cuMemHostGetFlags() not yet implemented!" << endl;
    return (CUresult) 1;
}

extern CUresult cuMemsetD16(CUdeviceptr dstDevice, unsigned short us, size_t N) {
    // FIXME: implement
    cerr << "*** Error: cuMemsetD16() not yet implemented!" << endl;
    return (CUresult) 1;
}

extern CUresult cuMemsetD2D16(CUdeviceptr dstDevice, size_t dstPitch, unsigned short us, size_t Width, size_t Height) {
    // FIXME: implement
    cerr << "*** Error: cuMemsetD2D16() not yet implemented!" << endl;
    return (CUresult) 1;
}

extern CUresult cuMemsetD2D32(CUdeviceptr dstDevice, size_t dstPitch, unsigned int ui, size_t Width, size_t Height) {
    // FIXME: implement
    cerr << "*** Error: cuMemsetD2D32() not yet implemented!" << endl;
    return (CUresult) 1;
}

extern CUresult cuMemsetD2D8(CUdeviceptr dstDevice, size_t dstPitch, unsigned char uc, size_t Width, size_t Height) {
    // FIXME: implement
    cerr << "*** Error: cuMemsetD2D8() not yet implemented!" << endl;
    return (CUresult) 1;
}

extern CUresult cuMemsetD32(CUdeviceptr dstDevice, unsigned int ui, size_t N) {
    // FIXME: implement
    cerr << "*** Error: cuMemsetD32() not yet implemented!" << endl;
    return (CUresult) 1;
}

extern CUresult cuMemsetD8(CUdeviceptr dstDevice, unsigned char uc, size_t N) {
    // FIXME: implement
    cerr << "*** Error: cuMemsetD8() not yet implemented!" << endl;
    return (CUresult) 1;
}
