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

#include "CudaRt.h"

/*
 Routines not found in the cuda's header files.
 KEEP THEM WITH CARE
 */

extern "C" __host__ void** __cudaRegisterFatBinary(void *fatCubin) {
    /* Fake host pointer */
    Buffer * input_buffer = new Buffer();
    input_buffer->AddString(CudaUtil::MarshalHostPointer((void **) fatCubin));
    input_buffer = CudaUtil::MarshalFatCudaBinary(
            (__cudaFatCudaBinary *) fatCubin, input_buffer);

    Frontend *f = Frontend::GetFrontend();
    f->Execute("cudaRegisterFatBinary", input_buffer);
    if (f->Success())
        return (void **) fatCubin;
    return NULL;
}

extern "C" __host__ void __cudaUnregisterFatBinary(void **fatCubinHandle) {
    Frontend *f = Frontend::GetFrontend();
    f->AddStringForArguments(CudaUtil::MarshalHostPointer(fatCubinHandle));

    f->Execute("cudaUnregisterFatBinary");
}

extern "C" __host__ void __cudaRegisterFunction(void **fatCubinHandle, const char *hostFun,
        char *deviceFun, const char *deviceName, int thread_limit, uint3 *tid,
        uint3 *bid, dim3 *bDim, dim3 *gDim, int *wSize) {
    Frontend *f = Frontend::GetFrontend();
    f->AddStringForArguments(CudaUtil::MarshalHostPointer(fatCubinHandle));
    f->AddStringForArguments(CudaUtil::MarshalHostPointer(hostFun));
    f->AddStringForArguments(deviceFun);
    f->AddStringForArguments(deviceName);
    f->AddVariableForArguments(thread_limit);
    f->AddHostPointerForArguments(tid);
    f->AddHostPointerForArguments(bid);
    f->AddHostPointerForArguments(bDim);
    f->AddHostPointerForArguments(gDim);
    f->AddHostPointerForArguments(wSize);

    f->Execute("cudaRegisterFunction");

    deviceFun = f->GetOutputString();
    tid = f->GetOutputHostPointer<uint3 > ();
    bid = f->GetOutputHostPointer<uint3 > ();
    bDim = f->GetOutputHostPointer<dim3 > ();
    gDim = f->GetOutputHostPointer<dim3 > ();
    wSize = f->GetOutputHostPointer<int>();
}

extern "C" __host__ void __cudaRegisterVar(void **fatCubinHandle, char *hostVar,
        char *deviceAddress, const char *deviceName, int ext, int size,
        int constant, int global) {
    Frontend *f = Frontend::GetFrontend();
    f->AddStringForArguments(CudaUtil::MarshalHostPointer(fatCubinHandle));
    f->AddStringForArguments(CudaUtil::MarshalHostPointer(hostVar));
    f->AddStringForArguments(deviceAddress);
    f->AddStringForArguments(deviceName);
    f->AddVariableForArguments(ext);
    f->AddVariableForArguments(size);
    f->AddVariableForArguments(constant);
    f->AddVariableForArguments(global);
    f->Execute("cudaRegisterVar");
}

extern "C" __host__ void __cudaRegisterShared(void **fatCubinHandle, void **devicePtr) {
    Frontend *f = Frontend::GetFrontend();
    f->AddStringForArguments(CudaUtil::MarshalHostPointer(fatCubinHandle));
    f->AddStringForArguments((char *) devicePtr);
    f->Execute("cudaRegisterShared");
}

extern "C" __host__ void __cudaRegisterSharedVar(void **fatCubinHandle, void **devicePtr,
        size_t size, size_t alignment, int storage) {
    Frontend *f = Frontend::GetFrontend();
    f->AddStringForArguments(CudaUtil::MarshalHostPointer(fatCubinHandle));
    f->AddStringForArguments((char *) devicePtr);
    f->AddVariableForArguments(size);
    f->AddVariableForArguments(alignment);
    f->AddVariableForArguments(storage);
    f->Execute("cudaRegisterSharedVar");
}

extern "C" __host__ void __cudaRegisterTexture(void **fatCubinHandle,
        const textureReference *hostVar, void **deviceAddress, char *deviceName,
        int dim, int norm, int ext) {
    Frontend *f = Frontend::GetFrontend();
    f->AddStringForArguments(CudaUtil::MarshalHostPointer(fatCubinHandle));
    // Achtung: passing the address and the content of the textureReference
    f->AddStringForArguments(CudaUtil::MarshalHostPointer(hostVar));
    f->AddHostPointerForArguments(hostVar);
    f->AddHostPointerForArguments(deviceAddress);
    f->AddStringForArguments(deviceName);
    f->AddVariableForArguments(dim);
    f->AddVariableForArguments(norm);
    f->AddVariableForArguments(ext);
    f->Execute("cudaRegisterTexture");
}


/* */

extern "C" __host__ int __cudaSynchronizeThreads(void** x, void* y) {
    // FIXME: implement
    cerr << "*** Error: __cudaSynchronizeThreads() not yet implemented!"
            << endl;
    return 0;
}

extern "C" __host__ void __cudaTextureFetch(const void *tex, void *index, int integer,
        void *val) {
    // FIXME: implement 
    cerr << "*** Error: __cudaTextureFetch() not yet implemented!" << endl;
}
