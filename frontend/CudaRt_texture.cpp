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

using namespace std;

extern "C" cudaError_t cudaBindTexture(size_t *offset,
        const textureReference *texref, const void *devPtr,
        const cudaChannelFormatDesc *desc, size_t size) {
    Frontend *f = Frontend::GetFrontend();
    f->AddHostPointerForArguments(offset);
    // Achtung: passing the address and the content of the textureReference
    f->AddStringForArguments(CudaUtil::MarshalHostPointer(texref));
    f->AddHostPointerForArguments(texref);
    f->AddDevicePointerForArguments(devPtr);
    f->AddHostPointerForArguments(desc);
    f->AddVariableForArguments(size);
    f->Execute("cudaBindTexture");
    if (f->Success())
        *offset = *(f->GetOutputHostPointer<size_t > ());
    return f->GetExitCode();
}

extern "C" cudaError_t cudaBindTexture2D(size_t *offset,
        const textureReference *texref, const void *devPtr,
        const cudaChannelFormatDesc *desc, size_t width, size_t height,
        size_t pitch) {
    Frontend *f = Frontend::GetFrontend();
    f->AddHostPointerForArguments(offset);
    // Achtung: passing the address and the content of the textureReference
    f->AddStringForArguments(CudaUtil::MarshalHostPointer(texref));
    f->AddHostPointerForArguments(texref);
    f->AddDevicePointerForArguments(devPtr);
    f->AddHostPointerForArguments(desc);
    f->AddVariableForArguments(width);
    f->AddVariableForArguments(height);
    f->AddVariableForArguments(pitch);
    f->Execute("cudaBindTexture2D");
    if (f->Success())
        *offset = *(f->GetOutputHostPointer<size_t > ());
    return f->GetExitCode();
}

extern "C" cudaError_t cudaBindTextureToArray(const textureReference *texref,
        const cudaArray *array, const cudaChannelFormatDesc *desc) {
    Frontend *f = Frontend::GetFrontend();
    // Achtung: passing the address and the content of the textureReference
    f->AddStringForArguments(CudaUtil::MarshalHostPointer(texref));
    f->AddHostPointerForArguments(texref);
    f->AddDevicePointerForArguments((void *) array);
    f->AddHostPointerForArguments(desc);
    f->Execute("cudaBindTextureToArray");
    return f->GetExitCode();
}

extern "C" cudaChannelFormatDesc cudaCreateChannelDesc(int x, int y, int z, int w,
        cudaChannelFormatKind f) {
    cudaChannelFormatDesc desc;
    desc.x = x;
    desc.y = y;
    desc.z = z;
    desc.w = w;
    desc.f = f;
    return desc;
}

extern "C" cudaError_t cudaGetChannelDesc(cudaChannelFormatDesc *desc,
        const cudaArray *array) {
    Frontend *f = Frontend::GetFrontend();
    f->AddHostPointerForArguments(desc);
    f->AddDevicePointerForArguments(array);
    f->Execute("cudaGetChannelDesc");
    if (f->Success())
        memmove(desc, f->GetOutputHostPointer<cudaChannelFormatDesc > (),
            sizeof (cudaChannelFormatDesc));
    return f->GetExitCode();
}

extern "C" cudaError_t cudaGetTextureAlignmentOffset(size_t *offset,
        const textureReference *texref) {
    Frontend *f = Frontend::GetFrontend();
    f->AddHostPointerForArguments(offset);
    f->AddStringForArguments(CudaUtil::MarshalHostPointer(texref));
    f->Execute("cudaGetTextureAlignmentOffset");
    if (f->Success())
        *offset = *(f->GetOutputHostPointer<size_t > ());
    return f->GetExitCode();
}

extern "C" cudaError_t cudaGetTextureReference(const textureReference **texref,
        const char *symbol) {
    Frontend *f = Frontend::GetFrontend();
    // Achtung: skipping to add texref
    // Achtung: passing the address and the content of symbol
    f->AddStringForArguments(CudaUtil::MarshalHostPointer((void *) symbol));
    f->AddStringForArguments(symbol);
    f->Execute("cudaGetTextureReference");
    if (f->Success()) {
        char *texrefHandler = f->GetOutputString();
        *texref = (textureReference *) strtoul(texrefHandler, NULL, 16);
    }
    return f->GetExitCode();
}

extern "C" cudaError_t cudaUnbindTexture(const textureReference *texref) {
    Frontend *f = Frontend::GetFrontend();
    f->AddStringForArguments(CudaUtil::MarshalHostPointer(texref));
    f->Execute("cudaUnbindTexture");
    return f->GetExitCode();
}
