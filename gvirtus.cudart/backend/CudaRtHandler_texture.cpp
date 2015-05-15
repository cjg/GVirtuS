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

extern const textureReference *getTexture(const textureReference *handler);

CUDA_ROUTINE_HANDLER(BindTexture) {
    Buffer *out = new Buffer();
    size_t *offset = out->Delegate<size_t>();
    *offset = *(input_buffer->Assign<size_t>());
    char * texrefHandler = input_buffer->AssignString();
    textureReference *guestTexref = input_buffer->Assign<textureReference>();
    textureReference *texref = pThis->GetTexture(texrefHandler);
    memmove(texref, guestTexref, sizeof(textureReference));
    void *devPtr = input_buffer->GetFromMarshal<void *>();
    cudaChannelFormatDesc *desc = input_buffer->Assign<cudaChannelFormatDesc>();
    size_t size = input_buffer->Get<size_t>();

    cudaError_t exit_code = cudaBindTexture(offset, texref, devPtr, desc, size);

    return new Result(exit_code, out);
}

#ifndef CUDART_VERSION
#error CUDART_VERSION not defined
#endif
#if CUDART_VERSION >= 2030
CUDA_ROUTINE_HANDLER(BindTexture2D) {
    Buffer *out = new Buffer();
    size_t *offset = out->Delegate<size_t>();
    *offset = *(input_buffer->Assign<size_t>());
    char * texrefHandler = input_buffer->AssignString();
    textureReference *guestTexref = input_buffer->Assign<textureReference>();
    textureReference *texref = pThis->GetTexture(texrefHandler);
    memmove(texref, guestTexref, sizeof(textureReference));
    void *devPtr = input_buffer->GetFromMarshal<void *>();
    cudaChannelFormatDesc *desc = input_buffer->Assign<cudaChannelFormatDesc>();
    size_t width = input_buffer->Get<size_t>();
    size_t height = input_buffer->Get<size_t>();
    size_t pitch = input_buffer->Get<size_t>();

    cudaError_t exit_code = cudaBindTexture2D(offset, texref, devPtr, desc, width,
            height, pitch);

    return new Result(exit_code, out);
}
#endif

CUDA_ROUTINE_HANDLER(BindTextureToArray) {
    const textureReference *texref = getTexture((const textureReference *) input_buffer->Get<uint64_t>());
    cudaArray *array = input_buffer->GetFromMarshal<cudaArray *>();
    cudaChannelFormatDesc *desc = input_buffer->Assign<cudaChannelFormatDesc>();

    cudaError_t exit_code = cudaBindTextureToArray(texref, array, desc);

    return new Result(exit_code);
}

CUDA_ROUTINE_HANDLER(GetChannelDesc) {
    cudaChannelFormatDesc *guestDesc =
            input_buffer->Assign<cudaChannelFormatDesc>();
    cudaArray *array = (cudaArray *) input_buffer->GetFromMarshal<cudaArray *>();
    Buffer *out = new Buffer();
    cudaChannelFormatDesc *desc = out->Delegate<cudaChannelFormatDesc>();
    memmove(desc, guestDesc, sizeof(cudaChannelFormatDesc));

    cudaError_t exit_code = cudaGetChannelDesc(desc, array);

    return new Result(exit_code, out);
}

CUDA_ROUTINE_HANDLER(GetTextureAlignmentOffset) {
    Buffer *out = new Buffer();
    size_t *offset = out->Delegate<size_t>();
    *offset = *(input_buffer->Assign<size_t>());
    char * texrefHandler = input_buffer->AssignString();
    textureReference *guestTexref = input_buffer->Assign<textureReference>();
    textureReference *texref = pThis->GetTexture(texrefHandler);
    memmove(texref, guestTexref, sizeof(textureReference));

    cudaError_t exit_code = cudaGetTextureAlignmentOffset(offset, texref);

    return new Result(exit_code, out);
}

CUDA_ROUTINE_HANDLER(GetTextureReference) {
    textureReference *texref;
    char *symbol_handler = input_buffer->AssignString();
    char *symbol = input_buffer->AssignString();

    char *our_symbol = const_cast<char *> (pThis->GetVar(symbol_handler));
    if (our_symbol != NULL)
        symbol = const_cast<char *> (our_symbol);

    cudaError_t exit_code = cudaGetTextureReference(
            (const textureReference ** ) &texref, symbol);

    Buffer *out = new Buffer();
    if(exit_code == cudaSuccess)
        out->AddString(pThis->GetTextureHandler(texref));
    else
        out->AddString("0x0");

    return new Result(exit_code, out);
}


CUDA_ROUTINE_HANDLER(UnbindTexture) {
    char * texrefHandler = input_buffer->AssignString();
    textureReference *texref = pThis->GetTexture(texrefHandler);

    cudaError_t exit_code = cudaUnbindTexture(texref);

    return new Result(exit_code);
}
