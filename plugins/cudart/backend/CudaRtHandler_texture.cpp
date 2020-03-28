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

// extern const textureReference *getTexture(const textureReference *handler);

CUDA_ROUTINE_HANDLER(BindTexture) {
  try {
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();

    size_t *offset = out->Delegate<size_t>();
    *offset = *(input_buffer->Assign<size_t>());
    char *texrefHandler = input_buffer->AssignString();
    textureReference *guestTexref = input_buffer->Assign<textureReference>();
    textureReference *texref = pThis->GetTexture(texrefHandler);
    memmove(texref, guestTexref, sizeof(textureReference));

    void *devPtr = input_buffer->GetFromMarshal<void *>();
    cudaChannelFormatDesc *desc = input_buffer->Assign<cudaChannelFormatDesc>();
    size_t size = input_buffer->Get<size_t>();
    cudaError_t exit_code = cudaBindTexture(offset, texref, devPtr, desc, size);
    return std::make_shared<Result>(exit_code, out);
  } catch (string e) {
    cerr << e << endl;
    return std::make_shared<Result>(cudaErrorMemoryAllocation);
  }
}

#ifndef CUDART_VERSION
#error CUDART_VERSION not defined
#endif
#if CUDART_VERSION >= 2030

CUDA_ROUTINE_HANDLER(BindTexture2D) {
  try {
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();

    size_t *offset = out->Delegate<size_t>();
    size_t *temp = input_buffer->Assign<size_t>();
    if (temp != NULL)
      *offset = *temp;
    else
      *offset = 0;
    char *texrefHandler = input_buffer->AssignString();
    textureReference *guestTexref = input_buffer->Assign<textureReference>();

    textureReference *texref = pThis->GetTexture(texrefHandler);
    memmove(texref, guestTexref, sizeof(textureReference));
    void *devPtr = (void *)input_buffer->Get<pointer_t>();
    cudaChannelFormatDesc *desc = input_buffer->Assign<cudaChannelFormatDesc>();
    size_t width = input_buffer->Get<size_t>();
    size_t height = input_buffer->Get<size_t>();
    size_t pitch = input_buffer->Get<size_t>();
    cudaError_t exit_code =
        cudaBindTexture2D(offset, texref, devPtr, desc, width, height, pitch);
    return std::make_shared<Result>(exit_code, out);
  } catch (string e) {
    cerr << e << endl;
    return std::make_shared<Result>(cudaErrorMemoryAllocation);
  }
}
#endif

CUDA_ROUTINE_HANDLER(BindTextureToArray) {
  try {
    char *texrefHandler = input_buffer->AssignString();
    textureReference *guestTexref = input_buffer->Assign<textureReference>();
    textureReference *texref = pThis->GetTexture(texrefHandler);
    memmove(texref, guestTexref, sizeof(textureReference));
    cudaArray *array = (cudaArray *)input_buffer->Get<pointer_t>();
    cudaChannelFormatDesc *desc = input_buffer->Assign<cudaChannelFormatDesc>();
    cudaError_t exit_code = cudaBindTextureToArray(texref, array, desc);
    return std::make_shared<Result>(exit_code);
  } catch (string e) {
    cerr << e << endl;
    return std::make_shared<Result>(cudaErrorMemoryAllocation);
  }
}

CUDA_ROUTINE_HANDLER(GetChannelDesc) {
  try {
    cudaChannelFormatDesc *guestDesc =
        input_buffer->Assign<cudaChannelFormatDesc>();
    cudaArray *array = (cudaArray *)input_buffer->GetFromMarshal<cudaArray *>();
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();

    cudaChannelFormatDesc *desc = out->Delegate<cudaChannelFormatDesc>();
    memmove(desc, guestDesc, sizeof(cudaChannelFormatDesc));
    cudaError_t exit_code = cudaGetChannelDesc(desc, array);
    return std::make_shared<Result>(exit_code, out);
  } catch (string e) {
    cerr << e << endl;
    return std::make_shared<Result>(cudaErrorMemoryAllocation);
  }
}

CUDA_ROUTINE_HANDLER(GetTextureAlignmentOffset) {
  try {
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();

    size_t *offset = out->Delegate<size_t>();
    *offset = *(input_buffer->Assign<size_t>());
    char *texrefHandler = input_buffer->AssignString();
    textureReference *guestTexref = input_buffer->Assign<textureReference>();
    textureReference *texref = pThis->GetTexture(texrefHandler);
    memmove(texref, guestTexref, sizeof(textureReference));
    cudaError_t exit_code = cudaGetTextureAlignmentOffset(offset, texref);
    return std::make_shared<Result>(exit_code, out);
  } catch (string e) {
    cerr << e << endl;
    return std::make_shared<Result>(cudaErrorMemoryAllocation);
  }
}

CUDA_ROUTINE_HANDLER(GetTextureReference) {
  textureReference *texref;
  try {
    char *symbol_handler = input_buffer->AssignString();
    char *symbol = input_buffer->AssignString();
    char *our_symbol = const_cast<char *>(pThis->GetVar(symbol_handler));
    if (our_symbol != NULL) symbol = const_cast<char *>(our_symbol);

    cudaError_t exit_code =
        cudaGetTextureReference((const textureReference **)&texref, symbol);

    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();

    if (exit_code == cudaSuccess)
      out->AddString(pThis->GetTextureHandler(texref));
    else
      out->AddString("0x0");
    return std::make_shared<Result>(exit_code, out);
  } catch (string e) {
    cerr << e << endl;
    return std::make_shared<Result>(cudaErrorMemoryAllocation);
  }
}

CUDA_ROUTINE_HANDLER(UnbindTexture) {
  try {
    char *texrefHandler = input_buffer->AssignString();
    textureReference *texref = pThis->GetTexture(texrefHandler);
    cudaError_t exit_code = cudaUnbindTexture(texref);
    return std::make_shared<Result>(exit_code);
  } catch (string e) {
    cerr << e << endl;
    return std::make_shared<Result>(cudaErrorMemoryAllocation);
  }
}

CUDA_ROUTINE_HANDLER(CreateTextureObject) {
  cudaTextureObject_t tex = 0;
  std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
  try {
    cudaResourceDesc *pResDesc = input_buffer->Assign<cudaResourceDesc>();
    cudaTextureDesc *pTexDesc =
        CudaUtil::UnmarshalTextureDesc(input_buffer.get());
    cudaResourceViewDesc *pResViewDesc =
        input_buffer->Assign<cudaResourceViewDesc>();

    cudaError_t exit_code =
        cudaCreateTextureObject(&tex, pResDesc, pTexDesc, pResViewDesc);

    out->Add<cudaTextureObject_t>(tex);
    return std::make_shared<Result>(exit_code, out);

  } catch (string e) {
    cerr << e << endl;
    return std::make_shared<Result>(cudaErrorUnknown);
  }
}
