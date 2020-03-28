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
 * Written by: Carlo Palmieri <carlo.palmieri@uniparthenope.it>,
 *             Department of Science and technology
 */
#include <iostream>

#include "CudaRtHandler.h"

using namespace std;

// extern const surfaceReference *getSurface(const surfaceReference *handler);

CUDA_ROUTINE_HANDLER(BindSurfaceToArray) {
  char *surfrefHandler = input_buffer->AssignString();

  surfaceReference *guestSurfref = input_buffer->Assign<surfaceReference>();

  surfaceReference *surfref = pThis->GetSurface(surfrefHandler);
  cudaChannelFormatDesc *a = &(surfref->channelDesc);
  memmove(surfref, guestSurfref, sizeof(surfaceReference));
  cudaArray *array = (cudaArray *)input_buffer->Get<pointer_t>();
  cudaChannelFormatDesc *desc = input_buffer->Assign<cudaChannelFormatDesc>();

  cudaError_t exit_code = cudaBindSurfaceToArray(surfref, array, desc);

  return std::make_shared<Result>(exit_code);
}

// CUDA_ROUTINE_HANDLER(GetTextureReference) {
//    textureReference *texref;
//    char *symbol_handler = input_buffer->AssignString();
//    char *symbol = input_buffer->AssignString();
//
//    char *our_symbol = const_cast<char *> (pThis->GetVar(symbol_handler));
//    if (our_symbol != NULL)
//        symbol = const_cast<char *> (our_symbol);
//
//    cudaError_t exit_code = cudaGetTextureReference(
//            (const textureReference ** ) &texref, symbol);
//
//    Buffer *out = new Buffer();
//    if(exit_code == cudaSuccess)
//        out->AddString(pThis->GetTextureHandler(texref));
//    else
//        out->AddString("0x0");
//
//    return std::make_shared<Result>(exit_code, out);
//}
