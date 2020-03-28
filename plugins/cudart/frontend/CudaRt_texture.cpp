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

extern "C" __host__ cudaError_t CUDARTAPI cudaBindTexture(
    size_t *offset, const textureReference *texref, const void *devPtr,
    const cudaChannelFormatDesc *desc, size_t size) {
  CudaRtFrontend::Prepare();
  CudaRtFrontend::AddHostPointerForArguments(offset);
  // Achtung: passing the address and the content of the textureReference
  CudaRtFrontend::AddStringForArguments(CudaUtil::MarshalHostPointer(texref));
  CudaRtFrontend::AddHostPointerForArguments(texref);
  CudaRtFrontend::AddDevicePointerForArguments(devPtr);
  CudaRtFrontend::AddHostPointerForArguments(desc);
  CudaRtFrontend::AddVariableForArguments(size);
  CudaRtFrontend::Execute("cudaBindTexture");
  if (CudaRtFrontend::Success())
    *offset = *(CudaRtFrontend::GetOutputHostPointer<size_t>());
  return CudaRtFrontend::GetExitCode();
}

extern "C" __host__ cudaError_t CUDARTAPI
cudaBindTexture2D(size_t *offset, const textureReference *texref,
                  const void *devPtr, const cudaChannelFormatDesc *desc,
                  size_t width, size_t height, size_t pitch) {
  CudaRtFrontend::Prepare();
  CudaRtFrontend::AddHostPointerForArguments(offset);
  // Achtung: passing the address and the content of the textureReference
  CudaRtFrontend::AddStringForArguments(CudaUtil::MarshalHostPointer(texref));
  CudaRtFrontend::AddHostPointerForArguments(texref);
  CudaRtFrontend::AddDevicePointerForArguments(devPtr);
  CudaRtFrontend::AddHostPointerForArguments(desc);
  CudaRtFrontend::AddVariableForArguments(width);
  CudaRtFrontend::AddVariableForArguments(height);
  CudaRtFrontend::AddVariableForArguments(pitch);
  CudaRtFrontend::Execute("cudaBindTexture2D");
  size_t tempOffset;

  if (CudaRtFrontend::Success())
    tempOffset = *(CudaRtFrontend::GetOutputHostPointer<size_t>());

  if (offset != NULL) {
    *offset = tempOffset;
  }

  return CudaRtFrontend::GetExitCode();
}

extern "C" __host__ cudaError_t CUDARTAPI
cudaBindTextureToArray(const textureReference *texref, const cudaArray *array,
                       const cudaChannelFormatDesc *desc) {
  CudaRtFrontend::Prepare();
  // Achtung: passing the address and the content of the textureReference
  CudaRtFrontend::AddStringForArguments(CudaUtil::MarshalHostPointer(texref));
  CudaRtFrontend::AddHostPointerForArguments(texref);
  CudaRtFrontend::AddDevicePointerForArguments((void *)array);
  CudaRtFrontend::AddHostPointerForArguments(desc);
  CudaRtFrontend::Execute("cudaBindTextureToArray");
  return CudaRtFrontend::GetExitCode();
}

extern "C" __host__ cudaChannelFormatDesc CUDARTAPI
cudaCreateChannelDesc(int x, int y, int z, int w, cudaChannelFormatKind f) {
  cudaChannelFormatDesc desc;
  desc.x = x;
  desc.y = y;
  desc.z = z;
  desc.w = w;
  desc.f = f;
  return desc;
}

extern "C" __host__ cudaError_t CUDARTAPI
cudaGetChannelDesc(cudaChannelFormatDesc *desc, const cudaArray *array) {
  CudaRtFrontend::Prepare();
  CudaRtFrontend::AddHostPointerForArguments(desc);
  CudaRtFrontend::AddDevicePointerForArguments(array);
  CudaRtFrontend::Execute("cudaGetChannelDesc");
  if (CudaRtFrontend::Success())
    memmove(desc, CudaRtFrontend::GetOutputHostPointer<cudaChannelFormatDesc>(),
            sizeof(cudaChannelFormatDesc));
  return CudaRtFrontend::GetExitCode();
}

extern "C" __host__ cudaError_t CUDARTAPI
cudaGetTextureAlignmentOffset(size_t *offset, const textureReference *texref) {
  CudaRtFrontend::Prepare();
  CudaRtFrontend::AddHostPointerForArguments(offset);
  CudaRtFrontend::AddStringForArguments(CudaUtil::MarshalHostPointer(texref));
  CudaRtFrontend::Execute("cudaGetTextureAlignmentOffset");
  if (CudaRtFrontend::Success())
    *offset = *(CudaRtFrontend::GetOutputHostPointer<size_t>());
  return CudaRtFrontend::GetExitCode();
}

extern "C" __host__ cudaError_t CUDARTAPI
cudaGetTextureReference(const textureReference **texref, const void *symbol) {
  CudaRtFrontend::Prepare();
  // Achtung: skipping to add texref
  // Achtung: passing the address and the content of symbol
  CudaRtFrontend::AddStringForArguments(CudaUtil::MarshalHostPointer(symbol));
  CudaRtFrontend::AddStringForArguments((char *)symbol);
  CudaRtFrontend::Execute("cudaGetTextureReference");
  if (CudaRtFrontend::Success()) {
    char *texrefHandler = CudaRtFrontend::GetOutputString();
    *texref = (textureReference *)strtoul(texrefHandler, NULL, 16);
  }
  return CudaRtFrontend::GetExitCode();
}

extern "C" __host__ cudaError_t CUDARTAPI
cudaUnbindTexture(const textureReference *texref) {
  CudaRtFrontend::Prepare();
  CudaRtFrontend::AddStringForArguments(CudaUtil::MarshalHostPointer(texref));
  CudaRtFrontend::Execute("cudaUnbindTexture");
  return CudaRtFrontend::GetExitCode();
}

extern "C" __host__ cudaError_t CUDARTAPI cudaCreateTextureObject(
    cudaTextureObject_t *pTexObject, const struct cudaResourceDesc *pResDesc,
    const struct cudaTextureDesc *pTexDesc,
    const struct cudaResourceViewDesc *pResViewDesc) {
  Buffer *input_buffer = new Buffer();

  input_buffer->Add(pResDesc);
  CudaUtil::MarshalTextureDescForArguments(pTexDesc, input_buffer);
  input_buffer->Add(pResDesc);
  CudaRtFrontend::Prepare();
  CudaRtFrontend::Execute("cudaCreateTextureObject", input_buffer);
  if (CudaRtFrontend::Success())
    (*pTexObject) = CudaRtFrontend::GetOutputVariable<cudaTextureObject_t>();
  return CudaRtFrontend::GetExitCode();
}
