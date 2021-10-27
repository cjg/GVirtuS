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

#if defined(__arm__) || defined(__aarch64__)
#include <GL/gl.h>
#endif

#include <cuda_gl_interop.h>

#include <iostream>

using namespace std;

CUDA_ROUTINE_HANDLER(GLSetGLDevice) {
  try {
    int device = input_buffer->Get<int>();
    cudaError_t exit_code = cudaGLSetGLDevice(device);
    return std::make_shared<Result>(exit_code);
  } catch (string e) {
    cerr << e << endl;
    return std::make_shared<Result>(cudaErrorMemoryAllocation);
  }
}

CUDA_ROUTINE_HANDLER(GraphicsGLRegisterBuffer) {
  struct cudaGraphicsResource *resource = NULL;
  try {
    GLuint buffer = input_buffer->Get<GLuint>();
    unsigned int flags = input_buffer->Get<unsigned int>();
    cudaError_t exit_code =
        cudaGraphicsGLRegisterBuffer(&resource, buffer, flags);
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();

    out->Add((pointer_t)resource);
    return std::make_shared<Result>(exit_code, out);
  } catch (string e) {
    cerr << e << endl;
    return std::make_shared<Result>(cudaErrorMemoryAllocation);
  }
}

CUDA_ROUTINE_HANDLER(GraphicsMapResources) {
  try {
    int count = input_buffer->Get<int>();
    cudaGraphicsResource_t *resources = new cudaGraphicsResource_t[count];
    for (int i = 0; i < count; i++)
      resources[i] = (cudaGraphicsResource_t)input_buffer->Get<pointer_t>();
    cudaStream_t stream = (cudaStream_t)input_buffer->Get<pointer_t>();
    cudaError_t exit_code = cudaGraphicsMapResources(count, resources, stream);
    return std::make_shared<Result>(exit_code);
  } catch (string e) {
    cerr << e << endl;
    return std::make_shared<Result>(cudaErrorMemoryAllocation);
  }
}

CUDA_ROUTINE_HANDLER(GraphicsResourceGetMappedPointer) {
  void *devPtr;
  size_t size;
  try {
    cudaGraphicsResource_t resource =
        (cudaGraphicsResource_t)input_buffer->Get<pointer_t>();
    cudaError_t exit_code =
        cudaGraphicsResourceGetMappedPointer(&devPtr, &size, resource);

    if (exit_code == cudaSuccess) {
      std::shared_ptr<Buffer> out = std::make_shared<Buffer>();

      out->Add((pointer_t)devPtr);
      out->Add(size);
      return std::make_shared<Result>(exit_code, out);
    }

    return std::make_shared<Result>(exit_code);
  } catch (string e) {
    cerr << e << endl;
    return std::make_shared<Result>(cudaErrorMemoryAllocation);
  }
}

CUDA_ROUTINE_HANDLER(GraphicsUnmapResources) {
  try {
    int count = input_buffer->Get<int>();
    cudaGraphicsResource_t *resources = new cudaGraphicsResource_t[count];
    for (int i = 0; i < count; i++)
      resources[i] = (cudaGraphicsResource_t)input_buffer->Get<pointer_t>();
    cudaStream_t stream = (cudaStream_t)input_buffer->Get<pointer_t>();
    cudaError_t exit_code =
        cudaGraphicsUnmapResources(count, resources, stream);
    return std::make_shared<Result>(exit_code);
  } catch (string e) {
    cerr << e << endl;
    return std::make_shared<Result>(cudaErrorMemoryAllocation);
  }
}

CUDA_ROUTINE_HANDLER(GraphicsUnregisterResource) {
  try {
    cudaGraphicsResource_t resource =
        (cudaGraphicsResource_t)input_buffer->Get<pointer_t>();
    return std::make_shared<Result>(cudaGraphicsUnregisterResource(resource));
  } catch (string e) {
    cerr << e << endl;
    return std::make_shared<Result>(cudaErrorMemoryAllocation);
  }
}

CUDA_ROUTINE_HANDLER(GraphicsResourceSetMapFlags) {
  try {
    cudaGraphicsResource_t resource =
        (cudaGraphicsResource_t)input_buffer->Get<pointer_t>();
    unsigned int flags = input_buffer->Get<unsigned int>();
    return std::make_shared<Result>(
        cudaGraphicsResourceSetMapFlags(resource, flags));
  } catch (string e) {
    cerr << e << endl;
    return std::make_shared<Result>(cudaErrorMemoryAllocation);
  }
}
