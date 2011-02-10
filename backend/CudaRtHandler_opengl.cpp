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

#include <cuda_gl_interop.h>

#include <iostream>

using namespace std;

CUDA_ROUTINE_HANDLER(GLSetGLDevice) {
    int device = input_buffer->Get<int>();

    cudaError_t exit_code = cudaGLSetGLDevice(device);

    return new Result(exit_code);
}

CUDA_ROUTINE_HANDLER(GraphicsGLRegisterBuffer) {
    struct cudaGraphicsResource *resource = NULL;
    GLuint buffer = input_buffer->Get<GLuint>();
    unsigned int flags = input_buffer->Get<unsigned int>();
    cudaError_t exit_code = cudaGraphicsGLRegisterBuffer(&resource, buffer,
            flags);

    Buffer *out = new Buffer();
    out->Add((uint64_t) resource);

    return new Result(exit_code, out);
}

CUDA_ROUTINE_HANDLER(GraphicsMapResources) {
    int count = input_buffer->Get<int>();
    cudaGraphicsResource_t *resources = new cudaGraphicsResource_t[count];
    for(int i = 0; i < count; i++)
        resources[i] = (cudaGraphicsResource_t) input_buffer->Get<uint64_t>();
    cudaStream_t stream = (cudaStream_t) input_buffer->Get<uint64_t>();

    cudaError_t exit_code = cudaGraphicsMapResources(count, resources, stream);

    return new Result(exit_code);
}

CUDA_ROUTINE_HANDLER(GraphicsResourceGetMappedPointer) {
    void *devPtr;
    size_t size;
    cudaGraphicsResource_t resource = (cudaGraphicsResource_t) input_buffer->Get<uint64_t>();

    cudaError_t exit_code = cudaGraphicsResourceGetMappedPointer(&devPtr, &size,
     resource);

    if(exit_code == cudaSuccess) {
        Buffer *out = new Buffer();
        out->Add((uint64_t) devPtr);
        out->Add(size);
        return new Result(exit_code, out);
    }

    return new Result(exit_code);
}

CUDA_ROUTINE_HANDLER(GraphicsUnmapResources) {
    int count = input_buffer->Get<int>();
    cudaGraphicsResource_t *resources = new cudaGraphicsResource_t[count];
    for(int i = 0; i < count; i++)
        resources[i] = (cudaGraphicsResource_t) input_buffer->Get<uint64_t>();
    cudaStream_t stream = (cudaStream_t) input_buffer->Get<uint64_t>();

    cudaError_t exit_code = cudaGraphicsUnmapResources(count, resources, stream);

    return new Result(exit_code);
}

CUDA_ROUTINE_HANDLER(GraphicsUnregisterResource) {
    cudaGraphicsResource_t resource = (cudaGraphicsResource_t) input_buffer->Get<uint64_t>();
    return new Result(cudaGraphicsUnregisterResource(resource));
}

CUDA_ROUTINE_HANDLER(GraphicsResourceSetMapFlags) {
    cudaGraphicsResource_t resource = (cudaGraphicsResource_t) input_buffer->Get<uint64_t>();
    unsigned int flags = input_buffer->Get<unsigned int>();
    return new Result(cudaGraphicsResourceSetMapFlags(resource, flags));
}
