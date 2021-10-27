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

#ifndef CUDART_VERSION
#error CUDART_VERSION not defined
#endif

#if CUDART_VERSION <= 2030

#include <cuda_gl_interop.h>

extern "C" __host__ cudaError_t CUDARTAPI cudaGLMapBufferObject(void **devPtr,
                                                                GLuint bufObj) {
  cerr << "I'm sorry but it isn't possibile to use OpenGL Interoperability "
       << "API." << endl
       << "Giving up ..." << endl;
  exit(-1);
  return cudaErrorUnknown;
}

extern "C" __host__ cudaError_t CUDARTAPI
cudaGLMapBufferObjectAsync(void **devPtr, GLuint bufObj, cudaStream_t stream) {
  cerr << "I'm sorry but it isn't possibile to use OpenGL Interoperability "
       << "API." << endl
       << "Giving up ..." << endl;
  exit(-1);
  return cudaErrorUnknown;
}

extern "C" __host__ cudaError_t CUDARTAPI
cudaGLRegisterBufferObject(GLuint bufObj) {
  cerr << "I'm sorry but it isn't possibile to use OpenGL Interoperability "
       << "API." << endl
       << "Giving up ..." << endl;
  exit(-1);
  return cudaErrorUnknown;
}

extern "C" __host__ cudaError_t CUDARTAPI
cudaGLSetBufferObjectMapFlags(GLuint bufObj, unsigned int flags) {
  cerr << "I'm sorry but it isn't possibile to use OpenGL Interoperability "
       << "API." << endl
       << "Giving up ..." << endl;
  exit(-1);
  return cudaErrorUnknown;
}

extern "C" __host__ cudaError_t CUDARTAPI cudaGLSetGLDevice(int device) {
  cerr << "I'm sorry but it isn't possibile to use OpenGL Interoperability "
       << "API." << endl
       << "Giving up ..." << endl;
  exit(-1);
  return cudaErrorUnknown;
}

extern "C" __host__ cudaError_t CUDARTAPI
cudaGLUnmapBufferObject(GLuint bufObj) {
  cerr << "I'm sorry but it isn't possibile to use OpenGL Interoperability "
       << "API." << endl
       << "Giving up ..." << endl;
  exit(-1);
  return cudaErrorUnknown;
}

extern "C" __host__ cudaError_t CUDARTAPI
cudaGLUnmapBufferObjectAsync(GLuint bufObj, cudaStream_t stream) {
  cerr << "I'm sorry but it isn't possibile to use OpenGL Interoperability "
       << "API." << endl
       << "Giving up ..." << endl;
  exit(-1);
  return cudaErrorUnknown;
}

extern "C" __host__ cudaError_t CUDARTAPI
cudaGLUnregisterBufferObject(GLuint bufObj) {
  cerr << "I'm sorry but it isn't possibile to use OpenGL Interoperability "
       << "API." << endl
       << "Giving up ..." << endl;
  exit(-1);
  return cudaErrorUnknown;
}
#else

#if defined(__arm__) || defined(__aarch64__)
#include <GL/gl.h>
#endif

#include <cuda_gl_interop.h>

// extern "C" void FlushRoutines();

extern "C" cudaError_t cudaGLSetGLDevice(int device) {
  // FlushRoutines();
  CudaRtFrontend::Prepare();
  CudaRtFrontend::AddVariableForArguments(device);
  CudaRtFrontend::Execute("cudaGLSetGLDevice");
  return CudaRtFrontend::GetExitCode();
}

extern "C" cudaError_t cudaGraphicsGLRegisterBuffer(
    struct cudaGraphicsResource **resource, GLuint buffer, unsigned int flags) {
  // FlushRoutines();
  CudaRtFrontend::Prepare();
  CudaRtFrontend::AddVariableForArguments(buffer);
  CudaRtFrontend::AddVariableForArguments(flags);
  CudaRtFrontend::Execute("cudaGraphicsGLRegisterBuffer");
  if (CudaRtFrontend::Success())
    *resource =
        (struct cudaGraphicsResource *)CudaRtFrontend::GetOutputDevicePointer();
  return CudaRtFrontend::GetExitCode();
}

extern "C" cudaError_t cudaGraphicsMapResources(
    int count, cudaGraphicsResource_t *resources, cudaStream_t stream) {
  CudaRtFrontend::Prepare();
  CudaRtFrontend::AddVariableForArguments(count);
  for (int i = 0; i < count; i++)
    CudaRtFrontend::AddDevicePointerForArguments(resources[i]);
  CudaRtFrontend::AddDevicePointerForArguments(stream);
  CudaRtFrontend::Execute("cudaGraphicsMapResources");
  return CudaRtFrontend::GetExitCode();
}

extern "C" cudaError_t cudaGraphicsResourceGetMappedPointer(
    void **devPtr, size_t *size, cudaGraphicsResource_t resource) {
  CudaRtFrontend::Prepare();
  CudaRtFrontend::AddDevicePointerForArguments(resource);
  CudaRtFrontend::Execute("cudaGraphicsResourceGetMappedPointer");
  if (CudaRtFrontend::Success()) {
    *devPtr = CudaRtFrontend::GetOutputDevicePointer();
    *size = CudaRtFrontend::GetOutputVariable<size_t>();
  }
  return CudaRtFrontend::GetExitCode();
}

extern "C" cudaError_t cudaGraphicsUnmapResources(
    int count, cudaGraphicsResource_t *resources, cudaStream_t stream) {
  CudaRtFrontend::Prepare();
  CudaRtFrontend::AddVariableForArguments(count);
  for (int i = 0; i < count; i++)
    CudaRtFrontend::AddDevicePointerForArguments(resources[i]);
  CudaRtFrontend::AddDevicePointerForArguments(stream);
  CudaRtFrontend::Execute("cudaGraphicsUnmapResources");
  return CudaRtFrontend::GetExitCode();
}

extern "C" cudaError_t cudaGraphicsUnregisterResource(
    cudaGraphicsResource_t resource) {
  CudaRtFrontend::Prepare();
  CudaRtFrontend::AddDevicePointerForArguments(resource);
  CudaRtFrontend::Execute("cudaGraphicsUnregisterResource");
  return CudaRtFrontend::GetExitCode();
}

extern "C" cudaError_t cudaGraphicsResourceSetMapFlags(
    struct cudaGraphicsResource *resource, unsigned int flags) {
  CudaRtFrontend::Prepare();
  CudaRtFrontend::AddDevicePointerForArguments(resource);
  CudaRtFrontend::AddVariableForArguments(flags);
  CudaRtFrontend::Execute("cudaGraphicsResourceSetMapFlags");
  return CudaRtFrontend::GetExitCode();
}

#endif
