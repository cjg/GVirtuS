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

#ifndef CUDART_VERSION
#error CUDART_VERSION not defined
#endif

using namespace std;

extern "C" __host__ cudaError_t CUDARTAPI cudaEventCreate(cudaEvent_t *event) {
  CudaRtFrontend::Prepare();
#if CUDART_VERSION >= 3010
  CudaRtFrontend::Execute("cudaEventCreate");
  if (CudaRtFrontend::Success())
    *event = (cudaEvent_t)CudaRtFrontend::GetOutputDevicePointer();
#else
  CudaRtFrontend::AddHostPointerForArguments(event);
  CudaRtFrontend::Execute("cudaEventCreate");
  if (CudaRtFrontend::Success())
    *event = *(CudaRtFrontend::GetOutputHostPointer<cudaEvent_t>());
#endif
  return CudaRtFrontend::GetExitCode();
}

#if CUDART_VERSION >= 3000
extern "C" __host__ cudaError_t CUDARTAPI
cudaEventCreateWithFlags(cudaEvent_t *event, unsigned int flags) {
#else
extern "C" __host__ cudaError_t CUDARTAPI
cudaEventCreateWithFlags(cudaEvent_t *event, int flags) {
#endif
  CudaRtFrontend::Prepare();
#if CUDART_VERSION >= 3010
  CudaRtFrontend::AddVariableForArguments(flags);
  CudaRtFrontend::Execute("cudaEventCreateWithFlags");
  if (CudaRtFrontend::Success())
    *event = (cudaEvent_t)CudaRtFrontend::GetOutputDevicePointer();
#else
  CudaRtFrontend::AddHostPointerForArguments(event);
  CudaRtFrontend::AddVariableForArguments(flags);
  CudaRtFrontend::Execute("cudaEventCreateWithFlags");
  if (CudaRtFrontend::Success())
    *event = *(CudaRtFrontend::GetOutputHostPointer<cudaEvent_t>());
#endif
  return CudaRtFrontend::GetExitCode();
}

extern "C" __host__ cudaError_t CUDARTAPI cudaEventDestroy(cudaEvent_t event) {
  CudaRtFrontend::Prepare();
#if CUDART_VERSION >= 3010
  CudaRtFrontend::AddDevicePointerForArguments(event);
#else
  CudaRtFrontend::AddVariableForArguments(event);
#endif
  CudaRtFrontend::Execute("cudaEventDestroy");
  return CudaRtFrontend::GetExitCode();
}

extern "C" __host__ cudaError_t CUDARTAPI
cudaEventElapsedTime(float *ms, cudaEvent_t start, cudaEvent_t end) {
  CudaRtFrontend::Prepare();
  CudaRtFrontend::AddHostPointerForArguments(ms);
#if CUDART_VERSION >= 3010
  CudaRtFrontend::AddDevicePointerForArguments(start);
  CudaRtFrontend::AddDevicePointerForArguments(end);
#else
  CudaRtFrontend::AddVariableForArguments(start);
  CudaRtFrontend::AddVariableForArguments(end);
#endif
  CudaRtFrontend::Execute("cudaEventElapsedTime");
  if (CudaRtFrontend::Success())
    *ms = *(CudaRtFrontend::GetOutputHostPointer<float>());
  return CudaRtFrontend::GetExitCode();
}

extern "C" __host__ cudaError_t CUDARTAPI cudaEventQuery(cudaEvent_t event) {
  CudaRtFrontend::Prepare();
#if CUDART_VERSION >= 3010
  CudaRtFrontend::AddDevicePointerForArguments(event);
#else
  CudaRtFrontend::AddVariableForArguments(event);
#endif
  CudaRtFrontend::Execute("cudaEventQuery");
  return CudaRtFrontend::GetExitCode();
}

extern "C" __host__ cudaError_t CUDARTAPI cudaEventRecord(cudaEvent_t event,
                                                          cudaStream_t stream) {
  CudaRtFrontend::Prepare();
#if CUDART_VERSION >= 3010
  CudaRtFrontend::AddDevicePointerForArguments(event);
  CudaRtFrontend::AddDevicePointerForArguments(stream);
#else
  CudaRtFrontend::AddVariableForArguments(event);
  CudaRtFrontend::AddVariableForArguments(stream);
#endif
  CudaRtFrontend::Execute("cudaEventRecord");
  return CudaRtFrontend::GetExitCode();
}

extern "C" __host__ cudaError_t CUDARTAPI
cudaEventSynchronize(cudaEvent_t event) {
  CudaRtFrontend::Prepare();
#if CUDART_VERSION >= 3010
  CudaRtFrontend::AddDevicePointerForArguments(event);
#else
  CudaRtFrontend::AddVariableForArguments(event);
#endif
  CudaRtFrontend::Execute("cudaEventSynchronize");
  return CudaRtFrontend::GetExitCode();
}
