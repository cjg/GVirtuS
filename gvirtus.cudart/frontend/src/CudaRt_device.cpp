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

extern "C" __host__ cudaError_t
CUDARTAPI cudaDeviceSetCacheConfig(cudaFuncCache cacheConfig) {
  CudaRtFrontend::Prepare();
  CudaRtFrontend::AddVariableForArguments(cacheConfig);
  CudaRtFrontend::Execute("cudaDeviceSetCacheConfig");

  return CudaRtFrontend::GetExitCode();
}

extern "C" __host__ cudaError_t
CUDARTAPI cudaChooseDevice(int *device, const cudaDeviceProp *prop) {
  CudaRtFrontend::Prepare();
  CudaRtFrontend::AddHostPointerForArguments(device);
  CudaRtFrontend::AddHostPointerForArguments(prop);
  CudaRtFrontend::Execute("cudaChooseDevice");
  if (CudaRtFrontend::Success())
    *device = *(CudaRtFrontend::GetOutputHostPointer<int>());
  return CudaRtFrontend::GetExitCode();
}

extern "C" __host__ cudaError_t
CUDARTAPI cudaGetDevice(int *device) {
  CudaRtFrontend::Prepare();
  CudaRtFrontend::AddHostPointerForArguments(device);
  CudaRtFrontend::Execute("cudaGetDevice");
  if (CudaRtFrontend::Success())
    *device = *(CudaRtFrontend::GetOutputHostPointer<int>());
  return CudaRtFrontend::GetExitCode();

}

extern "C" __host__ cudaError_t
CUDARTAPI cudaGetDeviceCount(int *count) {
  CudaRtFrontend::Prepare();
  CudaRtFrontend::AddHostPointerForArguments(count);
  CudaRtFrontend::Execute("cudaGetDeviceCount");
  if (CudaRtFrontend::Success())
    *count = *(CudaRtFrontend::GetOutputHostPointer<int>());
  return CudaRtFrontend::GetExitCode();
}

extern "C" __host__ cudaError_t
CUDARTAPI cudaGetDeviceProperties(cudaDeviceProp *prop, int device) {
  CudaRtFrontend::Prepare();
  CudaRtFrontend::AddHostPointerForArguments(prop);
  CudaRtFrontend::AddVariableForArguments(device);
  CudaRtFrontend::Execute("cudaGetDeviceProperties");
  if (CudaRtFrontend::Success()) {
    memmove(prop, CudaRtFrontend::GetOutputHostPointer<cudaDeviceProp>(),
            sizeof(cudaDeviceProp));
#ifndef CUDA_VERSION
#error CUDA_VERSION not defined
#endif
#if CUDA_VERSION >= 2030
    prop->canMapHostMemory = 0;
#endif
  }
  return CudaRtFrontend::GetExitCode();
}

extern "C" __host__ cudaError_t
cudaDeviceGetAttribute(int *value, cudaDeviceAttr attr, int device) {
  CudaRtFrontend::Prepare();
  CudaRtFrontend::AddHostPointerForArguments(value);
  CudaRtFrontend::AddVariableForArguments(attr);
  CudaRtFrontend::AddVariableForArguments(device);

  CudaRtFrontend::Execute("cudaDeviceGetAttribute");
  if (CudaRtFrontend::Success())
    *value = *(CudaRtFrontend::GetOutputHostPointer<int>());
  return CudaRtFrontend::GetExitCode();

}

extern "C" __host__ cudaError_t
CUDARTAPI cudaSetDevice(int device) {
  CudaRtFrontend::Prepare();
  CudaRtFrontend::AddVariableForArguments(device);
  CudaRtFrontend::Execute("cudaSetDevice");
  return CudaRtFrontend::GetExitCode();
}

#if CUDART_VERSION >= 3000
extern "C" __host__ cudaError_t CUDARTAPI cudaSetDeviceFlags(unsigned int flags) {
#else
extern "C" __host__ cudaError_t
CUDARTAPI cudaSetDeviceFlags(int flags) {
#endif
  CudaRtFrontend::Prepare();
  CudaRtFrontend::AddVariableForArguments(flags);
  CudaRtFrontend::Execute("cudaSetDeviceFlags");
  return CudaRtFrontend::GetExitCode();
}

extern "C" __host__ cudaError_t
CUDARTAPI cudaDeviceReset(void) {
  CudaRtFrontend::Prepare();
  CudaRtFrontend::Execute("cudaDeviceReset");
  return CudaRtFrontend::GetExitCode();

}

extern "C" __host__ cudaError_t
CUDARTAPI cudaDeviceSynchronize(void) {
  CudaRtFrontend::Prepare();
  CudaRtFrontend::Execute("cudaDeviceSynchronize");
  return CudaRtFrontend::GetExitCode();

}

extern "C" __host__ cudaError_t
CUDARTAPI cudaSetValidDevices(int *device_arr, int len) {
  CudaRtFrontend::Prepare();
  CudaRtFrontend::AddHostPointerForArguments(device_arr, len);
  CudaRtFrontend::AddVariableForArguments(len);
  CudaRtFrontend::Execute("cudaSetValidDevices");
  if (CudaRtFrontend::Success()) {
    int *out_device_arr = CudaRtFrontend::GetOutputHostPointer<int>();
    memmove(device_arr, out_device_arr, sizeof(int) * len);
  }
  return CudaRtFrontend::GetExitCode();
}
//testing vpelliccia
extern "C" __host__ cudaError_t
CUDARTAPI cudaIpcGetMemHandle(cudaIpcMemHandle_t *handle, void *devPtr) {
  CudaRtFrontend::Prepare();
  CudaRtFrontend::AddHostPointerForArguments(handle);
  CudaRtFrontend::AddDevicePointerForArguments(devPtr);
  CudaRtFrontend::Execute("cudaIpcGetMemHandle");
  if (CudaRtFrontend::Success())
    *handle = *(CudaRtFrontend::GetOutputHostPointer<cudaIpcMemHandle_t>());
  return CudaRtFrontend::GetExitCode();
}

extern "C" __host__ cudaError_t
CUDARTAPI cudaIpcGetEventHandle(cudaIpcEventHandle_t *handle, cudaEvent_t event) {
  CudaRtFrontend::Prepare();
  CudaRtFrontend::AddHostPointerForArguments(handle);
  CudaRtFrontend::AddDevicePointerForArguments(event);
  CudaRtFrontend::Execute("cudaIpcGetEventHandle");
  if (CudaRtFrontend::Success())
    *handle = *(CudaRtFrontend::GetOutputHostPointer<cudaIpcEventHandle_t>());
  return CudaRtFrontend::GetExitCode();
}
extern "C" __host__ cudaError_t
CUDARTAPI cudaDeviceSetLimit(cudaLimit limit, size_t value) {
  CudaRtFrontend::Prepare();
  CudaRtFrontend::AddVariableForArguments(limit);
  CudaRtFrontend::AddVariableForArguments(value);
  CudaRtFrontend::Execute("cudaDeviceSetLimit");
  return CudaRtFrontend::GetExitCode();
}

extern "C" __host__ cudaError_t
CUDARTAPI cudaIpcOpenEventHandle(cudaEvent_t *event, cudaIpcEventHandle_t handle) {
  CudaRtFrontend::Prepare();
  CudaRtFrontend::AddHostPointerForArguments(event);
  CudaRtFrontend::AddVariableForArguments(handle);
  CudaRtFrontend::Execute("cudaIpcOpenEventHandle");
  if (CudaRtFrontend::Success())
    *event = *(CudaRtFrontend::GetOutputHostPointer<cudaEvent_t>());
  return CudaRtFrontend::GetExitCode();
}

extern "C" __host__ cudaError_t
CUDARTAPI cudaIpcOpenMemHandle(void **devPtr, cudaIpcMemHandle_t handle, unsigned int flags) {
  CudaRtFrontend::Prepare();
  CudaRtFrontend::AddVariableForArguments(handle);
  CudaRtFrontend::AddVariableForArguments(flags);
  CudaRtFrontend::Execute("cudaIpcOpenMemHandle");
  if (CudaRtFrontend::Success())
    *devPtr = CudaRtFrontend::GetOutputDevicePointer();

  return CudaRtFrontend::GetExitCode();
}

extern "C" __host__ cudaError_t
CUDARTAPI cudaDeviceEnablePeerAccess(int peerDevice, unsigned int flags) {
  CudaRtFrontend::Prepare();
  CudaRtFrontend::AddVariableForArguments(peerDevice);
  CudaRtFrontend::AddVariableForArguments(flags);
  CudaRtFrontend::Execute("cudaDeviceEnablePeerAccess");
  return CudaRtFrontend::GetExitCode();
}

extern "C" __host__ cudaError_t
CUDARTAPI cudaDeviceCanAccessPeer(int *canAccessPeer, int device, int peerDevice) {
  CudaRtFrontend::Prepare();
  CudaRtFrontend::AddHostPointerForArguments(canAccessPeer);
  CudaRtFrontend::AddVariableForArguments(device);
  CudaRtFrontend::AddVariableForArguments(peerDevice);

  CudaRtFrontend::Execute("cudaDeviceCanAccessPeer");
  if (CudaRtFrontend::Success())
    *canAccessPeer = *(CudaRtFrontend::GetOutputHostPointer<int>());
  return CudaRtFrontend::GetExitCode();
}

extern "C" __host__ cudaError_t
CUDARTAPI cudaDeviceGetStreamPriorityRange(int *leastPriority, int *greatestPriority) {
  CudaRtFrontend::Prepare();
  CudaRtFrontend::AddHostPointerForArguments(leastPriority);
  CudaRtFrontend::AddHostPointerForArguments(greatestPriority);
  CudaRtFrontend::Execute("cudaDeviceGetStreamPriorityRange");
  if (CudaRtFrontend::Success())
    *leastPriority = *(CudaRtFrontend::GetOutputHostPointer<int>());
  *greatestPriority = *(CudaRtFrontend::GetOutputHostPointer<int>());
  return CudaRtFrontend::GetExitCode();
}

extern "C" __host__ cudaError_t
CUDARTAPI cudaDeviceDisablePeerAccess(int peerDevice) {
  CudaRtFrontend::Prepare();
  CudaRtFrontend::AddVariableForArguments(peerDevice);

  CudaRtFrontend::Execute("cudaDeviceDisablePeerAccess");
  return CudaRtFrontend::GetExitCode();
}


