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
 * Written by: Flora Giannone <flora.giannone@studenti.uniparthenope.it>,
 *             Department of Applied Science
 */

#include <cstring>
#include "CudaDrFrontend.h"
#include "CudaUtil.h"
#include "CudaDr.h"
#include <cuda.h>
#include <stdio.h>


using namespace std;

/*Returns the compute capability of the device*/
extern CUresult cuDeviceComputeCapability(int *major, int *minor, CUdevice dev) {
    CudaDrFrontend::Prepare();
    CudaDrFrontend::AddHostPointerForArguments(major);
    CudaDrFrontend::AddHostPointerForArguments(minor);
    CudaDrFrontend::AddVariableForArguments(dev);
    CudaDrFrontend::Execute("cuDeviceComputeCapability");
    if (CudaDrFrontend::Success()) {
        *major = *(CudaDrFrontend::GetOutputHostPointer<int>());
        *minor = *(CudaDrFrontend::GetOutputHostPointer<int>());
    }
    return (CUresult) (CudaDrFrontend::GetExitCode());
}

/*Returns a handle to a compute device*/
extern CUresult cuDeviceGet(CUdevice *device, int ordinal) {
    CudaDrFrontend::Prepare();
    CudaDrFrontend::AddHostPointerForArguments(device);
    CudaDrFrontend::AddVariableForArguments(ordinal);
    CudaDrFrontend::Execute("cuDeviceGet");
    if (CudaDrFrontend::Success()) {
        *device = *(CudaDrFrontend::GetOutputHostPointer<CUdevice > ());
    }
    return (CUresult) (CudaDrFrontend::GetExitCode());
}

/*Returns information about the device*/
extern CUresult cuDeviceGetAttribute(int *pi, CUdevice_attribute attrib, CUdevice dev) {
    CudaDrFrontend::Prepare();
    CudaDrFrontend::AddHostPointerForArguments(pi);
    CudaDrFrontend::AddVariableForArguments(attrib);
    CudaDrFrontend::AddVariableForArguments(dev);
    CudaDrFrontend::Execute("cuDeviceGetAttribute");
    if (CudaDrFrontend::Success())
        *pi = *(CudaDrFrontend::GetOutputHostPointer<int>());
    return (CUresult) (CudaDrFrontend::GetExitCode());
}

/*Returns the number of compute-capable devices. */
extern CUresult cuDeviceGetCount(int *count) {
    CudaDrFrontend::Prepare();
    CudaDrFrontend::AddHostPointerForArguments(count);
    CudaDrFrontend::Execute("cuDeviceGetCount");
    if (CudaDrFrontend::Success())
        *count = *(CudaDrFrontend::GetOutputHostPointer<int>());
    return (CUresult) (CudaDrFrontend::GetExitCode());
}

/*Returns an identifer string for the device.*/
extern CUresult cuDeviceGetName(char *name, int len, CUdevice dev) {
    CudaDrFrontend::Prepare();
    CudaDrFrontend::AddStringForArguments((char *) name);
    CudaDrFrontend::AddVariableForArguments(len);
    CudaDrFrontend::AddVariableForArguments(dev);
    char *temp = NULL;
    CudaDrFrontend::Execute("cuDeviceGetName");
    if (CudaDrFrontend::Success())
        temp = (CudaDrFrontend::GetOutputString());
    strcpy(name, temp);
    return (CUresult) (CudaDrFrontend::GetExitCode());
}

/*Returns properties for a selected device.*/
extern CUresult cuDeviceGetProperties(CUdevprop *prop, CUdevice dev) {
    CudaDrFrontend::Prepare();
    CudaDrFrontend::AddHostPointerForArguments(prop);
    CudaDrFrontend::AddVariableForArguments(dev);
    CudaDrFrontend::Execute("cuDeviceGetProperties");
    if (CudaDrFrontend::Success()) {
        memmove(prop, CudaDrFrontend::GetOutputHostPointer<CUdevprop > (), sizeof (CUdevprop));
    }
    return (CUresult) (CudaDrFrontend::GetExitCode());
}

/*Returns the total amount of memory on the device. */
extern CUresult cuDeviceTotalMem(size_t *bytes, CUdevice dev) {
    CudaDrFrontend::Prepare();
    CudaDrFrontend::AddHostPointerForArguments(bytes);
    CudaDrFrontend::AddVariableForArguments(dev);
    CudaDrFrontend::Execute("cuDeviceTotalMem");
    if (CudaDrFrontend::Success())
        *bytes = *(CudaDrFrontend::GetOutputHostPointer<size_t > ());
    return (CUresult) (CudaDrFrontend::GetExitCode());
}

#if (__CUDA_API_VERSION >= 7000)
extern cudaError_t cudaDeviceGetP2PAttribute ( int* value, cudaDeviceP2PAttr attr, int  srcDevice, int  dstDevice ) {
    cudaError_t error;
    return error;
}
#endif
