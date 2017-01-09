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

/*Create a CUDA context*/
extern CUresult cuCtxCreate(CUcontext *pctx, unsigned int flags, CUdevice dev) {
    CudaDrFrontend::Prepare();
    CudaDrFrontend::AddVariableForArguments(flags);
    CudaDrFrontend::AddVariableForArguments(dev);
    CudaDrFrontend::Execute("cuCtxCreate");
    if (CudaDrFrontend::Success()){
        *pctx = (CUcontext) (CudaDrFrontend::GetOutputDevicePointer());

    }
    return (CUresult) (CudaDrFrontend::GetExitCode());
}

/*Increment a context's usage-count*/
extern CUresult cuCtxAttach(CUcontext *pctx, unsigned int flags) {
    CudaDrFrontend::Prepare();
    CudaDrFrontend::AddVariableForArguments(flags);
    CudaDrFrontend::AddHostPointerForArguments(pctx);
    CudaDrFrontend::Execute("cuCtxAttach");
    if (CudaDrFrontend::Success())
        *pctx = (CUcontext) CudaDrFrontend::GetOutputDevicePointer();
    return (CUresult) (CudaDrFrontend::GetExitCode());
}

/*Destroy the current context or a floating CUDA context*/
extern CUresult cuCtxDestroy(CUcontext ctx) {
    CudaDrFrontend::Prepare();
    CudaDrFrontend::AddDevicePointerForArguments((void*) ctx);
    CudaDrFrontend::Execute("cuCtxDestroy");
    return (CUresult) CudaDrFrontend::GetExitCode();
}

/*Decrement a context's usage-count. */
extern CUresult cuCtxDetach(CUcontext ctx) {
    CudaDrFrontend::Prepare();
    CudaDrFrontend::AddDevicePointerForArguments((void*) ctx);
    CudaDrFrontend::Execute("cuCtxDetach");
    return (CUresult) CudaDrFrontend::GetExitCode();
}

/*Returns the device ID for the current context.*/
extern CUresult cuCtxGetDevice(CUdevice *device) {
    CudaDrFrontend::Prepare();
    CudaDrFrontend::AddHostPointerForArguments(device);
    CudaDrFrontend::Execute("cuCtxGetDevice");
    if (CudaDrFrontend::Success()) {
        *device = *(CudaDrFrontend::GetOutputHostPointer<CUdevice > ());
    }
    return (CUresult) CudaDrFrontend::GetExitCode();
}

/*Pops the current CUDA context from the current CPU thread.*/
extern CUresult cuCtxPopCurrent(CUcontext *pctx) {
    CudaDrFrontend::Prepare();
    CUcontext ctx;
    pctx = &ctx;
    CudaDrFrontend::Execute("cuCtxPopCurrent");
    if (CudaDrFrontend::Success())
        *pctx = (CUcontext) (CudaDrFrontend::GetOutputDevicePointer());
    return (CUresult) CudaDrFrontend::GetExitCode();
}

/*Pushes a floating context on the current CPU thread. */
extern CUresult cuCtxPushCurrent(CUcontext ctx) {
    CudaDrFrontend::Prepare();
    CudaDrFrontend::AddDevicePointerForArguments((void*) ctx);
    CudaDrFrontend::Execute("cuCtxPushCurrent");
    return (CUresult) CudaDrFrontend::GetExitCode();
}

/*Block for a context's tasks to complete.*/
extern CUresult cuCtxSynchronize(void) {
    CudaDrFrontend::Prepare();
    CudaDrFrontend::Execute("cuCtxSynchronize");
    return (CUresult) CudaDrFrontend::GetExitCode();
}

/* Disable peer access */
extern CUresult cuCtxDisablePeerAccess(CUcontext peerContext) {
    CudaDrFrontend::Prepare();
    CudaDrFrontend::AddDevicePointerForArguments((void*) peerContext);
    CudaDrFrontend::Execute("cuCtxDisablePeerAccess");
    return (CUresult) (CudaDrFrontend::GetExitCode());
}

/* Enable peer access */
extern CUresult cuCtxEnablePeerAccess(CUcontext peerContext, unsigned int flags) {
    CudaDrFrontend::Prepare();
    CudaDrFrontend::AddDevicePointerForArguments((void*) peerContext);
    CudaDrFrontend::AddVariableForArguments(flags);
    CudaDrFrontend::Execute("cuCtxEnablePeerAccess");
    return (CUresult) (CudaDrFrontend::GetExitCode());
}

/* Check if two devices could be connected using peer to peer */
extern CUresult cuDeviceCanAccessPeer(int *canAccessPeer, CUdevice dev, CUdevice peerDev) {
    CudaDrFrontend::Prepare();
    CudaDrFrontend::AddHostPointerForArguments(canAccessPeer);
    CudaDrFrontend::AddVariableForArguments(dev);
    CudaDrFrontend::AddVariableForArguments(peerDev);
    CudaDrFrontend::Execute("cuDeviceCanAccessPeer");
    if (CudaDrFrontend::Success())
        *canAccessPeer = *(CudaDrFrontend::GetOutputHostPointer<int>());
    return (CUresult) (CudaDrFrontend::GetExitCode());
}

