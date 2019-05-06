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



#ifndef _CUDADRHANDLER_H
#define	_CUDADRHANDLER_H

#include <unistd.h>
#include <iostream>
#include <map>
#include <string>
#include <cstdio>
#include <host_defines.h>
#include <builtin_types.h>
#include <driver_types.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include "communicator/Result.h"
#include <cuda.h>
#include "Handler.h"

#include "log4cplus/logger.h"
#include "log4cplus/loggingmacros.h"
#include "log4cplus/configurator.h"

class CudaDrHandler : public Handler{
public:
    CudaDrHandler();
    virtual ~CudaDrHandler();
    bool CanExecute(std::string routine);
    Result * Execute(std::string routine, Buffer * input_buffer);

    void RegisterFatBinary(std::string & handler, void **fatCubinHandle);
    void RegisterFatBinary(const char * handler, void **fatCubinHandle);
    void **GetFatBinary(std::string & handler);
    void **GetFatBinary(const char * handler);
    void UnregisterFatBinary(std::string & handler);
    void UnregisterFatBinary(const char * handler);

    void RegisterDeviceFunction(std::string & handler, std::string & function);
    void RegisterDeviceFunction(const char * handler, const char * function);
    const char *GetDeviceFunction(std::string & handler);
    const char *GetDeviceFunction(const char * handler);

    void RegisterVar(std::string & handler, std::string &deviceName);
    void RegisterVar(const char *handler, const char *deviceName);
    const char *GetVar(std::string & handler);
    const char *GetVar(const char *handler);

    void RegisterTexture(std::string & handler, textureReference *texref);
    void RegisterTexture(const char *handler, textureReference *texref);
    textureReference *GetTexture(std::string & handler);
    textureReference *GetTexture(const char *handler);
    const char *GetTextureHandler(textureReference *texref);

    const char *GetSymbol(Buffer * in);

    void RegisterSharedMemory(const char *name) {
        mShmFd = shm_open(name, O_RDWR, S_IRWXU);

	if((mpShm = mmap(NULL, 256 * 1024 * 1024, PROT_READ | PROT_WRITE, MAP_SHARED, mShmFd,
            0)) == MAP_FAILED) {
		std::cout << "Failed to mmap" << std::endl;
                mpShm = NULL;
        }
    }

    void RequestSharedMemory(char *name, size_t *size) {
        sprintf(name, "/gvirtus-%d", getpid());
        *size = 128 * 1024 * 1024;
        std::cout << "SHM name " << name << std::endl;

        mShmFd = shm_open(name, O_RDWR | O_CREAT, 00666);

        if(ftruncate(mShmFd, *size) != 0) {
            std::cout << "Failed to truncate" << std::endl;
            mpShm = NULL;
            return;
        }

	if((mpShm = mmap(NULL, *size, PROT_READ | PROT_WRITE, MAP_SHARED, mShmFd,
            0)) == MAP_FAILED) {
		std::cout << "Failed to mmap" << std::endl;
                mpShm = NULL;
        }
    }

    void *GetSharedMemory() {
        return mpShm;
    }

    bool HasSharedMemory() {
        return mpShm != NULL;
    }
    

private:
    log4cplus::Logger logger;
    void Initialize();
    typedef Result * (*CudaDriverHandler)(CudaDrHandler *, Buffer *);
    static std::map<std::string, CudaDriverHandler> * mspHandlers;
    std::map<std::string, void **> * mpFatBinary;
    std::map<std::string, std::string> * mpDeviceFunction;
    std::map<std::string, std::string> * mpVar;
    std::map<std::string, textureReference *> * mpTexture;
    void *mpShm;
    int mShmFd;
};


#define CUDA_DRIVER_HANDLER(name) Result * handle##name(CudaDrHandler * pThis, Buffer * input_buffer)
#define CUDA_DRIVER_HANDLER_PAIR(name) make_pair("cu" #name, handle##name)

/*CudaDrHandler_initialization*/
CUDA_DRIVER_HANDLER(Init);

/*CudaDrHandler_context*/
CUDA_DRIVER_HANDLER(CtxCreate);
CUDA_DRIVER_HANDLER(CtxAttach);
CUDA_DRIVER_HANDLER(CtxDestroy);
CUDA_DRIVER_HANDLER(CtxDetach);
CUDA_DRIVER_HANDLER(CtxGetDevice);
CUDA_DRIVER_HANDLER(CtxPopCurrent);
CUDA_DRIVER_HANDLER(CtxPushCurrent);
CUDA_DRIVER_HANDLER(CtxSynchronize);

CUDA_DRIVER_HANDLER(CtxDisablePeerAccess);
CUDA_DRIVER_HANDLER(CtxEnablePeerAccess);
CUDA_DRIVER_HANDLER(DeviceCanAccessPeer);

/*CudaDrHandler_device*/
CUDA_DRIVER_HANDLER(DeviceComputeCapability);
CUDA_DRIVER_HANDLER(DeviceGet);
CUDA_DRIVER_HANDLER(DeviceGetAttribute);
CUDA_DRIVER_HANDLER(DeviceGetCount);
CUDA_DRIVER_HANDLER(DeviceGetName);
CUDA_DRIVER_HANDLER(DeviceGetProperties);
CUDA_DRIVER_HANDLER(DeviceTotalMem);

/*CudaDrHandler_execution*/
CUDA_DRIVER_HANDLER(ParamSetSize);
CUDA_DRIVER_HANDLER(FuncSetBlockShape);
CUDA_DRIVER_HANDLER(LaunchGrid);
CUDA_DRIVER_HANDLER(FuncGetAttribute);
CUDA_DRIVER_HANDLER(FuncSetSharedSize);
CUDA_DRIVER_HANDLER(Launch);
CUDA_DRIVER_HANDLER(ParamSetf);
CUDA_DRIVER_HANDLER(ParamSeti);
CUDA_DRIVER_HANDLER(ParamSetv);
CUDA_DRIVER_HANDLER(ParamSetTexRef);
CUDA_DRIVER_HANDLER(LaunchGridAsync);
CUDA_DRIVER_HANDLER(FuncSetCacheConfig);

/*CudaDrHandler_memory*/
CUDA_DRIVER_HANDLER(MemFree);
CUDA_DRIVER_HANDLER(MemAlloc);
CUDA_DRIVER_HANDLER(MemcpyDtoH);
CUDA_DRIVER_HANDLER(MemcpyHtoD);
CUDA_DRIVER_HANDLER(ArrayCreate);
CUDA_DRIVER_HANDLER(Memcpy2D);
CUDA_DRIVER_HANDLER(ArrayDestroy);
CUDA_DRIVER_HANDLER(Array3DCreate);
CUDA_DRIVER_HANDLER(MemAllocPitch);
CUDA_DRIVER_HANDLER(MemGetAddressRange);
CUDA_DRIVER_HANDLER(MemGetInfo);


/*CudaDrHandler_module*/
CUDA_DRIVER_HANDLER(ModuleLoadData);
CUDA_DRIVER_HANDLER(ModuleLoad);
CUDA_DRIVER_HANDLER(ModuleLoadFatBinary);
CUDA_DRIVER_HANDLER(ModuleUnload);
CUDA_DRIVER_HANDLER(ModuleGetFunction);
CUDA_DRIVER_HANDLER(ModuleGetGlobal);
CUDA_DRIVER_HANDLER(ModuleLoadDataEx);
CUDA_DRIVER_HANDLER(ModuleGetTexRef);

/*CudaDrHandler_version*/
CUDA_DRIVER_HANDLER(DriverGetVersion);

/*CudaDrHandler_stream*/
CUDA_DRIVER_HANDLER(StreamCreate);
CUDA_DRIVER_HANDLER(StreamDestroy);
CUDA_DRIVER_HANDLER(StreamQuery);
CUDA_DRIVER_HANDLER(StreamSynchronize);

/*CudaDrHandler_event*/
CUDA_DRIVER_HANDLER(EventCreate);
CUDA_DRIVER_HANDLER(EventDestroy);
CUDA_DRIVER_HANDLER(EventElapsedTime);
CUDA_DRIVER_HANDLER(EventQuery);
CUDA_DRIVER_HANDLER(EventRecord);
CUDA_DRIVER_HANDLER(EventSynchronize);

/*CudaDrHandler_texture*/
CUDA_DRIVER_HANDLER(TexRefSetArray);
CUDA_DRIVER_HANDLER(TexRefSetAddressMode);
CUDA_DRIVER_HANDLER(TexRefSetFilterMode);
CUDA_DRIVER_HANDLER(TexRefSetFlags);
CUDA_DRIVER_HANDLER(TexRefSetFormat);
CUDA_DRIVER_HANDLER(TexRefGetAddress);
CUDA_DRIVER_HANDLER(TexRefGetArray);
CUDA_DRIVER_HANDLER(TexRefGetFlags);
CUDA_DRIVER_HANDLER(TexRefSetAddress);

/*New Cuda 6.5 functions*/
CUDA_DRIVER_HANDLER(LaunchKernel);

#endif	/* _CUDADRHANDLER_H */

