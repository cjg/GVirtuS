
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

#include <cstdio>
#include <elf.h>
#include <CudaRt_internal.h>

#include "CudaRt.h"

/*
 Routines not found in the cuda's header files.
 KEEP THEM WITH CARE
 */


extern "C" __host__ void **__cudaRegisterFatBinary(void *fatCubin) {



  /* Fake host pointer */
  __fatBinC_Wrapper_t *bin = (__fatBinC_Wrapper_t *)fatCubin;
  char *data = (char *)bin->data;

    NvFatCubin *pFatCubin = (NvFatCubin *)data;
    // check so its really an elf file
    Elf64_Ehdr *eh = &(pFatCubin->elf);
    if(!strncmp((char*)eh->e_ident, "\177ELF", 4)) {

        /* Section header table :  */
        Elf64_Shdr *sh_table = static_cast<Elf64_Shdr *>(malloc(eh->e_shentsize * eh->e_shnum));

        byte *baseAddr = (byte *) eh;
        for (uint32_t i = 0; i < eh->e_shnum; i++) {
            Elf64_Shdr *shdrSrc = (Elf64_Shdr *) (baseAddr + (off_t) eh->e_shoff + i * eh->e_shentsize);
            memcpy(&sh_table[i], shdrSrc, eh->e_shentsize);
        }

        char *sh_str = static_cast<char *>(malloc(sh_table[eh->e_shstrndx].sh_size));
        if (sh_str) {
            memcpy(sh_str, baseAddr + sh_table[eh->e_shstrndx].sh_offset, sh_table[eh->e_shstrndx].sh_size);

            for (uint32_t i = 0; i < eh->e_shnum; i++) {

                char *szSectionName = (sh_str + sh_table[i].sh_name);
                if (strncmp(".nv.info.", szSectionName, strlen(".nv.info.")) == 0) {
                    char *szFuncName = szSectionName + strlen(".nv.info.");
                    //printf("%s:\n", szFuncName);
                    byte *p = (byte *) eh + sh_table[i].sh_offset;

                    NvInfoFunction infoFunction;
                    size_t size;
                    NvInfoAttribute *pAttr = (NvInfoAttribute *) p;
                    while (pAttr < (NvInfoAttribute *) ((byte *) p + sh_table[i].sh_size)) {
                        size = 0;
                        switch (pAttr->fmt) {
                            case EIFMT_SVAL:
                                size = sizeof(NvInfoAttribute) + pAttr->value;
                                break;
                            case EIFMT_NVAL:
                                size = sizeof(NvInfoAttribute);
                                break;
                            case EIFMT_HVAL:
                                size = sizeof(NvInfoAttribute);
                                break;

                        }
                        if (pAttr->attr == EIATTR_KPARAM_INFO) {
                            NvInfoKParam *nvInfoKParam = (NvInfoKParam *) pAttr;

                            //printf("index:%d align:%x ordinal:%d offset:%d a:%x size:%d %d b:%x\n",  nvInfoKParam->index, nvInfoKParam->index, nvInfoKParam->ordinal,
                            //       nvInfoKParam->offset, nvInfoKParam->a, (nvInfoKParam->size & 0xf8) >> 2, nvInfoKParam->size & 0x07, nvInfoKParam->b);

                            NvInfoKParam nvInfoKParam1;
                            nvInfoKParam1.index = nvInfoKParam->index;
                            nvInfoKParam1.ordinal = nvInfoKParam->ordinal;
                            nvInfoKParam1.offset = nvInfoKParam->offset;
                            nvInfoKParam1.a = nvInfoKParam->a;
                            nvInfoKParam1.size = nvInfoKParam->size;
                            nvInfoKParam1.b = nvInfoKParam->b;
                            infoFunction.params.push_back(nvInfoKParam1);
                        }
                        pAttr = (NvInfoAttribute *) ((byte *) pAttr + size);
                    }
                    CudaRtFrontend::addDeviceFunc2InfoFunc(szFuncName, infoFunction);
                }
            }
            free(sh_str);
        }
        free(sh_table);


        Buffer *input_buffer = new Buffer();
        input_buffer->AddString(CudaUtil::MarshalHostPointer((void **) bin));
        input_buffer = CudaUtil::MarshalFatCudaBinary(bin, input_buffer);

        CudaRtFrontend::Prepare();
        CudaRtFrontend::Execute("cudaRegisterFatBinary", input_buffer);
        if (CudaRtFrontend::Success()) return (void **) fatCubin;
    }
  return NULL;
}

extern "C" __host__ void **__cudaRegisterFatBinaryEnd(void *fatCubin) {
  /* Fake host pointer */
  __fatBinC_Wrapper_t *bin = (__fatBinC_Wrapper_t *)fatCubin;
  char *data = (char *)bin->data;

  Buffer *input_buffer = new Buffer();
  input_buffer->AddString(CudaUtil::MarshalHostPointer((void **)bin));
  input_buffer = CudaUtil::MarshalFatCudaBinary(bin, input_buffer);

  CudaRtFrontend::Prepare();
  CudaRtFrontend::Execute("cudaRegisterFatBinaryEnd", input_buffer);
  if (CudaRtFrontend::Success()) return (void **)fatCubin;
  return NULL;
}

extern "C" __host__ void __cudaUnregisterFatBinary(void **fatCubinHandle) {
  CudaRtFrontend::Prepare();
  CudaRtFrontend::AddStringForArguments(
      CudaUtil::MarshalHostPointer(fatCubinHandle));
  CudaRtFrontend::Execute("cudaUnregisterFatBinary");
}

extern "C" __host__ void __cudaRegisterFunction(
    void **fatCubinHandle, const char *hostFun, char *deviceFun,
    const char *deviceName, int thread_limit, uint3 *tid, uint3 *bid,
    dim3 *bDim, dim3 *gDim, int *wSize) {

    //printf("__cudaRegisterFunction - hostFun:%x deviceFun:%s\n",hostFun,deviceFun);
  CudaRtFrontend::Prepare();
  CudaRtFrontend::AddStringForArguments(
      CudaUtil::MarshalHostPointer(fatCubinHandle));

  CudaRtFrontend::AddVariableForArguments((gvirtus::common::pointer_t)hostFun);
  CudaRtFrontend::AddStringForArguments(deviceFun);
  CudaRtFrontend::AddStringForArguments(deviceName);
  CudaRtFrontend::AddVariableForArguments(thread_limit);
  CudaRtFrontend::AddHostPointerForArguments(tid);
  CudaRtFrontend::AddHostPointerForArguments(bid);
  CudaRtFrontend::AddHostPointerForArguments(bDim);
  CudaRtFrontend::AddHostPointerForArguments(gDim);
  CudaRtFrontend::AddHostPointerForArguments(wSize);

  CudaRtFrontend::Execute("cudaRegisterFunction");

  deviceFun = CudaRtFrontend::GetOutputString();
  tid = CudaRtFrontend::GetOutputHostPointer<uint3>();
  bid = CudaRtFrontend::GetOutputHostPointer<uint3>();
  bDim = CudaRtFrontend::GetOutputHostPointer<dim3>();
  gDim = CudaRtFrontend::GetOutputHostPointer<dim3>();
  wSize = CudaRtFrontend::GetOutputHostPointer<int>();



  CudaRtFrontend::addHost2DeviceFunc((void*)hostFun,deviceFun);
}

extern "C" __host__ void __cudaRegisterVar(void **fatCubinHandle, char *hostVar,
                                           char *deviceAddress,
                                           const char *deviceName, int ext,
                                           int size, int constant, int global) {
  CudaRtFrontend::Prepare();
  CudaRtFrontend::AddStringForArguments(
      CudaUtil::MarshalHostPointer(fatCubinHandle));
  CudaRtFrontend::AddStringForArguments(CudaUtil::MarshalHostPointer(hostVar));
  CudaRtFrontend::AddStringForArguments(deviceAddress);
  CudaRtFrontend::AddStringForArguments(deviceName);
  CudaRtFrontend::AddVariableForArguments(ext);
  CudaRtFrontend::AddVariableForArguments(size);
  CudaRtFrontend::AddVariableForArguments(constant);
  CudaRtFrontend::AddVariableForArguments(global);
  CudaRtFrontend::Execute("cudaRegisterVar");
}

extern "C" __host__ void __cudaRegisterShared(void **fatCubinHandle,
                                              void **devicePtr) {
  CudaRtFrontend::Prepare();
  CudaRtFrontend::AddStringForArguments(
      CudaUtil::MarshalHostPointer(fatCubinHandle));
  CudaRtFrontend::AddStringForArguments((char *)devicePtr);
  CudaRtFrontend::Execute("cudaRegisterShared");
}

extern "C" __host__ void __cudaRegisterSharedVar(void **fatCubinHandle,
                                                 void **devicePtr, size_t size,
                                                 size_t alignment,
                                                 int storage) {
  CudaRtFrontend::Prepare();
  CudaRtFrontend::AddStringForArguments(
      CudaUtil::MarshalHostPointer(fatCubinHandle));
  CudaRtFrontend::AddStringForArguments((char *)devicePtr);
  CudaRtFrontend::AddVariableForArguments(size);
  CudaRtFrontend::AddVariableForArguments(alignment);
  CudaRtFrontend::AddVariableForArguments(storage);
  CudaRtFrontend::Execute("cudaRegisterSharedVar");
}

extern "C" __host__ void __cudaRegisterTexture(void **fatCubinHandle,
                                               const textureReference *hostVar,
                                               void **deviceAddress,
                                               char *deviceName, int dim,
                                               int norm, int ext) {
  CudaRtFrontend::Prepare();
  CudaRtFrontend::AddStringForArguments(
      CudaUtil::MarshalHostPointer(fatCubinHandle));
  CudaRtFrontend::AddStringForArguments(CudaUtil::MarshalHostPointer(hostVar));
  // Achtung: passing the address and the content of the textureReference
  CudaRtFrontend::AddHostPointerForArguments(hostVar);
  CudaRtFrontend::AddStringForArguments((char *)deviceAddress);
  CudaRtFrontend::AddStringForArguments(deviceName);
  CudaRtFrontend::AddVariableForArguments(dim);
  CudaRtFrontend::AddVariableForArguments(norm);
  CudaRtFrontend::AddVariableForArguments(ext);
  CudaRtFrontend::Execute("cudaRegisterTexture");
}

extern "C" __host__ void __cudaRegisterSurface(void **fatCubinHandle,
                                               const surfaceReference *hostVar,
                                               void **deviceAddress,
                                               char *deviceName, int dim,
                                               int ext) {
  CudaRtFrontend::Prepare();
  CudaRtFrontend::AddStringForArguments(
      CudaUtil::MarshalHostPointer(fatCubinHandle));
  CudaRtFrontend::AddStringForArguments(CudaUtil::MarshalHostPointer(hostVar));
  // Achtung: passing the address and the content of the textureReference
  CudaRtFrontend::AddHostPointerForArguments(hostVar);
  CudaRtFrontend::AddStringForArguments((char *)deviceAddress);
  CudaRtFrontend::AddStringForArguments(deviceName);
  CudaRtFrontend::AddVariableForArguments(dim);
  CudaRtFrontend::AddVariableForArguments(ext);
  CudaRtFrontend::Execute("cudaRegisterSurface");
}

/* */

extern "C" __host__ int __cudaSynchronizeThreads(void **x, void *y) {
  // FIXME: implement
  std::cerr << "*** Error: __cudaSynchronizeThreads() not yet implemented!"
            << std::endl;
  return 0;
}

extern "C" __host__ void __cudaTextureFetch(const void *tex, void *index,
                                            int integer, void *val) {
  // FIXME: implement
  std::cerr << "*** Error: __cudaTextureFetch() not yet implemented!"
            << std::endl;
}

#if CUDA_VERSION >= 9000
extern "C" __host__ __device__  unsigned CUDARTAPI __cudaPushCallConfiguration(dim3 gridDim, dim3 blockDim, size_t sharedMem = 0, void *stream = 0) {
    CudaRtFrontend::Prepare();
    CudaRtFrontend::AddVariableForArguments(gridDim);
    CudaRtFrontend::AddVariableForArguments(blockDim);
    CudaRtFrontend::AddVariableForArguments(sharedMem);

#if CUDART_VERSION >= 3010
    CudaRtFrontend::AddDevicePointerForArguments(stream);
#else
    CudaRtFrontend::AddVariableForArguments(stream);
#endif
    /*
    printf("cudaPushCallConfiguration:\n");
    printf("gridDim: %d,%d,%d\n",gridDim.x,gridDim.y,gridDim.z);
    printf("blockDim: %d,%d,%d\n",blockDim.x,blockDim.y,blockDim.z);
    printf("sharedMem: %d stream: %x\n",sharedMem,stream);
     */


    CudaRtFrontend::Execute("cudaPushCallConfiguration");
    cudaError_t cudaError=CudaRtFrontend::GetExitCode();
    //printf("cudaPushCallConfiguration:%d\n",cudaError);
    return cudaError;
}


extern "C" cudaError_t CUDARTAPI __cudaPopCallConfiguration( dim3 *gridDim,
                                                             dim3 *blockDim,
                                                             size_t *sharedMem,
                                                             void *stream) {
    /*
    printf("__cudaPopCallConfiguration:\n");
    printf("gridDim: %d,%d,%d\n",gridDim->x,gridDim->y,gridDim->z);
    printf("blockDim: %d,%d,%d\n",blockDim->x,blockDim->y,blockDim->z);
    printf("sharedMem: %ld stream: %x\n",*sharedMem, *(cudaStream_t*)stream);
    */
    CudaRtFrontend::Prepare();

    CudaRtFrontend::Execute("cudaPopCallConfiguration");
    cudaError_t cudaError=CudaRtFrontend::GetExitCode();
    //printf("__cudaPopCallConfiguration:%d\n",cudaError);

    *gridDim = CudaRtFrontend::GetOutputVariable<dim3>();
    *blockDim = CudaRtFrontend::GetOutputVariable<dim3>();
    *sharedMem = CudaRtFrontend::GetOutputVariable<size_t>();
    cudaStream_t stream1=CudaRtFrontend::GetOutputVariable<cudaStream_t>();
    //cudaStream_t stream1=0;
    memcpy(stream,&stream1,sizeof(cudaStream_t));

    /*
    printf("gridDim: %d,%d,%d\n",gridDim->x,gridDim->y,gridDim->z);
    printf("blockDim: %d,%d,%d\n",blockDim->x,blockDim->y,blockDim->z);
    printf("sharedMem: %ld stream: %x stream1: %x\n",*sharedMem,*(cudaStream_t*)stream,stream1);
    */
    return cudaError;
}
#endif
