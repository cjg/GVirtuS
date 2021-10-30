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

/**
 * @file   CudaUtil.cpp
 * @author Giuseppe Coviello <giuseppe.coviello@uniparthenope.it>
 * @date   Sun Oct 11 17:16:48 2009
 *
 * @brief
 *
 *
 */

#include "CudaUtil.h"

#include <cstdio>
#include <iostream>

#include <cuda.h>

#if CUDART_VERSION >= 11000
struct __align__(8) fatBinaryHeader
{
  unsigned int           magic;
  unsigned short         version;
  unsigned short         headerSize;
  unsigned long long int fatSize;
};
#endif

using namespace std;

CudaUtil::CudaUtil() {}

CudaUtil::CudaUtil(const CudaUtil& orig) {}

CudaUtil::~CudaUtil() {}

char* CudaUtil::MarshalHostPointer(const void* ptr) {
  char* marshal = new char[CudaUtil::MarshaledHostPointerSize];
  MarshalHostPointer(ptr, marshal);
  return marshal;
}

void CudaUtil::MarshalHostPointer(const void* ptr, char* marshal) {
#ifdef _WIN32
  sprintf_s(marshal, 10, "%p", ptr);
#else
  sprintf(marshal, "%p", ptr);
#endif
}

char* CudaUtil::MarshalDevicePointer(const void* devPtr) {
  char* marshal = new char[CudaUtil::MarshaledDevicePointerSize];
  MarshalDevicePointer(devPtr, marshal);
  return marshal;
}

void CudaUtil::MarshalDevicePointer(const void* devPtr, char* marshal) {
#ifdef _WIN32
  sprintf_s(marshal, 10, "%p", devPtr);
#else
  sprintf(marshal, "%p", devPtr);
#endif
}

Buffer* CudaUtil::MarshalFatCudaBinary(__cudaFatCudaBinary* bin,
                                       Buffer* marshal) {
  if (marshal == NULL) marshal = new Buffer();
  size_t size;
  int count;

  marshal->Add(bin->magic);
  marshal->Add(bin->version);
  marshal->Add(bin->gpuInfoVersion);

  size = strlen(bin->key) + 1;
  marshal->Add(size);
  marshal->Add(bin->key, size);

  size = strlen(bin->ident) + 1;
  marshal->Add(size);
  marshal->Add(bin->ident, size);

  size = strlen(bin->usageMode) + 1;
  marshal->Add(size);
  marshal->Add(bin->usageMode, size);

  for (count = 0; bin->ptx[count].gpuProfileName != NULL; count++)
    ;
  marshal->Add(count);
  for (int i = 0; i < count; i++) {
    size = strlen(bin->ptx[i].gpuProfileName) + 1;
    marshal->Add(size);
    marshal->Add(bin->ptx[i].gpuProfileName, size);

    size = strlen(bin->ptx[i].ptx) + 1;
    marshal->Add(size);
    marshal->Add(bin->ptx[i].ptx, size);
  }

  for (count = 0; bin->cubin[count].gpuProfileName != NULL; count++)
    ;
  marshal->Add(count);
  for (int i = 0; i < count; i++) {
    size = strlen(bin->cubin[i].gpuProfileName) + 1;
    marshal->Add(size);
    marshal->Add(bin->cubin[i].gpuProfileName, size);

    size = strlen(bin->cubin[i].cubin) + 1;
    marshal->Add(size);
    marshal->Add(bin->cubin[i].cubin, size);
  }

  /* Achtung: no debug is possible */
  marshal->Add(0);

#if 0
    for (count = 0; bin->exported != NULL && bin->exported[count].name != NULL; count++);
#else
  count = 0;
#endif
  marshal->Add(count);
  for (int i = 0; i < count; i++) {
    size = strlen(bin->exported[i].name) + 1;
    marshal->Add(size);
    marshal->Add(bin->exported[i].name, size);
  }

#if 0
    for (count = 0; bin->imported != NULL && bin->imported[count].name != NULL; count++);
#else
  count = 0;
#endif
  marshal->Add(count);
  for (int i = 0; i < count; i++) {
    size = strlen(bin->imported[i].name) + 1;
    marshal->Add(size);
    marshal->Add(bin->imported[i].name, size);
  }

  marshal->Add(bin->flags);

  /* Achtung: no dependends added */
  marshal->Add(0);

#ifndef CUDA_VERSION
#error CUDA_VERSION not defined
#endif
#if CUDA_VERSION >= 2030
  marshal->Add(bin->characteristic);
#endif

  return marshal;
}

Buffer* CudaUtil::MarshalFatCudaBinary(__fatBinC_Wrapper_t* bin,
                                       Buffer* marshal) {
  if (marshal == NULL) marshal = new Buffer();
  // size_t size = (unsigned long long*)&(bin->magic) - bin->data;

  marshal->Add(bin->magic);
  marshal->Add(bin->version);

  struct fatBinaryHeader* header = (fatBinaryHeader*)bin->data;
  //    size_t size = (header->fatSize / sizeof(unsigned long long)) + 2;
  //    size_t size = header->fatSize;
  size_t size = header->fatSize + (unsigned long long)header->headerSize;

  marshal->Add(size);
  //    marshal->Add((bin->data), size);
  marshal->Add((char*)(bin->data), size);

  return marshal;
}

__fatBinC_Wrapper_t* CudaUtil::UnmarshalFatCudaBinaryV2(Buffer* marshal) {
  __fatBinC_Wrapper_t* bin =
      new __fatBinC_Wrapper_t __attribute__((aligned(8)));
  size_t size;

  bin->magic = marshal->Get<int>();
  bin->version = marshal->Get<int>();
  size = marshal->Get<size_t>();
  //    bin->data = marshal->Get<unsigned long long int>(size);
  bin->data = (const long long unsigned int*)marshal->Get<char>(size);
  // bin->data= NULL;
  /*
  cerr << "**********DATA**********" << endl;
  fprintf(stderr, "data pointer: %p\n", bin->data);
  if (bin->data == NULL)
      throw "Error allocating";

  char* data = (char*)bin->data;
  for (int i = 0; i < (size * sizeof(long long int)); i++) {
      fprintf(stderr, "%x ", *(data + i));
  }
  cerr << endl << "********** END DATA**********" << endl;
   */
  bin->filename_or_fatbins = NULL;
  return bin;
}

__cudaFatCudaBinary* CudaUtil::UnmarshalFatCudaBinary(Buffer* marshal) {
  __cudaFatCudaBinary* bin = new __cudaFatCudaBinary;
  size_t size;
  int i, count;

  bin->magic = marshal->Get<unsigned long>();
  bin->version = marshal->Get<unsigned long>();
  bin->gpuInfoVersion = marshal->Get<unsigned long>();

  size = marshal->Get<size_t>();
  bin->key = marshal->Get<char>(size);

  size = marshal->Get<size_t>();
  bin->ident = marshal->Get<char>(size);

  size = marshal->Get<size_t>();
  bin->usageMode = marshal->Get<char>(size);

  count = marshal->Get<int>();
  bin->ptx = new __cudaFatPtxEntry[count + 1];
  for (i = 0; i < count; i++) {
    size = marshal->Get<size_t>();
    bin->ptx[i].gpuProfileName = marshal->Get<char>(size);

    size = marshal->Get<size_t>();
    bin->ptx[i].ptx = marshal->Get<char>(size);
  }
  bin->ptx[i].gpuProfileName = NULL;
  bin->ptx[i].ptx = NULL;

  count = marshal->Get<int>();
  bin->cubin = new __cudaFatCubinEntry[count + 1];
  for (i = 0; i < count; i++) {
    size = marshal->Get<size_t>();
    bin->cubin[i].gpuProfileName = marshal->Get<char>(size);

    size = marshal->Get<size_t>();
    bin->cubin[i].cubin = marshal->Get<char>(size);
  }
  bin->cubin[i].gpuProfileName = NULL;
  bin->cubin[i].cubin = NULL;

  /* Achtung: no debug is possible */
  marshal->Get<int>();
  bin->debug = new __cudaFatDebugEntry;
  bin->debug->gpuProfileName = NULL;
  bin->debug->debug = NULL;

  bin->debugInfo = NULL;

  count = marshal->Get<int>();
  if (count == 0)
    bin->exported = NULL;
  else {
    bin->exported = new __cudaFatSymbol[count + 1];
    for (i = 0; i < count; i++) {
      size = marshal->Get<size_t>();
      bin->exported[i].name = marshal->Get<char>(size);
    }
    bin->exported[i].name = NULL;
  }

  count = marshal->Get<int>();
  if (count == 0)
    bin->imported = NULL;
  else {
    bin->imported = new __cudaFatSymbol[count + 1];
    for (i = 0; i < count; i++) {
      size = marshal->Get<size_t>();
      bin->imported[i].name = marshal->Get<char>(size);
    }
    bin->imported[i].name = NULL;
  }

  bin->flags = marshal->Get<unsigned int>();

  marshal->Get<int>();
  bin->dependends = NULL;

#ifndef CUDA_VERSION
#error CUDA_VERSION not defined
#endif
#if CUDA_VERSION >= 2030
  bin->characteristic = marshal->Get<unsigned int>();
#endif
#if CUDA_VERSION >= 3010
  bin->elf = NULL;
#endif

  return bin;
}

void CudaUtil::DumpFatCudaBinary(__cudaFatCudaBinary* bin, ostream& out) {
  out << endl << "FatCudaBinary" << endl << "----------" << endl;
  out << "magic: " << bin->magic << endl;
  out << "version: " << bin->version << endl;
  out << "gpuInfoVersion: " << bin->gpuInfoVersion << endl;
  out << "key: " << bin->key << endl;
  out << "ident: " << bin->ident << endl;
  out << "usageMode: " << bin->usageMode << endl;
  out << "ptx:" << endl;
  int i;
  for (i = 0; bin->ptx[i].gpuProfileName != NULL; i++) {
    out << '\t' << "gpuProfileName[" << i << "]: " << bin->ptx[i].gpuProfileName
        << endl;
    out << '\t' << "ptx[" << i << "]: " << bin->ptx[i].ptx << endl;
  }
  out << "***" << i << endl;
  out << "cubin:" << endl;
  for (int i = 0; bin->cubin[i].gpuProfileName != NULL; i++) {
    out << '\t' << "gpuProfileName[" << i
        << "]: " << bin->cubin[i].gpuProfileName << endl;
    out << '\t' << "cubin[" << i << "]: " << bin->cubin[i].cubin << endl;
  }
#if 0
    out << "debug:" << endl;
    for (int i = 0; bin->debug[i].gpuProfileName != NULL; i++) {
        out << '\t' << "gpuProfileName[" << i << "]: " << bin->debug[i].gpuProfileName << endl;
        out << '\t' << "debug[" << i << "]: " << bin->debug[i].debug << endl;
    }
    out << "debugInfo: " << bin->debugInfo << endl;
#endif
  out << "exported:" << endl;
  for (int i = 0; bin->exported != NULL && bin->exported[i].name != NULL; i++) {
    out << '\t' << "name[" << i << "]: " << bin->exported[i].name << endl;
  }
  out << "imported:" << endl;
  for (int i = 0; bin->imported != NULL && bin->imported[i].name != NULL; i++) {
    out << '\t' << "name[" << i << "]: " << bin->imported[i].name << endl;
  }
  out << "flags: " << bin->flags << endl;
#if 0
    out << "dependends:" << endl;
    for (int i = 0; bin->dependends != NULL && bin->dependends[i].key != NULL; i++) {
        CudaUtil::DumpFatCudaBinary(bin->dependends + i);
    }
#endif
#ifndef CUDA_VERSION
#error CUDA_VERSION not defined
#endif
#if CUDA_VERSION == 2030
  out << "characteristic: " << bin->characteristic << endl;
#endif
  out << "----------" << endl << endl;
}

Buffer* CudaUtil::MarshalTextureDescForArguments(const cudaTextureDesc* tex,
                                                 Buffer* marshal = NULL) {
  if (marshal == NULL) marshal = new Buffer();
  if (tex->addressMode != NULL) {
    marshal->AddConst<int>(1);
    marshal->AddConst<cudaTextureAddressMode>(tex->addressMode[0]);
    marshal->AddConst<cudaTextureAddressMode>(tex->addressMode[1]);
    marshal->AddConst<cudaTextureAddressMode>(tex->addressMode[2]);
    marshal->AddConst<cudaTextureFilterMode>(tex->filterMode);
    marshal->AddConst<cudaTextureReadMode>(tex->readMode);
    marshal->AddConst<int>(tex->sRGB);
    marshal->AddConst<int>(tex->normalizedCoords);
    marshal->AddConst<unsigned int>(tex->maxAnisotropy);
    marshal->AddConst<cudaTextureFilterMode>(tex->mipmapFilterMode);
    marshal->AddConst<float>(tex->mipmapLevelBias);
    marshal->AddConst<float>(tex->minMipmapLevelClamp);
    marshal->AddConst<float>(tex->maxMipmapLevelClamp);
  } else {
    marshal->AddConst<int>(0);
    marshal->AddConst<cudaTextureDesc>(tex);
  }

  return marshal;
}

cudaTextureDesc* CudaUtil::UnmarshalTextureDesc(Buffer* marshal) {
  int check = marshal->Get<int>();
  cudaTextureDesc* tex = new cudaTextureDesc;
  if (check == 1) {
    tex->addressMode[0] = marshal->Get<cudaTextureAddressMode>();
    tex->addressMode[1] = marshal->Get<cudaTextureAddressMode>();
    tex->addressMode[2] = marshal->Get<cudaTextureAddressMode>();
    tex->filterMode = marshal->Get<cudaTextureFilterMode>();
    tex->readMode = marshal->Get<cudaTextureReadMode>();
    tex->sRGB = marshal->Get<int>();
    tex->normalizedCoords = marshal->Get<int>();
    tex->maxAnisotropy = marshal->Get<unsigned int>();
    tex->mipmapFilterMode = marshal->Get<cudaTextureFilterMode>();
    tex->mipmapLevelBias = marshal->Get<float>();
    tex->minMipmapLevelClamp = marshal->Get<float>();
    tex->maxMipmapLevelClamp = marshal->Get<float>();
    return tex;
  } else
    return marshal->Assign<cudaTextureDesc>();
}
