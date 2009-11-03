/* 
 * File:   CudaUtil.cpp
 * Author: cjg
 * 
 * Created on October 11, 2009, 5:16 PM
 */

#include <cstdio>
#include <iostream>
#include "CudaUtil.h"

using namespace std;

CudaUtil::CudaUtil() {
}

CudaUtil::CudaUtil(const CudaUtil& orig) {
}

CudaUtil::~CudaUtil() {
}

char * CudaUtil::MarshalHostPointer(const void* ptr) {
    char *marshal = new char[CudaUtil::MarshaledHostPointerSize];
    MarshalHostPointer(ptr, marshal);
    return marshal;
}

void CudaUtil::MarshalHostPointer(const void * ptr, char * marshal) {
    sprintf(marshal, "0x%x", ptr);
    //size_t len = strlen(marshal);
    //memmove(marshal + 2 + sizeof(void *) * 2 - len, marshal, len + 1);
    //marshal[0] = '0';
    //marshal[1] = 'x';
    //memset(marshal + 2, '0', sizeof(void *) * 2 - len);
}

char * CudaUtil::MarshalDevicePointer(const void* devPtr) {
    char *marshal = new char[CudaUtil::MarshaledDevicePointerSize];
    MarshalDevicePointer(devPtr, marshal);
    return marshal;
}

void CudaUtil::MarshalDevicePointer(const void* devPtr, char * marshal) {
    sprintf(marshal, "%x", devPtr);
    size_t len = strlen(marshal);
    memmove(marshal + 2 + sizeof(void *) * 2 - len, marshal, len + 1);
    marshal[0] = '0';
    marshal[1] = 'x';
    memset(marshal + 2, '0', sizeof(void *) * 2 - len);
}

#if 0
/*
 * Fat binary container.
 * A mix of ptx intermediate programs and cubins,
 * plus a global identifier that can be used for
 * further lookup in a translation cache or a resource
 * file. This key is a checksum over the device text.
 * The ptx and cubin array are each terminated with
 * entries that have NULL components.
 */

typedef struct __cudaFatCudaBinaryRec {
    unsigned long            magic;
    unsigned long            version;
    unsigned long            gpuInfoVersion;
    char*                   key;
    char*                   ident;
    char*                   usageMode;
    __cudaFatPtxEntry             *ptx;
    __cudaFatCubinEntry           *cubin;
    __cudaFatDebugEntry           *debug;
    void*                  debugInfo;
    unsigned int                   flags;
    __cudaFatSymbol               *exported;
    __cudaFatSymbol               *imported;
    struct __cudaFatCudaBinaryRec *dependends;
    unsigned int                   characteristic;
} __cudaFatCudaBinary
#endif

Buffer * CudaUtil::MarshalFatCudaBinary(__cudaFatCudaBinary* bin, Buffer * marshal) {
    if(marshal == NULL)
        marshal = new Buffer();
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


    for(count = 0; bin->ptx[count].gpuProfileName != NULL; count++);
    marshal->Add(count);
    for(int i = 0; i < count; i++) {
        size = strlen(bin->ptx[i].gpuProfileName) + 1;
        marshal->Add(size);
        marshal->Add(bin->ptx[i].gpuProfileName, size);

        size = strlen(bin->ptx[i].ptx) + 1;
        marshal->Add(size);
        marshal->Add(bin->ptx[i].ptx, size);
    }

    for(count = 0; bin->cubin[count].gpuProfileName != NULL; count++);
    marshal->Add(count);
    for(int i = 0; i < count; i++) {
        size = strlen(bin->cubin[i].gpuProfileName) + 1;
        marshal->Add(size);
        marshal->Add(bin->cubin[i].gpuProfileName, size);

        size = strlen(bin->cubin[i].cubin) + 1;
        marshal->Add(size);
        marshal->Add(bin->cubin[i].cubin, size);
    }

    /* Achtung: no debug is possible */
    marshal->Add(0);

    for(count = 0; bin->exported != NULL && bin->exported[count].name != NULL; count++);
    marshal->Add(count);
    for(int i = 0; i < count; i++) {
        size = strlen(bin->exported[i].name) + 1;
        marshal->Add(size);
        marshal->Add(bin->exported[i].name, size);
    }

    for(count = 0; bin->imported != NULL && bin->imported[count].name != NULL; count++);
    marshal->Add(count);
    for(int i = 0; i < count; i++) {
        size = strlen(bin->imported[i].name) + 1;
        marshal->Add(size);
        marshal->Add(bin->imported[i].name, size);
    }

    marshal->Add(bin->flags);

    /* Achtung: no dependends added */
    marshal->Add(0);

    marshal->Add(bin->characteristic);

    return marshal;
}

__cudaFatCudaBinary * CudaUtil::UnmarshalFatCudaBinary(Buffer* marshal) {
    __cudaFatCudaBinary * bin = new __cudaFatCudaBinary;
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
    for(i = 0; i < count; i++) {
        size = marshal->Get<size_t>();
        bin->ptx[i].gpuProfileName = marshal->Get<char>(size);

        size = marshal->Get<size_t>();
        bin->ptx[i].ptx = marshal->Get<char>(size);
    }
    bin->ptx[i].gpuProfileName = NULL;
    bin->ptx[i].ptx = NULL;

    count = marshal->Get<int>();
    bin->cubin = new __cudaFatCubinEntry[count + 1];
    for(i = 0; i < count; i++) {
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
    if(count == 0)
        bin->exported = NULL;
    else {
        bin->exported = new __cudaFatSymbol[count + 1];
        for(i = 0; i < count; i++) {
            size = marshal->Get<size_t>();
            bin->exported[i].name = marshal->Get<char>(size);
        }
        bin->exported[i].name = NULL;
    }

    count = marshal->Get<int>();
    if(count == 0)
        bin->imported = NULL;
    else {
        bin->imported = new __cudaFatSymbol[count + 1];
        for(i = 0; i < count; i++) {
            size = marshal->Get<size_t>();
            bin->imported[i].name = marshal->Get<char>(size);
        }
        bin->imported[i].name = NULL;
    }

    bin->flags = marshal->Get<unsigned int>();

    marshal->Get<int>();
    bin->dependends = NULL;

    bin->characteristic = marshal->Get<unsigned int>();
    
    return bin;
}

void CudaUtil::DumpFatCudaBinary(__cudaFatCudaBinary* bin, ostream & out) {
    out << endl << "FatCudaBinary" << endl << "----------" << endl;
    out << "magic: " << bin->magic << endl;
    out << "version: " << bin->version << endl;
    out << "gpuInfoVersion: " << bin->gpuInfoVersion << endl;
    out << "key: " << bin->key << endl;
    out << "ident: " << bin->ident << endl;
    out << "usageMode: " << bin->usageMode << endl;
    out << "ptx:" << endl;
    int i;
    for(i = 0; bin->ptx[i].gpuProfileName != NULL; i++) {
        out << '\t' << "gpuProfileName[" << i << "]: " << bin->ptx[i].gpuProfileName << endl;
        out << '\t' << "ptx[" << i << "]: " << bin->ptx[i].ptx << endl;
    }
    out << "***" << i << endl;
    out << "cubin:" << endl;
    for(int i = 0; bin->cubin[i].gpuProfileName != NULL; i++) {
        out << '\t' << "gpuProfileName[" << i << "]: " << bin->cubin[i].gpuProfileName << endl;
        out << '\t' << "cubin[" << i << "]: " << bin->cubin[i].cubin << endl;
    }
#if 0
    out << "debug:" << endl;
    for(int i = 0; bin->debug[i].gpuProfileName != NULL; i++) {
        out << '\t' << "gpuProfileName[" << i << "]: " << bin->debug[i].gpuProfileName << endl;
        out << '\t' << "debug[" << i << "]: " << bin->debug[i].debug << endl;
    }
    out << "debugInfo: " << bin->debugInfo << endl;
#endif
    out << "exported:" << endl;
    for(int i = 0; bin->exported != NULL && bin->exported[i].name != NULL; i++) {
        out << '\t' << "name[" << i << "]: " << bin->exported[i].name << endl;
    }
    out << "imported:" << endl;
    for(int i = 0; bin->imported != NULL && bin->imported[i].name != NULL; i++) {
        out << '\t' << "name[" << i << "]: " << bin->imported[i].name << endl;
    }
    out << "flags: " << bin->flags << endl;
#if 0
    out << "dependends:" << endl;
    for(int i = 0; bin->dependends != NULL && bin->dependends[i].key != NULL; i++) {
        CudaUtil::DumpFatCudaBinary(bin->dependends + i);
    }
#endif
    out << "characteristic: " << bin->characteristic << endl;
    out << "----------" << endl << endl;
}