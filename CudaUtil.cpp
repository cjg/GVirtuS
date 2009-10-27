/* 
 * File:   CudaUtil.cpp
 * Author: cjg
 * 
 * Created on October 11, 2009, 5:16 PM
 */

#include <iostream>
#include "CudaUtil.h"

using namespace std;

CudaUtil::CudaUtil() {
}

CudaUtil::CudaUtil(const CudaUtil& orig) {
}

CudaUtil::~CudaUtil() {
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

Buffer * CudaUtil::MarshalFatCudaBinary(__cudaFatCudaBinary* bin) {
    Buffer * marshal = new Buffer();
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

    cout << "cubin:" << endl;
    for(int i = 0; bin->cubin[i].gpuProfileName != NULL; i++) {
        cout << '\t' << "gpuProfileName[" << i << "]: " << bin->cubin[i].gpuProfileName << endl;
        cout << '\t' << "cubin[" << i << "]: " << bin->cubin[i].cubin << endl;
    }
    cout << "debug:" << endl;
    for(int i = 0; bin->debug[i].gpuProfileName != NULL; i++) {
        cout << '\t' << "gpuProfileName[" << i << "]: " << bin->debug[i].gpuProfileName << endl;
        cout << '\t' << "debug[" << i << "]: " << bin->debug[i].debug << endl;
    }
    cout << "debugInfo: " << bin->debugInfo << endl;
    cout << "exported:" << endl;
    for(int i = 0; bin->exported != NULL && bin->exported[i].name != NULL; i++) {
        cout << '\t' << "name[" << i << "]: " << bin->exported[i].name << endl;
    }
    cout << "imported:" << endl;
    for(int i = 0; bin->imported != NULL && bin->imported[i].name != NULL; i++) {
        cout << '\t' << "name[" << i << "]: " << bin->imported[i].name << endl;
    }
    cout << "flags: " << bin->flags << endl;
    cout << "dependends:" << endl;
    for(int i = 0; bin->dependends != NULL && bin->dependends[i].key != NULL; i++) {
        CudaUtil::DumpFatCudaBinary(bin->dependends + i);
    }
    cout << "characteristic: " << bin->characteristic << endl;
    cout << "----------" << endl << endl;
}

void CudaUtil::DumpFatCudaBinary(__cudaFatCudaBinary* bin) {
    cout << endl << "FatCudaBinary" << endl << "----------" << endl;
    cout << "magic: " << bin->magic << endl;
    cout << "version: " << bin->version << endl;
    cout << "gpuInfoVersion: " << bin->gpuInfoVersion << endl;
    cout << "key: " << bin->key << endl;
    cout << "ident: " << bin->ident << endl;
    cout << "usageMode: " << bin->usageMode << endl;
    cout << "ptx:" << endl;
    int i;
    for(i = 0; bin->ptx[i].gpuProfileName != NULL; i++) {
        cout << '\t' << "gpuProfileName[" << i << "]: " << bin->ptx[i].gpuProfileName << endl;
        cout << '\t' << "ptx[" << i << "]: " << bin->ptx[i].ptx << endl;
    }
    cout << "***" << i << endl;
    cout << "cubin:" << endl;
    for(int i = 0; bin->cubin[i].gpuProfileName != NULL; i++) {
        cout << '\t' << "gpuProfileName[" << i << "]: " << bin->cubin[i].gpuProfileName << endl;
        cout << '\t' << "cubin[" << i << "]: " << bin->cubin[i].cubin << endl;
    }
    cout << "debug:" << endl;
    for(int i = 0; bin->debug[i].gpuProfileName != NULL; i++) {
        cout << '\t' << "gpuProfileName[" << i << "]: " << bin->debug[i].gpuProfileName << endl;
        cout << '\t' << "debug[" << i << "]: " << bin->debug[i].debug << endl;
    }
    cout << "debugInfo: " << bin->debugInfo << endl;
    cout << "exported:" << endl;
    for(int i = 0; bin->exported != NULL && bin->exported[i].name != NULL; i++) {
        cout << '\t' << "name[" << i << "]: " << bin->exported[i].name << endl;
    }
    cout << "imported:" << endl;
    for(int i = 0; bin->imported != NULL && bin->imported[i].name != NULL; i++) {
        cout << '\t' << "name[" << i << "]: " << bin->imported[i].name << endl;
    }
    cout << "flags: " << bin->flags << endl;
    cout << "dependends:" << endl;
    for(int i = 0; bin->dependends != NULL && bin->dependends[i].key != NULL; i++) {
        CudaUtil::DumpFatCudaBinary(bin->dependends + i);
    }
    cout << "characteristic: " << bin->characteristic << endl;
    cout << "----------" << endl << endl;
}