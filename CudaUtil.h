/* 
 * File:   CudaUtil.h
 * Author: cjg
 *
 * Created on October 11, 2009, 5:16 PM
 */

#ifndef _CUDAUTIL_H
#define	_CUDAUTIL_H

#include <iostream>
#include <cstdlib>
#include "__cudaFatFormat.h"
#include "Buffer.h"

class CudaUtil {
public:
    CudaUtil();
    CudaUtil(const CudaUtil& orig);
    virtual ~CudaUtil();
    static const size_t MarshaledDevicePointerSize = sizeof(void *) * 2 + 3;
    static const size_t MarshaledHostPointerSize = sizeof(void *) * 2 + 3;
    static char * MarshalHostPointer(const void* ptr);
    static void MarshalHostPointer(const void * ptr, char * marshal);
    static char * MarshalDevicePointer(const void *devPtr);
    static void MarshalDevicePointer(const void *devPtr, char * marshal);
    static inline void * UnmarshalPointer(const char *marshal) {
        return (void *) strtoul(marshal, NULL, 16);
    }
    template <class T> static inline uint64_t MarshalPointer(const T ptr) {
        return static_cast<uint64_t>(ptr);
    }
    static Buffer * MarshalFatCudaBinary(__cudaFatCudaBinary * bin, Buffer * marshal = NULL);
    static __cudaFatCudaBinary * UnmarshalFatCudaBinary(Buffer * marshal);
    static void DumpFatCudaBinary(__cudaFatCudaBinary * bin, std::ostream & out);

    struct CudaVar {
        char fatCubinHandle[MarshaledHostPointerSize];
        char hostVar[MarshaledDevicePointerSize];
        char deviceAddress[255];
        char deviceName[255];
        int ext;
        int size;
        int constant;
        int global;
    };
private:
};

#endif	/* _CUDAUTIL_H */

