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
 * @file   CudaUtil.h
 * @author Giuseppe Coviello <giuseppe.coviello@uniparthenope.it>
 * @date   Sun Oct 11 17:16:48 2009
 * 
 * @brief  
 * 
 * 
 */

#ifndef _OPENCLUTIL_H
#define	_OPENCLUTIL_H

#include <iostream>
#include <cstdlib>

//#include "__cudaFatFormat.h"

#include "Buffer.h"

/**
 * CudaUtil contains facility functions used by gVirtuS. These functions
 * includes the ones for marshalling and unmarshalling pointers and "CUDA fat
 * binaries". 
 */
class OpenclUtil {
public:
    OpenclUtil();
    OpenclUtil(const OpenclUtil& orig);
    virtual ~OpenclUtil();
    //static const size_t MarshaledDevicePointerSize = sizeof(void *) * 2 + 3;
    static const size_t MarshaledHostPointerSize = sizeof(void *) * 2 + 3;
    static char * MarshalHostPointer(const void* ptr);
    static void MarshalHostPointer(const void * ptr, char * marshal);
    //static char * MarshalDevicePointer(const void *devPtr);
    //static void MarshalDevicePointer(const void *devPtr, char * marshal);
    static inline void * UnmarshalPointer(const char *marshal) {
        return (void *) strtoul(marshal, NULL, 16);
    }
    template <class T> static inline uint64_t MarshalPointer(const T ptr) {
        return static_cast<uint64_t>(ptr);
    }
    //static Buffer * MarshalFatCudaBinary(__cudaFatCudaBinary * bin, Buffer * marshal = NULL);
    //static __cudaFatCudaBinary * UnmarshalFatCudaBinary(Buffer * marshal);
    //static void DumpFatCudaBinary(__cudaFatCudaBinary * bin, std::ostream & out);

    /**
     * CudaVar is a data structure used for storing information about shared
     * variables.
     */
    struct OpenclVar {
        char fatCubinHandle[MarshaledHostPointerSize];
       // char hostVar[MarshaledDevicePointerSize];
        char deviceAddress[255];
        char deviceName[255];
        int ext;
        int size;
        int constant;
        int global;
    };
private:
};

#endif	/* _OPENCLUTIL_H */

