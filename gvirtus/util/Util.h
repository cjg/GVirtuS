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
 * @file   Util.h
 * @author Giuseppe Coviello <giuseppe.coviello@uniparthenope.it>
 * @date   Sun Oct 11 17:16:48 2009
 * 
 * @brief  
 * 
 * 
 */

#ifndef _UTIL_H
#define	_UTIL_H

#include <iostream>
#include <cstdlib>


#include "Buffer.h"


/**
 *Util contains facility functions used by gVirtuS. These functions
 * includes the ones for marshalling and unmarshalling pointers and "CUDA fat
 * binaries". 
 */
class Util {
public:
    Util();
    Util(const Util& orig);
    virtual ~Util();
    static const size_t MarshaledDevicePointerSize = sizeof(void *) * 2 + 3;
    static const size_t MarshaledHostPointerSize = sizeof(void *) * 2 + 3;
    static char * MarshalHostPointer(const void* ptr);
    static void MarshalHostPointer(const void * ptr, char * marshal);
    static char * MarshalDevicePointer(const void *devPtr);
    static void MarshalDevicePointer(const void *devPtr, char * marshal);
    static inline void * UnmarshalPointer(const char *marshal) {
        return (void *) strtoul(marshal, NULL, 16);
    }
    template <class T> static inline pointer_t MarshalPointer(const T ptr) {
        /*Verify the correctness*/
        return static_cast<pointer_t>(ptr);
    }

private:
};

#endif	/* _UTIL_H */

