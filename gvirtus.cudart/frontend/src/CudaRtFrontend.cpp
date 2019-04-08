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

#include "CudaRtFrontend.h"

using namespace std;

CudaRtFrontend msInstance __attribute_used__;

 map<const void*, mappedPointer>* CudaRtFrontend::mappedPointers = NULL;
 set<const void*>* CudaRtFrontend::devicePointers = NULL;
 map <pthread_t, stack<void*> *>* CudaRtFrontend::toManage = NULL;

CudaRtFrontend::CudaRtFrontend() {
    if (devicePointers == NULL)
        devicePointers = new set<const void*>();
    if (mappedPointers == NULL)
        mappedPointers = new map<const void*, mappedPointer>();
    if (toManage == NULL)
        toManage = new map <pthread_t, stack<void*> *>();
    Frontend::GetFrontend();
}

//static CudaRtFrontend* CudaRtFrontend::GetFrontend(){
//    return (CudaRtFrontend*) Frontend::GetFrontend();
//}
