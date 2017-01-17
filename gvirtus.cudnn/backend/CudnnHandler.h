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
 * Written by: Raffaele Montella <raffaele.montella@uniparthenope.it>,
 *             Department of Science and Technologies
 */

#include <iostream>
#include <map>
#include <string>
#include <cstdio>

#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>

#include <cudnn.h>

#include "Handler.h"
#include "Result.h"

class CudnnHandler : public Handler {
public:
    CudnnHandler();
    virtual ~CudnnHandler();
    bool CanExecute(std::string routine);
    Result * Execute(std::string routine, Buffer * input_buffer);


    void* RegisterPointer(void *,size_t);

    void RegisterMapObject(char *,char *);
    char * GetMapObject(char *);


private:
    void Initialize();
    typedef Result * (*CudnnRoutineHandler)(CudnnHandler *, Buffer *);
    static std::map<std::string, CudnnRoutineHandler> * mspHandlers;
    void **pointers;
    int nPointers;
    std::map<std::string, std::string> * mpMapObject;

    void *mpShm;
    int mShmFd;
};

#define CUDNN_ROUTINE_HANDLER(name) Result * handle##name(CudnnHandler * pThis, Buffer * input_buffer)
#define CUDNN_ROUTINE_HANDLER_PAIR(name) make_pair("cudnn" #name, handle##name)

/* CudnnHandler_Platform */
CUDNN_ROUTINE_HANDLER(Create);
CUDNN_ROUTINE_HANDLER(SetStream);
CUDNN_ROUTINE_HANDLER(GetStream);
CUDNN_ROUTINE_HANDLER(Sscal);
CUDNN_ROUTINE_HANDLER(Destroy);

