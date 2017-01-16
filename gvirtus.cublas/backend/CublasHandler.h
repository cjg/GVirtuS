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

#include <cublas_api.h>

#include "Handler.h"
#include "Result.h"

class CublasHandler : public Handler {
public:
    CublasHandler();
    virtual ~CublasHandler();
    bool CanExecute(std::string routine);
    Result * Execute(std::string routine, Buffer * input_buffer);


    void* RegisterPointer(void *,size_t);

    void RegisterMapObject(char *,char *);
    char * GetMapObject(char *);


private:
    void Initialize();
    typedef Result * (*CublasRoutineHandler)(CublasHandler *, Buffer *);
    static std::map<std::string, OpenclRoutineHandler> * mspHandlers;
    void **pointers;
    int nPointers;
    std::map<std::string, std::string> * mpMapObject;

    void *mpShm;
    int mShmFd;
};

#define CUBLAS_ROUTINE_HANDLER(name) Result * handle##name(CublasHandler * pThis, Buffer * input_buffer)
#define CUBLAS_ROUTINE_HANDLER_PAIR(name) make_pair("cublas" #name, handle##name)

/* CublasHandler_Platform */
CUBLAS_ROUTINE_HANDLER(Create);
CUBLAS_ROUTINE_HANDLER(SetMatrix);
CUBLAS_ROUTINE_HANDLER(GetMatrix);
CUBLAS_ROUTINE_HANDLER(Sscal);
CUBLAS_ROUTINE_HANDLER(Destroy);

