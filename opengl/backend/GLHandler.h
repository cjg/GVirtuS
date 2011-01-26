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
 * @file   CudaRtHandler.h
 * @author Giuseppe Coviello <giuseppe.coviello@uniparthenope.it>
 * @date   Sat Oct 10 10:51:58 2009
 * 
 * @brief  
 * 
 * 
 */

#ifndef _GLHANDLER_H
#define	_GLHANDLER_H

#include <iostream>
#include <map>
#include <string>
#include <cstdio>

#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>

#include <cuda_runtime_api.h>

#include "Handler.h"
#include "Result.h"

/**						
 * CudaRtHandler is used by Backend's Process(es) for storing and retrieving
 * device related data and functions. 
 * CudaRtHandler has also the method Execute() that is responsible to execute a
 * named CUDA Runtime routine unmarshalling the input parameters from the
 * provided Buffer.
 */
class GLHandler : public Handler {
public:
    GLHandler();
    virtual ~GLHandler();
    Result * Execute(std::string routine, Buffer * input_buffer);
    const char *InitFramebuffer(size_t size, bool use_shm);
    char *GetFramebuffer();
    inline void Lock() {
        if(mpLock)
            pthread_spin_lock(mpLock);
    }
    inline void Unlock() {
        if(mpLock)
            pthread_spin_unlock(mpLock);
    }
private:
    void Initialize();
    typedef Result * (*GLRoutineHandler)(GLHandler *, Buffer *);
    static std::map<std::string, GLRoutineHandler> * mspHandlers;
    char *mpFramebuffer;
    pthread_spinlock_t *mpLock;
};

#define GL_ROUTINE_HANDLER(name) Result * handle##name(GLHandler * pThis, Buffer * in)
#define GL_ROUTINE_HANDLER_PAIR(name) make_pair("gl" #name, handle##name)

GL_ROUTINE_HANDLER(XChooseVisual);
GL_ROUTINE_HANDLER(XCreateContext);
GL_ROUTINE_HANDLER(XMakeCurrent);
GL_ROUTINE_HANDLER(XQueryExtensionsString);
GL_ROUTINE_HANDLER(XQueryExtension);
GL_ROUTINE_HANDLER(GenLists);
GL_ROUTINE_HANDLER(NewList);
GL_ROUTINE_HANDLER(ShadeModel);
GL_ROUTINE_HANDLER(Normal3f);
GL_ROUTINE_HANDLER(Begin);
GL_ROUTINE_HANDLER(Vertex3f);
GL_ROUTINE_HANDLER(End);
GL_ROUTINE_HANDLER(EndList);
GL_ROUTINE_HANDLER(Viewport);
GL_ROUTINE_HANDLER(MatrixMode);
GL_ROUTINE_HANDLER(LoadIdentity);
GL_ROUTINE_HANDLER(Frustum);
GL_ROUTINE_HANDLER(Translatef);
GL_ROUTINE_HANDLER(CallList);
GL_ROUTINE_HANDLER(Clear);
GL_ROUTINE_HANDLER(PopMatrix);
GL_ROUTINE_HANDLER(PushMatrix);
GL_ROUTINE_HANDLER(Rotatef);
GL_ROUTINE_HANDLER(XSwapBuffers);
GL_ROUTINE_HANDLER(Enable);
GL_ROUTINE_HANDLER(Lightfv);
GL_ROUTINE_HANDLER(Materialfv);
GL_ROUTINE_HANDLER(__ExecuteRoutines);
GL_ROUTINE_HANDLER(__GetBuffer);


#endif	/* _CUDARTHANDLER_H */

