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

#ifndef _CUDNNHANDLER_H
#define _CUDNNHANDLER_H

#include "Handler.h"
#include "Result.h"
//#include "CudaUtil.h"
#include <cudnn.h>

#include "log4cplus/logger.h"
#include "log4cplus/loggingmacros.h"
#include "log4cplus/configurator.h"

class CudnnHandler : public Handler {
public:
    CudnnHandler();
    virtual ~CudnnHandler();
    bool CanExecute(std::string routine);
    Result * Execute(std::string routine, Buffer * input_buffer);

    /*void * RegisterPointer(void *,size_t);

    void RegisterMapObject(char *,char *);
    char * GetMapObject(char *);
    */
private:
    log4cplus::Logger logger;
    void Initialize();
    typedef Result * (*CudnnRoutineHandler)(CudnnHandler *, Buffer *);
    static std::map<std::string, CudnnRoutineHandler> * mspHandlers;
    //void **pointers;
    //int nPointers;
    
    //std::map<std::string, std::string> * mpMapObject;

    //void *mpShm;
    //int mShmFd;
};

#define CUDNN_ROUTINE_HANDLER(name) Result * handle##name(CudnnHandler * pThis, Buffer * in)
#define CUDNN_ROUTINE_HANDLER_PAIR(name) make_pair("cudnn" #name, handle##name)

/* CudnnHandler_Platform */
CUDNN_ROUTINE_HANDLER(GetVersion);
CUDNN_ROUTINE_HANDLER(Create);
CUDNN_ROUTINE_HANDLER(Destroy);
CUDNN_ROUTINE_HANDLER(GetErrorString);
CUDNN_ROUTINE_HANDLER(SetStream);
CUDNN_ROUTINE_HANDLER(GetStream);
CUDNN_ROUTINE_HANDLER(CreateTensorDescriptor);
CUDNN_ROUTINE_HANDLER(SetTensor4dDescriptor);
CUDNN_ROUTINE_HANDLER(SetTensor4dDescriptorEx);
CUDNN_ROUTINE_HANDLER(GetTensor4dDescriptor);
CUDNN_ROUTINE_HANDLER(SetTensorNdDescriptor);
CUDNN_ROUTINE_HANDLER(DestroyTensorDescriptor);
CUDNN_ROUTINE_HANDLER(TransformTensor);
CUDNN_ROUTINE_HANDLER(AddTensor);
CUDNN_ROUTINE_HANDLER(OpTensor);
CUDNN_ROUTINE_HANDLER(SetTensor);
CUDNN_ROUTINE_HANDLER(ScaleTensor);
CUDNN_ROUTINE_HANDLER(CreateFilterDescriptor);
CUDNN_ROUTINE_HANDLER(SetFilter4dDescriptor);
CUDNN_ROUTINE_HANDLER(GetFilter4dDescriptor);
CUDNN_ROUTINE_HANDLER(SetFilter4dDescriptor_v3);
CUDNN_ROUTINE_HANDLER(GetFilter4dDescriptor_v3);
CUDNN_ROUTINE_HANDLER(SetFilter4dDescriptor_v4);
CUDNN_ROUTINE_HANDLER(GetFilter4dDescriptor_v4);
CUDNN_ROUTINE_HANDLER(SetFilterNdDescriptor);
CUDNN_ROUTINE_HANDLER(GetFilterNdDescriptor);
CUDNN_ROUTINE_HANDLER(SetFilterNdDescriptor_v3);
CUDNN_ROUTINE_HANDLER(GetFilterNdDescriptor_v3);
CUDNN_ROUTINE_HANDLER(SetFilterNdDescriptor_v4);
CUDNN_ROUTINE_HANDLER(GetFilterNdDescriptor_v4);
CUDNN_ROUTINE_HANDLER(DestroyFilterDescriptor);
CUDNN_ROUTINE_HANDLER(CreateConvolutionDescriptor);
CUDNN_ROUTINE_HANDLER(SetConvolution2dDescriptor);
CUDNN_ROUTINE_HANDLER(GetConvolution2dDescriptor);
CUDNN_ROUTINE_HANDLER(GetConvolution2dForwardOutputDim);
#endif //_CUDNNHANDLER_H