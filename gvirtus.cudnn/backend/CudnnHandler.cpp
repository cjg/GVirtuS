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
 */

#include <cstring>
#include <map>
#include <errno.h>

#include "CudnnHandler.h"
#include "CudnnHandler_Helper.cpp"

using namespace std;
using namespace log4cplus;

std::map<string, CudnnHandler::CudnnRoutineHandler> * CudnnHandler::mspHandlers = NULL;

extern "C" int HandlerInit() {
    return 0;
}
extern "C" Handler *GetHandler() {
    return new CudnnHandler();
}

CudnnHandler::CudnnHandler() {
    logger=Logger::getInstance(LOG4CPLUS_TEXT("CudnnHandler"));
    Initialize();
}

CudnnHandler::~CudnnHandler() {

}

bool CudnnHandler::CanExecute(std::string routine) {
    return mspHandlers->find(routine) != mspHandlers->end();
}

Result * CudnnHandler::Execute(std::string routine, Buffer * in) {
    LOG4CPLUS_DEBUG(logger,"Called " << routine);
    map<string, CudnnHandler::CudnnRoutineHandler>::iterator it;
    it = mspHandlers->find(routine);
    if (it == mspHandlers->end())
        throw "No handler for '" + routine + "' found!";
    try {
        return it->second(this, in);
    } catch (const char *ex) {
        cout << ex << endl;
        cout << strerror(errno) << endl;
    }
    return NULL;
}


void CudnnHandler::Initialize() {
    if (mspHandlers != NULL)
        return;
    mspHandlers = new map<string, CudnnHandler::CudnnRoutineHandler> ();

    /* CublasHandler Query Platform Info */
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetVersion));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(Create));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetErrorString));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(Destroy));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(SetStream));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetStream));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(CreateTensorDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(SetTensor4dDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(SetTensor4dDescriptorEx));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetTensor4dDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(SetTensorNdDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(DestroyTensorDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(TransformTensor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(AddTensor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(OpTensor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(SetTensor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(ScaleTensor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(CreateFilterDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(SetFilter4dDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetFilter4dDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(SetFilter4dDescriptor_v3));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetFilter4dDescriptor_v3));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(SetFilter4dDescriptor_v4));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetFilter4dDescriptor_v4));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(SetFilterNdDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetFilterNdDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(SetFilterNdDescriptor_v3));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetFilterNdDescriptor_v3));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(SetFilterNdDescriptor_v4));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetFilterNdDescriptor_v4));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(DestroyFilterDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(CreateConvolutionDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(SetConvolution2dDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetConvolution2dDescriptor));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(GetConvolution2dForwardOutputDim));
}