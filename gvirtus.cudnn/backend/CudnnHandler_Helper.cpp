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

#include "CudnnHandler.h"
#include <iostream>
#include <cstdio>
#include <string>
using namespace std;

CUDNN_ROUTINE_HANDLER(Create) {
    cudnnHandle_t handle=input_buffer->Get<cudnnHandle_t>();
    cudnnStatus_t cudnn_status=cudnnCreate(&handle);

    Buffer *out=new Buffer();

    out->Add<cudnnHandle_t>(handle);
    return new Result(cudnn_status, out);
}
CUDNN_ROUTINE_HANDLER(SetStream) {
    cudnnHandle_t handle=input_buffer->Get<cudnnHandle_t>();   
    cudaStream_t streamId=input_buffer->Assign<cudaStream_t>();
    cudnnStatus_t cudnn_status=cudnnSetStream(handle,streamId);
    Buffer *out=new Buffer();
    return new Result(cudnn_status, out);
}

CUDNN_ROUTINE_HANDLER(GetStream) {
    cudnnHandle_t handle=input_buffer->Get<cudnnHandle_t>();
    cudaStream_t *streamId=input_buffer->GetDevicePointer<cudaStream_t *>();
    cudnnStatus_t cudnn_status=cublasGetStream(handle, streamId);
    Buffer *out=new Buffer();
    return new Result(cudnn_status, out);   
}

CUDNN_ROUTINE_HANDLER(Destroy) {
    cudnnHandle_t handle=input_buffer->Get<cudnnHandle_t>();
    cudnnStatus_t cudnn_status=cudnnDestroy(handle);

    Buffer *out=new Buffer();
    return new Result(cudnn_status, out);
} 
