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

#include "CublasHandler.h"
#include <iostream>
#include <cstdio>
#include <string>
using namespace std;

CUBLAS_ROUTINE_HANDLER(Create) {
    cublasHandle_t handle=input_buffer->Get<cublasHandle_t>();
    cublasStatus_t cublas_status=cublasCreate(&handle);

    Buffer *out=new Buffer();

    out->Add<cublasHandle_t>(handle);
    return new Result(cublas_status, out);
}
CUBLAS_ROUTINE_HANDLER(SetMatrix) {
    int rows=input_buffer->Assign<int>();
    int cols=input_buffer->Assign<int>();
    int elemSize=input_buffer->Assign<int>();
    const void *A=input_buffer->Get<void *>();
    int lda=input_buffer->Assign<int>();
    void *B=input_buffer->Assign<void *>();
    int ldb=input_buffer->Assign<int>();
    cublasStatus_t cublas_status=cublasSetMatrix(rows,cols,elemSize,A,lda,B,ldb);
    Buffer *out=new Buffer();
    return new Result(cublas_status, out);
}

CUBLAS_ROUTINE_HANDLER(GetMatrix) {
    int rows=input_buffer->Assign<int>();
    int cols=input_buffer->Assign<int>();
    int elemSize=input_buffer->Assign<int>();
    const void *A=input_buffer->Get<void *>();
    int lda=input_buffer->Assign<int>();
    void *B=input_buffer->Assign<void *>();
    int ldb=input_buffer->Assign<int>();
    cublasStatus_t cublas_status=cublasGetMatrix(rows,cols,elemSize,A,lda,B,ldb);
    Buffer *out=new Buffer();
    return new Result(cublas_status, out);   
}

CUBLAS_ROUTINE_HANDLER(Destroy) {
    cublasHandle_t handle=input_buffer->Get<cublasHandle_t>();
    cublasStatus_t cublas_status=cublasDestroy(handle);

    Buffer *out=new Buffer();
    return new Result(cublas_status, out);
} 
