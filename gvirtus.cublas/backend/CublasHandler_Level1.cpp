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

/*CUBLAS_ROUTINE_HANDLER(GetMatrix) {
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
*/

CUBLAS_ROUTINE_HANDLER(Sdot_v2){
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("Sdot_v2"));
    
    cublasHandle_t handle = (cublasHandle_t)in->Get<long long int>();
    int n = in->Get<int>();
    float * x = in->GetFromMarshal<float*>();
    int incx = in->Get<int>();
    float * y = in->GetFromMarshal<float*>();
    int incy = in->Get<int>();
    float * result = in->Assign<float>();
    
    cublasStatus_t cs = cublasSdot_v2(handle,n,x,incx,y,incy,result);
    
    cout << "DEBUG - cublasSdot_v2 Executed"<<endl;    
    return new Result(cs);
}


CUBLAS_ROUTINE_HANDLER(Ddot_v2){
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("Ddot_v2"));
    
    cublasHandle_t handle = (cublasHandle_t)in->Get<long long int>();
    int n = in->Get<int>();
    double * x = in->GetFromMarshal<double*>();
    int incx = in->Get<int>();
    double * y = in->GetFromMarshal<double*>();
    int incy = in->Get<int>();
    double * result = in->Assign<double>();
    
    cublasStatus_t cs = cublasDdot_v2(handle,n,x,incx,y,incy,result);
    
    cout << "DEBUG - cublasDdot_v2 Executed"<<endl;    
    return new Result(cs);
}


CUBLAS_ROUTINE_HANDLER(Cdotu_v2){
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("Cdotu_v2"));
    cublasHandle_t handle = (cublasHandle_t)in->Get<long long int>();
    int n = in->Get<int>();
    cuComplex * x = in->GetFromMarshal<cuComplex*>();
    int incx = in->Get<int>();
    cuComplex * y = in->GetFromMarshal<cuComplex*>();
    int incy = in->Get<int>();
    cuComplex * result = in->Assign<cuComplex>();
    
    cublasStatus_t cs = cublasCdotu_v2(handle,n,x,incx,y,incy,result);
    
    cout << "DEBUG - cublasCdotu_v2 Executed"<<endl;    
    return new Result(cs);
}

CUBLAS_ROUTINE_HANDLER(Cdotc_v2){
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("Cdotc_v2"));
    cublasHandle_t handle = (cublasHandle_t)in->Get<long long int>();
    int n = in->Get<int>();
    cuComplex * x = in->GetFromMarshal<cuComplex*>();
    int incx = in->Get<int>();
    cuComplex * y = in->GetFromMarshal<cuComplex*>();
    int incy = in->Get<int>();
    cuComplex * result = in->Assign<cuComplex>();
    
    cublasStatus_t cs = cublasCdotc_v2(handle,n,x,incx,y,incy,result);
    
    cout << "DEBUG - cublasCdotc_v2 Executed"<<endl;    
    return new Result(cs);
}

CUBLAS_ROUTINE_HANDLER(Zdotu_v2){
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("Zdotu_v2"));
    cublasHandle_t handle = (cublasHandle_t)in->Get<long long int>();
    int n = in->Get<int>();
    cuDoubleComplex * x = in->GetFromMarshal<cuDoubleComplex*>();
    int incx = in->Get<int>();
    cuDoubleComplex * y = in->GetFromMarshal<cuDoubleComplex*>();
    int incy = in->Get<int>();
    cuDoubleComplex * result = in->Assign<cuDoubleComplex>();
    
    cublasStatus_t cs = cublasZdotu_v2(handle,n,x,incx,y,incy,result);
    
    cout << "DEBUG - cublasZdotu_v2 Executed"<<endl;    
    return new Result(cs);
}

CUBLAS_ROUTINE_HANDLER(Zdotc_v2){
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("Zdotc_v2"));
    cublasHandle_t handle = (cublasHandle_t)in->Get<long long int>();
    int n = in->Get<int>();
    cuDoubleComplex * x = in->GetFromMarshal<cuDoubleComplex*>();
    int incx = in->Get<int>();
    cuDoubleComplex * y = in->GetFromMarshal<cuDoubleComplex*>();
    int incy = in->Get<int>();
    cuDoubleComplex * result = in->Assign<cuDoubleComplex>();
    
    cublasStatus_t cs = cublasZdotc_v2(handle,n,x,incx,y,incy,result);
    
    cout << "DEBUG - cublasZdotc_v2 Executed"<<endl;    
    return new Result(cs);
}

CUBLAS_ROUTINE_HANDLER(Sscal_v2) {
    cublasHandle_t handle = (cublasHandle_t)in->Get<long long int>();
    int n=in->Get<int>();
    float *alpha=in->Assign<float>();
    float *x=in->GetFromMarshal<float*>();
    int incx=in->Get<int>();
    cublasStatus_t cublas_status=cublasSscal(handle,n,alpha,x,incx);
    return new Result(cublas_status);
}

CUBLAS_ROUTINE_HANDLER(Dscal_v2) {
    cublasHandle_t handle = (cublasHandle_t)in->Get<long long int>();
    int n=in->Get<int>();
    double *alpha=in->Assign<double>();
    double *x=in->GetFromMarshal<double*>();
    int incx=in->Get<int>();
    cublasStatus_t cublas_status=cublasDscal(handle,n,alpha,x,incx);
    return new Result(cublas_status);
}

CUBLAS_ROUTINE_HANDLER(Cscal_v2) {
    cublasHandle_t handle = (cublasHandle_t)in->Get<long long int>();
    int n=in->Get<int>();
    cuComplex *alpha=in->Assign<cuComplex>();
    cuComplex *x=in->GetFromMarshal<cuComplex*>();
    int incx=in->Get<int>();
    cublasStatus_t cublas_status=cublasCscal(handle,n,alpha,x,incx);
    return new Result(cublas_status);
}

CUBLAS_ROUTINE_HANDLER(Csscal_v2) {
    cublasHandle_t handle = (cublasHandle_t)in->Get<long long int>();
    int n=in->Get<int>();
    float *alpha=in->Assign<float>();
    cuComplex *x=in->GetFromMarshal<cuComplex*>();
    int incx=in->Get<int>();
    cublasStatus_t cublas_status=cublasCsscal(handle,n,alpha,x,incx);
    return new Result(cublas_status);
}

CUBLAS_ROUTINE_HANDLER(Zscal_v2) {
    cublasHandle_t handle = (cublasHandle_t)in->Get<long long int>();
    int n=in->Get<int>();
    cuDoubleComplex *alpha=in->Assign<cuDoubleComplex>();
    cuDoubleComplex *x=in->GetFromMarshal<cuDoubleComplex*>();
    int incx=in->Get<int>();
    cublasStatus_t cublas_status=cublasZscal(handle,n,alpha,x,incx);
    return new Result(cublas_status);
}

CUBLAS_ROUTINE_HANDLER(Zdscal_v2) {
    cublasHandle_t handle = (cublasHandle_t)in->Get<long long int>();
    int n=in->Get<int>();
    double *alpha=in->Assign<double>();
    cuDoubleComplex *x=in->GetFromMarshal<cuDoubleComplex*>();
    int incx=in->Get<int>();
    cublasStatus_t cublas_status=cublasZdscal(handle,n,alpha,x,incx);
    return new Result(cublas_status);
}

CUBLAS_ROUTINE_HANDLER(Saxpy_v2){
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("Saxpy_v2"));
    
    cublasHandle_t handle = (cublasHandle_t)in->Get<long long int>();
    int n=in->Get<int>();
    float * alpha = in->Assign<float>();
    float * x = in->GetFromMarshal<float*>();
    int incx = in->Get<int>();
    float * y = in->GetFromMarshal<float*>();
    int incy = in->Get<int>();
    cublasStatus_t cs = cublasSaxpy_v2(handle,n,alpha,x,incx,y,incy);
    return new Result(cs);
}

CUBLAS_ROUTINE_HANDLER(Daxpy_v2){
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("Daxpy_v2"));
    
    cublasHandle_t handle = (cublasHandle_t)in->Get<long long int>();
    int n = in->Get<int>();
    double * alpha = in->Assign<double>();
    double * x = in->GetFromMarshal<double*>();
    int incx = in->Get<int>();
    double * y = in->GetFromMarshal<double*>();
    int incy = in->Get<int>();
    cublasStatus_t cs = cublasDaxpy_v2(handle,n,alpha,x,incx,y,incy);
    return new Result(cs);
}

CUBLAS_ROUTINE_HANDLER(Caxpy_v2){
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("Zaxpy_v2"));
    
    cublasHandle_t handle = (cublasHandle_t)in->Get<long long int>();
    int n = in->Get<int>();
    cuComplex * alpha = in->Assign<cuComplex>();
    cuComplex * x = in->GetFromMarshal<cuComplex*>();
    int incx = in->Get<int>();
    cuComplex * y = in->GetFromMarshal<cuComplex*>();
    int incy = in->Get<int>();
    cublasStatus_t cs = cublasCaxpy_v2(handle,n,alpha,x,incx,y,incy);
    return new Result(cs);
}

CUBLAS_ROUTINE_HANDLER(Zaxpy_v2){
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("Zaxpy_v2"));
    
    cublasHandle_t handle = (cublasHandle_t)in->Get<long long int>();
    int n = in->Get<int>();
    cuDoubleComplex * alpha = in->Assign<cuDoubleComplex>();
    cuDoubleComplex * x = in->GetFromMarshal<cuDoubleComplex*>();
    int incx = in->Get<int>();
    cuDoubleComplex * y = in->GetFromMarshal<cuDoubleComplex*>();
    int incy = in->Get<int>();
    cublasStatus_t cs = cublasZaxpy_v2(handle,n,alpha,x,incx,y,incy);
    return new Result(cs);
}


CUBLAS_ROUTINE_HANDLER(Scopy_v2){
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("Scopy_v2"));
    
    cublasHandle_t handle = (cublasHandle_t)in->Get<long long int>();
    int n = in->Get<int>();
    float * x = in->GetFromMarshal<float*>();
    int incx = in->Get<int>();
    float * y = in->GetFromMarshal<float*>();
    int incy = in->Get<int>();
    cublasStatus_t cs = cublasScopy_v2(handle,n,x,incx,y,incy);
    return new Result(cs);
}

CUBLAS_ROUTINE_HANDLER(Dcopy_v2){
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("Dcopy_v2"));
    
    cublasHandle_t handle = (cublasHandle_t)in->Get<long long int>();
    int n = in->Get<int>();
    double * x = in->GetFromMarshal<double*>();
    int incx = in->Get<int>();
    double * y = in->GetFromMarshal<double*>();
    int incy = in->Get<int>();
    cublasStatus_t cs = cublasDcopy_v2(handle,n,x,incx,y,incy);
    return new Result(cs);
}

CUBLAS_ROUTINE_HANDLER(Ccopy_v2){
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("Ccopy_v2"));
    
    cublasHandle_t handle = (cublasHandle_t)in->Get<long long int>();
    int n = in->Get<int>();
    cuComplex * x = in->GetFromMarshal<cuComplex*>();
    int incx = in->Get<int>();
    cuComplex * y = in->GetFromMarshal<cuComplex*>();
    int incy = in->Get<int>();
    cublasStatus_t cs = cublasCcopy_v2(handle,n,x,incx,y,incy);
    return new Result(cs);
}

CUBLAS_ROUTINE_HANDLER(Zcopy_v2){
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("Zcopy_v2"));
    
    cublasHandle_t handle = (cublasHandle_t)in->Get<long long int>();
    int n = in->Get<int>();
    cuDoubleComplex * x = in->GetFromMarshal<cuDoubleComplex*>();
    int incx = in->Get<int>();
    cuDoubleComplex * y = in->GetFromMarshal<cuDoubleComplex*>();
    int incy = in->Get<int>();
    cublasStatus_t cs = cublasZcopy_v2(handle,n,x,incx,y,incy);
    return new Result(cs);
}

CUBLAS_ROUTINE_HANDLER(Sswap_v2){
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("Sswap_v2"));
    
    cublasHandle_t handle = (cublasHandle_t)in->Get<long long int>();
    int n = in->Get<int>();
    float * x = in->GetFromMarshal<float*>();
    int incx = in->Get<int>();
    float * y = in->GetFromMarshal<float*>();
    int incy = in->Get<int>();
    cublasStatus_t cs = cublasSswap_v2(handle,n,x,incx,y,incy);
    return new Result(cs);
}

CUBLAS_ROUTINE_HANDLER(Dswap_v2){
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("Dswap_v2"));
    
    cublasHandle_t handle = (cublasHandle_t)in->Get<long long int>();
    int n = in->Get<int>();
    double * x = in->GetFromMarshal<double*>();
    int incx = in->Get<int>();
    double * y = in->GetFromMarshal<double*>();
    int incy = in->Get<int>();
    cublasStatus_t cs = cublasDswap_v2(handle,n,x,incx,y,incy);
    return new Result(cs);
}

CUBLAS_ROUTINE_HANDLER(Cswap_v2){
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("Dswap_v2"));
    
    cublasHandle_t handle = (cublasHandle_t)in->Get<long long int>();
    int n = in->Get<int>();
    cuComplex * x = in->GetFromMarshal<cuComplex*>();
    int incx = in->Get<int>();
    cuComplex * y = in->GetFromMarshal<cuComplex*>();
    int incy = in->Get<int>();
    cublasStatus_t cs = cublasCswap_v2(handle,n,x,incx,y,incy);
    return new Result(cs);
}

CUBLAS_ROUTINE_HANDLER(Zswap_v2){
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("Zswap_v2"));
    
    cublasHandle_t handle = (cublasHandle_t)in->Get<long long int>();
    int n = in->Get<int>();
    cuDoubleComplex * x = in->GetFromMarshal<cuDoubleComplex*>();
    int incx = in->Get<int>();
    cuDoubleComplex * y = in->GetFromMarshal<cuDoubleComplex*>();
    int incy = in->Get<int>();
    cublasStatus_t cs = cublasZswap_v2(handle,n,x,incx,y,incy);
    return new Result(cs);
}

CUBLAS_ROUTINE_HANDLER(Isamax_v2){
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("Isamax_v2"));
    
    cublasHandle_t handle = (cublasHandle_t)in->Get<long long int>();
    int n = in->Get<int>();
    float * x = in->GetFromMarshal<float*>();
    int incx = in->Get<int>();
    int * result = in->Assign<int>();
    
    cublasStatus_t cs = cublasIsamax_v2(handle,n,x,incx,result);
    Buffer * out = new Buffer();
    out->Add(result);
    return new Result(cs,out);
}

CUBLAS_ROUTINE_HANDLER(Idamax_v2){
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("Isamax_v2"));
    
    cublasHandle_t handle = (cublasHandle_t)in->Get<long long int>();
    int n = in->Get<int>();
    double * x = in->GetFromMarshal<double*>();
    int incx = in->Get<int>();
    int * result = in->Assign<int>();
    
    cublasStatus_t cs = cublasIdamax_v2(handle,n,x,incx,result);
    Buffer * out = new Buffer();
    out->Add(result);
    return new Result(cs,out);
}