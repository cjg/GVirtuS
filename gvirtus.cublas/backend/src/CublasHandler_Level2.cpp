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
 * Written by: Giuseppe Coviello <vincenzo.santopietro@uniparthenope.it>,
 *             Department of Science and Technologies
 */

#include "CublasHandler.h"
#include <iostream>
#include <cstdio>
#include <string>

using namespace std;
using namespace log4cplus;


CUBLAS_ROUTINE_HANDLER(Sgemv_v2){
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("Sgemv"));
    
    cublasHandle_t handle;
    handle = (cublasHandle_t) in->Get<long long int>();
    cublasOperation_t trans = in->Get<cublasOperation_t>();
    int m  = in->Get<int>();
    int n  = in->Get<int>();
    const float * alpha = in->Assign<float>();
    cout << "alpha: "<<*alpha<<endl;
    float * A = in->GetFromMarshal<float*>();
    int lda = in->Get<int>();
    float * x = in->GetFromMarshal<float*>();
    int incx = in->Get<int>();
    
    const float * beta = in->Assign<float>();
    float * y = in->GetFromMarshal<float*>();
    int incy = in->Get<int>();
    cublasStatus_t cs;
    Buffer * out = new Buffer();
    try{
        cs = cublasSgemv_v2(handle,trans,m,n,alpha,A,lda,x,incx,beta,y,incy);
        if (cs == CUBLAS_STATUS_INVALID_VALUE)
            cout<<"invalid value"<<endl;
        if (cs == CUBLAS_STATUS_ARCH_MISMATCH)
            cout<<"arch mismatch"<<endl;
        if( cs == CUBLAS_STATUS_EXECUTION_FAILED)
            cout<<"Execution failed"<<endl;
        out->AddMarshal<float *>(y);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return new Result(cudaErrorMemoryAllocation);
    }
    cout << "DEBUG - cublasSgemv_v2 Executed"<<endl;
    return new Result(cs,out);
}

CUBLAS_ROUTINE_HANDLER(Dgemv_v2){
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("Dgemv"));
    
    cublasHandle_t handle;
    handle = (cublasHandle_t) in->Get<long long int>();
    cublasOperation_t trans = in->Get<cublasOperation_t>();
    int m  = in->Get<int>();
    int n  = in->Get<int>();
    const double * alpha = in->Assign<double>();
    cout << "alpha: "<<*alpha<<endl;
    double * A = in->GetFromMarshal<double*>();
    int lda = in->Get<int>();
    double * x = in->GetFromMarshal<double*>();
    int incx = in->Get<int>();
    
    const double * beta = in->Assign<double>();
    double * y = in->GetFromMarshal<double*>();
    int incy = in->Get<int>();
    cublasStatus_t cs;
    Buffer * out = new Buffer();
    try{
        cs = cublasDgemv_v2(handle,trans,m,n,alpha,A,lda,x,incx,beta,y,incy);
        if (cs == CUBLAS_STATUS_INVALID_VALUE)
            cout<<"invalid value"<<endl;
        if (cs == CUBLAS_STATUS_ARCH_MISMATCH)
            cout<<"arch mismatch"<<endl;
        if( cs == CUBLAS_STATUS_EXECUTION_FAILED)
            cout<<"Execution failed"<<endl;
        out->AddMarshal<double *>(y);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return new Result(cudaErrorMemoryAllocation);
    }
    cout << "DEBUG - cublasDgemv_v2 Executed"<<endl;
    return new Result(cs,out);
}

CUBLAS_ROUTINE_HANDLER(Cgemv_v2){
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("Cgemv_v2"));
    
    cublasHandle_t handle;
    handle = (cublasHandle_t) in->Get<long long int>();
    cublasOperation_t trans = in->Get<cublasOperation_t>();
    int m  = in->Get<int>();
    int n  = in->Get<int>();
    const cuComplex * alpha = in->Assign<cuComplex>();
    cuComplex * A = in->GetFromMarshal<cuComplex*>();
    int lda = in->Get<int>();
    cuComplex * x = in->GetFromMarshal<cuComplex*>();
    int incx = in->Get<int>();
    
    const cuComplex * beta = in->Assign<cuComplex>();
    cuComplex * y = in->GetFromMarshal<cuComplex*>();
    int incy = in->Get<int>();
    cublasStatus_t cs;
    Buffer * out = new Buffer();
    try{
        cs = cublasCgemv_v2(handle,trans,m,n,alpha,A,lda,x,incx,beta,y,incy);
        if (cs == CUBLAS_STATUS_INVALID_VALUE)
            cout<<"invalid value"<<endl;
        if (cs == CUBLAS_STATUS_ARCH_MISMATCH)
            cout<<"arch mismatch"<<endl;
        if( cs == CUBLAS_STATUS_EXECUTION_FAILED)
            cout<<"Execution failed"<<endl;
        out->AddMarshal<cuComplex *>(y);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return new Result(cudaErrorMemoryAllocation);
    }
    cout << "DEBUG - cublasCgemv_v2 Executed"<<endl;
    return new Result(cs,out);
}


CUBLAS_ROUTINE_HANDLER(Zgemv_v2){
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("Zgemv_v2"));
    
    cublasHandle_t handle;
    handle = (cublasHandle_t) in->Get<long long int>();
    cublasOperation_t trans = in->Get<cublasOperation_t>();
    int m  = in->Get<int>();
    int n  = in->Get<int>();
    const cuDoubleComplex * alpha = in->Assign<cuDoubleComplex>();
    cuDoubleComplex * A = in->GetFromMarshal<cuDoubleComplex*>();
    int lda = in->Get<int>();
    cuDoubleComplex * x = in->GetFromMarshal<cuDoubleComplex*>();
    int incx = in->Get<int>();
    
    const cuDoubleComplex * beta = in->Assign<cuDoubleComplex>();
    cuDoubleComplex * y = in->GetFromMarshal<cuDoubleComplex*>();
    int incy = in->Get<int>();
    cublasStatus_t cs;
    Buffer * out = new Buffer();
    try{
        cs = cublasZgemv_v2(handle,trans,m,n,alpha,A,lda,x,incx,beta,y,incy);
        if (cs == CUBLAS_STATUS_INVALID_VALUE)
            cout<<"invalid value"<<endl;
        if (cs == CUBLAS_STATUS_ARCH_MISMATCH)
            cout<<"arch mismatch"<<endl;
        if( cs == CUBLAS_STATUS_EXECUTION_FAILED)
            cout<<"Execution failed"<<endl;
        out->AddMarshal<cuDoubleComplex *>(y);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return new Result(cudaErrorMemoryAllocation);
    }
    cout << "DEBUG - cublasZgemv_v2 Executed"<<endl;
    return new Result(cs,out);
}

CUBLAS_ROUTINE_HANDLER(Sgbmv_v2){
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("Sgbmv_v2"));
    
    cublasHandle_t handle;
    handle = (cublasHandle_t) in->Get<long long int>();
    cublasOperation_t trans = in->Get<cublasOperation_t>();
    int m  = in->Get<int>();
    int n  = in->Get<int>();
    int kl  = in->Get<int>();
    int ku  = in->Get<int>();
    const float * alpha = in->Assign<float>();
    float * A = in->GetFromMarshal<float*>();
    int lda  = in->Get<int>();
    const float * x = in->GetFromMarshal<float*>();
    int incx = in->Get<int>();
    const float * beta = in->Assign<float>();
    float * y = in->GetFromMarshal<float*>();
    int incy = in->Get<int>();
    
    cublasStatus_t cs = cublasSgbmv_v2(handle,trans,m,n,kl,ku,alpha,A,lda,x,incx,beta,y,incy);
    return new Result(cs);
}

CUBLAS_ROUTINE_HANDLER(Dgbmv_v2){
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("Dgbmv_v2"));
    
    cublasHandle_t handle;
    handle = (cublasHandle_t) in->Get<long long int>();
    cublasOperation_t trans = in->Get<cublasOperation_t>();
    int m  = in->Get<int>();
    int n  = in->Get<int>();
    int kl  = in->Get<int>();
    int ku  = in->Get<int>();
    const double * alpha = in->Assign<double>();
    double * A = in->GetFromMarshal<double*>();
    int lda  = in->Get<int>();
    const double * x = in->GetFromMarshal<double*>();
    int incx = in->Get<int>();
    const double * beta = in->Assign<double>();
    double * y = in->GetFromMarshal<double*>();
    int incy = in->Get<int>();
    
    cublasStatus_t cs = cublasDgbmv_v2(handle,trans,m,n,kl,ku,alpha,A,lda,x,incx,beta,y,incy);
    return new Result(cs);
}

CUBLAS_ROUTINE_HANDLER(Cgbmv_v2){
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("Cgbmv_v2"));
    
    cublasHandle_t handle;
    handle = (cublasHandle_t) in->Get<long long int>();
    cublasOperation_t trans = in->Get<cublasOperation_t>();
    int m  = in->Get<int>();
    int n  = in->Get<int>();
    int kl  = in->Get<int>();
    int ku  = in->Get<int>();
    const cuComplex * alpha = in->Assign<cuComplex>();
    cuComplex * A = in->GetFromMarshal<cuComplex*>();
    int lda  = in->Get<int>();
    const cuComplex * x = in->GetFromMarshal<cuComplex*>();
    int incx = in->Get<int>();
    const cuComplex * beta = in->Assign<cuComplex>();
    cuComplex * y = in->GetFromMarshal<cuComplex*>();
    int incy = in->Get<int>();
    
    cublasStatus_t cs = cublasCgbmv_v2(handle,trans,m,n,kl,ku,alpha,A,lda,x,incx,beta,y,incy);
    return new Result(cs);
}

CUBLAS_ROUTINE_HANDLER(Zgbmv_v2){
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("Zgbmv_v2"));
    
    cublasHandle_t handle;
    handle = (cublasHandle_t) in->Get<long long int>();
    cublasOperation_t trans = in->Get<cublasOperation_t>();
    int m  = in->Get<int>();
    int n  = in->Get<int>();
    int kl  = in->Get<int>();
    int ku  = in->Get<int>();
    const cuDoubleComplex * alpha = in->Assign<cuDoubleComplex>();
    cuDoubleComplex * A = in->GetFromMarshal<cuDoubleComplex*>();
    int lda  = in->Get<int>();
    const cuDoubleComplex * x = in->GetFromMarshal<cuDoubleComplex*>();
    int incx = in->Get<int>();
    const cuDoubleComplex * beta = in->Assign<cuDoubleComplex>();
    cuDoubleComplex * y = in->GetFromMarshal<cuDoubleComplex*>();
    int incy = in->Get<int>();
    
    cublasStatus_t cs = cublasZgbmv_v2(handle,trans,m,n,kl,ku,alpha,A,lda,x,incx,beta,y,incy);
    return new Result(cs);
}

CUBLAS_ROUTINE_HANDLER(Strmv_v2){
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("Strmv_v2"));
    
    cublasHandle_t handle;
    handle = (cublasHandle_t) in->Get<long long int>();
    cublasFillMode_t uplo = in->Get<cublasFillMode_t>();
    cublasOperation_t trans = in->Get<cublasOperation_t>();
    cublasDiagType_t diag = in->Get<cublasDiagType_t>();
    int n = in->Get<int>();
    float * A = in->GetFromMarshal<float*>();
    int lda = in->Get<int>();
    float * x = in->GetFromMarshal<float*>();
    int incx = in->Get<int>();
    
    cublasStatus_t cs = cublasStrmv_v2(handle,uplo,trans,diag,n,A,lda,x,incx);
    return new Result(cs);
}

CUBLAS_ROUTINE_HANDLER(Dtrmv_v2){
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("Dtrmv_v2"));
    
    cublasHandle_t handle;
    handle = (cublasHandle_t) in->Get<long long int>();
    cublasFillMode_t uplo = in->Get<cublasFillMode_t>();
    cublasOperation_t trans = in->Get<cublasOperation_t>();
    cublasDiagType_t diag = in->Get<cublasDiagType_t>();
    int n = in->Get<int>();
    double * A = in->GetFromMarshal<double*>();
    int lda = in->Get<int>();
    double * x = in->GetFromMarshal<double*>();
    int incx = in->Get<int>();
    
    cublasStatus_t cs = cublasDtrmv_v2(handle,uplo,trans,diag,n,A,lda,x,incx);
    return new Result(cs);
}

CUBLAS_ROUTINE_HANDLER(Ctrmv_v2){
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("Ctrmv_v2"));
    
    cublasHandle_t handle;
    handle = (cublasHandle_t) in->Get<long long int>();
    cublasFillMode_t uplo = in->Get<cublasFillMode_t>();
    cublasOperation_t trans = in->Get<cublasOperation_t>();
    cublasDiagType_t diag = in->Get<cublasDiagType_t>();
    int n = in->Get<int>();
    cuComplex * A = in->GetFromMarshal<cuComplex*>();
    int lda = in->Get<int>();
    cuComplex * x = in->GetFromMarshal<cuComplex*>();
    int incx = in->Get<int>();
    
    cublasStatus_t cs = cublasCtrmv_v2(handle,uplo,trans,diag,n,A,lda,x,incx);
    return new Result(cs);
}

CUBLAS_ROUTINE_HANDLER(Ztrmv_v2){
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("Ztrmv_v2"));
    
    cublasHandle_t handle;
    handle = (cublasHandle_t) in->Get<long long int>();
    cublasFillMode_t uplo = in->Get<cublasFillMode_t>();
    cublasOperation_t trans = in->Get<cublasOperation_t>();
    cublasDiagType_t diag = in->Get<cublasDiagType_t>();
    int n = in->Get<int>();
    cuDoubleComplex * A = in->GetFromMarshal<cuDoubleComplex*>();
    int lda = in->Get<int>();
    cuDoubleComplex * x = in->GetFromMarshal<cuDoubleComplex*>();
    int incx = in->Get<int>();
    
    cublasStatus_t cs = cublasZtrmv_v2(handle,uplo,trans,diag,n,A,lda,x,incx);
    return new Result(cs);
}

CUBLAS_ROUTINE_HANDLER(Stbmv_v2){
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("Stbmv_v2"));
    
    cublasHandle_t handle;
    handle = (cublasHandle_t) in->Get<long long int>();
    cublasFillMode_t uplo = in->Get<cublasFillMode_t>();
    cublasOperation_t trans = in->Get<cublasOperation_t>();
    cublasDiagType_t diag = in->Get<cublasDiagType_t>();
    int n = in->Get<int>();
    int k = in->Get<int>();
    float * A = in->GetFromMarshal<float*>();
    int lda = in->Get<int>();
    float * x = in->GetFromMarshal<float*>();
    int incx = in->Get<int>();
    
    cublasStatus_t cs = cublasStbmv_v2(handle,uplo,trans,diag,n,k,A,lda,x,incx);
    return new Result(cs);
}

CUBLAS_ROUTINE_HANDLER(Dtbmv_v2){
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("Dtbmv_v2"));
    
    cublasHandle_t handle;
    handle = (cublasHandle_t) in->Get<long long int>();
    cublasFillMode_t uplo = in->Get<cublasFillMode_t>();
    cublasOperation_t trans = in->Get<cublasOperation_t>();
    cublasDiagType_t diag = in->Get<cublasDiagType_t>();
    int n = in->Get<int>();
    int k = in->Get<int>();
    double * A = in->GetFromMarshal<double*>();
    int lda = in->Get<int>();
    double * x = in->GetFromMarshal<double*>();
    int incx = in->Get<int>();
    
    cublasStatus_t cs = cublasDtbmv_v2(handle,uplo,trans,diag,n,k,A,lda,x,incx);
    return new Result(cs);
}

CUBLAS_ROUTINE_HANDLER(Ctbmv_v2){
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("Ctbmv_v2"));
    
    cublasHandle_t handle;
    handle = (cublasHandle_t) in->Get<long long int>();
    cublasFillMode_t uplo = in->Get<cublasFillMode_t>();
    cublasOperation_t trans = in->Get<cublasOperation_t>();
    cublasDiagType_t diag = in->Get<cublasDiagType_t>();
    int n = in->Get<int>();
    int k = in->Get<int>();
    cuComplex * A = in->GetFromMarshal<cuComplex*>();
    int lda = in->Get<int>();
    cuComplex * x = in->GetFromMarshal<cuComplex*>();
    int incx = in->Get<int>();
    
    cublasStatus_t cs = cublasCtbmv_v2(handle,uplo,trans,diag,n,k,A,lda,x,incx);
    return new Result(cs);
}

CUBLAS_ROUTINE_HANDLER(Ztbmv_v2){
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("Ztbmv_v2"));
    
    cublasHandle_t handle;
    handle = (cublasHandle_t) in->Get<long long int>();
    cublasFillMode_t uplo = in->Get<cublasFillMode_t>();
    cublasOperation_t trans = in->Get<cublasOperation_t>();
    cublasDiagType_t diag = in->Get<cublasDiagType_t>();
    int n = in->Get<int>();
    int k = in->Get<int>();
    cuDoubleComplex * A = in->GetFromMarshal<cuDoubleComplex*>();
    int lda = in->Get<int>();
    cuDoubleComplex * x = in->GetFromMarshal<cuDoubleComplex*>();
    int incx = in->Get<int>();
    
    cublasStatus_t cs = cublasZtbmv_v2(handle,uplo,trans,diag,n,k,A,lda,x,incx);
    return new Result(cs);
}

CUBLAS_ROUTINE_HANDLER(Stpmv_v2){
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("Stpmv_v2"));
    
    cublasHandle_t handle;
    handle = (cublasHandle_t) in->Get<long long int>();
    cublasFillMode_t uplo = in->Get<cublasFillMode_t>();
    cublasOperation_t trans = in->Get<cublasOperation_t>();
    cublasDiagType_t diag = in->Get<cublasDiagType_t>();
    int n = in->Get<int>();
    float * AP = in->GetFromMarshal<float*>();
    float * x = in->GetFromMarshal<float*>();
    int incx = in->Get<int>();
    
    cublasStatus_t cs = cublasStpmv_v2(handle,uplo,trans,diag,n,AP,x,incx);
    return new Result(cs);
}

CUBLAS_ROUTINE_HANDLER(Dtpmv_v2){
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("Dtpmv_v2"));
    
    cublasHandle_t handle;
    handle = (cublasHandle_t) in->Get<long long int>();
    cublasFillMode_t uplo = in->Get<cublasFillMode_t>();
    cublasOperation_t trans = in->Get<cublasOperation_t>();
    cublasDiagType_t diag = in->Get<cublasDiagType_t>();
    int n = in->Get<int>();
    double * AP = in->GetFromMarshal<double*>();
    double * x = in->GetFromMarshal<double*>();
    int incx = in->Get<int>();
    
    cublasStatus_t cs = cublasDtpmv_v2(handle,uplo,trans,diag,n,AP,x,incx);
    return new Result(cs);
}

CUBLAS_ROUTINE_HANDLER(Ctpmv_v2){
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("Ctpmv_v2"));
    
    cublasHandle_t handle;
    handle = (cublasHandle_t) in->Get<long long int>();
    cublasFillMode_t uplo = in->Get<cublasFillMode_t>();
    cublasOperation_t trans = in->Get<cublasOperation_t>();
    cublasDiagType_t diag = in->Get<cublasDiagType_t>();
    int n = in->Get<int>();
    cuComplex * AP = in->GetFromMarshal<cuComplex*>();
    cuComplex * x = in->GetFromMarshal<cuComplex*>();
    int incx = in->Get<int>();
    
    cublasStatus_t cs = cublasCtpmv_v2(handle,uplo,trans,diag,n,AP,x,incx);
    return new Result(cs);
}

CUBLAS_ROUTINE_HANDLER(Ztpmv_v2){
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("Ztpmv_v2"));
    
    cublasHandle_t handle;
    handle = (cublasHandle_t) in->Get<long long int>();
    cublasFillMode_t uplo = in->Get<cublasFillMode_t>();
    cublasOperation_t trans = in->Get<cublasOperation_t>();
    cublasDiagType_t diag = in->Get<cublasDiagType_t>();
    int n = in->Get<int>();
    cuDoubleComplex * AP = in->GetFromMarshal<cuDoubleComplex*>();
    cuDoubleComplex * x = in->GetFromMarshal<cuDoubleComplex*>();
    int incx = in->Get<int>();
    
    cublasStatus_t cs = cublasZtpmv_v2(handle,uplo,trans,diag,n,AP,x,incx);
    return new Result(cs);
}

CUBLAS_ROUTINE_HANDLER(Strsv_v2){
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("Strsv_v2"));
    
    cublasHandle_t handle;
    handle = (cublasHandle_t) in->Get<long long int>();
    cublasFillMode_t uplo = in->Get<cublasFillMode_t>();
    cublasOperation_t trans = in->Get<cublasOperation_t>();
    cublasDiagType_t diag = in->Get<cublasDiagType_t>();
    int n = in->Get<int>();
    float * A = in->GetFromMarshal<float*>();
    int lda = in->Get<int>();
    float * x = in->GetFromMarshal<float*>();
    int incx = in->Get<int>();
    
    cublasStatus_t cs = cublasStrsv_v2(handle,uplo,trans,diag,n,A,lda,x,incx);
    return new Result(cs);
}

CUBLAS_ROUTINE_HANDLER(Dtrsv_v2){
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("Dtrsv_v2"));
    
    cublasHandle_t handle;
    handle = (cublasHandle_t) in->Get<long long int>();
    cublasFillMode_t uplo = in->Get<cublasFillMode_t>();
    cublasOperation_t trans = in->Get<cublasOperation_t>();
    cublasDiagType_t diag = in->Get<cublasDiagType_t>();
    int n = in->Get<int>();
    double * A = in->GetFromMarshal<double*>();
    int lda = in->Get<int>();
    double * x = in->GetFromMarshal<double*>();
    int incx = in->Get<int>();
    
    cublasStatus_t cs = cublasDtrsv_v2(handle,uplo,trans,diag,n,A,lda,x,incx);
    return new Result(cs);
}

CUBLAS_ROUTINE_HANDLER(Ctrsv_v2){
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("Ctrsv_v2"));
    
    cublasHandle_t handle;
    handle = (cublasHandle_t) in->Get<long long int>();
    cublasFillMode_t uplo = in->Get<cublasFillMode_t>();
    cublasOperation_t trans = in->Get<cublasOperation_t>();
    cublasDiagType_t diag = in->Get<cublasDiagType_t>();
    int n = in->Get<int>();
    cuComplex * A = in->GetFromMarshal<cuComplex*>();
    int lda = in->Get<int>();
    cuComplex * x = in->GetFromMarshal<cuComplex*>();
    int incx = in->Get<int>();
    
    cublasStatus_t cs = cublasCtrsv_v2(handle,uplo,trans,diag,n,A,lda,x,incx);
    return new Result(cs);
}

CUBLAS_ROUTINE_HANDLER(Ztrsv_v2){
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("Ztrsv_v2"));
    
    cublasHandle_t handle;
    handle = (cublasHandle_t) in->Get<long long int>();
    cublasFillMode_t uplo = in->Get<cublasFillMode_t>();
    cublasOperation_t trans = in->Get<cublasOperation_t>();
    cublasDiagType_t diag = in->Get<cublasDiagType_t>();
    int n = in->Get<int>();
    cuDoubleComplex * A = in->GetFromMarshal<cuDoubleComplex*>();
    int lda = in->Get<int>();
    cuDoubleComplex * x = in->GetFromMarshal<cuDoubleComplex*>();
    int incx = in->Get<int>();
    
    cublasStatus_t cs = cublasZtrsv_v2(handle,uplo,trans,diag,n,A,lda,x,incx);
    return new Result(cs);
}


CUBLAS_ROUTINE_HANDLER(Stpsv_v2){
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("Stpsv_v2"));
    
    cublasHandle_t handle;
    handle = (cublasHandle_t) in->Get<long long int>();
    cublasFillMode_t uplo = in->Get<cublasFillMode_t>();
    cublasOperation_t trans = in->Get<cublasOperation_t>();
    cublasDiagType_t diag = in->Get<cublasDiagType_t>();
    int n = in->Get<int>();
    float * AP = in->GetFromMarshal<float*>();
    float * x = in->GetFromMarshal<float*>();
    int incx = in->Get<int>();
    
    cublasStatus_t cs = cublasStpsv_v2(handle,uplo,trans,diag,n,AP,x,incx);
    return new Result(cs);
}

CUBLAS_ROUTINE_HANDLER(Dtpsv_v2){
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("Dtpsv_v2"));
    
    cublasHandle_t handle;
    handle = (cublasHandle_t) in->Get<long long int>();
    cublasFillMode_t uplo = in->Get<cublasFillMode_t>();
    cublasOperation_t trans = in->Get<cublasOperation_t>();
    cublasDiagType_t diag = in->Get<cublasDiagType_t>();
    int n = in->Get<int>();
    double * AP = in->GetFromMarshal<double*>();
    double * x = in->GetFromMarshal<double*>();
    int incx = in->Get<int>();
    
    cublasStatus_t cs = cublasDtpsv_v2(handle,uplo,trans,diag,n,AP,x,incx);
    return new Result(cs);
}

CUBLAS_ROUTINE_HANDLER(Ctpsv_v2){
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("Ctpsv_v2"));
    
    cublasHandle_t handle;
    handle = (cublasHandle_t) in->Get<long long int>();
    cublasFillMode_t uplo = in->Get<cublasFillMode_t>();
    cublasOperation_t trans = in->Get<cublasOperation_t>();
    cublasDiagType_t diag = in->Get<cublasDiagType_t>();
    int n = in->Get<int>();
    cuComplex * AP = in->GetFromMarshal<cuComplex*>();
    cuComplex * x = in->GetFromMarshal<cuComplex*>();
    int incx = in->Get<int>();
    
    cublasStatus_t cs = cublasCtpsv_v2(handle,uplo,trans,diag,n,AP,x,incx);
    return new Result(cs);
}

CUBLAS_ROUTINE_HANDLER(Ztpsv_v2){
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("Ztpsv_v2"));
    
    cublasHandle_t handle;
    handle = (cublasHandle_t) in->Get<long long int>();
    cublasFillMode_t uplo = in->Get<cublasFillMode_t>();
    cublasOperation_t trans = in->Get<cublasOperation_t>();
    cublasDiagType_t diag = in->Get<cublasDiagType_t>();
    int n = in->Get<int>();
    cuDoubleComplex * AP = in->GetFromMarshal<cuDoubleComplex*>();
    cuDoubleComplex * x = in->GetFromMarshal<cuDoubleComplex*>();
    int incx = in->Get<int>();
    
    cublasStatus_t cs = cublasZtpsv_v2(handle,uplo,trans,diag,n,AP,x,incx);
    return new Result(cs);
}

CUBLAS_ROUTINE_HANDLER(Stbsv_v2){
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("Stbsv_v2"));
    
    cublasHandle_t handle;
    handle = (cublasHandle_t) in->Get<long long int>();
    cublasFillMode_t uplo = in->Get<cublasFillMode_t>();
    cublasOperation_t trans = in->Get<cublasOperation_t>();
    cublasDiagType_t diag = in->Get<cublasDiagType_t>();
    int n = in->Get<int>();
    int k = in->Get<int>();
    float * A = in->GetFromMarshal<float*>();
    int lda = in->Get<int>();
    float * x = in->GetFromMarshal<float*>();
    int incx = in->Get<int>();

    cublasStatus_t cs = cublasStbsv_v2(handle,uplo,trans,diag,n,k,A,lda,x,incx);
    return new Result(cs);
}

CUBLAS_ROUTINE_HANDLER(Dtbsv_v2){
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("Dtbsv_v2"));
    
    cublasHandle_t handle;
    handle = (cublasHandle_t) in->Get<long long int>();
    cublasFillMode_t uplo = in->Get<cublasFillMode_t>();
    cublasOperation_t trans = in->Get<cublasOperation_t>();
    cublasDiagType_t diag = in->Get<cublasDiagType_t>();
    int n = in->Get<int>();
    int k = in->Get<int>();
    double * A = in->GetFromMarshal<double*>();
    int lda = in->Get<int>();
    double * x = in->GetFromMarshal<double*>();
    int incx = in->Get<int>();

    cublasStatus_t cs = cublasDtbsv_v2(handle,uplo,trans,diag,n,k,A,lda,x,incx);
    return new Result(cs);
}

CUBLAS_ROUTINE_HANDLER(Ctbsv_v2){
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("Ctbsv_v2"));
    
    cublasHandle_t handle;
    handle = (cublasHandle_t) in->Get<long long int>();
    cublasFillMode_t uplo = in->Get<cublasFillMode_t>();
    cublasOperation_t trans = in->Get<cublasOperation_t>();
    cublasDiagType_t diag = in->Get<cublasDiagType_t>();
    int n = in->Get<int>();
    int k = in->Get<int>();
    cuComplex * A = in->GetFromMarshal<cuComplex*>();
    int lda = in->Get<int>();
    cuComplex * x = in->GetFromMarshal<cuComplex*>();
    int incx = in->Get<int>();

    cublasStatus_t cs = cublasCtbsv_v2(handle,uplo,trans,diag,n,k,A,lda,x,incx);
    return new Result(cs);
}

CUBLAS_ROUTINE_HANDLER(Ztbsv_v2){
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("Ztbsv_v2"));
    
    cublasHandle_t handle;
    handle = (cublasHandle_t) in->Get<long long int>();
    cublasFillMode_t uplo = in->Get<cublasFillMode_t>();
    cublasOperation_t trans = in->Get<cublasOperation_t>();
    cublasDiagType_t diag = in->Get<cublasDiagType_t>();
    int n = in->Get<int>();
    int k = in->Get<int>();
    cuDoubleComplex * A = in->GetFromMarshal<cuDoubleComplex*>();
    int lda = in->Get<int>();
    cuDoubleComplex * x = in->GetFromMarshal<cuDoubleComplex*>();
    int incx = in->Get<int>();

    cublasStatus_t cs = cublasZtbsv_v2(handle,uplo,trans,diag,n,k,A,lda,x,incx);
    return new Result(cs);
}

CUBLAS_ROUTINE_HANDLER(Ssymv_v2){
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("Ssymv_v2"));
    
    cublasHandle_t handle;
    handle = (cublasHandle_t) in->Get<long long int>();
    cublasFillMode_t uplo = in->Get<cublasFillMode_t>();
    int n = in->Get<int>();
    const float * alpha = in->Assign<float>();
    float * A = in->GetFromMarshal<float*>();
    int lda = in->Get<int>();
    float * x = in->GetFromMarshal<float*>();
    int incx = in->Get<int>();
    
    const float * beta = in->Assign<float>();
    float * y = in->GetFromMarshal<float*>();
    int incy = in->Get<int>();
    
    cublasStatus_t cs = cublasSsymv_v2(handle,uplo,n,alpha,A,lda,x,incx,beta,y,incy);
    return new Result(cs);
}

CUBLAS_ROUTINE_HANDLER(Dsymv_v2){
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("Dsymv_v2"));
    
    cublasHandle_t handle;
    handle = (cublasHandle_t) in->Get<long long int>();
    cublasFillMode_t uplo = in->Get<cublasFillMode_t>();
    int n = in->Get<int>();
    const double * alpha = in->Assign<double>();
    double * A = in->GetFromMarshal<double*>();
    int lda = in->Get<int>();
    double * x = in->GetFromMarshal<double*>();
    int incx = in->Get<int>();
    
    const double * beta = in->Assign<double>();
    double * y = in->GetFromMarshal<double*>();
    int incy = in->Get<int>();
    
    cublasStatus_t cs = cublasDsymv_v2(handle,uplo,n,alpha,A,lda,x,incx,beta,y,incy);
    return new Result(cs);
}

CUBLAS_ROUTINE_HANDLER(Csymv_v2){
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("Csymv_v2"));
    
    cublasHandle_t handle;
    handle = (cublasHandle_t) in->Get<long long int>();
    cublasFillMode_t uplo = in->Get<cublasFillMode_t>();
    int n = in->Get<int>();
    const cuComplex * alpha = in->Assign<cuComplex>();
    cuComplex * A = in->GetFromMarshal<cuComplex*>();
    int lda = in->Get<int>();
    cuComplex * x = in->GetFromMarshal<cuComplex*>();
    int incx = in->Get<int>();
    
    const cuComplex * beta = in->Assign<cuComplex>();
    cuComplex * y = in->GetFromMarshal<cuComplex*>();
    int incy = in->Get<int>();
    
    cublasStatus_t cs = cublasCsymv_v2(handle,uplo,n,alpha,A,lda,x,incx,beta,y,incy);
    return new Result(cs);
}

CUBLAS_ROUTINE_HANDLER(Zsymv_v2){
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("Zsymv_v2"));
    
    cublasHandle_t handle;
    handle = (cublasHandle_t) in->Get<long long int>();
    cublasFillMode_t uplo = in->Get<cublasFillMode_t>();
    int n = in->Get<int>();
    const cuDoubleComplex * alpha = in->Assign<cuDoubleComplex>();
    cuDoubleComplex * A = in->GetFromMarshal<cuDoubleComplex*>();
    int lda = in->Get<int>();
    cuDoubleComplex * x = in->GetFromMarshal<cuDoubleComplex*>();
    int incx = in->Get<int>();
    
    const cuDoubleComplex * beta = in->Assign<cuDoubleComplex>();
    cuDoubleComplex * y = in->GetFromMarshal<cuDoubleComplex*>();
    int incy = in->Get<int>();
    
    cublasStatus_t cs = cublasZsymv_v2(handle,uplo,n,alpha,A,lda,x,incx,beta,y,incy);
    return new Result(cs);
}

CUBLAS_ROUTINE_HANDLER(Chemv_v2){
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("Chemv_v2"));
    
    cublasHandle_t handle;
    handle = (cublasHandle_t) in->Get<long long int>();
    cublasFillMode_t uplo = in->Get<cublasFillMode_t>();
    int n = in->Get<int>();
    const cuComplex * alpha = in->Assign<cuComplex>();
    cuComplex * A = in->GetFromMarshal<cuComplex*>();
    int lda = in->Get<int>();
    cuComplex * x = in->GetFromMarshal<cuComplex*>();
    int incx = in->Get<int>();
    
    const cuComplex * beta = in->Assign<cuComplex>();
    cuComplex * y = in->GetFromMarshal<cuComplex*>();
    int incy = in->Get<int>();
    
    cublasStatus_t cs = cublasChemv_v2(handle,uplo,n,alpha,A,lda,x,incx,beta,y,incy);
    return new Result(cs);
}

CUBLAS_ROUTINE_HANDLER(Zhemv_v2){
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("Zhemv_v2"));
    
    cublasHandle_t handle;
    handle = (cublasHandle_t) in->Get<long long int>();
    cublasFillMode_t uplo = in->Get<cublasFillMode_t>();
    int n = in->Get<int>();
    const cuDoubleComplex * alpha = in->Assign<cuDoubleComplex>();
    cuDoubleComplex * A = in->GetFromMarshal<cuDoubleComplex*>();
    int lda = in->Get<int>();
    cuDoubleComplex * x = in->GetFromMarshal<cuDoubleComplex*>();
    int incx = in->Get<int>();
    
    const cuDoubleComplex * beta = in->Assign<cuDoubleComplex>();
    cuDoubleComplex * y = in->GetFromMarshal<cuDoubleComplex*>();
    int incy = in->Get<int>();
    
    cublasStatus_t cs = cublasZhemv_v2(handle,uplo,n,alpha,A,lda,x,incx,beta,y,incy);
    return new Result(cs);
}

CUBLAS_ROUTINE_HANDLER(Ssbmv_v2){
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("Ssbmv_v2"));
    
    cublasHandle_t handle;
    handle = (cublasHandle_t) in->Get<long long int>();
    cublasFillMode_t uplo = in->Get<cublasFillMode_t>();
    int n = in->Get<int>();
    int k = in->Get<int>();
    const float * alpha = in->Assign<float>();
    float * A = in->GetFromMarshal<float*>();
    int lda = in->Get<int>();
    float * x = in->GetFromMarshal<float*>();
    int incx = in->Get<int>();
    
    const float * beta = in->Assign<float>();
    float * y = in->GetFromMarshal<float*>();
    int incy = in->Get<int>();
    
    cublasStatus cs = cublasSsbmv_v2(handle,uplo,n,k,alpha,A,lda,x,incx,beta,y,incy);
    return new Result(cs);
}

CUBLAS_ROUTINE_HANDLER(Dsbmv_v2){
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("Dsbmv_v2"));
    
    cublasHandle_t handle;
    handle = (cublasHandle_t) in->Get<long long int>();
    cublasFillMode_t uplo = in->Get<cublasFillMode_t>();
    int n = in->Get<int>();
    int k = in->Get<int>();
    const double * alpha = in->Assign<double>();
    double * A = in->GetFromMarshal<double*>();
    int lda = in->Get<int>();
    double * x = in->GetFromMarshal<double*>();
    int incx = in->Get<int>();
    
    const double * beta = in->Assign<double>();
    double * y = in->GetFromMarshal<double*>();
    int incy = in->Get<int>();
    
    cublasStatus cs = cublasDsbmv_v2(handle,uplo,n,k,alpha,A,lda,x,incx,beta,y,incy);
    return new Result(cs);
}

CUBLAS_ROUTINE_HANDLER(Chbmv_v2){
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("Chbmv_v2"));
    
    cublasHandle_t handle;
    handle = (cublasHandle_t) in->Get<long long int>();
    cublasFillMode_t uplo = in->Get<cublasFillMode_t>();
    int n = in->Get<int>();
    int k = in->Get<int>();
    const cuComplex * alpha = in->Assign<cuComplex>();
    cuComplex * A = in->GetFromMarshal<cuComplex*>();
    int lda = in->Get<int>();
    cuComplex * x = in->GetFromMarshal<cuComplex*>();
    int incx = in->Get<int>();
    
    const cuComplex * beta = in->Assign<cuComplex>();
    cuComplex * y = in->GetFromMarshal<cuComplex*>();
    int incy = in->Get<int>();
    
    cublasStatus cs = cublasChbmv_v2(handle,uplo,n,k,alpha,A,lda,x,incx,beta,y,incy);
    return new Result(cs);
}

CUBLAS_ROUTINE_HANDLER(Zhbmv_v2){
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("Zhbmv_v2"));
    
    cublasHandle_t handle;
    handle = (cublasHandle_t) in->Get<long long int>();
    cublasFillMode_t uplo = in->Get<cublasFillMode_t>();
    int n = in->Get<int>();
    int k = in->Get<int>();
    const cuDoubleComplex * alpha = in->Assign<cuDoubleComplex>();
    cuDoubleComplex * A = in->GetFromMarshal<cuDoubleComplex*>();
    int lda = in->Get<int>();
    cuDoubleComplex * x = in->GetFromMarshal<cuDoubleComplex*>();
    int incx = in->Get<int>();
    
    const cuDoubleComplex * beta = in->Assign<cuDoubleComplex>();
    cuDoubleComplex * y = in->GetFromMarshal<cuDoubleComplex*>();
    int incy = in->Get<int>();
    
    cublasStatus cs = cublasZhbmv_v2(handle,uplo,n,k,alpha,A,lda,x,incx,beta,y,incy);
    return new Result(cs);
}

CUBLAS_ROUTINE_HANDLER(Sspmv_v2){
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("Sspmv_v2"));
    
    cublasHandle_t handle;
    handle = (cublasHandle_t) in->Get<long long int>();
    cublasFillMode_t uplo = in->Get<cublasFillMode_t>();
    int n = in->Get<int>();
    const float * alpha = in->Assign<float>();
    float * AP = in->GetFromMarshal<float*>();
    float * x = in->GetFromMarshal<float*>();
    int incx = in->Get<int>();
    
    const float * beta = in->Assign<float>();
    float * y = in->GetFromMarshal<float*>();
    int incy = in->Get<int>();
    
    cublasStatus cs = cublasSspmv_v2(handle,uplo,n,alpha,AP,x,incx,beta,y,incy);
    return new Result(cs);
}

CUBLAS_ROUTINE_HANDLER(Dspmv_v2){
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("Dspmv_v2"));
    
    cublasHandle_t handle;
    handle = (cublasHandle_t) in->Get<long long int>();
    cublasFillMode_t uplo = in->Get<cublasFillMode_t>();
    int n = in->Get<int>();
    const double * alpha = in->Assign<double>();
    double * AP = in->GetFromMarshal<double*>();
    double * x = in->GetFromMarshal<double*>();
    int incx = in->Get<int>();
    
    const double * beta = in->Assign<double>();
    double * y = in->GetFromMarshal<double*>();
    int incy = in->Get<int>();
    
    cublasStatus cs = cublasDspmv_v2(handle,uplo,n,alpha,AP,x,incx,beta,y,incy);
    return new Result(cs);
}

CUBLAS_ROUTINE_HANDLER(Chpmv_v2){
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("Chpmv_v2"));
    
    cublasHandle_t handle;
    handle = (cublasHandle_t) in->Get<long long int>();
    cublasFillMode_t uplo = in->Get<cublasFillMode_t>();
    int n = in->Get<int>();
    const cuComplex * alpha = in->Assign<cuComplex>();
    cuComplex * AP = in->GetFromMarshal<cuComplex*>();
    cuComplex * x = in->GetFromMarshal<cuComplex*>();
    int incx = in->Get<int>();
    
    const cuComplex * beta = in->Assign<cuComplex>();
    cuComplex * y = in->GetFromMarshal<cuComplex*>();
    int incy = in->Get<int>();
    
    cublasStatus cs = cublasChpmv_v2(handle,uplo,n,alpha,AP,x,incx,beta,y,incy);
    return new Result(cs);
}

CUBLAS_ROUTINE_HANDLER(Zhpmv_v2){
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("Zhpmv_v2"));
    
    cublasHandle_t handle;
    handle = (cublasHandle_t) in->Get<long long int>();
    cublasFillMode_t uplo = in->Get<cublasFillMode_t>();
    int n = in->Get<int>();
    const cuDoubleComplex * alpha = in->Assign<cuDoubleComplex>();
    cuDoubleComplex * AP = in->GetFromMarshal<cuDoubleComplex*>();
    cuDoubleComplex * x = in->GetFromMarshal<cuDoubleComplex*>();
    int incx = in->Get<int>();
    
    const cuDoubleComplex * beta = in->Assign<cuDoubleComplex>();
    cuDoubleComplex * y = in->GetFromMarshal<cuDoubleComplex*>();
    int incy = in->Get<int>();
    
    cublasStatus cs = cublasZhpmv_v2(handle,uplo,n,alpha,AP,x,incx,beta,y,incy);
    return new Result(cs);
}


CUBLAS_ROUTINE_HANDLER(Sger_v2){
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("Sger_v2"));
    
    cublasHandle_t handle;
    handle = (cublasHandle_t) in->Get<long long int>();
    int m = in->Get<int>();
    int n = in->Get<int>();
    const float * alpha = in->Assign<float>();
    float * x = in->GetFromMarshal<float*>();
    int incx = in->Get<int>();
    float * y = in->GetFromMarshal<float*>();
    int incy = in->Get<int>();
    float * A = in->GetFromMarshal<float*>();
    int lda = in->Get<int>();

    cublasStatus_t cs = cublasSger_v2(handle,m,n,alpha,x,incx,y,incy,A,lda);
    return new Result(cs);
}

CUBLAS_ROUTINE_HANDLER(Dger_v2){
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("Dger_v2"));
    
    cublasHandle_t handle;
    handle = (cublasHandle_t) in->Get<long long int>();
    int m = in->Get<int>();
    int n = in->Get<int>();
    const double * alpha = in->Assign<double>();
    double * x = in->GetFromMarshal<double*>();
    int incx = in->Get<int>();
    double * y = in->GetFromMarshal<double*>();
    int incy = in->Get<int>();
    double * A = in->GetFromMarshal<double*>();
    int lda = in->Get<int>();

    cublasStatus_t cs = cublasDger_v2(handle,m,n,alpha,x,incx,y,incy,A,lda);
    return new Result(cs);
}

CUBLAS_ROUTINE_HANDLER(Cgeru_v2){
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("Cgeru_v2"));
    
    cublasHandle_t handle;
    handle = (cublasHandle_t) in->Get<long long int>();
    int m = in->Get<int>();
    int n = in->Get<int>();
    const cuComplex * alpha = in->Assign<cuComplex>();
    cuComplex * x = in->GetFromMarshal<cuComplex*>();
    int incx = in->Get<int>();
    cuComplex * y = in->GetFromMarshal<cuComplex*>();
    int incy = in->Get<int>();
    cuComplex * A = in->GetFromMarshal<cuComplex*>();
    int lda = in->Get<int>();

    cublasStatus_t cs = cublasCgeru_v2(handle,m,n,alpha,x,incx,y,incy,A,lda);
    return new Result(cs);
}

CUBLAS_ROUTINE_HANDLER(Cgerc_v2){
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("Cgerc_v2"));
    
    cublasHandle_t handle;
    handle = (cublasHandle_t) in->Get<long long int>();
    int m = in->Get<int>();
    int n = in->Get<int>();
    const cuComplex * alpha = in->Assign<cuComplex>();
    cuComplex * x = in->GetFromMarshal<cuComplex*>();
    int incx = in->Get<int>();
    cuComplex * y = in->GetFromMarshal<cuComplex*>();
    int incy = in->Get<int>();
    cuComplex * A = in->GetFromMarshal<cuComplex*>();
    int lda = in->Get<int>();

    cublasStatus_t cs = cublasCgerc_v2(handle,m,n,alpha,x,incx,y,incy,A,lda);
    return new Result(cs);
}

CUBLAS_ROUTINE_HANDLER(Zgeru_v2){
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("Zgeru_v2"));
    
    cublasHandle_t handle;
    handle = (cublasHandle_t) in->Get<long long int>();
    int m = in->Get<int>();
    int n = in->Get<int>();
    const cuDoubleComplex * alpha = in->Assign<cuDoubleComplex>();
    cuDoubleComplex * x = in->GetFromMarshal<cuDoubleComplex*>();
    int incx = in->Get<int>();
    cuDoubleComplex * y = in->GetFromMarshal<cuDoubleComplex*>();
    int incy = in->Get<int>();
    cuDoubleComplex * A = in->GetFromMarshal<cuDoubleComplex*>();
    int lda = in->Get<int>();

    cublasStatus_t cs = cublasZgeru_v2(handle,m,n,alpha,x,incx,y,incy,A,lda);
    return new Result(cs);
}

CUBLAS_ROUTINE_HANDLER(Zgerc_v2){
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("Zgerc_v2"));
    
    cublasHandle_t handle;
    handle = (cublasHandle_t) in->Get<long long int>();
    int m = in->Get<int>();
    int n = in->Get<int>();
    const cuDoubleComplex * alpha = in->Assign<cuDoubleComplex>();
    cuDoubleComplex * x = in->GetFromMarshal<cuDoubleComplex*>();
    int incx = in->Get<int>();
    cuDoubleComplex * y = in->GetFromMarshal<cuDoubleComplex*>();
    int incy = in->Get<int>();
    cuDoubleComplex * A = in->GetFromMarshal<cuDoubleComplex*>();
    int lda = in->Get<int>();

    cublasStatus_t cs = cublasZgerc_v2(handle,m,n,alpha,x,incx,y,incy,A,lda);
    return new Result(cs);
}


CUBLAS_ROUTINE_HANDLER(Ssyr_v2){
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("Ssyr_v2"));
    
    cublasHandle_t handle;
    handle = (cublasHandle_t) in->Get<long long int>();
    cublasFillMode_t uplo = in->Get<cublasFillMode_t>();
    int n = in->Get<int>();
    const float * alpha = in->Assign<float>();
    float * x = in->GetFromMarshal<float*>();
    int incx = in->Get<int>();
    float * A = in->GetFromMarshal<float*>();
    int lda = in->Get<int>();
    
    cublasStatus_t cs = cublasSsyr_v2(handle,uplo,n,alpha,x,incx,A,lda);
    return new Result(cs);
}

CUBLAS_ROUTINE_HANDLER(Dsyr_v2){
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("Dsyr_v2"));
    
    cublasHandle_t handle;
    handle = (cublasHandle_t) in->Get<long long int>();
    cublasFillMode_t uplo = in->Get<cublasFillMode_t>();
    int n = in->Get<int>();
    const double * alpha = in->Assign<double>();
    double * x = in->GetFromMarshal<double*>();
    int incx = in->Get<int>();
    double * A = in->GetFromMarshal<double*>();
    int lda = in->Get<int>();
    
    cublasStatus_t cs = cublasDsyr_v2(handle,uplo,n,alpha,x,incx,A,lda);
    return new Result(cs);
}

CUBLAS_ROUTINE_HANDLER(Csyr_v2){
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("Csyr_v2"));
    
    cublasHandle_t handle;
    handle = (cublasHandle_t) in->Get<long long int>();
    cublasFillMode_t uplo = in->Get<cublasFillMode_t>();
    int n = in->Get<int>();
    const cuComplex * alpha = in->Assign<cuComplex>();
    cuComplex * x = in->GetFromMarshal<cuComplex*>();
    int incx = in->Get<int>();
    cuComplex * A = in->GetFromMarshal<cuComplex*>();
    int lda = in->Get<int>();
    
    cublasStatus_t cs = cublasCsyr_v2(handle,uplo,n,alpha,x,incx,A,lda);
    return new Result(cs);
}

CUBLAS_ROUTINE_HANDLER(Zsyr_v2){
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("Zsyr_v2"));
    
    cublasHandle_t handle;
    handle = (cublasHandle_t) in->Get<long long int>();
    cublasFillMode_t uplo = in->Get<cublasFillMode_t>();
    int n = in->Get<int>();
    const cuDoubleComplex * alpha = in->Assign<cuDoubleComplex>();
    cuDoubleComplex * x = in->GetFromMarshal<cuDoubleComplex*>();
    int incx = in->Get<int>();
    cuDoubleComplex * A = in->GetFromMarshal<cuDoubleComplex*>();
    int lda = in->Get<int>();
    
    cublasStatus_t cs = cublasZsyr_v2(handle,uplo,n,alpha,x,incx,A,lda);
    return new Result(cs);
}

CUBLAS_ROUTINE_HANDLER(Cher_v2){
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("Cher_v2"));
    
    cublasHandle_t handle;
    handle = (cublasHandle_t) in->Get<long long int>();
    cublasFillMode_t uplo = in->Get<cublasFillMode_t>();
    int n = in->Get<int>();
    const float * alpha = in->Assign<float>();
    cuComplex * x = in->GetFromMarshal<cuComplex*>();
    int incx = in->Get<int>();
    cuComplex * A = in->GetFromMarshal<cuComplex*>();
    int lda = in->Get<int>();
    
    cublasStatus_t cs = cublasCher_v2(handle,uplo,n,alpha,x,incx,A,lda);
    return new Result(cs);
}

CUBLAS_ROUTINE_HANDLER(Zher_v2){
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("Zher_v2"));
    
    cublasHandle_t handle;
    handle = (cublasHandle_t) in->Get<long long int>();
    cublasFillMode_t uplo = in->Get<cublasFillMode_t>();
    int n = in->Get<int>();
    const double * alpha = in->Assign<double>();
    cuDoubleComplex * x = in->GetFromMarshal<cuDoubleComplex*>();
    int incx = in->Get<int>();
    cuDoubleComplex * A = in->GetFromMarshal<cuDoubleComplex*>();
    int lda = in->Get<int>();
    
    cublasStatus_t cs = cublasZher_v2(handle,uplo,n,alpha,x,incx,A,lda);
    return new Result(cs);
}


CUBLAS_ROUTINE_HANDLER(Sspr_v2){
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("Sspr_v2"));
    
    cublasHandle_t handle;
    handle = (cublasHandle_t) in->Get<long long int>();
    cublasFillMode_t uplo = in->Get<cublasFillMode_t>();
    int n = in->Get<int>();
    const float * alpha = in->Assign<float>();
    float * x = in->GetFromMarshal<float*>();
    int incx = in->Get<int>();
    float * AP = in->GetFromMarshal<float*>();
    
    cublasStatus_t cs = cublasSspr_v2(handle,uplo,n,alpha,x,incx,AP);
    return new Result(cs);
}


CUBLAS_ROUTINE_HANDLER(Dspr_v2){
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("Dspr_v2"));
    
    cublasHandle_t handle;
    handle = (cublasHandle_t) in->Get<long long int>();
    cublasFillMode_t uplo = in->Get<cublasFillMode_t>();
    int n = in->Get<int>();
    const double * alpha = in->Assign<double>();
    double * x = in->GetFromMarshal<double*>();
    int incx = in->Get<int>();
    double * AP = in->GetFromMarshal<double*>();
    
    cublasStatus_t cs = cublasDspr_v2(handle,uplo,n,alpha,x,incx,AP);
    return new Result(cs);
}

CUBLAS_ROUTINE_HANDLER(Chpr_v2){
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("Chpr_v2"));
    
    cublasHandle_t handle;
    handle = (cublasHandle_t) in->Get<long long int>();
    cublasFillMode_t uplo = in->Get<cublasFillMode_t>();
    int n = in->Get<int>();
    const float * alpha = in->Assign<float>();
    cuComplex * x = in->GetFromMarshal<cuComplex*>();
    int incx = in->Get<int>();
    cuComplex * AP = in->GetFromMarshal<cuComplex*>();
    
    cublasStatus_t cs = cublasChpr_v2(handle,uplo,n,alpha,x,incx,AP);
    return new Result(cs);
}

CUBLAS_ROUTINE_HANDLER(Zhpr_v2){
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("Zhpr_v2"));
    
    cublasHandle_t handle;
    handle = (cublasHandle_t) in->Get<long long int>();
    cublasFillMode_t uplo = in->Get<cublasFillMode_t>();
    int n = in->Get<int>();
    const double * alpha = in->Assign<double>();
    cuDoubleComplex * x = in->GetFromMarshal<cuDoubleComplex*>();
    int incx = in->Get<int>();
    cuDoubleComplex * AP = in->GetFromMarshal<cuDoubleComplex*>();
    
    cublasStatus_t cs = cublasZhpr_v2(handle,uplo,n,alpha,x,incx,AP);
    return new Result(cs);
}

CUBLAS_ROUTINE_HANDLER(Ssyr2_v2){
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("Ssyr2_v2"));
    
    cublasHandle_t handle;
    handle = (cublasHandle_t) in->Get<long long int>();
    cublasFillMode_t uplo = in->Get<cublasFillMode_t>();
    int n = in->Get<int>();
    const float * alpha = in->Assign<float>();
    float * x = in->GetFromMarshal<float*>();
    int incx = in->Get<int>();
    float * y = in->GetFromMarshal<float*>();
    int incy = in->Get<int>();
    float * A = in->GetFromMarshal<float*>();
    int lda = in->Get<int>();
    
    cublasStatus_t cs = cublasSsyr2_v2(handle,uplo,n,alpha,x,incx,y,incy,A,lda);
    return new Result(cs);
}

CUBLAS_ROUTINE_HANDLER(Dsyr2_v2){
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("Dsyr2_v2"));
    
    cublasHandle_t handle;
    handle = (cublasHandle_t) in->Get<long long int>();
    cublasFillMode_t uplo = in->Get<cublasFillMode_t>();
    int n = in->Get<int>();
    const double * alpha = in->Assign<double>();
    double * x = in->GetFromMarshal<double*>();
    int incx = in->Get<int>();
    double * y = in->GetFromMarshal<double*>();
    int incy = in->Get<int>();
    double * A = in->GetFromMarshal<double*>();
    int lda = in->Get<int>();
    
    cublasStatus_t cs = cublasDsyr2_v2(handle,uplo,n,alpha,x,incx,y,incy,A,lda);
    return new Result(cs);
}

CUBLAS_ROUTINE_HANDLER(Csyr2_v2){
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("Csyr2_v2"));
    
    cublasHandle_t handle;
    handle = (cublasHandle_t) in->Get<long long int>();
    cublasFillMode_t uplo = in->Get<cublasFillMode_t>();
    int n = in->Get<int>();
    const cuComplex * alpha = in->Assign<cuComplex>();
    cuComplex * x = in->GetFromMarshal<cuComplex*>();
    int incx = in->Get<int>();
    cuComplex * y = in->GetFromMarshal<cuComplex*>();
    int incy = in->Get<int>();
    cuComplex * A = in->GetFromMarshal<cuComplex*>();
    int lda = in->Get<int>();
    
    cublasStatus_t cs = cublasCsyr2_v2(handle,uplo,n,alpha,x,incx,y,incy,A,lda);
    return new Result(cs);
}

CUBLAS_ROUTINE_HANDLER(Zsyr2_v2){
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("Zsyr2_v2"));
    
    cublasHandle_t handle;
    handle = (cublasHandle_t) in->Get<long long int>();
    cublasFillMode_t uplo = in->Get<cublasFillMode_t>();
    int n = in->Get<int>();
    const cuDoubleComplex * alpha = in->Assign<cuDoubleComplex>();
    cuDoubleComplex * x = in->GetFromMarshal<cuDoubleComplex*>();
    int incx = in->Get<int>();
    cuDoubleComplex * y = in->GetFromMarshal<cuDoubleComplex*>();
    int incy = in->Get<int>();
    cuDoubleComplex * A = in->GetFromMarshal<cuDoubleComplex*>();
    int lda = in->Get<int>();
    
    cublasStatus_t cs = cublasZsyr2_v2(handle,uplo,n,alpha,x,incx,y,incy,A,lda);
    return new Result(cs);
}

CUBLAS_ROUTINE_HANDLER(Cher2_v2){
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("Cher2_v2"));
    
    cublasHandle_t handle;
    handle = (cublasHandle_t) in->Get<long long int>();
    cublasFillMode_t uplo = in->Get<cublasFillMode_t>();
    int n = in->Get<int>();
    const cuComplex * alpha = in->Assign<cuComplex>();
    cuComplex * x = in->GetFromMarshal<cuComplex*>();
    int incx = in->Get<int>();
    cuComplex * y = in->GetFromMarshal<cuComplex*>();
    int incy = in->Get<int>();
    cuComplex * A = in->GetFromMarshal<cuComplex*>();
    int lda = in->Get<int>();
    
    cublasStatus_t cs = cublasCher2_v2(handle,uplo,n,alpha,x,incx,y,incy,A,lda);
    return new Result(cs);
}

CUBLAS_ROUTINE_HANDLER(Zher2_v2){
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("Zsyr2_v2"));
    
    cublasHandle_t handle;
    handle = (cublasHandle_t) in->Get<long long int>();
    cublasFillMode_t uplo = in->Get<cublasFillMode_t>();
    int n = in->Get<int>();
    const cuDoubleComplex * alpha = in->Assign<cuDoubleComplex>();
    cuDoubleComplex * x = in->GetFromMarshal<cuDoubleComplex*>();
    int incx = in->Get<int>();
    cuDoubleComplex * y = in->GetFromMarshal<cuDoubleComplex*>();
    int incy = in->Get<int>();
    cuDoubleComplex * A = in->GetFromMarshal<cuDoubleComplex*>();
    int lda = in->Get<int>();
    
    cublasStatus_t cs = cublasZher2_v2(handle,uplo,n,alpha,x,incx,y,incy,A,lda);
    return new Result(cs);
}


CUBLAS_ROUTINE_HANDLER(Sspr2_v2){
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("Sspr2_v2"));
    
    cublasHandle_t handle;
    handle = (cublasHandle_t) in->Get<long long int>();
    cublasFillMode_t uplo = in->Get<cublasFillMode_t>();
    int n = in->Get<int>();
    const float * alpha = in->Assign<float>();
    float * x = in->GetFromMarshal<float*>();
    int incx = in->Get<int>();
    float * y = in->GetFromMarshal<float*>();
    int incy = in->Get<int>();
    float * AP = in->GetFromMarshal<float*>();
    
    cublasStatus_t cs = cublasSspr2_v2(handle,uplo,n,alpha,x,incx,y,incy,AP);
    return new Result(cs);
}

CUBLAS_ROUTINE_HANDLER(Dspr2_v2){
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("Sspr2_v2"));
    
    cublasHandle_t handle;
    handle = (cublasHandle_t) in->Get<long long int>();
    cublasFillMode_t uplo = in->Get<cublasFillMode_t>();
    int n = in->Get<int>();
    const double * alpha = in->Assign<double>();
    double * x = in->GetFromMarshal<double*>();
    int incx = in->Get<int>();
    double * y = in->GetFromMarshal<double*>();
    int incy = in->Get<int>();
    double * AP = in->GetFromMarshal<double*>();
    
    cublasStatus_t cs = cublasDspr2_v2(handle,uplo,n,alpha,x,incx,y,incy,AP);
    return new Result(cs);
}

CUBLAS_ROUTINE_HANDLER(Chpr2_v2){
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("Chpr2_v2"));
    
    cublasHandle_t handle;
    handle = (cublasHandle_t) in->Get<long long int>();
    cublasFillMode_t uplo = in->Get<cublasFillMode_t>();
    int n = in->Get<int>();
    const cuComplex * alpha = in->Assign<cuComplex>();
    cuComplex * x = in->GetFromMarshal<cuComplex*>();
    int incx = in->Get<int>();
    cuComplex * y = in->GetFromMarshal<cuComplex*>();
    int incy = in->Get<int>();
    cuComplex * AP = in->GetFromMarshal<cuComplex*>();
    
    cublasStatus_t cs = cublasChpr2_v2(handle,uplo,n,alpha,x,incx,y,incy,AP);
    return new Result(cs);
}

CUBLAS_ROUTINE_HANDLER(Zhpr2_v2){
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("Zhpr2_v2"));
    
    cublasHandle_t handle;
    handle = (cublasHandle_t) in->Get<long long int>();
    cublasFillMode_t uplo = in->Get<cublasFillMode_t>();
    int n = in->Get<int>();
    const cuDoubleComplex * alpha = in->Assign<cuDoubleComplex>();
    cuDoubleComplex * x = in->GetFromMarshal<cuDoubleComplex*>();
    int incx = in->Get<int>();
    cuDoubleComplex * y = in->GetFromMarshal<cuDoubleComplex*>();
    int incy = in->Get<int>();
    cuDoubleComplex * AP = in->GetFromMarshal<cuDoubleComplex*>();
    
    cublasStatus_t cs = cublasZhpr2_v2(handle,uplo,n,alpha,x,incx,y,incy,AP);
    return new Result(cs);
}