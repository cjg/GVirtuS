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