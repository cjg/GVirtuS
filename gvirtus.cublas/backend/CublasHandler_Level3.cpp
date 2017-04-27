#include "CublasHandler.h"
#include <iostream>
#include <cstdio>
#include <string>

using namespace std;
using namespace log4cplus;

CUBLAS_ROUTINE_HANDLER(Sgemm_v2) {
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("Sgemm"));
    
    cublasHandle_t handle;
    handle = (cublasHandle_t) in->Get<long long int>();
    
    cublasOperation_t transa = in->Get<cublasOperation_t>();
    cublasOperation_t transb = in->Get<cublasOperation_t>();
    int m  = in->Get<int>();
    int n  = in->Get<int>();
    int k  = in->Get<int>();
    const float * alpha = in->Assign<float>();
    
    float * A = in->GetFromMarshal<float*>();
    int lda = in->Get<int>();
    float * B = in->GetFromMarshal<float*>();
    int ldb = in->Get<int>();
    const float * beta = in->Assign<float>();
    float * C = in->GetFromMarshal<float*>();
    int ldc = in->Get<int>();
    cublasStatus_t cs;
    Buffer * out = new Buffer();
    try{
        cs = cublasSgemm(handle,transa,transb,m,n,k,alpha,A,lda,B,ldb,beta,C,ldc);
        out->AddMarshal<float *>(C);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return new Result(cudaErrorMemoryAllocation);
    }
    cout << "DEBUG - cublasSgemm_v2 Executed"<<endl;
    return new Result(cs,out);
}


CUBLAS_ROUTINE_HANDLER(Snrm2_v2){
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("Sgemm"));
    
    cublasHandle_t handle;
    handle = (cublasHandle_t) in->Get<long long int>();
    int n = in->Get<int>();
    float * x = in->GetFromMarshal<float*>();
    int incx = in->Get<int>();
    float * result;
    
    cublasStatus_t cs = cublasSnrm2_v2(handle,n,x,incx,result);
    Buffer * out = new Buffer();
    try{
        out->Add<float>(*result);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return new Result(cudaErrorMemoryAllocation);
    }
    cout << "DEBUG - cublasSnrm2_v2 Executed"<<endl;
    return new Result(cs,out);
}