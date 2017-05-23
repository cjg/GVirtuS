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

CUBLAS_ROUTINE_HANDLER(Dgemm_v2) {
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("Dgemm"));
    
    cublasHandle_t handle;
    handle = (cublasHandle_t) in->Get<long long int>();
    
    cublasOperation_t transa = in->Get<cublasOperation_t>();
    cublasOperation_t transb = in->Get<cublasOperation_t>();
    int m  = in->Get<int>();
    int n  = in->Get<int>();
    int k  = in->Get<int>();
    const double * alpha = in->Assign<double>();
    
    double * A = in->GetFromMarshal<double*>();
    int lda = in->Get<int>();
    double * B = in->GetFromMarshal<double*>();
    int ldb = in->Get<int>();
    const double * beta = in->Assign<double>();
    double * C = in->GetFromMarshal<double*>();
    int ldc = in->Get<int>();
    cublasStatus_t cs;
    Buffer * out = new Buffer();
    try{
        cs = cublasDgemm(handle,transa,transb,m,n,k,alpha,A,lda,B,ldb,beta,C,ldc);
        out->AddMarshal<double *>(C);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return new Result(cudaErrorMemoryAllocation);
    }
    cout << "DEBUG - cublasDgemm_v2 Executed"<<endl;
    return new Result(cs,out);
}

CUBLAS_ROUTINE_HANDLER(Cgemm_v2) {
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("Cgemm"));
    
    cublasHandle_t handle;
    handle = (cublasHandle_t) in->Get<long long int>();
    
    cublasOperation_t transa = in->Get<cublasOperation_t>();
    cublasOperation_t transb = in->Get<cublasOperation_t>();
    int m  = in->Get<int>();
    int n  = in->Get<int>();
    int k  = in->Get<int>();
    const cuComplex * alpha = in->Assign<cuComplex>();
    
    cuComplex * A = in->GetFromMarshal<cuComplex*>();
    int lda = in->Get<int>();
    cuComplex * B = in->GetFromMarshal<cuComplex*>();
    int ldb = in->Get<int>();
    const cuComplex * beta = in->Assign<cuComplex>();
    cuComplex * C = in->GetFromMarshal<cuComplex*>();
    int ldc = in->Get<int>();
    cublasStatus_t cs;
    Buffer * out = new Buffer();
    try{
        cs = cublasCgemm(handle,transa,transb,m,n,k,alpha,A,lda,B,ldb,beta,C,ldc);
        out->AddMarshal<cuComplex *>(C);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return new Result(cudaErrorMemoryAllocation);
    }
    cout << "DEBUG - cublasCgemm_v2 Executed"<<endl;
    return new Result(cs,out);
}

CUBLAS_ROUTINE_HANDLER(Zgemm_v2) {
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("Zgemm"));
    
    cublasHandle_t handle;
    handle = (cublasHandle_t) in->Get<long long int>();
    
    cublasOperation_t transa = in->Get<cublasOperation_t>();
    cublasOperation_t transb = in->Get<cublasOperation_t>();
    int m  = in->Get<int>();
    int n  = in->Get<int>();
    int k  = in->Get<int>();
    const cuDoubleComplex * alpha = in->Assign<cuDoubleComplex>();
    
    cuDoubleComplex * A = in->GetFromMarshal<cuDoubleComplex*>();
    int lda = in->Get<int>();
    cuDoubleComplex * B = in->GetFromMarshal<cuDoubleComplex*>();
    int ldb = in->Get<int>();
    const cuDoubleComplex * beta = in->Assign<cuDoubleComplex>();
    cuDoubleComplex * C = in->GetFromMarshal<cuDoubleComplex*>();
    int ldc = in->Get<int>();
    cublasStatus_t cs;
    Buffer * out = new Buffer();
    try{
        cs = cublasZgemm(handle,transa,transb,m,n,k,alpha,A,lda,B,ldb,beta,C,ldc);
        out->AddMarshal<cuDoubleComplex *>(C);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return new Result(cudaErrorMemoryAllocation);
    }
    cout << "DEBUG - cublasZgemm_v2 Executed"<<endl;
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

CUBLAS_ROUTINE_HANDLER(Ssyrk_v2){
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("Ssyrk_v2"));
    
    cublasHandle_t handle;
    handle = (cublasHandle_t) in->Get<long long int>();
    cublasFillMode_t uplo = in->Get<cublasFillMode_t>();
    cublasOperation_t trans = in->Get<cublasOperation_t>();
    int n = in->Get<int>();
    int k = in->Get<int>();
    const float * alpha = in->Assign<float>();
    float * A = in->GetFromMarshal<float*>();
    int lda = in->Get<int>();
    const float * beta = in->Assign<float>();
    float * C = in->GetFromMarshal<float*>();
    int ldc = in->Get<int>();
    
    cublasStatus_t cs = cublasSsyrk_v2(handle,uplo,trans,n,k,alpha,A,lda,beta,C,ldc);
    return new Result(cs);
}

CUBLAS_ROUTINE_HANDLER(Dsyrk_v2){
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("Dsyrk_v2"));
    
    cublasHandle_t handle;
    handle = (cublasHandle_t) in->Get<long long int>();
    cublasFillMode_t uplo = in->Get<cublasFillMode_t>();
    cublasOperation_t trans = in->Get<cublasOperation_t>();
    int n = in->Get<int>();
    int k = in->Get<int>();
    const double * alpha = in->Assign<double>();
    double * A = in->GetFromMarshal<double*>();
    int lda = in->Get<int>();
    const double * beta = in->Assign<double>();
    double * C = in->GetFromMarshal<double*>();
    int ldc = in->Get<int>();
    
    cublasStatus_t cs = cublasDsyrk_v2(handle,uplo,trans,n,k,alpha,A,lda,beta,C,ldc);
    return new Result(cs);
}

CUBLAS_ROUTINE_HANDLER(Csyrk_v2){
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("Csyrk_v2"));
    
    cublasHandle_t handle;
    handle = (cublasHandle_t) in->Get<long long int>();
    cublasFillMode_t uplo = in->Get<cublasFillMode_t>();
    cublasOperation_t trans = in->Get<cublasOperation_t>();
    int n = in->Get<int>();
    int k = in->Get<int>();
    const cuComplex * alpha = in->Assign<cuComplex>();
    cuComplex * A = in->GetFromMarshal<cuComplex*>();
    int lda = in->Get<int>();
    const cuComplex * beta = in->Assign<cuComplex>();
    cuComplex * C = in->GetFromMarshal<cuComplex*>();
    int ldc = in->Get<int>();
    
    cublasStatus_t cs = cublasCsyrk_v2(handle,uplo,trans,n,k,alpha,A,lda,beta,C,ldc);
    return new Result(cs);
}

CUBLAS_ROUTINE_HANDLER(Zsyrk_v2){
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("Zsyrk_v2"));
    
    cublasHandle_t handle;
    handle = (cublasHandle_t) in->Get<long long int>();
    cublasFillMode_t uplo = in->Get<cublasFillMode_t>();
    cublasOperation_t trans = in->Get<cublasOperation_t>();
    int n = in->Get<int>();
    int k = in->Get<int>();
    const cuDoubleComplex * alpha = in->Assign<cuDoubleComplex>();
    cuDoubleComplex * A = in->GetFromMarshal<cuDoubleComplex*>();
    int lda = in->Get<int>();
    const cuDoubleComplex * beta = in->Assign<cuDoubleComplex>();
    cuDoubleComplex * C = in->GetFromMarshal<cuDoubleComplex*>();
    int ldc = in->Get<int>();
    
    cublasStatus_t cs = cublasZsyrk_v2(handle,uplo,trans,n,k,alpha,A,lda,beta,C,ldc);
    return new Result(cs);
}


CUBLAS_ROUTINE_HANDLER(Cherk_v2){
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("Cherk_v2"));
    
    cublasHandle_t handle;
    handle = (cublasHandle_t) in->Get<long long int>();
    cublasFillMode_t uplo = in->Get<cublasFillMode_t>();
    cublasOperation_t trans = in->Get<cublasOperation_t>();
    int n = in->Get<int>();
    int k = in->Get<int>();
    const float * alpha = in->Assign<float>();
    cuComplex * A = in->GetFromMarshal<cuComplex*>();
    int lda = in->Get<int>();
    const float * beta = in->Assign<float>();
    cuComplex * C = in->GetFromMarshal<cuComplex*>();
    int ldc = in->Get<int>();
    
    cublasStatus_t cs = cublasCherk_v2(handle,uplo,trans,n,k,alpha,A,lda,beta,C,ldc);
    return new Result(cs);
}

CUBLAS_ROUTINE_HANDLER(Zherk_v2){
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("Zherk_v2"));
    
    cublasHandle_t handle;
    handle = (cublasHandle_t) in->Get<long long int>();
    cublasFillMode_t uplo = in->Get<cublasFillMode_t>();
    cublasOperation_t trans = in->Get<cublasOperation_t>();
    int n = in->Get<int>();
    int k = in->Get<int>();
    const double * alpha = in->Assign<double>();
    cuDoubleComplex * A = in->GetFromMarshal<cuDoubleComplex*>();
    int lda = in->Get<int>();
    const double * beta = in->Assign<double>();
    cuDoubleComplex * C = in->GetFromMarshal<cuDoubleComplex*>();
    int ldc = in->Get<int>();
    
    cublasStatus_t cs = cublasZherk_v2(handle,uplo,trans,n,k,alpha,A,lda,beta,C,ldc);
    return new Result(cs);
}