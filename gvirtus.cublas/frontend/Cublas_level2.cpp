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

#include <iostream>
#include <cstdio>
#include <string>

#include "Frontend.h"
#include "CublasFrontend.h"

using namespace std;


extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSgemv_v2(cublasHandle_t handle, cublasOperation_t trans, int m, int n, const float *alpha, const float *A, int lda, const float *x, int incx, const float *beta, float *y, int incy){
    
    CublasFrontend::Prepare();
    CublasFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CublasFrontend::AddVariableForArguments<cublasOperation_t>(trans);
    CublasFrontend::AddVariableForArguments<int>(m);
    CublasFrontend::AddVariableForArguments<int>(n);
    float * _alpha = const_cast<float *>(alpha);
    CublasFrontend::AddHostPointerForArguments(_alpha);
    CublasFrontend::AddDevicePointerForArguments(A);
    CublasFrontend::AddVariableForArguments<int>(lda);
    CublasFrontend::AddDevicePointerForArguments(x);
    CublasFrontend::AddVariableForArguments<int>(incx);
    
    float * _beta = const_cast<float *>(beta);
    CublasFrontend::AddHostPointerForArguments(_beta);
    CublasFrontend::AddDevicePointerForArguments(y);
    CublasFrontend::AddVariableForArguments<int>(incy);
    CublasFrontend::Execute("cublasSgemv_v2");
    if (CublasFrontend::Success())
        y = (float *)CublasFrontend::GetOutputDevicePointer();
    return CublasFrontend::GetExitCode();
}

extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDgemv_v2(cublasHandle_t handle, cublasOperation_t trans, int m, int n, const double *alpha, const double *A, int lda, const double *x, int incx, const double *beta, double *y, int incy){
    
    CublasFrontend::Prepare();
    CublasFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CublasFrontend::AddVariableForArguments<cublasOperation_t>(trans);
    CublasFrontend::AddVariableForArguments<int>(m);
    CublasFrontend::AddVariableForArguments<int>(n);
    double * _alpha = const_cast<double *>(alpha);
    CublasFrontend::AddHostPointerForArguments(_alpha);
    CublasFrontend::AddDevicePointerForArguments(A);
    CublasFrontend::AddVariableForArguments<int>(lda);
    CublasFrontend::AddDevicePointerForArguments(x);
    CublasFrontend::AddVariableForArguments<int>(incx);
    
    double * _beta = const_cast<double *>(beta);
    CublasFrontend::AddHostPointerForArguments(_beta);
    CublasFrontend::AddDevicePointerForArguments(y);
    CublasFrontend::AddVariableForArguments<int>(incy);
    CublasFrontend::Execute("cublasDgemv_v2");
    if (CublasFrontend::Success())
        y = (double *)CublasFrontend::GetOutputDevicePointer();
    return CublasFrontend::GetExitCode();
}

extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCgemv_v2(cublasHandle_t handle, cublasOperation_t trans, int m, int n, const cuComplex *alpha, const cuComplex *A, int lda, const cuComplex *x, int incx, const cuComplex *beta, cuComplex *y, int incy){
    
    CublasFrontend::Prepare();
    CublasFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CublasFrontend::AddVariableForArguments<cublasOperation_t>(trans);
    CublasFrontend::AddVariableForArguments<int>(m);
    CublasFrontend::AddVariableForArguments<int>(n);
    cuComplex * _alpha = const_cast<cuComplex *>(alpha);
    CublasFrontend::AddHostPointerForArguments(_alpha);
    CublasFrontend::AddDevicePointerForArguments(A);
    CublasFrontend::AddVariableForArguments<int>(lda);
    CublasFrontend::AddDevicePointerForArguments(x);
    CublasFrontend::AddVariableForArguments<int>(incx);
    
    cuComplex * _beta = const_cast<cuComplex *>(beta);
    CublasFrontend::AddHostPointerForArguments(_beta);
    CublasFrontend::AddDevicePointerForArguments(y);
    CublasFrontend::AddVariableForArguments<int>(incy);
    CublasFrontend::Execute("cublasCgemv_v2");
    if (CublasFrontend::Success())
        y = (cuComplex *)CublasFrontend::GetOutputDevicePointer();
    return CublasFrontend::GetExitCode();
}

extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZgemv_v2(cublasHandle_t handle, cublasOperation_t trans, int m, int n, const cuDoubleComplex *alpha, const cuDoubleComplex *A, int lda, const cuDoubleComplex *x, int incx, const cuDoubleComplex *beta, cuDoubleComplex *y, int incy){
    
    CublasFrontend::Prepare();
    CublasFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CublasFrontend::AddVariableForArguments<cublasOperation_t>(trans);
    CublasFrontend::AddVariableForArguments<int>(m);
    CublasFrontend::AddVariableForArguments<int>(n);
    cuDoubleComplex * _alpha = const_cast<cuDoubleComplex *>(alpha);
    CublasFrontend::AddHostPointerForArguments(_alpha);
    CublasFrontend::AddDevicePointerForArguments(A);
    CublasFrontend::AddVariableForArguments<int>(lda);
    CublasFrontend::AddDevicePointerForArguments(x);
    CublasFrontend::AddVariableForArguments<int>(incx);
    
    cuDoubleComplex * _beta = const_cast<cuDoubleComplex *>(beta);
    CublasFrontend::AddHostPointerForArguments(_beta);
    CublasFrontend::AddDevicePointerForArguments(y);
    CublasFrontend::AddVariableForArguments<int>(incy);
    CublasFrontend::Execute("cublasZgemv_v2");
    if (CublasFrontend::Success())
        y = (cuDoubleComplex *)CublasFrontend::GetOutputDevicePointer();
    return CublasFrontend::GetExitCode();
}

extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSgbmv_v2(cublasHandle_t handle, cublasOperation_t trans, int m, int n, int kl, int ku, const float *alpha, const float *A, int lda, const float *x, int incx, const float *beta, float *y, int incy){
    CublasFrontend::Prepare();
    
    CublasFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CublasFrontend::AddVariableForArguments<cublasOperation_t>(trans);
    CublasFrontend::AddVariableForArguments<int>(m);
    CublasFrontend::AddVariableForArguments<int>(n);
    CublasFrontend::AddVariableForArguments<int>(kl);
    CublasFrontend::AddVariableForArguments<int>(ku);
    float * _alpha = const_cast<float *>(alpha);
    CublasFrontend::AddHostPointerForArguments(_alpha);
    CublasFrontend::AddDevicePointerForArguments(A);
    CublasFrontend::AddVariableForArguments<int>(lda);
    CublasFrontend::AddDevicePointerForArguments(x);
    CublasFrontend::AddVariableForArguments<int>(incx);
    float * _beta = const_cast<float *>(beta);
    CublasFrontend::AddHostPointerForArguments(_beta);
    CublasFrontend::AddDevicePointerForArguments(y);
    CublasFrontend::AddVariableForArguments<int>(incy);
    
    CublasFrontend::Execute("cublasSgbmv_v2");
    return CublasFrontend::GetExitCode();
}

extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDgbmv_v2(cublasHandle_t handle, cublasOperation_t trans, int m, int n, int kl, int ku, const double *alpha, const double *A, int lda, const double *x, int incx, const double *beta, double *y, int incy){
    CublasFrontend::Prepare();
    
    CublasFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CublasFrontend::AddVariableForArguments<cublasOperation_t>(trans);
    CublasFrontend::AddVariableForArguments<int>(m);
    CublasFrontend::AddVariableForArguments<int>(n);
    CublasFrontend::AddVariableForArguments<int>(kl);
    CublasFrontend::AddVariableForArguments<int>(ku);
    double * _alpha = const_cast<double *>(alpha);
    CublasFrontend::AddHostPointerForArguments(_alpha);
    CublasFrontend::AddDevicePointerForArguments(A);
    CublasFrontend::AddVariableForArguments<int>(lda);
    CublasFrontend::AddDevicePointerForArguments(x);
    CublasFrontend::AddVariableForArguments<int>(incx);
    double * _beta = const_cast<double *>(beta);
    CublasFrontend::AddHostPointerForArguments(_beta);
    CublasFrontend::AddDevicePointerForArguments(y);
    CublasFrontend::AddVariableForArguments<int>(incy);
    
    CublasFrontend::Execute("cublasDgbmv_v2");
    return CublasFrontend::GetExitCode();
}

extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCgbmv_v2(cublasHandle_t handle, cublasOperation_t trans, int m, int n, int kl, int ku, const cuComplex *alpha, const cuComplex *A, int lda, const cuComplex *x, int incx, const cuComplex *beta, cuComplex *y, int incy){
    CublasFrontend::Prepare();
    
    CublasFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CublasFrontend::AddVariableForArguments<cublasOperation_t>(trans);
    CublasFrontend::AddVariableForArguments<int>(m);
    CublasFrontend::AddVariableForArguments<int>(n);
    CublasFrontend::AddVariableForArguments<int>(kl);
    CublasFrontend::AddVariableForArguments<int>(ku);
    cuComplex * _alpha = const_cast<cuComplex *>(alpha);
    CublasFrontend::AddHostPointerForArguments(_alpha);
    CublasFrontend::AddDevicePointerForArguments(A);
    CublasFrontend::AddVariableForArguments<int>(lda);
    CublasFrontend::AddDevicePointerForArguments(x);
    CublasFrontend::AddVariableForArguments<int>(incx);
    cuComplex * _beta = const_cast<cuComplex *>(beta);
    CublasFrontend::AddHostPointerForArguments(_beta);
    CublasFrontend::AddDevicePointerForArguments(y);
    CublasFrontend::AddVariableForArguments<int>(incy);
    
    CublasFrontend::Execute("cublasCgbmv_v2");
    return CublasFrontend::GetExitCode();
}

extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZgbmv_v2(cublasHandle_t handle, cublasOperation_t trans, int m, int n, int kl, int ku, const cuDoubleComplex *alpha, const cuDoubleComplex *A, int lda, const cuDoubleComplex *x, int incx, const cuDoubleComplex *beta, cuDoubleComplex *y, int incy){
    CublasFrontend::Prepare();
    
    CublasFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CublasFrontend::AddVariableForArguments<cublasOperation_t>(trans);
    CublasFrontend::AddVariableForArguments<int>(m);
    CublasFrontend::AddVariableForArguments<int>(n);
    CublasFrontend::AddVariableForArguments<int>(kl);
    CublasFrontend::AddVariableForArguments<int>(ku);
    cuDoubleComplex * _alpha = const_cast<cuDoubleComplex *>(alpha);
    CublasFrontend::AddHostPointerForArguments(_alpha);
    CublasFrontend::AddDevicePointerForArguments(A);
    CublasFrontend::AddVariableForArguments<int>(lda);
    CublasFrontend::AddDevicePointerForArguments(x);
    CublasFrontend::AddVariableForArguments<int>(incx);
    cuDoubleComplex * _beta = const_cast<cuDoubleComplex *>(beta);
    CublasFrontend::AddHostPointerForArguments(_beta);
    CublasFrontend::AddDevicePointerForArguments(y);
    CublasFrontend::AddVariableForArguments<int>(incy);
    
    CublasFrontend::Execute("cublasZgbmv_v2");
    return CublasFrontend::GetExitCode();
}

extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasStrmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const float *A, int lda, float *x, int incx) {
    CublasFrontend::Prepare();
    
    CublasFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CublasFrontend::AddVariableForArguments<cublasFillMode_t>(uplo);
    CublasFrontend::AddVariableForArguments<cublasOperation_t>(trans);
    CublasFrontend::AddVariableForArguments<cublasDiagType_t>(diag);
    CublasFrontend::AddVariableForArguments<int>(n);
    CublasFrontend::AddDevicePointerForArguments(A);
    CublasFrontend::AddVariableForArguments<int>(lda);
    CublasFrontend::AddDevicePointerForArguments(x);
    CublasFrontend::AddVariableForArguments<int>(incx);
    
    CublasFrontend::Execute("cublasStrmv_v2");
    return CublasFrontend::GetExitCode();
}

extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDtrmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const double *A, int lda, double *x, int incx) {
    CublasFrontend::Prepare();
    
    CublasFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CublasFrontend::AddVariableForArguments<cublasFillMode_t>(uplo);
    CublasFrontend::AddVariableForArguments<cublasOperation_t>(trans);
    CublasFrontend::AddVariableForArguments<cublasDiagType_t>(diag);
    CublasFrontend::AddVariableForArguments<int>(n);
    CublasFrontend::AddDevicePointerForArguments(A);
    CublasFrontend::AddVariableForArguments<int>(lda);
    CublasFrontend::AddDevicePointerForArguments(x);
    CublasFrontend::AddVariableForArguments<int>(incx);
    
    CublasFrontend::Execute("cublasDtrmv_v2");
    return CublasFrontend::GetExitCode();
}

extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCtrmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const cuComplex *A, int lda, cuComplex *x, int incx) {
    CublasFrontend::Prepare();
    
    CublasFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CublasFrontend::AddVariableForArguments<cublasFillMode_t>(uplo);
    CublasFrontend::AddVariableForArguments<cublasOperation_t>(trans);
    CublasFrontend::AddVariableForArguments<cublasDiagType_t>(diag);
    CublasFrontend::AddVariableForArguments<int>(n);
    CublasFrontend::AddDevicePointerForArguments(A);
    CublasFrontend::AddVariableForArguments<int>(lda);
    CublasFrontend::AddDevicePointerForArguments(x);
    CublasFrontend::AddVariableForArguments<int>(incx);
    
    CublasFrontend::Execute("cublasCtrmv_v2");
    return CublasFrontend::GetExitCode();
}

extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZtrmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const cuDoubleComplex *A, int lda, cuDoubleComplex *x, int incx) {
    CublasFrontend::Prepare();
    
    CublasFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CublasFrontend::AddVariableForArguments<cublasFillMode_t>(uplo);
    CublasFrontend::AddVariableForArguments<cublasOperation_t>(trans);
    CublasFrontend::AddVariableForArguments<cublasDiagType_t>(diag);
    CublasFrontend::AddVariableForArguments<int>(n);
    CublasFrontend::AddDevicePointerForArguments(A);
    CublasFrontend::AddVariableForArguments<int>(lda);
    CublasFrontend::AddDevicePointerForArguments(x);
    CublasFrontend::AddVariableForArguments<int>(incx);
    
    CublasFrontend::Execute("cublasZtrmv_v2");
    return CublasFrontend::GetExitCode();
}

extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasStbmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, int k, const float *A, int lda, float *x, int incx){
    CublasFrontend::Prepare();
    
    CublasFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CublasFrontend::AddVariableForArguments<cublasFillMode_t>(uplo);
    CublasFrontend::AddVariableForArguments<cublasOperation_t>(trans);
    CublasFrontend::AddVariableForArguments<cublasDiagType_t>(diag);
    CublasFrontend::AddVariableForArguments<int>(n);
    CublasFrontend::AddVariableForArguments<int>(k);
    CublasFrontend::AddDevicePointerForArguments(A);
    CublasFrontend::AddVariableForArguments<int>(lda);
    CublasFrontend::AddDevicePointerForArguments(x);
    CublasFrontend::AddVariableForArguments<int>(incx);
    
    CublasFrontend::Execute("cublasStbmv_v2");
    return CublasFrontend::GetExitCode();
}

extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDtbmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, int k, const double *A, int lda, double *x, int incx){
    CublasFrontend::Prepare();
    
    CublasFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CublasFrontend::AddVariableForArguments<cublasFillMode_t>(uplo);
    CublasFrontend::AddVariableForArguments<cublasOperation_t>(trans);
    CublasFrontend::AddVariableForArguments<cublasDiagType_t>(diag);
    CublasFrontend::AddVariableForArguments<int>(n);
    CublasFrontend::AddVariableForArguments<int>(k);
    CublasFrontend::AddDevicePointerForArguments(A);
    CublasFrontend::AddVariableForArguments<int>(lda);
    CublasFrontend::AddDevicePointerForArguments(x);
    CublasFrontend::AddVariableForArguments<int>(incx);
    
    CublasFrontend::Execute("cublasDtbmv_v2");
    return CublasFrontend::GetExitCode();
}

extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCtbmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, int k, const cuComplex *A, int lda, cuComplex *x, int incx){
    CublasFrontend::Prepare();
    
    CublasFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CublasFrontend::AddVariableForArguments<cublasFillMode_t>(uplo);
    CublasFrontend::AddVariableForArguments<cublasOperation_t>(trans);
    CublasFrontend::AddVariableForArguments<cublasDiagType_t>(diag);
    CublasFrontend::AddVariableForArguments<int>(n);
    CublasFrontend::AddVariableForArguments<int>(k);
    CublasFrontend::AddDevicePointerForArguments(A);
    CublasFrontend::AddVariableForArguments<int>(lda);
    CublasFrontend::AddDevicePointerForArguments(x);
    CublasFrontend::AddVariableForArguments<int>(incx);
    
    CublasFrontend::Execute("cublasCtbmv_v2");
    return CublasFrontend::GetExitCode();
}

extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZtbmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, int k, const cuDoubleComplex *A, int lda, cuDoubleComplex *x, int incx){
    CublasFrontend::Prepare();
    
    CublasFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CublasFrontend::AddVariableForArguments<cublasFillMode_t>(uplo);
    CublasFrontend::AddVariableForArguments<cublasOperation_t>(trans);
    CublasFrontend::AddVariableForArguments<cublasDiagType_t>(diag);
    CublasFrontend::AddVariableForArguments<int>(n);
    CublasFrontend::AddVariableForArguments<int>(k);
    CublasFrontend::AddDevicePointerForArguments(A);
    CublasFrontend::AddVariableForArguments<int>(lda);
    CublasFrontend::AddDevicePointerForArguments(x);
    CublasFrontend::AddVariableForArguments<int>(incx);
    
    CublasFrontend::Execute("cublasZtbmv_v2");
    return CublasFrontend::GetExitCode();
}

extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasStpmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const float *AP, float *x, int incx){
    CublasFrontend::Prepare();
    
    CublasFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CublasFrontend::AddVariableForArguments<cublasFillMode_t>(uplo);
    CublasFrontend::AddVariableForArguments<cublasOperation_t>(trans);
    CublasFrontend::AddVariableForArguments<cublasDiagType_t>(diag);
    CublasFrontend::AddVariableForArguments<int>(n);
    CublasFrontend::AddDevicePointerForArguments(AP);
    CublasFrontend::AddDevicePointerForArguments(x);
    CublasFrontend::AddVariableForArguments<int>(incx);
    
    CublasFrontend::Execute("cublasStpmv_v2");
    return CublasFrontend::GetExitCode();
}

extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDtpmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const double *AP, double *x, int incx){
    CublasFrontend::Prepare();
    
    CublasFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CublasFrontend::AddVariableForArguments<cublasFillMode_t>(uplo);
    CublasFrontend::AddVariableForArguments<cublasOperation_t>(trans);
    CublasFrontend::AddVariableForArguments<cublasDiagType_t>(diag);
    CublasFrontend::AddVariableForArguments<int>(n);
    CublasFrontend::AddDevicePointerForArguments(AP);
    CublasFrontend::AddDevicePointerForArguments(x);
    CublasFrontend::AddVariableForArguments<int>(incx);
    
    CublasFrontend::Execute("cublasDtpmv_v2");
    return CublasFrontend::GetExitCode();
}

extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCtpmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const cuComplex *AP, cuComplex *x, int incx){
    CublasFrontend::Prepare();
    
    CublasFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CublasFrontend::AddVariableForArguments<cublasFillMode_t>(uplo);
    CublasFrontend::AddVariableForArguments<cublasOperation_t>(trans);
    CublasFrontend::AddVariableForArguments<cublasDiagType_t>(diag);
    CublasFrontend::AddVariableForArguments<int>(n);
    CublasFrontend::AddDevicePointerForArguments(AP);
    CublasFrontend::AddDevicePointerForArguments(x);
    CublasFrontend::AddVariableForArguments<int>(incx);
    
    CublasFrontend::Execute("cublasCtpmv_v2");
    return CublasFrontend::GetExitCode();
}

extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZtpmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const cuDoubleComplex *AP, cuDoubleComplex *x, int incx){
    CublasFrontend::Prepare();
    
    CublasFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CublasFrontend::AddVariableForArguments<cublasFillMode_t>(uplo);
    CublasFrontend::AddVariableForArguments<cublasOperation_t>(trans);
    CublasFrontend::AddVariableForArguments<cublasDiagType_t>(diag);
    CublasFrontend::AddVariableForArguments<int>(n);
    CublasFrontend::AddDevicePointerForArguments(AP);
    CublasFrontend::AddDevicePointerForArguments(x);
    CublasFrontend::AddVariableForArguments<int>(incx);
    
    CublasFrontend::Execute("cublasZtpmv_v2");
    return CublasFrontend::GetExitCode();
}

extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasStrsv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const float *A, int lda, float *x, int incx){
    CublasFrontend::Prepare();
    
    CublasFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CublasFrontend::AddVariableForArguments<cublasFillMode_t>(uplo);
    CublasFrontend::AddVariableForArguments<cublasOperation_t>(trans);
    CublasFrontend::AddVariableForArguments<cublasDiagType_t>(diag);
    CublasFrontend::AddVariableForArguments<int>(n);
    CublasFrontend::AddDevicePointerForArguments(A);
    CublasFrontend::AddVariableForArguments<int>(lda);
    CublasFrontend::AddDevicePointerForArguments(x);
    CublasFrontend::AddVariableForArguments<int>(incx);
    
    CublasFrontend::Execute("cublasStrsv_v2");
    return CublasFrontend::GetExitCode();
}

extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDtrsv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const double *A, int lda, double *x, int incx){
    CublasFrontend::Prepare();
    
    CublasFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CublasFrontend::AddVariableForArguments<cublasFillMode_t>(uplo);
    CublasFrontend::AddVariableForArguments<cublasOperation_t>(trans);
    CublasFrontend::AddVariableForArguments<cublasDiagType_t>(diag);
    CublasFrontend::AddVariableForArguments<int>(n);
    CublasFrontend::AddDevicePointerForArguments(A);
    CublasFrontend::AddVariableForArguments<int>(lda);
    CublasFrontend::AddDevicePointerForArguments(x);
    CublasFrontend::AddVariableForArguments<int>(incx);
    
    CublasFrontend::Execute("cublasDtrsv_v2");
    return CublasFrontend::GetExitCode();
}

extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCtrsv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const cuComplex *A, int lda, cuComplex *x, int incx){
    CublasFrontend::Prepare();
    
    CublasFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CublasFrontend::AddVariableForArguments<cublasFillMode_t>(uplo);
    CublasFrontend::AddVariableForArguments<cublasOperation_t>(trans);
    CublasFrontend::AddVariableForArguments<cublasDiagType_t>(diag);
    CublasFrontend::AddVariableForArguments<int>(n);
    CublasFrontend::AddDevicePointerForArguments(A);
    CublasFrontend::AddVariableForArguments<int>(lda);
    CublasFrontend::AddDevicePointerForArguments(x);
    CublasFrontend::AddVariableForArguments<int>(incx);
    
    CublasFrontend::Execute("cublasCtrsv_v2");
    return CublasFrontend::GetExitCode();
}

extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZtrsv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const cuDoubleComplex *A, int lda, cuDoubleComplex *x, int incx){
    CublasFrontend::Prepare();
    
    CublasFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CublasFrontend::AddVariableForArguments<cublasFillMode_t>(uplo);
    CublasFrontend::AddVariableForArguments<cublasOperation_t>(trans);
    CublasFrontend::AddVariableForArguments<cublasDiagType_t>(diag);
    CublasFrontend::AddVariableForArguments<int>(n);
    CublasFrontend::AddDevicePointerForArguments(A);
    CublasFrontend::AddVariableForArguments<int>(lda);
    CublasFrontend::AddDevicePointerForArguments(x);
    CublasFrontend::AddVariableForArguments<int>(incx);
    
    CublasFrontend::Execute("cublasZtrsv_v2");
    return CublasFrontend::GetExitCode();
}

extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasStpsv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const float *AP, float *x, int incx){
    CublasFrontend::Prepare();
    
    CublasFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CublasFrontend::AddVariableForArguments<cublasFillMode_t>(uplo);
    CublasFrontend::AddVariableForArguments<cublasOperation_t>(trans);
    CublasFrontend::AddVariableForArguments<cublasDiagType_t>(diag);
    CublasFrontend::AddVariableForArguments<int>(n);
    CublasFrontend::AddDevicePointerForArguments(AP);
    CublasFrontend::AddDevicePointerForArguments(x);
    CublasFrontend::AddVariableForArguments<int>(incx);
    
    CublasFrontend::Execute("cublasStpsv_v2");
    return CublasFrontend::GetExitCode();
}

extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDtpsv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const double *AP, double *x, int incx){
    CublasFrontend::Prepare();
    
    CublasFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CublasFrontend::AddVariableForArguments<cublasFillMode_t>(uplo);
    CublasFrontend::AddVariableForArguments<cublasOperation_t>(trans);
    CublasFrontend::AddVariableForArguments<cublasDiagType_t>(diag);
    CublasFrontend::AddVariableForArguments<int>(n);
    CublasFrontend::AddDevicePointerForArguments(AP);
    CublasFrontend::AddDevicePointerForArguments(x);
    CublasFrontend::AddVariableForArguments<int>(incx);
    
    CublasFrontend::Execute("cublasDtpsv_v2");
    return CublasFrontend::GetExitCode();
}

extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCtpsv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const cuComplex *AP, cuComplex *x, int incx){
    CublasFrontend::Prepare();
    
    CublasFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CublasFrontend::AddVariableForArguments<cublasFillMode_t>(uplo);
    CublasFrontend::AddVariableForArguments<cublasOperation_t>(trans);
    CublasFrontend::AddVariableForArguments<cublasDiagType_t>(diag);
    CublasFrontend::AddVariableForArguments<int>(n);
    CublasFrontend::AddDevicePointerForArguments(AP);
    CublasFrontend::AddDevicePointerForArguments(x);
    CublasFrontend::AddVariableForArguments<int>(incx);
    
    CublasFrontend::Execute("cublasCtpsv_v2");
    return CublasFrontend::GetExitCode();
}

extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZtpsv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, const cuDoubleComplex *AP, cuDoubleComplex *x, int incx){
    CublasFrontend::Prepare();
    
    CublasFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CublasFrontend::AddVariableForArguments<cublasFillMode_t>(uplo);
    CublasFrontend::AddVariableForArguments<cublasOperation_t>(trans);
    CublasFrontend::AddVariableForArguments<cublasDiagType_t>(diag);
    CublasFrontend::AddVariableForArguments<int>(n);
    CublasFrontend::AddDevicePointerForArguments(AP);
    CublasFrontend::AddDevicePointerForArguments(x);
    CublasFrontend::AddVariableForArguments<int>(incx);
    
    CublasFrontend::Execute("cublasZtpsv_v2");
    return CublasFrontend::GetExitCode();
}

extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasStbsv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, int k, const float *A, int lda, float *x, int incx){
    CublasFrontend::Prepare();
    
    CublasFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CublasFrontend::AddVariableForArguments<cublasFillMode_t>(uplo);
    CublasFrontend::AddVariableForArguments<cublasOperation_t>(trans);
    CublasFrontend::AddVariableForArguments<cublasDiagType_t>(diag);
    CublasFrontend::AddVariableForArguments<int>(n);
    CublasFrontend::AddVariableForArguments<int>(k);
    CublasFrontend::AddDevicePointerForArguments(A);
    CublasFrontend::AddVariableForArguments<int>(lda);
    CublasFrontend::AddDevicePointerForArguments(x);
    CublasFrontend::AddVariableForArguments<int>(incx);
    
    CublasFrontend::Execute("cublasStbsv_v2");
    return CublasFrontend::GetExitCode();
}  

extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDtbsv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, int k, const double *A, int lda, double *x, int incx){
    CublasFrontend::Prepare();
    
    CublasFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CublasFrontend::AddVariableForArguments<cublasFillMode_t>(uplo);
    CublasFrontend::AddVariableForArguments<cublasOperation_t>(trans);
    CublasFrontend::AddVariableForArguments<cublasDiagType_t>(diag);
    CublasFrontend::AddVariableForArguments<int>(n);
    CublasFrontend::AddVariableForArguments<int>(k);
    CublasFrontend::AddDevicePointerForArguments(A);
    CublasFrontend::AddVariableForArguments<int>(lda);
    CublasFrontend::AddDevicePointerForArguments(x);
    CublasFrontend::AddVariableForArguments<int>(incx);
    
    CublasFrontend::Execute("cublasDtbsv_v2");
    return CublasFrontend::GetExitCode();
}  

extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCtbsv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, int k, const cuComplex *A, int lda, cuComplex *x, int incx){
    CublasFrontend::Prepare();
    
    CublasFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CublasFrontend::AddVariableForArguments<cublasFillMode_t>(uplo);
    CublasFrontend::AddVariableForArguments<cublasOperation_t>(trans);
    CublasFrontend::AddVariableForArguments<cublasDiagType_t>(diag);
    CublasFrontend::AddVariableForArguments<int>(n);
    CublasFrontend::AddVariableForArguments<int>(k);
    CublasFrontend::AddDevicePointerForArguments(A);
    CublasFrontend::AddVariableForArguments<int>(lda);
    CublasFrontend::AddDevicePointerForArguments(x);
    CublasFrontend::AddVariableForArguments<int>(incx);
    
    CublasFrontend::Execute("cublasCtbsv_v2");
    return CublasFrontend::GetExitCode();
}

extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZtbsv_v2(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int n, int k, const cuDoubleComplex *A, int lda, cuDoubleComplex *x, int incx){
    CublasFrontend::Prepare();
    
    CublasFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CublasFrontend::AddVariableForArguments<cublasFillMode_t>(uplo);
    CublasFrontend::AddVariableForArguments<cublasOperation_t>(trans);
    CublasFrontend::AddVariableForArguments<cublasDiagType_t>(diag);
    CublasFrontend::AddVariableForArguments<int>(n);
    CublasFrontend::AddVariableForArguments<int>(k);
    CublasFrontend::AddDevicePointerForArguments(A);
    CublasFrontend::AddVariableForArguments<int>(lda);
    CublasFrontend::AddDevicePointerForArguments(x);
    CublasFrontend::AddVariableForArguments<int>(incx);
    
    CublasFrontend::Execute("cublasZtbsv_v2");
    return CublasFrontend::GetExitCode();
}

extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSsymv_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const float *alpha, const float *A, int lda, const float *x, int incx, const float *beta, float *y, int incy){
    CublasFrontend::Prepare();
    
    CublasFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CublasFrontend::AddVariableForArguments<cublasFillMode_t>(uplo);
    CublasFrontend::AddVariableForArguments<int>(n);
    float * _alpha = const_cast<float *>(alpha);
    CublasFrontend::AddHostPointerForArguments(_alpha);
    CublasFrontend::AddDevicePointerForArguments(A);
    CublasFrontend::AddVariableForArguments<int>(lda);
    CublasFrontend::AddDevicePointerForArguments(x);
    CublasFrontend::AddVariableForArguments<int>(incx);
    float * _beta = const_cast<float *>(beta);
    CublasFrontend::AddHostPointerForArguments(_beta);
    CublasFrontend::AddDevicePointerForArguments(y);
    CublasFrontend::AddVariableForArguments<int>(incy);
    
    CublasFrontend::Execute("cublasSsymv_v2");
    return CublasFrontend::GetExitCode();
}

extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDsymv_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const double *alpha, const double *A, int lda, const double *x, int incx, const double *beta, double *y, int incy){
    CublasFrontend::Prepare();
    
    CublasFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CublasFrontend::AddVariableForArguments<cublasFillMode_t>(uplo);
    CublasFrontend::AddVariableForArguments<int>(n);
    double * _alpha = const_cast<double *>(alpha);
    CublasFrontend::AddHostPointerForArguments(_alpha);
    CublasFrontend::AddDevicePointerForArguments(A);
    CublasFrontend::AddVariableForArguments<int>(lda);
    CublasFrontend::AddDevicePointerForArguments(x);
    CublasFrontend::AddVariableForArguments<int>(incx);
    double * _beta = const_cast<double *>(beta);
    CublasFrontend::AddHostPointerForArguments(_beta);
    CublasFrontend::AddDevicePointerForArguments(y);
    CublasFrontend::AddVariableForArguments<int>(incy);
    
    CublasFrontend::Execute("cublasDsymv_v2");
    return CublasFrontend::GetExitCode();
}

extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCsymv_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuComplex *alpha, const cuComplex *A, int lda, const cuComplex *x, int incx, const cuComplex *beta, cuComplex *y, int incy){
    CublasFrontend::Prepare();
    
    CublasFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CublasFrontend::AddVariableForArguments<cublasFillMode_t>(uplo);
    CublasFrontend::AddVariableForArguments<int>(n);
    cuComplex * _alpha = const_cast<cuComplex *>(alpha);
    CublasFrontend::AddHostPointerForArguments(_alpha);
    CublasFrontend::AddDevicePointerForArguments(A);
    CublasFrontend::AddVariableForArguments<int>(lda);
    CublasFrontend::AddDevicePointerForArguments(x);
    CublasFrontend::AddVariableForArguments<int>(incx);
    cuComplex * _beta = const_cast<cuComplex *>(beta);
    CublasFrontend::AddHostPointerForArguments(_beta);
    CublasFrontend::AddDevicePointerForArguments(y);
    CublasFrontend::AddVariableForArguments<int>(incy);
    
    CublasFrontend::Execute("cublasCsymv_v2");
    return CublasFrontend::GetExitCode();
}

extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZsymv_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuDoubleComplex *alpha, const cuDoubleComplex *A, int lda, const cuDoubleComplex *x, int incx, const cuDoubleComplex *beta, cuDoubleComplex *y, int incy){
    CublasFrontend::Prepare();
    
    CublasFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CublasFrontend::AddVariableForArguments<cublasFillMode_t>(uplo);
    CublasFrontend::AddVariableForArguments<int>(n);
    cuDoubleComplex * _alpha = const_cast<cuDoubleComplex *>(alpha);
    CublasFrontend::AddHostPointerForArguments(_alpha);
    CublasFrontend::AddDevicePointerForArguments(A);
    CublasFrontend::AddVariableForArguments<int>(lda);
    CublasFrontend::AddDevicePointerForArguments(x);
    CublasFrontend::AddVariableForArguments<int>(incx);
    cuDoubleComplex * _beta = const_cast<cuDoubleComplex *>(beta);
    CublasFrontend::AddHostPointerForArguments(_beta);
    CublasFrontend::AddDevicePointerForArguments(y);
    CublasFrontend::AddVariableForArguments<int>(incy);
    
    CublasFrontend::Execute("cublasZsymv_v2");
    return CublasFrontend::GetExitCode();
}

extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasChemv_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuComplex *alpha, const cuComplex *A, int lda, const cuComplex *x, int incx, const cuComplex *beta, cuComplex *y, int incy){
    CublasFrontend::Prepare();
    
    CublasFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CublasFrontend::AddVariableForArguments<cublasFillMode_t>(uplo);
    CublasFrontend::AddVariableForArguments<int>(n);
    cuComplex * _alpha = const_cast<cuComplex *>(alpha);
    CublasFrontend::AddHostPointerForArguments(_alpha);
    CublasFrontend::AddDevicePointerForArguments(A);
    CublasFrontend::AddVariableForArguments<int>(lda);
    CublasFrontend::AddDevicePointerForArguments(x);
    CublasFrontend::AddVariableForArguments<int>(incx);
    cuComplex * _beta = const_cast<cuComplex *>(beta);
    CublasFrontend::AddHostPointerForArguments(_beta);
    CublasFrontend::AddDevicePointerForArguments(y);
    CublasFrontend::AddVariableForArguments<int>(incy);
    
    CublasFrontend::Execute("cublasChemv_v2");
    return CublasFrontend::GetExitCode();
}

extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZhemv_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuDoubleComplex *alpha, const cuDoubleComplex *A, int lda, const cuDoubleComplex *x, int incx, const cuDoubleComplex *beta, cuDoubleComplex *y, int incy){
    CublasFrontend::Prepare();
    
    CublasFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CublasFrontend::AddVariableForArguments<cublasFillMode_t>(uplo);
    CublasFrontend::AddVariableForArguments<int>(n);
    cuDoubleComplex * _alpha = const_cast<cuDoubleComplex *>(alpha);
    CublasFrontend::AddHostPointerForArguments(_alpha);
    CublasFrontend::AddDevicePointerForArguments(A);
    CublasFrontend::AddVariableForArguments<int>(lda);
    CublasFrontend::AddDevicePointerForArguments(x);
    CublasFrontend::AddVariableForArguments<int>(incx);
    cuDoubleComplex * _beta = const_cast<cuDoubleComplex *>(beta);
    CublasFrontend::AddHostPointerForArguments(_beta);
    CublasFrontend::AddDevicePointerForArguments(y);
    CublasFrontend::AddVariableForArguments<int>(incy);
    
    CublasFrontend::Execute("cublasZhemv_v2");
    return CublasFrontend::GetExitCode();
}

extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSsbmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, int k, const float *alpha, const float *A, int lda, const float *x, int incx, const float *beta, float *y, int incy){
    CublasFrontend::Prepare();
    
    CublasFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CublasFrontend::AddVariableForArguments<cublasFillMode_t>(uplo);
    CublasFrontend::AddVariableForArguments<int>(n);
    CublasFrontend::AddVariableForArguments<int>(k);
    float * _alpha = const_cast<float *>(alpha);
    CublasFrontend::AddHostPointerForArguments(_alpha);
    CublasFrontend::AddDevicePointerForArguments(A);
    CublasFrontend::AddVariableForArguments<int>(lda);
    CublasFrontend::AddDevicePointerForArguments(x);
    CublasFrontend::AddVariableForArguments<int>(incx);
    float * _beta = const_cast<float *>(beta);
    CublasFrontend::AddHostPointerForArguments(_beta);
    CublasFrontend::AddDevicePointerForArguments(y);
    CublasFrontend::AddVariableForArguments<int>(incy);
    
    CublasFrontend::Execute("cublasSsbmv_v2");
    return CublasFrontend::GetExitCode();
}

extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDsbmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, int k, const double *alpha, const double *A, int lda, const double *x, int incx, const double *beta, double *y, int incy){
    CublasFrontend::Prepare();
    
    CublasFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CublasFrontend::AddVariableForArguments<cublasFillMode_t>(uplo);
    CublasFrontend::AddVariableForArguments<int>(n);
    CublasFrontend::AddVariableForArguments<int>(k);
    double * _alpha = const_cast<double *>(alpha);
    CublasFrontend::AddHostPointerForArguments(_alpha);
    CublasFrontend::AddDevicePointerForArguments(A);
    CublasFrontend::AddVariableForArguments<int>(lda);
    CublasFrontend::AddDevicePointerForArguments(x);
    CublasFrontend::AddVariableForArguments<int>(incx);
    double * _beta = const_cast<double *>(beta);
    CublasFrontend::AddHostPointerForArguments(_beta);
    CublasFrontend::AddDevicePointerForArguments(y);
    CublasFrontend::AddVariableForArguments<int>(incy);
    
    CublasFrontend::Execute("cublasDsbmv_v2");
    return CublasFrontend::GetExitCode();
}

extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasChbmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, int k, const cuComplex *alpha, const cuComplex *A, int lda, const cuComplex *x, int incx, const cuComplex *beta, cuComplex *y, int incy){
    CublasFrontend::Prepare();
    
    CublasFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CublasFrontend::AddVariableForArguments<cublasFillMode_t>(uplo);
    CublasFrontend::AddVariableForArguments<int>(n);
    CublasFrontend::AddVariableForArguments<int>(k);
    cuComplex * _alpha = const_cast<cuComplex *>(alpha);
    CublasFrontend::AddHostPointerForArguments(_alpha);
    CublasFrontend::AddDevicePointerForArguments(A);
    CublasFrontend::AddVariableForArguments<int>(lda);
    CublasFrontend::AddDevicePointerForArguments(x);
    CublasFrontend::AddVariableForArguments<int>(incx);
    cuComplex * _beta = const_cast<cuComplex *>(beta);
    CublasFrontend::AddHostPointerForArguments(_beta);
    CublasFrontend::AddDevicePointerForArguments(y);
    CublasFrontend::AddVariableForArguments<int>(incy);
    
    CublasFrontend::Execute("cublasChbmv_v2");
    return CublasFrontend::GetExitCode();
}

extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZhbmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, int k, const cuDoubleComplex *alpha, const cuDoubleComplex *A, int lda, const cuDoubleComplex *x, int incx, const cuDoubleComplex *beta, cuDoubleComplex *y, int incy){
    CublasFrontend::Prepare();
    
    CublasFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CublasFrontend::AddVariableForArguments<cublasFillMode_t>(uplo);
    CublasFrontend::AddVariableForArguments<int>(n);
    CublasFrontend::AddVariableForArguments<int>(k);
    cuDoubleComplex * _alpha = const_cast<cuDoubleComplex *>(alpha);
    CublasFrontend::AddHostPointerForArguments(_alpha);
    CublasFrontend::AddDevicePointerForArguments(A);
    CublasFrontend::AddVariableForArguments<int>(lda);
    CublasFrontend::AddDevicePointerForArguments(x);
    CublasFrontend::AddVariableForArguments<int>(incx);
    cuDoubleComplex * _beta = const_cast<cuDoubleComplex *>(beta);
    CublasFrontend::AddHostPointerForArguments(_beta);
    CublasFrontend::AddDevicePointerForArguments(y);
    CublasFrontend::AddVariableForArguments<int>(incy);
    
    CublasFrontend::Execute("cublasZhbmv_v2");
    return CublasFrontend::GetExitCode();
}

extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSspmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const float *alpha, const float *AP, const float *x, int incx, const float *beta, float *y, int incy){
    CublasFrontend::Prepare();
    
    CublasFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CublasFrontend::AddVariableForArguments<cublasFillMode_t>(uplo);
    CublasFrontend::AddVariableForArguments<int>(n);
    float * _alpha = const_cast<float *>(alpha);
    CublasFrontend::AddHostPointerForArguments(_alpha);
    
    CublasFrontend::AddDevicePointerForArguments(AP);
    CublasFrontend::AddDevicePointerForArguments(x);
    CublasFrontend::AddVariableForArguments<int>(incx);
    
    float * _beta = const_cast<float *>(beta);
    CublasFrontend::AddHostPointerForArguments(_beta);
    CublasFrontend::AddDevicePointerForArguments(y);
    CublasFrontend::AddVariableForArguments<int>(incy);
    
    CublasFrontend::Execute("cublasSspmv_v2");
    return CublasFrontend::GetExitCode();
}

extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDspmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const double *alpha, const double *AP, const double *x, int incx, const double *beta, double *y, int incy){
    CublasFrontend::Prepare();
    
    CublasFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CublasFrontend::AddVariableForArguments<cublasFillMode_t>(uplo);
    CublasFrontend::AddVariableForArguments<int>(n);
    double * _alpha = const_cast<double *>(alpha);
    CublasFrontend::AddHostPointerForArguments(_alpha);
    
    CublasFrontend::AddDevicePointerForArguments(AP);
    CublasFrontend::AddDevicePointerForArguments(x);
    CublasFrontend::AddVariableForArguments<int>(incx);
    
    double * _beta = const_cast<double *>(beta);
    CublasFrontend::AddHostPointerForArguments(_beta);
    CublasFrontend::AddDevicePointerForArguments(y);
    CublasFrontend::AddVariableForArguments<int>(incy);
    
    CublasFrontend::Execute("cublasDspmv_v2");
    return CublasFrontend::GetExitCode();
}

extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasChpmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuComplex *alpha, const cuComplex *AP, const cuComplex *x, int incx, const cuComplex *beta, cuComplex *y, int incy){
    CublasFrontend::Prepare();
    
    CublasFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CublasFrontend::AddVariableForArguments<cublasFillMode_t>(uplo);
    CublasFrontend::AddVariableForArguments<int>(n);
    cuComplex * _alpha = const_cast<cuComplex *>(alpha);
    CublasFrontend::AddHostPointerForArguments(_alpha);
    
    CublasFrontend::AddDevicePointerForArguments(AP);
    CublasFrontend::AddDevicePointerForArguments(x);
    CublasFrontend::AddVariableForArguments<int>(incx);
    
    cuComplex * _beta = const_cast<cuComplex *>(beta);
    CublasFrontend::AddHostPointerForArguments(_beta);
    CublasFrontend::AddDevicePointerForArguments(y);
    CublasFrontend::AddVariableForArguments<int>(incy);
    
    CublasFrontend::Execute("cublasChpmv_v2");
    return CublasFrontend::GetExitCode();
}

extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZhpmv_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuDoubleComplex *alpha, const cuDoubleComplex *AP, const cuDoubleComplex *x, int incx, const cuDoubleComplex *beta, cuDoubleComplex *y, int incy){
    CublasFrontend::Prepare();
    
    CublasFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CublasFrontend::AddVariableForArguments<cublasFillMode_t>(uplo);
    CublasFrontend::AddVariableForArguments<int>(n);
    cuDoubleComplex * _alpha = const_cast<cuDoubleComplex *>(alpha);
    CublasFrontend::AddHostPointerForArguments(_alpha);
    
    CublasFrontend::AddDevicePointerForArguments(AP);
    CublasFrontend::AddDevicePointerForArguments(x);
    CublasFrontend::AddVariableForArguments<int>(incx);
    
    cuDoubleComplex * _beta = const_cast<cuDoubleComplex *>(beta);
    CublasFrontend::AddHostPointerForArguments(_beta);
    CublasFrontend::AddDevicePointerForArguments(y);
    CublasFrontend::AddVariableForArguments<int>(incy);
    
    CublasFrontend::Execute("cublasZhpmv_v2");
    return CublasFrontend::GetExitCode();
}


extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSger_v2(cublasHandle_t handle, int m, int n, const float *alpha, const float *x, int incx, const float *y, int incy, float *A, int lda){
    CublasFrontend::Prepare();
    CublasFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CublasFrontend::AddVariableForArguments<int>(m);
    CublasFrontend::AddVariableForArguments<int>(n);
    float * _alpha = const_cast<float *>(alpha);
    CublasFrontend::AddHostPointerForArguments(_alpha);
    CublasFrontend::AddDevicePointerForArguments(x);
    CublasFrontend::AddVariableForArguments<int>(incx);
    CublasFrontend::AddDevicePointerForArguments(y);
    CublasFrontend::AddVariableForArguments<int>(incy);
    CublasFrontend::AddDevicePointerForArguments(A);
    CublasFrontend::AddVariableForArguments<int>(lda);
    
    CublasFrontend::Execute("cublasSger_v2");
    return CublasFrontend::GetExitCode();
}

extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDger_v2(cublasHandle_t handle, int m, int n, const double *alpha, const double *x, int incx, const double *y, int incy, double *A, int lda){
    CublasFrontend::Prepare();
    CublasFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CublasFrontend::AddVariableForArguments<int>(m);
    CublasFrontend::AddVariableForArguments<int>(n);
    double * _alpha = const_cast<double *>(alpha);
    CublasFrontend::AddHostPointerForArguments(_alpha);
    CublasFrontend::AddDevicePointerForArguments(x);
    CublasFrontend::AddVariableForArguments<int>(incx);
    CublasFrontend::AddDevicePointerForArguments(y);
    CublasFrontend::AddVariableForArguments<int>(incy);
    CublasFrontend::AddDevicePointerForArguments(A);
    CublasFrontend::AddVariableForArguments<int>(lda);
    
    CublasFrontend::Execute("cublasDger_v2");
    return CublasFrontend::GetExitCode();
}

extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCgeru_v2(cublasHandle_t handle, int m, int n, const cuComplex *alpha, const cuComplex *x, int incx, const cuComplex *y, int incy, cuComplex *A, int lda){
    CublasFrontend::Prepare();
    CublasFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CublasFrontend::AddVariableForArguments<int>(m);
    CublasFrontend::AddVariableForArguments<int>(n);
    cuComplex * _alpha = const_cast<cuComplex *>(alpha);
    CublasFrontend::AddHostPointerForArguments(_alpha);
    CublasFrontend::AddDevicePointerForArguments(x);
    CublasFrontend::AddVariableForArguments<int>(incx);
    CublasFrontend::AddDevicePointerForArguments(y);
    CublasFrontend::AddVariableForArguments<int>(incy);
    CublasFrontend::AddDevicePointerForArguments(A);
    CublasFrontend::AddVariableForArguments<int>(lda);
    
    CublasFrontend::Execute("cublasCgeru_v2");
    return CublasFrontend::GetExitCode();
}

extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCgerc_v2(cublasHandle_t handle, int m, int n, const cuComplex *alpha, const cuComplex *x, int incx, const cuComplex *y, int incy, cuComplex *A, int lda){
    CublasFrontend::Prepare();
    CublasFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CublasFrontend::AddVariableForArguments<int>(m);
    CublasFrontend::AddVariableForArguments<int>(n);
    cuComplex * _alpha = const_cast<cuComplex *>(alpha);
    CublasFrontend::AddHostPointerForArguments(_alpha);
    CublasFrontend::AddDevicePointerForArguments(x);
    CublasFrontend::AddVariableForArguments<int>(incx);
    CublasFrontend::AddDevicePointerForArguments(y);
    CublasFrontend::AddVariableForArguments<int>(incy);
    CublasFrontend::AddDevicePointerForArguments(A);
    CublasFrontend::AddVariableForArguments<int>(lda);
    
    CublasFrontend::Execute("cublasCgerc_v2");
    return CublasFrontend::GetExitCode();
}

extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZgeru_v2(cublasHandle_t handle, int m, int n, const cuDoubleComplex *alpha, const cuDoubleComplex *x, int incx, const cuDoubleComplex *y, int incy, cuDoubleComplex *A, int lda){
    CublasFrontend::Prepare();
    CublasFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CublasFrontend::AddVariableForArguments<int>(m);
    CublasFrontend::AddVariableForArguments<int>(n);
    cuDoubleComplex * _alpha = const_cast<cuDoubleComplex *>(alpha);
    CublasFrontend::AddHostPointerForArguments(_alpha);
    CublasFrontend::AddDevicePointerForArguments(x);
    CublasFrontend::AddVariableForArguments<int>(incx);
    CublasFrontend::AddDevicePointerForArguments(y);
    CublasFrontend::AddVariableForArguments<int>(incy);
    CublasFrontend::AddDevicePointerForArguments(A);
    CublasFrontend::AddVariableForArguments<int>(lda);
    
    CublasFrontend::Execute("cublasZgeru_v2");
    return CublasFrontend::GetExitCode();
}

extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZgerc_v2(cublasHandle_t handle, int m, int n, const cuDoubleComplex *alpha, const cuDoubleComplex *x, int incx, const cuDoubleComplex *y, int incy, cuDoubleComplex *A, int lda){
    CublasFrontend::Prepare();
    CublasFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CublasFrontend::AddVariableForArguments<int>(m);
    CublasFrontend::AddVariableForArguments<int>(n);
    cuDoubleComplex * _alpha = const_cast<cuDoubleComplex *>(alpha);
    CublasFrontend::AddHostPointerForArguments(_alpha);
    CublasFrontend::AddDevicePointerForArguments(x);
    CublasFrontend::AddVariableForArguments<int>(incx);
    CublasFrontend::AddDevicePointerForArguments(y);
    CublasFrontend::AddVariableForArguments<int>(incy);
    CublasFrontend::AddDevicePointerForArguments(A);
    CublasFrontend::AddVariableForArguments<int>(lda);
    
    CublasFrontend::Execute("cublasZgerc_v2");
    return CublasFrontend::GetExitCode();
}

extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSsyr_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const float *alpha, const float *x, int incx, float *A, int lda){
    CublasFrontend::Prepare();
    
    CublasFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CublasFrontend::AddVariableForArguments<cublasFillMode_t>(uplo);
    CublasFrontend::AddVariableForArguments<int>(n);
    float * _alpha = const_cast<float *>(alpha);
    CublasFrontend::AddHostPointerForArguments(_alpha);
    CublasFrontend::AddDevicePointerForArguments(x);
    CublasFrontend::AddVariableForArguments<int>(incx);
    CublasFrontend::AddDevicePointerForArguments(A);
    CublasFrontend::AddVariableForArguments<int>(lda);
    
    CublasFrontend::Execute("cublasSsyr_v2");
    return CublasFrontend::GetExitCode();
}

extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDsyr_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const double *alpha, const double *x, int incx, double *A, int lda){
    CublasFrontend::Prepare();
    
    CublasFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CublasFrontend::AddVariableForArguments<cublasFillMode_t>(uplo);
    CublasFrontend::AddVariableForArguments<int>(n);
    double * _alpha = const_cast<double *>(alpha);
    CublasFrontend::AddHostPointerForArguments(_alpha);
    CublasFrontend::AddDevicePointerForArguments(x);
    CublasFrontend::AddVariableForArguments<int>(incx);
    CublasFrontend::AddDevicePointerForArguments(A);
    CublasFrontend::AddVariableForArguments<int>(lda);
    
    CublasFrontend::Execute("cublasDsyr_v2");
    return CublasFrontend::GetExitCode();
}

extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCsyr_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuComplex *alpha, const cuComplex *x, int incx, cuComplex *A, int lda){
    CublasFrontend::Prepare();
    
    CublasFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CublasFrontend::AddVariableForArguments<cublasFillMode_t>(uplo);
    CublasFrontend::AddVariableForArguments<int>(n);
    cuComplex * _alpha = const_cast<cuComplex *>(alpha);
    CublasFrontend::AddHostPointerForArguments(_alpha);
    CublasFrontend::AddDevicePointerForArguments(x);
    CublasFrontend::AddVariableForArguments<int>(incx);
    CublasFrontend::AddDevicePointerForArguments(A);
    CublasFrontend::AddVariableForArguments<int>(lda);
    
    CublasFrontend::Execute("cublasCsyr_v2");
    return CublasFrontend::GetExitCode();
}

extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZsyr_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuDoubleComplex *alpha, const cuDoubleComplex *x, int incx, cuDoubleComplex *A, int lda){
    CublasFrontend::Prepare();
    
    CublasFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CublasFrontend::AddVariableForArguments<cublasFillMode_t>(uplo);
    CublasFrontend::AddVariableForArguments<int>(n);
    cuDoubleComplex * _alpha = const_cast<cuDoubleComplex *>(alpha);
    CublasFrontend::AddHostPointerForArguments(_alpha);
    CublasFrontend::AddDevicePointerForArguments(x);
    CublasFrontend::AddVariableForArguments<int>(incx);
    CublasFrontend::AddDevicePointerForArguments(A);
    CublasFrontend::AddVariableForArguments<int>(lda);
    
    CublasFrontend::Execute("cublasZsyr_v2");
    return CublasFrontend::GetExitCode();
}

extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCher_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const float *alpha, const cuComplex *x, int incx, cuComplex *A, int lda){
    CublasFrontend::Prepare();
    
    CublasFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CublasFrontend::AddVariableForArguments<cublasFillMode_t>(uplo);
    CublasFrontend::AddVariableForArguments<int>(n);
    float * _alpha = const_cast<float *>(alpha);
    CublasFrontend::AddHostPointerForArguments(_alpha);
    CublasFrontend::AddDevicePointerForArguments(x);
    CublasFrontend::AddVariableForArguments<int>(incx);
    CublasFrontend::AddDevicePointerForArguments(A);
    CublasFrontend::AddVariableForArguments<int>(lda);
    
    CublasFrontend::Execute("cublasCher_v2");
    return CublasFrontend::GetExitCode();
}

extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZher_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const double *alpha, const cuDoubleComplex *x, int incx, cuDoubleComplex *A, int lda){
    CublasFrontend::Prepare();
    
    CublasFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CublasFrontend::AddVariableForArguments<cublasFillMode_t>(uplo);
    CublasFrontend::AddVariableForArguments<int>(n);
    double * _alpha = const_cast<double *>(alpha);
    CublasFrontend::AddHostPointerForArguments(_alpha);
    CublasFrontend::AddDevicePointerForArguments(x);
    CublasFrontend::AddVariableForArguments<int>(incx);
    CublasFrontend::AddDevicePointerForArguments(A);
    CublasFrontend::AddVariableForArguments<int>(lda);
    
    CublasFrontend::Execute("cublasZher_v2");
    return CublasFrontend::GetExitCode();
}


extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSspr_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const float *alpha, const float *x, int incx, float *AP){
    CublasFrontend::Prepare();
    
    CublasFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CublasFrontend::AddVariableForArguments<cublasFillMode_t>(uplo);
    CublasFrontend::AddVariableForArguments<int>(n);
    float * _alpha = const_cast<float *>(alpha);
    CublasFrontend::AddHostPointerForArguments(_alpha);
    CublasFrontend::AddDevicePointerForArguments(x);
    CublasFrontend::AddVariableForArguments<int>(incx);
    CublasFrontend::AddDevicePointerForArguments(AP);
    
    CublasFrontend::Execute("cublasSspr_v2");
    return CublasFrontend::GetExitCode();
}

extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDspr_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const double *alpha, const double *x, int incx, double *AP){
    CublasFrontend::Prepare();
    
    CublasFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CublasFrontend::AddVariableForArguments<cublasFillMode_t>(uplo);
    CublasFrontend::AddVariableForArguments<int>(n);
    double * _alpha = const_cast<double *>(alpha);
    CublasFrontend::AddHostPointerForArguments(_alpha);
    CublasFrontend::AddDevicePointerForArguments(x);
    CublasFrontend::AddVariableForArguments<int>(incx);
    CublasFrontend::AddDevicePointerForArguments(AP);
    
    CublasFrontend::Execute("cublasDspr_v2");
    return CublasFrontend::GetExitCode();
}

extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasChpr_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const float *alpha, const cuComplex *x, int incx, cuComplex *AP){
    CublasFrontend::Prepare();
    
    CublasFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CublasFrontend::AddVariableForArguments<cublasFillMode_t>(uplo);
    CublasFrontend::AddVariableForArguments<int>(n);
    float * _alpha = const_cast<float *>(alpha);
    CublasFrontend::AddHostPointerForArguments(_alpha);
    CublasFrontend::AddDevicePointerForArguments(x);
    CublasFrontend::AddVariableForArguments<int>(incx);
    CublasFrontend::AddDevicePointerForArguments(AP);
    
    CublasFrontend::Execute("cublasChpr_v2");
    return CublasFrontend::GetExitCode();
}

extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZhpr_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const double *alpha, const cuDoubleComplex *x, int incx, cuDoubleComplex *AP){
    CublasFrontend::Prepare();
    
    CublasFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CublasFrontend::AddVariableForArguments<cublasFillMode_t>(uplo);
    CublasFrontend::AddVariableForArguments<int>(n);
    double * _alpha = const_cast<double *>(alpha);
    CublasFrontend::AddHostPointerForArguments(_alpha);
    CublasFrontend::AddDevicePointerForArguments(x);
    CublasFrontend::AddVariableForArguments<int>(incx);
    CublasFrontend::AddDevicePointerForArguments(AP);
    
    CublasFrontend::Execute("cublasZhpr_v2");
    return CublasFrontend::GetExitCode();
}

extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSsyr2_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const float *alpha, const float *x, int incx, const float *y, int incy, float *A, int lda){
    CublasFrontend::Prepare();
    
    CublasFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CublasFrontend::AddVariableForArguments<cublasFillMode_t>(uplo);
    CublasFrontend::AddVariableForArguments<int>(n);
    float * _alpha = const_cast<float *>(alpha);
    CublasFrontend::AddHostPointerForArguments(_alpha);
    CublasFrontend::AddDevicePointerForArguments(x);
    CublasFrontend::AddVariableForArguments<int>(incx);
    CublasFrontend::AddDevicePointerForArguments(y);
    CublasFrontend::AddVariableForArguments<int>(incy);
    CublasFrontend::AddDevicePointerForArguments(A);
    CublasFrontend::AddVariableForArguments<int>(lda);
    
    CublasFrontend::Execute("cublasSsyr2_v2");
    return CublasFrontend::GetExitCode();
}


extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDsyr2_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const double *alpha, const double *x, int incx, const double *y, int incy, double *A, int lda){
    CublasFrontend::Prepare();
    
    CublasFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CublasFrontend::AddVariableForArguments<cublasFillMode_t>(uplo);
    CublasFrontend::AddVariableForArguments<int>(n);
    double * _alpha = const_cast<double *>(alpha);
    CublasFrontend::AddHostPointerForArguments(_alpha);
    CublasFrontend::AddDevicePointerForArguments(x);
    CublasFrontend::AddVariableForArguments<int>(incx);
    CublasFrontend::AddDevicePointerForArguments(y);
    CublasFrontend::AddVariableForArguments<int>(incy);
    CublasFrontend::AddDevicePointerForArguments(A);
    CublasFrontend::AddVariableForArguments<int>(lda);
    
    CublasFrontend::Execute("cublasDsyr2_v2");
    return CublasFrontend::GetExitCode();
}

extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCsyr2_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuComplex *alpha, const cuComplex *x, int incx, const cuComplex *y, int incy, cuComplex *A, int lda){
    CublasFrontend::Prepare();
    
    CublasFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CublasFrontend::AddVariableForArguments<cublasFillMode_t>(uplo);
    CublasFrontend::AddVariableForArguments<int>(n);
    cuComplex * _alpha = const_cast<cuComplex *>(alpha);
    CublasFrontend::AddHostPointerForArguments(_alpha);
    CublasFrontend::AddDevicePointerForArguments(x);
    CublasFrontend::AddVariableForArguments<int>(incx);
    CublasFrontend::AddDevicePointerForArguments(y);
    CublasFrontend::AddVariableForArguments<int>(incy);
    CublasFrontend::AddDevicePointerForArguments(A);
    CublasFrontend::AddVariableForArguments<int>(lda);
    
    CublasFrontend::Execute("cublasCsyr2_v2");
    return CublasFrontend::GetExitCode();
}

extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZsyr2_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuDoubleComplex *alpha, const cuDoubleComplex *x, int incx, const cuDoubleComplex *y, int incy, cuDoubleComplex *A, int lda){
    CublasFrontend::Prepare();
    
    CublasFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CublasFrontend::AddVariableForArguments<cublasFillMode_t>(uplo);
    CublasFrontend::AddVariableForArguments<int>(n);
    cuDoubleComplex * _alpha = const_cast<cuDoubleComplex *>(alpha);
    CublasFrontend::AddHostPointerForArguments(_alpha);
    CublasFrontend::AddDevicePointerForArguments(x);
    CublasFrontend::AddVariableForArguments<int>(incx);
    CublasFrontend::AddDevicePointerForArguments(y);
    CublasFrontend::AddVariableForArguments<int>(incy);
    CublasFrontend::AddDevicePointerForArguments(A);
    CublasFrontend::AddVariableForArguments<int>(lda);
    
    CublasFrontend::Execute("cublasZsyr2_v2");
    return CublasFrontend::GetExitCode();
}

extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCher2_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuComplex *alpha, const cuComplex *x, int incx, const cuComplex *y, int incy, cuComplex *A, int lda){
    CublasFrontend::Prepare();
    
    CublasFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CublasFrontend::AddVariableForArguments<cublasFillMode_t>(uplo);
    CublasFrontend::AddVariableForArguments<int>(n);
    cuComplex * _alpha = const_cast<cuComplex *>(alpha);
    CublasFrontend::AddHostPointerForArguments(_alpha);
    CublasFrontend::AddDevicePointerForArguments(x);
    CublasFrontend::AddVariableForArguments<int>(incx);
    CublasFrontend::AddDevicePointerForArguments(y);
    CublasFrontend::AddVariableForArguments<int>(incy);
    CublasFrontend::AddDevicePointerForArguments(A);
    CublasFrontend::AddVariableForArguments<int>(lda);
    
    CublasFrontend::Execute("cublasCher2_v2");
    return CublasFrontend::GetExitCode();
}

extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZher2_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuDoubleComplex *alpha, const cuDoubleComplex *x, int incx, const cuDoubleComplex *y, int incy, cuDoubleComplex *A, int lda){
    CublasFrontend::Prepare();
    
    CublasFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CublasFrontend::AddVariableForArguments<cublasFillMode_t>(uplo);
    CublasFrontend::AddVariableForArguments<int>(n);
    cuDoubleComplex * _alpha = const_cast<cuDoubleComplex *>(alpha);
    CublasFrontend::AddHostPointerForArguments(_alpha);
    CublasFrontend::AddDevicePointerForArguments(x);
    CublasFrontend::AddVariableForArguments<int>(incx);
    CublasFrontend::AddDevicePointerForArguments(y);
    CublasFrontend::AddVariableForArguments<int>(incy);
    CublasFrontend::AddDevicePointerForArguments(A);
    CublasFrontend::AddVariableForArguments<int>(lda);
    
    CublasFrontend::Execute("cublasZher2_v2");
    return CublasFrontend::GetExitCode();
}

extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSspr2_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const float *alpha, const float *x, int incx, const float *y, int incy, float *AP){
    CublasFrontend::Prepare();
    
    CublasFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CublasFrontend::AddVariableForArguments<cublasFillMode_t>(uplo);
    CublasFrontend::AddVariableForArguments<int>(n);
    float * _alpha = const_cast<float *>(alpha);
    CublasFrontend::AddHostPointerForArguments(_alpha);
    CublasFrontend::AddDevicePointerForArguments(x);
    CublasFrontend::AddVariableForArguments<int>(incx);
    CublasFrontend::AddDevicePointerForArguments(y);
    CublasFrontend::AddVariableForArguments<int>(incy);
    CublasFrontend::AddDevicePointerForArguments(AP);
    
    CublasFrontend::Execute("cublasSspr2_v2");
    return CublasFrontend::GetExitCode();
}

extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDspr2_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const double *alpha, const double *x, int incx, const double *y, int incy, double *AP){
    CublasFrontend::Prepare();
    
    CublasFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CublasFrontend::AddVariableForArguments<cublasFillMode_t>(uplo);
    CublasFrontend::AddVariableForArguments<int>(n);
    double * _alpha = const_cast<double *>(alpha);
    CublasFrontend::AddHostPointerForArguments(_alpha);
    CublasFrontend::AddDevicePointerForArguments(x);
    CublasFrontend::AddVariableForArguments<int>(incx);
    CublasFrontend::AddDevicePointerForArguments(y);
    CublasFrontend::AddVariableForArguments<int>(incy);
    CublasFrontend::AddDevicePointerForArguments(AP);
    
    CublasFrontend::Execute("cublasDspr2_v2");
    return CublasFrontend::GetExitCode();
}

extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasChpr2_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuComplex *alpha, const cuComplex *x, int incx, const cuComplex *y, int incy, cuComplex *AP){
    CublasFrontend::Prepare();
    
    CublasFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CublasFrontend::AddVariableForArguments<cublasFillMode_t>(uplo);
    CublasFrontend::AddVariableForArguments<int>(n);
    cuComplex * _alpha = const_cast<cuComplex *>(alpha);
    CublasFrontend::AddHostPointerForArguments(_alpha);
    CublasFrontend::AddDevicePointerForArguments(x);
    CublasFrontend::AddVariableForArguments<int>(incx);
    CublasFrontend::AddDevicePointerForArguments(y);
    CublasFrontend::AddVariableForArguments<int>(incy);
    CublasFrontend::AddDevicePointerForArguments(AP);
    
    CublasFrontend::Execute("cublasChpr2_v2");
    return CublasFrontend::GetExitCode();
}

extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZhpr2_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n, const cuDoubleComplex *alpha, const cuDoubleComplex *x, int incx, const cuDoubleComplex *y, int incy, cuDoubleComplex *AP){
    CublasFrontend::Prepare();
    
    CublasFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CublasFrontend::AddVariableForArguments<cublasFillMode_t>(uplo);
    CublasFrontend::AddVariableForArguments<int>(n);
    cuDoubleComplex * _alpha = const_cast<cuDoubleComplex *>(alpha);
    CublasFrontend::AddHostPointerForArguments(_alpha);
    CublasFrontend::AddDevicePointerForArguments(x);
    CublasFrontend::AddVariableForArguments<int>(incx);
    CublasFrontend::AddDevicePointerForArguments(y);
    CublasFrontend::AddVariableForArguments<int>(incy);
    CublasFrontend::AddDevicePointerForArguments(AP);
    
    CublasFrontend::Execute("cublasZhpr2_v2");
    return CublasFrontend::GetExitCode();
}

