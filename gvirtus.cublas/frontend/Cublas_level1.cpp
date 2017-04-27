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

#include <iostream>
#include <cstdio>
#include <string>

#include "Frontend.h"
#include "CublasFrontend.h"

using namespace std;


extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSdot_v2(cublasHandle_t handle, int n, const float *x, int incx, const float *y, int incy, float *result){
    CublasFrontend::Prepare();
    
    CublasFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CublasFrontend::AddVariableForArguments<int>(n);
    CublasFrontend::AddDevicePointerForArguments(x);
    CublasFrontend::AddVariableForArguments<int>(incx);
    CublasFrontend::AddDevicePointerForArguments(y);
    CublasFrontend::AddVariableForArguments<int>(incy);
    CublasFrontend::AddHostPointerForArguments<float>(result);
    
    CublasFrontend::Execute("cublasSdot_v2");
    
    return CublasFrontend::GetExitCode();
}

extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDdot_v2(cublasHandle_t handle, int n, const double *x, int incx, const double *y, int incy, double *result){
    CublasFrontend::Prepare();
    
    CublasFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CublasFrontend::AddVariableForArguments<int>(n);
    CublasFrontend::AddDevicePointerForArguments(x);
    CublasFrontend::AddVariableForArguments<int>(incx);
    CublasFrontend::AddDevicePointerForArguments(y);
    CublasFrontend::AddVariableForArguments<int>(incy);
    CublasFrontend::AddHostPointerForArguments<double>(result);
    
    CublasFrontend::Execute("cublasDdot_v2");
    
    return CublasFrontend::GetExitCode();
}


extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCdotu_v2(cublasHandle_t handle, int n, const cuComplex *x, int incx, const cuComplex *y, int incy, cuComplex *result){
    CublasFrontend::Prepare();
    
    CublasFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CublasFrontend::AddVariableForArguments<int>(n);
    CublasFrontend::AddDevicePointerForArguments(x);
    CublasFrontend::AddVariableForArguments<int>(incx);
    CublasFrontend::AddDevicePointerForArguments(y);
    CublasFrontend::AddVariableForArguments<int>(incy);
    CublasFrontend::AddHostPointerForArguments<cuComplex>(result);
    
    CublasFrontend::Execute("cublasCdotu_v2");
    
    return CublasFrontend::GetExitCode();
}

extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCdotc_v2(cublasHandle_t handle, int n, const cuComplex *x, int incx, const cuComplex *y, int incy, cuComplex *result){
    CublasFrontend::Prepare();
    
    CublasFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CublasFrontend::AddVariableForArguments<int>(n);
    CublasFrontend::AddDevicePointerForArguments(x);
    CublasFrontend::AddVariableForArguments<int>(incx);
    CublasFrontend::AddDevicePointerForArguments(y);
    CublasFrontend::AddVariableForArguments<int>(incy);
    CublasFrontend::AddHostPointerForArguments<cuComplex>(result);
    
    CublasFrontend::Execute("cublasCdotc_v2");
    
    return CublasFrontend::GetExitCode();
}


extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZdotu_v2(cublasHandle_t handle, int n, const cuDoubleComplex *x, int incx, const cuDoubleComplex *y, int incy, cuDoubleComplex *result){
    CublasFrontend::Prepare();
    
    CublasFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CublasFrontend::AddVariableForArguments<int>(n);
    CublasFrontend::AddDevicePointerForArguments(x);
    CublasFrontend::AddVariableForArguments<int>(incx);
    CublasFrontend::AddDevicePointerForArguments(y);
    CublasFrontend::AddVariableForArguments<int>(incy);
    CublasFrontend::AddHostPointerForArguments<cuDoubleComplex>(result);
    
    CublasFrontend::Execute("cublasZdotu_v2");
    
    return CublasFrontend::GetExitCode();
}

extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZdotc_v2(cublasHandle_t handle, int n, const cuDoubleComplex *x, int incx, const cuDoubleComplex *y, int incy, cuDoubleComplex *result){
    CublasFrontend::Prepare();
    
    CublasFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CublasFrontend::AddVariableForArguments<int>(n);
    CublasFrontend::AddDevicePointerForArguments(x);
    CublasFrontend::AddVariableForArguments<int>(incx);
    CublasFrontend::AddDevicePointerForArguments(y);
    CublasFrontend::AddVariableForArguments<int>(incy);
    CublasFrontend::AddHostPointerForArguments<cuDoubleComplex>(result);
    
    CublasFrontend::Execute("cublasZdotc_v2");
    
    return CublasFrontend::GetExitCode();
}

extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSscal_v2(cublasHandle_t handle,
                                                     int n,
                                                     const float *alpha,  /* host or device pointer */
                                                     float *x,
                                                     int incx){
    CublasFrontend::Prepare();
    CublasFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CublasFrontend::AddVariableForArguments<int>(n);
    CublasFrontend::AddHostPointerForArguments<float>((float*)alpha);
    //CublasFrontend::AddDevicePointerForArguments(alpha); 
    CublasFrontend::AddDevicePointerForArguments(x); 
    CublasFrontend::AddVariableForArguments(incx);
    CublasFrontend::Execute("cublasSscal_v2");
    return CublasFrontend::GetExitCode(); 
}

extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDscal_v2(cublasHandle_t handle,
                                                     int n,
                                                     const double *alpha,  /* host or device pointer */
                                                     double *x,
                                                     int incx){
    CublasFrontend::Prepare();
    CublasFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CublasFrontend::AddVariableForArguments<int>(n);
    CublasFrontend::AddHostPointerForArguments<double>((double*)alpha);
    //CublasFrontend::AddDevicePointerForArguments(alpha); 
    CublasFrontend::AddDevicePointerForArguments(x); 
    CublasFrontend::AddVariableForArguments(incx);
    CublasFrontend::Execute("cublasDscal_v2");
    return CublasFrontend::GetExitCode(); 
}

extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCscal_v2(cublasHandle_t handle,
                                                     int n,
                                                     const cuComplex *alpha,  /* host or device pointer */
                                                     cuComplex *x,
                                                     int incx){
    CublasFrontend::Prepare();
    CublasFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CublasFrontend::AddVariableForArguments<int>(n);
    CublasFrontend::AddHostPointerForArguments<cuComplex>((cuComplex*)alpha);
    //CublasFrontend::AddDevicePointerForArguments(alpha); 
    CublasFrontend::AddDevicePointerForArguments(x); 
    CublasFrontend::AddVariableForArguments(incx);
    CublasFrontend::Execute("cublasCscal_v2");
    return CublasFrontend::GetExitCode(); 
}

extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCsscal_v2(cublasHandle_t handle,
                                                     int n,
                                                     const float *alpha,  /* host or device pointer */
                                                     cuComplex *x,
                                                     int incx){
    CublasFrontend::Prepare();
    CublasFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CublasFrontend::AddVariableForArguments<int>(n);
    CublasFrontend::AddHostPointerForArguments<float>((float*)alpha);
    //CublasFrontend::AddDevicePointerForArguments(alpha); 
    CublasFrontend::AddDevicePointerForArguments(x); 
    CublasFrontend::AddVariableForArguments(incx);
    CublasFrontend::Execute("cublasCsscal_v2");
    return CublasFrontend::GetExitCode(); 
}

extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZscal_v2(cublasHandle_t handle,
                                                     int n,
                                                     const cuDoubleComplex *alpha,  /* host or device pointer */
                                                     cuDoubleComplex *x,
                                                     int incx){
    CublasFrontend::Prepare();
    CublasFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CublasFrontend::AddVariableForArguments<int>(n);
    CublasFrontend::AddHostPointerForArguments<cuDoubleComplex>((cuDoubleComplex*)alpha);
    //CublasFrontend::AddDevicePointerForArguments(alpha); 
    CublasFrontend::AddDevicePointerForArguments(x); 
    CublasFrontend::AddVariableForArguments(incx);
    CublasFrontend::Execute("cublasZscal_v2");
    return CublasFrontend::GetExitCode(); 
}

extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZdscal_v2(cublasHandle_t handle,
                                                     int n,
                                                     const double *alpha,  /* host or device pointer */
                                                     cuDoubleComplex *x,
                                                     int incx){
    CublasFrontend::Prepare();
    CublasFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CublasFrontend::AddVariableForArguments<int>(n);
    CublasFrontend::AddHostPointerForArguments<double>((double*)alpha);
    //CublasFrontend::AddDevicePointerForArguments(alpha); 
    CublasFrontend::AddDevicePointerForArguments(x); 
    CublasFrontend::AddVariableForArguments(incx);
    CublasFrontend::Execute("cublasZdscal_v2");
    return CublasFrontend::GetExitCode(); 
}

extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSaxpy_v2(cublasHandle_t handle, int n, const float *alpha, const float *x, int incx, float *y, int incy){
    CublasFrontend::Prepare();
    CublasFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CublasFrontend::AddVariableForArguments<int>(n);
    CublasFrontend::AddHostPointerForArguments<float>((float*)alpha);
    CublasFrontend::AddDevicePointerForArguments(x);
    CublasFrontend::AddVariableForArguments<int>(incx); 
    CublasFrontend::AddDevicePointerForArguments(y);
    CublasFrontend::AddVariableForArguments<int>(incy);
    
    CublasFrontend::Execute("cublasSaxpy_v2");
    
    return CublasFrontend::GetExitCode();
} 

extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDaxpy_v2(cublasHandle_t handle, int n, const double *alpha, const double *x, int incx, double *y, int incy){
    CublasFrontend::Prepare();
    CublasFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CublasFrontend::AddVariableForArguments<int>(n);
    CublasFrontend::AddHostPointerForArguments<double>((double*)alpha);
    CublasFrontend::AddDevicePointerForArguments(x);
    CublasFrontend::AddVariableForArguments<int>(incx); 
    CublasFrontend::AddDevicePointerForArguments(y);
    CublasFrontend::AddVariableForArguments<int>(incy);
    
    CublasFrontend::Execute("cublasDaxpy_v2");
    
    return CublasFrontend::GetExitCode();
}

extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCaxpy_v2(cublasHandle_t handle, int n, const cuComplex *alpha, const cuComplex *x, int incx, cuComplex *y, int incy){
    CublasFrontend::Prepare();
    CublasFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CublasFrontend::AddVariableForArguments<int>(n);
    CublasFrontend::AddHostPointerForArguments<cuComplex>((cuComplex*)alpha);
    CublasFrontend::AddDevicePointerForArguments(x);
    CublasFrontend::AddVariableForArguments<int>(incx); 
    CublasFrontend::AddDevicePointerForArguments(y);
    CublasFrontend::AddVariableForArguments<int>(incy);
    
    CublasFrontend::Execute("cublasCaxpy_v2");
    
    return CublasFrontend::GetExitCode();
}

extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZaxpy_v2(cublasHandle_t handle, int n, const cuDoubleComplex *alpha, const cuDoubleComplex *x, int incx, cuDoubleComplex *y, int incy){
    CublasFrontend::Prepare();
    CublasFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CublasFrontend::AddVariableForArguments<int>(n);
    CublasFrontend::AddHostPointerForArguments<cuDoubleComplex>((cuDoubleComplex*)alpha);
    CublasFrontend::AddDevicePointerForArguments(x);
    CublasFrontend::AddVariableForArguments<int>(incx); 
    CublasFrontend::AddDevicePointerForArguments(y);
    CublasFrontend::AddVariableForArguments<int>(incy);
    
    CublasFrontend::Execute("cublasZaxpy_v2");
    
    return CublasFrontend::GetExitCode();
}


extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasScopy_v2(cublasHandle_t handle, int n, const float *x, int incx, float *y, int incy){
    CublasFrontend::Prepare();
    CublasFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CublasFrontend::AddVariableForArguments<int>(n);
    CublasFrontend::AddDevicePointerForArguments(x);
    CublasFrontend::AddVariableForArguments<int>(incx);
    CublasFrontend::AddDevicePointerForArguments(y);
    CublasFrontend::AddVariableForArguments<int>(incy);
    
    CublasFrontend::Execute("cublasScopy_v2");
    return CublasFrontend::GetExitCode();
}

extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDcopy_v2(cublasHandle_t handle, int n, const double *x, int incx, double *y, int incy){
    CublasFrontend::Prepare();
    CublasFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CublasFrontend::AddVariableForArguments<int>(n);
    CublasFrontend::AddDevicePointerForArguments(x);
    CublasFrontend::AddVariableForArguments<int>(incx);
    CublasFrontend::AddDevicePointerForArguments(y);
    CublasFrontend::AddVariableForArguments<int>(incy);
    
    CublasFrontend::Execute("cublasDcopy_v2");
    return CublasFrontend::GetExitCode();
}


extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCcopy_v2(cublasHandle_t handle, int n, const cuComplex *x, int incx, cuComplex *y, int incy){
    CublasFrontend::Prepare();
    CublasFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CublasFrontend::AddVariableForArguments<int>(n);
    CublasFrontend::AddDevicePointerForArguments(x);
    CublasFrontend::AddVariableForArguments<int>(incx);
    CublasFrontend::AddDevicePointerForArguments(y);
    CublasFrontend::AddVariableForArguments<int>(incy);
    
    CublasFrontend::Execute("cublasCcopy_v2");
    return CublasFrontend::GetExitCode();
}

extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZcopy_v2(cublasHandle_t handle, int n, const cuDoubleComplex *x, int incx, cuDoubleComplex *y, int incy){
    CublasFrontend::Prepare();
    CublasFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CublasFrontend::AddVariableForArguments<int>(n);
    CublasFrontend::AddDevicePointerForArguments(x);
    CublasFrontend::AddVariableForArguments<int>(incx);
    CublasFrontend::AddDevicePointerForArguments(y);
    CublasFrontend::AddVariableForArguments<int>(incy);
    
    CublasFrontend::Execute("cublasZcopy_v2");
    return CublasFrontend::GetExitCode();
}


extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSswap_v2(cublasHandle_t handle, int n, float *x, int incx, float *y, int incy){
    CublasFrontend::Prepare();
    CublasFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CublasFrontend::AddVariableForArguments<int>(n);
    CublasFrontend::AddDevicePointerForArguments(x);
    CublasFrontend::AddVariableForArguments<int>(incx);
    CublasFrontend::AddDevicePointerForArguments(y);
    CublasFrontend::AddVariableForArguments<int>(incy);
    
    CublasFrontend::Execute("cublasSswap_v2");
    return CublasFrontend::GetExitCode();
}

extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDswap_v2(cublasHandle_t handle, int n, double *x, int incx, double *y, int incy){
    CublasFrontend::Prepare();
    CublasFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CublasFrontend::AddVariableForArguments<int>(n);
    CublasFrontend::AddDevicePointerForArguments(x);
    CublasFrontend::AddVariableForArguments<int>(incx);
    CublasFrontend::AddDevicePointerForArguments(y);
    CublasFrontend::AddVariableForArguments<int>(incy);
    
    CublasFrontend::Execute("cublasDswap_v2");
    return CublasFrontend::GetExitCode();
}

extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCswap_v2(cublasHandle_t handle, int n, cuComplex *x, int incx, cuComplex *y, int incy){
    CublasFrontend::Prepare();
    CublasFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CublasFrontend::AddVariableForArguments<int>(n);
    CublasFrontend::AddDevicePointerForArguments(x);
    CublasFrontend::AddVariableForArguments<int>(incx);
    CublasFrontend::AddDevicePointerForArguments(y);
    CublasFrontend::AddVariableForArguments<int>(incy);
    
    CublasFrontend::Execute("cublasCswap_v2");
    return CublasFrontend::GetExitCode();
}

extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZswap_v2(cublasHandle_t handle, int n, cuDoubleComplex *x, int incx, cuDoubleComplex *y, int incy){
    CublasFrontend::Prepare();
    CublasFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CublasFrontend::AddVariableForArguments<int>(n);
    CublasFrontend::AddDevicePointerForArguments(x);
    CublasFrontend::AddVariableForArguments<int>(incx);
    CublasFrontend::AddDevicePointerForArguments(y);
    CublasFrontend::AddVariableForArguments<int>(incy);
    
    CublasFrontend::Execute("cublasZswap_v2");
    return CublasFrontend::GetExitCode();
}

extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasIsamax_v2(cublasHandle_t handle, int n, const float *x, int incx, int *result){
    CublasFrontend::Prepare();
    
    CublasFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CublasFrontend::AddVariableForArguments<int>(n);
    CublasFrontend::AddDevicePointerForArguments(x);
    CublasFrontend::AddVariableForArguments<int>(incx);
    CublasFrontend::AddHostPointerForArguments<int>(result);
    
    CublasFrontend::Execute("cublasIsamax_v2");
    if(CublasFrontend::Success())
        result = CublasFrontend::GetOutputHostPointer<int>();
    return CublasFrontend::GetExitCode();
}

extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasIdamax_v2(cublasHandle_t handle, int n, const double *x, int incx, int *result){
    CublasFrontend::Prepare();
    
    CublasFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CublasFrontend::AddVariableForArguments<int>(n);
    CublasFrontend::AddDevicePointerForArguments(x);
    CublasFrontend::AddVariableForArguments<int>(incx);
    CublasFrontend::AddHostPointerForArguments<int>(result);
    
    CublasFrontend::Execute("cublasIdamax_v2");
    if(CublasFrontend::Success())
        result = CublasFrontend::GetOutputHostPointer<int>();
    return CublasFrontend::GetExitCode();
}