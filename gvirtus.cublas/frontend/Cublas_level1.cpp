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

extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasIcamax_v2(cublasHandle_t handle, int n, const cuComplex *x, int incx, int *result){
    CublasFrontend::Prepare();
    
    CublasFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CublasFrontend::AddVariableForArguments<int>(n);
    CublasFrontend::AddDevicePointerForArguments(x);
    CublasFrontend::AddVariableForArguments<int>(incx);
    CublasFrontend::AddHostPointerForArguments<int>(result);
    
    CublasFrontend::Execute("cublasIcamax_v2");
    if(CublasFrontend::Success())
        result = CublasFrontend::GetOutputHostPointer<int>();
    return CublasFrontend::GetExitCode();
}

extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasIzamax_v2(cublasHandle_t handle, int n, const cuDoubleComplex *x, int incx, int *result){
    CublasFrontend::Prepare();
    
    CublasFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CublasFrontend::AddVariableForArguments<int>(n);
    CublasFrontend::AddDevicePointerForArguments(x);
    CublasFrontend::AddVariableForArguments<int>(incx);
    CublasFrontend::AddHostPointerForArguments<int>(result);
    
    CublasFrontend::Execute("cublasIzamax_v2");
    if(CublasFrontend::Success())
        result = CublasFrontend::GetOutputHostPointer<int>();
    return CublasFrontend::GetExitCode();
}


extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasIsamin_v2(cublasHandle_t handle, int n, const float *x, int incx, int *result){
    CublasFrontend::Prepare();
    
    CublasFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CublasFrontend::AddVariableForArguments<int>(n);
    CublasFrontend::AddDevicePointerForArguments(x);
    CublasFrontend::AddVariableForArguments<int>(incx);
    CublasFrontend::AddHostPointerForArguments<int>(result);
    
    CublasFrontend::Execute("cublasIsamin_v2");
    if(CublasFrontend::Success())
        result = CublasFrontend::GetOutputHostPointer<int>();
    return CublasFrontend::GetExitCode();
}

extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasIdamin_v2(cublasHandle_t handle, int n, const double *x, int incx, int *result){
    CublasFrontend::Prepare();
    
    CublasFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CublasFrontend::AddVariableForArguments<int>(n);
    CublasFrontend::AddDevicePointerForArguments(x);
    CublasFrontend::AddVariableForArguments<int>(incx);
    CublasFrontend::AddHostPointerForArguments<int>(result);
    
    CublasFrontend::Execute("cublasIdamin_v2");
    if(CublasFrontend::Success())
        result = CublasFrontend::GetOutputHostPointer<int>();
    return CublasFrontend::GetExitCode();
}

extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasIcamin_v2(cublasHandle_t handle, int n, const cuComplex *x, int incx, int *result){
    CublasFrontend::Prepare();
    
    CublasFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CublasFrontend::AddVariableForArguments<int>(n);
    CublasFrontend::AddDevicePointerForArguments(x);
    CublasFrontend::AddVariableForArguments<int>(incx);
    CublasFrontend::AddHostPointerForArguments<int>(result);
    
    CublasFrontend::Execute("cublasIcamin_v2");
    if(CublasFrontend::Success())
        result = CublasFrontend::GetOutputHostPointer<int>();
    return CublasFrontend::GetExitCode();
}

extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasIzamin_v2(cublasHandle_t handle, int n, const cuDoubleComplex *x, int incx, int *result){
    CublasFrontend::Prepare();
    
    CublasFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CublasFrontend::AddVariableForArguments<int>(n);
    CublasFrontend::AddDevicePointerForArguments(x);
    CublasFrontend::AddVariableForArguments<int>(incx);
    CublasFrontend::AddHostPointerForArguments<int>(result);
    
    CublasFrontend::Execute("cublasIzamin_v2");
    if(CublasFrontend::Success())
        result = CublasFrontend::GetOutputHostPointer<int>();
    return CublasFrontend::GetExitCode();
}

extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSasum_v2(cublasHandle_t handle, int n, const float *x, int incx, float *result) {
    CublasFrontend::Prepare();
    
    CublasFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CublasFrontend::AddVariableForArguments<int>(n);
    CublasFrontend::AddDevicePointerForArguments(x);
    CublasFrontend::AddVariableForArguments<int>(incx);
    CublasFrontend::AddHostPointerForArguments<float>(result);
    
    CublasFrontend::Execute("cublasSasum_v2");
    if(CublasFrontend::Success())
        result = CublasFrontend::GetOutputHostPointer<float>();
    return CublasFrontend::GetExitCode();
}

extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDasum_v2(cublasHandle_t handle, int n, const double *x, int incx, double *result) {
    CublasFrontend::Prepare();
    
    CublasFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CublasFrontend::AddVariableForArguments<int>(n);
    CublasFrontend::AddDevicePointerForArguments(x);
    CublasFrontend::AddVariableForArguments<int>(incx);
    CublasFrontend::AddHostPointerForArguments<double>(result);
    
    CublasFrontend::Execute("cublasDasum_v2");
    if(CublasFrontend::Success())
        result = CublasFrontend::GetOutputHostPointer<double>();
    return CublasFrontend::GetExitCode();
}

extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasScasum_v2(cublasHandle_t handle, int n, const cuComplex *x, int incx, float *result) {
    CublasFrontend::Prepare();
    
    CublasFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CublasFrontend::AddVariableForArguments<int>(n);
    CublasFrontend::AddDevicePointerForArguments(x);
    CublasFrontend::AddVariableForArguments<int>(incx);
    CublasFrontend::AddHostPointerForArguments<float>(result);
    
    CublasFrontend::Execute("cublasScasum_v2");
    if(CublasFrontend::Success())
        result = CublasFrontend::GetOutputHostPointer<float>();
    return CublasFrontend::GetExitCode();
}

extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDzasum_v2(cublasHandle_t handle, int n, const cuDoubleComplex *x, int incx, double *result) {
    CublasFrontend::Prepare();
    
    CublasFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CublasFrontend::AddVariableForArguments<int>(n);
    CublasFrontend::AddDevicePointerForArguments(x);
    CublasFrontend::AddVariableForArguments<int>(incx);
    CublasFrontend::AddHostPointerForArguments<double>(result);
    
    CublasFrontend::Execute("cublasDzasum_v2");
    if(CublasFrontend::Success())
        result = CublasFrontend::GetOutputHostPointer<double>();
    return CublasFrontend::GetExitCode();
}

extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSrot_v2(cublasHandle_t handle, int n, float *x, int incx, float *y, int incy, const float *c, const float *s){
    CublasFrontend::Prepare();
    
    CublasFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CublasFrontend::AddVariableForArguments<int>(n);
    CublasFrontend::AddDevicePointerForArguments(x);
    CublasFrontend::AddVariableForArguments<int>(incx);
    CublasFrontend::AddDevicePointerForArguments(y);
    CublasFrontend::AddVariableForArguments<int>(incy);
    CublasFrontend::AddHostPointerForArguments<float>((float*)c);
    CublasFrontend::AddHostPointerForArguments<float>((float*)s);
    
    CublasFrontend::Execute("cublasSrot_v2");
    return CublasFrontend::GetExitCode();
}

extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDrot_v2(cublasHandle_t handle, int n, double *x, int incx, double *y, int incy, const double *c, const double *s){
    CublasFrontend::Prepare();
    
    CublasFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CublasFrontend::AddVariableForArguments<int>(n);
    CublasFrontend::AddDevicePointerForArguments(x);
    CublasFrontend::AddVariableForArguments<int>(incx);
    CublasFrontend::AddDevicePointerForArguments(y);
    CublasFrontend::AddVariableForArguments<int>(incy);
    CublasFrontend::AddHostPointerForArguments<double>((double*)c);
    CublasFrontend::AddHostPointerForArguments<double>((double*)s);
    
    CublasFrontend::Execute("cublasDrot_v2");
    return CublasFrontend::GetExitCode();
}

extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCrot_v2(cublasHandle_t handle, int n, cuComplex *x, int incx, cuComplex *y, int incy, const float *c, const cuComplex *s){
    CublasFrontend::Prepare();
    
    CublasFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CublasFrontend::AddVariableForArguments<int>(n);
    CublasFrontend::AddDevicePointerForArguments(x);
    CublasFrontend::AddVariableForArguments<int>(incx);
    CublasFrontend::AddDevicePointerForArguments(y);
    CublasFrontend::AddVariableForArguments<int>(incy);
    CublasFrontend::AddHostPointerForArguments<float>((float*)c);
    CublasFrontend::AddHostPointerForArguments<cuComplex>((cuComplex*)s);
    
    CublasFrontend::Execute("cublasCrot_v2");
    return CublasFrontend::GetExitCode();
}

extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCsrot_v2(cublasHandle_t handle, int n, cuComplex *x, int incx, cuComplex *y, int incy, const float *c, const float *s){
    CublasFrontend::Prepare();
    
    CublasFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CublasFrontend::AddVariableForArguments<int>(n);
    CublasFrontend::AddDevicePointerForArguments(x);
    CublasFrontend::AddVariableForArguments<int>(incx);
    CublasFrontend::AddDevicePointerForArguments(y);
    CublasFrontend::AddVariableForArguments<int>(incy);
    CublasFrontend::AddHostPointerForArguments<float>((float*)c);
    CublasFrontend::AddHostPointerForArguments<float>((float*)s);
    
    CublasFrontend::Execute("cublasCsrot_v2");
    return CublasFrontend::GetExitCode();
}


extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZrot_v2(cublasHandle_t handle, int n, cuDoubleComplex *x, int incx, cuDoubleComplex *y, int incy, const double *c, const cuDoubleComplex *s){
    CublasFrontend::Prepare();
    
    CublasFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CublasFrontend::AddVariableForArguments<int>(n);
    CublasFrontend::AddDevicePointerForArguments(x);
    CublasFrontend::AddVariableForArguments<int>(incx);
    CublasFrontend::AddDevicePointerForArguments(y);
    CublasFrontend::AddVariableForArguments<int>(incy);
    CublasFrontend::AddHostPointerForArguments<double>((double*)c);
    CublasFrontend::AddHostPointerForArguments<cuDoubleComplex>((cuDoubleComplex*)s);
    
    CublasFrontend::Execute("cublasZrot_v2");
    return CublasFrontend::GetExitCode();
}

extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZdrot_v2(cublasHandle_t handle, int n, cuDoubleComplex *x, int incx, cuDoubleComplex *y, int incy, const double *c, const double *s){
    CublasFrontend::Prepare();
    
    CublasFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CublasFrontend::AddVariableForArguments<int>(n);
    CublasFrontend::AddDevicePointerForArguments(x);
    CublasFrontend::AddVariableForArguments<int>(incx);
    CublasFrontend::AddDevicePointerForArguments(y);
    CublasFrontend::AddVariableForArguments<int>(incy);
    CublasFrontend::AddHostPointerForArguments<double>((double*)c);
    CublasFrontend::AddHostPointerForArguments<double>((double*)s);
    
    CublasFrontend::Execute("cublasZdrot_v2");
    return CublasFrontend::GetExitCode();
}


extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSrotg_v2(cublasHandle_t handle, float *a, float *b, float *c, float *s){
    CublasFrontend::Prepare();
    
    CublasFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CublasFrontend::AddHostPointerForArguments<float>(a);
    CublasFrontend::AddHostPointerForArguments<float>(b);
    CublasFrontend::AddHostPointerForArguments<float>(c);
    CublasFrontend::AddHostPointerForArguments<float>(s);
    
    CublasFrontend::Execute("cublasSrotg_v2");
    return CublasFrontend::GetExitCode();
}

extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDrotg_v2(cublasHandle_t handle, double *a, double *b, double *c, double *s){
    CublasFrontend::Prepare();
    
    CublasFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CublasFrontend::AddHostPointerForArguments<double>(a);
    CublasFrontend::AddHostPointerForArguments<double>(b);
    CublasFrontend::AddHostPointerForArguments<double>(c);
    CublasFrontend::AddHostPointerForArguments<double>(s);
    
    CublasFrontend::Execute("cublasDrotg_v2");
    return CublasFrontend::GetExitCode();
}

extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasCrotg_v2(cublasHandle_t handle, cuComplex *a, cuComplex *b, float *c, cuComplex *s){
    CublasFrontend::Prepare();
    
    CublasFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CublasFrontend::AddHostPointerForArguments<cuComplex>(a);
    CublasFrontend::AddHostPointerForArguments<cuComplex>(b);
    CublasFrontend::AddHostPointerForArguments<float>(c);
    CublasFrontend::AddHostPointerForArguments<cuComplex>(s);
    
    CublasFrontend::Execute("cublasCrotg_v2");
    return CublasFrontend::GetExitCode();
}

extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasZrotg_v2(cublasHandle_t handle, cuDoubleComplex *a, cuDoubleComplex *b, double *c, cuDoubleComplex *s){
    CublasFrontend::Prepare();
    
    CublasFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CublasFrontend::AddHostPointerForArguments<cuDoubleComplex>(a);
    CublasFrontend::AddHostPointerForArguments<cuDoubleComplex>(b);
    CublasFrontend::AddHostPointerForArguments<double>(c);
    CublasFrontend::AddHostPointerForArguments<cuDoubleComplex>(s);
    
    CublasFrontend::Execute("cublasZrotg_v2");
    return CublasFrontend::GetExitCode();
}

extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSrotm_v2(cublasHandle_t handle, int n, float *x, int incx, float *y, int incy, const float* param){
    CublasFrontend::Prepare();
    
    CublasFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CublasFrontend::AddVariableForArguments<int>(n);
    CublasFrontend::AddDevicePointerForArguments(x);
    CublasFrontend::AddVariableForArguments<int>(incx);
    CublasFrontend::AddDevicePointerForArguments(y);
    CublasFrontend::AddVariableForArguments<int>(incy);
    CublasFrontend::AddHostPointerForArguments<float>((float*)param);
    
    CublasFrontend::Execute("cublasSrotm_v2");
    return CublasFrontend::GetExitCode();
}

extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDrotm_v2(cublasHandle_t handle, int n, double *x, int incx, double *y, int incy, const double* param){
    CublasFrontend::Prepare();
    
    CublasFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CublasFrontend::AddVariableForArguments<int>(n);
    CublasFrontend::AddDevicePointerForArguments(x);
    CublasFrontend::AddVariableForArguments<int>(incx);
    CublasFrontend::AddDevicePointerForArguments(y);
    CublasFrontend::AddVariableForArguments<int>(incy);
    CublasFrontend::AddHostPointerForArguments<double>((double*)param);
    
    CublasFrontend::Execute("cublasDrotm_v2");
    return CublasFrontend::GetExitCode();
}

extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSrotmg_v2(cublasHandle_t handle, float *d1, float *d2, float *x1, const float *y1, float *param){
    CublasFrontend::Prepare();
    
    CublasFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CublasFrontend::AddHostPointerForArguments<float>(d1);
    CublasFrontend::AddHostPointerForArguments<float>(d2);
    CublasFrontend::AddHostPointerForArguments<float>(x1);
    CublasFrontend::AddHostPointerForArguments<float>((float*)y1);
    CublasFrontend::AddHostPointerForArguments<float>(param);
    
    CublasFrontend::Execute("cublasSrotmg_v2");
    if(CublasFrontend::Success())
        param = CublasFrontend::GetOutputHostPointer<float>();
    return CublasFrontend::GetExitCode();
}

extern "C" CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDrotmg_v2(cublasHandle_t handle, double *d1, double *d2, double *x1, const double *y1, double *param){
    CublasFrontend::Prepare();
    
    CublasFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CublasFrontend::AddHostPointerForArguments<double>(d1);
    CublasFrontend::AddHostPointerForArguments<double>(d2);
    CublasFrontend::AddHostPointerForArguments<double>(x1);
    CublasFrontend::AddHostPointerForArguments<double>((double*)y1);
    CublasFrontend::Execute("cublasDrotmg_v2");
    if(CublasFrontend::Success())
        param = CublasFrontend::GetOutputHostPointer<double>();
    return CublasFrontend::GetExitCode();
}

