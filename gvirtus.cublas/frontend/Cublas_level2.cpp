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