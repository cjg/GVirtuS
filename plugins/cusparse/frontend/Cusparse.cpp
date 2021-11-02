/*
 *  * gVirtuS -- A GPGPU transparent virtualization component.
 *   *
 *    * Copyright (C) 2009-2010  The University of Napoli Parthenope at Naples.
 *     *
 *      * This file is part of gVirtuS.
 *       *
 *        * gVirtuS is free software; you can redistribute it and/or modify
 *         * it under the terms of the GNU General Public License as published by
 *          * the Free Software Foundation; either version 2 of the License, or
 *           * (at your option) any later version.
 *            *
 *             * gVirtuS is distributed in the hope that it will be useful,
 *              * but WITHOUT ANY WARRANTY; without even the implied warranty of
 *               * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *                * GNU Lesser General Public License for more details.
 *                 *
 *                  * You should have received a copy of the GNU General Public License
 *                   * along with gVirtuS; if not, write to the Free Software
 *                    * Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
 *                     *
 *                      * Written by: Antonio Pilato <antonio.pilato001@studenti.uniparthenope.it>,
 *                       *             Department of Science and Technologies
 *                        */

#include <iostream>
#include <cstdio>
#include <string>

#include "CusparseFrontend.h"

using namespace std;

extern "C" cusparseStatus_t cusparseGetVersion(cusparseHandle_t handle, int* version){
    CusparseFrontend::Prepare();
    CusparseFrontend::AddHostPointerForArguments<cusparseHandle_t>(handle);
    CusparseFrontend::AddHostPointerForArguments<int>(version);
    CusparseFrontend::Execute("cusparseGetVersion");
    return CusparseFrontend::GetExitCode();
}

extern "C" const char *cusparseGetErrorString(cusparseStatus_t status){
    CusparseFrontend::Prepare();

    CusparseFrontend::AddVariableForArguments<cusparseStatus_t>(status);
    CusparseFrontend::Execute("cusparseGetErrorString");
    return (const char *) CusparseFrontend::GetOutputHostPointer<char *>();
}

/*
extern "C" cusparseStatus_t CUSPARSEAPI cusparseCreate(cusparseHandle_t *handle){
    CusparseFrontend::Prepare();
    CusparseFrontend::AddHostPointerForArguments<cusparseHandle_t>(handle);
    CusparseFrontend::Execute("cusparseCreate");
    if(CusparseFrontend::Success())
        *handle = CusparseFrontend::GetOutputVariable<cusparseHandle_t>();
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseDestroy(cusparseHandle_t handle){
    CusparseFrontend::Prepare();

    CusparseFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CusparseFrontend::Execute("cusparseDestroy");
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseSetStream(cusparseHandle_t handle, cudaStream_t streamId){
    CusparseFrontend::Prepare();

    CusparseFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CusparseFrontend::AddVariableForArguments<long long int>((long long int)streamId);
    CusparseFrontend::Execute("cusparseSetStream");
    return CusparseFrontend::GetExitCode();
}

extern "C" cusparseStatus_t CUSPARSEAPI cusparseGetStream(cusparseHandle_t handle, cudaStream_t *streamId){
    CusparseFrontend::Prepare();

    CusparseFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CusparseFrontend::Execute("cusparseGetStream");
    if(CusparseFrontend::Success())
        *streamId = (cudaStream_t) CusparseFrontend::GetOutputVariable<long long int>();
    return CusparseFrontend::GetExitCode();
}
*/