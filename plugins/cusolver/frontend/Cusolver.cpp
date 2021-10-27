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
 * Department of Applied Science
 */

#include <iostream>
#include <cstdio>
#include <string>

#include "CusolverFrontend.h"

using namespace std;


size_t CUSOLVERAPI cusolverDnGetVersion(){
    CusolverFrontend::Prepare();

    CusolverFrontend::Execute("cusolverDnGetVersion");
    return CusolverFrontend::GetExitCode();
}

extern "C" const char * CUSOLVERAPI cusolverDnGetErrorString(cusolverStatus_t status){
    CusolverFrontend::Prepare();

    CusolverFrontend::AddVariableForArguments<cusolverStatus_t>(status);
    CusolverFrontend::Execute("cusolverDnGetErrorString");
    return (const char *) CusolverFrontend::GetOutputHostPointer<char *>();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnCreate(cusolverDnHandle_t *handle){
    CusolverFrontend::Prepare();
    CusolverFrontend::AddHostPointerForArguments<cusolverDnHandle_t>(handle);
    CusolverFrontend::Execute("cusolverDnCreate");
    if(CusolverFrontend::Success())
        *handle = CusolverFrontend::GetOutputVariable<cusolverDnHandle_t>();
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnDestroy(cusolverDnHandle_t handle){
    CusolverFrontend::Prepare();

    CusolverFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CusolverFrontend::Execute("cusolverDnDestroy");
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnSetStream(cusolverDnHandle_t handle, cudaStream_t streamId){
    CusolverFrontend::Prepare();

    CusolverFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CusolverFrontend::AddVariableForArguments<long long int>((long long int)streamId);
    CusolverFrontend::Execute("cusolverSetStream");
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnGetStream(cusolverDnHandle_t handle, cudaStream_t *streamId){
    CusolverFrontend::Prepare();

    CusolverFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CusolverFrontend::Execute("cusolverDnGetStream");
    if(CusolverFrontend::Success())
        *streamId = (cudaStream_t) CusolverFrontend::GetOutputVariable<long long int>();
    return CusolverFrontend::GetExitCode();
}
