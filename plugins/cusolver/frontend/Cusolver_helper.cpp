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
 * Written by: Antonio Pilato <antonio.pilato001@studenti.uniparthenope.it>,
 *              Department of Science andTechnology
 */

#include <iostream>
#include <cstdio>
#include <string>

#include "CusolverFrontend.h"

using namespace std;

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnCreate(cusolverDnHandle_t *handle) {
    CusolverFrontend::Prepare();
    //CusolverFrontend::AddHostPointerForArguments<cusolverDnHandle_t>(handle);
    CusolverFrontend::Execute("cusolverDnCreate");
    if(CusolverFrontend::Success())
    	*handle = *(CusolverFrontend::GetOutputHostPointer<cusolverDnHandle_t>());
    return CusolverFrontend::GetExitCode();
}
    
extern "C" cusolverStatus_t CUSOLVERAPI cuSolverDnDestroy(cusolverDnHandle_t handle) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments(handle);
    CusolverFrontend::Execute("cusolverDnDestroy");
    return CusolverFrontend::GetExitCode();
}
extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnSetStream(cusolverDnHandle_t handle, cudaStream_t streamId) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments(handle);
    CusolverFrontend::AddVariableForArguments(streamId);
    CusolverFrontend::Execute("cusolverDnSetStream");
    return CusolverFrontend::GetExitCode();
}
extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnGetStream(cusolverDnHandle_t handle, cudaStream_t *streamId) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments(handle);
    CusolverFrontend::Execute("cusolverDnGetStream");
    return CusolverFrontend::GetExitCode();
}   
