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

#include "Opencl_gv.h"
#include <iostream>
#include <cstdio>
#include <string>

using namespace std;

extern "C" CL_API_ENTRY cl_kernel CL_API_CALL
clCreateKernel (cl_program  program,
  	const char *kernel_name,
  	cl_int *errcode_ret)  {

    OpenclFrontend::Prepare();
    OpenclFrontend::AddHostPointerForArguments<cl_program>(&program);
    OpenclFrontend::AddStringForArguments(kernel_name);

    OpenclFrontend::Execute("clCreateKernel");

    cl_kernel kernel=NULL;
    if (OpenclFrontend::Success()){
        if (errcode_ret != NULL){
            *errcode_ret = OpenclFrontend::GetExitCode();
        }
        kernel = *(OpenclFrontend::GetOutputHostPointer<cl_kernel>());
    }
    return kernel;
}

extern "C" CL_API_ENTRY cl_int CL_API_CALL
clSetKernelArg (cl_kernel kernel,
  	cl_uint arg_index,
  	size_t arg_size,
  	const void *arg_value) {

    OpenclFrontend::Prepare();
    OpenclFrontend::AddHostPointerForArguments<cl_kernel>(&kernel);
    OpenclFrontend::AddVariableForArguments<cl_uint>(arg_index);
    OpenclFrontend::AddVariableForArguments<size_t>(arg_size);
    OpenclFrontend::AddHostPointerForArguments((char *)arg_value,arg_size);

    OpenclFrontend::Execute("clSetKernelArg");

    return OpenclFrontend::GetExitCode();
}

extern "C" CL_API_ENTRY cl_int CL_API_CALL
clReleaseKernel (cl_kernel kernel) {
    OpenclFrontend::Prepare();
    OpenclFrontend::AddStringForArguments(OpenclUtil::MarshalHostPointer(kernel));

    OpenclFrontend::Execute("clReleaseKernel");

    return OpenclFrontend::GetExitCode();
}