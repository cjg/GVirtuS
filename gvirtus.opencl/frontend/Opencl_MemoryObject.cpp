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

extern "C" CL_API_ENTRY cl_mem CL_API_CALL
clCreateBuffer (cl_context context,
  	cl_mem_flags flags,
  	size_t size,
  	void *host_ptr,
  	cl_int *errcode_ret){

    OpenclFrontend::Prepare();
    OpenclFrontend::AddStringForArguments(OpenclUtil::MarshalHostPointer(context));
    OpenclFrontend::AddVariableForArguments<cl_mem_flags>(flags);
    OpenclFrontend::AddVariableForArguments<size_t>(size);
    OpenclFrontend::AddHostPointerForArguments<char>((char*)host_ptr,size);

    OpenclFrontend::Execute("clCreateBuffer");

    cl_mem mem=NULL;
    if (OpenclFrontend::Success()){
        mem = (cl_mem)OpenclUtil::UnmarshalPointer(OpenclFrontend::GetOutputString());
        if (errcode_ret !=NULL)
            *errcode_ret=OpenclFrontend::GetExitCode();
    }

    return mem;
}

extern "C" CL_API_ENTRY cl_int CL_API_CALL
clReleaseMemObject (cl_mem memobj){
    OpenclFrontend::Prepare();

    OpenclFrontend::AddHostPointerForArguments<cl_mem>(&memobj);

    OpenclFrontend::Execute("clReleaseMemObject");

    return OpenclFrontend::GetExitCode();


}
