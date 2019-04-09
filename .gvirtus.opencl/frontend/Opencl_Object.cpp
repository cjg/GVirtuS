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

extern "C" CL_API_ENTRY cl_program CL_API_CALL
clCreateProgramWithSource (cl_context context,
  	cl_uint count,
  	const char **strings,
  	const size_t *lengths,
  	cl_int *errcode_ret) {

    OpenclFrontend::Prepare();
    OpenclFrontend::AddStringForArguments(OpenclUtil::MarshalHostPointer(context));
    OpenclFrontend::AddVariableForArguments<cl_uint>(count);
    OpenclFrontend::AddHostPointerForArguments<const size_t>(lengths,count);
    if (lengths != NULL){
        for (cl_uint i=0;i<count;i++){
            OpenclFrontend::AddHostPointerForArguments(strings[i],lengths[i]);
        }
    }else{
        for (cl_uint i=0;i<count;i++){
            OpenclFrontend::AddStringForArguments(strings[i]);
        }
    }

    OpenclFrontend::Execute("clCreateProgramWithSource");

    cl_program program=NULL;
    if(OpenclFrontend::Success()){
        program = (cl_program)OpenclUtil::UnmarshalPointer(OpenclFrontend::GetOutputString());
        if (errcode_ret != NULL){
            *errcode_ret=OpenclFrontend::GetExitCode();
        }
    }

    return program;
}

extern "C" CL_API_ENTRY cl_int CL_API_CALL
clBuildProgram (cl_program program,
  	cl_uint num_devices,
  	const cl_device_id *device_list,
  	const char *options,
  	void (*pfn_notify)(cl_program program, void *user_data),
  	void *user_data) {

    OpenclFrontend::Prepare();
    OpenclFrontend::AddHostPointerForArguments<cl_program>(&program);
    OpenclFrontend::AddVariableForArguments<cl_uint>(num_devices);
    OpenclFrontend::AddHostPointerForArguments(device_list,num_devices);
    if (options == NULL)
        OpenclFrontend::AddStringForArguments("");
    else
        OpenclFrontend::AddStringForArguments(options);

    OpenclFrontend::Execute("clBuildProgram");

    return OpenclFrontend::GetExitCode();
}


extern "C" CL_API_ENTRY cl_int CL_API_CALL
clReleaseProgram (cl_program program) {

    OpenclFrontend::Prepare();
    OpenclFrontend::AddStringForArguments(OpenclUtil::MarshalHostPointer(program));

    OpenclFrontend::Execute("clReleaseProgram");

    return OpenclFrontend::GetExitCode();

}
