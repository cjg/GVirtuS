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

extern "C" CL_API_ENTRY cl_context CL_API_CALL

clCreateContext(const cl_context_properties *properties,

  	cl_uint num_devices,

  	const cl_device_id *devices,

  	void (*pfn_notify)(

            const char *errinfo,

            const void *private_info,

            size_t cb,

            void *user_data),

  	void *user_data,

  	cl_int *errcode_ret){


    OpenclFrontend::Prepare();
    unsigned count = 0;
    for(; properties != NULL && properties[count] != 0; ++count){}

    cout<<"count"<<count<<endl;
    OpenclFrontend::AddHostPointerForArguments<const cl_context_properties>(properties, count+1);

    OpenclFrontend::AddVariableForArguments<cl_uint>(num_devices);


    OpenclFrontend::AddHostPointerForArguments(devices,num_devices);

//cl_context context=clCreateContext(properties, 1, &device, NULL, NULL, &error);

    OpenclFrontend::Execute("clCreateContext");



    cl_context ret_context=NULL;

    if (OpenclFrontend::Success()){

        if (errcode_ret != NULL)

            *errcode_ret=OpenclFrontend::GetExitCode();

        ret_context=*(OpenclFrontend::GetOutputHostPointer<cl_context>());

    }


    return ret_context;


}
extern "C" CL_API_ENTRY cl_int CL_API_CALL
clGetContextInfo (cl_context context,
  	cl_context_info param_name,
  	size_t param_value_size,
  	void *param_value,
  	size_t *param_value_size_ret){

    OpenclFrontend::Prepare();
    OpenclFrontend::AddHostPointerForArguments<cl_context>(&context);
    OpenclFrontend::AddVariableForArguments<cl_context_info>(param_name);
    OpenclFrontend::AddVariableForArguments<size_t>(param_value_size);
    OpenclFrontend::AddHostPointerForArguments((char*)param_value,param_value_size);
    OpenclFrontend::AddHostPointerForArguments<size_t>(param_value_size_ret);

    OpenclFrontend::Execute("clGetContextInfo");

    if (OpenclFrontend::Success()){
        size_t *tmp_param_value_size_ret = OpenclFrontend::GetOutputHostPointer<size_t>();
        if (param_value_size_ret !=NULL)
            *param_value_size_ret=*tmp_param_value_size_ret;
        if (param_value!=NULL)
            memmove(param_value,OpenclFrontend::GetOutputHostPointer<char>(),*tmp_param_value_size_ret);
    }

    return OpenclFrontend::GetExitCode();
}

extern "C" CL_API_ENTRY cl_int CL_API_CALL
clReleaseContext (cl_context context){

    OpenclFrontend::Prepare();
    OpenclFrontend::AddStringForArguments(OpenclUtil::MarshalHostPointer(context));

    OpenclFrontend::Execute("clReleaseContext");

    return OpenclFrontend::GetExitCode();

}