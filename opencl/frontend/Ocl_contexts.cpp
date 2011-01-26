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
 * Written by: Roberto Di Lauro <roberto.dilauro@uniparthenope.it>,
 *             Department of Applied Science
 */

#include <string.h>

#include "Ocl.h"

using namespace std;


extern "C" cl_context clCreateContext(const cl_context_properties *properties,
  	cl_uint num_devices,
  	const cl_device_id *devices,
  	void (*pfn_notify)(
            const char *errinfo,
            const void *private_info,
            size_t cb,
            void *user_data),
  	void *user_data,
  	cl_int *errcode_ret){

    OclFrontend::Prepare();
    OclFrontend::AddHostPointerForArguments<const cl_context_properties>(properties);
    OclFrontend::AddVariableForArguments<cl_uint>(num_devices);

    OclFrontend::AddHostPointerForArguments(devices,num_devices);

    OclFrontend::Execute("clCreateContext");


    cl_context ret_context;
    if (OclFrontend::Success()){
        if (errcode_ret != NULL)
            *errcode_ret=OclFrontend::GetExitCode();
        ret_context=*(OclFrontend::GetOutputHostPointer<cl_context>());
    }

    return ret_context;

}

extern "C" cl_int clGetContextInfo (cl_context context,
  	cl_context_info param_name,
  	size_t param_value_size,
  	void *param_value,
  	size_t *param_value_size_ret){

    OclFrontend::Prepare();
    OclFrontend::AddHostPointerForArguments<cl_context>(&context);
    OclFrontend::AddVariableForArguments<cl_context_info>(param_name);
    OclFrontend::AddVariableForArguments<size_t>(param_value_size);
    OclFrontend::AddHostPointerForArguments((char*)param_value,param_value_size);
    OclFrontend::AddHostPointerForArguments<size_t>(param_value_size_ret);

    OclFrontend::Execute("clGetContextInfo");

    if (OclFrontend::Success()){
        size_t *tmp_param_value_size_ret = OclFrontend::GetOutputHostPointer<size_t>();
        if (param_value_size_ret !=NULL)
            *param_value_size_ret=*tmp_param_value_size_ret;
        if (param_value!=NULL)
            memmove(param_value,OclFrontend::GetOutputHostPointer<char>(),*tmp_param_value_size_ret);
    }

    return OclFrontend::GetExitCode();
}


extern "C" cl_context clCreateContextFromType (const cl_context_properties   *properties,
  	cl_device_type  device_type,
  	void  (*pfn_notify) (const char *errinfo,
  	const void  *private_info,
  	size_t  cb,
  	void  *user_data),
  	void  *user_data,
  	cl_int  *errcode_ret){

    cerr << "*** Error: clCreateContextFromType not yet implemented!" << endl;
    return 0;
}


extern "C" cl_int clRetainContext (cl_context context){
    cerr << "*** Error: clRetainContext not yet implemented!" << endl;
    return 0;
}

extern "C" cl_int clReleaseContext (cl_context context){

    OclFrontend::Prepare();
    OclFrontend::AddStringForArguments(CudaUtil::MarshalHostPointer(context));

    OclFrontend::Execute("clReleaseContext");

    return OclFrontend::GetExitCode();

}


