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
#include "OclFrontend.h"

using namespace std;


extern "C" cl_command_queue clCreateCommandQueue(cl_context context,
  	cl_device_id device,
  	cl_command_queue_properties properties,
  	cl_int *errcode_ret){

    OclFrontend::Prepare();

    OclFrontend::AddHostPointerForArguments<cl_context>(&context);
    OclFrontend::AddHostPointerForArguments<cl_device_id>(&device);
    OclFrontend::AddVariableForArguments<cl_command_queue_properties>(properties);

     OclFrontend::Execute("clCreateCommandQueue");

    cl_command_queue command_queue;
    if (OclFrontend::Success()){
        command_queue=*(OclFrontend::GetOutputHostPointer<cl_command_queue>());
        if (errcode_ret != NULL)
            *errcode_ret=OclFrontend::GetExitCode();
    }

     return command_queue;
}


cl_int clGetCommandQueueInfo( 	cl_command_queue command_queue,
  	cl_command_queue_info param_name,
  	size_t param_value_size,
  	void *param_value,
  	size_t *param_value_size_ret){

    OclFrontend::Prepare();
    OclFrontend::AddStringForArguments(CudaUtil::MarshalHostPointer(command_queue));
    OclFrontend::AddVariableForArguments(param_name);
    OclFrontend::AddVariableForArguments(param_value_size);
    OclFrontend::AddHostPointerForArguments((char*)param_value,param_value_size);

    OclFrontend::Execute("clGetCommandQueueInfo");

     if (OclFrontend::Success()){
        size_t * tmp_param_value_size_ret=OclFrontend::GetOutputHostPointer<size_t>();
        if (param_value_size_ret != NULL){
            *param_value_size_ret = *tmp_param_value_size_ret;
        }
        memcpy(param_value,OclFrontend::GetOutputHostPointer<char>(),*tmp_param_value_size_ret);
        
    }
    return OclFrontend::GetExitCode();

}

extern "C" cl_int clReleaseCommandQueue(cl_command_queue command_queue){
    OclFrontend::Prepare();
    OclFrontend::AddStringForArguments(CudaUtil::MarshalHostPointer(command_queue));

    OclFrontend::Execute("clReleaseCommandQueue");

    return OclFrontend::GetExitCode();


}

extern "C" cl_int clRetainCommandQueue(cl_command_queue command_queue){

    cerr << "*** Error: clRetainCommandQueue not yet implemented!" << endl;
    return 0;
}



extern "C" cl_int clSetCommandQueueProperty(cl_command_queue command_queue,
  	cl_command_queue_properties properties,
  	cl_bool enable,
  	cl_command_queue_properties *old_properties){

    cerr << "*** Error: clSetCommandQueueProperty not yet implemented!" << endl;
    return 0;
}

