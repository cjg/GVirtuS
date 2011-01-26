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

#include "OclHandler.h"

OCL_ROUTINE_HANDLER(CreateCommandQueue) {
   
    cl_context *context=input_buffer->Assign<cl_context>();
    cl_device_id *device=input_buffer->Assign<cl_device_id>();
    cl_command_queue_properties properties=input_buffer->Get<cl_command_queue_properties>();
    cl_int errcode_ret=0;

    cl_command_queue command_queue=clCreateCommandQueue(*context,*device,properties,&errcode_ret);
    
    Buffer *out = new Buffer();
   
    out->Add(&command_queue);

    return new Result(errcode_ret, out); 
}

OCL_ROUTINE_HANDLER(ReleaseCommandQueue){

    cl_command_queue command_queue = (cl_command_queue)CudaUtil::UnmarshalPointer(input_buffer->AssignString());

    cl_int exit_code = clReleaseCommandQueue(command_queue);

    return new Result(exit_code);

}

OCL_ROUTINE_HANDLER(GetCommandQueueInfo){

    cl_command_queue command_queue = (cl_command_queue)CudaUtil::UnmarshalPointer(input_buffer->AssignString());
    cl_command_queue_info param_name = input_buffer->Get<cl_command_queue_info>();
    size_t param_value_size = input_buffer->Get<size_t>();
    void *param_value = input_buffer->AssignAll<char>();
    size_t param_value_size_ret = 0;

    cl_int exit_code = clGetCommandQueueInfo(command_queue,param_name,param_value_size,param_value,&param_value_size_ret);

    Buffer *out = new Buffer();
    out->Add(&param_value_size_ret);
    out->Add((char*)param_value,param_value_size_ret);
    return new Result(exit_code,out);

}