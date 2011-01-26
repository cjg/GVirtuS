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
#include "CudaUtil.h"

OCL_ROUTINE_HANDLER(CreateKernel){
    cl_program * program = input_buffer->Assign<cl_program>();
    const char * kernel_name = input_buffer->AssignString();
    cl_int errcode_ret = 0;

    cl_kernel kernel = clCreateKernel(*program,kernel_name,&errcode_ret);

    Buffer *out = new Buffer();

    out->Add(&kernel);

    return new Result(errcode_ret,out);
}

OCL_ROUTINE_HANDLER(SetKernelArg){
 
    cl_kernel *kernel = input_buffer->Assign<cl_kernel>();
    cl_uint arg_index= input_buffer->Get<cl_uint>();
    size_t arg_size = input_buffer->Get<size_t>();
    void *arg_value = (void*)input_buffer->AssignAll<char>();

    cl_int exit_code = clSetKernelArg(*kernel,arg_index,arg_size,arg_value);

    return new Result(exit_code);

}

OCL_ROUTINE_HANDLER(ReleaseKernel){

    cl_kernel kernel = (cl_kernel)CudaUtil::UnmarshalPointer(input_buffer->AssignString());

    cl_int exit_code = clReleaseKernel(kernel);

    return new Result(exit_code);

}

OCL_ROUTINE_HANDLER(GetKernelWorkGroupInfo){
    
    cl_kernel kernel = (cl_kernel)CudaUtil::UnmarshalPointer(input_buffer->AssignString());
    cl_device_id device = (cl_device_id)CudaUtil::UnmarshalPointer(input_buffer->AssignString());
    cl_kernel_work_group_info param_name = input_buffer->Get<cl_kernel_work_group_info>();
    size_t param_value_size = input_buffer->Get<size_t>();
    void *param_value = input_buffer->AssignAll<char>();
    size_t param_value_size_ret = 0;

    cl_int exit_code = clGetKernelWorkGroupInfo(kernel,device,param_name,param_value_size,param_value,&param_value_size_ret);

    Buffer *out = new Buffer();
    out->Add(&param_value_size_ret);
    out->Add((char*)param_value,param_value_size_ret);
    return new Result(exit_code,out);
    
}