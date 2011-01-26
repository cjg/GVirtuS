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

#include "Ocl.h"

using namespace std;

extern "C" cl_kernel clCreateKernel ( 	cl_program  program,
  	const char *kernel_name,
  	cl_int *errcode_ret)  {

    OclFrontend::Prepare();
    OclFrontend::AddHostPointerForArguments<cl_program>(&program);
    OclFrontend::AddStringForArguments(kernel_name);

    OclFrontend::Execute("clCreateKernel");

    cl_kernel kernel;
    if (OclFrontend::Success()){
        if (errcode_ret != NULL){
            *errcode_ret = OclFrontend::GetExitCode();
        }
        kernel = *(OclFrontend::GetOutputHostPointer<cl_kernel>());
    }
    return kernel;
}

extern "C"  cl_int clSetKernelArg (cl_kernel kernel,
  	cl_uint arg_index,
  	size_t arg_size,
  	const void *arg_value) {

    OclFrontend::Prepare();
    OclFrontend::AddHostPointerForArguments<cl_kernel>(&kernel);
    OclFrontend::AddVariableForArguments<cl_uint>(arg_index);
    OclFrontend::AddVariableForArguments<size_t>(arg_size);
    OclFrontend::AddHostPointerForArguments((char *)arg_value,arg_size);

    OclFrontend::Execute("clSetKernelArg");

    return OclFrontend::GetExitCode();
}

extern "C" cl_int clReleaseKernel (cl_kernel kernel) {
    OclFrontend::Prepare();
    OclFrontend::AddStringForArguments(CudaUtil::MarshalHostPointer(kernel));

    OclFrontend::Execute("clReleaseKernel");

    return OclFrontend::GetExitCode();
}

extern "C" cl_int clGetKernelWorkGroupInfo (cl_kernel kernel,
  	cl_device_id device,
  	cl_kernel_work_group_info param_name,
  	size_t param_value_size,
  	void *param_value,
  	size_t *param_value_size_ret)  {

    OclFrontend::Prepare();
    OclFrontend::AddStringForArguments(CudaUtil::MarshalHostPointer(kernel));
    OclFrontend::AddStringForArguments(CudaUtil::MarshalHostPointer(device));
    OclFrontend::AddVariableForArguments(param_name);
    OclFrontend::AddVariableForArguments(param_value_size);
    OclFrontend::AddHostPointerForArguments((char*)param_value,param_value_size);

    OclFrontend::Execute("clGetKernelWorkGroupInfo");

    if (OclFrontend::Success()){
        size_t * tmp_param_value_size_ret=OclFrontend::GetOutputHostPointer<size_t>();
        if (param_value_size_ret != NULL){
            *param_value_size_ret = *tmp_param_value_size_ret;
        }
        memcpy(param_value,OclFrontend::GetOutputHostPointer<char>(),*tmp_param_value_size_ret);
    }
    return OclFrontend::GetExitCode();
    
}


extern "C"  cl_int clCreateKernelsInProgram ( 	cl_program  program,
  	cl_uint num_kernels,
  	cl_kernel *kernels,
  	cl_uint *num_kernels_ret) {
    cerr << "*** Error: clCreateKernelsInProgram not yet implemented!" << endl;
    return 0;
}

extern "C"  cl_int clRetainKernel (cl_kernel kernel) {
    cerr << "*** Error: clRetainKernel not yet implemented!" << endl;
    return 0;
}





extern "C"  cl_int clGetKernelInfo (cl_kernel kernel,
  	cl_kernel_info param_name,
  	size_t param_value_size,
  	void *param_value,
  	size_t *param_value_size_ret) {
    cerr << "*** Error: clGetKernelInfo not yet implemented!" << endl;
    return 0;
}



