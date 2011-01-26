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

extern "C" cl_program clCreateProgramWithSource (cl_context context,
  	cl_uint count,
  	const char **strings,
  	const size_t *lengths,
  	cl_int *errcode_ret) {

    OclFrontend::Prepare();
    OclFrontend::AddStringForArguments(CudaUtil::MarshalHostPointer(context));
    OclFrontend::AddVariableForArguments<cl_uint>(count);
    OclFrontend::AddHostPointerForArguments<const size_t>(lengths,count);
    if (lengths != NULL){
        for (int i=0;i<count;i++){
            OclFrontend::AddHostPointerForArguments(strings[i],lengths[i]);
        }
    }else{
        for (int i=0;i<count;i++){
            OclFrontend::AddStringForArguments(strings[i]);
        }
    }

    OclFrontend::Execute("clCreateProgramWithSource");

    cl_program program;
    if(OclFrontend::Success()){
        program = (cl_program)CudaUtil::UnmarshalPointer(OclFrontend::GetOutputString());
        if (errcode_ret != NULL){
            *errcode_ret=OclFrontend::GetExitCode();
        }
    }

    return program;
}

extern "C" cl_int clBuildProgram (cl_program program,
  	cl_uint num_devices,
  	const cl_device_id *device_list,
  	const char *options,
  	void (*pfn_notify)(cl_program program, void *user_data),
  	void *user_data) {

    OclFrontend::Prepare();
    OclFrontend::AddHostPointerForArguments<cl_program>(&program);
    OclFrontend::AddVariableForArguments<cl_uint>(num_devices);
    OclFrontend::AddHostPointerForArguments(device_list,num_devices);
    if (options == NULL)
        OclFrontend::AddStringForArguments("");
    else
        OclFrontend::AddStringForArguments(options);
    
    OclFrontend::Execute("clBuildProgram");
    
    return OclFrontend::GetExitCode();
}

extern "C" cl_int clReleaseProgram (cl_program program) {

    OclFrontend::Prepare();
    OclFrontend::AddStringForArguments(CudaUtil::MarshalHostPointer(program));

    OclFrontend::Execute("clReleaseProgram");

    return OclFrontend::GetExitCode();

}

extern "C" cl_int clGetProgramInfo ( 	cl_program program,
  	cl_program_info param_name,
  	size_t param_value_size,
  	void *param_value,
  	size_t *param_value_size_ret) {


    OclFrontend::Prepare();
    OclFrontend::AddHostPointerForArguments<cl_program>(&program);
    OclFrontend::AddVariableForArguments<cl_program_info>(param_name);
    OclFrontend::AddVariableForArguments<size_t>(param_value_size);
    OclFrontend::AddHostPointerForArguments((char*)param_value,param_value_size);
    OclFrontend::AddHostPointerForArguments<size_t>(param_value_size_ret);

    OclFrontend::Execute("clGetProgramInfo");

    if (OclFrontend::Success()){

        if (param_name != CL_PROGRAM_BINARIES){
            size_t *tmp_param_value_size_ret = OclFrontend::GetOutputHostPointer<size_t>();
            if (param_value_size_ret !=NULL)
                *param_value_size_ret=*tmp_param_value_size_ret;
            if (param_value!=NULL)
                memmove(param_value,OclFrontend::GetOutputHostPointer<char>(),*tmp_param_value_size_ret);
        }else{
            cl_uint num_devices = OclFrontend::GetOutputVariable<cl_uint>();
            for (int i=0; i< num_devices; i++){
               *(((char **)param_value)[i])=*(OclFrontend::GetOutputHostPointer<char>());
            }
        }
    }

    return OclFrontend::GetExitCode();


}

extern "C" cl_program clCreateProgramWithBinary ( 	cl_context context,
  	cl_uint num_devices,
  	const cl_device_id *device_list,
  	const size_t *lengths,
  	const unsigned char **binaries,
  	cl_int *binary_status,
  	cl_int *errcode_ret) {

    cerr << "*** Error: clCreateProgramWithBinary not yet implemented!" << endl;
    return 0;
}

extern "C" cl_int clRetainProgram (cl_program program) {

    cerr << "*** Error: clRetainProgram not yet implemented!" << endl;
    return 0;
}


extern "C" cl_int clUnloadCompiler (void) {

    cerr << "*** Error: clUnloadCompiler not yet implemented!" << endl;
    return 0;
}




extern "C" cl_int clGetProgramBuildInfo ( 	cl_program  program,
  	cl_device_id  device,
  	cl_program_build_info  param_name,
  	size_t  param_value_size,
  	void  *param_value,
  	size_t  *param_value_size_ret) {

    cerr << "*** Error: clGetProgramBuildInfo not yet implemented!" << endl;
    return 0;
}
