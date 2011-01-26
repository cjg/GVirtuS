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

#include <stdlib.h>


#include <string.h>

#include "OclHandler.h"
#include "CudaUtil.h"

OCL_ROUTINE_HANDLER(CreateProgramWithSource) {

    cl_context context= (cl_context)CudaUtil::UnmarshalPointer(input_buffer->AssignString());
    cl_uint count=input_buffer->Get<cl_uint>();
    const size_t *lengths=input_buffer->AssignAll<const size_t>();
    char **strings;
    strings=(char **)malloc( sizeof(char *) * count);
    if (lengths != NULL){
        for (int i=0;i<count;i++){
            strings[i]=(char*)malloc(sizeof(char) * lengths[i]);
            memcpy(strings[i],input_buffer->Assign<char>(),lengths[i]);
            strings[i][lengths[i]]='\0';
        }
    }else{
         for (int i=0;i<count;i++){
            char *tmp_string=input_buffer->AssignString();
            int length=strlen(tmp_string);
            strings[i]=(char *)malloc(sizeof(char) * length+1);
            memcpy(strings[i],input_buffer->Assign<char>(),length+1);
        }
    }

    cl_int errcode_ret=0;

    cl_program program=clCreateProgramWithSource(context,count,(const char**)strings,lengths,&errcode_ret);

    Buffer *out=new Buffer();
    out->AddString(CudaUtil::MarshalHostPointer(program));

    return new Result(errcode_ret, out); 
    
}

OCL_ROUTINE_HANDLER(BuildProgram) {
  
    cl_program * program = input_buffer->Assign<cl_program>();
    cl_uint num_devices = input_buffer->Get<cl_uint>();
    cl_device_id *device_list = input_buffer->AssignAll<cl_device_id>();
    const char * options = input_buffer->AssignString();

    cl_int errcode_ret = clBuildProgram(*program,num_devices,device_list,options,NULL,NULL);

    return new Result(errcode_ret);

}

OCL_ROUTINE_HANDLER(ReleaseProgram){
    cl_program program = (cl_program)CudaUtil::UnmarshalPointer(input_buffer->AssignString());

    cl_int exit_code = clReleaseProgram(program);

    return new Result(exit_code);
}

OCL_ROUTINE_HANDLER(GetProgramInfo){
    cl_program *program=input_buffer->Assign<cl_program>();
    cl_program_info param_name=input_buffer->Get<cl_program_info>();
    size_t param_value_size=input_buffer->Get<size_t>();
    char *param_value=input_buffer->AssignAll<char>();
    size_t *param_value_size_ret=input_buffer->Assign<size_t>();
    size_t tmp_param_value_size_ret=0;
    if (param_value_size_ret == NULL ){
        param_value_size_ret=&tmp_param_value_size_ret;
    }

    cl_uint num_devices;
    size_t* binary_sizes;
    cl_int exit_code;
    char** ptx_code;
    if (param_name == CL_PROGRAM_BINARIES){

        clGetProgramInfo(*program, CL_PROGRAM_NUM_DEVICES, sizeof(cl_uint), &num_devices, NULL);

        binary_sizes = (size_t*)malloc(num_devices * sizeof(size_t));
        clGetProgramInfo(*program, CL_PROGRAM_BINARY_SIZES, num_devices * sizeof(size_t), binary_sizes, NULL);

        ptx_code = (char**)malloc(num_devices * sizeof(char*));
        for( unsigned int i=0; i<num_devices; ++i)
        {
            ptx_code[i] = (char*)malloc(binary_sizes[i]);
        }

            exit_code=clGetProgramInfo(*program,param_name,param_value_size,ptx_code,param_value_size_ret);

    }else{

        exit_code=clGetProgramInfo(*program,param_name,param_value_size,param_value,param_value_size_ret);
    }
    Buffer *out=new Buffer();
    
    if (param_name != CL_PROGRAM_BINARIES){
        out->Add(param_value_size_ret);
        out->Add(param_value,*param_value_size_ret);
    }else{
        out->Add(num_devices);
        for (int i=0; i<num_devices; i++){
            out->Add(ptx_code[i],binary_sizes[i]);
        }
    }

    return new Result(exit_code,out);


}