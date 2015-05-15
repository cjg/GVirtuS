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

#include "OpenclHandler.h"
#include "OpenclUtil.h"
#include <iostream>
#include <cstdio>
#include <string>
using namespace std;

OPENCL_ROUTINE_HANDLER(CreateProgramWithSource) {

    cl_context context= (cl_context)OpenclUtil::UnmarshalPointer(input_buffer->AssignString());
    cl_uint count=input_buffer->Get<cl_uint>();
    const size_t *lengths=input_buffer->AssignAll<const size_t>();
    char **strings;
    strings=(char **)malloc( sizeof(char *) * count);
    if (lengths != NULL){
        for (cl_uint i=0;i<count;i++){
            strings[i]=(char*)malloc(sizeof(char) * lengths[i]);
            memcpy(strings[i],input_buffer->Assign<char>(),lengths[i]);
            strings[i][lengths[i]]='\0';
        }
    }else{
         for (cl_uint i=0;i<count;i++){
            char *tmp_string=input_buffer->AssignString();
            int length=strlen(tmp_string);
            strings[i]=(char *)malloc(sizeof(char) * length+1);
            memcpy(strings[i],input_buffer->Assign<char>(),length+1);
        }
    }

    cl_int errcode_ret=0;

    cl_program program=clCreateProgramWithSource(context,count,(const char**)strings,lengths,&errcode_ret);

    Buffer *out=new Buffer();
    out->AddString(OpenclUtil::MarshalHostPointer(program));

    return new Result(errcode_ret, out);

}
OPENCL_ROUTINE_HANDLER(BuildProgram) {
    cl_program * program = input_buffer->Assign<cl_program>();
    cl_uint num_devices = input_buffer->Get<cl_uint>();
    cl_device_id *device_list = input_buffer->AssignAll<cl_device_id>();
    const char * options = input_buffer->AssignString();

    cl_int errcode_ret = clBuildProgram(*program,num_devices,device_list,options,NULL,NULL);

    return new Result(errcode_ret);
}

OPENCL_ROUTINE_HANDLER(ReleaseProgram){
    cl_program program = (cl_program)OpenclUtil::UnmarshalPointer(input_buffer->AssignString());

    cl_int exit_code = clReleaseProgram(program);

    return new Result(exit_code);
}