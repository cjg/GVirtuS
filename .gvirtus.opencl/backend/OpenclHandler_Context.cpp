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
#include <iostream>
#include <cstdio>
#include <string>
#include <stdint.h>

using namespace std;

OPENCL_ROUTINE_HANDLER(CreateContext) {
    cout<<"input buffer size"<< input_buffer->GetBufferSize()<<endl;
    cout<<"esegui clceateContext"<<endl;
    cl_context_properties *properties=input_buffer->AssignAll<cl_context_properties>();
        cout<<"estrai properties"<<endl;

    cl_uint num_devices=input_buffer->Get<cl_uint>();
            cout<<"estrai num devices="<<num_devices<<endl;

    cl_device_id *devices=input_buffer->AssignAll<cl_device_id>();
    cout<<"estrai devices id:"<< (intptr_t)devices<<endl;

    cl_int exit_code=0;
//cl_context context=clCreateContext(properties, 1, &device, NULL, NULL, &error);
    cl_context ret_context = (clCreateContext(properties,num_devices,devices,NULL,NULL,&exit_code));
    cout<<"clceateContext eseguita"<<endl;
    Buffer *out = new Buffer();
    out->Add(&ret_context);

    return new Result(exit_code, out);
}

OPENCL_ROUTINE_HANDLER(GetContextInfo) {

    cl_context *context=input_buffer->Assign<cl_context>();
    cl_context_info param_name=input_buffer->Get<cl_context_info>();
    size_t param_value_size=input_buffer->Get<size_t>();
    char *param_value=input_buffer->AssignAll<char>();
    size_t *param_value_size_ret=input_buffer->Assign<size_t>();
    size_t tmp_param_value_size_ret=0;
    if (param_value_size_ret == NULL ){
        param_value_size_ret=&tmp_param_value_size_ret;
    }

    cl_int exit_code=clGetContextInfo(*context,param_name,param_value_size,param_value,param_value_size_ret);

    Buffer *out=new Buffer();
    out->Add(param_value_size_ret);
    out->Add(param_value,*param_value_size_ret);

    return new Result(exit_code,out);
    
}

OPENCL_ROUTINE_HANDLER(ReleaseContext){
    cl_context context = (cl_context)OpenclUtil::UnmarshalPointer(input_buffer->AssignString());

    cl_int exit_code = clReleaseContext(context);

    return new Result(exit_code);
}
