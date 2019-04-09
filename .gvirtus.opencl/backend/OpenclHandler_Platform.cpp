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

OPENCL_ROUTINE_HANDLER(GetPlatformIDs) {
    cout<<"input buffer size"<< input_buffer->GetBufferSize()<<endl;

    cl_int num_entries = input_buffer->Get<cl_int> ();
    cout<< "num_entries =" << num_entries << endl;

    cl_platform_id *platforms = input_buffer->Assign<cl_platform_id> ();
    cout<< "platfom ID =" << (intptr_t)platforms << endl;

    cl_uint *num_platforms = input_buffer->Assign<cl_uint> ();
    cout<< "num platfoms =" << *num_platforms << endl;

    cl_uint exit_code = clGetPlatformIDs(num_entries, platforms, num_platforms);
    cout<< "exit_code =" << exit_code << endl;

    Buffer *out = new Buffer();
    out->Add(num_platforms);
    out->Add(platforms);

    return new Result(exit_code, out);
 
}

OPENCL_ROUTINE_HANDLER(GetPlatformInfo) {

    cl_platform_id *platform=input_buffer->Assign<cl_platform_id>();
    cl_platform_info param_name=input_buffer->Get<cl_platform_info>();
    size_t param_value_size=input_buffer->Get<size_t>();
    char * param_value=new char[param_value_size];
    size_t param_value_size_ret=0;

    cl_int exit_code = clGetPlatformInfo(*platform,param_name,param_value_size,param_value,&param_value_size_ret);

    Buffer *out = new Buffer();
    out->Add(&param_value_size_ret);
    out->AddString(param_value);
    return new Result(exit_code, out); 
}
