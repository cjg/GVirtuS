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

OPENCL_ROUTINE_HANDLER(CreateBuffer) {

    cl_context context = (cl_context)OpenclUtil::UnmarshalPointer(input_buffer->AssignString());
    cl_mem_flags flags=input_buffer->Get<cl_mem_flags>();
    size_t size=input_buffer->Get<size_t>();
    void *host=input_buffer->AssignAll<char>();
    cl_int errcode_ret=0;
    cl_mem mem;
    if ((flags & 8 /*1000*/)  != 0){
        mem=clCreateBuffer(context,flags,size,pThis->RegisterPointer(host,size),&errcode_ret);
    }else{
        mem=clCreateBuffer(context,flags,size,host,&errcode_ret);
    }

    Buffer *out=new Buffer();
    out->AddString(OpenclUtil::MarshalHostPointer(mem));

    return new Result(errcode_ret, out);
}

OPENCL_ROUTINE_HANDLER(ReleaseMemObject){

    cl_mem *memObj = input_buffer->Assign<cl_mem>();

    cl_int exit_code = clReleaseMemObject(*memObj);

    return new Result(exit_code);
}