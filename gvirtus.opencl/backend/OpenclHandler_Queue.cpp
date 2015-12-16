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
using namespace std;

OPENCL_ROUTINE_HANDLER(CreateCommandQueue) {

    cl_context *context=input_buffer->Assign<cl_context>();
    cl_device_id *device=input_buffer->Assign<cl_device_id>();
    cl_command_queue_properties properties=input_buffer->Get<cl_command_queue_properties>();
    cl_int errcode_ret=0;

    cl_command_queue command_queue=clCreateCommandQueue(*context,*device,properties,&errcode_ret);

    Buffer *out = new Buffer();

    out->Add(&command_queue);

    return new Result(errcode_ret, out);
}

OPENCL_ROUTINE_HANDLER(ReleaseCommandQueue){

    cl_command_queue command_queue = (cl_command_queue)OpenclUtil::UnmarshalPointer(input_buffer->AssignString());

    cl_int exit_code = clReleaseCommandQueue(command_queue);

    return new Result(exit_code);

}
