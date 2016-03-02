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

OPENCL_ROUTINE_HANDLER(GetDeviceIDs) {
    cout<<"input buffer size"<< input_buffer->GetBufferSize()<<endl;
    cl_platform_id *platform=input_buffer->Assign<cl_platform_id>();
    cl_device_type device_type=input_buffer->Get<cl_device_type>();
    cl_uint num_entries=input_buffer->Get<cl_uint>();
    cl_device_id *devices=input_buffer->AssignAll<cl_device_id>();
    cl_uint *num_devices=input_buffer->Assign<cl_uint>();

    cl_int exit_code=clGetDeviceIDs(*platform,device_type,num_entries,devices,num_devices);

    Buffer *out=new Buffer();

    out->Add<cl_uint>(num_devices);
    out->Add<cl_device_id>(devices,num_entries);
    return new Result(exit_code, out);
 
}

