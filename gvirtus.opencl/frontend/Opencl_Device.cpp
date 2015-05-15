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

#include "Opencl_gv.h"
#include <iostream>
#include <cstdio>
#include <string>

using namespace std;

extern "C" CL_API_ENTRY cl_int CL_API_CALL
clGetDeviceIDs(cl_platform_id platform,
               cl_device_type device_type,
               cl_uint num_entries,
               cl_device_id *devices,
               cl_uint *num_devices){

    OpenclFrontend::Prepare();

    OpenclFrontend::AddHostPointerForArguments<cl_platform_id>(&platform);
    OpenclFrontend::AddVariableForArguments<cl_device_type>(device_type);
    OpenclFrontend::AddVariableForArguments<cl_uint>(num_entries);
    OpenclFrontend::AddHostPointerForArguments<cl_device_id>(devices,num_entries);
    OpenclFrontend::AddHostPointerForArguments<cl_uint>(num_devices);

    OpenclFrontend::Execute("clGetDeviceIDs");
    if(OpenclFrontend::Success()){
          cl_uint *tmp_num_devices;
        tmp_num_devices = OpenclFrontend::GetOutputHostPointer<cl_uint>();
        if (tmp_num_devices != NULL)
            *num_devices=*tmp_num_devices;

        cl_device_id *tmp_devices;
        tmp_devices = OpenclFrontend::GetOutputHostPointer<cl_device_id>();
        if (tmp_devices != NULL){
            memmove(devices,tmp_devices,sizeof(cl_device_id)*num_entries);
        }


    }
    return OpenclFrontend::GetExitCode();
}

