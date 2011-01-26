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

extern "C" cl_int clGetDeviceIDs(cl_platform_id platform,
  	cl_device_type device_type,
  	cl_uint num_entries,
  	cl_device_id *devices,
  	cl_uint *num_devices){

    OclFrontend::Prepare();

    OclFrontend::AddHostPointerForArguments<cl_platform_id>(&platform);
    OclFrontend::AddVariableForArguments<cl_device_type>(device_type);
    OclFrontend::AddVariableForArguments<cl_uint>(num_entries);
    OclFrontend::AddHostPointerForArguments<cl_device_id>(devices,num_entries);
    OclFrontend::AddHostPointerForArguments<cl_uint>(num_devices);

    OclFrontend::Execute("clGetDeviceIDs");
    if(OclFrontend::Success()){
          cl_uint *tmp_num_devices;
        tmp_num_devices = OclFrontend::GetOutputHostPointer<cl_uint>();
        if (tmp_num_devices != NULL)
            *num_devices=*tmp_num_devices;
       
        cl_device_id *tmp_devices;
        tmp_devices = OclFrontend::GetOutputHostPointer<cl_device_id>();
        if (tmp_devices != NULL){
            memmove(devices,tmp_devices,sizeof(cl_device_id)*num_entries);
        }

      
    }
    return OclFrontend::GetExitCode();
}

extern "C" cl_int clGetDeviceInfo(cl_device_id device,cl_device_info param_name,
        size_t param_value_size,void *param_value,size_t *param_value_size_ret){

    OclFrontend::Prepare();
    OclFrontend::AddHostPointerForArguments<cl_device_id>(&device);
    OclFrontend::AddVariableForArguments<cl_device_info>(param_name);
    OclFrontend::AddVariableForArguments<size_t>(param_value_size);


    OclFrontend::Execute("clGetDeviceInfo");

    if (OclFrontend::Success()){
         size_t tmp_param_value_size_ret=OclFrontend::GetOutputVariable<size_t>();
         param_value_size_ret=&tmp_param_value_size_ret;

         char* tmp_param_value=NULL;
         tmp_param_value=(OclFrontend::GetOutputHostPointer<char>());
         memcpy(param_value,tmp_param_value,*param_value_size_ret);
     }
    return OclFrontend::GetExitCode();

}