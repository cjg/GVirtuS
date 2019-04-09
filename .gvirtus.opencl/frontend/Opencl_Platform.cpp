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
clGetPlatformIDs(cl_uint num_entries,
                 cl_platform_id *platforms,
                 cl_uint *num_platforms){

    OpenclFrontend::Prepare();
    OpenclFrontend::AddVariableForArguments(num_entries);
    OpenclFrontend::AddVariableForArguments(platforms);
    OpenclFrontend::AddVariableForArguments(num_platforms);

    OpenclFrontend::Execute("clGetPlatformIDs");

    if(OpenclFrontend::Success()){
       cout << "Success " << endl;
       cl_uint *tmp_num_platform;
       cl_platform_id *tmp_platform;
       tmp_num_platform = OpenclFrontend::GetOutputHostPointer<cl_uint>();
       if (tmp_num_platform != NULL)
         *num_platforms= *tmp_num_platform;

       tmp_platform = (OpenclFrontend::GetOutputHostPointer<cl_platform_id>());
       if (tmp_platform !=NULL)
       *platforms=*tmp_platform;

    }

    return  OpenclFrontend::GetExitCode();

}

extern "C" CL_API_ENTRY cl_int CL_API_CALL
clGetPlatformInfo(cl_platform_id platform,
  	cl_platform_info param_name,
  	size_t param_value_size,
  	void *param_value,
  	size_t *param_value_size_ret){

    OpenclFrontend::Prepare();
    OpenclFrontend::AddHostPointerForArguments<cl_platform_id>(&platform); //TEST
    OpenclFrontend::AddVariableForArguments<cl_platform_info>(param_name);
    OpenclFrontend::AddVariableForArguments<size_t>(param_value_size);

    OpenclFrontend::Execute("clGetPlatformInfo");
    if(OpenclFrontend::Success()){
        param_value_size_ret=OpenclFrontend::GetOutputHostPointer<size_t>();
        char* tmp_param_value=OpenclFrontend::GetOutputString();
        memcpy(param_value,tmp_param_value,strlen(tmp_param_value)+1);
    }
    return OpenclFrontend::GetExitCode();

}
