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

extern "C" cl_int clGetEventProfilingInfo ( 	cl_event event,
  	cl_profiling_info param_name,
  	size_t param_value_size,
  	void *param_value,
  	size_t *param_value_size_ret) {

    OclFrontend::Prepare();
    OclFrontend::AddStringForArguments(CudaUtil::MarshalHostPointer(event));
    OclFrontend::AddVariableForArguments(param_name);
    OclFrontend::AddVariableForArguments(param_value_size);
    OclFrontend::AddHostPointerForArguments((char*)param_value,param_value_size);

    OclFrontend::Execute("clGetEventProfilingInfo");

     if (OclFrontend::Success()){
        size_t * tmp_param_value_size_ret=OclFrontend::GetOutputHostPointer<size_t>();
        if (param_value_size_ret != NULL){
            *param_value_size_ret = *tmp_param_value_size_ret;
        }
        memcpy(param_value,OclFrontend::GetOutputHostPointer<char>(),*tmp_param_value_size_ret);

    }
    return OclFrontend::GetExitCode();


}
