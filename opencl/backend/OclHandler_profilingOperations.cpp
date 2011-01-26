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

#include "OclHandler.h"


OCL_ROUTINE_HANDLER(GetEventProfilingInfo){

    cl_event event = (cl_event)CudaUtil::UnmarshalPointer(input_buffer->AssignString());
    cl_profiling_info param_name = input_buffer->Get<cl_profiling_info>();
    size_t param_value_size = input_buffer->Get<size_t>();
    void *param_value = input_buffer->AssignAll<char>();
    size_t param_value_size_ret = 0;

    cl_int exit_code = clGetEventProfilingInfo(event,param_name,param_value_size,param_value,&param_value_size_ret);
    
    Buffer *out = new Buffer();
    out->Add(&param_value_size_ret);
    out->Add((char*)param_value,param_value_size_ret);
    return new Result(exit_code,out);

}