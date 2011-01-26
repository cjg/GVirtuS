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

extern "C" cl_int clEnqueueNDRangeKernel (cl_command_queue command_queue,
  	cl_kernel kernel,
  	cl_uint work_dim,
  	const size_t *global_work_offset,
  	const size_t *global_work_size,
  	const size_t *local_work_size,
  	cl_uint num_events_in_wait_list,
  	const cl_event *event_wait_list,
  	cl_event *event)  {

    OclFrontend::Prepare();
    OclFrontend::AddHostPointerForArguments<cl_command_queue>(&command_queue);
    OclFrontend::AddHostPointerForArguments<cl_kernel>(&kernel);
    OclFrontend::AddVariableForArguments<cl_uint>(work_dim);
    OclFrontend::AddHostPointerForArguments<const size_t>(global_work_offset,work_dim);
    OclFrontend::AddHostPointerForArguments<const size_t>(global_work_size,work_dim);
    OclFrontend::AddHostPointerForArguments<const size_t>(local_work_size,work_dim);
    OclFrontend::AddVariableForArguments<cl_uint>(num_events_in_wait_list);
    for (int i=0; i<num_events_in_wait_list; i++){
        OclFrontend::AddStringForArguments(CudaUtil::MarshalHostPointer(event_wait_list[i]));
    }
    if (event != NULL)
        OclFrontend::AddVariableForArguments(true);
     else
        OclFrontend::AddVariableForArguments(false);
    
    OclFrontend::Execute("clEnqueueNDRangeKernel");

    if(OclFrontend::Success()){
        if (event!=NULL){
            *event=(cl_event)(CudaUtil::UnmarshalPointer(OclFrontend::GetOutputString()));
        }
    }

    return OclFrontend::GetExitCode();
}

extern "C"  cl_int clEnqueueTask ( 	cl_command_queue command_queue,
  	cl_kernel kernel,
  	cl_uint num_events_in_wait_list,
  	const cl_event *event_wait_list,
  	cl_event *event) {
    cerr << "*** Error: clEnqueueTask not yet implemented!" << endl;
    return 0;
}

extern "C"  cl_int clEnqueueNativeKernel ( 	cl_command_queue command_queue,
  	void (*user_func)(void *),
  	void *args,
  	size_t cb_args,
  	cl_uint num_mem_objects,
  	const cl_mem *mem_list,
  	const void **args_mem_loc,
  	cl_uint num_events_in_wait_list,
  	const cl_event *event_wait_list,
  	cl_event *event) {
    cerr << "*** Error: clEnqueueNativeKernel not yet implemented!" << endl;
    return 0;
}
