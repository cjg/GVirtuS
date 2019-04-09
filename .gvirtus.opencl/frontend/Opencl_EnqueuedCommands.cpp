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
clEnqueueWriteBuffer (cl_command_queue command_queue,
  	cl_mem buffer,
  	cl_bool blocking_write,
  	size_t offset,
  	size_t cb,
  	const void *ptr,
  	cl_uint num_events_in_wait_list,
  	const cl_event *event_wait_list,
  	cl_event *event){

    OpenclFrontend::Prepare();
    OpenclFrontend::AddHostPointerForArguments<cl_command_queue>(&command_queue);
    OpenclFrontend::AddHostPointerForArguments<cl_mem>(&buffer);
    OpenclFrontend::AddVariableForArguments<cl_bool>(blocking_write);
    OpenclFrontend::AddVariableForArguments<size_t>(offset);
    OpenclFrontend::AddVariableForArguments<size_t>(cb);
    OpenclFrontend::AddHostPointerForArguments((char*)ptr,cb);
    OpenclFrontend::AddVariableForArguments<cl_uint>(num_events_in_wait_list);
    for (cl_uint i=0; i<num_events_in_wait_list; i++){
        OpenclFrontend::AddStringForArguments(OpenclUtil::MarshalHostPointer(event_wait_list[i]));
    }
    if (event != NULL)
        OpenclFrontend::AddVariableForArguments(true);
     else
        OpenclFrontend::AddVariableForArguments(false);


    OpenclFrontend::Execute("clEnqueueWriteBuffer");


    if (OpenclFrontend::Success()){
        if (event != NULL){
            *event = (cl_event)OpenclUtil::UnmarshalPointer(OpenclFrontend::GetOutputString());
        }

    }

    return OpenclFrontend::GetExitCode();

}
extern "C" CL_API_ENTRY cl_int CL_API_CALL
clEnqueueNDRangeKernel (cl_command_queue command_queue,
  	cl_kernel kernel,
  	cl_uint work_dim,
  	const size_t *global_work_offset,
  	const size_t *global_work_size,
  	const size_t *local_work_size,
  	cl_uint num_events_in_wait_list,
  	const cl_event *event_wait_list,
  	cl_event *event)  {

    OpenclFrontend::Prepare();
    OpenclFrontend::AddHostPointerForArguments<cl_command_queue>(&command_queue);
    OpenclFrontend::AddHostPointerForArguments<cl_kernel>(&kernel);
    OpenclFrontend::AddVariableForArguments<cl_uint>(work_dim);
    OpenclFrontend::AddHostPointerForArguments<const size_t>(global_work_offset,work_dim);
    OpenclFrontend::AddHostPointerForArguments<const size_t>(global_work_size,work_dim);
    OpenclFrontend::AddHostPointerForArguments<const size_t>(local_work_size,work_dim);
    OpenclFrontend::AddVariableForArguments<cl_uint>(num_events_in_wait_list);
    for (cl_uint i=0; i<num_events_in_wait_list; i++){
        OpenclFrontend::AddStringForArguments(OpenclUtil::MarshalHostPointer(event_wait_list[i]));
    }
    if (event != NULL)
        OpenclFrontend::AddVariableForArguments(true);
     else
        OpenclFrontend::AddVariableForArguments(false);

    OpenclFrontend::Execute("clEnqueueNDRangeKernel");

    if(OpenclFrontend::Success()){
        if (event!=NULL){
            *event=(cl_event)(OpenclUtil::UnmarshalPointer(OpenclFrontend::GetOutputString()));
        }
    }

    return OpenclFrontend::GetExitCode();
}

extern "C" CL_API_ENTRY cl_int CL_API_CALL
clEnqueueReadBuffer (cl_command_queue command_queue,
  	cl_mem buffer,
  	cl_bool blocking_read,
  	size_t offset,
  	size_t cb,
  	void *ptr,
  	cl_uint num_events_in_wait_list,
  	const cl_event *event_wait_list,
  	cl_event *event){


    OpenclFrontend::Prepare();
    OpenclFrontend::AddHostPointerForArguments<cl_command_queue>(&command_queue);
    OpenclFrontend::AddHostPointerForArguments<cl_mem>(&buffer);
    OpenclFrontend::AddVariableForArguments<cl_bool>(blocking_read);
    OpenclFrontend::AddVariableForArguments<size_t>(offset);
    OpenclFrontend::AddVariableForArguments<size_t>(cb);
    OpenclFrontend::AddHostPointerForArguments((char*)ptr,cb);
    OpenclFrontend::AddVariableForArguments<cl_uint>(num_events_in_wait_list);
    for (cl_uint i=0; i<num_events_in_wait_list; i++){
        OpenclFrontend::AddStringForArguments(OpenclUtil::MarshalHostPointer(event_wait_list[i]));
    }
    if (event != NULL)
        OpenclFrontend::AddVariableForArguments(true);
     else
        OpenclFrontend::AddVariableForArguments(false);


    OpenclFrontend::Execute("clEnqueueReadBuffer");

    if (OpenclFrontend::Success()){
        if (event != NULL){
            *event = (cl_event)OpenclUtil::UnmarshalPointer(OpenclFrontend::GetOutputString());

        }
        if (ptr!=NULL){
            memcpy(ptr,OpenclFrontend::GetOutputHostPointer<char>(),cb);
        }

    }

    return OpenclFrontend::GetExitCode();

}

extern "C" CL_API_ENTRY cl_int CL_API_CALL
clEnqueueCopyBuffer (cl_command_queue command_queue,
  	cl_mem src_buffer,
  	cl_mem dst_buffer,
  	size_t src_offset,
  	size_t dst_offset,
  	size_t cb,
  	cl_uint num_events_in_wait_list,
  	const cl_event *event_wait_list,
  	cl_event *event){

    OpenclFrontend::Prepare();
    
    OpenclFrontend::AddStringForArguments(OpenclUtil::MarshalHostPointer(command_queue));

    OpenclFrontend::AddStringForArguments(OpenclUtil::MarshalHostPointer(src_buffer));
    OpenclFrontend::AddStringForArguments(OpenclUtil::MarshalHostPointer(dst_buffer));
    OpenclFrontend::AddVariableForArguments<size_t>(src_offset);
    OpenclFrontend::AddVariableForArguments<size_t>(dst_offset);
    OpenclFrontend::AddVariableForArguments<size_t>(cb);
    OpenclFrontend::AddVariableForArguments<cl_uint>(num_events_in_wait_list);
    for (unsigned i=0; i<num_events_in_wait_list; i++){
        OpenclFrontend::AddStringForArguments(OpenclUtil::MarshalHostPointer(event_wait_list[i]));
    }
     if (event != NULL)
        OpenclFrontend::AddVariableForArguments(true);
     else
        OpenclFrontend::AddVariableForArguments(false);
    

    OpenclFrontend::Execute("clEnqueueCopyBuffer");

    if (OpenclFrontend::GetExitCode()){
        if (event!=NULL){
            *event=*(OpenclFrontend::GetOutputHostPointer<cl_event>());
        }
    }
    return OpenclFrontend::GetExitCode();

}
