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

OPENCL_ROUTINE_HANDLER(EnqueueWriteBuffer) {

    cl_command_queue *command_queue= input_buffer->Assign<cl_command_queue>();
    cl_mem *buffer = input_buffer->Assign<cl_mem>();
    cl_bool blocking_write = input_buffer->Get<cl_bool>();
    size_t offset = input_buffer->Get<size_t>();
    size_t cb = input_buffer->Get<size_t>();
    void *ptr = (void*)input_buffer->AssignAll<char>();
    cl_uint num_events_in_wait_list = input_buffer->Get<cl_uint>();
    cl_event *event_wait_list = new cl_event[num_events_in_wait_list];
    for (cl_uint i=0; i<num_events_in_wait_list;i++){
        event_wait_list[i] = (cl_event)OpenclUtil::UnmarshalPointer(input_buffer->AssignString());
    }
    bool catch_event = input_buffer->Get<bool>();
 cl_event *event;
 cl_event tmp_event;
 if (catch_event)
     event = &tmp_event;
 else
     event = NULL;

    /* FIXME: There is a problem with non-blocking read/write,
     * so read/write will be always blocking */
   blocking_write = CL_TRUE;

    cl_int exit_code = clEnqueueWriteBuffer(*command_queue,*buffer,blocking_write,
            offset,cb,ptr,num_events_in_wait_list,event_wait_list,event);

    Buffer *out = new Buffer();

    if (event != NULL)
        out->AddString(OpenclUtil::MarshalHostPointer(*event));

    return new Result(exit_code,out);
}

OPENCL_ROUTINE_HANDLER(EnqueueNDRangeKernel){
    cl_command_queue *command_queue = input_buffer->Assign<cl_command_queue>();
    cl_kernel *kernel = input_buffer->Assign<cl_kernel>();
    cl_uint work_dim = input_buffer->Get<cl_uint>();
    const size_t *global_work_offset = input_buffer->AssignAll<const size_t>();
    const size_t *global_work_size = input_buffer->AssignAll<const  size_t>();
    const size_t *local_work_size = input_buffer->AssignAll<const  size_t>();
    cl_uint num_events_in_wait_list = input_buffer->Get<cl_uint>();
     cl_event *event_wait_list = new cl_event[num_events_in_wait_list];
    for (cl_uint i=0; i<num_events_in_wait_list;i++){
        event_wait_list[i] = (cl_event)OpenclUtil::UnmarshalPointer(input_buffer->AssignString());
    }
bool catch_event = input_buffer->Get<bool>();
 cl_event *event;
 cl_event tmp_event;
 if (catch_event)
     event = &tmp_event;
 else
     event = NULL;

    cl_int exit_code = clEnqueueNDRangeKernel(*command_queue,*kernel,work_dim,
            global_work_offset,global_work_size,local_work_size,num_events_in_wait_list,
            NULL,event);


    Buffer *out = new Buffer();
    if (event != NULL)
        out->AddString(OpenclUtil::MarshalHostPointer(*event));

    return new Result(exit_code,out);


}

OPENCL_ROUTINE_HANDLER(EnqueueReadBuffer){
    
cl_command_queue *command_queue= input_buffer->Assign<cl_command_queue>();
    cl_mem *buffer = input_buffer->Assign<cl_mem>();
    cl_bool blocking_read = input_buffer->Get<cl_bool>();
    size_t offset = input_buffer->Get<size_t>();
    size_t cb = input_buffer->Get<size_t>();
    void *ptr = (void*)input_buffer->AssignAll<char>();
    cl_uint num_events_in_wait_list = input_buffer->Get<cl_uint>();
    cl_event *event_wait_list = new cl_event[num_events_in_wait_list];
    for (cl_uint i=0; i<num_events_in_wait_list;i++){
        event_wait_list[i] = (cl_event)OpenclUtil::UnmarshalPointer(input_buffer->AssignString());
    }
    bool catch_event = input_buffer->Get<bool>();
     cl_event *event;
 cl_event tmp_event;
 if (catch_event)
     event = &tmp_event;
 else
     event = NULL;


    /* FIXME: There is a problem with non-blocking read/write, so read/write will be always blocking */
    blocking_read = CL_TRUE;

    cl_int exit_code = clEnqueueReadBuffer(*command_queue,*buffer,blocking_read,
            offset,cb,ptr,num_events_in_wait_list,event_wait_list,event);

    Buffer *out = new Buffer();

    if (event != NULL)
        out->AddString(OpenclUtil::MarshalHostPointer(*event));
    out->Add((char *)ptr,cb);

    return new Result(exit_code,out);

}

OPENCL_ROUTINE_HANDLER(EnqueueCopyBuffer) {

 cl_command_queue command_queue = (cl_command_queue)OpenclUtil::UnmarshalPointer(input_buffer->AssignString());
 cl_mem src_buffer = (cl_mem)OpenclUtil::UnmarshalPointer(input_buffer->AssignString());
 cl_mem dst_buffer = (cl_mem)OpenclUtil::UnmarshalPointer(input_buffer->AssignString());

 size_t src_offset= input_buffer->Get<size_t>();
 size_t dst_offset= input_buffer->Get<size_t>();
 size_t cb = input_buffer->Get<size_t>();
 cl_uint num_events_in_wait_list = input_buffer->Get<cl_uint>();
 cl_event *event_wait_list = new cl_event[num_events_in_wait_list];
    for (unsigned i=0; i<num_events_in_wait_list;i++){
        event_wait_list[i] = (cl_event)OpenclUtil::UnmarshalPointer(input_buffer->AssignString());
    }
 bool catch_event = input_buffer->Get<bool>();
 cl_event *event;
 cl_event tmp_event;
 if (catch_event)
     event = &tmp_event;
 else
     event = NULL;

 cl_int errcode_ret = clEnqueueCopyBuffer(command_queue,src_buffer,dst_buffer,src_offset,dst_offset,cb,num_events_in_wait_list,event_wait_list,event);

 
 Buffer *out = new Buffer();
 if (event!=NULL)
    out->AddString(OpenclUtil::MarshalHostPointer(*event));

 return new Result(errcode_ret,out);
}
