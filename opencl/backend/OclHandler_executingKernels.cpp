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

OCL_ROUTINE_HANDLER(EnqueueNDRangeKernel){
    cl_command_queue *command_queue = input_buffer->Assign<cl_command_queue>();
    cl_kernel *kernel = input_buffer->Assign<cl_kernel>();
    cl_uint work_dim = input_buffer->Get<cl_uint>();
    const size_t *global_work_offset = input_buffer->AssignAll<const size_t>();
    const size_t *global_work_size = input_buffer->AssignAll<const  size_t>();
    const size_t *local_work_size = input_buffer->AssignAll<const  size_t>();
    cl_uint num_events_in_wait_list = input_buffer->Get<cl_uint>();
     cl_event *event_wait_list = new cl_event[num_events_in_wait_list];
    for (int i=0; i<num_events_in_wait_list;i++){
        event_wait_list[i] = (cl_event)CudaUtil::UnmarshalPointer(input_buffer->AssignString());
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
        out->AddString(CudaUtil::MarshalHostPointer(*event));
    
    return new Result((cudaError_t)exit_code,out);
            

}