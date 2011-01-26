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

#include <signal.h>

#include "OclHandler.h"
#include "CudaUtil.h"

OCL_ROUTINE_HANDLER(GetSupportedImageFormats) {
    
    
    cl_context *context=input_buffer->Assign<cl_context>();
    cl_mem_flags flags=input_buffer->Get<cl_mem_flags>();
    cl_mem_object_type image_type=input_buffer->Get<cl_mem_object_type>();
    cl_uint num_entries=input_buffer->Get<cl_uint>();
    cl_image_format *image_formats=input_buffer->AssignAll<cl_image_format>();
    cl_uint *num_image_formats=input_buffer->Assign<cl_uint>();
    cl_int exit_code=0;

    exit_code=clGetSupportedImageFormats (*context,flags,image_type,num_entries,image_formats,num_image_formats);

    Buffer *out = new Buffer();
    out->Add(num_image_formats);
    out->Add(image_formats,num_entries);

    return new Result(exit_code, out); 
}

OCL_ROUTINE_HANDLER(CreateBuffer) {

    cl_context context = (cl_context)CudaUtil::UnmarshalPointer(input_buffer->AssignString());
    cl_mem_flags flags=input_buffer->Get<cl_mem_flags>();
    size_t size=input_buffer->Get<size_t>();
    void *host=input_buffer->AssignAll<char>();
    cl_int errcode_ret=0;
    cl_mem mem;
    if ((flags & 8 /*1000*/)  != 0){
        mem=clCreateBuffer(context,flags,size,pThis->RegisterPointer(host,size),&errcode_ret);
    }else{
        mem=clCreateBuffer(context,flags,size,host,&errcode_ret);
    }

    Buffer *out=new Buffer();
    out->AddString(CudaUtil::MarshalHostPointer(mem));
    
    return new Result(errcode_ret, out); 
}

OCL_ROUTINE_HANDLER(EnqueueCopyBuffer) {

 cl_command_queue command_queue = (cl_command_queue)CudaUtil::UnmarshalPointer(input_buffer->AssignString());
 cl_mem src_buffer = (cl_mem)CudaUtil::UnmarshalPointer(input_buffer->AssignString());
 cl_mem dst_buffer = (cl_mem)CudaUtil::UnmarshalPointer(input_buffer->AssignString());

 size_t src_offset= input_buffer->Get<size_t>();
 size_t dst_offset= input_buffer->Get<size_t>();
 size_t cb = input_buffer->Get<size_t>();
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

 cl_int errcode_ret = clEnqueueCopyBuffer(command_queue,src_buffer,dst_buffer,src_offset,dst_offset,cb,num_events_in_wait_list,event_wait_list,event);

 
 Buffer *out = new Buffer();
 if (event!=NULL)
    out->AddString(CudaUtil::MarshalHostPointer(*event));

 return new Result(errcode_ret,out);
}

OCL_ROUTINE_HANDLER(EnqueueReadBuffer) {
    
    cl_command_queue *command_queue= input_buffer->Assign<cl_command_queue>();
    cl_mem *buffer = input_buffer->Assign<cl_mem>();
    cl_bool blocking_read = input_buffer->Get<cl_bool>();
    size_t offset = input_buffer->Get<size_t>();
    size_t cb = input_buffer->Get<size_t>();
    void *ptr = (void*)input_buffer->AssignAll<char>();
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

    
    /* FIXME: There is a problem with non-blocking read/write, so read/write will be always blocking */
    blocking_read = CL_TRUE;

    cl_int exit_code = clEnqueueReadBuffer(*command_queue,*buffer,blocking_read,
            offset,cb,ptr,num_events_in_wait_list,event_wait_list,event);

    Buffer *out = new Buffer();

    if (event != NULL)
        out->AddString(CudaUtil::MarshalHostPointer(*event));
    out->Add((char *)ptr,cb);

    return new Result(exit_code,out);
}

OCL_ROUTINE_HANDLER(EnqueueWriteBuffer) {
    
    cl_command_queue *command_queue= input_buffer->Assign<cl_command_queue>();
    cl_mem *buffer = input_buffer->Assign<cl_mem>();
    cl_bool blocking_write = input_buffer->Get<cl_bool>();
    size_t offset = input_buffer->Get<size_t>();
    size_t cb = input_buffer->Get<size_t>();
    void *ptr = (void*)input_buffer->AssignAll<char>();
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

    /* FIXME: There is a problem with non-blocking read/write,
     * so read/write will be always blocking */
   blocking_write = CL_TRUE;

    cl_int exit_code = clEnqueueWriteBuffer(*command_queue,*buffer,blocking_write,
            offset,cb,ptr,num_events_in_wait_list,event_wait_list,event);

    Buffer *out = new Buffer();

    if (event != NULL)
        out->AddString(CudaUtil::MarshalHostPointer(*event));

    return new Result(exit_code,out);
}

OCL_ROUTINE_HANDLER(ReleaseMemObject){

    cl_mem *memObj = input_buffer->Assign<cl_mem>();

    cl_int exit_code = clReleaseMemObject(*memObj);

    return new Result(exit_code);
}

OCL_ROUTINE_HANDLER(EnqueueMapBuffer){

    
    cl_command_queue commad_queue = (cl_command_queue)CudaUtil::UnmarshalPointer(input_buffer->AssignString());
    cl_mem buffer = (cl_mem)CudaUtil::UnmarshalPointer(input_buffer->AssignString());
    cl_bool blocking_map = input_buffer->Get<cl_bool>();
    cl_map_flags map_flags = input_buffer->Get<cl_map_flags>();
    size_t offset = input_buffer->Get<size_t>();
    size_t cb = input_buffer->Get<size_t>();
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

    cl_int errcode_ret = 0;
    
   blocking_map = CL_TRUE; //FIXME

    void *return_pointer = clEnqueueMapBuffer(commad_queue,buffer,blocking_map,map_flags,
            offset,cb,num_events_in_wait_list,event_wait_list,event,&errcode_ret);

    pThis->RegisterMapObject(CudaUtil::MarshalHostPointer(buffer),CudaUtil::MarshalHostPointer(return_pointer));

    Buffer *out = new Buffer();
    out->Add((char *)return_pointer,cb);
    if (event != NULL)
        out->AddString(CudaUtil::MarshalHostPointer(*event));
    
    return new Result(errcode_ret,out);
}

OCL_ROUTINE_HANDLER(EnqueueUnmapMemObject){

    cl_command_queue command_queue = (cl_command_queue)CudaUtil::UnmarshalPointer(input_buffer->AssignString());
    cl_mem memobj = (cl_mem)CudaUtil::UnmarshalPointer(input_buffer->AssignString());
    void *mapped_ptr = (void *)CudaUtil::UnmarshalPointer((input_buffer->AssignString()));
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

    mapped_ptr = CudaUtil::UnmarshalPointer(pThis->GetMapObject(CudaUtil::MarshalHostPointer(memobj)));

    cl_int errcode_ret = clEnqueueUnmapMemObject(command_queue,memobj,mapped_ptr,num_events_in_wait_list,event_wait_list,event);

    Buffer *out = new Buffer();
    if (event != NULL)
        out->AddString(CudaUtil::MarshalHostPointer(*event));
    

    return new Result(errcode_ret,out);

}