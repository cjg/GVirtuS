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

#include <stdlib.h>


#include <string.h>

#include "Ocl.h"

using namespace std;


extern "C" cl_int clGetSupportedImageFormats (cl_context context,
  	cl_mem_flags flags,
  	cl_mem_object_type image_type,
  	cl_uint num_entries,
  	cl_image_format *image_formats,
  	cl_uint *num_image_formats){

    OclFrontend::Prepare();
    OclFrontend::AddHostPointerForArguments<cl_context>(&context);
    OclFrontend::AddVariableForArguments<cl_mem_flags>(flags);
    OclFrontend::AddVariableForArguments<cl_mem_object_type>(image_type);
    OclFrontend::AddVariableForArguments<cl_uint>(num_entries);
    OclFrontend::AddHostPointerForArguments<cl_image_format>(image_formats,num_entries);
    OclFrontend::AddHostPointerForArguments<cl_uint>(num_image_formats);

    OclFrontend::Execute("clGetSupportedImageFormats");

    if (OclFrontend::Success()){

        cl_uint *tmp_num_image_formats;
        tmp_num_image_formats=(OclFrontend::GetOutputHostPointer<cl_uint>());
        if (tmp_num_image_formats!=NULL){
            *num_image_formats=*tmp_num_image_formats;
        }
        cl_image_format *tmp_image_formats;
        tmp_image_formats=OclFrontend::GetOutputHostPointer<cl_image_format>();
        memcpy(image_formats,tmp_image_formats,sizeof(image_formats)*num_entries);
    }

    return OclFrontend::GetExitCode();

}


extern "C" cl_mem clCreateBuffer (cl_context context,
  	cl_mem_flags flags,
  	size_t size,
  	void *host_ptr,
  	cl_int *errcode_ret){

    OclFrontend::Prepare();
    OclFrontend::AddStringForArguments(CudaUtil::MarshalHostPointer(context));
    OclFrontend::AddVariableForArguments<cl_mem_flags>(flags);
    OclFrontend::AddVariableForArguments<size_t>(size);
    OclFrontend::AddHostPointerForArguments<char>((char*)host_ptr,size);

    OclFrontend::Execute("clCreateBuffer");

    cl_mem mem;
    if (OclFrontend::Success()){
        mem = (cl_mem)CudaUtil::UnmarshalPointer(OclFrontend::GetOutputString());
        if (errcode_ret !=NULL)
            *errcode_ret=OclFrontend::GetExitCode();
    }

    return mem;
}

extern "C" cl_int clEnqueueCopyBuffer (cl_command_queue command_queue,
  	cl_mem src_buffer,
  	cl_mem dst_buffer,
  	size_t src_offset,
  	size_t dst_offset,
  	size_t cb,
  	cl_uint num_events_in_wait_list,
  	const cl_event *event_wait_list,
  	cl_event *event){

    OclFrontend::Prepare();
    
    OclFrontend::AddStringForArguments(CudaUtil::MarshalHostPointer(command_queue));

    OclFrontend::AddStringForArguments(CudaUtil::MarshalHostPointer(src_buffer));
    OclFrontend::AddStringForArguments(CudaUtil::MarshalHostPointer(dst_buffer));
    OclFrontend::AddVariableForArguments<size_t>(src_offset);
    OclFrontend::AddVariableForArguments<size_t>(dst_offset);
    OclFrontend::AddVariableForArguments<size_t>(cb);
    OclFrontend::AddVariableForArguments<cl_uint>(num_events_in_wait_list);
    for (int i=0; i<num_events_in_wait_list; i++){
        OclFrontend::AddStringForArguments(CudaUtil::MarshalHostPointer(event_wait_list[i]));
    }
     if (event != NULL)
        OclFrontend::AddVariableForArguments(true);
     else
        OclFrontend::AddVariableForArguments(false);
    

    OclFrontend::Execute("clEnqueueCopyBuffer");

    if (OclFrontend::GetExitCode()){
        if (event!=NULL){
            *event=*(OclFrontend::GetOutputHostPointer<cl_event>());
        }
    }
    return OclFrontend::GetExitCode();

}

extern "C" cl_int clEnqueueReadBuffer (cl_command_queue command_queue,
  	cl_mem buffer,
  	cl_bool blocking_read,
  	size_t offset,
  	size_t cb,
  	void *ptr,
  	cl_uint num_events_in_wait_list,
  	const cl_event *event_wait_list,
  	cl_event *event){

    
    OclFrontend::Prepare();
    OclFrontend::AddHostPointerForArguments<cl_command_queue>(&command_queue);
    OclFrontend::AddHostPointerForArguments<cl_mem>(&buffer);
    OclFrontend::AddVariableForArguments<cl_bool>(blocking_read);
    OclFrontend::AddVariableForArguments<size_t>(offset);
    OclFrontend::AddVariableForArguments<size_t>(cb);
    OclFrontend::AddHostPointerForArguments((char*)ptr,cb);
    OclFrontend::AddVariableForArguments<cl_uint>(num_events_in_wait_list);
    for (int i=0; i<num_events_in_wait_list; i++){
        OclFrontend::AddStringForArguments(CudaUtil::MarshalHostPointer(event_wait_list[i]));
    }
    if (event != NULL)
        OclFrontend::AddVariableForArguments(true);
     else
        OclFrontend::AddVariableForArguments(false);
    
    
    OclFrontend::Execute("clEnqueueReadBuffer");

    cl_event *tmp_event;
    char * tmp;
    if (OclFrontend::Success()){
        if (event != NULL){
            *event = (cl_event)CudaUtil::UnmarshalPointer(OclFrontend::GetOutputString());
      
        }
        if (ptr!=NULL){
            memcpy(ptr,OclFrontend::GetOutputHostPointer<char>(),cb);
        }
        
    }

    return OclFrontend::GetExitCode();
    
}

extern "C" cl_int clEnqueueWriteBuffer (cl_command_queue command_queue,
  	cl_mem buffer,
  	cl_bool blocking_write,
  	size_t offset,
  	size_t cb,
  	const void *ptr,
  	cl_uint num_events_in_wait_list,
  	const cl_event *event_wait_list,
  	cl_event *event){

    OclFrontend::Prepare();
    OclFrontend::AddHostPointerForArguments<cl_command_queue>(&command_queue);
    OclFrontend::AddHostPointerForArguments<cl_mem>(&buffer);
    OclFrontend::AddVariableForArguments<cl_bool>(blocking_write);
    OclFrontend::AddVariableForArguments<size_t>(offset);
    OclFrontend::AddVariableForArguments<size_t>(cb);
    OclFrontend::AddHostPointerForArguments((char*)ptr,cb);
    OclFrontend::AddVariableForArguments<cl_uint>(num_events_in_wait_list);
    for (int i=0; i<num_events_in_wait_list; i++){
        OclFrontend::AddStringForArguments(CudaUtil::MarshalHostPointer(event_wait_list[i]));
    }
    if (event != NULL)
        OclFrontend::AddVariableForArguments(true);
     else
        OclFrontend::AddVariableForArguments(false);
    

    OclFrontend::Execute("clEnqueueWriteBuffer");

    
    char * tmp;
    if (OclFrontend::Success()){
        if (event != NULL){
            *event = (cl_event)CudaUtil::UnmarshalPointer(OclFrontend::GetOutputString());
        } 
     
    }

    return OclFrontend::GetExitCode();

}

extern "C" void * clEnqueueMapBuffer ( 	cl_command_queue command_queue,
  	cl_mem buffer,
  	cl_bool blocking_map,
  	cl_map_flags map_flags,
  	size_t offset,
  	size_t cb,
  	cl_uint num_events_in_wait_list,
  	const cl_event *event_wait_list,
  	cl_event *event,
  	cl_int *errcode_ret) {

    OclFrontend::Prepare();
    OclFrontend::AddStringForArguments(CudaUtil::MarshalHostPointer(command_queue));
    OclFrontend::AddStringForArguments(CudaUtil::MarshalHostPointer(buffer));
    OclFrontend::AddVariableForArguments(blocking_map);
    OclFrontend::AddVariableForArguments(map_flags);
    OclFrontend::AddVariableForArguments(offset);
    OclFrontend::AddVariableForArguments(cb);
    OclFrontend::AddVariableForArguments(num_events_in_wait_list);
    for (int i=0; i<num_events_in_wait_list; i++){
        OclFrontend::AddStringForArguments(CudaUtil::MarshalHostPointer(event_wait_list[i]));
    }
    if (event != NULL)
        OclFrontend::AddVariableForArguments(true);
     else
        OclFrontend::AddVariableForArguments(false);
    

    OclFrontend::Execute("clEnqueueMapBuffer");
    void *return_pointer = (void *)malloc(cb * sizeof(char));
    if (OclFrontend::Success()){
        memcpy(return_pointer,OclFrontend::GetOutputHostPointer<char>(),cb);
        if (event != NULL){
                *event = (cl_event)CudaUtil::UnmarshalPointer(OclFrontend::GetOutputString());
        }
        if (errcode_ret != NULL){
                *errcode_ret = OclFrontend::GetExitCode();
        }
    }

    return return_pointer;

}

extern "C" cl_int clEnqueueUnmapMemObject (cl_command_queue command_queue,
  	cl_mem memobj,
  	void *mapped_ptr,
  	cl_uint num_events_in_wait_list,
  	const cl_event *event_wait_list,
  	cl_event *event) {

    OclFrontend::Prepare();
    OclFrontend::AddStringForArguments(CudaUtil::MarshalHostPointer(command_queue));
    OclFrontend::AddStringForArguments(CudaUtil::MarshalHostPointer(memobj));
    OclFrontend::AddStringForArguments(CudaUtil::MarshalHostPointer(mapped_ptr));
    OclFrontend::AddVariableForArguments(num_events_in_wait_list);
    for (int i=0; i<num_events_in_wait_list; i++){
        OclFrontend::AddStringForArguments(CudaUtil::MarshalHostPointer(event_wait_list[i]));
    }
    if (event != NULL)
        OclFrontend::AddVariableForArguments(true);
     else
        OclFrontend::AddVariableForArguments(false);
    
    
    OclFrontend::Execute("clEnqueueUnmapMemObject");
    
    if (OclFrontend::Success()){
           if (event != NULL){
                *event = (cl_event)CudaUtil::UnmarshalPointer(OclFrontend::GetOutputString());
           }
    }

    return OclFrontend::GetExitCode();

}

extern "C" cl_int clReleaseMemObject (cl_mem memobj){
    OclFrontend::Prepare();

    OclFrontend::AddHostPointerForArguments<cl_mem>(&memobj);

    OclFrontend::Execute("clReleaseMemObject");

    return OclFrontend::GetExitCode();


}


extern "C" cl_int clRetainMemObject (cl_mem memobj){
    cerr << "*** Error: clRetainMemObject not yet implemented!" << endl;     return 0;
}



extern "C" cl_mem clCreateImage2D (cl_context context,
  	cl_mem_flags flags,
  	const cl_image_format *image_format,
  	size_t image_width,
  	size_t image_height,
  	size_t image_row_pitch,
  	void *host_ptr,
  	cl_int *errcode_ret) {

    cerr << "*** Error: clCreateImage2D not yet implemented!" << endl;     return 0;
}

extern "C" cl_mem clCreateImage3D (cl_context context,
  	cl_mem_flags flags,
  	const cl_image_format *image_format,
  	size_t image_width,
  	size_t image_height,
  	size_t image_depth,
  	size_t image_row_pitch,
  	size_t image_slice_pitch,
  	void *host_ptr,
  	cl_int *errcode_ret) {

    cerr << "*** Error: clCreateImage3D not yet implemented!" << endl;     return 0;
}
extern "C" cl_int clEnqueueReadImage ( 	cl_command_queue command_queue,
  	cl_mem image,
  	cl_bool blocking_read,
  	const size_t origin[3],
  	const size_t region[3],
  	size_t row_pitch,
  	size_t slice_pitch,
  	void *ptr,
  	cl_uint num_events_in_wait_list,
  	const cl_event *event_wait_list,
  	cl_event *event) {

    cerr << "*** Error: clEnqueueReadImage not yet implemented!" << endl;     return 0;
}
extern "C" cl_int clEnqueueWriteImage ( 	cl_command_queue command_queue,
  	cl_mem image,
  	cl_bool blocking_write,
  	const size_t origin[3],
  	const size_t region[3],
  	size_t input_row_pitch,
  	size_t input_slice_pitch,
  	const void * ptr,
  	cl_uint num_events_in_wait_list,
  	const cl_event *event_wait_list,
  	cl_event *event) {

    cerr << "*** Error: clEnqueueWriteImage not yet implemented!" << endl;     return 0;
}
extern "C" cl_int clEnqueueCopyImage ( 	cl_command_queue command_queue,
  	cl_mem src_image,
  	cl_mem dst_image,
  	const size_t src_origin[3],
  	const size_t dst_origin[3],
  	const size_t region[3],
  	cl_uint num_events_in_wait_list,
  	const cl_event *event_wait_list,
  	cl_event *event) {

    cerr << "*** Error: clEnqueueCopyImage not yet implemented!" << endl;     return 0;
}
extern "C" cl_int clEnqueueCopyImageToBuffer ( 	cl_command_queue command_queue,
  	cl_mem src_image,
  	cl_mem  dst_buffer,
  	const size_t src_origin[3],
  	const size_t region[3],
  	size_t dst_offset,
  	cl_uint num_events_in_wait_list,
  	const cl_event *event_wait_list,
  	cl_event *event)
{

    cerr << "*** Error: clEnqueueCopyImageToBuffer not yet implemented!" << endl;     return 0;
}
extern "C" cl_int clEnqueueCopyBufferToImage ( 	cl_command_queue command_queue,
  	cl_mem src_buffer,
  	cl_mem  dst_image,
  	size_t src_offset,
  	const size_t dst_origin[3],
  	const size_t region[3],
  	cl_uint num_events_in_wait_list,
  	const cl_event *event_wait_list,
  	cl_event *event) {

    cerr << "*** Error: clEnqueueCopyBufferToImage not yet implemented!" << endl;     return 0;
}

extern "C" void * clEnqueueMapImage ( 	cl_command_queue command_queue,
  	cl_mem image,
  	cl_bool blocking_map,
  	cl_map_flags map_flags,
  	const size_t origin[3],
  	const size_t region[3],
  	size_t *image_row_pitch,
  	size_t *image_slice_pitch,
  	cl_uint num_events_in_wait_list,
  	const cl_event *event_wait_list,
  	cl_event *event,
  	cl_int *errcode_ret) {

    cerr << "*** Error: clEnqueueMapImage not yet implemented!" << endl;     return 0;
}

extern "C" cl_int clGetMemObjectInfo ( 	cl_mem memobj,
  	cl_mem_info param_name,
  	size_t param_value_size,
  	void *param_value,
  	size_t *param_value_size_ret) {

    cerr << "*** Error: clGetMemObjectInfo not yet implemented!" << endl;     return 0;
}

extern "C" cl_int  clGetImageInfo (cl_mem image,
  	cl_image_info param_name,
  	size_t param_value_size,
  	void *param_value,
  	size_t *param_value_size_ret) {

    cerr << "*** Error: clGetImageInfo not yet implemented!" << endl;     return 0;
}

