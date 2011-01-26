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
#include <CL/cl_gl.h>
#include <GL/glu.h>
#include <GL/gl.h>



using namespace std;

extern "C" cl_mem clCreateFromGLBuffer ( 	cl_context context,
  	cl_mem_flags flags,
  	GLuint bufobj,
  	cl_int * errcode_ret){

}

extern "C" cl_mem clCreateFromGLTexture2D ( 	cl_context context,
  	cl_mem_flags flags,
  	GLenum texture_target,
  	GLint miplevel,
  	GLuint texture,
  	cl_int *errcode_ret){

}

extern "C" cl_mem clCreateFromGLTexture3D ( 	cl_context context,
  	cl_mem_flags flags,
  	GLenum texture_target,
  	GLint miplevel,
  	GLuint texture,
  	cl_int * errcode_ret){

}

extern "C" cl_mem clCreateFromGLRenderbuffer ( 	cl_context context,
  	cl_mem_flags flags,
  	GLuint renderbuffer,
  	cl_int * errcode_ret){

}
extern "C" cl_int clGetGLObjectInfo ( 	cl_mem memobj,
  	cl_gl_object_type *gl_object_type,
  	GLuint *gl_object_name){

}

extern "C" cl_int clGetGLTextureInfo ( 	cl_mem memobj,
  	cl_gl_texture_info param_name,
  	size_t param_value_size,
  	void *param_value,
  	size_t *param_value_size_ret){

}

extern "C" cl_int clEnqueueAcquireGLObjects ( 	cl_command_queue command_queue,
  	cl_uint num_objects,
  	const cl_mem *mem_objects,
  	cl_uint num_events_in_wait_list,
  	const cl_event *event_wait_list,
  	cl_event *event){

}

extern "C" cl_int clEnqueueReleaseGLObjects ( 	cl_command_queue command_queue,
  	cl_uint num_objects,
  	const cl_mem *mem_objects,
  	cl_uint num_events_in_wait_list,
  	const cl_event *event_wait_list,
  	cl_event *event){

}

extern "C" cl_event clCreateEventFromGLsyncKHR ( 	cl_context context,
  	GLsync sync,
  	cl_int *errcode_ret){

}
