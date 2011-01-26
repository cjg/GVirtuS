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

extern "C" cl_sampler clCreateSampler ( 	cl_context context,
  	cl_bool normalized_coords,
  	cl_addressing_mode addressing_mode,
  	cl_filter_mode filter_mode,
  	cl_int *errcode_ret)  {

    cerr << "*** Error: clCreateSampler not yet implemented!" << endl;
    return 0;
}

extern "C"  cl_int clRetainSampler( 	cl_sampler sampler) {

    cerr << "*** Error: clRetainSampler not yet implemented!" << endl;
    return 0;
}

extern "C"  cl_int clReleaseSampler ( 	cl_sampler sampler) {

    cerr << "*** Error: clReleaseSampler not yet implemented!" << endl;
    return 0;
}


extern "C"  cl_int clGetSamplerInfo ( 	cl_sampler sampler,
  	cl_sampler_info param_name,
  	size_t param_value_size,
  	void *param_value,
  	size_t *param_value_size_ret) {

    cerr << "*** Error: clGetSamplerInfo not yet implemented!" << endl;
    return 0;
}

