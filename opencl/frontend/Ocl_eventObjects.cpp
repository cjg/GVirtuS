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

extern "C" cl_int clWaitForEvents (cl_uint num_events,
  	const cl_event *event_list)  {

    OclFrontend::Prepare();

    OclFrontend::AddVariableForArguments<cl_uint>(num_events);
    OclFrontend::AddHostPointerForArguments(event_list,num_events);
    


    OclFrontend::Execute("clWaitForEvents");

    return OclFrontend::GetExitCode();

}

extern "C" cl_int clReleaseEvent (cl_event event)  {

    OclFrontend::Prepare();
    OclFrontend::AddHostPointerForArguments(&event);
    OclFrontend::Execute("clReleaseEvent");
    return OclFrontend::GetExitCode();
}

extern "C"  cl_int clGetEventInfo ( 	cl_event event,
  	cl_event_info param_name,
  	size_t param_value_size,
  	void *param_value,
  	size_t *param_value_size_ret) {
    cerr << "*** Error: clGetEventInfo not yet implemented!" << endl;
    return 0;
}

extern "C" cl_int clRetainEvent ( 	cl_event event)  {
    cerr << "*** Error: clRetainEvent not yet implemented!" << endl;
    return 0;
}




