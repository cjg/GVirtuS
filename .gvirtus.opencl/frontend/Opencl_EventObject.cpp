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
clWaitForEvents (cl_uint num_events,
  	const cl_event *event_list)  {

    OpenclFrontend::Prepare();

    OpenclFrontend::AddVariableForArguments<cl_uint>(num_events);
    OpenclFrontend::AddHostPointerForArguments(event_list,num_events);
    

    OpenclFrontend::Execute("clWaitForEvents");

    return OpenclFrontend::GetExitCode();

}
extern "C" CL_API_ENTRY cl_int CL_API_CALL
clReleaseEvent (cl_event event)  {

    OpenclFrontend::Prepare();
    OpenclFrontend::AddHostPointerForArguments(&event);
    OpenclFrontend::Execute("clReleaseEvent");
    return OclFrontend::GetExitCode();
}