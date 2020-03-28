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
 * Written by: Carlo Palmieri <carlo.palmieri@uniparthenope.it>,
 *             Department of Science and Technology
 */

#include <ios>
#include <iostream>

#include "CudaRt.h"

using namespace std;

extern "C" __host__ cudaError_t CUDARTAPI
cudaBindSurfaceToArray(const surfaceReference *surfref, const cudaArray *array,
                       const cudaChannelFormatDesc *desc) {
  CudaRtFrontend::Prepare();

  //    cerr << CudaUtil::MarshalHostPointer(surfref) << " " << hex << array <<
  //    endl;

  // Achtung: passing the address and the content of the textureReference
  CudaRtFrontend::AddStringForArguments(CudaUtil::MarshalHostPointer(surfref));
  CudaRtFrontend::AddHostPointerForArguments(surfref);
  CudaRtFrontend::AddDevicePointerForArguments((void *)array);
  CudaRtFrontend::AddHostPointerForArguments(desc);
  CudaRtFrontend::Execute("cudaBindSurfaceToArray");
  return CudaRtFrontend::GetExitCode();
}

// extern "C" __host__ cudaError_t CUDARTAPI cudaGetTextureReference(const
// textureReference **texref,
//        const void *symbol) {
//    CudaRtFrontend::Prepare();
//    // Achtung: skipping to add texref
//    // Achtung: passing the address and the content of symbol
//    CudaRtFrontend::AddStringForArguments(CudaUtil::MarshalHostPointer(symbol));
//    CudaRtFrontend::AddStringForArguments((char*)symbol);
//    CudaRtFrontend::Execute("cudaGetTextureReference");
//    if (CudaRtFrontend::Success()) {
//        char *texrefHandler = CudaRtFrontend::GetOutputString();
//        *texref = (textureReference *) strtoul(texrefHandler, NULL, 16);
//    }
//    return CudaRtFrontend::GetExitCode();
//}
//
