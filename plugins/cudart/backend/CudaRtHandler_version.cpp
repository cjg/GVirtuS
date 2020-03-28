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

#include "CudaRtHandler.h"

#ifndef CUDART_VERSION
#error CUDART_VERSION not defined
#endif
#if CUDART_VERSION >= 2030
CUDA_ROUTINE_HANDLER(DriverGetVersion) {
  try {
    int *driverVersion = input_buffer->Assign<int>();
    cudaError_t exit_code = cudaDriverGetVersion(driverVersion);

    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();

    out->Add(driverVersion);
    return std::make_shared<Result>(exit_code, out);
  } catch (string e) {
    cerr << e << endl;
    return std::make_shared<Result>(cudaErrorMemoryAllocation);
  }
}

CUDA_ROUTINE_HANDLER(RuntimeGetVersion) {
  try {
    int *runtimeVersion = input_buffer->Assign<int>();
    cudaError_t exit_code = cudaRuntimeGetVersion(runtimeVersion);
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();

    out->Add(runtimeVersion);
    return std::make_shared<Result>(exit_code, out);
  } catch (string e) {
    cerr << e << endl;
    return std::make_shared<Result>(cudaErrorMemoryAllocation);
  }
}
#endif
