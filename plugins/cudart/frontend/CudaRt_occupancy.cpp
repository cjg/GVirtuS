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
 * Written by: Raffaele Montella <raffaele.montella@uniparthenope.it>,
 *             Department of Science and Technologies
 */

#include "CudaRt.h"
using namespace std;

using gvirtus::common::pointer_t;

/* cudaOccupancyMaxActiveBlocksPerMultiprocessor */
extern "C" __host__ cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    int* numBlocks, const void* func, int blockSize, size_t dynamicSMemSize) {
  CudaRtFrontend::Prepare();
  CudaRtFrontend::AddHostPointerForArguments(numBlocks);
  CudaRtFrontend::AddVariableForArguments((pointer_t)func);
  CudaRtFrontend::AddVariableForArguments(blockSize);
  CudaRtFrontend::AddVariableForArguments(dynamicSMemSize);
  CudaRtFrontend::Execute("cudaOccupancyMaxActiveBlocksPerMultiprocessor");

  if (CudaRtFrontend::Success())
    *numBlocks = *(CudaRtFrontend::GetOutputHostPointer<int>());
  return CudaRtFrontend::GetExitCode();
}

/* cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags */
#if (CUDART_VERSION >= 7000)
extern "C" __host__ cudaError_t
cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int* numBlocks,
                                                       const void* func,
                                                       int blockSize,
                                                       size_t dynamicSMemSize,
                                                       unsigned int flags) {
  CudaRtFrontend::Prepare();
  CudaRtFrontend::AddHostPointerForArguments(numBlocks);
  CudaRtFrontend::AddVariableForArguments((pointer_t)func);
  CudaRtFrontend::AddVariableForArguments(blockSize);
  CudaRtFrontend::AddVariableForArguments(dynamicSMemSize);
  CudaRtFrontend::AddVariableForArguments(flags);
  CudaRtFrontend::Execute(
      "cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags");

  if (CudaRtFrontend::Success())
    *numBlocks = *(CudaRtFrontend::GetOutputHostPointer<int>());
  return CudaRtFrontend::GetExitCode();
}
#endif
