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
 * Written by: Flora Giannone <flora.giannone@studenti.uniparthenope.it>,
 *             Department of Applied Science
 */

#include "CudaRtHandler.h"

#ifndef CUDART_VERSION
#error CUDART_VERSION not defined
#endif

/*OccupancyMaxActiveBlocksPerMultiprocessor.*/
CUDA_ROUTINE_HANDLER(OccupancyMaxActiveBlocksPerMultiprocessor) {
  int *numBlocks = input_buffer->Assign<int>();
  const char *func = (const char *)(input_buffer->Get<pointer_t>());
  int blockSize = input_buffer->Get<int>();
  size_t dynamicSMemSize = input_buffer->Get<size_t>();

  cudaError_t exit_code = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      numBlocks, func, blockSize, dynamicSMemSize);

  std::shared_ptr<Buffer> out = std::make_shared<Buffer>();

  try {
    out->Add(numBlocks);
  } catch (string e) {
    cerr << e << endl;
    return std::make_shared<Result>(cudaErrorMemoryAllocation);
  }

  return std::make_shared<Result>(exit_code, out);
}

/*OccupancyMaxActiveBlocksPerMultiprocessorWithFlags.*/
#if (CUDART_VERSION >= 7000)
CUDA_ROUTINE_HANDLER(OccupancyMaxActiveBlocksPerMultiprocessorWithFlags) {
  int *numBlocks = input_buffer->Assign<int>();
  const char *func = (const char *)(input_buffer->Get<pointer_t>());
  int blockSize = input_buffer->Get<int>();
  size_t dynamicSMemSize = input_buffer->Get<size_t>();
  unsigned int flags = input_buffer->Get<unsigned int>();

  cudaError_t exit_code =
      cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
          numBlocks, func, blockSize, dynamicSMemSize, flags);

  std::shared_ptr<Buffer> out = std::make_shared<Buffer>();

  try {
    out->Add(numBlocks);
  } catch (string e) {
    cerr << e << endl;
    return std::make_shared<Result>(cudaErrorMemoryAllocation);
  }

  return std::make_shared<Result>(exit_code, out);
}
#endif
