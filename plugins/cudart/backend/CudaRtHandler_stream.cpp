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

CUDA_ROUTINE_HANDLER(StreamCreate) {
  try {
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();

#if CUDART_VERSION >= 3010
    cudaStream_t pStream;  // = input_buffer->Assign<cudaStream_t>();
    cudaError_t exit_code = cudaStreamCreate(&pStream);

    out->Add((pointer_t)pStream);
    return std::make_shared<Result>(exit_code, out);
  } catch (string e) {
    cerr << e << endl;
    return std::make_shared<Result>(cudaErrorMemoryAllocation);
  }
#else
    try {
      cudaStream_t *pStream = input_buffer->Assign<cudaStream_t>();
    } catch (string e) {
      cerr << e << endl;
      return std::make_shared<Result>(cudaErrorMemoryAllocation);
    }
    cudaError_t exit_code = cudaStreamCreate(pStream);
    try {
      out->Add(pStream);
    } catch (string e) {
      cerr << e << endl;
      return std::make_shared<Result>(cudaErrorMemoryAllocation);
    }
#endif
}

CUDA_ROUTINE_HANDLER(StreamCreateWithPriority) {
  try {
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();

    cudaStream_t pStream;
    unsigned int flags = input_buffer->Get<unsigned int>();
    int priority = input_buffer->Get<int>();
    cudaError_t exit_code =
        cudaStreamCreateWithPriority(&pStream, flags, priority);

    out->Add((pointer_t)pStream);
    return std::make_shared<Result>(exit_code, out);
  } catch (string e) {
    cerr << e << endl;
    return std::make_shared<Result>(cudaErrorMemoryAllocation);
  }
}

CUDA_ROUTINE_HANDLER(StreamCreateWithFlags) {
  try {
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();

#if CUDART_VERSION >= 3010
    cudaStream_t pStream;  // = input_buffer->Assign<cudaStream_t>();
    unsigned int flags = input_buffer->Get<unsigned int>();
    cudaError_t exit_code = cudaStreamCreateWithFlags(&pStream, flags);
    out->Add((pointer_t)pStream);
    return std::make_shared<Result>(exit_code, out);
  } catch (string e) {
    cerr << e << endl;
    return std::make_shared<Result>(cudaErrorMemoryAllocation);  //???
  }
#else
      try {
        cudaStream_t *pStream = input_buffer->Assign<cudaStream_t>();
      } catch (string e) {
        cerr << e << endl;
        return std::make_shared<Result>(cudaErrorMemoryAllocation);
      }
      cudaError_t exit_code = cudaStreamCreate(pStream);
      try {
        out->Add(pStream);
      } catch (string e) {
        cerr << e << endl;
        return std::make_shared<Result>(cudaErrorMemoryAllocation);
      }
#endif
}

CUDA_ROUTINE_HANDLER(StreamDestroy) {
  try {
    cudaStream_t stream = input_buffer->Get<cudaStream_t>();
    return std::make_shared<Result>(cudaStreamDestroy(stream));
  } catch (string e) {
    cerr << e << endl;
    return std::make_shared<Result>(cudaErrorMemoryAllocation);
  }
}

CUDA_ROUTINE_HANDLER(StreamWaitEvent) {
  try {
    cudaStream_t stream = input_buffer->Get<cudaStream_t>();
    cudaEvent_t event = input_buffer->Get<cudaEvent_t>();
    unsigned int flags = input_buffer->Get<unsigned int>();
    return std::make_shared<Result>(cudaStreamWaitEvent(stream, event, flags));
  } catch (string e) {
    cerr << e << endl;
    return std::make_shared<Result>(cudaErrorMemoryAllocation);
  }
}

CUDA_ROUTINE_HANDLER(StreamQuery) {
  try {
    cudaStream_t stream = input_buffer->Get<cudaStream_t>();
    return std::make_shared<Result>(cudaStreamQuery(stream));
  } catch (string e) {
    cerr << e << endl;
    return std::make_shared<Result>(cudaErrorMemoryAllocation);
  }
}

/*
CUDA_ROUTINE_HANDLER(StreamAddCallback ) {
    try {
        cudaStream_t stream = input_buffer->Get<cudaStream_t>();
        cudaStreamCallback_t callback =
input_buffer->Get<cudaStreamCallback_t>();
        ////
        void *userData = input_buffer->GetFromMarshal<void *>();
        unsigned int flags = input_buffer->Get<unsigned int>();
        return
std::make_shared<Result>(cudaStreamAddCallback(stream,callback,userData,flags));
    } catch (string e) {
        cerr << e << endl;
        return std::make_shared<Result>(cudaErrorMemoryAllocation);
    }
}*/

CUDA_ROUTINE_HANDLER(StreamSynchronize) {
  try {
    cudaStream_t stream = input_buffer->Get<cudaStream_t>();
    return std::make_shared<Result>(cudaStreamSynchronize(stream));
  } catch (string e) {
    cerr << e << endl;
    return std::make_shared<Result>(cudaErrorMemoryAllocation);
  }
}
