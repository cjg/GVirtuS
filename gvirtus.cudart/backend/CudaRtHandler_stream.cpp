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
        Buffer *out = new Buffer();
  
    #if CUDART_VERSION >= 3010
        cudaStream_t pStream; // = input_buffer->Assign<cudaStream_t>();
        cudaError_t exit_code = cudaStreamCreate(&pStream);

        out->Add((pointer_t) pStream);
           return new Result(exit_code, out);
    } catch (string e) {
        cerr << e << endl;
        return new Result(cudaErrorMemoryAllocation);
    }
#else
    try {
        cudaStream_t *pStream = input_buffer->Assign<cudaStream_t>();
    } catch (string e) {
        cerr << e << endl;
        return new Result(cudaErrorMemoryAllocation);
    }
    cudaError_t exit_code = cudaStreamCreate(pStream);
    try {
        out->Add(pStream);
    } catch (string e) {
        cerr << e << endl;
        return new Result(cudaErrorMemoryAllocation);
    }
#endif
 
}

CUDA_ROUTINE_HANDLER(StreamDestroy) {
    try {
        cudaStream_t stream = input_buffer->Get<cudaStream_t>();
        return new Result(cudaStreamDestroy(stream));
    } catch (string e) {
        cerr << e << endl;
        return new Result(cudaErrorMemoryAllocation);
    }

    
}

CUDA_ROUTINE_HANDLER(StreamQuery) {
    try {
        cudaStream_t stream = input_buffer->Get<cudaStream_t>();
        return new Result(cudaStreamQuery(stream));
    } catch (string e) {
        cerr << e << endl;
        return new Result(cudaErrorMemoryAllocation);
    }

    
}

CUDA_ROUTINE_HANDLER(StreamSynchronize) {
    try {
        cudaStream_t stream = input_buffer->Get<cudaStream_t>();
        return new Result(cudaStreamSynchronize(stream));
    } catch (string e) {
        cerr << e << endl;
        return new Result(cudaErrorMemoryAllocation);
    }

    
}
