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

CUDA_ROUTINE_HANDLER(EventCreate) {
    try {
        Buffer *out = new Buffer();
    
    #if CUDART_VERSION >= 3010
        cudaEvent_t event;
        cudaError_t exit_code = cudaEventCreate(&event);
        out->Add((pointer_t) event);
        return new Result(exit_code, out);
    } catch (string e) {
        cerr << e << endl;
        return new Result(cudaErrorMemoryAllocation);
    }
#else
    
    try {
        cudaEvent_t *event = input_buffer->Assign<cudaEvent_t>();
        cudaError_t exit_code = cudaEventCreate(event);
        out->Add(event);
    } catch (string e) {
        cerr << e << endl;
        return new Result(cudaErrorMemoryAllocation);
    }
#endif
    
}

#if CUDART_VERSION >= 2030
CUDA_ROUTINE_HANDLER(EventCreateWithFlags) {
    try {
        Buffer *out = new Buffer();
   
    #if CUDART_VERSION >= 3010
        cudaEvent_t event;
        int flags = input_buffer->Get<int>();
        cudaError_t exit_code = cudaEventCreateWithFlags(&event, flags);
        out->Add((pointer_t) event);
        return new Result(exit_code, out);
    } catch (string e) {
        cerr << e << endl;
        return new Result(cudaErrorMemoryAllocation);
    }
#else
    try {
        cudaEvent_t *event = input_buffer->Assign<cudaEvent_t>();
        int flags = input_buffer->Get<int>();
        cudaError_t exit_code = cudaEventCreateWithFlags(event, flags);
        out->Add(event);
    } catch (string e) {
        cerr << e << endl;
        return new Result(cudaErrorMemoryAllocation);
    }
#endif
   
}
#endif

CUDA_ROUTINE_HANDLER(EventDestroy) {
    try {
        cudaEvent_t event = input_buffer->Get<cudaEvent_t>();
        return new Result(cudaEventDestroy(event));
    } catch (string e) {
        cerr << e << endl;
        return new Result(cudaErrorMemoryAllocation);
    }
    
}

CUDA_ROUTINE_HANDLER(EventElapsedTime) {
    try {
        float *ms = input_buffer->Assign<float>();
        cudaEvent_t start = input_buffer->Get<cudaEvent_t>();
        cudaEvent_t end = input_buffer->Get<cudaEvent_t>();
        cudaError_t exit_code = cudaEventElapsedTime(ms, start, end);
        Buffer *out = new Buffer();
        out->Add(ms);
        return new Result(exit_code, out);
    } catch (string e) {
        cerr << e << endl;
        return new Result(cudaErrorMemoryAllocation);
    }

    
}

CUDA_ROUTINE_HANDLER(EventQuery) {
    try {
        cudaEvent_t event = input_buffer->Get<cudaEvent_t>();
        return new Result(cudaEventQuery(event));
    } catch (string e) {
        cerr << e << endl;
        return new Result(cudaErrorMemoryAllocation);
    }
        
    
}

CUDA_ROUTINE_HANDLER(EventRecord) {
    try {
        cudaEvent_t event = input_buffer->Get<cudaEvent_t>();
        cudaStream_t stream = input_buffer->Get<cudaStream_t>();
        return new Result(cudaEventRecord(event, stream));
    } catch (string e) {
        cerr << e << endl;
        return new Result(cudaErrorMemoryAllocation);
    }
    
}

CUDA_ROUTINE_HANDLER(EventSynchronize) {
    try {
        cudaEvent_t event = input_buffer->Get<cudaEvent_t>();
        return new Result(cudaEventSynchronize(event));
    } catch (string e) {
        cerr << e << endl;
        return new Result(cudaErrorMemoryAllocation);
    }

    
}

