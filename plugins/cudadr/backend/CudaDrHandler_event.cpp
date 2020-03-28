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


#include <cuda.h>
#include "CudaDrHandler.h"
#include <driver_types.h>
#include <stdio.h>
#include <string.h>

/*Creates an event.*/
CUDA_DRIVER_HANDLER(EventCreate) {
    CUevent phEvent = NULL;
    unsigned int Flags = input_buffer->Get<unsigned int>();
    CUresult exit_code = cuEventCreate(&phEvent, Flags);
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    out->AddMarshal(phEvent);
    return std::make_shared<Result>((cudaError_t) exit_code, out);
}

/*Destroys an event.*/
CUDA_DRIVER_HANDLER(EventDestroy) {
    CUevent phEvent = input_buffer->Get<CUevent > ();
    CUresult exit_code = cuEventDestroy(phEvent);
    return std::make_shared<Result>((cudaError_t) exit_code);
}

/*Computes the elapsed time between two events.*/
CUDA_DRIVER_HANDLER(EventElapsedTime) {
    float *pMilliseconds = input_buffer->Assign<float>();
    CUevent hStart = input_buffer->Get<CUevent > ();
    CUevent hEnd = input_buffer->Get<CUevent > ();
    CUresult exit_code = cuEventElapsedTime(pMilliseconds, hStart, hEnd);
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    out->Add(pMilliseconds);
    return std::make_shared<Result>((cudaError_t) exit_code, out);
}

/*Queries an event's status.*/
CUDA_DRIVER_HANDLER(EventQuery) {
    CUevent hEvent = input_buffer->Get<CUevent > ();
    CUresult exit_code = cuEventQuery(hEvent);
    return std::make_shared<Result>((cudaError_t) exit_code);
}

/*Records an event. */
CUDA_DRIVER_HANDLER(EventRecord) {
    CUevent hEvent = input_buffer->Get<CUevent > ();
    CUstream hStream = input_buffer->Get<CUstream > ();
    CUresult exit_code = cuEventRecord(hEvent, hStream);
    return std::make_shared<Result>((cudaError_t) exit_code);
}

/*Waits for an event to complete.*/
CUDA_DRIVER_HANDLER(EventSynchronize) {
    CUevent hEvent = input_buffer->Get<CUevent > ();
    CUresult exit_code = cuEventSynchronize(hEvent);
    return std::make_shared<Result>((cudaError_t) exit_code);
}
