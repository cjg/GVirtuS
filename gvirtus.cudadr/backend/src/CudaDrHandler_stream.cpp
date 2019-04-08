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

/*Create a stream.*/
CUDA_DRIVER_HANDLER(StreamCreate) {
    CUstream phStream = NULL;
    unsigned int Flags = input_buffer->Get<unsigned int>();
    CUresult exit_code = cuStreamCreate(&phStream, Flags);
    Buffer *out = new Buffer();
    out->AddMarshal(phStream);
    return new Result((cudaError_t) exit_code, out);
}

/*Destroys a stream.*/
CUDA_DRIVER_HANDLER(StreamDestroy) {
    CUstream phStream = input_buffer->Get<CUstream > ();
    CUresult exit_code = cuStreamDestroy(phStream);
    return new Result((cudaError_t) exit_code);
}

/*Determine status of a compute stream.*/
CUDA_DRIVER_HANDLER(StreamQuery) {
    CUstream phStream = input_buffer->Get<CUstream > ();
    CUresult exit_code = cuStreamQuery(phStream);
    return new Result((cudaError_t) exit_code);
}

/*Wait until a stream's tasks are completed.*/
CUDA_DRIVER_HANDLER(StreamSynchronize) {
    CUstream phStream = input_buffer->Get<CUstream > ();
    CUresult exit_code = cuStreamSynchronize(phStream);
    return new Result((cudaError_t) exit_code);
}
