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

/*Sets the parameter size for the function.*/
CUDA_DRIVER_HANDLER(ParamSetSize) {
    unsigned int numbytes = input_buffer->Get<unsigned int>();
    CUfunction hfunc = input_buffer->Get<CUfunction > ();
    CUresult exit_code = cuParamSetSize(hfunc, numbytes);
    return new Result((cudaError_t) exit_code);
}

/*Sets the block-dimensions for the function.*/
CUDA_DRIVER_HANDLER(FuncSetBlockShape) {
    int x = input_buffer->Get<int>();
    int y = input_buffer->Get<int>();
    int z = input_buffer->Get<int>();
    CUfunction hfunc = input_buffer->Get<CUfunction > ();
    CUresult exit_code = cuFuncSetBlockShape(hfunc, x, y, z);
    return new Result((cudaError_t) exit_code);
}

/*Launches a CUDA function.*/
CUDA_DRIVER_HANDLER(LaunchGrid) {
    int grid_width = input_buffer->Get<int>();
    int grid_height = input_buffer->Get<int>();
    CUfunction f = input_buffer->Get<CUfunction > ();
    CUresult exit_code = cuLaunchGrid(f, grid_width, grid_height);
    return new Result((cudaError_t) exit_code);
}

/*Returns information about a function.*/
CUDA_DRIVER_HANDLER(FuncGetAttribute) {
    int *pi = input_buffer->Assign<int>();
    CUfunction_attribute attrib = input_buffer->Get<CUfunction_attribute > ();
    CUfunction hfunc = input_buffer->Get<CUfunction > ();
    CUresult exit_code = cuFuncGetAttribute(pi, attrib, hfunc);
    Buffer * out = new Buffer();
    out->Add(pi);
    return new Result((cudaError_t) exit_code, out);
}

/*Sets the dynamic shared-memory size for the function.*/
CUDA_DRIVER_HANDLER(FuncSetSharedSize) {
    unsigned int bytes = input_buffer->Get<unsigned int>();
    CUfunction hfunc = input_buffer->Get<CUfunction > ();
    CUresult exit_code = cuFuncSetSharedSize(hfunc, bytes);
    return new Result((cudaError_t) exit_code);
}

/*Launches a CUDA function.*/
CUDA_DRIVER_HANDLER(Launch) {
    CUfunction f = input_buffer->Get<CUfunction > ();
    CUresult exit_code = cuLaunch(f);
    return new Result((cudaError_t) exit_code);
}

/*Adds a floating-point parameter to the function's argument list.*/
CUDA_DRIVER_HANDLER(ParamSetf) {
    int offset = input_buffer->Get<int>();
    float value = input_buffer->Get<float>();
    CUfunction hfunc = input_buffer->Get<CUfunction > ();
    CUresult exit_code = cuParamSetf(hfunc, offset, value);
    return new Result((cudaError_t) exit_code);
}

/*Adds an integer parameter to the function's argument list.*/
CUDA_DRIVER_HANDLER(ParamSeti) {
    int offset = input_buffer->Get<int>();
    unsigned int value = input_buffer->Get<unsigned int>();
    CUfunction hfunc = input_buffer->Get<CUfunction > ();
    CUresult exit_code = cuParamSeti(hfunc, offset, value);
    return new Result((cudaError_t) exit_code);
}

/*Adds arbitrary data to the function's argument list.*/
CUDA_DRIVER_HANDLER(ParamSetv) {
    int offset = input_buffer->Get<int>();
    unsigned int numbytes = input_buffer->Get<unsigned int>();
    void *ptr = input_buffer->Assign<void *>();
    CUfunction hfunc = input_buffer->Get<CUfunction > ();
    CUresult exit_code = cuParamSetv(hfunc, offset, ptr, numbytes);
    return new Result((cudaError_t) exit_code);
}

/*Adds a texture-reference to the function's argument list. */
CUDA_DRIVER_HANDLER(ParamSetTexRef) {
    CUfunction hfunc = input_buffer->GetFromMarshal<CUfunction > ();
    int texunit = input_buffer->Get<int>();
    CUtexref hTexRef = input_buffer->GetFromMarshal<CUtexref > ();
    CUresult exit_code = cuParamSetTexRef(hfunc, texunit, hTexRef);
    return new Result((cudaError_t) exit_code);
}

/*Launches a CUDA function.*/
CUDA_DRIVER_HANDLER(LaunchGridAsync) {
    int grid_width = input_buffer->Get<int>();
    int grid_height = input_buffer->Get<int>();
    CUfunction f = input_buffer->Get<CUfunction > ();
    CUstream hStream = input_buffer->Get<CUstream > ();
    CUresult exit_code = cuLaunchGridAsync(f, grid_width, grid_height, hStream);
    return new Result((cudaError_t) exit_code);
}

/*Sets the preferred cache configuration for a device function. */
CUDA_DRIVER_HANDLER(FuncSetCacheConfig) {
    CUfunc_cache config = input_buffer->Get<CUfunc_cache > ();
    CUfunction f = input_buffer->Get<CUfunction > ();
    CUresult exit_code = cuFuncSetCacheConfig(f, config);
    return new Result((cudaError_t) exit_code);
}
