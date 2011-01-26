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

using namespace std;

/*Load a module's data. */
CUDA_DRIVER_HANDLER(ModuleLoadData) {
    CUmodule module = NULL;
    char *image = input_buffer->AssignString();
    CUresult exit_code = cuModuleLoadData(&module, image);
    Buffer *out = new Buffer();
    out->AddMarshal(module);
    return new Result((cudaError_t) exit_code, out);
}

/*Returns a function handle*/
CUDA_DRIVER_HANDLER(ModuleGetFunction) {
    CUfunction hfunc = 0;
    char *name = input_buffer->AssignString();
    CUmodule hmod = input_buffer->Get<CUmodule > ();
    CUresult exit_code = cuModuleGetFunction(&hfunc, hmod, name);
    Buffer *out = new Buffer();
    out->AddMarshal(hfunc);
    return new Result((cudaError_t) exit_code, out);
}

/*Returns a global pointer from a module.*/
CUDA_DRIVER_HANDLER(ModuleGetGlobal) {
    CUdeviceptr dptr;
    size_t bytes;
    char *name = input_buffer->AssignString();
    CUmodule hmod = input_buffer->Get<CUmodule > ();
    CUresult exit_code = cuModuleGetGlobal(&dptr, &bytes, hmod, name);
    Buffer * output_buffer = new Buffer();
    output_buffer->AddMarshal(dptr);
    output_buffer->AddMarshal(bytes);
    return new Result((cudaError_t) exit_code, output_buffer);
}

/*Load a module's data with options.*/
CUDA_DRIVER_HANDLER(ModuleLoadDataEx) {
    CUmodule module;
    unsigned int numOptions = input_buffer->Get<unsigned int>();
    CUjit_option *options = input_buffer->AssignAll<CUjit_option > ();
    char *image = input_buffer->AssignString();
    void **optionValues = new void*[numOptions];
    int log_buffer_size_bytes = 1024;
    int error_log_buffer_size_bytes = 1024;

    for (int i = 0; i < numOptions; i++) {

        switch (options[i]) {
            case CU_JIT_INFO_LOG_BUFFER:
                *(optionValues + i) = input_buffer->Assign<char>();
                optionValues[i] = malloc(log_buffer_size_bytes * sizeof (char));
                break;
            case CU_JIT_ERROR_LOG_BUFFER:
                *(optionValues + i) = input_buffer->Assign<char>();
                optionValues[i] = malloc(error_log_buffer_size_bytes * sizeof (char));
                break;
            case CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES:
                log_buffer_size_bytes = (*(input_buffer->Assign<unsigned int>()));
                optionValues[i] = (void *) log_buffer_size_bytes;
                break;
            case CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES:
                error_log_buffer_size_bytes = (*(input_buffer->Assign<unsigned int>()));
                optionValues[i] = (void *) log_buffer_size_bytes;
                break;
            default:
                optionValues[i] = (void *) (*(input_buffer->Assign<unsigned int>()));
        }
    }
    CUresult exit_code = cuModuleLoadDataEx(&module, image, numOptions, options, optionValues);
    Buffer * out = new Buffer();
    out->AddMarshal(module);
    for (int i = 0; i < numOptions; i++) {
        if (options[i] == CU_JIT_INFO_LOG_BUFFER || options[i] == CU_JIT_ERROR_LOG_BUFFER) {
            int len_string = strlen((char *) (optionValues[i]));
            out->Add<int>(len_string);
            out->Add((char *) (optionValues[i]), len_string);
        } else
            out->Add(&optionValues[i]);
    }
    return new Result((cudaError_t) exit_code, out);
}

/*Returns a handle to a texture-reference.*/
CUDA_DRIVER_HANDLER(ModuleGetTexRef) {
    CUtexref pTexRef;
    char *name = input_buffer->AssignString();
    CUmodule hmod = input_buffer->Get<CUmodule > ();
    CUresult exit_code = cuModuleGetTexRef(&pTexRef, hmod, name);
    printf("DEBUG : %p\n", (void*) pTexRef);
    Buffer * out = new Buffer();
    out->AddMarshal(pTexRef);
    return new Result((cudaError_t) exit_code, out);
}
