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
#include <fstream>


#include "util/Decoder.h"

using namespace std;
using namespace log4cplus;

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
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("ModuleLoadDataEx"));
    //std::cout <<"Start CULAUNCHKERNEL"<<std::endl;
    LOG4CPLUS_DEBUG(logger,"Start ModuleLoadDataEx");
    
    CUmodule module;
    unsigned int numOptions = input_buffer->Get<unsigned int>();
    CUjit_option *options = input_buffer->AssignAll<CUjit_option > ();
    char *image = input_buffer->AssignString();
    void **optionValues = new void*[numOptions];
    int log_buffer_size_bytes = 1024;
    int error_log_buffer_size_bytes = 1024;
    for (unsigned int i = 0; i < numOptions; i++) {
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
    for (unsigned int i = 0; i < numOptions; i++) {
        if (options[i] == CU_JIT_INFO_LOG_BUFFER || options[i] == CU_JIT_ERROR_LOG_BUFFER) {
            int len_string = strlen((char *) (optionValues[i]));
            out->Add<int>(len_string);
            out->Add((char *) (optionValues[i]), len_string);
        } else{
            out->Add(&optionValues[i]);

        }
           
    }
    LOG4CPLUS_DEBUG(logger,"End ModuleLoadDataEx");
    return new Result((cudaError_t) exit_code, out);
}

/*Returns a handle to a texture-reference.*/
CUDA_DRIVER_HANDLER(ModuleGetTexRef) {
    CUtexref pTexRef;
    char *name = input_buffer->AssignString();
    CUmodule hmod = input_buffer->Get<CUmodule > ();
    CUresult exit_code = cuModuleGetTexRef(&pTexRef, hmod, name);
    Buffer * out = new Buffer();
    out->AddMarshal(pTexRef);
    return new Result((cudaError_t) exit_code, out);
}

/*Load a module's data with options.*/
CUDA_DRIVER_HANDLER(ModuleLoad) {
    Decoder *decoder=new Decoder();
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("ModuleLoad"));
    LOG4CPLUS_DEBUG(logger,"Start ModuleLoad");
    char *fname=input_buffer->AssignString();
    LOG4CPLUS_DEBUG(logger,"Module name:" << fname);
    char *moduleLoad=input_buffer->AssignString();
    LOG4CPLUS_DEBUG(logger,"Calling decoder->Decode");
    std::istringstream iss(moduleLoad);
    fstream fout;
    fout.open("/tmp/file.bin", ios::binary | ios::out);
    decoder->Decode(iss,fout); 
    fout.close();
    CUmodule module;
    CUresult exit_code = cuModuleLoad(&module,"/tmp/file.bin");	
    Buffer * out = new Buffer();
    out->AddMarshal(module);
    return new Result((cudaError_t) exit_code, out);
}

/*Load a module's data with options.*/
CUDA_DRIVER_HANDLER(ModuleLoadFatBinary) {
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("ModuleLoadFatBinary"));
    LOG4CPLUS_DEBUG(logger,"Start ModuleLoadFatBinary");
    char *fname=input_buffer->AssignString();
    LOG4CPLUS_DEBUG(logger,"Module name:" << fname);
    CUmodule module;
    CUresult exit_code = cuModuleLoadFatBinary(&module,fname);
    Buffer * out = new Buffer();
    out->AddMarshal(module);
    return new Result((cudaError_t) exit_code, out);
}

/*Load a module's data with options.*/
CUDA_DRIVER_HANDLER(ModuleUnload) {
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("ModuleUnLoad"));
    LOG4CPLUS_DEBUG(logger,"Start ModuleUnLoad");
    CUmodule module=input_buffer->Get<CUmodule> ();
    Buffer * out = new Buffer();
    CUresult exit_code = cuModuleUnload(module);
    return new Result((cudaError_t) exit_code, out);
}

