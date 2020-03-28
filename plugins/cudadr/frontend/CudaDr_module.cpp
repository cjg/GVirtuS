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

#include <fstream>
#include <iostream>
#include <string>
#include <sstream>
#include <iomanip>
#include <cstring>
#include "CudaDrFrontend.h"
#include "CudaUtil.h"
#include "CudaDr.h"
#include <cuda.h>

#include "util/Encoder.h"

using namespace std;

/*Load a module's data. */
extern CUresult cuModuleLoadData(CUmodule *module, const void *image) {
    CudaDrFrontend::Prepare();
    CudaDrFrontend::AddStringForArguments((char *) image);
    CudaDrFrontend::Execute("cuModuleLoadData");
    if (CudaDrFrontend::Success())
        *module = (CUmodule) (CudaDrFrontend::GetOutputDevicePointer());
    return (CUresult) (CudaDrFrontend::GetExitCode());

}

/*Returns a function handle*/
extern CUresult cuModuleGetFunction(CUfunction *hfunc, CUmodule hmod, const char *name) {
    CudaDrFrontend::Prepare();
    CudaDrFrontend::AddStringForArguments((char*) name);
    CudaDrFrontend::AddDevicePointerForArguments((void*) hmod);
    CudaDrFrontend::Execute("cuModuleGetFunction");
    void* tmp;
    if (CudaDrFrontend::Success()) {
        tmp = (CUfunction) (CudaDrFrontend::GetOutputDevicePointer());
        *hfunc = (CUfunction) tmp;
    }
    return (CUresult) (CudaDrFrontend::GetExitCode());
}

/*Returns a global pointer from a module.*/
extern CUresult cuModuleGetGlobal(CUdeviceptr *dptr, size_t *bytes, CUmodule hmod, const char *name) {
    CudaDrFrontend::Prepare();
    CudaDrFrontend::AddStringForArguments((char*) name);
    CudaDrFrontend::AddDevicePointerForArguments((void*) hmod);
    CudaDrFrontend::Execute("cuModuleGetGlobal");
    if (CudaDrFrontend::Success()) {
        *dptr = (CUdeviceptr) (CudaDrFrontend::GetOutputDevicePointer());
        *bytes = (size_t) (CudaDrFrontend::GetOutputDevicePointer());
    }
    return (CUresult) (CudaDrFrontend::GetExitCode());
}

/*Returns a handle to a texture-reference.*/
extern CUresult cuModuleGetTexRef(CUtexref *pTexRef, CUmodule hmod, const char *name) {
    CudaDrFrontend::Prepare();
    CudaDrFrontend::AddStringForArguments((char*) name);
    CudaDrFrontend::AddDevicePointerForArguments((void*) hmod);
    CudaDrFrontend::Execute("cuModuleGetTexRef");
    if (CudaDrFrontend::Success()) {
        *pTexRef = (CUtexref) (CudaDrFrontend::GetOutputDevicePointer());
    }
    return (CUresult) (CudaDrFrontend::GetExitCode());
}

/*Load a module's data with options.*/
extern CUresult cuModuleLoadDataEx(CUmodule *module, const void *image, unsigned int numOptions, CUjit_option *options, void **optionValues) {
    CudaDrFrontend::Prepare();
    char *tmp;
    char *tmp2;
    CudaDrFrontend::AddVariableForArguments(numOptions);
    CudaDrFrontend::AddHostPointerForArguments(options, numOptions);
    CudaDrFrontend::AddStringForArguments((char *) image);
    for (unsigned int i = 0; i < numOptions; i++) {
        CudaDrFrontend::AddHostPointerForArguments(&optionValues[i]);
    }
    CudaDrFrontend::Execute("cuModuleLoadDataEx");
    if (CudaDrFrontend::Success()) {
        *module = (CUmodule) (CudaDrFrontend::GetOutputDevicePointer());
        int len_str;
        for (unsigned int i = 0; i < numOptions-1; i++) {
            switch (options[i]) {
                case CU_JIT_INFO_LOG_BUFFER:
                    len_str = CudaDrFrontend::GetOutputVariable<int>();
                    if (len_str >0){
                        tmp = CudaDrFrontend::GetOutputHostPointer<char>(len_str);
                    strcpy((char *) *(optionValues + i), tmp);
                    }
                 
                    break;
                case CU_JIT_ERROR_LOG_BUFFER:
                    tmp2 = (CudaDrFrontend::GetOutputString());
                    strcpy((char *) *(optionValues + i), tmp2);
                    break;

                default:
                    *(optionValues + i) = (void *) (*(CudaDrFrontend::GetOutputHostPointer<unsigned int>()));

            }
            
        }
    }
    return (CUresult) (CudaDrFrontend::GetExitCode());
}

extern CUresult cuModuleLoad(CUmodule *module, const char *fname) {
    Encoder *encoder=new Encoder();
    std::ostringstream res;
    std::ifstream input("helloWorldDriverAPI.ptx", std::ios::binary);
    encoder->Encode(input,res);
    //input >> std::noskipws;
    
    std::string moduleLoad = res.str();
    cout << "Encoded module:\n"; 
    cout << moduleLoad;
    cout << "--------------\n";

    CudaDrFrontend::Prepare();
    CudaDrFrontend::AddStringForArguments((char *) fname);
    CudaDrFrontend::AddStringForArguments(moduleLoad.c_str());
    CudaDrFrontend::Execute("cuModuleLoad");
    if (CudaDrFrontend::Success())
        *module = (CUmodule) (CudaDrFrontend::GetOutputDevicePointer());
    return (CUresult) (CudaDrFrontend::GetExitCode());

    // FIXME: implement
    //cerr << "*** Error: cuModuleLoad not yet implemented!" << endl;
    //return (CUresult) 1;
}

extern CUresult cuModuleLoadFatBinary(CUmodule *module, const void *fatCubin) {
    CudaDrFrontend::Prepare();
    CudaDrFrontend::AddStringForArguments((char *) fatCubin);
    CudaDrFrontend::Execute("cuModuleLoadFatBinary");
    if (CudaDrFrontend::Success())
        *module = (CUmodule) (CudaDrFrontend::GetOutputDevicePointer());
    return (CUresult) (CudaDrFrontend::GetExitCode());
    // FIXME: implement
    //cerr << "*** Error: cuModuleLoadFatBinary not yet implemented!" << endl;
    //return (CUresult) 1;
}

extern CUresult cuModuleUnload(CUmodule hmod) {
    CudaDrFrontend::Prepare();
    CudaDrFrontend::AddDevicePointerForArguments((char *)hmod);
    CudaDrFrontend::Execute("cuModuleUnload");
    return (CUresult) (CudaDrFrontend::GetExitCode());
    // FIXME: implement
    //cerr << "*** Error: cuModuleUnload not yet implemented!" << endl;
    //return (CUresult) 1;
}
