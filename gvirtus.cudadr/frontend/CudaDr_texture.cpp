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


#include <cstring>
#include "CudaDrFrontend.h"
#include "CudaUtil.h"
#include "CudaDr.h"
#include <cuda.h>
#include <stdio.h>


using namespace std;

/*Binds an address as a texture reference.*/
extern CUresult cuTexRefSetArray(CUtexref hTexRef, CUarray hArray, unsigned int Flags) {
    CudaDrFrontend::Prepare();
    CudaDrFrontend::AddDevicePointerForArguments((void*) hArray);
    CudaDrFrontend::AddVariableForArguments(Flags);
    CudaDrFrontend::AddDevicePointerForArguments((void*) hTexRef);
    CudaDrFrontend::Execute("cuTexRefSetArray");
    return (CUresult) (CudaDrFrontend::GetExitCode());
}

/*Sets the addressing mode for a texture reference.*/
extern CUresult cuTexRefSetAddressMode(CUtexref hTexRef, int dim, CUaddress_mode am) {
    CudaDrFrontend::Prepare();
    CudaDrFrontend::AddVariableForArguments(dim);
    CudaDrFrontend::AddVariableForArguments(am);
    CudaDrFrontend::AddDevicePointerForArguments((void*) hTexRef);
    CudaDrFrontend::Execute("cuTexRefSetAddressMode");
    return (CUresult) (CudaDrFrontend::GetExitCode());
}

/*Sets the filtering mode for a texture reference.*/
extern CUresult cuTexRefSetFilterMode(CUtexref hTexRef, CUfilter_mode fm) {
    CudaDrFrontend::Prepare();
    CudaDrFrontend::AddVariableForArguments(fm);
    CudaDrFrontend::AddDevicePointerForArguments((void*) hTexRef);
    CudaDrFrontend::Execute("cuTexRefSetFilterMode");
    return (CUresult) (CudaDrFrontend::GetExitCode());
}

/*Sets the flags for a texture reference.*/
extern CUresult cuTexRefSetFlags(CUtexref hTexRef, unsigned int Flags) {
    CudaDrFrontend::Prepare();
    CudaDrFrontend::AddVariableForArguments(Flags);
    CudaDrFrontend::AddDevicePointerForArguments((void*) hTexRef);
    CudaDrFrontend::Execute("cuTexRefSetFlags");
    return (CUresult) (CudaDrFrontend::GetExitCode());
}

/*Sets the format for a texture reference.*/
extern CUresult cuTexRefSetFormat(CUtexref hTexRef, CUarray_format fmt, int NumPackedComponents) {
    CudaDrFrontend::Prepare();
    CudaDrFrontend::AddVariableForArguments(NumPackedComponents);
    CudaDrFrontend::AddVariableForArguments(fmt);
    CudaDrFrontend::AddDevicePointerForArguments((void*) hTexRef);
    CudaDrFrontend::Execute("cuTexRefSetFormat");
    return (CUresult) (CudaDrFrontend::GetExitCode());
}

/*Gets the address associated with a texture reference. */
extern CUresult cuTexRefGetAddress(CUdeviceptr *pdptr, CUtexref hTexRef) {
    CudaDrFrontend::Prepare();
    CudaDrFrontend::AddDevicePointerForArguments((void*) hTexRef);
    CudaDrFrontend::Execute("cuTexRefGetAddress");
    if (CudaDrFrontend::Success())
        *pdptr = (CUdeviceptr) (CudaDrFrontend::GetOutputDevicePointer());
    return (CUresult) CudaDrFrontend::GetExitCode();
}

/*Gets the array bound to a texture reference.*/
extern CUresult cuTexRefGetArray(CUarray *phArray, CUtexref hTexRef) {
    CudaDrFrontend::Prepare();
    CudaDrFrontend::AddDevicePointerForArguments((void*) hTexRef);
    CudaDrFrontend::Execute("cuTexRefGetArray");
    if (CudaDrFrontend::Success())
        *phArray = (CUarray) (CudaDrFrontend::GetOutputDevicePointer());
    return (CUresult) CudaDrFrontend::GetExitCode();
}

/*Gets the flags used by a texture reference. */
extern CUresult cuTexRefGetFlags(unsigned int *pFlags, CUtexref hTexRef) {
    CudaDrFrontend::Prepare();
    CudaDrFrontend::AddDevicePointerForArguments((void*) hTexRef);
    CudaDrFrontend::Execute("cuTexRefGetFlags");
    if (CudaDrFrontend::Success())
        *pFlags = *(CudaDrFrontend::GetOutputHostPointer<size_t > ());
    return (CUresult) CudaDrFrontend::GetExitCode();
}

/*Binds an address as a texture reference.*/
extern CUresult cuTexRefSetAddress(size_t *ByteOffset, CUtexref hTexRef, CUdeviceptr dptr, size_t bytes) {
    CudaDrFrontend::Prepare();
    CudaDrFrontend::AddDevicePointerForArguments((void*) hTexRef);
    CudaDrFrontend::AddVariableForArguments(dptr);
    CudaDrFrontend::AddVariableForArguments(bytes);
    CudaDrFrontend::Execute("cuTexRefSetAddress");
    if (CudaDrFrontend::Success())
        *ByteOffset = *(CudaDrFrontend::GetOutputHostPointer<size_t > ());
    return (CUresult) CudaDrFrontend::GetExitCode();
}

extern CUresult cuTexRefGetAddressMode(CUaddress_mode *pam, CUtexref hTexRef, int dim) {
    // FIXME: implement
    cerr << "*** Error: cuTexRefGetAddressMode() not yet implemented!" << endl;
    return (CUresult) 1;
}

extern CUresult cuTexRefGetFilterMode(CUfilter_mode *pfm, CUtexref hTexRef) {
    // FIXME: implement
    cerr << "*** Error: cuTexRefGetFilterMode() not yet implemented!" << endl;
    return (CUresult) 1;
}

extern CUresult cuTexRefGetFormat(CUarray_format *pFormat, int *pNumChannels, CUtexref hTexRef) {
    // FIXME: implement
    cerr << "*** Error: cuTexRefGetFormat() not yet implemented!" << endl;
    return (CUresult) 1;
}

extern CUresult cuTexRefSetAddress2D(CUtexref hTexRef, const CUDA_ARRAY_DESCRIPTOR *desc, CUdeviceptr dptr, size_t Pitch) {
    // FIXME: implement
    cerr << "*** Error: cuTexRefSetAddress2D() not yet implemented!" << endl;
    return (CUresult) 1;
}

extern CUresult cuTexRefCreate(CUtexref *pTexRef) {
    // FIXME: implement
    cerr << "*** Error: cuTexRefCreate() not yet implemented! : DEPRECATED" << endl;
    return (CUresult) 1;
}

extern CUresult cuTexRefDestroy(CUtexref hTexRef) {
    // FIXME: implement
    cerr << "*** Error: cuTexRefDestroy() not yet implemented! : DEPRECATED" << endl;
    return (CUresult) 1;
}
