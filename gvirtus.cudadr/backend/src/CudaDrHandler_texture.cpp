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

/*Binds an address as a texture reference. */
CUDA_DRIVER_HANDLER(TexRefSetArray) {
    CUarray hArray = input_buffer->GetFromMarshal<CUarray > ();
    unsigned int Flags = input_buffer->Get<unsigned int>();
    CUtexref hTexRef = input_buffer->Get<CUtexref > ();
    CUresult exit_code = cuTexRefSetArray(hTexRef, hArray, Flags);
    return std::make_shared<Result>((cudaError_t) exit_code);
}

/*Sets the addressing mode for a texture reference.*/
CUDA_DRIVER_HANDLER(TexRefSetAddressMode) {
    int dim = input_buffer->Get<int>();
    CUaddress_mode am = input_buffer->Get<CUaddress_mode > ();
    CUtexref hTexRef = input_buffer->Get<CUtexref > ();
    CUresult exit_code = cuTexRefSetAddressMode(hTexRef, dim, am);
    return std::make_shared<Result>((cudaError_t) exit_code);
}

/*Gets the filter-mode used by a texture reference.*/
CUDA_DRIVER_HANDLER(TexRefSetFilterMode) {
    CUfilter_mode fm = input_buffer->Get<CUfilter_mode > ();
    CUtexref hTexRef = input_buffer->Get<CUtexref > ();
    CUresult exit_code = cuTexRefSetFilterMode(hTexRef, fm);
    return std::make_shared<Result>((cudaError_t) exit_code);
}

/*Sets the flags for a texture reference.*/
CUDA_DRIVER_HANDLER(TexRefSetFlags) {
    unsigned int Flags = input_buffer->Get<unsigned int>();
    CUtexref hTexRef = input_buffer->Get<CUtexref > ();
    CUresult exit_code = cuTexRefSetFlags(hTexRef, Flags);
    return std::make_shared<Result>((cudaError_t) exit_code);
}

/*Sets the format for a texture reference. */
CUDA_DRIVER_HANDLER(TexRefSetFormat) {
    int NumPackedComponents = input_buffer->Get<int>();
    CUarray_format fmt = input_buffer->Get<CUarray_format > ();
    CUtexref hTexRef = input_buffer->Get<CUtexref > ();
    CUresult exit_code = cuTexRefSetFormat(hTexRef, fmt, NumPackedComponents);
    return std::make_shared<Result>((cudaError_t) exit_code);
}

/*Gets the address associated with a texture reference. */
CUDA_DRIVER_HANDLER(TexRefGetAddress) {
    CUdeviceptr pdptr;
    CUtexref hTexRef = input_buffer->Get<CUtexref > ();
    CUresult exit_code = cuTexRefGetAddress(&pdptr, hTexRef);
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    out->AddMarshal(pdptr);
    return std::make_shared<Result>((cudaError_t) exit_code, out);
}

/*Gets the array bound to a texture reference.*/
CUDA_DRIVER_HANDLER(TexRefGetArray) {
    CUarray hArray;
    CUtexref hTexRef = input_buffer->Get<CUtexref > ();
    CUresult exit_code = cuTexRefGetArray(&hArray, hTexRef);
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    out->AddMarshal(hArray);
    return std::make_shared<Result>((cudaError_t) exit_code, out);
}

/*Gets the flags used by a texture reference. */
CUDA_DRIVER_HANDLER(TexRefGetFlags) {
    unsigned int pFlags;
    CUtexref hTexRef = input_buffer->Get<CUtexref > ();
    CUresult exit_code = cuTexRefGetFlags(&pFlags, hTexRef);
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    out->Add(pFlags);
    return std::make_shared<Result>((cudaError_t) exit_code, out);
}

/*Binds an address as a texture reference.*/
CUDA_DRIVER_HANDLER(TexRefSetAddress) {
    size_t ByteOffset;
    CUtexref hTexRef = input_buffer->Get<CUtexref > ();
    CUdeviceptr dptr = input_buffer->Get<CUdeviceptr > ();
    size_t bytes = input_buffer->Get<size_t > ();
    CUresult exit_code = cuTexRefSetAddress(&ByteOffset, hTexRef, dptr, bytes);
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    out->Add(ByteOffset);
    return std::make_shared<Result>((cudaError_t) exit_code, out);
}

