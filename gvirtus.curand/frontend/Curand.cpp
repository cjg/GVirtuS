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

#include <iostream>
#include <cstdio>
#include <string>

#include "CurandFrontend.h"

using namespace std;

extern "C" curandStatus_t curandGenerate( curandGenerator_t generator, unsigned int *outputPtr, size_t num){
    CurandFrontend::Prepare();
    
    CurandFrontend::AddVariableForArguments<long long int>((long long int)generator);
    CurandFrontend::AddHostPointerForArguments<unsigned int>(outputPtr);
    CurandFrontend::AddVariableForArguments<size_t>(num);
    
    CurandFrontend::Execute("curandGenerate");
    return CurandFrontend::GetExitCode();
}

extern "C" curandStatus_t curandGenerateLongLong( curandGenerator_t generator, unsigned long long *outputPtr, size_t num){
    CurandFrontend::Prepare();
    
    CurandFrontend::AddVariableForArguments<long long int>((long long int)generator);
    CurandFrontend::AddHostPointerForArguments<unsigned long long>(outputPtr);
    CurandFrontend::AddVariableForArguments<size_t>(num);
    
    CurandFrontend::Execute("curandGenerateLongLong");
    return CurandFrontend::GetExitCode();
}

extern "C" curandStatus_t curandGenerateUniform( curandGenerator_t generator, float *outputPtr, size_t num){
    CurandFrontend::Prepare();
    
    CurandFrontend::AddVariableForArguments<long long int>((long long int)generator);
    CurandFrontend::AddHostPointerForArguments<float>(outputPtr);
    CurandFrontend::AddVariableForArguments<size_t>(num);
    
    CurandFrontend::Execute("curandGenerateUniform");
    return CurandFrontend::GetExitCode();
}

extern "C" curandStatus_t curandGenerateNormal( curandGenerator_t generator, float *outputPtr, size_t n, float mean, float stddev){
    CurandFrontend::Prepare();
    
    CurandFrontend::AddVariableForArguments<long long int>((long long int)generator);
    CurandFrontend::AddHostPointerForArguments<float>(outputPtr);
    CurandFrontend::AddVariableForArguments<size_t>(n);
    CurandFrontend::AddVariableForArguments<float>(mean);
    CurandFrontend::AddVariableForArguments<float>(stddev);
    
    CurandFrontend::Execute("curandGenerateNormal");
    return CurandFrontend::GetExitCode();
}


extern "C" curandStatus_t curandGenerateLogNormal( curandGenerator_t generator, float *outputPtr, size_t n, float mean, float stddev){
    CurandFrontend::Prepare();
    
    CurandFrontend::AddVariableForArguments<long long int>((long long int)generator);
    CurandFrontend::AddHostPointerForArguments<float>(outputPtr);
    CurandFrontend::AddVariableForArguments<size_t>(n);
    CurandFrontend::AddVariableForArguments<float>(mean);
    CurandFrontend::AddVariableForArguments<float>(stddev);
    
    CurandFrontend::Execute("curandGenerateLogNormal");
    return CurandFrontend::GetExitCode();
}

