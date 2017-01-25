/*
 * gVirtuS -- A GPGPU transparent virtualization component.
 *
 * Copyright (C) 2009-2011  The University of Napoli Parthenope at Naples.
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

#include <map>
#include <errno.h>

/**
 * @file   Backend.cpp
 * @author Giuseppe Coviello <giuseppe.coviello@uniparthenope.it>
 * @author Vincenzo Santopietro <vincenzo.santopietro@uniparthenope.it>
 * @date   Sat Oct 10 10:51:58 2009
 *
 * @brief
 *
 *
 */

#include "CufftHandler.h"

using namespace std;
using namespace log4cplus;

map<string, CufftHandler::CufftRoutineHandler> *CufftHandler::mspHandlers = NULL;

extern "C" int HandlerInit() {
    return 0;
}

extern "C" Handler *GetHandler() {
    return new CufftHandler();
}

CufftHandler::CufftHandler() {
    logger=Logger::getInstance(LOG4CPLUS_TEXT("CufftHandler"));
    Initialize();
}

CufftHandler::~CufftHandler() {

}

bool CufftHandler::CanExecute(std::string routine) {
    return mspHandlers->find(routine) != mspHandlers->end();
}

Result * CufftHandler::Execute(std::string routine, Buffer * input_buffer) {
    LOG4CPLUS_DEBUG(logger,"Called " << routine);
    map<string, CufftHandler::CufftRoutineHandler>::iterator it;
    it = mspHandlers->find(routine);
    if (it == mspHandlers->end())
        throw "No handler for '" + routine + "' found!";
    try {
        return it->second(this, input_buffer);
    } catch (const char *ex) {
        cout << ex << endl;
        cout << strerror(errno) << endl;
    }
    return NULL;
}

/*
 * cufftResult cufftPlan1d(cufftHandle *plan, int nx, cufftType type, int batch);
 * Creates a 1D FFT plan configuration for a specified signal size and data type.
 * The batch input parameter tells cuFFT how many 1D transforms to configure.
 */
CUFFT_ROUTINE_HANDLER(Plan1d) {
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Plan1D"));
    try {
        cufftHandle plan;
        int nx = in->Get<int>();
        cufftType type = in->Get<cufftType> ();
        int batch = in->Get<int>();
        cufftResult ec = cufftPlan1d(&plan, nx, type,batch);
        Buffer *out = new Buffer();
        out->Add(plan);
        return new Result(ec, out);
    } catch (string e) {
        LOG4CPLUS_DEBUG(logger,e);
        return new Result(cudaErrorMemoryAllocation); //???
    }
}

/*
 * cufftResult cufftPlan2d(cufftHandle *plan, int nx, int ny, cufftType type);
 * Creates a 2D FFT plan configuration according to specified signal sizes and data type.
 */
CUFFT_ROUTINE_HANDLER(Plan2d) {
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Plan1D"));
    try {
        cufftHandle plan;
        int nx = in->Get<int>();
        int ny = in->Get<int>();
        cufftType type = in->Get<cufftType > ();
        cufftResult ec = cufftPlan2d(&plan, nx, ny, type);
        Buffer *out = new Buffer();
        out->Add(plan);
        return new Result(ec, out);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return new Result(cudaErrorMemoryAllocation); //???
    }
}

/*
 * cufftResult cufftPlan3d(cufftHandle *plan, int nx, int ny, int nz, cufftType type);
 * Creates a 3D FFT plan configuration according to specified signal sizes and data type.
 * This function is the same as cufftPlan2d() except that it takes a third size parameter nz.
 */
CUFFT_ROUTINE_HANDLER(Plan3d) {
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Plan1D"));
    try{
        cufftHandle plan;
        int nx = in->Get<int>();
        int ny = in->Get<int>();
        int nz = in->Get<int>();
        cufftType type = in->Get<cufftType > ();
        cufftResult ec = cufftPlan3d(&plan, nx, ny, nz, type);
        Buffer *out = new Buffer();
        out->Add(plan);
        return new Result(ec, out);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return new Result(cudaErrorMemoryAllocation); //???
    }
}


CUFFT_ROUTINE_HANDLER(ExecC2R) {
    cufftHandle plan = in->Get<cufftHandle > ();
    cufftComplex *idata = (cufftComplex *) in->Get<uint64_t > ();
    cufftReal *odata = (cufftReal *) in->Get<uint64_t > ();
    return new Result(cufftExecC2R(plan, idata, odata));
}

CUFFT_ROUTINE_HANDLER(SetCompatibilityMode) {
    cufftHandle plan = in->Get<cufftHandle > ();
    cufftCompatibility mode = in->Get<cufftCompatibility > ();
    return new Result(cufftSetCompatibilityMode(plan, mode));
}

CUFFT_ROUTINE_HANDLER(Create) {
    cufftHandle *plan_adv = in->Assign<cufftHandle>();
    cufftResult ec = cufftCreate(plan_adv);
    Buffer *out = new Buffer();
    out->Add(plan_adv);
    cout <<"plan: "<< *plan_adv<<"\n";
    return new Result(ec, out);
}

/**
    cufftXtMakePlanMAny Handler
    @param    plan     cufftHandle returned by cufftCreate
    @param    rank    Dimensionality of the transform (1, 2, or 3)
    @param    n   Array of size rank, describing the size of each dimension, n[0] being the size of the innermost deminsion. For multiple GPUs and rank equal to 1, the sizes must be a power of 2. For multiple GPUs and rank equal to 2 or 3, the sizes must be factorable into primes less than or equal to 127.
    @param    inembed     Pointer of size rank that indicates the storage dimensions of the input data in memory, inembed[0] being the storage dimension of the innermost dimension. If set to NULL all other advanced data layout parameters are ignored.
    @param    istride     Indicates the distance between two successive input elements in the least significant (i.e., innermost) dimension
    @param    idist   Indicates the distance between the first element of two consecutive signals in a batch of the input data
    @param    inputtype   Type of input data.
    @param    onembed     Pointer of size rank that indicates the storage dimensions of the output data in memory, inembed[0] being the storage dimension of the innermost dimension. If set to NULL all other advanced data layout parameters are ignored.
    @param    ostride     Indicates the distance between two successive output elements in the output array in the least significant (i.e., innermost) dimension
    @param    odist   Indicates the distance between the first element of two consecutive signals in a batch of the output data
    @param    outputtype  Type of output data.
    @param    batch   Batch size for this transform
    @param    *workSize   Pointer to the size(s), in bytes, of the work areas. For example for two GPUs worksize must be declared to have two elements.
    @param    executiontype   Type of data to be used for computations.
        
    @return    *workSize   Pointer to the size(s) of the work areas.
*/
CUFFT_ROUTINE_HANDLER(XtMakePlanMany) {
    cufftHandle plan = in->Get<cufftHandle>();
    int rank = in->Get<int>();
    cout <<"HelloXtPlanMAny"<<endl;
    long long int *n = in->Assign<long long int>();//long long int's address -> uint64_t
    long long int *inembed = in->Assign<long long int>();
    long long int istride = in->Get<long long int>();
    long long int idist = in->Get<long long int>();
    cudaDataType inputtype = in->Get<cudaDataType>();

    long long int *onembed = in->Assign<long long int>();
    long long int ostride = in->Get<long long int>();
    long long int odist = in->Get<long long int>();
    cudaDataType outputtype = in->Get<cudaDataType>();
    
    long long int batch = in->Get<long long int>();
    size_t * workSize = in->Assign<size_t>();
    cudaDataType executiontype = in->Get<cudaDataType>();
    
    //Buffer *out = new Buffer();
    //out->Add(workSize);
    //return new Result(ec,out);
    return new Result(cufftXtMakePlanMany(plan,rank,n,inembed,istride,idist,inputtype,onembed,ostride,odist,outputtype,batch,workSize,executiontype));
}

void CufftHandler::Initialize() {
    if (mspHandlers != NULL)
        return;
    mspHandlers = new map<string, CufftHandler::CufftRoutineHandler > ();
    mspHandlers->insert(CUFFT_ROUTINE_HANDLER_PAIR(Plan1d));
    mspHandlers->insert(CUFFT_ROUTINE_HANDLER_PAIR(Plan2d));
    mspHandlers->insert(CUFFT_ROUTINE_HANDLER_PAIR(Plan3d));
    mspHandlers->insert(CUFFT_ROUTINE_HANDLER_PAIR(ExecC2R));
    mspHandlers->insert(CUFFT_ROUTINE_HANDLER_PAIR(SetCompatibilityMode));
    mspHandlers->insert(CUFFT_ROUTINE_HANDLER_PAIR(Create));
    mspHandlers->insert(CUFFT_ROUTINE_HANDLER_PAIR(XtMakePlanMany));
}


