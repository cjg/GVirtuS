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
#include <bits/stl_map.h>

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
    
    cufftHandle *plan_adv = in->Assign<cufftHandle>();
    int nx = in->Get<int>();
    cufftType type = in->Get<cufftType>();
    int batch = in->Get<int>();
    
    cufftResult exit_code = cufftPlan1d(plan_adv, nx, type,batch);
    Buffer *out = new Buffer();
    
    try{
        out->Add(plan_adv);
    } catch (string e) {
        LOG4CPLUS_DEBUG(logger,e);
        return new Result(cudaErrorMemoryAllocation); //???
    }
    cout <<"DEBUG - Plan: "<< *plan_adv<<"\n";
    cout<<"DEBUG - cufftPlan1d Executed\n";
    return new Result(exit_code, out);
}

/*
 * cufftResult cufftPlan2d(cufftHandle *plan, int nx, int ny, cufftType type);
 * Creates a 2D FFT plan configuration according to specified signal sizes and data type.
 */
CUFFT_ROUTINE_HANDLER(Plan2d) {
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Plan2d"));

    cufftHandle plan;// = in->Assign<cufftHandle>();
    int nx = in->Get<int>();
    int ny = in->Get<int>();
    cufftType type = in->Get<cufftType>();
    
    cufftResult exit_code = cufftPlan2d(&plan, nx, ny, type);
    Buffer *out = new Buffer();
    try {
        out->Add(&plan);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return new Result(cudaErrorMemoryAllocation); //???
    }
    cout <<"DEBUG - Plan: "<< plan<<"\n";
    cout<<"DEBUG - cufftPlan2d Executed\n";
    return new Result(exit_code, out);
}

/*
 * cufftResult cufftPlan3d(cufftHandle *plan, int nx, int ny, int nz, cufftType type);
 * Creates a 3D FFT plan configuration according to specified signal sizes and data type.
 * This function is the same as cufftPlan2d() except that it takes a third size parameter nz.
 */
CUFFT_ROUTINE_HANDLER(Plan3d) {
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Plan3D"));
    try{
        cufftHandle *plan = in->Assign<cufftHandle>();
        int nx = in->Get<int>();
        int ny = in->Get<int>();
        int nz = in->Get<int>();
        cufftType type = in->Get<cufftType>();
        cufftResult ec = cufftPlan3d(plan, nx, ny, nz, type);
        Buffer *out = new Buffer();
        out->Add(plan);
        return new Result(ec, out);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return new Result(cudaErrorMemoryAllocation); //???
    }
}

/*
 * Creates a FFT plan configuration of dimension rank, with sizes specified in the array n. 
 * The batch input parameter tells cuFFT how many transforms to configure. With this function, batched plans of 1, 2, or 3 dimensions may be created.
 */
CUFFT_ROUTINE_HANDLER(PlanMany) {
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Plan3D"));
    
    cufftHandle *plan = in->Assign<cufftHandle>();
    int rank = in->Get<int>();
    int * n = in->Assign<int>();
    int * inembed = in->Assign<int>();
    int istride = in->Get<int>();
    int idist = in->Get<int>();
    
    int * onembed = in->Assign<int>();
    int ostride = in->Get<int>();
    int odist = in->Get<int>();
    
    cufftType type = in->Get<cufftType>();
    int batch = in->Get<int>();
    try{
        cufftResult exit_code = cufftPlanMany(plan,rank,n,inembed,istride,idist,onembed,ostride,odist,type,batch);
        Buffer *out = new Buffer();
        out->Add(plan);
        return new Result(exit_code, out);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return new Result(cudaErrorMemoryAllocation); //???
    }
}


CUFFT_ROUTINE_HANDLER(ExecC2R) {
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("ExecC2R"));
    
    cufftHandle plan = in->Get<cufftHandle>();
    cufftReal *odata;
    cufftComplex *idata;
    
    idata = (in->GetFromMarshal<cufftComplex*>());
    
    try{
        odata = (in->GetFromMarshal<cufftReal*>());    
    } catch (std::string e){
        odata = (cufftReal*) idata;
        cout << e <<endl;
    }
    
    cufftResult exit_code = cufftExecC2R(plan,idata,odata);
    
    Buffer *out = new Buffer();
    try{
        out->AddMarshal(odata);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return new Result(cudaErrorMemoryAllocation);
    }
    cout<<"DEBUG - cufftExecC2R Executed\n";
    return new Result(exit_code,out);
}

CUFFT_ROUTINE_HANDLER(SetCompatibilityMode) {
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("SetCompatibilityMode"));
    cufftHandle plan = in->Get<cufftHandle>();
    cufftCompatibility mode = in->Get<cufftCompatibility>();
    cufftResult exit_code = cufftSetCompatibilityMode(plan, mode);
    cout<<"DEBUG - cufftSetCompatibilityMode Executed\n";
    return new Result(exit_code);
}

CUFFT_ROUTINE_HANDLER(Create) {
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("Create"));
    cufftHandle *plan_adv = in->Assign<cufftHandle>();
    cufftResult exit_code = cufftCreate(plan_adv);
    Buffer *out = new Buffer();
    try {
        out->Add(plan_adv);
    } catch (string e) {
        LOG4CPLUS_DEBUG(logger,e);
        return new Result(cudaErrorMemoryAllocation);
    }
    cout <<"DEBUG - Plan: "<< *plan_adv<<"\n";
    cout<<"DEBUG - cufftCreate Executed\n";
    
    return new Result(exit_code, out);
    //return new Result(ec, out);
}
/*
 * cufftResult cufftDestroy(cufftHandle plan);
 * Frees all GPU resources associated with a cuFFT plan and destroys the internal plan data structure.
 * This function should be called once a plan is no longer needed, to avoid wasting GPU memory.
 */
CUFFT_ROUTINE_HANDLER(Destroy) {
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("Create"));
    
    cufftHandle plan = in->Get<cufftHandle>();
    cufftResult exit_code = cufftDestroy(plan);
    
    cout << "DEBUG - cufftDestroy Executed\n";
    return new Result(exit_code);
}

CUFFT_ROUTINE_HANDLER(SetWorkArea){
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("SetWorkArea"));
    cufftHandle plan = in->Get<cufftHandle>();
    void * workArea = in->GetFromMarshal<void*>();
    cufftResult exit_code = cufftSetWorkArea(plan,workArea);
    cout << "DEBUG - cufftSetWorkArea Executed\n";
    return new Result(exit_code);
}


CUFFT_ROUTINE_HANDLER(SetAutoAllocation){
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("SetAutoallocation"));
    cufftHandle plan = in->Get<cufftHandle>();
    int autoAllocate = in->Get<int>();
    cufftResult exit_code = cufftSetAutoAllocation(plan,autoAllocate);
    cout << "DEBUG - cufftSetAutoallocation Executed\n";
    return new Result(exit_code);
}

#if __CUDA_API_VERSION >= 7000
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
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("XtMakePlanMany"));
    
    cufftHandle plan = in->Get<cufftHandle>();
    int rank = in->Get<int>();
    long long int *n = in->Assign<long long int>();
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
    
    cufftResult exit_code = cufftXtMakePlanMany(plan,rank,n,inembed,istride,idist,inputtype,onembed,ostride,odist,outputtype,batch,workSize,executiontype);
    Buffer *out = new Buffer();
    try{
        //out->Add(n);
        //out->Add(inembed);
        //out->Add(onembed);
        out->Add(workSize);   
    } catch (string e) {
        LOG4CPLUS_DEBUG(logger,e);
        return new Result(cudaErrorMemoryAllocation);
    }
    cout<<"DEBUG - cufftXtMakePlanMany Executed\n";
    return new Result(exit_code,out);
}
#endif

/*
 *cufftResult cufftExecC2C(cufftHandle plan, cufftComplex *idata, cufftComplex *odata, int direction);
 *cufftExecC2C() (cufftExecZ2Z()) executes a single-precision (double-precision) complex-to-complex transform plan in the transform direction as specified by direction parameter.
 *cuFFT uses the GPU memory pointed to by the idata parameter as input data. 
 *This function stores the Fourier coefficients in the odata array. If idata and odata are the same, this method does an in-place transform. 
 */
CUFFT_ROUTINE_HANDLER(ExecC2C){
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("ExecC2C"));
    
    cufftHandle plan = in->Get<cufftHandle>();
    cufftComplex *idata,*odata;
    
    idata = (in->GetFromMarshal<cufftComplex*>());
    
    try{
        odata = (in->GetFromMarshal<cufftComplex*>());    
    } catch (std::string e){
        odata = idata;
        cout << e <<endl;
    }
    
    int direction = in->Get<int>();
    
    cufftResult exit_code = cufftExecC2C(plan,idata,odata,direction);
    
    Buffer *out = new Buffer();
    try{
        out->AddMarshal(odata);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return new Result(cudaErrorMemoryAllocation);
    }
    cout<<"DEBUG - cufftExecC2C Executed\n";
    return new Result(exit_code,out);
}

CUFFT_ROUTINE_HANDLER(ExecR2C){
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("ExecR2C"));
    
    cufftHandle plan = in->Get<cufftHandle>();
    cufftReal *idata;
    cufftComplex *odata;
    idata = (in->GetFromMarshal<cufftReal*>());
    try{
        odata = (in->GetFromMarshal<cufftComplex*>());    
    } catch (std::string e){
        odata = (cufftComplex*) idata;
    }
    
    cufftResult exit_code = cufftExecR2C(plan,idata,odata);
    
    Buffer *out = new Buffer();
    try{
        out->AddMarshal(odata);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return new Result(cudaErrorMemoryAllocation);
    }
    cout<<"DEBUG - cufftExecR2C Executed\n";
    return new Result(exit_code,out);
}


CUFFT_ROUTINE_HANDLER(ExecZ2Z){
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("ExecZ2Z"));
    
    cufftHandle plan = in->Get<cufftHandle>();
    cufftDoubleComplex *idata,*odata;
    idata = (in->GetFromMarshal<cufftDoubleComplex*>());
    
    try{
        odata = (in->GetFromMarshal<cufftDoubleComplex*>());    
    } catch (std::string e){
        odata = idata;
        cout << e <<endl;
    }
    int direction = in->Get<int>();
    cufftResult exit_code = cufftExecZ2Z(plan,idata,odata,direction);
    
    Buffer *out = new Buffer();
    try{
        out->AddMarshal(odata);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return new Result(cudaErrorMemoryAllocation);
    }
    cout<<"DEBUG - cufftExecZ2Z Executed\n";
    return new Result(exit_code,out);
}


CUFFT_ROUTINE_HANDLER(ExecD2Z){
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("ExecD2Z"));
    
    cufftHandle plan = in->Get<cufftHandle>();
    cufftDoubleReal *idata;
    cufftDoubleComplex *odata;
    idata = (in->GetFromMarshal<cufftDoubleReal*>());
    
    try{
        odata = (in->GetFromMarshal<cufftDoubleComplex*>());    
    } catch (std::string e){
        odata = (cufftDoubleComplex*)idata;
        cout << e <<endl;
    }
    
    int direction = in->BackGet<int>();
    cufftResult exit_code = cufftExecD2Z(plan,idata,odata);
    
    Buffer *out = new Buffer();
    try{
        out->AddMarshal(odata);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return new Result(cudaErrorMemoryAllocation);
    }
    cout<<"DEBUG - cufftExecD2Z Executed\n";
    return new Result(exit_code,out);
}


CUFFT_ROUTINE_HANDLER(ExecZ2D){
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("ExecZ2D"));
    
    cufftHandle plan = in->Get<cufftHandle>();
    cufftDoubleComplex *idata;
    cufftDoubleReal *odata;
    idata = (in->GetFromMarshal<cufftDoubleComplex*>());
    
    try{
        odata = (in->GetFromMarshal<cufftDoubleReal*>());    
    } catch (std::string e){
        odata = (cufftDoubleReal*)idata;
        cout << e <<endl;
    }
    cufftResult exit_code = cufftExecZ2D(plan,idata,odata);
    
    Buffer *out = new Buffer();
    try{
        out->AddMarshal(odata);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return new Result(cudaErrorMemoryAllocation);
    }
    cout<<"DEBUG - cufftExecZ2D Executed\n";
    return new Result(exit_code,out);
}


/*
 * cufftResult 
    cufftXtSetGPUs(cufftHandle plan, int nGPUs, int *whichGPUs);
 *  cufftXtSetGPUs() indentifies which GPUs are to be used with the plan. 
 * @brief As in the single GPU case cufftCreate() creates a plan and cufftMakePlan*() does the plan generation.This call will return an error if a non-default stream has been associated with the plan.
 * @param plan
 * @param nGPUs
 * @param *whichGPUs
 */
CUFFT_ROUTINE_HANDLER(XtSetGPUs){
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("XtSetGPUs"));
    
    cufftHandle plan = in->Get<cufftHandle>();
    int nGPUs = in->Get<int>();
    int *whichGPUs = (in->Assign<int>());
    cufftResult exit_code = cufftXtSetGPUs(plan,nGPUs,whichGPUs);
    
    cout<<"DEBUG - cufftXtSetGPUs Executed\n";
    return new Result(exit_code);
}


CUFFT_ROUTINE_HANDLER(Estimate1d) {
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("Estimate1d"));
    
    int nx = in->Get<int>();
    cufftType type = in->Get<cufftType>();
    int batch = in->Get<int>();
    size_t * workSize = (in->Assign<size_t>());
    cufftResult exit_code = cufftEstimate1d(nx,type,batch,workSize);
    
    Buffer *out = new Buffer();
    try{
        out->Add(workSize);
    } catch (string e){
        cout << 'DEBUG - ' <<  e << endl;
        LOG4CPLUS_DEBUG(logger,e);
        return new Result(cudaErrorMemoryAllocation);
    }
    cout<<"DEBUG - cufftEstimate1d Executed"<<endl;
    return new Result(exit_code,out);
}

CUFFT_ROUTINE_HANDLER(Estimate2d) {
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("Estimate2d"));
    
    int nx = in->Get<int>();
    int ny = in->Get<int>();
    cufftType type = in->Get<cufftType>();
    size_t * workSize = (in->Assign<size_t>());
    cufftResult exit_code = cufftEstimate2d(nx,ny,type,workSize);
    
    Buffer *out = new Buffer();
    try{
        out->Add(workSize);
    } catch (string e){
        cout << 'DEBUG - ' <<  e << endl;
        LOG4CPLUS_DEBUG(logger,e);
        return new Result(cudaErrorMemoryAllocation);
    }
    cout<<"DEBUG - cufftEstimate2d Executed"<<endl;
    return new Result(exit_code,out);
}


CUFFT_ROUTINE_HANDLER(Estimate3d) {
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("Estimate3d"));
    
    int nx = in->Get<int>();
    int ny = in->Get<int>();
    int nz = in->Get<int>();
    cufftType type = in->Get<cufftType>();
    size_t * workSize = (in->Assign<size_t>());
    cufftResult exit_code = cufftEstimate3d(nx,ny,nz,type,workSize);
    
    Buffer *out = new Buffer();
    try{
        out->Add(workSize);
    } catch (string e){
        cout << 'DEBUG - ' <<  e << endl;
        LOG4CPLUS_DEBUG(logger,e);
        return new Result(cudaErrorMemoryAllocation);
    }
    cout<<"DEBUG - cufftEstimate3d Executed"<<endl;
    return new Result(exit_code,out);
}

CUFFT_ROUTINE_HANDLER(EstimateMany) {
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("EstimateMany"));
    
    int rank = in->Get<int>();
    int * n = in->Assign<int>();
    int * inembed = in->Assign<int>();
    int istride = in->Get<int>();
    int idist = in->Get<int>();
    
    int * onembed = in->Assign<int>();
    int ostride = in->Get<int>();
    int odist = in->Get<int>();
    
    cufftType type = in->Get<cufftType>();
    int batch = in->Get<int>();
    size_t * workSize = (in->Assign<size_t>());
    
    cufftResult exit_code = cufftEstimateMany(rank,n,inembed,istride,idist,onembed,ostride,odist,type,batch,workSize);
    
    Buffer *out = new Buffer();
    try{
        out->Add(workSize);
    } catch (string e){
        cout << 'DEBUG - ' <<  e << endl;
        LOG4CPLUS_DEBUG(logger,e);
        return new Result(cudaErrorMemoryAllocation);
    }
    cout<<"DEBUG - cufftEstimateMany Executed"<<endl;
    return new Result(exit_code,out);
}


/*
 * cufftResult 
    cufftMakePlan1d(cufftHandle plan, int nx, cufftType type, int batch, 
        size_t *workSize);
 * @brief Following a call to cufftCreate() makes a 1D FFT plan configuration for a specified signal size and data type. The batch input parameter tells cuFFT how many 1D transforms to configure.
 * @param plan
 * @param nx
 * @param type
 * @param batch
 * @param *workSize 
 */
CUFFT_ROUTINE_HANDLER(MakePlan1d) {
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("MakePlan1d"));
    
    cufftHandle plan = in->Get<cufftHandle>();
    int nx = in->Get<int>();
    cufftType type = in->Get<cufftType>();
    int batch = in->Get<int>();
    size_t * workSize = (in->Assign<size_t>());
    cufftResult exit_code = cufftMakePlan1d(plan,nx,type,batch,workSize);
    
    Buffer *out = new Buffer();
    try{
        out->Add(workSize);
    } catch (string e){
        cout << 'DEBUG - ' <<  e << endl;
        LOG4CPLUS_DEBUG(logger,e);
        return new Result(cudaErrorMemoryAllocation);
    }
    cout<<"DEBUG - cufftMakePlan1d Executed"<<endl;
    return new Result(exit_code,out);
}

CUFFT_ROUTINE_HANDLER(MakePlan2d) {
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("MakePlan2d"));
    
    cufftHandle plan = in->Get<cufftHandle>();
    int nx = in->Get<int>();
    int ny = in->Get<int>();
    cufftType type = in->Get<cufftType>();
    size_t * workSize = (in->Assign<size_t>());
    cufftResult exit_code = cufftMakePlan2d(plan,nx,ny,type,workSize);
    
    Buffer *out = new Buffer();
    try{
        out->Add(workSize);
    } catch (string e){
        cout << 'DEBUG - ' <<  e << endl;
        LOG4CPLUS_DEBUG(logger,e);
        return new Result(cudaErrorMemoryAllocation);
    }
    cout<<"DEBUG - cufftMakePlan2d Executed"<<endl;
    return new Result(exit_code,out);
}

CUFFT_ROUTINE_HANDLER(MakePlan3d) {
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("MakePlan3d"));
    
    cufftHandle plan = in->Get<cufftHandle>();
    int nx = in->Get<int>();
    int ny = in->Get<int>();
    int nz = in->Get<int>();
    cufftType type = in->Get<cufftType>();
    size_t * workSize = (in->Assign<size_t>());
    cufftResult exit_code = cufftMakePlan3d(plan,nx,ny,nz,type,workSize);
    
    Buffer *out = new Buffer();
    try{
        out->Add(workSize);
    } catch (string e){
        cout << 'DEBUG - ' <<  e << endl;
        LOG4CPLUS_DEBUG(logger,e);
        return new Result(cudaErrorMemoryAllocation);
    }
    cout<<"DEBUG - cufftMakePlan3d Executed"<<endl;
    return new Result(exit_code,out);
}


CUFFT_ROUTINE_HANDLER(MakePlanMany) {
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("MakePlanMany"));
    
    cufftHandle plan = in->Get<cufftHandle>();
    int rank = in->Get<int>();
    int * n = in->Assign<int>();
    int * inembed = in->Assign<int>();
    int istride = in->Get<int>();
    int idist = in->Get<int>();
    
    int * onembed = in->Assign<int>();
    int ostride = in->Get<int>();
    int odist = in->Get<int>();
    
    cufftType type = in->Get<cufftType>();
    int batch = in->Get<int>();
    size_t * workSize = (in->Assign<size_t>());
    
    cufftResult exit_code = cufftMakePlanMany(plan,rank,n,inembed,istride,idist,onembed,ostride,odist,type,batch,workSize);
    Buffer *out = new Buffer();
    try{
        out->Add(workSize);
    } catch (string e){
        cout << 'DEBUG - ' <<  e << endl;
        LOG4CPLUS_DEBUG(logger,e);
        return new Result(cudaErrorMemoryAllocation);
    }
    cout<<"DEBUG - cufftMakePlanMany Executed"<<endl;
    return new Result(exit_code,out);
}

#if CUDART_VERSION >= 7000
CUFFT_ROUTINE_HANDLER(MakePlanMany64) {
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("MakePlanMany64"));
    
    cufftHandle plan = in->Get<cufftHandle>();
    int rank = in->Get<int>();
    long long int *n = in->Assign<long long int>();
    long long int *inembed = in->Assign<long long int>();
    long long int istride = in->Get<long long int>();
    long long int idist = in->Get<long long int>();

    long long int *onembed = in->Assign<long long int>();
    long long int ostride = in->Get<long long int>();
    long long int odist = in->Get<long long int>();
    
    cufftType type = in->Get<cufftType>();
    long long int batch = in->Get<long long int>();
    size_t * workSize = in->Assign<size_t>();
    
    cufftResult exit_code = cufftMakePlanMany64(plan,rank,n,inembed,istride,idist,onembed,ostride,odist,type,batch,workSize);
    Buffer *out = new Buffer();
    try{
        out->Add(workSize);   
    } catch (string e) {
        LOG4CPLUS_DEBUG(logger,e);
        return new Result(cudaErrorMemoryAllocation);
    }
    cout<<"DEBUG - cufftMakePlanMany64 Executed\n";
    return new Result(exit_code,out);
}
#endif

CUFFT_ROUTINE_HANDLER(GetSize1d) {
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("GetSize1d"));
        
    cufftHandle handle = in->Get<cufftHandle>();
    int nx = in->Get<int>();
    cufftType type = in->Get<cufftType>();
    int batch = in->Get<int>();
    size_t * workSize = (in->Assign<size_t>());
    cufftResult exit_code = cufftGetSize1d(handle,nx,type,batch,workSize);
    
    Buffer *out = new Buffer();
    try{
        out->Add(workSize);
    } catch (string e){
        cout << 'DEBUG - ' <<  e << endl;
        LOG4CPLUS_DEBUG(logger,e);
        return new Result(cudaErrorMemoryAllocation);
    }
    cout<<"DEBUG - cufftGetSize1d Executed"<<endl;
    return new Result(exit_code,out);
}


CUFFT_ROUTINE_HANDLER(GetSize2d) {
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("GetSize2d"));
        
    cufftHandle handle = in->Get<cufftHandle>();
    int nx = in->Get<int>();
    int ny = in->Get<int>();
    cufftType type = in->Get<cufftType>();
    size_t * workSize = (in->Assign<size_t>());
    cufftResult exit_code = cufftGetSize2d(handle,nx,ny,type,workSize);
    
    Buffer *out = new Buffer();
    try{
        out->Add(workSize);
    } catch (string e){
        cout << 'DEBUG - ' <<  e << endl;
        LOG4CPLUS_DEBUG(logger,e);
        return new Result(cudaErrorMemoryAllocation);
    }
    cout<<"DEBUG - cufftGetSize2d Executed"<<endl;
    return new Result(exit_code,out);
}

CUFFT_ROUTINE_HANDLER(GetSize3d) {
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("GetSize3d"));
        
    cufftHandle handle = in->Get<cufftHandle>();
    int nx = in->Get<int>();
    int ny = in->Get<int>();
    int nz = in->Get<int>();
    cufftType type = in->Get<cufftType>();
    size_t * workSize = (in->Assign<size_t>());
    cufftResult exit_code = cufftGetSize3d(handle,nx,ny,nz,type,workSize);
    
    Buffer *out = new Buffer();
    try{
        out->Add(workSize);
    } catch (string e){
        cout << 'DEBUG - ' <<  e << endl;
        LOG4CPLUS_DEBUG(logger,e);
        return new Result(cudaErrorMemoryAllocation);
    }
    cout<<"DEBUG - cufftGetSize3d Executed"<<endl;
    return new Result(exit_code,out);
}

CUFFT_ROUTINE_HANDLER(GetSizeMany) {
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("GetSizeMany"));
    
    cufftHandle handle = in->Get<cufftHandle>();
    int rank = in->Get<int>();
    int * n = in->Assign<int>();
    int * inembed = in->Assign<int>();
    int istride = in->Get<int>();
    int idist = in->Get<int>();
    
    int * onembed = in->Assign<int>();
    int ostride = in->Get<int>();
    int odist = in->Get<int>();
    
    cufftType type = in->Get<cufftType>();
    int batch = in->Get<int>();
    size_t * workSize = (in->Assign<size_t>());
    
    cufftResult exit_code = cufftGetSizeMany(handle,rank,n,inembed,istride,idist,onembed,ostride,odist,type,batch,workSize);
    Buffer *out = new Buffer();
    try{
        out->Add(workSize);
    } catch (string e){
        cout << 'DEBUG - ' <<  e << endl;
        LOG4CPLUS_DEBUG(logger,e);
        return new Result(cudaErrorMemoryAllocation);
    }
    cout<<"DEBUG - cufftGetSizeMany Executed"<<endl;
    return new Result(exit_code,out);
}

#if CUDART_VERSION >= 7000
CUFFT_ROUTINE_HANDLER(GetSizeMany64) {
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("GetSizeMany64"));
    
    cufftHandle plan = in->Get<cufftHandle>();
    int rank = in->Get<int>();
    long long int *n = in->Assign<long long int>();
    long long int *inembed = in->Assign<long long int>();
    long long int istride = in->Get<long long int>();
    long long int idist = in->Get<long long int>();

    long long int *onembed = in->Assign<long long int>();
    long long int ostride = in->Get<long long int>();
    long long int odist = in->Get<long long int>();
    
    cufftType type = in->Get<cufftType>();
    long long int batch = in->Get<long long int>();
    size_t * workSize = in->Assign<size_t>();
    
    cufftResult exit_code = cufftGetSizeMany64(plan,rank,n,inembed,istride,idist,onembed,ostride,odist,type,batch,workSize);
    Buffer *out = new Buffer();
    try{
        out->Add(workSize);   
    } catch (string e) {
        LOG4CPLUS_DEBUG(logger,e);
        return new Result(cudaErrorMemoryAllocation);
    }
    cout<<"DEBUG - cufftGetSizeMany64 Executed\n";
    return new Result(exit_code,out);
}
#endif

CUFFT_ROUTINE_HANDLER(GetSize) {
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("GetSize"));
    
    cufftHandle handle = in->Get<cufftHandle>();
    size_t * workSize = in->Assign<size_t>();
    cufftResult exit_code = cufftGetSize(handle,workSize);
    Buffer *out = new Buffer();
    try{
        out->Add(workSize);   
    } catch (string e) {
        LOG4CPLUS_DEBUG(logger,e);
        return new Result(cudaErrorMemoryAllocation);
    }
    cout<<"DEBUG - cufftGetSize Executed\n";
    return new Result(exit_code,out);
}

CUFFT_ROUTINE_HANDLER(SetStream) {
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("SetStream"));
    
    cufftHandle plan = in->Get<cufftHandle>();
    cudaStream_t stream = in->GetFromMarshal<cudaStream_t>();
    
    cufftResult exit_code = cufftSetStream(plan,stream);
    
    cout<<"DEBUG - cufftSetStream Executed\n";
    return new Result(exit_code);
}

#if __CUDA_API_VERSION >= 7000
CUFFT_ROUTINE_HANDLER(GetProperty){
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("GetProperty"));
    
    libraryPropertyType type = in->Get<libraryPropertyType>();
    int * value = in->Assign<int>();
    cufftResult exit_code = cufftGetProperty(type,value);
    cout<<"DEBUG - cufftGetProperty Executed\n";
    return new Result(exit_code);
}
#endif

/*
 * cufftResult cufftXtMalloc(cufftHandle plan, cudaLibXtDesc **descriptor, 
        cufftXtSubFormat format);
 * @param plan 
 * @param descriptor
 * @param format
 */
CUFFT_ROUTINE_HANDLER(XtMalloc){
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("XtMalloc"));
    
    cufftHandle plan = in->Get<cufftHandle>();
    cudaLibXtDesc ** desc = in->Assign<cudaLibXtDesc*>();
    cufftXtSubFormat format = in->Get<cufftXtSubFormat>();
    
    cufftResult exit_code = cufftXtMalloc(plan,desc,format);
    Buffer * out = new Buffer();
    try{
        out->Add(desc);
    } catch (string e){
        cout << 'DEBUG - ' <<  e << endl;
        LOG4CPLUS_DEBUG(logger,e);
        return new Result(cudaErrorMemoryAllocation);
    }
    cout<<"DEBUG - cufftXtMalloc Executed"<<endl;
    return new Result(exit_code,out);
}

/*
 * cufftResult cufftXtMemcpy(cufftHandle plan, void *dstPointer, void *srcPointer, cufftXtCopyType type);
 * @param plan
 * @param *dstPointer
 * @param *srcPointer
 * @param type
 */
CUFFT_ROUTINE_HANDLER(XtMemcpy){
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("XtMemcpy"));
    
    cufftHandle plan = in->Get<cufftHandle>();
    cout<<"plan_input: "<< plan<<endl;
    void * dstPointer = NULL;
    
    void * srcPointer = NULL;
    cufftXtCopyType type = in->BackGet<cufftXtCopyType>();
    cufftResult exit_code;
    Buffer *out;
    try{
        switch(type){
            case CUFFT_COPY_HOST_TO_DEVICE:
                cout<<"type: HOST_TO_DEVICE"<< endl;  
                dstPointer = in->GetFromMarshal<void*>();
                cout<<"dstPointer:"<< dstPointer <<endl;
                srcPointer = (in->Assign<void>());
                cout<<"srcPointer:"<< srcPointer <<endl;
                
                exit_code = cufftXtMemcpy(plan,dstPointer,srcPointer,CUFFT_COPY_HOST_TO_DEVICE);
                
                cout<<"type: HOST_TO_DEVICE"<< endl;  
                if (exit_code == CUFFT_INVALID_PLAN)
                    cout << "A"<<endl;
                if (exit_code == CUFFT_INVALID_VALUE)
                    cout << "B"<<endl;
                if (exit_code == CUFFT_INTERNAL_ERROR)
                    cout << "C"<<endl;
                if (exit_code == CUFFT_SETUP_FAILED)
                    cout << "D"<<endl;
                if (exit_code == CUFFT_INVALID_DEVICE)
                    cout << "E"<<endl;
                if (exit_code == CUFFT_SUCCESS)
                    cout << "F"<<endl;
                cout <<"exit_code: "<<exit_code<<endl;
                //out->AddMarshal(dstPointer);
                break;
            /*case CUFFT_COPY_DEVICE_TO_DEVICE:
                cout<<"type: DEVICE_TO_DEVICE"<< endl;  
                
                dstPointer = in->GetFromMarshal<void*>();
                srcPointer = in->GetFromMarshal<void*>();
                exit_code = cufftXtMemcpy(plan,dstPointer,srcPointer,type);
                out = new Buffer();
                out->AddMarshal(dstPointer);
                
                break;
            case CUFFT_COPY_DEVICE_TO_HOST:
                cout<<"type: DEVICE_TO_HOST"<< endl;  
                dstPointer = in->Assign<char>();
                srcPointer = in->GetFromMarshal<void*>();
                exit_code = cufftXtMemcpy(plan,dstPointer,srcPointer,type);
                out = new Buffer();
                out->Add(dstPointer);
                break;*/
            default:
                break;
        }
    } catch (string e){
        cout << 'EXCEPTION - ' <<  e << endl;
        LOG4CPLUS_DEBUG(logger,e);
        return new Result(cudaErrorMemoryAllocation);
    }
    cout<<"DEBUG - cufftXtMemcpy Executed"<<endl;
    return new Result(exit_code,out);
}

/*Da testare*/
CUFFT_ROUTINE_HANDLER(XtExecDescriptorC2C){
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("XtExecDescriptorC2C"));
    
    cufftHandle plan = in->Get<cufftHandle>();
    cudaLibXtDesc *input = in->GetFromMarshal<cudaLibXtDesc*>();
    cudaLibXtDesc *output = in->GetFromMarshal<cudaLibXtDesc*>();
    int direction = in->Get<int>();
    Buffer *out = new Buffer();
    cufftResult exit_code;
    try{
        exit_code = cufftXtExecDescriptorC2C(plan,input,output,direction);
        out->AddMarshal(output);
    } catch (string e){
        cout << 'EXCEPTION - ' <<  e << endl;
        LOG4CPLUS_DEBUG(logger,e);
        return new Result(cudaErrorMemoryAllocation);
    }
    cout<<"DEBUG - cufftXtExecDescriptorC2C Executed"<<endl;
    return new Result(exit_code,out);
}

/*Da testare*/
CUFFT_ROUTINE_HANDLER(XtFree){
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("XtFree"));
    cudaLibXtDesc *descriptor = in->Assign<cudaLibXtDesc>();
    cufftResult exit_code = cufftXtFree(descriptor);
    cout<<"DEBUG - cufftXtMemcpy Executed"<<endl;
    return new Result(exit_code);
}


/* -- FUNCTION NOT SUPPORTED IN GVIRTUS -- */
CUFFT_ROUTINE_HANDLER(XtSetCallback){
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("XtSetCallback"));
    
    /*cufftHandle plan = in->Get<cufftHandle>();
    cout << "plan:"<< plan<<endl;
    void ** callbackRoutine = in->Assign<void*>();
    cout << "callbackroutine:"<< callbackRoutine<<endl;
    cufftXtCallbackType type = in->Get<cufftXtCallbackType>();
    void ** callerInfo = in->GetFromMarshal<void**>();
    cout<< "callerinfo: "<< callerInfo<<endl;
    cufftResult exit_code = cufftXtSetCallback(plan,callbackRoutine,type,callerInfo);*/
    return new Result(CUFFT_NOT_IMPLEMENTED);
}

CUFFT_ROUTINE_HANDLER(GetVersion){
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("GetVersion"));
    int *version = in->Assign<int>();
    cufftResult exit_code = cufftGetVersion(version);
    cout<<"DEBUG - cufftGetVersion Executed"<<endl;
}


void CufftHandler::Initialize() {
    if (mspHandlers != NULL)
        return;
    mspHandlers = new map<string, CufftHandler::CufftRoutineHandler > ();
    /* - Plan - */
    mspHandlers->insert(CUFFT_ROUTINE_HANDLER_PAIR(Plan1d));
    mspHandlers->insert(CUFFT_ROUTINE_HANDLER_PAIR(Plan2d));
    mspHandlers->insert(CUFFT_ROUTINE_HANDLER_PAIR(Plan3d));
    mspHandlers->insert(CUFFT_ROUTINE_HANDLER_PAIR(PlanMany));
    /* - MakePlan - */
    mspHandlers->insert(CUFFT_ROUTINE_HANDLER_PAIR(MakePlan1d));
    mspHandlers->insert(CUFFT_ROUTINE_HANDLER_PAIR(MakePlan2d));
    mspHandlers->insert(CUFFT_ROUTINE_HANDLER_PAIR(MakePlan3d));
    mspHandlers->insert(CUFFT_ROUTINE_HANDLER_PAIR(MakePlanMany));
#if CUDART_VERSION >= 7000
    mspHandlers->insert(CUFFT_ROUTINE_HANDLER_PAIR(MakePlanMany64));
#endif
    /* - GetSize - */
    mspHandlers->insert(CUFFT_ROUTINE_HANDLER_PAIR(GetSize1d));
    mspHandlers->insert(CUFFT_ROUTINE_HANDLER_PAIR(GetSize2d));
    mspHandlers->insert(CUFFT_ROUTINE_HANDLER_PAIR(GetSize3d));
    mspHandlers->insert(CUFFT_ROUTINE_HANDLER_PAIR(GetSizeMany));
#if CUDART_VERSION >= 7000
    mspHandlers->insert(CUFFT_ROUTINE_HANDLER_PAIR(GetSizeMany64));
#endif
    mspHandlers->insert(CUFFT_ROUTINE_HANDLER_PAIR(GetSize));
    /* - Estimate - */
    mspHandlers->insert(CUFFT_ROUTINE_HANDLER_PAIR(SetWorkArea));
    mspHandlers->insert(CUFFT_ROUTINE_HANDLER_PAIR(SetCompatibilityMode));
    mspHandlers->insert(CUFFT_ROUTINE_HANDLER_PAIR(SetAutoAllocation));
    mspHandlers->insert(CUFFT_ROUTINE_HANDLER_PAIR(GetVersion));
    mspHandlers->insert(CUFFT_ROUTINE_HANDLER_PAIR(SetStream));
#if __CUDA_API_VERSION >= 7000
    mspHandlers->insert(CUFFT_ROUTINE_HANDLER_PAIR(GetProperty));
#endif
    /* - Estimate - */
    mspHandlers->insert(CUFFT_ROUTINE_HANDLER_PAIR(Estimate1d));
    mspHandlers->insert(CUFFT_ROUTINE_HANDLER_PAIR(Estimate2d));
    mspHandlers->insert(CUFFT_ROUTINE_HANDLER_PAIR(Estimate3d));
    mspHandlers->insert(CUFFT_ROUTINE_HANDLER_PAIR(EstimateMany));
    
    /* - Create/Destroy - */
    mspHandlers->insert(CUFFT_ROUTINE_HANDLER_PAIR(Create));
    mspHandlers->insert(CUFFT_ROUTINE_HANDLER_PAIR(Destroy));
    /* - Exec - */
    mspHandlers->insert(CUFFT_ROUTINE_HANDLER_PAIR(ExecC2C));
    mspHandlers->insert(CUFFT_ROUTINE_HANDLER_PAIR(ExecR2C));
    mspHandlers->insert(CUFFT_ROUTINE_HANDLER_PAIR(ExecC2R));
    mspHandlers->insert(CUFFT_ROUTINE_HANDLER_PAIR(ExecZ2Z));
    mspHandlers->insert(CUFFT_ROUTINE_HANDLER_PAIR(ExecD2Z));
    mspHandlers->insert(CUFFT_ROUTINE_HANDLER_PAIR(ExecZ2D));
    /* -- CufftX -- */
    #if __CUDA_API_VERSION >= 7000
    mspHandlers->insert(CUFFT_ROUTINE_HANDLER_PAIR(XtMakePlanMany));
    #endif
    mspHandlers->insert(CUFFT_ROUTINE_HANDLER_PAIR(XtSetGPUs));
    mspHandlers->insert(CUFFT_ROUTINE_HANDLER_PAIR(XtExecDescriptorC2C));
    mspHandlers->insert(CUFFT_ROUTINE_HANDLER_PAIR(XtSetCallback));
    /* - Memory Management - */
    mspHandlers->insert(CUFFT_ROUTINE_HANDLER_PAIR(XtMalloc));
    mspHandlers->insert(CUFFT_ROUTINE_HANDLER_PAIR(XtMemcpy));
    mspHandlers->insert(CUFFT_ROUTINE_HANDLER_PAIR(XtFree));
}


