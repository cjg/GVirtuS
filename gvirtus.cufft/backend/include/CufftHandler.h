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

/**
 * @file   CudaRtHandler.h
 * @author Giuseppe Coviello <giuseppe.coviello@uniparthenope.it>
 * @author Vincenzo Santopietro <vincenzo.santopietro@uniparthenope.it>
 * @date   Sat Oct 10 10:51:58 2009
 * 
 * @brief  
 * 
 * 
 */

#ifndef _CUFFTHANDLER_H
#define	_CUFFTHANDLER_H

#include "Handler.h"
#include "Result.h"

#include <cufft.h>
#include <cufftXt.h>

#include "log4cplus/logger.h"
#include "log4cplus/loggingmacros.h"
#include "log4cplus/configurator.h"

#include <limits.h>
#if ( __WORDSIZE == 64 )
    #define BUILD_64   1
#endif

/**						
 * CudaRtHandler is used by Backend's Process(es) for storing and retrieving
 * device related data and functions. 
 * CudaRtHandler has also the method Execute() that is responsible to execute a
 * named CUDA Runtime routine unmarshalling the input parameters from the
 * provided Buffer.
 */
class CufftHandler : public Handler {
public:
    CufftHandler();
    virtual ~CufftHandler();
    bool CanExecute(std::string routine);
    Result * Execute(std::string routine, Buffer * input_buffer);
private:
    log4cplus::Logger logger;
    void Initialize();
    typedef Result * (*CufftRoutineHandler)(CufftHandler *, Buffer *);
    static std::map<std::string, CufftRoutineHandler> * mspHandlers;
};

#define CUFFT_ROUTINE_HANDLER(name) Result * handle##name(CufftHandler * pThis, Buffer * in)
#define CUFFT_ROUTINE_HANDLER_PAIR(name) make_pair("cufft" #name, handle##name)

/* CufftHandler.cpp */
CUFFT_ROUTINE_HANDLER(Plan1d);
CUFFT_ROUTINE_HANDLER(Plan2d);
CUFFT_ROUTINE_HANDLER(Plan3d);
CUFFT_ROUTINE_HANDLER(PlanMany);
CUFFT_ROUTINE_HANDLER(Estimate1d);
CUFFT_ROUTINE_HANDLER(Estimate2d);
CUFFT_ROUTINE_HANDLER(Estimate3d);
CUFFT_ROUTINE_HANDLER(EstimateMany);
CUFFT_ROUTINE_HANDLER(MakePlan1d);
CUFFT_ROUTINE_HANDLER(MakePlan2d);
CUFFT_ROUTINE_HANDLER(MakePlan3d);
CUFFT_ROUTINE_HANDLER(MakePlanMany);
#if CUDART_VERSION >= 7000
CUFFT_ROUTINE_HANDLER(MakePlanMany64);
#endif
CUFFT_ROUTINE_HANDLER(GetSize1d);
CUFFT_ROUTINE_HANDLER(GetSize2d);
CUFFT_ROUTINE_HANDLER(GetSize3d);
CUFFT_ROUTINE_HANDLER(GetSizeMany);
#if CUDART_VERSION >= 7000
CUFFT_ROUTINE_HANDLER(GetSizeMany64);
#endif
CUFFT_ROUTINE_HANDLER(GetSize);
CUFFT_ROUTINE_HANDLER(SetWorkArea);
CUFFT_ROUTINE_HANDLER(ExecC2C);
CUFFT_ROUTINE_HANDLER(ExecR2C);
CUFFT_ROUTINE_HANDLER(ExecC2R);
CUFFT_ROUTINE_HANDLER(ExecZ2Z);
CUFFT_ROUTINE_HANDLER(SetCompatibilityMode);
CUFFT_ROUTINE_HANDLER(SetWorkArea);
CUFFT_ROUTINE_HANDLER(SetAutoAllocation);
CUFFT_ROUTINE_HANDLER(Create);
#if __CUDA_API_VERSION >= 7000 
CUFFT_ROUTINE_HANDLER(XtMakePlanMany);
#endif
CUFFT_ROUTINE_HANDLER(XtExecDescriptorC2C);
CUFFT_ROUTINE_HANDLER(XtSetCallback);
/* Memory Management */
CUFFT_ROUTINE_HANDLER(XtMalloc);
CUFFT_ROUTINE_HANDLER(XtMemcpy);
CUFFT_ROUTINE_HANDLER(XtFree);
#endif	/* _CUFFTHANDLER_H */

