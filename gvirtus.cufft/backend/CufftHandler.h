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
    void Initialize();
    typedef Result * (*CufftRoutineHandler)(CufftHandler *, Buffer *);
    static std::map<std::string, CufftRoutineHandler> * mspHandlers;
};

#define CUFFT_ROUTINE_HANDLER(name) Result * handle##name(CufftHandler * pThis, Buffer * in)
#define CUFFT_ROUTINE_HANDLER_PAIR(name) make_pair("cufft" #name, handle##name)

#endif	/* _CUDARTHANDLER_H */

