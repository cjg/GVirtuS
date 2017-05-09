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
 * Written by: Raffaele Montella <raffaele.montella@uniparthenope.it>,
 *             Department of Science and Technologies
 *             Vincenzo Santopietro <vincenzo.santopietro@uniparthenope.it>,
 *             Department of Science and Technologies
 */
#ifndef _CUBLASHANDLER
#define _CUBLASHANDLER

#ifndef CUBLASAPI
    #ifdef _WIN32
        #define CUBLASAPI __stdcall
    #else
        #define CUBLASAPI
    #endif
#endif

#include <iostream>
#include <map>
#include <string>

#include <cublas.h>
#include "cublas_v2.h"

#include "Handler.h"
#include "Result.h"

#include "log4cplus/logger.h"
#include "log4cplus/loggingmacros.h"
#include "log4cplus/configurator.h"

class CublasHandler : public Handler {
public:
    CublasHandler();
    virtual ~CublasHandler();
    bool CanExecute(std::string routine);
    Result * Execute(std::string routine, Buffer * input_buffer);

    /*void * RegisterPointer(void *,size_t);

    void RegisterMapObject(char *,char *);
    char * GetMapObject(char *);
*/
private:
    log4cplus::Logger logger;
    void Initialize();
    typedef Result * (*CublasRoutineHandler)(CublasHandler *, Buffer *);
    static std::map<std::string, CublasRoutineHandler> * mspHandlers;
    //void **pointers;
    //int nPointers;
    
    //std::map<std::string, std::string> * mpMapObject;

    //void *mpShm;
    //int mShmFd;
};

#define CUBLAS_ROUTINE_HANDLER(name) Result * handle##name(CublasHandler * pThis, Buffer * in)
#define CUBLAS_ROUTINE_HANDLER_PAIR(name) make_pair("cublas" #name, handle##name)

/* CublasHandler_Helper */
CUBLAS_ROUTINE_HANDLER(GetVersion_v2);
CUBLAS_ROUTINE_HANDLER(Create_v2);
CUBLAS_ROUTINE_HANDLER(Destroy_v2);
CUBLAS_ROUTINE_HANDLER(SetVector);
CUBLAS_ROUTINE_HANDLER(GetVector);
CUBLAS_ROUTINE_HANDLER(SetMatrix);
CUBLAS_ROUTINE_HANDLER(GetMatrix);
CUBLAS_ROUTINE_HANDLER(SetStream_v2);
CUBLAS_ROUTINE_HANDLER(GetPointerMode_v2);
CUBLAS_ROUTINE_HANDLER(SetPointerMode_v2);

/* CublasHandler_Level1 */
CUBLAS_ROUTINE_HANDLER(Sdot_v2);
CUBLAS_ROUTINE_HANDLER(Ddot_v2);
CUBLAS_ROUTINE_HANDLER(Cdotu_v2);
CUBLAS_ROUTINE_HANDLER(Cdotc_v2);
CUBLAS_ROUTINE_HANDLER(Zdotu_v2);
CUBLAS_ROUTINE_HANDLER(Zdotc_v2);

CUBLAS_ROUTINE_HANDLER(Sscal_v2);
CUBLAS_ROUTINE_HANDLER(Dscal_v2);
CUBLAS_ROUTINE_HANDLER(Cscal_v2);
CUBLAS_ROUTINE_HANDLER(Csscal_v2);
CUBLAS_ROUTINE_HANDLER(Zscal_v2);
CUBLAS_ROUTINE_HANDLER(Zdscal_v2);

CUBLAS_ROUTINE_HANDLER(Saxpy_v2);
CUBLAS_ROUTINE_HANDLER(Caxpy_v2);
CUBLAS_ROUTINE_HANDLER(Zaxpy_v2);
CUBLAS_ROUTINE_HANDLER(Daxpy_v2);

CUBLAS_ROUTINE_HANDLER(Scopy_v2);
CUBLAS_ROUTINE_HANDLER(Dcopy_v2);
CUBLAS_ROUTINE_HANDLER(Ccopy_v2);
CUBLAS_ROUTINE_HANDLER(Zcopy_v2);

CUBLAS_ROUTINE_HANDLER(Sswap_v2);
CUBLAS_ROUTINE_HANDLER(Dswap_v2);
CUBLAS_ROUTINE_HANDLER(Cswap_v2);
CUBLAS_ROUTINE_HANDLER(Zswap_v2);

CUBLAS_ROUTINE_HANDLER(Isamax_v2);
CUBLAS_ROUTINE_HANDLER(Idamax_v2);
CUBLAS_ROUTINE_HANDLER(Icamax_v2);
CUBLAS_ROUTINE_HANDLER(Izamax_v2);

CUBLAS_ROUTINE_HANDLER(Sasum_v2);
CUBLAS_ROUTINE_HANDLER(Dasum_v2);
CUBLAS_ROUTINE_HANDLER(Scasum_v2);
CUBLAS_ROUTINE_HANDLER(Dzasum_v2);

CUBLAS_ROUTINE_HANDLER(Srot_v2);
CUBLAS_ROUTINE_HANDLER(Drot_v2);
CUBLAS_ROUTINE_HANDLER(Crot_v2);
CUBLAS_ROUTINE_HANDLER(Csrot_v2);
CUBLAS_ROUTINE_HANDLER(Zrot_v2);
CUBLAS_ROUTINE_HANDLER(Zdrot_v2);

CUBLAS_ROUTINE_HANDLER(Srotg_v2);
CUBLAS_ROUTINE_HANDLER(Drotg_v2);
CUBLAS_ROUTINE_HANDLER(Crotg_v2);
CUBLAS_ROUTINE_HANDLER(Zrotg_v2);

CUBLAS_ROUTINE_HANDLER(Srotm_v2);
CUBLAS_ROUTINE_HANDLER(Drotm_v2);

CUBLAS_ROUTINE_HANDLER(Srotmg_v2);
CUBLAS_ROUTINE_HANDLER(Drotmg_v2);
/* CublasHandler_Level2 */
CUBLAS_ROUTINE_HANDLER(Sgemv);
/* CublasHandler_Level3 */
CUBLAS_ROUTINE_HANDLER(Sgemm_v2);
CUBLAS_ROUTINE_HANDLER(Snrm2_v2);
/*CUBLAS_ROUTINE_HANDLER(Dnrm2_v2);
CUBLAS_ROUTINE_HANDLER(Scnrm2_v2);
CUBLAS_ROUTINE_HANDLER(Dznrm2_v2);*/

/*CUBLAS_ROUTINE_HANDLER(Sscal);
CUBLAS_ROUTINE_HANDLER(Destroy);*/
#endif