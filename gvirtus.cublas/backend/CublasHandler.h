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
CUBLAS_ROUTINE_HANDLER(Dgemv_v2);
CUBLAS_ROUTINE_HANDLER(Cgemv_v2);
CUBLAS_ROUTINE_HANDLER(Zgemv_v2);

CUBLAS_ROUTINE_HANDLER(Sgbmv_v2);
CUBLAS_ROUTINE_HANDLER(Dgbmv_v2);
CUBLAS_ROUTINE_HANDLER(Cgbmv_v2);
CUBLAS_ROUTINE_HANDLER(Zgbmv_v2);

CUBLAS_ROUTINE_HANDLER(Strmv_v2);
CUBLAS_ROUTINE_HANDLER(Dtrmv_v2);
CUBLAS_ROUTINE_HANDLER(Ctrmv_v2);
CUBLAS_ROUTINE_HANDLER(Ztrmv_v2);

CUBLAS_ROUTINE_HANDLER(Stbmv_v2);
CUBLAS_ROUTINE_HANDLER(Dtbmv_v2);
CUBLAS_ROUTINE_HANDLER(Ctbmv_v2);
CUBLAS_ROUTINE_HANDLER(Ztbmv_v2);

CUBLAS_ROUTINE_HANDLER(Stpmv_v2);
CUBLAS_ROUTINE_HANDLER(Dtpmv_v2);
CUBLAS_ROUTINE_HANDLER(Ctpmv_v2);
CUBLAS_ROUTINE_HANDLER(Ztpmv_v2);

CUBLAS_ROUTINE_HANDLER(Stpsv_v2);
CUBLAS_ROUTINE_HANDLER(Dtpsv_v2);
CUBLAS_ROUTINE_HANDLER(Ctpsv_v2);
CUBLAS_ROUTINE_HANDLER(Ztpsv_v2);

CUBLAS_ROUTINE_HANDLER(Stbsv_v2);
CUBLAS_ROUTINE_HANDLER(Dtbsv_v2);
CUBLAS_ROUTINE_HANDLER(Ctbsv_v2);
CUBLAS_ROUTINE_HANDLER(Ztbsv_v2);

CUBLAS_ROUTINE_HANDLER(Ssymv_v2);
CUBLAS_ROUTINE_HANDLER(Dsymv_v2);
CUBLAS_ROUTINE_HANDLER(Csymv_v2);
CUBLAS_ROUTINE_HANDLER(Zsymv_v2);
CUBLAS_ROUTINE_HANDLER(Chemv_v2);
CUBLAS_ROUTINE_HANDLER(Zhemv_v2);

CUBLAS_ROUTINE_HANDLER(Ssbmv_v2);
CUBLAS_ROUTINE_HANDLER(Dsbmv_v2);
CUBLAS_ROUTINE_HANDLER(Chbmv_v2);
CUBLAS_ROUTINE_HANDLER(Zhbmv_v2);

CUBLAS_ROUTINE_HANDLER(Sspmv_v2);
CUBLAS_ROUTINE_HANDLER(Dspmv_v2);
CUBLAS_ROUTINE_HANDLER(Chpmv_v2);
CUBLAS_ROUTINE_HANDLER(Zhpmv_v2);

CUBLAS_ROUTINE_HANDLER(Sger_v2);
CUBLAS_ROUTINE_HANDLER(Dger_v2);
CUBLAS_ROUTINE_HANDLER(Cgeru_v2);
CUBLAS_ROUTINE_HANDLER(Cgerc_v2);
CUBLAS_ROUTINE_HANDLER(Zgeru_v2);
CUBLAS_ROUTINE_HANDLER(Zgerc_v2);

CUBLAS_ROUTINE_HANDLER(Ssyr_v2);
CUBLAS_ROUTINE_HANDLER(Dsyr_v2);
CUBLAS_ROUTINE_HANDLER(Csyr_v2);
CUBLAS_ROUTINE_HANDLER(Zsyr_v2);
CUBLAS_ROUTINE_HANDLER(Cher_v2);
CUBLAS_ROUTINE_HANDLER(Zher_v2);

CUBLAS_ROUTINE_HANDLER(Sspr_v2);
CUBLAS_ROUTINE_HANDLER(Dspr_v2);
CUBLAS_ROUTINE_HANDLER(Chpr_v2);
CUBLAS_ROUTINE_HANDLER(Zhpr_v2);

CUBLAS_ROUTINE_HANDLER(Ssyr2_v2);
CUBLAS_ROUTINE_HANDLER(Dsyr2_v2);
CUBLAS_ROUTINE_HANDLER(Csyr2_v2);
CUBLAS_ROUTINE_HANDLER(Zsyr2_v2);
CUBLAS_ROUTINE_HANDLER(Cher2_v2);
CUBLAS_ROUTINE_HANDLER(Zher2_v2);

CUBLAS_ROUTINE_HANDLER(Sspr2_v2);
CUBLAS_ROUTINE_HANDLER(Dspr2_v2);
CUBLAS_ROUTINE_HANDLER(Chpr2_v2);
CUBLAS_ROUTINE_HANDLER(Zhpr2_v2);
/* CublasHandler_Level3 */
CUBLAS_ROUTINE_HANDLER(Sgemm_v2);
CUBLAS_ROUTINE_HANDLER(Dgemm_v2);
CUBLAS_ROUTINE_HANDLER(Cgemm_v2);
CUBLAS_ROUTINE_HANDLER(Zgemm_v2);

CUBLAS_ROUTINE_HANDLER(SgemmBatched_v2);
CUBLAS_ROUTINE_HANDLER(DgemmBatched_v2);
CUBLAS_ROUTINE_HANDLER(CgemmBatched_v2);
CUBLAS_ROUTINE_HANDLER(ZgemmBatched_v2);

CUBLAS_ROUTINE_HANDLER(Snrm2_v2);
CUBLAS_ROUTINE_HANDLER(Dnrm2_v2);
CUBLAS_ROUTINE_HANDLER(Scnrm2_v2);
CUBLAS_ROUTINE_HANDLER(Dznrm2_v2);

CUBLAS_ROUTINE_HANDLER(Ssyrk_v2);
CUBLAS_ROUTINE_HANDLER(Dsyrk_v2);
CUBLAS_ROUTINE_HANDLER(Csyrk_v2);
CUBLAS_ROUTINE_HANDLER(Zsyrk_v2);
CUBLAS_ROUTINE_HANDLER(Cherk_v2);
CUBLAS_ROUTINE_HANDLER(Zherk_v2);

CUBLAS_ROUTINE_HANDLER(Ssyr2k_v2);
CUBLAS_ROUTINE_HANDLER(Dsyr2k_v2);
CUBLAS_ROUTINE_HANDLER(Csyr2k_v2);
CUBLAS_ROUTINE_HANDLER(Zsyr2k_v2);
CUBLAS_ROUTINE_HANDLER(Cher2k_v2);
CUBLAS_ROUTINE_HANDLER(Zher2k_v2);

CUBLAS_ROUTINE_HANDLER(Ssymm_v2);
CUBLAS_ROUTINE_HANDLER(Dsymm_v2);
CUBLAS_ROUTINE_HANDLER(Csymm_v2);
CUBLAS_ROUTINE_HANDLER(Zsymm_v2);
CUBLAS_ROUTINE_HANDLER(Chemm_v2);
CUBLAS_ROUTINE_HANDLER(Zhemm_v2);

CUBLAS_ROUTINE_HANDLER(Strsm_v2);
CUBLAS_ROUTINE_HANDLER(Dtrsm_v2);
CUBLAS_ROUTINE_HANDLER(Ctrsm_v2);
CUBLAS_ROUTINE_HANDLER(Ztrsm_v2);

CUBLAS_ROUTINE_HANDLER(Strmm_v2);
CUBLAS_ROUTINE_HANDLER(Dtrmm_v2);
CUBLAS_ROUTINE_HANDLER(Ctrmm_v2);
CUBLAS_ROUTINE_HANDLER(Ztrmm_v2);
/*CUBLAS_ROUTINE_HANDLER(Sscal);
CUBLAS_ROUTINE_HANDLER(Destroy);*/
#endif