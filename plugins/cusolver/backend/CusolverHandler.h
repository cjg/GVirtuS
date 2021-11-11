
/*
 *   gVirtuS -- A GPGPU transparent virtualization component.
 *   
 *  Copyright (C) 2009-2010  The University of Napoli Parthenope at Naples.
 *     
 *  This file is part of gVirtuS.
 *       
 *  gVirtuS is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *            
 *  gVirtuS is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU Lesser General Public License for more details.
 *                 
 *  You should have received a copy of the GNU General Public License
 *  along with gVirtuS; if not, write to the Free Software
 *  Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
 *                    
 *  Written by: Antonio Pilato <antonio.pilato001@studenti.uniparthenope.it>,
 *  Department of Science and Technologies
 */

#ifndef CUSOLVERHANDLER_H
#define CUSOLVERHANDLER_H

#include <gvirtus/backend/Handler.h>
#include <gvirtus/communicators/Result.h>

#include <cusolverDn.h>

using gvirtus::common::pointer_t;
using gvirtus::communicators::Buffer;
using gvirtus::communicators::Result;

#include "log4cplus/logger.h"
#include "log4cplus/loggingmacros.h"
#include "log4cplus/configurator.h"

#include <limits.h>
#if ( __WORDSIZE == 64 )
    #define BUILD_64   1
#endif

using namespace std;
using namespace log4cplus;

class CusolverHandler : public gvirtus::backend::Handler {
public:
    CusolverHandler();
    virtual ~CusolverHandler();
    bool CanExecute(std::string routine);
    std::shared_ptr<gvirtus::communicators::Result> Execute(std::string routine, std::shared_ptr<gvirtus::communicators::Buffer> input_buffer);
    static void setLogLevel(Logger *logger);
private:
    log4cplus::Logger logger;
    void Initialize();
    typedef std::shared_ptr<gvirtus::communicators::Result> (*CusolverRoutineHandler)(CusolverHandler *, std::shared_ptr<gvirtus::communicators::Result>);
    static std::map<std::string, CusolverRoutineHandler> * mspHandlers;
};

#define CUSOLVER_ROUTINE_HANDLER(name) std::shared_ptr<Result> handle##name(CusolverHandler * pThis, std::shared_ptr<Buffer> in)
#define CUSOLVER_ROUTINE_HANDLER_PAIR(name) make_pair("cusolver" #name, handle##name)

CUSOLVER_ROUTINE_HANDLER(GetVersion);
CUSOLVER_ROUTINE_HANDLER(GetErrorString);
CUSOLVER_ROUTINE_HANDLER(Create);
CUSOLVER_ROUTINE_HANDLER(Destroy);
CUSOLVER_ROUTINE_HANDLER(SetStream);
CUSOLVER_ROUTINE_HANDLER(GetStream);

#endif  /* CUSOLVERHANDLER_H */
