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
 * @date   Sat Oct 10 10:51:58 2009
 *
 * @brief
 *
 *
 */

#include "CufftHandler.h"

using namespace std;

map<string, CufftHandler::CufftRoutineHandler> *CufftHandler::mspHandlers = NULL;

extern "C" int HandlerInit() {
    return 0;
}

extern "C" Handler *GetHandler() {
    return new CufftHandler();
}

CufftHandler::CufftHandler() {
    Initialize();
}

CufftHandler::~CufftHandler() {

}

bool CufftHandler::CanExecute(std::string routine) {
    return mspHandlers->find(routine) != mspHandlers->end();
}

Result * CufftHandler::Execute(std::string routine, Buffer * input_buffer) {
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

CUFFT_ROUTINE_HANDLER(Plan2d) {
    cufftHandle plan;
    int nx = in->Get<int>();
    int ny = in->Get<int>();
    cufftType type = in->Get<cufftType > ();
    cufftResult ec = cufftPlan2d(&plan, nx, ny, type);
    Buffer *out = new Buffer();
    out->Add(plan);
    return new Result(ec, out);
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

void CufftHandler::Initialize() {
    if (mspHandlers != NULL)
        return;
    mspHandlers = new map<string, CufftHandler::CufftRoutineHandler > ();
    mspHandlers->insert(CUFFT_ROUTINE_HANDLER_PAIR(Plan2d));
    mspHandlers->insert(CUFFT_ROUTINE_HANDLER_PAIR(ExecC2R));
    mspHandlers->insert(CUFFT_ROUTINE_HANDLER_PAIR(SetCompatibilityMode));
}

