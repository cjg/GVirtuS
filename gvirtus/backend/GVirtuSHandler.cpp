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
 */
#include "GVirtuSHandler.h"
#include "GVirtuSHandler_host.cpp"

#include <cstring>
#include <map>
#include <errno.h>


using namespace std;
using namespace log4cplus;

std::map<string, GVirtuSHandler::GVirtuSRoutineHandler> * GVirtuSHandler::mspHandlers = NULL;

extern "C" int HandlerInit() {
    return 0;
}
extern "C" Handler *GetHandler() {
    return new GVirtuSHandler();
}

GVirtuSHandler::GVirtuSHandler() {
    logger=Logger::getInstance(LOG4CPLUS_TEXT("GVirtuSHandler"));
    Initialize();
}

GVirtuSHandler::~GVirtuSHandler() {
}

bool GVirtuSHandler::CanExecute(std::string routine) {
    return mspHandlers->find(routine) != mspHandlers->end();
}

Result * GVirtuSHandler::Execute(std::string routine, Buffer * in) {
    LOG4CPLUS_DEBUG(logger,"Called " << routine);
    map<string, GVirtuSHandler::GVirtuSRoutineHandler>::iterator it;
    it = mspHandlers->find(routine);
    if (it == mspHandlers->end())
        throw "No handler for '" + routine + "' found!";
    try {
        return it->second(this, in);
    } catch (const char *ex) {
        cout << ex << endl;
        cout << strerror(errno) << endl;
    }
    return NULL;
}


void GVirtuSHandler::Initialize() {
    if (mspHandlers != NULL)
        return;
    mspHandlers = new map<string, GVirtuSHandler::GVirtuSRoutineHandler> ();

    /* GVirtuSHandler Query Platform Info */
    mspHandlers->insert(GVIRTUS_ROUTINE_HANDLER_PAIR(getVersion));
}
