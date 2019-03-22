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

#include <vector>
#include <cstring>
#include <cstdio>

/**
 * @file   Backend.cpp
 * @author Giuseppe Coviello <giuseppe.coviello@uniparthenope.it>
 * @date   Wed Sep 30 12:29:26 2009
 *
 * @brief
 *
 *
 */

#include "Backend.h"

#include <iostream>

#include "Process.h"
#include <dlfcn.h>
#include <string>
#include <cstring>

using namespace std;
using namespace log4cplus;

Backend::Backend(vector<string> &plugins) {
    logger=Logger::getInstance(LOG4CPLUS_TEXT("Backend"));
    mPlugins = plugins;
}

void Backend::Start(Communicator * communicator) {
    communicator->Serve();
    //aux communicator
    while (true) {
        Communicator *client =
                const_cast<Communicator *> (communicator->Accept());
        LOG4CPLUS_DEBUG(logger, "Connection accepted" );   
        Process *process = new Process(client, mPlugins);
        process->Start(NULL);
    }
}

void Backend::EventOccurred(std::string& event, void* object) {
    LOG4CPLUS_DEBUG(logger, "EventOccurred: " << event); 
}
