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

/**
 * @file   Process.h
 * @author Giuseppe Coviello <giuseppe.coviello@uniparthenope.it>
 * @date   Wed Sep 30 10:45:40 2009
 * 
 * @brief  
 * 
 * 
 */

#ifndef _PROCESS_H
#define	_PROCESS_H

#include <vector>

#include "Subprocess.h"
#include "Observable.h"
#include "Communicator.h"
#include "Handler.h"

/**
 * Process is the object used by the Backend to handle the request from a single
 * Frontend.
 */
class Process : public Subprocess, public Observable {
public:
    Process(const Communicator *communicator, std::vector<std::string> &plugins);
    virtual ~Process();
    void Setup();
    void Execute(void * arg);
private:
    Communicator * mpCommunicator;
    std::vector<std::string> mPlugins;
    std::vector<Handler *> mHandlers;
};

#endif	/* _PROCESS_H */

