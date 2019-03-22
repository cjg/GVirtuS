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
 * @file   Communicator.cpp
 * @author Giuseppe Coviello <giuseppe.coviello@uniparthenope.it>
 * @date   Wed Sep 30 11:56:44 2009
 *
 * @brief
 *
 *
 */

#include "Communicator.h"

#include <cstring>
#include <cstdlib>

#ifndef _WIN32
#include "AfUnixCommunicator.h"
//#include "VmciCommunicator.h"
#include "ShmCommunicator.h"
#include "VMShmCommunicator.h"
#include "VMSocketCommunicator.h"
#include "VirtioCommunicator.h"
#endif
#include "TcpCommunicator.h"

using namespace std;

Communicator * Communicator::Get(const std::string & communicator) {
    const char *s = communicator.c_str();
    const char *tmp = strstr(s, "://");
    if (tmp == NULL)
        throw "Invalid communicator string.";
    char *type = new char[tmp - s + 1];
    memmove(type, s, tmp - s);
    type[tmp - s] = 0;
#ifndef _WIN32
    if (strcmp(type, "afunix") == 0)
        return new AfUnixCommunicator(communicator);
    if (strcmp(type, "shm") == 0)
        return new ShmCommunicator(communicator);
    if (strcmp(type, "vmshm") == 0)
        return new VMShmCommunicator(communicator);
    if (strcmp(type, "vmsocket") == 0)
        return new VMSocketCommunicator(communicator);
    if (strcmp(type, "virtio") == 0)
        return new VirtioCommunicator(communicator);
#endif
    if (strcmp(type, "tcp") == 0)
        return new TcpCommunicator(communicator);
    throw "Not a valid communicator type!";
    return NULL;
}

Communicator::~Communicator() {
}

