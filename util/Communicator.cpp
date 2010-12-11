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

#include "AfUnixCommunicator.h"
#include "TcpCommunicator.h"
#include "VmciCommunicator.h"
#include "ShmCommunicator.h"
#include "VMShmCommunicator.h"

Communicator * Communicator::Create(ConfigFile::Element & config) {
    const char *type = config.GetValue("type").c_str();
    if (strcasecmp(type, "AfUnix") == 0) {
        mode_t mode = 0660;
        if (config.HasKey("mode"))
            mode = config.GetShortValueFromOctal("mode");
        return new AfUnixCommunicator(config.GetValue("path"), mode);
    } else if (strcasecmp(type, "Tcp") == 0)
        return new TcpCommunicator(
            config.GetValue("hostname").c_str(),
            config.GetShortValue("port"));
#ifdef HAVE_VMCI
    else if (strcasecmp(type, "Vmci") == 0)
        return new VmciCommunicator(config.GetShortValue("port"),
            config.GetShortValue("cid"));
#endif
    else if (strcasecmp(type, "Shm") == 0)
        return new ShmCommunicator();
    else if (strcasecmp(type, "VMShm") == 0)
        return new VMShmCommunicator(
            config.GetValue("hostname").c_str(),
            config.GetShortValue("port"));
    else
        throw "Not a valid type!";
    return NULL;
}

Communicator::~Communicator() {
}

