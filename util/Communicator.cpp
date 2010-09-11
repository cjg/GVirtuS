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

#include <cstring>
#include <cstdlib>
#include "AfUnixCommunicator.h"
#include "TcpCommunicator.h"
#include "VmciCommunicator.h"
#include "VMSocketCommunicator.h"
#include "Communicator.h"

Communicator * Communicator::Create(ConfigFile::Element & config) {
    const char *type = config.GetValue("type").c_str();
    if (strcasecmp(type, "AfUnix") == 0) {
        mode_t mode = 0660;
        bool use_shm = false;
        if (config.HasKey("mode"))
            mode = config.GetShortValueFromOctal("mode");
        if (config.HasKey("use_shm"))
            use_shm = config.GetBoolValue("use_shm");
        return new AfUnixCommunicator(config.GetValue("path"), mode, use_shm);
    } else if (strcasecmp(type, "Tcp") == 0)
        return new TcpCommunicator(
            config.GetValue("hostname").c_str(),
            config.GetShortValue("port"));
#ifdef HAVE_VMCI
    else if (strcasecmp(type, "Vmci") == 0)
        return new VmciCommunicator(config.GetShortValue("port"),
            config.GetShortValue("cid"));
#endif
    else if (strcasecmp(type, "VMSocket") == 0)
        if (config.HasKey("shm"))
            return new VMSocketCommunicator(config.GetValue("device"),
                config.GetValue("shm"));
        else
            return new VMSocketCommunicator(config.GetValue("device"));
    else
        throw "Not a valid type!";
    return NULL;
}

Communicator::~Communicator() {
}

