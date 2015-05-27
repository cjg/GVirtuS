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
 * @file   VirtioCommunicator.cpp
 * @author Abhijeet Dev <abhijeet@abhijeet-dev.net>
 * @date   Tue Mar 6 15:31:54 2012
 * 
 * @brief  
 * 
 */

#include <sys/fcntl.h>
#include <sys/ioctl.h>

#include <unistd.h>

#include <cstring>

#include "VirtioCommunicator.h"

using namespace std;

VirtioCommunicator::VirtioCommunicator(const std::string& communicator) {
    const char *deviceptr = strstr(communicator.c_str(), "://") + 3;
    mDevice = string(deviceptr);
}

void VirtioCommunicator::Serve() {
    throw "VirtioCommunicator: cannot Serve";
}

const Communicator * const VirtioCommunicator::Accept() const {
    throw "VirtioCommunicator: cannot Accept";
}

void VirtioCommunicator::Connect() {
    if((mFd = open(mDevice.c_str(), O_RDWR)) < 0)
        throw "VirtioCommunicator: cannot open device.";
}

size_t VirtioCommunicator::Read(char* buffer, size_t size) {
    size_t offset = 0;
    int readed;
    while(offset < size) {
        readed = read(mFd, buffer + offset, size - offset);
        if(readed <= 0 && offset == 0)
            return readed;
        offset += readed;
    }
    return size;
}

size_t VirtioCommunicator::Write(const char* buffer, size_t size) {
    size_t offset = 0;
    int written;
    while(offset < size) {
        written = write(mFd, buffer + offset, size - offset);
        offset += written;
    }
    return size;
}

void VirtioCommunicator::Sync() {
    fsync(mFd);
}

void VirtioCommunicator::Close() {
    close(mFd);
}
