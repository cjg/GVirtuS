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
 * @file   VMSocketCommunicator.h
 * @author Giuseppe Coviello <giuseppe.coviello@uniparthenope.it>
 * @date   Mon Nov 30 15:44:05 2009
 * 
 * @brief  
 * 
 * 
 */

#ifndef _VMSOCKETCOMMUNICATOR_H
#define	_VMSOCKETCOMMUNICATOR_H

#include <ext/stdio_filebuf.h>
#include "Communicator.h"

/**
 * VMSocketCommunicator implements a Communicator for the VMSocket virtual PCI
 * device.
 */
class VMSocketCommunicator : public Communicator {
public:
    VMSocketCommunicator(std::string &path);
    VMSocketCommunicator(const char * path);
    VMSocketCommunicator(std::string & path, std::string & shm);
    virtual ~VMSocketCommunicator();
    void Serve();
    const Communicator * const Accept() const;
    void Connect();
    std::istream & GetInputStream() const;
    std::ostream & GetOutputStream() const;
    void Close();
    bool HasSharedMemory();
    void * GetSharedMemory();
    const char * GetSharedMemoryName();
    size_t GetSharedMemorySize();
    void SetSharedMemory(const char *name, size_t size);
private:
    void InitializeStream();
    std::istream *mpInput;
    std::ostream *mpOutput;
    std::string mPath;
    int mFd;
    __gnu_cxx::stdio_filebuf<char> *mpInputBuf;
    __gnu_cxx::stdio_filebuf<char> *mpOutputBuf;
    bool mHasSharedMemory;
    std::string mSharedMemoryDev;
    int mSharedMemoryFd;
    void *mpSharedMemory;
    char *mpSharedMemoryName;
    int mSharedMemorySize;
};


#endif	/* _VMSOCKETCOMMUNICATOR_H */

