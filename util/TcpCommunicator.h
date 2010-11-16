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
 * @file   TcpCommunicator.h
 * @author Giuseppe Coviello <giuseppe.coviello@uniparthenope.it>
 * @date   Thu Oct 8 12:08:33 2009
 * 
 * @brief  
 * 
 * 
 */

#ifndef _TCPCOMMUNICATOR_H
#define	_TCPCOMMUNICATOR_H

#include <ext/stdio_filebuf.h>
#include "Communicator.h"

/**
 * TcpCommunicator implements a Communicator for the TCP/IP socket.
 */
class TcpCommunicator : public Communicator {
public:
    TcpCommunicator(const char *hostname, short port);
    TcpCommunicator(int fd, const char *hostname);
    virtual ~TcpCommunicator();
    void Serve();
    const Communicator * const Accept() const;
    void Connect();
    size_t Read(char *buffer, size_t size);
    size_t Write(const char *buffer, size_t size);
    void Sync();
    void Close();
    bool HasSharedMemory() {
        return false;
    }
    void * GetSharedMemory() { return NULL; }
    const char * GetSharedMemoryName() { return NULL; }
    size_t GetSharedMemorySize() { return 0; }
    void SetSharedMemory(const char *name, size_t size) { }
private:
    void InitializeStream();
    std::istream *mpInput;
    std::ostream *mpOutput;
    std::string mHostname;
    char * mInAddr;
    int mInAddrSize;
    short mPort;
    int mSocketFd;
    __gnu_cxx::stdio_filebuf<char> *mpInputBuf;
    __gnu_cxx::stdio_filebuf<char> *mpOutputBuf;
};

#endif	/* _TCPCOMMUNICATOR_H */

