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
 * @file   VmciCommunicator.h
 * @author Giuseppe Coviello <giuseppe.coviello@uniparthenope.it>
 * @date   Wed Nov 4 14:50:05 2009
 *
 * @brief
 *
 *
 */

#ifndef _VMCICOMMUNICATOR_H
#define	_VMCICOMMUNICATOR_H

#include "../config.h"

#ifdef HAVE_VMCI_VMCI_SOCKETS_H

#include <ext/stdio_filebuf.h>

#include <vmci/vmci_sockets.h>

#include "Communicator.h"

#define AF_VMCI VMCISock_GetAFValue()

class VmciCommunicator : public Communicator {
public:
    VmciCommunicator(short port, short cid = -1);
    VmciCommunicator(unsigned fd);
    virtual ~VmciCommunicator();
    void Serve();
    const Communicator * const Accept() const;
    void Connect();
    std::istream & GetInputStream() const;
    std::ostream & GetOutputStream() const;
    void Close();

    /* Semaphores and Shared Memory */
    bool HasSemaphoresAndShm() {
        return false;
    }
    void HostWait() {}
    void HostPost() {}
    void HostSet(int value) {}
    void GuestWait() {}
    void GuestPost() {}
    void GuestSet(int value) {}
    void * GetShm() { return NULL; }
    const char * GetHostSemName() { return NULL; }
    const char * GetGuestSemName() { return NULL; }
    const char * GetShmName() { return NULL; }
private:
    void InitializeStream();
    std::istream *mpInput;
    std::ostream *mpOutput;
    short mCid;
    short mPort;
    int mSocketFd;
    __gnu_cxx::stdio_filebuf<char> *mpInputBuf;
    __gnu_cxx::stdio_filebuf<char> *mpOutputBuf;
};

#endif /* HAVE_VMCI_VMCI_SOCKETS_H */

#endif	/* _VMCICOMMUNICATOR_H */

