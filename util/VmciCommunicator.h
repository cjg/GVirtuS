/* 
 * File:   VmciCommunicator.h
 * Author: cjg
 *
 * Created on November 4, 2009, 2:50 PM
 */

#ifndef _VMCICOMMUNICATOR_H
#define	_VMCICOMMUNICATOR_H

#include "../config.h"

#ifdef HAVE_VMCI_VMCI_SOCKETS_H

#include <vmci/vmci_sockets.h>
#include <ext/stdio_filebuf.h>
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

