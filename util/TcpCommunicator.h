/* 
 * File:   TcpCommunicator.h
 * Author: cjg
 *
 * Created on 8 ottobre 2009, 12.08
 */

#ifndef _TCPCOMMUNICATOR_H
#define	_TCPCOMMUNICATOR_H

#include <ext/stdio_filebuf.h>
#include "Communicator.h"

class TcpCommunicator : public Communicator {
public:
    TcpCommunicator(const char *hostname, short port);
    TcpCommunicator(int fd);
    virtual ~TcpCommunicator();
    void Serve();
    const Communicator * const Accept() const;
    void Connect();
    std::istream & GetInputStream() const;
    std::ostream & GetOutputStream() const;
    void Close();
    bool HasSharedMemory() {
        return false;
    }
    void * GetSharedMemory() { return NULL; }
    const char * GetSharedMemoryName() { return NULL; }
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

