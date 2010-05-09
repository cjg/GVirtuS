/* 
 * File:   VMSocketCommunicator.h
 * Author: cjg
 *
 * Created on November 30, 2009, 3:44 PM
 */

#ifndef _VMSOCKETCOMMUNICATOR_H
#define	_VMSOCKETCOMMUNICATOR_H

#include <ext/stdio_filebuf.h>
#include "Communicator.h"

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
    size_t GetSharedMemorySize() { return 0; }
    void SetSharedMemory(const char *name, size_t size) { }
private:
    void InitializeStream();
    std::istream *mpInput;
    std::ostream *mpOutput;
    std::string mPath;
    int mFd;
    __gnu_cxx::stdio_filebuf<char> *mpInputBuf;
    __gnu_cxx::stdio_filebuf<char> *mpOutputBuf;
    bool mHasSharedMemory;
    int mSharedMemoryFd;
    void *mpSharedMemory;
    char *mpSharedMemoryName;
};


#endif	/* _VMSOCKETCOMMUNICATOR_H */

