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
    virtual ~VMSocketCommunicator();
    void Serve();
    const Communicator * const Accept() const;
    void Connect();
    std::istream & GetInputStream() const;
    std::ostream & GetOutputStream() const;
    void Close();
private:
    void InitializeStream();
    std::istream *mpInput;
    std::ostream *mpOutput;
    std::string mPath;
    int mFd;
    __gnu_cxx::stdio_filebuf<char> *mpInputBuf;
    __gnu_cxx::stdio_filebuf<char> *mpOutputBuf;
};


#endif	/* _VMSOCKETCOMMUNICATOR_H */

