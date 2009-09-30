/* 
 * File:   AfUnixCommunicator.h
 * Author: cjg
 *
 * Created on 30 settembre 2009, 12.01
 */

#ifndef _AFUNIXCOMMUNICATOR_H
#define	_AFUNIXCOMMUNICATOR_H

#include <ext/stdio_filebuf.h>
#include "Communicator.h"

class AfUnixCommunicator : public Communicator {
public:
    AfUnixCommunicator(std::string &path);
    AfUnixCommunicator(int fd);
    AfUnixCommunicator(const char * path);
    virtual ~AfUnixCommunicator();
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
    int mSocketFd;
    __gnu_cxx::stdio_filebuf<char> *mpInputBuf;
    __gnu_cxx::stdio_filebuf<char> *mpOutputBuf;
};

#endif	/* _AFUNIXCOMMUNICATOR_H */

