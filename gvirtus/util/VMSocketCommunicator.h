/* 
 * File:   VMSocketCommunicator.h
 * Author: cjg
 *
 * Created on February 14, 2011, 3:57 PM
 */

#ifndef VMSOCKETCOMMUNICATOR_H
#define	VMSOCKETCOMMUNICATOR_H

#include "Communicator.h"

class VMSocketCommunicator : public Communicator {
public:
    VMSocketCommunicator(const std::string &communicator);
    void Serve();
    const Communicator * const Accept() const;
    void Connect();
    size_t Read(char *buffer, size_t size);
    size_t Write(const char *buffer, size_t size);
    void Sync();
    void Close();
private:
    int mFd;
    std::string mPath;
    std::string mDevice;
};

#endif	/* VMSOCKETCOMMUNICATOR_H */

