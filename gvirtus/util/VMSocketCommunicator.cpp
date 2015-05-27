/* 
 * File:   VMSocketCommunicator.cpp
 * Author: cjg
 * 
 * Created on February 14, 2011, 3:57 PM
 */

#include <sys/fcntl.h>
#include <sys/ioctl.h>

#include <unistd.h>

#include <cstring>

#include "VMSocketCommunicator.h"

using namespace std;

VMSocketCommunicator::VMSocketCommunicator(const std::string& communicator) {
    const char *deviceptr = strstr(communicator.c_str(), "://") + 3;
    const char *pathptr = strchr(deviceptr, ':');
    if(pathptr == NULL)
        throw "VMSocketCommunicator: path not specified.";
    mDevice = string(deviceptr).substr(0, pathptr - deviceptr);
    mPath = string(pathptr + 1);
}

void VMSocketCommunicator::Serve() {
    throw "VMSocketCommunicator: cannot Serve";
}

const Communicator * const VMSocketCommunicator::Accept() const {
    throw "VMSocketCommunicator: cannot Accept";
}

void VMSocketCommunicator::Connect() {
    if((mFd = open(mDevice.c_str(), O_RDWR)) < 0)
        throw "VMSocketCommunicator: cannot open device.";
    if(ioctl(mFd, 0, mPath.c_str()) != 0)
        throw "VMSocketCommunicator: cannot connect.";
}

size_t VMSocketCommunicator::Read(char* buffer, size_t size) {
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

size_t VMSocketCommunicator::Write(const char* buffer, size_t size) {
    size_t offset = 0;
    int written;
    while(offset < size) {
        written = write(mFd, buffer + offset, size - offset);
	offset += written;
    }
    return size;
}

void VMSocketCommunicator::Sync() {
    fsync(mFd);
}

void VMSocketCommunicator::Close() {
    close(mFd);
}
