/* 
 * File:   VMSocketCommunicator.cpp
 * Author: cjg
 * 
 * Created on November 30, 2009, 3:44 PM
 */

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <csignal>
#include "VMSocketCommunicator.h"

using namespace std;

VMSocketCommunicator::VMSocketCommunicator(string &path) {
    mPath = path;
}

VMSocketCommunicator::VMSocketCommunicator(const char *path) {
    mPath = string(path);
}

VMSocketCommunicator::~VMSocketCommunicator() {
    // TODO Auto-generated destructor stub
}

void VMSocketCommunicator::Serve() {
    throw "VMSocketCommunicator: Can't start server.";
}

const Communicator * const VMSocketCommunicator::Accept() const {
    return NULL;
}

void VMSocketCommunicator::Connect() {
    if((mFd = open(mPath.c_str(), O_RDWR)) < 0)
        throw "VMSocketCommunicator: Can't create socket.";

    InitializeStream();
}

void VMSocketCommunicator::Close() {
}

std::istream & VMSocketCommunicator::GetInputStream() const {
    return *(this->mpInput);
}

std::ostream & VMSocketCommunicator::GetOutputStream() const {
    return *(this->mpOutput);
}

void VMSocketCommunicator::InitializeStream() {
    mpInputBuf = new __gnu_cxx::stdio_filebuf<char>(mFd, std::ios_base::in);
    mpOutputBuf = new __gnu_cxx::stdio_filebuf<char>(mFd, std::ios_base::out);
    mpInput = new istream(mpInputBuf);
    mpOutput = new ostream(mpOutputBuf);
    /* FIXME: handle SIGPIPE instead of just ignoring it */
    signal(SIGPIPE, SIG_IGN);
}
