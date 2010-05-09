/* 
 * File:   VMSocketCommunicator.cpp
 * Author: cjg
 * 
 * Created on November 30, 2009, 3:44 PM
 */

#include <sys/types.h>
#include <sys/stat.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <csignal>
#include "VMSocketCommunicator.h"

using namespace std;

VMSocketCommunicator::VMSocketCommunicator(string &path) {
    mPath = path;
    mHasSharedMemory = false;
}

VMSocketCommunicator::VMSocketCommunicator(const char *path) {
    mPath = string(path);
    mHasSharedMemory = false;
}

VMSocketCommunicator::VMSocketCommunicator(std::string& path, std::string& shm) {
    mPath = path;
    mSharedMemoryFd = open(shm.c_str(), O_RDWR);
    mpSharedMemory = mmap(NULL, 256 * 1024 * 1024, PROT_READ | PROT_WRITE,
            MAP_SHARED, mSharedMemoryFd, 0);
    mHasSharedMemory = true;
    mpSharedMemoryName = new char[1024];
    ssize_t readed = read(mSharedMemoryFd, mpSharedMemoryName, 1024);
    mpSharedMemoryName[readed] = 0;
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

/*
 * bool HasSemaphoresAndShm();
    void HostWait();
    void HostPost();
    void HostSet(int value);
    void GuestWait();
    void GuestPost();
    void GuestSet(int value);
    void * GetShm();*/

bool VMSocketCommunicator::HasSharedMemory() {
    return mHasSharedMemory;
}

void * VMSocketCommunicator::GetSharedMemory() {
    return mpSharedMemory;
}

const char * VMSocketCommunicator::GetSharedMemoryName() {
    return mpSharedMemoryName;
}