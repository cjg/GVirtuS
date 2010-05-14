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
#include <cstring>
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

VMSocketCommunicator::VMSocketCommunicator(std::string& path, string &shm) {
    mPath = path;
    mHasSharedMemory = true;
    mSharedMemoryDev = shm;
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

size_t VMSocketCommunicator::GetSharedMemorySize() {
    return mSharedMemorySize;
}

void VMSocketCommunicator::SetSharedMemory(const char* name, size_t size) {
    mpSharedMemoryName = strdup(name);
    if((mSharedMemoryFd = open(mSharedMemoryDev.c_str(), O_RDWR)) < 0) {
        cout << "Failed to open " << mSharedMemoryDev << endl;
        return;
    }
    if((mSharedMemorySize = write(mSharedMemoryFd, mpSharedMemoryName,
            strlen(mpSharedMemoryName) + 1)) < 0) {
        cout << "Failed to obtain shared memory " << mpSharedMemoryName << endl;
        return;
    }

    mpSharedMemory = mmap(NULL, mSharedMemorySize, PROT_READ | PROT_WRITE,
            MAP_SHARED, mSharedMemoryFd, 0);
    mHasSharedMemory = true;
}