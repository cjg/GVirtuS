/*
 * gVirtuS -- A GPGPU transparent virtualization component.
 *
 * Copyright (C) 2009-2010  The University of Napoli Parthenope at Naples.
 *
 * This file is part of gVirtuS.
 *
 * gVirtuS is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * gVirtuS is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with gVirtuS; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
 *
 * Written by: Giuseppe Coviello <giuseppe.coviello@uniparthenope.it>,
 *             Department of Applied Science
 */

/**
 * @file   VMShmCommunicator.cpp
 * @author Giuseppe Coviello <giuseppe.coviello@uniparthenope.it>
 * @date   Tue Nov 16 12:34:26 2010
 *
 * @brief
 *
 *
 */

#ifndef _WIN32

#include "VMShmCommunicator.h"

#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/un.h>
#include <netinet/in.h>
#include <fcntl.h>
#include <arpa/inet.h>
#include <errno.h>
#include <sys/ioctl.h>

#include <unistd.h>

#include <cstring>
#include <cstdio>
#include <cstdlib>

using namespace std;

void vmshm_sem_init(vmshm_sem_t *self, void *shm, size_t *offset, int value) {
    self->mLock = (pthread_spinlock_t *) (((char *) shm) + *offset);
    *offset += sizeof (pthread_spinlock_t);
    self->mCounter = (int *) (((char *) shm) + *offset);
    *offset += sizeof (int);
    pthread_spin_init(self->mLock, PTHREAD_PROCESS_SHARED);
    *(self->mCounter) = value;
}

void vmshm_sem_get(vmshm_sem_t *self, void *shm, size_t *offset) {
    self->mLock = (pthread_spinlock_t *) (((char *) shm) + *offset);
    *offset += sizeof (pthread_spinlock_t);
    self->mCounter = (int *) (((char *) shm) + *offset);
    *offset += sizeof (int);
}

int vmshm_sem_post(vmshm_sem_t self) {
    int value;
    pthread_spin_lock(self.mLock);
    self.mCounter[0] += 1;
    value = *(self.mCounter);
    pthread_spin_unlock(self.mLock);
    return value;
}

int vmshm_sem_wait(vmshm_sem_t self) {
    int value = -1;
    while (value < 0) {
        pthread_spin_lock(self.mLock);
        if (*(self.mCounter) > 0) {
            *self.mCounter -= 1;
            value = *(self.mCounter);
        }
        pthread_spin_unlock(self.mLock);
    }
    return value;
}

VMShmCommunicator::VMShmCommunicator(const std::string& communicator) {
    const char *valueptr = strstr(communicator.c_str(), "://") + 3;
    const char *portptr = strchr(valueptr, ':');
    if (portptr == NULL)
        throw "Port not specified.";
    mPort = strtol(portptr + 1, NULL, 10);
    char *hostname = strdup(valueptr);
    hostname[portptr - valueptr] = 0;
    mHostname = string(hostname);
    free(hostname);
}

VMShmCommunicator::VMShmCommunicator(const char *hostname, short port) {
    mHostname = string(hostname);
    mPort = port;
}

VMShmCommunicator::VMShmCommunicator(const char* name) {
    shm_unlink(name);

    mFd = shm_open(name, O_CREAT | O_RDWR, S_IRUSR | S_IWUSR);
    if (mFd == -1)
        throw "VMShmCommunicator: cannot open shared memory";

    if (ftruncate(mFd, 1024 * 1024) == -1)
        throw "VMShmCommunicator: cannot request size";

    mpVMShm = reinterpret_cast<char *> (mmap(NULL, 1024 * 1024,
            PROT_READ | PROT_WRITE, MAP_SHARED, mFd, 0));
    if (mpVMShm == MAP_FAILED)
        throw "VMShmCommunicator: cannot map shared memory";

    size_t offset = 0;

    /* semaphores */
    vmshm_sem_init(&mpInEmpty, mpVMShm, &offset, 1);
    vmshm_sem_init(&mpInFull, mpVMShm, &offset, 0);
    vmshm_sem_init(&mpOutEmpty, mpVMShm, &offset, 1);
    vmshm_sem_init(&mpOutFull, mpVMShm, &offset, 0);

    /* sizes */
    mpInSize = reinterpret_cast<size_t *> (mpVMShm + offset);
    offset += sizeof (size_t);
    mpOutSize = reinterpret_cast<size_t *> (mpVMShm + offset);
    offset += sizeof (size_t);

    /* buffers */
    mIOSize = (1024 * 1024 - offset) / 2;
    mpIn = mpVMShm + offset;
    offset += mIOSize;
    mpOut = mpVMShm + offset;

    mpLocalIn = new char[mIOSize];
    mLocalInSize = 0;
    mLocalInOffset = 0;

    mpLocalOut = new char[mIOSize];
    mLocalOutSize = mIOSize;
    mLocalOutOffset = 0;
}

VMShmCommunicator::~VMShmCommunicator() {
}

void VMShmCommunicator::Serve() {
    struct sockaddr_in addr;
    if ((mSocketFd = socket(AF_INET, SOCK_DGRAM, 0)) < 0)
        throw ("VMShmCommunicator: Socket creation error");
    memset((void *) &addr, 0, sizeof (addr)); /* clear server address */
    addr.sin_family = AF_INET; /* address type is INET */
    addr.sin_port = htons(mPort); /* daytime port is 13 */
    addr.sin_addr.s_addr = htonl(INADDR_ANY); /* connect from anywhere */
    /* bind socket */
    if (bind(mSocketFd, (struct sockaddr *) &addr, sizeof (addr)) < 0)
        throw ("VMShmCommunicator: Bind error");
}

const Communicator * const VMShmCommunicator::Accept() const {
    char name[1024];
    struct sockaddr_in addr;
    socklen_t len = sizeof (addr);
    memset(&addr, 0, sizeof (addr));
    recvfrom(mSocketFd, name, 1024, 0, (struct sockaddr *) &addr, &len);
    snprintf(name, 1024, "/gvirtus-%d", rand());
    Communicator *client = new VMShmCommunicator(name);
    sendto(mSocketFd, name, 1024, 0, (struct sockaddr *) &addr, sizeof (addr));
    return client;
}

void VMShmCommunicator::Connect() {
    char name[1024];
    struct sockaddr_in addr;

    if ((mSocketFd = socket(AF_INET, SOCK_DGRAM, 0)) < 0)
        throw ("VMShmCommunicator: Socket creation error");
    memset((void *) &addr, 0, sizeof (addr)); /* clear server address */
    addr.sin_addr.s_addr = inet_addr(mHostname.c_str());
    addr.sin_family = AF_INET; /* address type is INET */
    addr.sin_port = htons(mPort); /* daytime port is 13 */
    /* build address using inet_pton */
    memset(name, 0, 1024);
    sendto(mSocketFd, name, 1024, 0, (struct sockaddr *) &addr, sizeof (addr));
    recvfrom(mSocketFd, name, 1024, 0, NULL, NULL);

    mFd = open("/dev/vmshm0", O_RDWR);
    if (mFd == -1)
        throw "VMShmCommunicator: cannot open VMShm device";


    if (ioctl(mFd, 0, name))
        throw "VMShmCommunicator: cannot request shared memory";

    mpVMShm = reinterpret_cast<char *> (mmap(NULL, 1024 * 1024,
            PROT_READ | PROT_WRITE, MAP_SHARED, mFd, 0));
    if (mpVMShm == MAP_FAILED)
        throw "VMShmCommunicator: cannot map shared memory";

    size_t offset = 0;
    /* semaphores */
    vmshm_sem_get(&mpOutEmpty, mpVMShm, &offset);
    vmshm_sem_get(&mpOutFull, mpVMShm, &offset);
    vmshm_sem_get(&mpInEmpty, mpVMShm, &offset);
    vmshm_sem_get(&mpInFull, mpVMShm, &offset);

    /* sizes */
    mpOutSize = reinterpret_cast<size_t *> (mpVMShm + offset);
    offset += sizeof (size_t);
    mpInSize = reinterpret_cast<size_t *> (mpVMShm + offset);
    offset += sizeof (size_t);

    /* buffers */
    mIOSize = (1024 * 1024 - offset) / 2;
    mpOut = mpVMShm + offset;
    offset += mIOSize;
    mpIn = mpVMShm + offset;

    mpLocalIn = new char[mIOSize];
    mLocalInSize = 0;
    mLocalInOffset = 0;

    mpLocalOut = new char[mIOSize];
    mLocalOutSize = mIOSize;
    mLocalOutOffset = 0;
}

#define min(a,b) ((a) < (b) ? (a) : (b))

size_t VMShmCommunicator::ReadPacket(char* buffer) {
    size_t size;
    vmshm_sem_wait(mpInFull);
    size = *mpInSize;
    memmove(buffer, mpIn, size);
    vmshm_sem_post(mpInEmpty);
    return size;
}

size_t VMShmCommunicator::Read(char* buffer, size_t size) {
    size_t chunk, offset = 0;

    /* consume bytes in LocalIn */
    chunk = min(mLocalInSize, size);
    if(chunk > 0) {
        memmove(buffer, mpLocalIn + mLocalInOffset, chunk);
        mLocalInSize -= chunk;
        mLocalInOffset += chunk;
        offset = chunk;
    }
    
    /* read only complete packets */
    while(offset < size && size - offset >= mIOSize) {
        if((chunk = ReadPacket(buffer + offset)) == 0)
            return 0;
        offset += chunk;
    }
    
    /* if it is needed some other spare byte we read a full packet storing it
     * in LocalIn */
    if(offset < size) {
        chunk = size - offset;
        if((mLocalInSize = ReadPacket(mpLocalIn)) == 0)
            return 0;
        memmove(buffer + offset, mpLocalIn, chunk);
        mLocalInSize -= chunk;
        mLocalInOffset = chunk;
        offset += chunk;
    }
    return offset;
}

size_t VMShmCommunicator::Write(const char* buffer, size_t size) {
    size_t chunk, offset = 0;

    /* fill LocalOut if there is something in it */
    if(mLocalOutOffset > 0) {
        chunk = min(size, mLocalOutSize);
        memmove(mpLocalOut + mLocalOutOffset, buffer, chunk);
        mLocalOutSize -= chunk;
        mLocalOutOffset += chunk;
        offset = chunk;
    }

    /* sync if localout is full */
    if(mLocalOutSize == 0)
        Sync();

    /* write only complete packets */
    while(offset < size && size - offset >= mIOSize) {
        vmshm_sem_wait(mpOutEmpty);
        *mpOutSize = mIOSize;
        memmove(mpOut, buffer + offset, mIOSize);
        vmshm_sem_post(mpOutFull);
        offset += mIOSize;
    }

    /* fill LocalOut with spare bytes */
    if(offset < size) {
        chunk = size - offset;
        memmove(mpLocalOut, buffer + offset, chunk);
        mLocalOutSize = mIOSize - chunk;
        mLocalOutOffset = chunk;
    }

    return size;
}

void VMShmCommunicator::Sync() {
    if (mLocalOutOffset == 0)
        return;
    vmshm_sem_wait(mpOutEmpty);
    *mpOutSize = mLocalOutOffset;
    memmove(mpOut, mpLocalOut, mLocalOutOffset);
    vmshm_sem_post(mpOutFull);
    mLocalOutSize = mIOSize;
    mLocalOutOffset = 0;
}

void VMShmCommunicator::Close() {
    vmshm_sem_wait(mpInEmpty);
    *mpOutSize = 0;
    vmshm_sem_post(mpOutFull);
}

#endif
