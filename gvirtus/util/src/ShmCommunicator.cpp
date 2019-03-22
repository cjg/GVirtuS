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
 * @file   ShmCommunicator.cpp
 * @author Giuseppe Coviello <giuseppe.coviello@uniparthenope.it>
 * @date   Tue Nov 16 9:52:26 2010
 *
 * @brief
 *
 *
 */

#ifndef _WIN32

#include "ShmCommunicator.h"

#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/un.h>
#include <netinet/in.h>
#include <fcntl.h>
#include <arpa/inet.h>
#include <errno.h>

#include <unistd.h>

#include <cstring>
#include <cstdio>

#include <cstdlib>

using namespace std;

ShmCommunicator::ShmCommunicator(const std::string& communicator) {
    
}

ShmCommunicator::ShmCommunicator() {
}

ShmCommunicator::ShmCommunicator(const char* name) {
    shm_unlink(name);

    mFd = shm_open(name, O_CREAT | O_RDWR, S_IRUSR | S_IWUSR);
    if (mFd == -1)
        throw "ShmCommunicator: cannot open shared memory";

    if (ftruncate(mFd, 2 * 1024 * 1024) == -1)
        throw "ShmCommunicator: cannot request size";

    mpShm = reinterpret_cast<char *> (mmap(NULL, 2 * 1024 * 1024,
            PROT_READ | PROT_WRITE, MAP_SHARED, mFd, 0));
    if (mpShm == MAP_FAILED)
        throw "ShmCommunicator: cannot map shared memory";

    size_t offset = 0;

    /* semaphores */
    mpInEmpty = reinterpret_cast<sem_t *> (mpShm + offset);
    offset += sizeof (sem_t);
    mpInFull = reinterpret_cast<sem_t *> (mpShm + offset);
    offset += sizeof (sem_t);
    mpOutEmpty = reinterpret_cast<sem_t *> (mpShm + offset);
    offset += sizeof (sem_t);
    mpOutFull = reinterpret_cast<sem_t *> (mpShm + offset);
    offset += sizeof (sem_t);

    /* sizes */
    mpInSize = reinterpret_cast<size_t *> (mpShm + offset);
    offset += sizeof (size_t);
    mpOutSize = reinterpret_cast<size_t *> (mpShm + offset);
    offset += sizeof (size_t);

    /* buffers */
    mIOSize = (2 * 1024 * 1024 - offset) / 2;
    mpIn = mpShm + offset;
    offset += mIOSize;
    mpOut = mpShm + offset;

    sem_init(mpInEmpty, 1, 1);
    sem_init(mpInFull, 1, 0);
    sem_init(mpOutEmpty, 1, 1);
    sem_init(mpOutFull, 1, 0);

    mpLocalIn = new char[mIOSize];
    mLocalInSize = 0;
    mLocalInOffset = 0;

    mpLocalOut = new char[mIOSize];
    mLocalOutSize = mIOSize;
    mLocalOutOffset = 0;
}

ShmCommunicator::~ShmCommunicator() {
}

void ShmCommunicator::Serve() {
    struct sockaddr_in addr;
    if ((mSocketFd = socket(AF_INET, SOCK_DGRAM, 0)) < 0)
        throw ("ShmCommunicator: Socket creation error");
    memset((void *) &addr, 0, sizeof (addr)); /* clear server address */
    addr.sin_family = AF_INET; /* address type is INET */
    addr.sin_port = htons(6666); /* daytime port is 13 */
    addr.sin_addr.s_addr = htonl(INADDR_ANY); /* connect from anywhere */
    /* bind socket */
    if (bind(mSocketFd, (struct sockaddr *) &addr, sizeof (addr)) < 0)
        throw ("ShmCommunicator: Bind error");
}

const Communicator * const ShmCommunicator::Accept() const {
    char name[1024];
    struct sockaddr_in addr;
    socklen_t len = sizeof (addr);
    memset(&addr, 0, sizeof (addr));
    recvfrom(mSocketFd, name, 1024, 0, (struct sockaddr *) &addr, &len);
    snprintf(name, 1024, "/gvirtus-%d", rand());
    Communicator *client = new ShmCommunicator(name);
    sendto(mSocketFd, name, 1024, 0, (struct sockaddr *) &addr, sizeof (addr));
    return client;
}

void ShmCommunicator::Connect() {
    char name[1024];
    struct sockaddr_in addr;

    if ((mSocketFd = socket(AF_INET, SOCK_DGRAM, 0)) < 0)
        throw ("ShmCommunicator: Socket creation error");
    memset((void *) &addr, 0, sizeof (addr)); /* clear server address */
    addr.sin_addr.s_addr = inet_addr("127.0.0.1");
    addr.sin_family = AF_INET; /* address type is INET */
    addr.sin_port = htons(6666); /* daytime port is 13 */
    /* build address using inet_pton */
    memset(name, 0, 1024);
    sendto(mSocketFd, name, 1024, 0, (struct sockaddr *) &addr, sizeof (addr));
    recvfrom(mSocketFd, name, 1024, 0, NULL, NULL);

    mFd = shm_open(name, O_RDWR, S_IRUSR | S_IWUSR);
    if (mFd == -1)
        throw "ShmCommunicator: cannot open shared memory";


    if (ftruncate(mFd, 2 * 1024 * 1024) == -1)
        throw "ShmCommunicator: cannot request size";

    mpShm = reinterpret_cast<char *> (mmap(NULL, 2 * 1024 * 1024,
            PROT_READ | PROT_WRITE, MAP_SHARED, mFd, 0));
    if (mpShm == MAP_FAILED)
        throw "ShmCommunicator: cannot map shared memory";

    size_t offset = 0;
    /* semaphores */
    mpOutEmpty = reinterpret_cast<sem_t *> (mpShm + offset);
    offset += sizeof (sem_t);
    mpOutFull = reinterpret_cast<sem_t *> (mpShm + offset);
    offset += sizeof (sem_t);
    mpInEmpty = reinterpret_cast<sem_t *> (mpShm + offset);
    offset += sizeof (sem_t);
    mpInFull = reinterpret_cast<sem_t *> (mpShm + offset);
    offset += sizeof (sem_t);

    /* sizes */
    mpOutSize = reinterpret_cast<size_t *> (mpShm + offset);
    offset += sizeof (size_t);
    mpInSize = reinterpret_cast<size_t *> (mpShm + offset);
    offset += sizeof (size_t);

    /* buffers */
    mIOSize = (2 * 1024 * 1024 - offset) / 2;
    mpOut = mpShm + offset;
    offset += mIOSize;
    mpIn = mpShm + offset;

    mpLocalIn = new char[mIOSize];
    mLocalInSize = 0;
    mLocalInOffset = 0;

    mpLocalOut = new char[mIOSize];
    mLocalOutSize = mIOSize;
    mLocalOutOffset = 0;
}

#define min(a,b) ((a) < (b) ? (a) : (b))

size_t ShmCommunicator::ReadPacket(char* buffer) {
    size_t size;
    sem_wait(mpInFull);
    size = *mpInSize;
    memmove(buffer, mpIn, size);
    sem_post(mpInEmpty);
    return size;
}

size_t ShmCommunicator::Read(char* buffer, size_t size) {
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

size_t ShmCommunicator::Write(const char* buffer, size_t size) {
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
        sem_wait(mpOutEmpty);
        *mpOutSize = mIOSize;
        memmove(mpOut, buffer + offset, mIOSize);
        sem_post(mpOutFull);
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

void ShmCommunicator::Sync() {
    if (mLocalOutOffset == 0)
        return;
    sem_wait(mpOutEmpty);
    *mpOutSize = mLocalOutOffset;
    memmove(mpOut, mpLocalOut, mLocalOutOffset);
    sem_post(mpOutFull);
    mLocalOutSize = mIOSize;
    mLocalOutOffset = 0;
}

void ShmCommunicator::Close() {
    sem_wait(mpInEmpty);
    *mpOutSize = 0;
    sem_post(mpOutFull);
}

#endif
