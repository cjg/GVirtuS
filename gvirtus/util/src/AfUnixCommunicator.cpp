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
 * @file   AfUnixCommunicator.cpp
 * @author Giuseppe Coviello <giuseppe.coviello@uniparthenope.it>
 * @date   Wed Sep 30 12:01:12 2009
 *
 * @brief
 *
 *
 */

#ifndef _WIN32

#include "AfUnixCommunicator.h"

#include <sys/types.h>
#include <sys/socket.h>
#include <sys/un.h>

#include <csignal>
#include <stdlib.h>
#include <stdio.h>
#include <stddef.h>

#include <unistd.h>

using namespace std;

AfUnixCommunicator::AfUnixCommunicator(const std::string& communicator) {
    const char *valueptr = strstr(communicator.c_str(), "://") + 3;
    const char *modeptr = strchr(valueptr, ':');
    if(modeptr != NULL) {
        mMode = strtol(modeptr + 1, NULL, 8);
        char *path = strdup(valueptr);
        path[modeptr - valueptr] = 0;
        mPath = string(path);
        free(path);
    } else {
        mPath = string(valueptr);
        mMode = 0660;
    }
}

AfUnixCommunicator::AfUnixCommunicator(string &path, mode_t mode) {
    mPath = path;
    mMode = mode;
}

AfUnixCommunicator::AfUnixCommunicator(int fd) {
    mSocketFd = fd;
    InitializeStream();
}

AfUnixCommunicator::AfUnixCommunicator(const char *path, mode_t mode) {
    mPath = string(path);
    mMode = mode;
}

AfUnixCommunicator::~AfUnixCommunicator() {
    // TODO Auto-generated destructor stub
}

void AfUnixCommunicator::Serve() {
    struct sockaddr_un socket_addr;

    unlink(mPath.c_str());

    if ((mSocketFd = socket(AF_UNIX, SOCK_STREAM, 0)) == 0)
        throw "AfUnixCommunicator: Can't create socket.";

    socket_addr.sun_family = AF_UNIX;
    strcpy(socket_addr.sun_path, mPath.c_str());

    if (bind(mSocketFd, (struct sockaddr *) & socket_addr,
            sizeof (struct sockaddr_un)) != 0)
        throw "AfUnixCommunicator: Can't bind socket.";

    if (listen(mSocketFd, 5) != 0)
        throw "AfUnixCommunicator: Can't listen from socket.";

    chmod(mPath.c_str(), mMode);
}

const Communicator * const AfUnixCommunicator::Accept() const {
    unsigned client_socket_fd;
    struct sockaddr_un client_socket_addr;
    unsigned client_socket_addr_size;

    client_socket_addr_size = sizeof (struct sockaddr_un);
    if ((client_socket_fd = accept(mSocketFd,
        (sockaddr *) & client_socket_addr,
        &client_socket_addr_size)) == 0)
        throw "AfUnixCommunicator: Error while accepting connection.";

    return new AfUnixCommunicator(client_socket_fd);
}

void AfUnixCommunicator::Connect() {
    int len;
    struct sockaddr_un remote;

    if((mSocketFd = socket(AF_UNIX, SOCK_STREAM, 0)) == 0)
        throw "AfUnixCommunicator: Can't create socket.";

    remote.sun_family = AF_UNIX;
    strcpy(remote.sun_path, mPath.c_str());
    len = offsetof(struct sockaddr_un, sun_path) + strlen(remote.sun_path);
    if (connect(mSocketFd, (struct sockaddr *) & remote, len) != 0) 
        throw "AfUnixCommunicator: Can't connect to socket.";
    InitializeStream();
}

void AfUnixCommunicator::Close() {
}

size_t AfUnixCommunicator::Read(char* buffer, size_t size) {
    mpInput->read(buffer, size);
    if(mpInput->bad() || mpInput->eof())
        return 0;
    return size;
}

size_t AfUnixCommunicator::Write(const char* buffer, size_t size) {
    mpOutput->write(buffer, size);
    return size;
}

void AfUnixCommunicator::Sync() {
    mpOutput->flush();
}

void AfUnixCommunicator::InitializeStream() {
    mpInputBuf = new __gnu_cxx::stdio_filebuf<char>(mSocketFd,
            std::ios_base::in);
    mpOutputBuf = new __gnu_cxx::stdio_filebuf<char>(mSocketFd,
            std::ios_base::out);
    mpInput = new istream(mpInputBuf);
    mpOutput = new ostream(mpOutputBuf);
    /* FIXME: handle SIGPIPE instead of just ignoring it */
    signal(SIGPIPE, SIG_IGN);
}

#endif
