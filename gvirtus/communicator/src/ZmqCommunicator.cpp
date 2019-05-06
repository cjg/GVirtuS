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
 * @file   ZmqCommunicator.cpp
 * @author Giuseppe Coviello <giuseppe.coviello@uniparthenope.it>
 * @date   Thu Oct 8 12:08:33 2009
 *
 * @brief
 *
 *
 */
//#define DEBUG

#include "ZmqCommunicator.h"

#include <iostream>
#include <string>
#include <zmq.hpp>
#ifndef _WIN32
#include <unistd.h>
#else
#include <windows.h>

#define sleep(n) Sleep(n)
#endif

#ifndef _WIN32
#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/types.h>
#else
#include <WinSock2.h>
static bool initialized = false;
#endif

#include <cstdlib>
#include <cstring>

using namespace std;

namespace gvirtus::comm {

  ZmqCommunicator::ZmqCommunicator(const std::string &communicator) {
#ifdef _WIN32
    if (!initialized) {
      WSADATA data;
      if (WSAStartup(MAKEWORD(2, 2), &data) != 0)
        throw "Cannot initialized WinSock.";
      initialized = true;
    }
#endif
    const char *valueptr = strstr(communicator.c_str(), "://") + 3;
    const char *portptr = strchr(valueptr, ':');
    if (portptr == NULL)
      throw "Port not specified.";
    mPort = (short)strtol(portptr + 1, NULL, 10);
#ifdef _WIN32
    char *hostname = _strdup(valueptr);
#else
    char *hostname = strdup(valueptr);
#endif
    hostname[portptr - valueptr] = 0;
    mHostname = string(hostname);
    struct hostent *ent = gethostbyname(hostname);
    free(hostname);
    if (ent == NULL)
      throw "ZmqCommunicator: Can't resolve hostname '" + mHostname + "'.";
    mInAddrSize = ent->h_length;
    mInAddr = new char[mInAddrSize];
    memcpy(mInAddr, *ent->h_addr_list, mInAddrSize);
  }

  ZmqCommunicator::ZmqCommunicator(const char *hostname, short port) {
    mHostname = string(hostname);
    struct hostent *ent = gethostbyname(hostname);
    if (ent == NULL)
      throw "ZmqCommunicator: Can't resolve hostname '" + mHostname + "'.";
    mInAddrSize = ent->h_length;
    mInAddr = new char[mInAddrSize];
    memcpy(mInAddr, *ent->h_addr_list, mInAddrSize);
    mPort = port;
  }

  ZmqCommunicator::ZmqCommunicator(int fd, const char *hostname) {
    mSocketFd = fd;
    InitializeStream();
  }

  ZmqCommunicator::~ZmqCommunicator() { delete[] mInAddr; }

  void
  ZmqCommunicator::Serve() {

    zmq_context = new zmq::context_t(1);
    zmq_socket = new zmq::socket_t(*zmq_context, ZMQ_REP);
    zmq_socket->bind("tcp://*:5555");

    struct sockaddr_in socket_addr;

    if ((mSocketFd = socket(AF_INET, SOCK_STREAM, 0)) == 0)
      throw "**ZmqCommunicator: Can't create socket.";

    memset((char *)&socket_addr, 0, sizeof(struct sockaddr_in));

    socket_addr.sin_family = AF_INET;
    socket_addr.sin_port = htons(mPort);
    socket_addr.sin_addr.s_addr = INADDR_ANY;

    char on = 1;
    setsockopt(mSocketFd, SOL_SOCKET, SO_REUSEADDR, &on, sizeof(on));

    int result = bind(mSocketFd, (struct sockaddr *)&socket_addr, sizeof(struct sockaddr_in));
    if (result != 0)
      throw "ZmqCommunicator: Can't bind socket.";

    if (listen(mSocketFd, 5) != 0)
      throw "AfUnixCommunicator: Can't listen from socket.";
  }

  const Communicator *const
  ZmqCommunicator::Accept() const {
    unsigned client_socket_fd;
    struct sockaddr_in client_socket_addr;
#ifndef _WIN32
    unsigned client_socket_addr_size;
#else
    int client_socket_addr_size;
#endif
    client_socket_addr_size = sizeof(struct sockaddr_in);
    if ((client_socket_fd = accept(mSocketFd, (sockaddr *)&client_socket_addr, &client_socket_addr_size)) == 0)
      throw "ZmqCommunicator: Error while accepting connection.";

    return new ZmqCommunicator(client_socket_fd, inet_ntoa(client_socket_addr.sin_addr));
  }

  void
  ZmqCommunicator::Connect() {
    struct sockaddr_in remote;

    if ((mSocketFd = socket(AF_INET, SOCK_STREAM, 0)) == 0)
      throw "ZmqCommunicator: Can't create socket.";

    remote.sin_family = AF_INET;
    remote.sin_port = htons(mPort);
    memcpy(&remote.sin_addr, mInAddr, mInAddrSize);

    if (connect(mSocketFd, (struct sockaddr *)&remote, sizeof(struct sockaddr_in)) != 0)
      throw "ZmqCommunicator: Can't connect to socket.";
    InitializeStream();
  }

  void
  ZmqCommunicator::Close() {
    throw "ZmqCommunicator::Close()\n";
  }

  size_t
  ZmqCommunicator::Read(char *buffer, size_t size) {
    mpInput->read(buffer, size);
#ifdef DEBUG
    for (unsigned int i = 0; i < size; i++)
      printf("%d LETTO %02X\n", i, buffer[i]);
#endif
    if (mpInput->bad() || mpInput->eof())
      return 0;
    return size;
  }

  size_t
  ZmqCommunicator::Write(const char *buffer, size_t size) {
    mpOutput->write(buffer, size);
#ifdef DEBUG
    for (unsigned int i = 0; i < size; i++)
      printf("%d SCRITTO %02X \n", i, buffer[i]);
#endif
    return size;
  }

  void
  ZmqCommunicator::Sync() {
    mpOutput->flush();
  }

  void
  ZmqCommunicator::InitializeStream() {
#ifdef _WIN32
    FILE *i = _fdopen(mSocketFd, "r");
    FILE *o = _fdopen(mSocketFd, "w");
    mpInputBuf = new filebuf(i);
    mpOutputBuf = new filebuf(o);
#else
    mpInputBuf = new __gnu_cxx::stdio_filebuf<char>(mSocketFd, ios_base::in);
    mpOutputBuf = new __gnu_cxx::stdio_filebuf<char>(mSocketFd, ios_base::out);
#endif
    mpInput = new istream(mpInputBuf);
    mpOutput = new ostream(mpOutputBuf);
  }

} // namespace gvirtus::comm