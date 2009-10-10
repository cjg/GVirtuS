/* 
 * File:   TcpCommunicator.cpp
 * Author: cjg
 * 
 * Created on 8 ottobre 2009, 12.08
 */

#include <cstring>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <netdb.h>
#include "TcpCommunicator.h"

using namespace std;

TcpCommunicator::TcpCommunicator(const char *hostname, short port) {
    mHostname = string(hostname);
    struct hostent *ent = gethostbyname(hostname);
    if(ent == NULL)
        throw "TcpCommunicator: Can't resolve hostname '" + mHostname + "'.";
    mInAddrSize = ent->h_length;
    mInAddr = new char[mInAddrSize];
    memcpy(mInAddr, *ent->h_addr_list, mInAddrSize);
    mPort = port;
}

TcpCommunicator::TcpCommunicator(int fd) {
    mSocketFd = fd;
    InitializeStream();
}

TcpCommunicator::~TcpCommunicator() {
    delete[] mInAddr;
}

void TcpCommunicator::Serve() {
    struct sockaddr_in socket_addr;


    if ((mSocketFd = socket(AF_INET, SOCK_STREAM, 0)) == 0)
        throw "TcpCommunicator: Can't create socket.";

    socket_addr.sin_family = AF_INET;
    socket_addr.sin_port = htons(mPort);
    memcpy(&socket_addr.sin_addr, mInAddr, mInAddrSize);

    if (bind(mSocketFd, (struct sockaddr *) & socket_addr,
            sizeof (struct sockaddr_in)) != 0)
        throw "TcpCommunicator: Can't bind socket.";

    if (listen(mSocketFd, 5) != 0)
        throw "AfUnixCommunicator: Can't listen from socket.";
}

const Communicator * const TcpCommunicator::Accept() const {
    unsigned client_socket_fd;
    struct sockaddr_in client_socket_addr;
    unsigned client_socket_addr_size;

    client_socket_addr_size = sizeof (struct sockaddr_in);
    if ((client_socket_fd = accept(mSocketFd,
        (sockaddr *) & client_socket_addr,
        &client_socket_addr_size)) == 0)
        throw "TcpCommunicator: Error while accepting connection.";

    return new TcpCommunicator(client_socket_fd);
}

void TcpCommunicator::Connect() {
    struct sockaddr_in remote;

    if((mSocketFd = socket(AF_INET, SOCK_STREAM, 0)) == 0)
        throw "TcpCommunicator: Can't create socket.";

    remote.sin_family = AF_INET;
    remote.sin_port = htons(mPort);
    memcpy(&remote.sin_addr, mInAddr, mInAddrSize);

    if (connect(mSocketFd, (struct sockaddr *) & remote, 
        sizeof(struct sockaddr_in)) != 0)
        throw "TcpCommunicator: Can't connect to socket.";
    InitializeStream();
}

void TcpCommunicator::Close() {
}

std::istream & TcpCommunicator::GetInputStream() const {
    return *(this->mpInput);
}

std::ostream & TcpCommunicator::GetOutputStream() const {
    return *(this->mpOutput);
}

void TcpCommunicator::InitializeStream() {
    mpInputBuf = new __gnu_cxx::stdio_filebuf<char>(mSocketFd,
            std::ios_base::in);
    mpOutputBuf = new __gnu_cxx::stdio_filebuf<char>(mSocketFd,
            std::ios_base::out);
    mpInput = new istream(mpInputBuf);
    mpOutput = new ostream(mpOutputBuf);
}

