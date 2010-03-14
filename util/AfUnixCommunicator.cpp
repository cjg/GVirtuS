/* 
 * File:   AfUnixCommunicator.cpp
 * Author: cjg
 * 
 * Created on 30 settembre 2009, 12.01
 */

#include <sys/types.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <csignal>
#include "AfUnixCommunicator.h"

using namespace std;

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

std::istream & AfUnixCommunicator::GetInputStream() const {
    return *(this->mpInput);
}

std::ostream & AfUnixCommunicator::GetOutputStream() const {
    return *(this->mpOutput);
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
