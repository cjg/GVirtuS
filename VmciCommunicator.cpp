/* 
 * File:   VmciCommunicator.cpp
 * Author: cjg
 * 
 * Created on November 4, 2009, 2:50 PM
 */

#include <cstring>
#include "VmciCommunicator.h"

using namespace std;


VmciCommunicator::VmciCommunicator(short port, short cid) {
    mPort = port;
    if(cid == -1)
        mCid = VMCISock_GetLocalCID();
    else
        mCid = cid;
}

VmciCommunicator::VmciCommunicator(unsigned fd) {
    mSocketFd = fd;
    InitializeStream();
}

VmciCommunicator::~VmciCommunicator() {
}

void VmciCommunicator::Serve() {
    struct sockaddr_vm socket_addr = {0};

    if ((mSocketFd = socket(AF_VMCI, SOCK_STREAM, 0)) == 0)
        throw "VmciCommunicator: Can't create socket.";

    socket_addr.svm_family = AF_VMCI;
    socket_addr.svm_cid = VMADDR_CID_ANY;
    socket_addr.svm_port = mPort;

    if (bind(mSocketFd, (struct sockaddr *) & socket_addr,
            sizeof (struct sockaddr_vm)) != 0)
        throw "VmciCommunicator: Can't bind socket.";

    if (listen(mSocketFd, 5) != 0)
        throw "AfUnixCommunicator: Can't listen from socket.";
}

const Communicator * const VmciCommunicator::Accept() const {
    unsigned client_socket_fd;
    struct sockaddr_vm client_socket_addr;
    unsigned client_socket_addr_size;

    client_socket_addr_size = sizeof (struct sockaddr_vm);
    if ((client_socket_fd = accept(mSocketFd,
        (sockaddr *) & client_socket_addr,
        &client_socket_addr_size)) == 0)
        throw "VmciCommunicator: Error while accepting connection.";

    return new VmciCommunicator(client_socket_fd);
}

void VmciCommunicator::Connect() {
    struct sockaddr_vm remote = {0};

    if((mSocketFd = socket(AF_VMCI, SOCK_STREAM, 0)) == 0)
        throw "VmciCommunicator: Can't create socket.";

    remote.svm_family = AF_VMCI;
    remote.svm_cid = mCid;
    remote.svm_port = mPort;

    if (connect(mSocketFd, (struct sockaddr *) & remote,
        sizeof(struct sockaddr_vm)) != 0)
        throw "VmciCommunicator: Can't connect to socket.";
    InitializeStream();
}

void VmciCommunicator::Close() {
}

std::istream & VmciCommunicator::GetInputStream() const {
    return *(this->mpInput);
}

std::ostream & VmciCommunicator::GetOutputStream() const {
    return *(this->mpOutput);
}

void VmciCommunicator::InitializeStream() {
    mpInputBuf = new __gnu_cxx::stdio_filebuf<char>(mSocketFd,
            std::ios_base::in);
    mpOutputBuf = new __gnu_cxx::stdio_filebuf<char>(mSocketFd,
            std::ios_base::out);
    mpInput = new istream(mpInputBuf);
    mpOutput = new ostream(mpOutputBuf);
}
