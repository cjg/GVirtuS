/* 
 * File:   AfUnixCommunicator.h
 * Author: cjg
 *
 * Created on 30 settembre 2009, 12.01
 */

#ifndef _AFUNIXCOMMUNICATOR_H
#define	_AFUNIXCOMMUNICATOR_H

#include <ext/stdio_filebuf.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include "Communicator.h"

class AfUnixCommunicator : public Communicator {
public:
    AfUnixCommunicator(std::string &path, mode_t mode = 00660);
    AfUnixCommunicator(int fd);
    AfUnixCommunicator(const char * path, mode_t mode = 00660);
    virtual ~AfUnixCommunicator();
    void Serve();
    const Communicator * const Accept() const;
    void Connect();
    std::istream & GetInputStream() const;
    std::ostream & GetOutputStream() const;
    void Close();
    bool HasSharedMemory() {
        return true;
    }

    void * GetSharedMemory() {
        return mpSharedMemory;
    }

    const char * GetSharedMemoryName() {
        return mpSharedMemoryName;
    }

    void SetSharedMemory(const char *name, size_t size) {
        mpSharedMemoryName = strdup(name);
        mSharedMemoryFd = shm_open(name, O_RDWR, S_IRWXU);

	if((mpSharedMemory = mmap(NULL, size, PROT_READ | PROT_WRITE,
            MAP_SHARED, mSharedMemoryFd, 0)) == MAP_FAILED) {
		std::cout << "Failed to mmap" << std::endl;
                mpSharedMemory = NULL;
        }
    }

private:
    void InitializeStream();
    std::istream *mpInput;
    std::ostream *mpOutput;
    std::string mPath;
    mode_t mMode;
    int mSocketFd;
    __gnu_cxx::stdio_filebuf<char> *mpInputBuf;
    __gnu_cxx::stdio_filebuf<char> *mpOutputBuf;
    int mSharedMemoryFd;
    void *mpSharedMemory;
    char *mpSharedMemoryName;
};

#endif	/* _AFUNIXCOMMUNICATOR_H */

