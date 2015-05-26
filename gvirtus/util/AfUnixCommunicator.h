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
 * @file   AfUnixCommunicator.h
 * @author Giuseppe Coviello <giuseppe.coviello@uniparthenope.it>
 * @date   Wed Sep 30 12:01:12 2009
 * 
 * @brief  
 * 
 * 
 */

#ifndef _AFUNIXCOMMUNICATOR_H
#define	_AFUNIXCOMMUNICATOR_H

#ifndef __WIN32

#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>

#include <ext/stdio_filebuf.h>

#include "Communicator.h"



/**
 * AfUnixCommunicator implements a Communicator for the AF_UNIX socket in the unix domain.
 */
class AfUnixCommunicator : public Communicator {
public:
    AfUnixCommunicator(const std::string &communicator);
    /** 
     * Creates a new AfUnixCommunicator for binding or connecting it to the
     * AF_UNIX socket specified from path.
     * 
     * @param path the path the AF_UNIX socket.
     * @param mode
     * @param use_shm	
     */
    AfUnixCommunicator(std::string &path, mode_t mode);

    /** 
     * Creates a new AfUnixCommunicator for binding or connecting it to the
     * AF_UNIX socket specified from path.
     * 
     * @param path the path the AF_UNIX socket.
     * @param mode
     * @param use_shm	
     */
    AfUnixCommunicator(const char * path, mode_t mode);

    /**
     * Creates a new AfUnixCommunicator for binding or connecting it to the
     * AF_UNIX socket specified from path.
     * 
     * @param path the path the AF_UNIX socket.
     */
    AfUnixCommunicator(int fd);

    virtual ~AfUnixCommunicator();
    void Serve();
    const Communicator * const Accept() const;
    void Connect();
    size_t Read(char *buffer, size_t size);
    size_t Write(const char *buffer, size_t size);
    void Sync();
    void Close();

private:
    /** 
     * Initializes the input and output streams.
     */
    void InitializeStream();
    std::istream *mpInput; /**< the input stream for sending */
    std::ostream *mpOutput;	/**< the output stream for receiving data */
    std::string mPath;		/**< the path of the AF_UNIX socket */
    int mSocketFd;		/**< the file descriptor of the connected socket */
    __gnu_cxx::stdio_filebuf<char> *mpInputBuf;
    __gnu_cxx::stdio_filebuf<char> *mpOutputBuf;
    mode_t mMode;
};

#endif

#endif	/* _AFUNIXCOMMUNICATOR_H */

