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
 * @file   Communicator.h
 * @author Giuseppe Coviello <giuseppe.coviello@uniparthenope.it>
 * @date   Wed Sep 30 11:56:44 2009
 * 
 * @brief  
 * 
 * 
 */

#ifndef _COMMUNICATOR_H
#define	_COMMUNICATOR_H

#include <iostream>

#include "ConfigFile.h"

/**
 * Communicator is an abstract class that implements a simple stream oriented
 * mechanism for communicating with two end points.
 * Communicator use a client/server approach, for having a Communicator server
 * the application must call Serve() and the Accept() for accepting the
 * connection by clients and communicating to them. 
 * The client has to use just the Connect() method.
 * For sending and receiving data through the communicator is possible the use
 * the input and output stream. Warning: _never_ try to communicate through the
 * streams of a server Communicator, for communicating with the client the
 * Communicator returned from the Accept() must be used.
 */
class Communicator {
public:
    /** 
     * Creates a new communicator. The real type of the communicator and his
     * parameters are obtained from the ConfigFile::Element @arg config.
     * 
     * @param config the ConfigFile::Element that stores the configuration.
     * 
     * @return a new Communicator.
     */
    static Communicator * Get(const std::string & communicator);

    virtual ~Communicator();

    /** 
     * Sets the communicator as a server.
     */
    virtual void Serve() = 0;

    /** 
     * Accepts a new connection. The call to the first Accept() must follow a
     * call to Serve().
     * 
     * @return a Communicator to the connected peer.
     */
    virtual const Communicator * const Accept() const = 0;

    /** 
     * Sets the communicator as a client and connects it to the end point
     * specified in the ConfigFile::Element used to build this Communicator.
     */
    virtual void Connect() = 0;

    virtual size_t Read(char *buffer, size_t size) = 0;
    virtual size_t Write(const char *buffer, size_t size) = 0;
    virtual void Sync() = 0;
    
    /** 
     * Closes the connection with the end point.
     */
    virtual void Close() = 0;
private:

};

#endif	/* _COMMUNICATOR_H */

