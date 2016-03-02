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
 * @file   VirtioCommunicator.h
 * @author Abhijeet Dev <abhijeet@abhijeet-dev.net>
 * @date   Tue Mar 6 15:30:21 2012
 * 
 * @brief  
 * 
 */

#ifndef VIRTIOCOMMUNICATOR_H
#define VIRTIOCOMMUNICATOR_H

#include "Communicator.h"

class VirtioCommunicator : public Communicator {
public:
    VirtioCommunicator(const std::string &communicator);
    void Serve();
    const Communicator * const Accept() const;
    void Connect();
    size_t Read(char *buffer, size_t size);
    size_t Write(const char *buffer, size_t size);
    void Sync();
    void Close();
private:
    int mFd;
    std::string mDevice;
};

#endif	/* VIRTIOCOMMUNICATOR_H */

