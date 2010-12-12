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
 * @file   ShmCommunicator.h
 * @author Giuseppe Coviello <giuseppe.coviello@uniparthenope.it>
 * @date   Tue Nov 16 9:52:26 2010
 *
 * @brief
 *
 *
 */

#ifndef SHMCOMMUNICATOR_H
#define	SHMCOMMUNICATOR_H

#include <semaphore.h>

#include "Communicator.h"

class ShmCommunicator : public Communicator {
public:
    ShmCommunicator(const std::string & communicator);
    ShmCommunicator();
    virtual ~ShmCommunicator();
    void Serve();
    const Communicator * const Accept() const;
    void Connect();
    size_t Read(char *buffer, size_t size);
    size_t Write(const char *buffer, size_t size);
    void Sync();
    void Close();
private:
    ShmCommunicator(const char *name);
    size_t ReadPacket(char *buffer);
    int mSocketFd;
    int mFd;
    char *mpShm;
    size_t mIOSize;
    int *mpClosed;
    sem_t *mpInEmpty;
    sem_t *mpInFull;
    size_t *mpInSize;
    char *mpIn;
    sem_t *mpOutEmpty;
    sem_t *mpOutFull;
    size_t *mpOutSize;
    char *mpOut;
    char *mpLocalIn;
    size_t mLocalInSize;
    size_t mLocalInOffset;
    char *mpLocalOut;
    size_t mLocalOutSize;
    size_t mLocalOutOffset;
};

#endif	/* SHMCOMMUNICATOR_H */

