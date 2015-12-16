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
 * @file   Mutex.h
 * @author Giuseppe Coviello <giuseppe.coviello@uniparthenope.it>
 * @date   Thu Oct 1 11:54:04 2009
 * 
 * @brief  
 * 
 * 
 */

#ifndef _MUTEX_H
#define	_MUTEX_H

#include <pthread.h>

/**
 * Mutex can be used to implement the mutual exclusion on blocks of code.
 */
class Mutex {
public:
    Mutex();
    virtual ~Mutex();
    bool Lock();
    void Unlock();
private:
    pthread_mutex_t mMutex;
};

#define synchronized(mutex) for(bool __mutex_lock = (mutex).Lock(); __mutex_lock; __mutex_lock = false, (mutex).Unlock())

#endif	/* _MUTEX_H */

