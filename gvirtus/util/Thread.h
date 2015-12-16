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
 * @file   Thread.h
 * @author Giuseppe Coviello <giuseppe.coviello@uniparthenope.it>
 * @date   Thu Oct 1 12:22:44 2009
 * 
 * @brief  
 * 
 * 
 */

#ifndef _THREAD_H
#define	_THREAD_H

#include <pthread.h>

/**
 * Thread is an abstract class which permits to spawn thread(s) to classes that
 * expandes it.
 * Thread has two virtual methods that the class that uses it must implement:
 * - Setup() that is invoked before spawning the new thread.
 * - Run() that is the method that will be runned in the new thread.
 * To start a new thread running the Run() methods must be called the Start()
 * method.
 */
class Thread {
public:
    Thread();
    virtual ~Thread();
    int Start(void * arg);
    void Join();
protected:
    int Run(void * arg);
    static void * EntryPoint(void *);
    virtual void Setup() = 0;
    virtual void Execute(void *) = 0;
    pthread_t GetThreadId();
    void * Arg() const {return mpArg;}
    void Arg(void* a){mpArg = a;}
private:
    pthread_t mThreadId;
    void * mpArg;
};

#endif	/* _THREAD_H */

