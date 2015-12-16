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
 * @file   Subprocess.h
 * @author Giuseppe Coviello <giuseppe.coviello@uniparthenope.it>
 * @date   Thu Jul 15 10:33:17 2010
 * 
 * @brief  
 * 
 * 
 */

#ifndef _SUBPROCESS_H
#define	_SUBPROCESS_H

#include <sys/types.h>
#include <unistd.h>

/**
 * Subprocess is an abstract class which permits to spawn process(s) to classes
 * that expandes it.
 * Subprocess has two virtual methods that the class that uses it must
 * implement:
 * - Setup() that is invoked before spawning the new process.
 * - Run() that is the method that will be runned in the new process. 
 * To start a new process running the Run() methods must be called the Start()
 * method.
 */
class Subprocess {
public:
    Subprocess();
    virtual ~Subprocess();
    int Start(void * arg);
    void Wait();
protected:
    int Run(void * arg);
    static void * EntryPoint(void *);
    virtual void Setup() = 0;
    virtual void Execute(void *) = 0;
    pid_t GetPid();
    void * Arg() const {return mpArg;}
    void Arg(void* a){mpArg = a;}
private:
    pid_t mPid;
    void * mpArg;
};

#endif	/* _SUBPROCESS_H */

