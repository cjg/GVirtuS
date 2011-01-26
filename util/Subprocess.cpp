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
 * @file   Subprocess.cpp
 * @author Giuseppe Coviello <giuseppe.coviello@uniparthenope.it>
 * @date   Thu Jul 15 10:33:17 2010
 *
 * @brief
 *
 *
 */

#ifndef _WIN32

#include "Subprocess.h"

#include <sys/wait.h>
#include <stdlib.h>

#include <csignal>

Subprocess::Subprocess() {
    signal(SIGCHLD, SIG_IGN);
}

Subprocess::~Subprocess() {
}

int Subprocess::Start(void * arg) {
    mpArg = arg;
    if((mPid = fork()) < 0)
        throw "Can't instantiate Subprocess.";
    if(mPid == 0) {
        this->EntryPoint(this);
        exit(0);
    }
    return mPid;
}

void Subprocess::Wait() {
    int status;
    waitpid(mPid, &status, 0);
}

int Subprocess::Run(void * arg) {
    Setup();
    Execute(arg);
    return 0;
}

/*static */
void * Subprocess::EntryPoint(void * pthis) {
    ((Subprocess *) pthis)->Run(((Subprocess *) pthis)->mpArg);
    return NULL;
}

pid_t Subprocess::GetPid() {
    if(mPid == 0)
        mPid = getpid();
    return mPid;
}

#endif
