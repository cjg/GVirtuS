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
 * @file   Thread.cpp
 * @author Giuseppe Coviello <giuseppe.coviello@uniparthenope.it>
 * @date   Thu Oct 1 12:22:44 2009
 *
 * @brief
 *
 *
 */

#include "Thread.h"

Thread::Thread() {
}

Thread::~Thread() {
}

int Thread::Start(void * arg) {
    mpArg = arg;
    return pthread_create(&mThreadId, NULL, Thread::EntryPoint, this);
}

void Thread::Join() {
    pthread_join(mThreadId, NULL);
}

int Thread::Run(void * arg) {
    Setup();
    Execute(arg);
    return 0;
}

/*static */
void * Thread::EntryPoint(void * pthis) {
    ((Thread *) pthis)->Run(((Thread *) pthis)->mpArg);
    return NULL;
}

pthread_t Thread::GetThreadId() {
    return mThreadId;
}
