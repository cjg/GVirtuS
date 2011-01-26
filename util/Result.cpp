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
 * @file   Result.cpp
 * @author Giuseppe Coviello <giuseppe.coviello@uniparthenope.it>
 * @date   Sun Oct 18 13:23:56 2009
 *
 * @brief
 *
 *
 */

#include "Result.h"

Result::Result(int exit_code) {
    mExitCode = exit_code;
    mpOutputBuffer = NULL;
}

Result::Result(int exit_code, const Buffer* output_buffer) {
    mExitCode = exit_code;
    mpOutputBuffer = const_cast<Buffer *>(output_buffer);
}

Result::Result(const Result& orig) {
}

Result::Result(std::istream & in) {
    in.read((char *) &mExitCode, sizeof(int));
    mpOutputBuffer = new Buffer(in);
}

Result::~Result() {
    delete mpOutputBuffer;
}

int Result::GetExitCode() {
    return mExitCode;
}

const Buffer * Result::GetOutputBufffer() const {
    return mpOutputBuffer;
}

void Result::Dump(Communicator * c) {
    c->Write((char *) &mExitCode, sizeof(int));
    if(mpOutputBuffer != NULL)
        mpOutputBuffer->Dump(c);
    else {
        size_t size = 0;
        c->Write((char *) &size, sizeof(size_t));
        c->Sync();
    }
}
