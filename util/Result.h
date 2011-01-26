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
 * @file   Result.h
 * @author Giuseppe Coviello <giuseppe.coviello@uniparthenope.it>
 * @date   Sun Oct 18 13:23:56 2009
 * 
 * @brief  
 * 
 * 
 */

#ifndef _RESULT_H
#define	_RESULT_H

#include <iostream>

#include "Buffer.h"

/**
 * Result is used to store the results of a CUDA Runtime routine.
 */
class Result {
public:
    Result(int exit_code);
    Result(int exit_code, const Buffer * output_buffer);
    Result(const Result& orig);
    Result(std::istream & in);
    virtual ~Result();
    int GetExitCode();
    const Buffer * GetOutputBufffer() const;
    void Dump(Communicator * c);
private:
    int mExitCode;
    Buffer * mpOutputBuffer;
};

#endif	/* _RESULT_H */

