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
 * @file   Process.cpp
 * @author Giuseppe Coviello <giuseppe.coviello@uniparthenope.it>
 * @date   Wed Sep 30 10:45:40 2009
 *
 * @brief
 *
 *
 */

#include <iostream>
#include <cstdio>
#include <string>
#include "cuda_runtime_api.h"
#include "Process.h"

using namespace std;

Process::Process(const Communicator *communicator)
: Subprocess(), Observable(),
mpInput(const_cast<Communicator *> (communicator)->GetInputStream()),
mpOutput(const_cast<Communicator *> (communicator)->GetOutputStream()) {
    mpCommunicator = const_cast<Communicator *> (communicator);
    mpHandler = new CudaRtHandler();
}

Process::~Process() {
    cout << "[Process " << GetPid() << "]: Destroyed." << endl;
}

void Process::Setup() {

}

void Process::Execute(void * arg) {
    cout << "[Process " << GetPid() << "]: Started." << endl;

    string routine;
    Buffer * input_buffer = new Buffer();
    while (getline(mpInput, routine)) {
        input_buffer->Reset(mpInput);
        Result * result;
        try {
            result = mpHandler->Execute(routine, input_buffer);
        } catch (string e) {
            cout << "[Process " << GetPid() << "]: Exception " << e
                    << "." << endl;
            result = new Result(cudaErrorUnknown, new Buffer());
        }
        result->Dump(mpOutput);
        if (result->GetExitCode() != cudaSuccess) {
            cout << "[Process " << GetPid() << "]: Requested '" << routine
                    << "' routine." << endl;
            cout << "[Process " << GetPid() << "]: Exit Code '"
                    << cudaGetErrorString(result->GetExitCode()) << "'." << endl;
        }
        delete result;
    }
    delete input_buffer;
    Notify("process-ended");
    delete this;
}

