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
 * @file   Frontend.cpp
 * @author Giuseppe Coviello <giuseppe.coviello@uniparthenope.it>
 * @date   Wed Sep 30 12:57:11 2009
 *
 * @brief
 *
 *
 */

#include "Frontend.h"

#include <iostream>
#include <pthread.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <unistd.h>

#include "communicator/CommunicatorFactory.h"
#include "communicator/EndpointFactory.h"
#include "nlohmann/json.hpp"

using namespace std;

static Frontend msFrontend;
map<pthread_t, Frontend *> *Frontend::mpFrontends = NULL;
static bool initialized = false;

/**
 *
 */
void
Frontend::Init(gvirtus::Communicator *c) {
#if 1
  const char *config_file;
  pid_t tid = syscall(SYS_gettid);

  std::string config_path;
#ifdef _CONFIG_FILE_JSON
  config_path = _CONFIG_FILE_JSON;
#else
  config_path = "properties.json";
#endif

  std::unique_ptr<char> default_endpoint;

  if (mpFrontends->find(tid) == mpFrontends->end()) { // no frontend found
    Frontend *f = new Frontend();
    mpFrontends->insert(make_pair(tid, f));
  }

  auto end = gvirtus::EndpointFactory::get_endpoint(config_path);
  mpFrontends->find(tid)->second->_communicator = gvirtus::CommunicatorFactory::get_communicator(end);
#else
  mpCommunicator = c;
#endif

  mpFrontends->find(tid)->second->_communicator->obj_ptr()->Connect();

  mpFrontends->find(tid)->second->mpInputBuffer = new Buffer();
  mpFrontends->find(tid)->second->mpOutputBuffer = new Buffer();
  mpFrontends->find(tid)->second->mpLaunchBuffer = new Buffer();
  mpFrontends->find(tid)->second->mExitCode = -1;
  mpFrontends->find(tid)->second->mpInitialized = true;
}

Frontend::~Frontend() {

  // cout<< "distruttore "<<endl;
  if (mpFrontends != NULL) {

    pid_t tid = syscall(SYS_gettid); // getting frontend's tid

    map<pthread_t, Frontend *>::iterator it;
    for (it = mpFrontends->begin(); it != mpFrontends->end(); it++) {
      cout << "~Frontend: " << it->second << endl;
      mpFrontends->erase(it);
      // delete it->second;
    }

    // delete mpFrontends->find(tid);
  } else {
    //        mpCommunicator->Close();
    //        delete mpCommunicator;
    delete mpFrontends;
  }
  //    mpCommunicator->Close();
  //    delete mpCommunicator;
}

Frontend *
Frontend::GetFrontend(gvirtus::Communicator *c) {
  if (mpFrontends == NULL)
    mpFrontends = new map<pthread_t, Frontend *>();

  pid_t tid = syscall(SYS_gettid); // getting frontend's tid
  // cout << "tid Get Frontend: "<< tid<< endl;
  if (mpFrontends->find(tid) != mpFrontends->end())
    return mpFrontends->find(tid)->second;
  else {
    Frontend *f = new Frontend();
    // if (!f->initialized()) {
    try {
      f->Init(c);
      mpFrontends->insert(make_pair(tid, f));
    } catch (const char *e) {
      cerr << "Error: cannot create Frontend ('" << e << "')" << endl;
    }
    //}
    return f;
  }
  /*if (!initialized) {
      try {
          msFrontend.Init(c);
      } catch (const char *e) {
          cerr << "Error: cannot create Frontend ('" << e << "')" << endl;
      }
  }
  return &msFrontend;*/
}

void
Frontend::Execute(const char *routine, const Buffer *input_buffer) {
  if (input_buffer == NULL)
    input_buffer = mpInputBuffer;

  pid_t tid = syscall(SYS_gettid);
  if (mpFrontends->find(tid) != mpFrontends->end()) {
    /* sending job */
    Frontend *frontend = new Frontend();
    frontend = mpFrontends->find(tid)->second;
    frontend->_communicator->obj_ptr()->Write(routine, strlen(routine) + 1);
    input_buffer->Dump(frontend->_communicator->obj_ptr().get());
    frontend->_communicator->obj_ptr()->Sync();

    // std::istream &in = frontend->mpCommunicator->GetInputStream();

    frontend->mpOutputBuffer->Reset();
    frontend->_communicator->obj_ptr()->Read((char *)&frontend->mExitCode, sizeof(int));
    size_t out_buffer_size;
    frontend->_communicator->obj_ptr()->Read((char *)&out_buffer_size, sizeof(size_t));
    if (out_buffer_size > 0)
      frontend->mpOutputBuffer->Read<char>(frontend->_communicator->obj_ptr().get(), out_buffer_size);
  } else {
    /* error */
    cerr << " ERROR - can't send any job request " << endl;
  }

  /* sending job */
  /*mpCommunicator->Write(routine, strlen(routine) + 1);
  input_buffer->Dump(mpCommunicator);
  mpCommunicator->Sync();

  // receiving output
  //std::istream &in = mpCommunicator->GetInputStream();

  mpOutputBuffer->Reset();

  mpCommunicator->Read((char *) & mExitCode, sizeof (int));
  size_t out_buffer_size;
  mpCommunicator->Read((char *) & out_buffer_size, sizeof (size_t));
  if (out_buffer_size > 0)
      mpOutputBuffer->Read<char>(mpCommunicator, out_buffer_size);*/
}

void
Frontend::Prepare() {
  // mpInputBuffer->Reset();
  pid_t tid = syscall(SYS_gettid);
  if (this->mpFrontends->find(tid) != mpFrontends->end())
    mpFrontends->find(tid)->second->mpInputBuffer->Reset();
}
