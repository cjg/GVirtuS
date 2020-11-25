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

#include <gvirtus/communicators/CommunicatorFactory.h>
#include <gvirtus/communicators/EndpointFactory.h>
#include <gvirtus/frontend/Frontend.h>

#include <pthread.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <unistd.h>
#include <iostream>

#include <chrono>

#include <stdlib.h> /* getenv */
#include "log4cplus/configurator.h"
#include "log4cplus/logger.h"
#include "log4cplus/loggingmacros.h"

using namespace std;

using gvirtus::communicators::Buffer;
using gvirtus::communicators::Communicator;
using gvirtus::communicators::CommunicatorFactory;
using gvirtus::communicators::EndpointFactory;
using gvirtus::frontend::Frontend;

using std::chrono::steady_clock;

static Frontend msFrontend;
map<pthread_t, Frontend *> *Frontend::mpFrontends = NULL;
static bool initialized = false;

log4cplus::Logger logger;

std::string getEnvVar(std::string const &key) {
  char *val = getenv(key.c_str());
  return val == NULL ? std::string("") : std::string(val);
}

void Frontend::Init(Communicator *c) {
  log4cplus::BasicConfigurator config;
  config.configure();
  logger = log4cplus::Logger::getInstance(LOG4CPLUS_TEXT("GVirtuS Frontend"));

  // Set the logging level
  log4cplus::LogLevel logLevel = log4cplus::INFO_LOG_LEVEL;
  std::string logLevelString = getEnvVar("GVIRTUS_LOGLEVEL");
  if (logLevelString != "") {
    logLevel = std::stoi(logLevelString);
  }

  logger.setLogLevel(logLevel);
  
  pid_t tid = syscall(SYS_gettid);

  // Get the GVIRTUS_CONFIG environment varibale
  std::string config_path = getEnvVar("GVIRTUS_CONFIG");

  // Check if the configuration file is defined
  if (config_path == "" ) {

    // Check if the configuration file is in the GVIRTUS_HOME directory
    config_path = getEnvVar("GVIRTUS_HOME")+"/etc/properties.json";
    if (config_path == "") {

      // Finally consider the current directory
      config_path = "./properties.json";
    }
  }

  std::unique_ptr<char> default_endpoint;

  if (mpFrontends->find(tid) == mpFrontends->end()) {  // no frontend found
    Frontend *f = new Frontend();
    mpFrontends->insert(make_pair(tid, f));
  }

  LOG4CPLUS_INFO(logger, "🛈  - GVirtuS frontend version "+config_path);

  auto end = EndpointFactory::get_endpoint(config_path);

  mpFrontends->find(tid)->second->_communicator =
      CommunicatorFactory::get_communicator(end);
  mpFrontends->find(tid)->second->_communicator->obj_ptr()->Connect();


  mpFrontends->find(tid)->second->mpInputBuffer = std::make_shared<Buffer>();
  mpFrontends->find(tid)->second->mpOutputBuffer = std::make_shared<Buffer>();
  mpFrontends->find(tid)->second->mpLaunchBuffer = std::make_shared<Buffer>();
  mpFrontends->find(tid)->second->mExitCode = -1;
  mpFrontends->find(tid)->second->mpInitialized = true;
}

Frontend::~Frontend() {
  if (mpFrontends != NULL) {
    pid_t tid = syscall(SYS_gettid);  // getting frontend's tid
    auto env = getenv("GVIRTUS_DUMP_STATS");
    auto dump_stats = env != nullptr && (strcasecmp(env, "on") == 0 || strcasecmp(env, "true") == 0 ||
        strcmp(env, "1") == 0);
    map<pthread_t, Frontend *>::iterator it;
    for (it = mpFrontends->begin(); it != mpFrontends->end(); it++) {
      if (dump_stats) {
        std::cerr << "[GVIRTUS_STATS] Executed " << it->second->mRoutinesExecuted << " routine(s) in "
                  << it->second->mRoutineExecutionTime << " second(s)\n"
                  << "[GVIRTUS_STATS] Sent " << it->second->mDataSent / (1024 * 1024.0) << " Mb(s) in " << it->second->mSendingTime
                  << " second(s)\n"
                  << "[GVIRTUS_STATS] Received " << it->second->mDataReceived / (1024 * 1024.0) << " Mb(s) in "
                  << it->second->mReceivingTime
                  << " second(s)\n";
      }
      mpFrontends->erase(it);
    }
  } else
    delete mpFrontends;
}

Frontend *Frontend::GetFrontend(Communicator *c) {
  if (mpFrontends == NULL) mpFrontends = new map<pthread_t, Frontend *>();

  pid_t tid = syscall(SYS_gettid);  // getting frontend's tid
  if (mpFrontends->find(tid) != mpFrontends->end())
    return mpFrontends->find(tid)->second;
  else {
    Frontend *f = new Frontend();
    try {
      f->Init(c);
      mpFrontends->insert(make_pair(tid, f));
    } catch (const char *e) {
      cerr << "Error: cannot create Frontend ('" << e << "')" << endl;
    }

    return f;
  }
}

void Frontend::Execute(const char *routine, const Buffer *input_buffer) {
  if (input_buffer == nullptr) input_buffer = mpInputBuffer.get();

  pid_t tid = syscall(SYS_gettid);
  if (mpFrontends->find(tid) != mpFrontends->end()) {
    /* sending job */
    auto frontend = mpFrontends->find(tid)->second;
    frontend->mRoutinesExecuted++;
    auto start = steady_clock::now();
    frontend->_communicator->obj_ptr()->Write(routine, strlen(routine) + 1);
    frontend->mDataSent += input_buffer->GetBufferSize();
    input_buffer->Dump(frontend->_communicator->obj_ptr().get());
    frontend->_communicator->obj_ptr()->Sync();
    frontend->mSendingTime += std::chrono::duration_cast<std::chrono::milliseconds>(steady_clock::now() - start)
        .count() / 1000.0;
    frontend->mpOutputBuffer->Reset();

    frontend->_communicator->obj_ptr()->Read((char *) &frontend->mExitCode,
                                             sizeof(int));
    double time_taken;
    frontend->_communicator->obj_ptr()->Read(reinterpret_cast<char *>(&time_taken), sizeof(time_taken));
    frontend->mRoutineExecutionTime += time_taken;

    start = steady_clock::now();
    size_t out_buffer_size;
    frontend->_communicator->obj_ptr()->Read((char *) &out_buffer_size,
                                             sizeof(size_t));
    frontend->mDataReceived += out_buffer_size;
    if (out_buffer_size > 0)
      frontend->mpOutputBuffer->Read<char>(
          frontend->_communicator->obj_ptr().get(), out_buffer_size);
    frontend->mReceivingTime += std::chrono::duration_cast<std::chrono::milliseconds>(steady_clock::now() - start)
        .count() / 1000.0;
  } else {
    /* error */
    cerr << " ERROR - can't send any job request " << endl;
  }
}

void Frontend::Prepare() {
  pid_t tid = syscall(SYS_gettid);
  if (this->mpFrontends->find(tid) != mpFrontends->end())
    mpFrontends->find(tid)->second->mpInputBuffer->Reset();
}
