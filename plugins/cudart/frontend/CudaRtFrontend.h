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

#ifndef CUDARTFRONTEND_H
#define CUDARTFRONTEND_H

//#define DEBUG

#include <list>
#include <map>
#include <set>
#include <stack>

#include <CudaUtil.h>
#include <cuda_runtime_api.h>
#include <CudaRt_internal.h>
#include <gvirtus/frontend/Frontend.h>

#include "CudaRt.h"

using namespace std;

typedef struct __configureFunction {
  gvirtus::common::funcs __f;
  Buffer* buffer;
} configureFunction;

class CudaRtFrontend {
 public:
  static inline void Execute(const char* routine,
                             const Buffer* input_buffer = NULL) {
#ifdef DEBUG
    if (string(routine) != "cudaLaunch")
      cerr << "Requesting " << routine << endl;
#endif
    gvirtus::frontend::Frontend::GetFrontend()->Execute(routine, input_buffer);
  }

  /**
   * Prepares the Frontend for the execution. This method _must_ be called
   * before any requests of execution or any method for adding parameters for
   * the next execution.
   */
  static inline void Prepare() {
    gvirtus::frontend::Frontend::GetFrontend()->Prepare();
  }

  static inline Buffer* GetLaunchBuffer() {
    return gvirtus::frontend::Frontend::GetFrontend()->GetLaunchBuffer();
  }

  /**
   * Adds a scalar variabile as an input parameter for the next execution
   * request.
   *
   * @param var the variable to add as a parameter.
   */
  template <class T>
  static inline void AddVariableForArguments(T var) {
    gvirtus::frontend::Frontend::GetFrontend()->GetInputBuffer()->Add(var);
  }

  /**
   * Adds a string (array of char(s)) as an input parameter for the next
   * execution request.
   *
   * @param s the string to add as a parameter.
   */
  static inline void AddStringForArguments(const char* s) {
    gvirtus::frontend::Frontend::GetFrontend()->GetInputBuffer()->AddString(s);
  }

  /**
   * Adds, marshalling it, an host pointer as an input parameter for the next
   * execution request.
   * The optional parameter n is useful when adding an array: with n is
   * possible to specify the length of the array in terms of elements.
   *
   * @param ptr the pointer to add as a parameter.
   * @param n the length of the array, if ptr is an array.
   */
  template <class T>
  static inline void AddHostPointerForArguments(T* ptr, size_t n = 1) {
    gvirtus::frontend::Frontend::GetFrontend()->GetInputBuffer()->Add(ptr, n);
  }

  /**
   * Adds a device pointer as an input parameter for the next execution
   * request.
   *
   * @param ptr the pointer to add as a parameter.
   */
  static inline void AddDevicePointerForArguments(const void* ptr) {
    gvirtus::frontend::Frontend::GetFrontend()->GetInputBuffer()->Add(
        (gvirtus::common::pointer_t)ptr);
  }

  /**
   * Adds a symbol, a named variable, as an input parameter for the next
   * execution request.
   *
   * @param symbol the symbol to add as a parameter.
   */
  static inline void AddSymbolForArguments(const char* symbol) {
    AddStringForArguments(CudaUtil::MarshalHostPointer((void*)symbol));
    AddStringForArguments(symbol);
  }

  static inline cudaError_t GetExitCode() {
    return (cudaError_t)gvirtus::frontend::Frontend::GetFrontend()
        ->GetExitCode();
  }

  static inline bool Success() {
    return gvirtus::frontend::Frontend::GetFrontend()->Success(cudaSuccess);
  }

  template <class T>
  static inline T GetOutputVariable() {
    return gvirtus::frontend::Frontend::GetFrontend()
        ->GetOutputBuffer()
        ->Get<T>();
  }

  /**
   * Retrieves an host pointer from the output parameters of the last execution
   * request.
   * The optional parameter n is useful when retrieving an array: with n is
   * possible to specify the length of the array in terms of elements.
   *
   * @param n the length of the array.
   *
   * @return the pointer from the output parameters.
   */
  template <class T>
  static inline T* GetOutputHostPointer(size_t n = 1) {
    return gvirtus::frontend::Frontend::GetFrontend()
        ->GetOutputBuffer()
        ->Assign<T>(n);
  }

  /**
   * Retrives a device pointer from the output parameters of the last
   * execution request.
   *
   * @return the pointer to the device memory.
   */
  static inline void* GetOutputDevicePointer() {
    return (void*)gvirtus::frontend::Frontend::GetFrontend()
        ->GetOutputBuffer()
        ->Get<gvirtus::common::pointer_t>();
  }

  /**
   * Retrives a string, array of chars, from the output parameters of the last
   * execution request.
   *
   * @return the string from the output parameters.
   */
  static inline char* GetOutputString() {
    return gvirtus::frontend::Frontend::GetFrontend()
        ->GetOutputBuffer()
        ->AssignString();
  }

  static inline void addMappedPointer(void* device,
                                      gvirtus::common::mappedPointer host) {
    mappedPointers->insert(make_pair(device, host));
  }

  static void addtoManage(void* manage) {
    pid_t tid = syscall(SYS_gettid);
    stack<void*>* toHandle;
    if (toManage->find(tid) != toManage->end())
      toHandle = toManage->find(tid)->second;
    else {
      toHandle = new stack<void*>();
      toManage->insert(make_pair(tid, toHandle));
    }
    toHandle->push(manage);
#ifdef DEBUG
    cerr << "Added: " << std::hex << manage << endl;
#endif
  };

  static void manage() {
    pid_t tid = syscall(SYS_gettid);
    stack<void*>* toHandle;
    if (toManage->find(tid) != toManage->end()) {
      toHandle = toManage->find(tid)->second;
      while (!toHandle->empty()) {
        void* p = toHandle->top();
        toHandle->pop();
        gvirtus::common::mappedPointer mP = getMappedPointer(p);
#ifdef DEBUG
        cerr << "copying " << mP.size << ": " << std::hex << mP.pointer
             << " to " << std::hex << p << endl;
#endif
        cudaMemcpy(p, mP.pointer, mP.size, cudaMemcpyDeviceToHost);
      }
    }
  }

  static void configure() {
    pid_t tid = syscall(SYS_gettid);
    stack<void*>* toHandle;
    if (toManage->find(tid) != toManage->end()) {
      toHandle = toManage->find(tid)->second;
      while (!toHandle->empty()) {
        void* p = toHandle->top();
        toHandle->pop();
        gvirtus::common::mappedPointer mP = getMappedPointer(p);
#ifdef DEBUG
        cerr << "copying " << mP.size << ": " << std::hex << mP.pointer
             << " to " << std::hex << p << endl;
#endif
        cudaMemcpy(p, mP.pointer, mP.size, cudaMemcpyDeviceToHost);
      }
    }
  }

  static inline bool isMappedMemory(const void* p) {
    return (mappedPointers->find(p) == mappedPointers->end() ? false : true);
  }

  static inline void addDevicePointer(void* device) {
#ifdef DEBUG
    cerr << endl << "Added device pointer: " << hex << device << endl;
#endif
    devicePointers->insert(device);
  };

  static inline void removeDevicePointer(void* device) {
    devicePointers->erase(device);
  };

  static inline bool isDevicePointer(const void* p) {
#ifdef DEBUG
    cerr << endl << "Looking for device pointer: " << hex << p << endl;
#endif
    return (devicePointers->find(p) == devicePointers->end() ? false : true);
  }

  static inline gvirtus::common::mappedPointer getMappedPointer(void* device) {
    return mappedPointers->find(device)->second;
  };

  static inline void removeMappedPointer(void* device) {
    mappedPointers->erase(device);
  };

  static inline void addConfigureElement() {}

    static inline void addDeviceFunc2InfoFunc(std::string deviceFunc, NvInfoFunction infoFunction) {
        mapDeviceFunc2InfoFunc->insert(make_pair(deviceFunc, infoFunction));
    }

    static inline NvInfoFunction getInfoFunc(std::string deviceFunc) {
        return mapDeviceFunc2InfoFunc->find(deviceFunc)->second;
    };

    static inline void addHost2DeviceFunc(void* hostFunc, std::string deviceFunc) {
        mapHost2DeviceFunc->insert(make_pair(hostFunc, deviceFunc));
    }

    static inline std::string getDeviceFunc(void *hostFunc) {
        return mapHost2DeviceFunc->find(hostFunc)->second;
    };

  CudaRtFrontend();

    static void hexdump(void *ptr, int buflen) {
        unsigned char *buf = (unsigned char*)ptr;
        int i, j;
        for (i=0; i<buflen; i+=16) {
            printf("%06x: ", i);
            for (j=0; j<16; j++)
                if (i+j < buflen)
                    printf("%02x ", buf[i+j]);
                else
                    printf("   ");
            printf(" ");
            for (j=0; j<16; j++)
                if (i+j < buflen)
                    printf("%c", isprint(buf[i+j]) ? buf[i+j] : '.');
            printf("\n");
        }
    }

 private:
  static map<const void*, gvirtus::common::mappedPointer>* mappedPointers;
  static set<const void*>* devicePointers;
  static map<pthread_t, stack<void*>*>* toManage;
  static list<configureFunction>* setup;
  Buffer* mpInputBuffer;
  bool configured;
  static map<std::string, NvInfoFunction>* mapDeviceFunc2InfoFunc;
  static map<const void *,std::string>* mapHost2DeviceFunc;
};

#endif /* CUDARTFRONTEND_H */
