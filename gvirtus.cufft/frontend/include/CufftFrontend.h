/* 
 * File:   CufftFrontend.h
 * Author: vincenzo
 *
 * Created on January 26, 2017, 10:01 AM
 */

#ifndef CUFFTFRONTEND_H
#define	CUFFTFRONTEND_H

//#define DEBUG

#include <map>
#include <set>
#include <stack>
#include <list>
#include <iostream>

#include <cufft.h>
#include <cuda_runtime_api.h>

#include "Frontend.h"
#include "Cufft.h"

using namespace std;

typedef struct __configureFunction{
      funcs __f;
      Buffer* buffer;
  } configureFunction;

class CufftFrontend {
public:
    static inline void Execute(const char *routine, const Buffer *input_buffer = NULL) {
        #ifdef DEBUG
                if (string(routine) != "cudaLaunch")
                    cerr << "Requesting " << routine << endl;
        #endif
                Frontend::GetFrontend()->Execute(routine,input_buffer);
    }
    /**
     * Prepares the Frontend for the execution. This method _must_ be called
     * before any requests of execution or any method for adding parameters for
     * the next execution.
     */
    
    static inline void Prepare() {
        Frontend::GetFrontend()->Prepare();
    }
    
    static inline Buffer *GetLaunchBuffer() {
        return Frontend::GetFrontend()->GetLaunchBuffer();
    }
    
    /**
     * Adds a scalar variabile as an input parameter for the next execution
     * request.
     *
     * @param var the variable to add as a parameter.
     */
    template <class T> static inline void AddVariableForArguments(T var) {
        Frontend::GetFrontend()->GetInputBuffer()->Add(var);
    }
    
    /**
     * Adds a string (array of char(s)) as an input parameter for the next
     * execution request.
     *
     * @param s the string to add as a parameter.
     */
    static inline void AddStringForArguments(const char *s) {
        Frontend::GetFrontend()->GetInputBuffer()->AddString(s);
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
    template <class T>static inline void AddHostPointerForArguments(T *ptr, size_t n = 1) {
        Frontend::GetFrontend()->GetInputBuffer()->Add(ptr, n);
    }

    /**
     * Adds a device pointer as an input parameter for the next execution
     * request.
     *
     * @param ptr the pointer to add as a parameter.
     */
    static inline void AddDevicePointerForArguments(const void *ptr) {
        Frontend::GetFrontend()->GetInputBuffer()->Add((pointer_t) ptr);
    }

    /**
     * Adds a symbol, a named variable, as an input parameter for the next
     * execution request.
     *
     * @param symbol the symbol to add as a parameter.
     */
    static inline void AddSymbolForArguments(const char *symbol) {
        /* TODO: implement AddSymbolForArguments
         * AddStringForArguments(CudaUtil::MarshalHostPointer((void *) symbol));
         * AddStringForArguments(symbol);
         * */
    }
    
    static inline cudaError_t GetExitCode() {
        return (cudaError_t) Frontend::GetFrontend()->GetExitCode();
    }

    static inline bool Success() {
        return Frontend::GetFrontend()->Success(cudaSuccess);
    }

    template <class T> static inline T GetOutputVariable() {
        return Frontend::GetFrontend()->GetOutputBuffer()->Get<T> ();
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
    template <class T>static inline T * GetOutputHostPointer(size_t n = 1) {
        return Frontend::GetFrontend()->GetOutputBuffer()->Assign<T> (n);
    }

    /**
     * Retrives a device pointer from the output parameters of the last
     * execution request.
     *
     * @return the pointer to the device memory.
     */
    static inline void * GetOutputDevicePointer() {
        return (void *) Frontend::GetFrontend()->GetOutputBuffer()->Get<pointer_t>();
    }

    /**
     * Retrives a string, array of chars, from the output parameters of the last
     * execution request.
     *
     * @return the string from the output parameters.
     */
    static inline char * GetOutputString() {
        return Frontend::GetFrontend()->GetOutputBuffer()->AssignString();
    }
    
    static inline void addMappedPointer(void* device, mappedPointer host) {
        mappedPointers->insert(make_pair(device, host)); 
        
    };
    
    static void addtoManage(void* manage) {
        pid_t tid = syscall(SYS_gettid);
        stack<void*> * toHandle;
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
        stack<void*> * toHandle;
        if (toManage->find(tid) != toManage->end()) {
            toHandle = toManage->find(tid)->second;
            while (!toHandle->empty()) {
                void* p = toHandle->top();
                toHandle->pop();
                mappedPointer mP = getMappedPointer(p);
#ifdef DEBUG
                cerr << "copying " << mP.size << ": " << std::hex << mP.pointer << " to "
                        << std::hex << p << endl;
#endif
                cudaMemcpy(p, mP.pointer, mP.size, cudaMemcpyDeviceToHost);
            }
        }
    }
    
    static void configure() {
        pid_t tid = syscall(SYS_gettid);
        stack<void*> * toHandle;
        if (toManage->find(tid) != toManage->end()) {
            toHandle = toManage->find(tid)->second;
            while (!toHandle->empty()) {
                void* p = toHandle->top();
                toHandle->pop();
                mappedPointer mP = getMappedPointer(p);
#ifdef DEBUG
                cerr << "copying " << mP.size << ": " << std::hex << mP.pointer << " to "
                        << std::hex << p << endl;
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

    static inline mappedPointer getMappedPointer(void* device) {
        return mappedPointers->find(device)->second;
    };
    
    static inline void removeMappedPointer(void* device) {
        mappedPointers->erase(device);
    };
    
    static inline void addConfigureElement() {
        
    }

    CufftFrontend();
    virtual ~CufftFrontend();

private:
    
    static map <const void*, mappedPointer>* mappedPointers;
    static set <const void*>* devicePointers;
    static map <pthread_t, stack<void*> *>* toManage;
    static list <configureFunction>* setup;
    Buffer * mpInputBuffer;
    bool configured;
};

#endif	/* CUFFTFRONTEND_H */

