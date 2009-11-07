/* 
 * File:   Frontend.h
 * Author: cjg
 *
 * Created on 30 settembre 2009, 12.57
 */

#ifndef _FRONTEND_H
#define	_FRONTEND_H

#include <builtin_types.h>
#include "Communicator.h"
#include "Result.h"
#include "CudaUtil.h"

class Frontend {
public:
    virtual ~Frontend();
    static Frontend * GetFrontend();
    void Execute(const char *routine, const Buffer *input_buffer = NULL);
    void Prepare();

    template <class T>void AddVariableForArguments(T var) {
        mpInputBuffer->Add(var);
    }

    void AddStringForArguments(const char *s) {
        mpInputBuffer->AddString(s);
    }

    template <class T>void AddHostPointerForArguments(T *ptr, size_t n = 1) {
        mpInputBuffer->Add(ptr, n);
    }

    void AddDevicePointerForArguments(const void *ptr) {
        char *tmp = CudaUtil::MarshalDevicePointer(ptr);
        mpInputBuffer->Add(tmp, CudaUtil::MarshaledDevicePointerSize);
        delete[] tmp;
    }

    /*
    void Execute(const char *routine) {
        mpResult = Execute(routine, mpInputBuffer);
        mExitCode = mpResult->GetExitCode();
    }
    */
    
    bool Success() {
        return mExitCode == cudaSuccess;
    }

    template <class T>T GetOutputVariable() {
        return const_cast<Buffer *> (mpResult->GetOutputBufffer())->Get<T > ();
    }

    template <class T>T * GetOutputHostPointer(size_t n = 1) {
        return const_cast<Buffer *> (mpResult->GetOutputBufffer())->Assign<T > (n);
    }

    char * GetOutputString() {
        return const_cast<Buffer *> (mpResult->GetOutputBufffer())->AssignString();
    }

    cudaError_t GetExitCode() {
        return mExitCode;
    }
private:
    Frontend();
    Communicator *mpCommunicator;
    Buffer * mpInputBuffer;
    Result * mpResult;
    cudaError_t mExitCode;
    static Frontend *mspFrontend;
};

#endif	/* _FRONTEND_H */

