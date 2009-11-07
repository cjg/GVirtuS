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

    bool Success() {
        return mExitCode == cudaSuccess;
    }

    template <class T>T GetOutputVariable() {
        return mpOutputBuffer->Get<T> ();
    }

    template <class T>T * GetOutputHostPointer(size_t n = 1) {
        return mpOutputBuffer->Assign<T> (n);
    }

    char * GetOutputString() {
        return mpOutputBuffer->AssignString();
    }

    cudaError_t GetExitCode() {
        return mExitCode;
    }
private:
    Frontend();
    Communicator *mpCommunicator;
    Buffer * mpInputBuffer;
    Buffer * mpOutputBuffer;
    cudaError_t mExitCode;
    static Frontend *mspFrontend;
};

#endif	/* _FRONTEND_H */

