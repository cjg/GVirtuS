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
#include <vector>

class Frontend {
public:
    virtual ~Frontend();
    static Frontend * GetFrontend(bool register_var = false);
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
        mpInputBuffer->Add((uint64_t) ptr);
    }

    void AddSymbolForArguments(const char *symbol) {
        AddStringForArguments(CudaUtil::MarshalHostPointer((void *) symbol));
        AddStringForArguments(symbol);
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

    void * GetOutputDevicePointer() {
        return (void *) mpOutputBuffer->Get<uint64_t>();
    }

    char * GetOutputString() {
        return mpOutputBuffer->AssignString();
    }

    cudaError_t GetExitCode() {
        return mExitCode;
    }

    void AddVar(CudaUtil::CudaVar * var);

    inline Buffer * GetLaunchBuffer() {
        return mpLaunchBuffer;
    }

    Communicator * GetCommunicator() {
        return mpCommunicator;
    }
    
private:
    Frontend();
    Communicator *mpCommunicator;
    Buffer * mpInputBuffer;
    Buffer * mpOutputBuffer;
    Buffer * mpLaunchBuffer;
    cudaError_t mExitCode;
    static Frontend *mspFrontend;
    std::vector<CudaUtil::CudaVar *> * mpVar;
    bool mAddingVar; 
};

#endif	/* _FRONTEND_H */

