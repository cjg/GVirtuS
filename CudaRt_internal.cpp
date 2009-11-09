#include "Frontend.h"
#include "CudaRt.h"

/*
 Routines not found in the cuda's header files.
 KEEP THEM WITH CARE
 */

void** __cudaRegisterFatBinary(void *fatCubin) {
    /* Fake host pointer */
    void ** handler = (void **) & fatCubin;
    Buffer * input_buffer = new Buffer();
    input_buffer->AddString(CudaUtil::MarshalHostPointer(handler));
    input_buffer = CudaUtil::MarshalFatCudaBinary(
            (__cudaFatCudaBinary *) fatCubin, input_buffer);

    Frontend *f = Frontend::GetFrontend();
    f->Execute("cudaRegisterFatBinary", input_buffer);
    if (f->Success())
        return handler;
    return NULL;
}

void __cudaUnregisterFatBinary(void **fatCubinHandle) {
    /* FIXME: implement */
    cerr << "*** Error: __cudaUnregisterFatBinary() not yet implemented!"
            << endl;
}

void __cudaRegisterFunction(void **fatCubinHandle, const char *hostFun, 
        char *deviceFun, const char *deviceName, int thread_limit, uint3 *tid,
        uint3 *bid, dim3 *bDim, dim3 *gDim, int *wSize) {
    Frontend *f = Frontend::GetFrontend();
    f->AddStringForArguments(CudaUtil::MarshalHostPointer(fatCubinHandle));
    f->AddStringForArguments(hostFun);
    f->AddStringForArguments(deviceFun);
    f->AddStringForArguments(deviceName);
    f->AddVariableForArguments(thread_limit);
    f->AddHostPointerForArguments(tid);
    f->AddHostPointerForArguments(bid);
    f->AddHostPointerForArguments(bDim);
    f->AddHostPointerForArguments(gDim);
    f->AddHostPointerForArguments(wSize);

    f->Execute("cudaRegisterFunction");

    deviceFun = f->GetOutputString();
    tid = f->GetOutputHostPointer<uint3 > ();
    bid = f->GetOutputHostPointer<uint3 > ();
    bDim = f->GetOutputHostPointer<dim3 > ();
    gDim = f->GetOutputHostPointer<dim3 > ();
    wSize = f->GetOutputHostPointer<int>();
}

void __cudaRegisterVar(void **fatCubinHandle, char *hostVar, 
        char *deviceAddress, const char *deviceName, int ext, int size,
        int constant, int global) {
    // FIXME: implement
    cerr << "*** Error: __cudaRegisterVar() not yet implemented!" << endl;
}

void __cudaRegisterShared(void **fatCubinHandle, void **devicePtr) {
    // FIXME: implement
    cerr << "*** Error: __cudaRegisterShared() not yet implemented!" << endl;
}

void __cudaRegisterSharedVar(void **fatCubinHandle, void **devicePtr, 
        size_t size, size_t alignment, int storage) {
    // FIXME: implement
    cerr << "*** Error: __cudaRegisterSharedVar() not yet implemented!" << endl;
}

/* */

extern int __cudaSynchronizeThreads(void** x, void* y) {
    // FIXME: implement
    cerr << "*** Error: __cudaSynchronizeThreads() not yet implemented!"
            << endl;
    return 0;
}

extern void __cudaTextureFetch(const void *tex, void *index, int integer,
        void *val) {
    // FIXME: implement 
    cerr << "*** Error: __cudaTextureFetch() not yet implemented!" << endl;
}
