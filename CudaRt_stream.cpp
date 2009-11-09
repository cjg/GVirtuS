#include <cstring>
#include "Frontend.h"
#include "CudaUtil.h"
#include "CudaRt.h"

using namespace std;

extern cudaError_t cudaStreamCreate(cudaStream_t *pStream) {
    Frontend *f = Frontend::GetFrontend();
    f->AddHostPointerForArguments(pStream);
    f->Execute("cudaStreamCreate");
    if(f->Success())
        *pStream = *(f->GetOutputHostPointer<cudaStream_t>());
    return f->GetExitCode();
}

extern cudaError_t cudaStreamDestroy(cudaStream_t stream) {
    Frontend *f = Frontend::GetFrontend();
    f->AddVariableForArguments(stream);
    f->Execute("cudaStreamDestroy");
    return f->GetExitCode();
}

extern cudaError_t cudaStreamQuery(cudaStream_t stream) {
    Frontend *f = Frontend::GetFrontend();
    f->AddVariableForArguments(stream);
    f->Execute("cudaStreamQuery");
    return f->GetExitCode();
}

extern cudaError_t cudaStreamSynchronize(cudaStream_t stream) {
    Frontend *f = Frontend::GetFrontend();
    f->AddVariableForArguments(stream);
    f->Execute("cudaStreamSynchronize");
    return f->GetExitCode();
}