#include "Frontend.h"
#include "CudaRt.h"

using namespace std;

extern cudaError_t cudaEventCreate(cudaEvent_t *event) {
    Frontend *f = Frontend::GetFrontend();
    f->AddHostPointerForArguments(event);
    f->Execute("cudaEventCreate");
    if(f->Success())
        *event = *(f->GetOutputHostPointer<cudaEvent_t>());
    return f->GetExitCode();
}

extern cudaError_t cudaEventCreateWithFlags(cudaEvent_t *event, int flags) {
    Frontend *f = Frontend::GetFrontend();
    f->AddHostPointerForArguments(event);
    f->AddVariableForArguments(flags);
    f->Execute("cudaEventCreateWithFlags");
    if(f->Success())
        *event = *(f->GetOutputHostPointer<cudaEvent_t>());
    return f->GetExitCode();
}

extern cudaError_t cudaEventDestroy(cudaEvent_t event) {
    Frontend *f = Frontend::GetFrontend();
    f->AddVariableForArguments(event);
    f->Execute("cudaEventDestroy");
    return f->GetExitCode();
}

extern cudaError_t cudaEventElapsedTime(float *ms, cudaEvent_t start, cudaEvent_t end) {
    Frontend *f = Frontend::GetFrontend();
    f->AddHostPointerForArguments(ms);
    f->AddVariableForArguments(start);
    f->AddVariableForArguments(end);
    f->Execute("cudaEventElapsedTime");
    if(f->Success())
        *ms = *(f->GetOutputHostPointer<float>());
    return f->GetExitCode();
}

extern cudaError_t cudaEventQuery(cudaEvent_t event) {
    Frontend *f = Frontend::GetFrontend();
    f->AddVariableForArguments(event);
    f->Execute("cudaEventQuery");
    return f->GetExitCode();
}

extern cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream) {
    Frontend *f = Frontend::GetFrontend();
    f->AddVariableForArguments(event);
    f->AddVariableForArguments(stream);
    f->Execute("cudaEventRecord");
    return f->GetExitCode();
}

extern cudaError_t cudaEventSynchronize(cudaEvent_t event) {
    Frontend *f = Frontend::GetFrontend();
    f->AddVariableForArguments(event);
    f->Execute("cudaEventSynchronize");
    return f->GetExitCode();
}
