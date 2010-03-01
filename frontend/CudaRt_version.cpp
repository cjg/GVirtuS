#include "Frontend.h"
#include "CudaRt.h"

using namespace std;

extern cudaError_t cudaDriverGetVersion(int *driverVersion) {
    Frontend *f = Frontend::GetFrontend();
    f->AddHostPointerForArguments(driverVersion);
    f->Execute("cudaDriverGetVersion");
    if(f->Success())
        *driverVersion = *(f->GetOutputHostPointer<int>());
    return f->GetExitCode();
}

extern cudaError_t cudaRuntimeGetVersion(int *runtimeVersion) {
    Frontend *f = Frontend::GetFrontend();
    f->AddHostPointerForArguments(runtimeVersion);
    f->Execute("cudaDriverGetVersion");
    if(f->Success())
        *runtimeVersion = *(f->GetOutputHostPointer<int>());
    return f->GetExitCode();
}
