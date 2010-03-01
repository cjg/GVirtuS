#include <cstring>
#include "Frontend.h"
#include "CudaUtil.h"
#include "CudaRt.h"

using namespace std;

extern const char* cudaGetErrorString(cudaError_t error) {
    Frontend *f = Frontend::GetFrontend();
    f->AddVariableForArguments(error);
    f->Execute("cudaGetErrorString");
    char *error_string = strdup(f->GetOutputString());
    return error_string;
}

extern cudaError_t cudaGetLastError(void) {
    Frontend *f = Frontend::GetFrontend();
    f->Execute("cudaGetLastError");
    return f->GetExitCode();
}
