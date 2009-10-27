#include <cstring>
#include "Frontend.h"
#include "CudaUtil.h"
#include "CudaRt.h"

using namespace std;

extern const char* cudaGetErrorString(cudaError_t error) {
    CudaRt *c = new CudaRt("cudaGetErrorString");
    c->AddVariableForArguments(error);
    c->Execute();
    char *error_string = strdup(c->GetOutputHostPointer<char>());
    CudaRt::Finalize(c);
    return error_string;
}

extern cudaError_t cudaGetLastError(void) {
    CudaRt *c = new CudaRt("cudaGetLastError");
    c->Execute();
    return CudaRt::Finalize(c);
}
