#include <cstring>
#include "Frontend.h"
#include "CudaUtil.h"
#include "CudaRt.h"

using namespace std;

extern cudaError_t cudaConfigureCall(dim3 gridDim, dim3 blockDim,
        size_t sharedMem, cudaStream_t stream)
{
    CudaRt *c = new CudaRt("cudaConfigureCall");
    c->AddVariableForArguments(gridDim);
    c->AddVariableForArguments(blockDim);
    c->AddVariableForArguments(sharedMem);
    c->AddVariableForArguments(stream);
    c->Execute();
    return CudaRt::Finalize(c);
}

extern cudaError_t cudaLaunch(const char *entry)
{
    CudaRt *c = new CudaRt("cudaLaunch");
    c->AddStringForArguments(entry);
    c->Execute();
    return CudaRt::Finalize(c);
}

extern cudaError_t cudaSetupArgument(const void *arg, size_t size, size_t offset)
{
    CudaRt *c = new CudaRt("cudaSetupArgument");
    cout << ((void *)(static_cast<const char *>(arg))) << endl;
    cout << ((void *)(static_cast<const char *>(arg) + offset)) << endl << endl;
    c->AddHostPointerForArguments(static_cast<const char *>(arg), size);
    c->AddVariableForArguments(size);
    c->AddVariableForArguments(offset);
    c->Execute();
    return CudaRt::Finalize(c);
}


