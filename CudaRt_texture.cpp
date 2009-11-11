#include "Frontend.h"
#include "CudaRt.h"

using namespace std;

extern cudaError_t cudaBindTexture(size_t *offset,
        const textureReference *texref, const void *devPtr,
        const cudaChannelFormatDesc *desc, size_t size) {
    // FIXME: implement
    cerr << "*** Error: cudaBindTexture() not yet implemented!" << endl;
    return cudaErrorUnknown;
}

extern cudaError_t cudaBindTexture2D(size_t *offset,
        const textureReference *texref, const void *devPtr,
        const cudaChannelFormatDesc *desc, size_t width, size_t height,
        size_t pitch) {
    // FIXME: implement
    cerr << "*** Error: cudaBindTexture2D() not yet implemented!" << endl;
    return cudaErrorUnknown;
}

extern cudaError_t cudaBindTextureToArray(const textureReference *texref, 
        const cudaArray *array, const cudaChannelFormatDesc *desc) {
    Frontend *f = Frontend::GetFrontend();
    // Achtung: passing the address and the content of the textureReference
    f->AddStringForArguments(CudaUtil::MarshalHostPointer(texref));
    f->AddHostPointerForArguments(texref);
    f->AddDevicePointerForArguments((void *) array);
    f->AddHostPointerForArguments(desc);
    f->Execute("cudaBindTextureToArray");
    return f->GetExitCode();
}

extern cudaChannelFormatDesc cudaCreateChannelDesc(int x, int y, int z, int w,
        cudaChannelFormatKind f) {
    cudaChannelFormatDesc desc;
    desc.x = x;
    desc.y = y;
    desc.z = z;
    desc.w = w;
    desc.f = f;
    return desc;
}

extern cudaError_t cudaGetChannelDesc(cudaChannelFormatDesc *desc,
        const cudaArray *array) {
    // FIXME: implement
    cerr << "*** Error: cudaGetChannelDesc() not yet implemented!" << endl;
    return cudaErrorUnknown;
}

extern cudaError_t cudaGetTextureAlignmentOffset(size_t *offset,
        const textureReference *texref) {
    // FIXME: implement
    cerr << "*** Error: cudaGetTextureAlignmentOffset() not yet implemented!"
            << endl;
    return cudaErrorUnknown;
}

extern cudaError_t cudaGetTextureReference(const textureReference **texref,
        const char *symbol) {
    // FIXME: implement
    cerr << "*** Error: cudaGetTextureReference() not yet implemented!" << endl;
    return cudaErrorUnknown;
}

extern cudaError_t cudaUnbindTexture(const textureReference *texref) {
    // FIXME: implement
    cerr << "*** Error: cudaUnbindTexture() not yet implemented!" << endl;
    return cudaErrorUnknown;
}