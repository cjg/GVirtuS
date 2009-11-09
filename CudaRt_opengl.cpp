#include "Frontend.h"
#include "CudaRt.h"

using namespace std;

extern cudaError_t cudaGLMapBufferObject(void **devPtr, GLuint bufObj) {
    // FIXME: implement
    cerr << "*** Error: cudaGlMapBufferObject() not yet implemented!" << endl;
    return cudaErrorUnknown;
}

extern cudaError_t cudaGLMapBufferObjectAsync(void **devPtr, GLuint bufObj,
        cudaStream_t stream) {
    // FIXME: implement
    cerr << "*** Error: cudaGlMapBufferObjectAsync() not yet implemented!"
            << endl;
    return cudaErrorUnknown;
}

extern cudaError_t cudaGLRegisterBufferObject(GLuint bufObj) {
    // FIXME: implement
    cerr << "*** Error: cudaGlRegisterBufferObject() not yet implemented!"
            << endl;
    return cudaErrorUnknown;
}

extern cudaError_t cudaGLSetBufferObjectMapFlags(GLuint bufObj,
        unsigned int flags) {
    // FIXME: implement
    cerr << "*** Error: cudaGLSetBufferObjectMapFlags() not yet implemented!"
            << endl;
    return cudaErrorUnknown;
}

extern cudaError_t cudaGLSetGLDevice(int device) {
    // FIXME: implement
    cerr << "*** Error: cudaGlSetGLDevice() not yet implemented!" << endl;
    return cudaErrorUnknown;
}

extern cudaError_t cudaGLUnmapBufferObject(GLuint bufObj) {
    // FIXME: implement
    cerr << "*** Error: cudaGLUnmapBufferObject() not yet implemented!" << endl;
    return cudaErrorUnknown;
}

extern cudaError_t cudaGLUnmapBufferObjectAsync(GLuint bufObj,
        cudaStream_t stream) {
    // FIXME: implement
    cerr << "*** Error: cudaGLUnmapBufferObjectAsync() not yet implemented!"
            << endl;
    return cudaErrorUnknown;
}

extern cudaError_t cudaGLUnregisterBufferObject(GLuint bufObj) {
    // FIXME: implement
    cerr << "*** Error: cudaGLUnregisterBufferObject() not yet implemented!"
            << endl;
    return cudaErrorUnknown;
}
