#include "Frontend.h"
#include "CudaRt.h"

using namespace std;

extern cudaError_t cudaGLMapBufferObject(void **devPtr, GLuint bufObj) {
    cerr << "I'm sorry but it isn't possibile to use OpenGL Interoperability "
            << "API." << endl << "Giving up ..." << endl;
    exit(-1);
    return cudaErrorUnknown;
}

extern cudaError_t cudaGLMapBufferObjectAsync(void **devPtr, GLuint bufObj,
        cudaStream_t stream) {
    cerr << "I'm sorry but it isn't possibile to use OpenGL Interoperability "
            << "API." << endl << "Giving up ..." << endl;
    exit(-1);
    return cudaErrorUnknown;
}

extern cudaError_t cudaGLRegisterBufferObject(GLuint bufObj) {
    cerr << "I'm sorry but it isn't possibile to use OpenGL Interoperability "
            << "API." << endl << "Giving up ..." << endl;
    exit(-1);
    return cudaErrorUnknown;
}

extern cudaError_t cudaGLSetBufferObjectMapFlags(GLuint bufObj,
        unsigned int flags) {
    cerr << "I'm sorry but it isn't possibile to use OpenGL Interoperability "
            << "API." << endl << "Giving up ..." << endl;
    exit(-1);
    return cudaErrorUnknown;
}

extern cudaError_t cudaGLSetGLDevice(int device) {
    cerr << "I'm sorry but it isn't possibile to use OpenGL Interoperability "
            << "API." << endl << "Giving up ..." << endl;
    exit(-1);
    return cudaErrorUnknown;
}

extern cudaError_t cudaGLUnmapBufferObject(GLuint bufObj) {
    cerr << "I'm sorry but it isn't possibile to use OpenGL Interoperability "
            << "API." << endl << "Giving up ..." << endl;
    exit(-1);
    return cudaErrorUnknown;
}

extern cudaError_t cudaGLUnmapBufferObjectAsync(GLuint bufObj,
        cudaStream_t stream) {
    cerr << "I'm sorry but it isn't possibile to use OpenGL Interoperability "
            << "API." << endl << "Giving up ..." << endl;
    exit(-1);
    return cudaErrorUnknown;
}

extern cudaError_t cudaGLUnregisterBufferObject(GLuint bufObj) {
    cerr << "I'm sorry but it isn't possibile to use OpenGL Interoperability "
            << "API." << endl << "Giving up ..." << endl;
    exit(-1);
    return cudaErrorUnknown;
}
