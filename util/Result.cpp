/* 
 * File:   Result.cpp
 * Author: cjg
 * 
 * Created on October 18, 2009, 1:23 PM
 */

#include "Result.h"

Result::Result(cudaError_t exit_code) {
    mExitCode = exit_code;
    mpOutputBuffer = NULL;
}

Result::Result(cudaError_t exit_code, const Buffer* output_buffer) {
    mExitCode = exit_code;
    mpOutputBuffer = const_cast<Buffer *>(output_buffer);
}

Result::Result(const Result& orig) {
}

Result::Result(std::istream & in) {
    in.read((char *) &mExitCode, sizeof(cudaError_t));
    mpOutputBuffer = new Buffer(in);
}

Result::~Result() {
    delete mpOutputBuffer;
}

cudaError_t Result::GetExitCode() {
    return mExitCode;
}

const Buffer * Result::GetOutputBufffer() const {
    return mpOutputBuffer;
}

void Result::Dump(std::ostream& out) {
    out.write((char *) &mExitCode, sizeof(cudaError_t));
    if(mpOutputBuffer != NULL)
        mpOutputBuffer->Dump(out);
    else {
        size_t size = 0;
        out.write((char *) &size, sizeof(size_t));
        out.flush();
    }
}
