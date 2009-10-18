/* 
 * File:   Result.cpp
 * Author: cjg
 * 
 * Created on October 18, 2009, 1:23 PM
 */

#include "Result.h"

Result::Result(cudaError_t exit_code, const Buffer* output_buffer) {
    mExitCode = exit_code;
    mpOutputBuffer = const_cast<Buffer *>(output_buffer);
}

Result::Result(const Result& orig) {
}

Result::~Result() {
}

cudaError_t Result::GetExitCode() {
    return mExitCode;
}

const Buffer & Result::GetOutputBufffer() const {
    return *(mpOutputBuffer);
}

