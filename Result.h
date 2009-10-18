/* 
 * File:   Result.h
 * Author: cjg
 *
 * Created on October 18, 2009, 1:23 PM
 */

#ifndef _RESULT_H
#define	_RESULT_H

#include <builtin_types.h>
#include "Buffer.h"

class Result {
public:
    Result(cudaError_t exit_code, const Buffer * output_buffer);
    Result(const Result& orig);
    virtual ~Result();
    cudaError_t GetExitCode();
    const Buffer & GetOutputBufffer() const;
private:
    cudaError_t mExitCode;
    Buffer * mpOutputBuffer;
};

#endif	/* _RESULT_H */

