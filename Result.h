/* 
 * File:   Result.h
 * Author: cjg
 *
 * Created on October 18, 2009, 1:23 PM
 */

#ifndef _RESULT_H
#define	_RESULT_H

#include <iostream>
#include <builtin_types.h>
#include "Buffer.h"

class Result {
public:
    Result(cudaError_t exit_code, const Buffer * output_buffer);
    Result(const Result& orig);
    Result(std::istream & in);
    virtual ~Result();
    cudaError_t GetExitCode();
    const Buffer * GetOutputBufffer() const;
    void Dump(std::ostream & out);
private:
    cudaError_t mExitCode;
    Buffer * mpOutputBuffer;
};

#endif	/* _RESULT_H */

