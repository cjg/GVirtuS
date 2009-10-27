/* 
 * File:   CudaUtil.h
 * Author: cjg
 *
 * Created on October 11, 2009, 5:16 PM
 */

#ifndef _CUDAUTIL_H
#define	_CUDAUTIL_H

#include <cstdlib>
#include "__cudaFatFormat.h"
#include "Buffer.h"

class CudaUtil {
public:
    CudaUtil();
    CudaUtil(const CudaUtil& orig);
    virtual ~CudaUtil();
    static const size_t MarshaledDevicePointerSize = sizeof(void *) * 2 + 3;
    static Buffer * MarshalFatCudaBinary(__cudaFatCudaBinary * bin);
    static void DumpFatCudaBinary(__cudaFatCudaBinary * bin);
private:
};

#endif	/* _CUDAUTIL_H */

