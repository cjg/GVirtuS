/* 
 * File:   CudaUtil.h
 * Author: cjg
 *
 * Created on October 11, 2009, 5:16 PM
 */

#ifndef _CUDAUTIL_H
#define	_CUDAUTIL_H

#include <cstdlib>

class CudaUtil {
public:
    CudaUtil();
    CudaUtil(const CudaUtil& orig);
    virtual ~CudaUtil();
    static const size_t MarshaledDevicePointerSize = sizeof(void *) * 2 + 3;
private:

};

#endif	/* _CUDAUTIL_H */

