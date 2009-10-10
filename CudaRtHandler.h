/* 
 * File:   CudaRtHandler.h
 * Author: cjg
 *
 * Created on October 10, 2009, 10:51 PM
 */

#ifndef _CUDARTHANDLER_H
#define	_CUDARTHANDLER_H

#include <iostream>

class CudaRtHandler {
public:
    CudaRtHandler(std::istream & input, std::ostream & output);
    virtual ~CudaRtHandler();
    void GetDeviceCount();
    void GetDeviceProperties();
private:
    std::istream & mInput;
    std::ostream & mOutput;
};

#endif	/* _CUDARTHANDLER_H */

