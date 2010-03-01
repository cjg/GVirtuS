/* 
 * File:   Communicator.h
 * Author: cjg
 *
 * Created on 30 settembre 2009, 11.56
 */

#ifndef _COMMUNICATOR_H
#define	_COMMUNICATOR_H

#include <iostream>
#include "ConfigFile.h"

class Communicator {
public:
    static Communicator * Create(ConfigFile::Element & config);
    virtual ~Communicator();
    virtual void Serve() = 0;
    virtual const Communicator * const Accept() const = 0;
    virtual void Connect() = 0;
    virtual std::istream & GetInputStream() const = 0;
    virtual std::ostream & GetOutputStream() const = 0;
    virtual void Close() = 0;
private:

};

#endif	/* _COMMUNICATOR_H */

