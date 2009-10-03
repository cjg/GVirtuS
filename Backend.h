/* 
 * File:   Backend.h
 * Author: cjg
 *
 * Created on 30 settembre 2009, 12.29
 */

#ifndef _BACKEND_H
#define	_BACKEND_H

#include "Observer.h"
#include "Communicator.h"

class Backend : public Observer {
public:
    Backend(const Communicator * communicator);
    void Start();
    void EventOccurred(std::string & event, void * object);
private:
    Communicator *mpCommunicator;
};

#endif	/* _BACKEND_H */

