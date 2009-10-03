/* 
 * File:   Observer.h
 * Author: cjg
 *
 * Created on October 2, 2009, 3:54 PM
 */

#ifndef _OBSERVER_H
#define	_OBSERVER_H

#include <string>

class Observer {
public:
    virtual ~Observer();
    virtual void EventOccurred(std::string & event, void * object);
private:

};

#endif	/* _OBSERVER_H */

