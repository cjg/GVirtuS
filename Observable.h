/* 
 * File:   Observable.h
 * Author: cjg
 *
 * Created on 3 ottobre 2009, 9.26
 */

#ifndef _OBSERVABLE_H
#define	_OBSERVABLE_H

#include <string>
#include <vector>
#include <map>
#include "Observer.h"

class Observable {
public:
    Observable();
    virtual ~Observable();
    void AddObserver(std::string & event, const Observer * observer);
    void AddObserver(const char * event, const Observer * observer);
protected:
    void Notify(std::string & event);
    void Notify(const char * event);
private:
    std::map<std::string, std::vector<Observer *> *> * mObservers;
};

#endif	/* _OBSERVABLE_H */

