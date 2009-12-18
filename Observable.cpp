/* 
 * File:   Observable.cpp
 * Author: cjg
 * 
 * Created on 3 ottobre 2009, 9.26
 */

#include "Observable.h"

using namespace std;

Observable::Observable() {
    mObservers = new map<string, vector<Observer *> *>();
}

Observable::~Observable() {
    for (map<string, vector<Observer *> *>::iterator it = mObservers->begin();
            it != mObservers->end(); it++)
        delete it->second;
    delete mObservers;
}

void Observable::AddObserver(std::string& event, const Observer* observer) {
    vector<Observer *> *observers;
    map<string, vector<Observer *> *>::iterator it = mObservers->find(event);
    if (it == mObservers->end()) {
        observers = new vector<Observer *>();
        mObservers->insert(make_pair(event, observers));
    } else
        observers = it->second;
    for (vector<Observer *>::iterator it = observers->begin();
            it != observers->end(); it++)
        if (*it == observer)
            return;
    observers->push_back(const_cast<Observer *>(observer));
}

void Observable::AddObserver(const char* event, const Observer* observer) {
    string tmp(event);
    AddObserver(tmp, observer);
}

void Observable::Notify(std::string& event) {
    vector<Observer *> *observers;
    map<string, vector<Observer *> *>::iterator it = mObservers->find(event);
    if (it == mObservers->end())
        return;
    observers = it->second;
    for (vector<Observer *>::iterator it = observers->begin();
            it != observers->end(); it++)
        (*it)->EventOccurred(event, this);
}

void Observable::Notify(const char* event) {
    string tmp(event);
    Notify(tmp);
}
