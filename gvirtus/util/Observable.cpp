/*
 * gVirtuS -- A GPGPU transparent virtualization component.
 *
 * Copyright (C) 2009-2010  The University of Napoli Parthenope at Naples.
 *
 * This file is part of gVirtuS.
 *
 * gVirtuS is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * gVirtuS is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with gVirtuS; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
 *
 * Written by: Giuseppe Coviello <giuseppe.coviello@uniparthenope.it>,
 *             Department of Applied Science
 */

/**
 * @file   Observable.cpp
 * @author Giuseppe Coviello <giuseppe.coviello@uniparthenope.it>
 * @date   Sat Oct 3 9:26:26 2009
 *
 * @brief
 *
 *
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
