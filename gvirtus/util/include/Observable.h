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
 * @file   Observable.h
 * @author Giuseppe Coviello <giuseppe.coviello@uniparthenope.it>
 * @date   Sat Oct 3 9:26:26 2009
 * 
 * @brief  
 * 
 * 
 */

#ifndef _OBSERVABLE_H
#define	_OBSERVABLE_H

#include <string>
#include <vector>
#include <map>

#include "Observer.h"

/**
 * Observable emits signal that can be catched by registred Observer(s).
 */
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

