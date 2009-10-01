/* 
 * File:   Mutex.h
 * Author: cjg
 *
 * Created on October 1, 2009, 11:54 AM
 */

#ifndef _MUTEX_H
#define	_MUTEX_H

#include <pthread.h>

class Mutex {
public:
    Mutex();
    virtual ~Mutex();
    void Lock();
    void Unlock();
private:
    pthread_mutex_t mMutex;
};

#define synchronized(mutex) for(bool __mutex_lock = true, (mutex).Lock(); __mutex_lock; __mutex_lock = false, (mutex).Unlock())

#endif	/* _MUTEX_H */

