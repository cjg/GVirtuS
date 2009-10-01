/* 
 * File:   Mutex.cpp
 * Author: cjg
 * 
 * Created on October 1, 2009, 11:54 AM
 */

#include "Mutex.h"

Mutex::Mutex() {
    pthread_mutex_init(&mMutex, NULL);
}

Mutex::~Mutex() {
    pthread_mutex_destroy(&mMutex);
}

void Mutex::Lock() {
    pthread_mutex_lock(&mMutex);
}

void Mutex::Unlock() {
    pthread_mutex_unlock(&mMutex);
}