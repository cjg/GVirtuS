/**
 * @file   CufftFrontend.cpp
 * @author Vincenzo Santopietro <vincenzo.santopietro@uniparthenope.it>
 * @date   Jan 26, 2017, 10:01 AM
 * 
 * @brief  
 * 
 */

#include "CufftFrontend.h"

using namespace std;

CufftFrontend msInstance __attribute_used__;

map<const void*, mappedPointer>* CufftFrontend::mappedPointers = NULL;
set<const void*>* CufftFrontend::devicePointers = NULL;
map <pthread_t, stack<void*> *>* CufftFrontend::toManage = NULL;


CufftFrontend::CufftFrontend() {
    if (devicePointers == NULL)
        devicePointers = new set<const void*>();
    if (mappedPointers == NULL)
        mappedPointers = new map<const void*, mappedPointer>();
    if (toManage == NULL)
        toManage = new map <pthread_t, stack<void*> *>();
    Frontend::GetFrontend();
}


CufftFrontend::~CufftFrontend() {
}

