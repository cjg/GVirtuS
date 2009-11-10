/* 
 * File:   MemoryEntry.cpp
 * Author: cjg
 * 
 * Created on November 10, 2009, 11:23 AM
 */

#include <iostream>
#include "MemoryEntry.h"

using namespace std;

MemoryEntry::MemoryEntry(void* pHostHandler, void* pDeviceHandler,
        size_t size) {
    mpHostHander = pHostHandler;
    mpDeviceHandler = pDeviceHandler;

    mSize = size;
    cout << "mpHostHandler: " << mpHostHander << " mpDeviceHandler: "
        << mpDeviceHandler << " mSize: " << mSize << endl;
}

MemoryEntry::MemoryEntry(const MemoryEntry& orig) {
}

MemoryEntry::~MemoryEntry() {
}

void *MemoryEntry::Get() {
    return mpDeviceHandler;
}

void *MemoryEntry::Get(void* pHostHandler) {
    size_t offset = (size_t) pHostHandler - (size_t) mpHostHander;
    if(offset < mSize)
        return (void *) ((char *) mpDeviceHandler + offset);
    else
        return NULL;
}
