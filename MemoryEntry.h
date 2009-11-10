/* 
 * File:   MemoryEntry.h
 * Author: cjg
 *
 * Created on November 10, 2009, 11:23 AM
 */

#ifndef _MEMORYENTRY_H
#define	_MEMORYENTRY_H

#include <cstdlib>

class MemoryEntry {
public:
    MemoryEntry(void *pHostHandler, void *pDeviceHandler, size_t size);
    MemoryEntry(const MemoryEntry& orig);
    virtual ~MemoryEntry();
    void *Get();
    void *Get(void *pHostHandler);
private:
    void *mpHostHander;
    void *mpDeviceHandler;
    size_t mSize;
};

#endif	/* _MEMORYENTRY_H */

