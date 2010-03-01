/* 
 * File:   Buffer.h
 * Author: cjg
 *
 * Created on October 18, 2009, 1:16 PM
 */

#ifndef _BUFFER_H
#define	_BUFFER_H

#include <iostream>
#include <cstdlib>
#include <typeinfo>
#include <cstring>
#include <stdint.h>

#define BLOCK_SIZE  4096

using namespace std;

class Buffer {
public:
    Buffer(size_t initial_size = 0, size_t block_size = BLOCK_SIZE);
    Buffer(const Buffer& orig);
    Buffer(std::istream & in);
    Buffer(char * buffer, size_t buffer_size, size_t block_size = BLOCK_SIZE);
    virtual ~Buffer();

    template <class T> void Add(T item) {
        if ((mLength + (sizeof (T))) >= mSize) {
            mSize = ((mLength + (sizeof (T))) / mBlockSize + 1) * mBlockSize;
            if ((mpBuffer = (char *) realloc(mpBuffer, mSize)) == NULL)
                throw "Can't reallocate memory.";
        }
        memmove(mpBuffer + mLength, (char *) & item, sizeof (T));
        mLength += sizeof (T);
        mBackOffset = mLength;
    };

    template <class T> void Add(T *item, size_t n = 1) {
        if(item == NULL) {
            Add((size_t) 0);
            return;
        }
        size_t size = sizeof(T) * n;
        Add(size);
        if ((mLength + size) >= mSize) {
            mSize = ((mLength + size) / mBlockSize + 1) * mBlockSize;
            if ((mpBuffer = (char *) realloc(mpBuffer, mSize)) == NULL)
                throw "Can't reallocate memory.";
        }
        memmove(mpBuffer + mLength, (char *) item, size);
        mLength += size;
        mBackOffset = mLength;
    }

    void AddString(const char *s) {
        size_t size = strlen(s) + 1;
        Add(size);
        Add(s, size);
    }

    template <class T> void AddMarshal(T item) {
        Add((uint64_t) item);
    }


    template <class T> void Read(std::istream & in) {
        while ((mLength + sizeof (T)) >= mSize) {
            mSize += mBlockSize;
            if ((mpBuffer = (char *) realloc(mpBuffer, mSize)) == NULL)
                throw "Can't reallocate memory.";
        }
        in.read(mpBuffer + mLength, sizeof (T));
        mLength += sizeof (T);
        mBackOffset = mLength;
    }

    template <class T> void Read(std::istream & in, size_t n = 1) {
        while ((mLength + (sizeof (T) * n)) >= mSize) {
            mSize += mBlockSize;
            if ((mpBuffer = (char *) realloc(mpBuffer, mSize)) == NULL)
                throw "Can't reallocate memory.";
        }
        in.read(mpBuffer + mLength, sizeof (T) * n);
        mLength += sizeof (T) * n;
        mBackOffset = mLength;
    }

    template <class T> T Get() {
        if (mOffset + sizeof (T) > mLength)
            throw "Can't read any " + std::string(typeid (T).name()) + ".";
        T result = *((T *) (mpBuffer + mOffset));
        mOffset += sizeof (T);
        return result;
    }

    template <class T>T BackGet() {
        if (mBackOffset - sizeof (T) > mLength)
            throw "Can't read  " + std::string(typeid (T).name()) + ".";
        T result = *((T *) (mpBuffer + mBackOffset - sizeof(T)));
        mBackOffset -= sizeof (T);
        return result;
    }

    template <class T> T * Get(size_t n) {
        if(Get<size_t>() == 0)
            return NULL;
        if (mOffset + sizeof (T) * n > mLength)
            throw "Can't read  " + std::string(typeid (T).name()) + ".";
        T *result = new T[n];
        memmove((char *) result, mpBuffer + mOffset, sizeof (T) * n);
        mOffset += sizeof (T) * n;
        return result;
    }

    template <class T>T * Delegate(size_t n = 1) {
        size_t size = sizeof(T) * n;
        Add(size);
        if ((mLength + size) >= mSize) {
            mSize = ((mLength + size) / mBlockSize + 1) * mBlockSize;
            if ((mpBuffer = (char *) realloc(mpBuffer, mSize)) == NULL)
                throw "Can't reallocate memory.";
        }
        T * dst = (T *) (mpBuffer + mLength);
        mLength += size;
        mBackOffset = mLength;
        return dst;
    }

    template <class T>T * Assign(size_t n = 1) {
        if(Get<size_t>() == 0)
            return NULL;
        if (mOffset + sizeof (T) * n > mLength)
            throw "Can't read  " + std::string(typeid (T).name()) + ".";
        T * result = (T *) (mpBuffer + mOffset);
        mOffset += sizeof (T) * n;
        return result;
    }

    template <class T>T * AssignAll() {
        size_t size = Get<size_t>();
        if(size == 0)
            return NULL;
        size_t n = size / sizeof(T);
        if (mOffset + sizeof (T) * n > mLength)
            throw "Can't read  " + std::string(typeid (T).name()) + ".";
        T * result = (T *) (mpBuffer + mOffset);
        mOffset += sizeof (T) * n;
        return result;
    }

    char * AssignString() {
        size_t size = Get<size_t>();
        return Assign<char>(size);
    }

    template <class T>T * BackAssign(size_t n = 1) {
        if (mBackOffset - sizeof (T) * n > mLength)
            throw "Can't read  " + std::string(typeid (T).name()) + ".";
        T * result = (T *) (mpBuffer + mBackOffset - sizeof(T) * n);
        mBackOffset -= sizeof (T) * n + sizeof(size_t);
        return result;
    }

    template <class T>T GetFromMarshal() {
        return (T) Get<uint64_t>();
    }

    void Reset();
    void Reset(std::istream & in);
    const char * const GetBuffer() const;
    size_t GetBufferSize() const;
    void Dump(std::ostream & out) const;
private:
    size_t mBlockSize;
    size_t mSize;
    size_t mLength;
    size_t mOffset;
    size_t mBackOffset;
    char * mpBuffer;
};

#endif	/* _BUFFER_H */
