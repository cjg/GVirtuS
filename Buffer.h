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

#define BLOCK_SIZE  1024

class Buffer {
public:
    Buffer(size_t initial_size = 0, size_t block_size = BLOCK_SIZE);
    Buffer(const Buffer& orig);
    Buffer(std::istream & in);
    virtual ~Buffer();

    template <class T> void Add(T item) {
        while ((mLength + sizeof (T)) >= mSize) {
            mSize += mBlockSize;
            if ((mpBuffer = (char *) realloc(mpBuffer, mSize)) == NULL)
                throw "Can't reallocate memory.";
        }
        memmove(mpBuffer + mLength, (char *) & item, sizeof (T));
        mLength += sizeof (T);
    };

    template <class T> void Add(T *item, size_t n = 1) {
        while ((mLength + (sizeof (T) * n)) >= mSize) {
            mSize += mBlockSize;
            if (realloc(&mpBuffer, mSize))
                throw "Can't reallocate memory.";
        }
        memmove(mpBuffer + mLength, (char *) item, sizeof (T) * n);
        mLength += sizeof (T) * n;
    }

    template <class T> void Read(std::istream & in) {
        while ((mLength + sizeof (T)) >= mSize) {
            mSize += mBlockSize;
            if ((mpBuffer = (char *) realloc(mpBuffer, mSize)) == NULL)
                throw "Can't reallocate memory.";
        }
        in.read(mpBuffer + mLength, sizeof (T));
        mLength += sizeof (T);
    }

    template <class T> void Read(std::istream & in, size_t n = 1) {
        while ((mLength + (sizeof (T) * n)) >= mSize) {
            mSize += mBlockSize;
            if (realloc(&mpBuffer, mSize))
                throw "Can't reallocate memory.";
        }
        in.read(mpBuffer + mLength, sizeof (T) * n);
        mLength += sizeof (T) * n;
    }

    template <class T> T Get() {
        if (mOffset + sizeof (T) > mLength)
            throw "Can't read any " + std::string(typeid (T).name()) + ".";
        T result = *((T *) (mpBuffer + mOffset));
        mOffset += sizeof (T);
        return result;
    }

    template <class T> T * Get(size_t n) {
        if (mOffset + sizeof (T) * n > mLength)
            throw "Can't read  " + std::string(typeid (T).name()) + ".";
        T *result = new T[n];
        memmove((char *) result, mpBuffer + mOffset, sizeof (T) * n);
        mOffset += sizeof (T) * n;
        return result;
    }

    template <class T>T * Assign(size_t n = 1) {
        if (mOffset + sizeof (T) * n > mLength)
            throw "Can't read  " + std::string(typeid (T).name()) + ".";
        T * result = (T *) (mpBuffer + mOffset);
        mOffset += sizeof (T) * n;
        return result;
    }
    const char * const GetBuffer() const;
    size_t GetBufferSize() const;
    void Dump(std::ostream & out) const;
private:
    size_t mBlockSize;
    size_t mSize;
    size_t mLength;
    size_t mOffset;
    char * mpBuffer;
};

#endif	/* _BUFFER_H */
