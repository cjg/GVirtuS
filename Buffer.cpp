/* 
 * File:   Buffer.cpp
 * Author: cjg
 * 
 * Created on October 18, 2009, 1:16 PM
 */

#include <cstring>
#include "Buffer.h"

Buffer::Buffer(size_t initial_size, size_t block_size) {
    mSize = initial_size;
    mBlockSize = block_size;
    mLength = 0;
    mOffset = 0;
    mpBuffer = NULL;
    if (mSize == 0)
        mSize = 0;
    if ((mSize = (mSize / mBlockSize) * mBlockSize) == 0)
        mSize = mBlockSize;
    if ((mpBuffer = (char *) malloc(mSize)) == NULL)
        throw "Can't allocate memory.";
}

Buffer::Buffer(const Buffer& orig) {
    mBlockSize = orig.mBlockSize;
    mLength = orig.mLength;
    mSize = orig.mSize;
    mOffset = orig.mOffset;
    if ((mpBuffer = (char *) malloc(mSize)) == NULL)
        throw "Can't allocate memory.";
    memmove(mpBuffer, orig.mpBuffer, mLength);
}

Buffer::~Buffer() {
    free(mpBuffer);
}

const char * const Buffer::GetBuffer() const {
    return mpBuffer;
}

size_t Buffer::GetBufferSize() const {
    return mLength;
}

void Buffer::Dump(std::ostream& out) const {
    out.write(mpBuffer, mLength);
}

