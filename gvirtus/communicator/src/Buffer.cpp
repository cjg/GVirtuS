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
 * @file   Buffer.cpp
 * @author Giuseppe Coviello <giuseppe.coviello@uniparthenope.it>
 * @date   Sun Oct 18 13:16:46 2009
 *
 * @brief
 *
 *
 */
//#define DEBUG
#include "Buffer.h"

using namespace std;

Buffer::Buffer(size_t initial_size, size_t block_size) {
  mSize = initial_size;
  mBlockSize = block_size;
  mLength = 0;
  mOffset = 0;
  mpBuffer = NULL;
  mOwnBuffer = true;
  if (mSize == 0)
    mSize = 0;
  if ((mSize = (mSize / mBlockSize) * mBlockSize) == 0)
    mSize = mBlockSize;
  if ((mpBuffer = (char *)malloc(mSize)) == NULL)
    throw "Can't allocate memory.";
  mBackOffset = mLength;
}

Buffer::Buffer(const Buffer &orig) {
  mBlockSize = orig.mBlockSize;
  mLength = orig.mLength;
  mSize = orig.mLength;
  mOffset = orig.mOffset;
  mLength = orig.mLength;
  mOwnBuffer = true;
  if ((mpBuffer = (char *)malloc(mSize)) == NULL)
    throw "Can't allocate memory.";
  memmove(mpBuffer, orig.mpBuffer, mLength);
  mBackOffset = mLength;
}

Buffer::Buffer(istream &in) {
  in.read((char *)&mSize, sizeof(size_t));
  mBlockSize = BLOCK_SIZE;
  mLength = mSize;
  mOffset = 0;
  mOwnBuffer = true;
  if ((mpBuffer = (char *)malloc(mSize)) == NULL)
    throw "Can't allocate memory.";
  in.read(mpBuffer, mSize);
  mBackOffset = mLength;
}

Buffer::Buffer(char *buffer, size_t buffer_size, size_t block_size) {
  mSize = buffer_size;
  mBlockSize = block_size;
  mLength = mSize;
  mOffset = 0;
  mpBuffer = buffer;
  mOwnBuffer = false;
  mBackOffset = mLength;
}

Buffer::~Buffer() {
  if (mOwnBuffer)
    free(mpBuffer);
}

void
Buffer::Reset() {
  mLength = 0;
  mOffset = 0;
  mBackOffset = 0;
}

void
Buffer::Reset(gvirtus::comm::Communicator *c) {
  c->Read((char *)&mLength, sizeof(size_t));
  std::cout << "FTERCREAD\n\n";

#ifdef DEBUG
  std::cout << "readed size of buffer " << mLength << std::endl;
#endif
  mOffset = 0;
  mBackOffset = mLength;
  if (mLength >= mSize) {
    mSize = (mLength / mBlockSize + 1) * mBlockSize;
    if ((mpBuffer = (char *)realloc(mpBuffer, mSize)) == NULL)
      throw "Can't reallocate memory.";
  }
  c->Read(mpBuffer, mLength);
}

const char *const
Buffer::GetBuffer() const {
  return mpBuffer;
}

size_t
Buffer::GetBufferSize() const {
  return mLength;
}

void
Buffer::Dump(gvirtus::comm::Communicator *c) const {
  /**
   *  TO-DO scrivi al message dispatcher che stai per scrivere
   *  acquisisci il lock
   *  scrivi
   *  md->write(communicator out, tid, mpBuffer, mLenght);
   */
  c->Write((char *)&mLength, sizeof(size_t));
  c->Write(mpBuffer, mLength);
  c->Sync();
  /**
   * TO-DO rilascia il lock
   * notifica
   *
   */
}
