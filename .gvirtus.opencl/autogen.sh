#!/bin/bash

aclocal || exit $?
autoconf || exit $?
libtoolize --force || exit $?
automake --add-missing --copy || exit $?
#echo ./configure "$@" --host=i686-linux-gnu "CFLAGS=-m32" "CXXFLAGS=-m32" "LDFLAGS=-m32"
#./configure "$@" "CFLAGS=-m32" "CXXFLAGS=-m32" "LDFLAGS=-m32"
./configure "$@"
# End Of File
