#!/bin/bash

aclocal || exit $?
autoconf || exit $?
libtoolize --force || exit $?
automake --add-missing --copy || exit $?
echo ./configure "$@"
./configure "$@"

# End Of File
