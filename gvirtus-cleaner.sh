#!/bin/bash

if [ $# -ne 1 ]; then
    dir=./
else
    dir=$1
fi

find $dir -name '*.so' -delete -print
find $dir -name '*.a' -delete -print
find $dir -name '*.cbp' -delete -print
find $dir -name 'gvirtus-backend' -delete -print
find $dir/*/.graphviz -name '*.dot*' | grep -vP ".*dot$" | xargs rm -fv
find $dir -name 'install_manifest.txt' -delete -print
find $dir -name 'cmake_install.cmake' -delete -print
find $dir -name 'CMakeCache.txt' -delete -print
find $dir -name 'Makefile' -delete -print
find $dir -name 'CMakeFiles' -exec rm -rv {} +
find $dir -name 'cmake-build-debug' -exec rm -rv {} +
find $dir -name 'build' -exec rm -rv {} +
find $dir -name 'uvw-prefix' -exec rm -frv {} +
find $dir -name 'libuv-prefix' -exec rm -frv {} +
find $dir -name 'nlohmann-prefix' -exec rm -frv {} +

echo
echo
echo "***********************************"
echo "*       CMakeFiles removed        *"
echo "***********************************"
echo
