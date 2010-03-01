aclocal
autoconf
libtoolize --force
automake --add-missing
echo ./configure "$@"
./configure "$@"
