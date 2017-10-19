frontendlibdir = $(libdir)/frontend
frontendlib_LTLIBRARIES = libcublas.la
libcublas_la_SOURCES = \
CublasFrontend.cpp\
CublasFrontend.h\
Cublas_helper.cpp\
Cublas_level1.cpp\
Cublas_level2.cpp\
Cublas_level3.cpp
libcublas_la_LIBADD = -lgvirtus-frontend -lrt -llog4cplus
libcublas_la_LDFLAGS = -version-info 8
