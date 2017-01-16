backendlibdir = $(libdir)/backend
backendlib_LTLIBRARIES = libcublas-backend.la
libOpencl_backend_la_SOURCES = \
CublasHandler.cpp\
CublasHandler.h\
libCublas_backend_la_AMCXXFLAGS = -fPIC
libCublas_backend_la_LIBADD = $(OPENCL_LIB_FLAGS) -lrt -lcublas

