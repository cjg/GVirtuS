backendlibdir = $(libdir)/backend
backendlib_LTLIBRARIES = libcublas-backend.la
libcublas_backend_la_SOURCES = \
CublasHandler.cpp\
CublasHandler.h\
libcublas_backend_la_AMCXXFLAGS = -fPIC
libcublas_backend_la_LIBADD = $(CUBLAS_LIB_FLAGS) -lrt -lcublas

