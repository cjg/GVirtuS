backendlibdir = $(libdir)/backend
backendlib_LTLIBRARIES = libcudnn-backend.la
libOpencl_backend_la_SOURCES = \
CudnnHandler.cpp\
CudnnHandler.h\
libCudnn_backend_la_AMCXXFLAGS = -fPIC
libCudnn_backend_la_LIBADD = $(CUDNN_LIB_FLAGS) -lrt -lcudnn

