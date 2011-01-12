#lib_LTLIBRARIES = libcudart.la
#libcudart_la_SOURCES = \
CudaRt.cpp\
CudaRt_device.cpp\
CudaRt_error.cpp\
CudaRt_event.cpp\
CudaRt_execution.cpp\
CudaRt.h\
CudaRt_internal.cpp\
CudaRt_memory.cpp\
CudaRt_opengl.cpp\
CudaRt_stream.cpp\
CudaRt_texture.cpp\
CudaRt_thread.cpp\
CudaRt_version.cpp\
Frontend.cpp\
Frontend.h
#libcudart_la_LIBADD = ../util/libutil.a -lrt
#libcudart_la_LDFLAGS = -version-info 3:14:0

noinst_LIBRARIES = libfrontend.a
libfrontend_a_SOURCES = \
Frontend.cpp\
Frontend.h
libfrontend_a_LIBADD = ../util/libutil.a
libfrontend_a_CXXFLAGS = -fPIC

