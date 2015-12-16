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

lib_LTLIBRARIES = libgvirtus-frontend.la
libgvirtus_frontend_la_SOURCES = \
Frontend.cpp\
Frontend.h
libgvirtus_frontend_includedir=$(includedir)/gvirtus
libgvirtus_frontend_include_HEADERS= \
Frontend.h
libgvirtus_frontend_la_LDFLAGS = -Wl,--whole-archive -Wl,../util/libutil.a
libgvirtus_frontend_la_LIBADD = -lrt
libgvirtus_frontend_la_CXXFLAGS = -fPIC

