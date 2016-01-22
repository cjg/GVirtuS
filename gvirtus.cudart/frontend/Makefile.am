frontendlibdir = $(libdir)/frontend
frontendlib_LTLIBRARIES = libcudart.la
libcudart_la_SOURCES = \
CudaRt.cpp\
CudaRt.h\
CudaRtFrontend.cpp\
CudaRtFrontend.h\
CudaRt_device.cpp\
CudaRt_error.cpp\
CudaRt_event.cpp\
CudaRt_execution.cpp\
CudaRt_opengl.cpp\
CudaRt_internal.cpp\
CudaRt_memory.cpp\
CudaRt_stream.cpp\
CudaRt_surface.cpp\
CudaRt_texture.cpp\
CudaRt_thread.cpp\
CudaRt_version.cpp

libcudart_la_LIBADD = -lgvirtus-frontend -lrt
libcudart_la_LDFLAGS = -Wl,--whole-archive -Wl,../util/libcudautil.a -version-info 11:14:5


