backendlibdir = $(libdir)/backend
backendlib_LTLIBRARIES = libcudart-backend.la
libcudart_backend_la_SOURCES = \
CudaRtHandler.cpp\
CudaRtHandler.h\
CudaRtHandler_device.cpp\
CudaRtHandler_error.cpp\
CudaRtHandler_event.cpp\
CudaRtHandler_execution.cpp\
CudaRtHandler_internal.cpp\
CudaRtHandler_memory.cpp\
CudaRtHandler_opengl.cpp\
CudaRtHandler_stream.cpp\
CudaRtHandler_surface.cpp\
CudaRtHandler_texture.cpp\
CudaRtHandler_thread.cpp\
CudaRtHandler_version.cpp
libcudart_backend_la_AMCXXFLAGS = -fPIC
libcudart_backend_la_LDFLAGS = -Wl,--whole-archive -Wl,../util/libcudautil.a
libcudart_backend_la_LIBADD = $(CUDA_LIB_FLAGS) -lrt -lcudart

