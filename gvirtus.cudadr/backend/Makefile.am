backendlibdir = $(libdir)/backend
backendlib_LTLIBRARIES = libcudadr-backend.la
libcudadr_backend_la_SOURCES = \
CudaDrBackend.cpp\
CudaDrBackend.h\
CudaDrHandler.cpp\
CudaDrHandler.h\
CudaDrHandler_context.cpp\
CudaDrHandler_device.cpp\
CudaDrHandler_execution.cpp\
CudaDrHandler_initialization.cpp\
CudaDrHandler_memory.cpp\
CudaDrHandler_module.cpp\
CudaDrHandler_version.cpp\
CudaDrHandler_stream.cpp\
CudaDrHandler_event.cpp\
CudaDrHandler_texture.cpp
libcudadr_backend_la_AMCXXFLAGS = -fPIC
libcudadr_backend_la_LDFLAGS = -Wl,--whole-archive -Wl,../util/libcudautil.a -Wl,--no-whole-archive
libcudadr_backend_la_LIBADD = $(CUDA_LIB_FLAGS) -lrt -lcuda -llog4cplus

