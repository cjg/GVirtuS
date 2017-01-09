frontendlibdir = $(libdir)/frontend
frontendlib_LTLIBRARIES = libcuda.la
libcuda_la_SOURCES = \
CudaDr.cpp\
CudaDr.h\
CudaDrFrontend.cpp\
CudaDrFrontend.h\
CudaDr_context.cpp\
CudaDr_device.cpp\
CudaDr_execution.cpp\
CudaDr_initialization.cpp\
CudaDr_memory.cpp\
CudaDr_module.cpp\
CudaDr_version.cpp\
CudaDr_stream.cpp\
CudaDr_event.cpp\
CudaDr_texture.cpp

libcuda_la_LIBADD = -lgvirtus-frontend -lrt -llog4cplus
libcuda_la_LDFLAGS = -Wl,--whole-archive -Wl,../util/libcudautil.a -Wl,--no-whole-archive -version-info 1:0:0


