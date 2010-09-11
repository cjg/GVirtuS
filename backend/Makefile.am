sbin_PROGRAMS = gvirtus-backend
gvirtus_backend_SOURCES = \
Backend.cpp\
Backend.h\
CudaRtHandler.cpp\
CudaRtHandler_device.cpp\
CudaRtHandler_error.cpp\
CudaRtHandler_event.cpp\
CudaRtHandler_execution.cpp\
CudaRtHandler.h\
CudaRtHandler_internal.cpp\
CudaRtHandler_memory.cpp\
CudaRtHandler_stream.cpp\
CudaRtHandler_texture.cpp\
CudaRtHandler_thread.cpp\
CudaRtHandler_version.cpp\
main.cpp\
Process.cpp\
Process.h
gvirtus_backend_LDADD = ../util/libutil.a -lexpat $(CUDA_LIB_FLAGS) -lcudart
