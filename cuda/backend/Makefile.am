sbin_PROGRAMS = gvirtus-cuda-backend
gvirtus_cuda_backend_SOURCES = \
CudaRtBackend.cpp\
CudaRtBackend.h\
CudaRtHandler.cpp\
CudaRtHandler.h\
CudaRtHandler_device.cpp\
CudaRtHandler_error.cpp\
CudaRtHandler_event.cpp\
CudaRtHandler_execution.cpp\
CudaRtHandler_internal.cpp\
CudaRtHandler_memory.cpp\
CudaRtHandler_stream.cpp\
CudaRtHandler_texture.cpp\
CudaRtHandler_thread.cpp\
CudaRtHandler_version.cpp\
main.cpp
gvirtus_cuda_backend_LDADD = ../../backend/libbackend.a ../../util/libutil.a $(CUDA_LIB_FLAGS) -lcudart -lrt

