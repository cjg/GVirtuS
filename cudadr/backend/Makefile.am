sbin_PROGRAMS = gvirtus-cudadr-backend
gvirtus_cudadr_backend_SOURCES = \
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
CudaDrHandler_texture.cpp\
main.cpp
gvirtus_cudadr_backend_LDADD = ../../backend/libbackend.a ../../util/libutil.a $(CUDA_LIB_FLAGS) -lcuda -lcudart -lrt

