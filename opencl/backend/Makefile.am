sbin_PROGRAMS = gvirtus-opencl-backend
gvirtus_opencl_backend_SOURCES = \
OclBackend.cpp\
OclBackend.h\
OclHandler.cpp\
OclHandler.h\
OclHandler_queryPlatformInfo.cpp\
OclHandler_queryDevices.cpp\
OclHandler_contexts.cpp\
OclHandler_memoryObjects.cpp\
OclHandler_commandQueues.cpp\
OclHandler_programObjects.cpp\
OclHandler_kernelObjects.cpp\
OclHandler_executingKernels.cpp\
OclHandler_flushAndFinish.cpp\
OclHandler_eventObjects.cpp\
OclHandler_profilingOperations.cpp\
main.cpp
gvirtus_opencl_backend_LDADD = ../../backend/libbackend.a ../../util/libutil.a  $(OPENCL_LIB_FLAGS) -lOpenCL -lrt
