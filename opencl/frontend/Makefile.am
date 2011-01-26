lib_LTLIBRARIES = libOpenCL.la

libOpenCL_la_SOURCES = Ocl.cpp\
Ocl.h\
OclFrontend.cpp\
OclFrontend.h\
Ocl_commandQueues.cpp\
Ocl_contexts.cpp\
Ocl_eventObjects.cpp\
Ocl_executingKernels.cpp\
Ocl_ExecutionKernelsMemoryObjectCommands.cpp\
Ocl_flushAndFinish.cpp\
Ocl_kernelObjects.cpp\
Ocl_memoryObjects.cpp\
Ocl_profilingOperations.cpp\
Ocl_programObjects.cpp\
Ocl_queryDevices.cpp\
Ocl_queryPlatformInfo.cpp\
Ocl_samplerObjects.cpp\
Ocl_clGlSharingApis.cpp
libOpenCL_la_LIBADD = ../../frontend/libfrontend.a ../../util/libutil.a -lrt -lOpenCL
libOpenCL_la_LDFLAGS = -version-info 1:1:0