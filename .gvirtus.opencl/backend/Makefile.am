backendlibdir = $(libdir)/backend
backendlib_LTLIBRARIES = libOpencl-backend.la
libOpencl_backend_la_SOURCES = \
OpenclHandler.cpp\
OpenclHandler.h\
OpenclHandler_Platform.cpp\
OpenclHandler_Context.cpp\
OpenclHandler_Queue.cpp\
OpenclHandler_Object.cpp\
OpenclHandler_KernelObject.cpp\
OpenclHandler_MemoryObject.cpp\
OpenclHandler_EnqueuedCommands.cpp\
OpenclHandler_FlushandFinish.cpp\
OpenclHandler_Device.cpp\
../util/OpenclUtil.cpp\
../util/OpenclUtil.h
libOpencl_backend_la_AMCXXFLAGS = -fPIC
libOpencl_backend_la_LIBADD = $(OPENCL_LIB_FLAGS) -lrt -lOpenCL

