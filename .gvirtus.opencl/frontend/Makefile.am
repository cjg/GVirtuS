frontendlibdir = $(libdir)/frontend
frontendlib_LTLIBRARIES = libOpencl.la
libOpencl_la_SOURCES = \
../util/OpenclUtil.cpp\
../util/OpenclUtil.h\
Opencl_gv.cpp\
Opencl_gv.h\
OpenclFrontend.cpp\
OpenclFrontend.h\
Opencl_Platform.cpp\
Opencl_Context.cpp\
Opencl_Queue.cpp\
Opencl_Object.cpp\
Opencl_KernelObject.cpp\
Opencl_EnqueuedCommands.cpp\
Opencl_FlushandFinish.cpp\
Opencl_MemoryObject.cpp\
Opencl_Device.cpp
libOpencl_la_LIBADD = -lgvirtus-frontend -lrt
#libOpencl_la_LDFLAGS = -Wl,--whole-archive -Wl,../util/libopenclutil.a -version-info 3:14:0

