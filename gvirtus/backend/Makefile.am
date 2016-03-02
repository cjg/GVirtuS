bin_PROGRAMS = gvirtus-backend
gvirtus_backend_SOURCES = \
Backend.cpp\
Backend.h\
Handler.cpp\
Handler.h\
Process.cpp\
Process.h\
main.cpp
gvirtus_backend_includedir=$(includedir)/gvirtus
gvirtus_backend_include_HEADERS= \
Backend.h\
Handler.h\
Process.h
gvirtus_backend_CXXFLAGS = -fPIC
gvirtus_backend_LDFLAGS = -Wl,--export-dynamic -Wl,--whole-archive -Wl,../util/libutil.a -Wl,--no-whole-archive
gvirtus_backend_LDADD = -lrt -ldl
