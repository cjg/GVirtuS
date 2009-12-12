#
# Generated Makefile - do not edit!
#
# Edit the Makefile in the project folder instead (../Makefile). Each target
# has a -pre and a -post target defined where you can add customized code.
#
# This makefile implements configuration specific macros and targets.


# Environment
MKDIR=mkdir
CP=cp
CCADMIN=CCadmin
RANLIB=ranlib
CC=gcc
CCC=g++
CXX=g++
FC=
AS=

# Macros
CND_PLATFORM=GNU-Linux-x86
CND_CONF=Debug
CND_DISTDIR=dist

# Include project Makefile
include Makefile

# Object Directory
OBJECTDIR=build/${CND_CONF}/${CND_PLATFORM}

# Object Files
OBJECTFILES= \
	${OBJECTDIR}/CudaRt_internal.o \
	${OBJECTDIR}/CudaRt_device.o \
	${OBJECTDIR}/CudaRt_texture.o \
	${OBJECTDIR}/CudaRt.o \
	${OBJECTDIR}/CudaRt_opengl.o \
	${OBJECTDIR}/CudaRt_thread.o \
	${OBJECTDIR}/Frontend.o \
	${OBJECTDIR}/CudaRt_memory.o \
	${OBJECTDIR}/CudaRt_event.o \
	${OBJECTDIR}/CudaRt_stream.o \
	${OBJECTDIR}/CudaRt_error.o \
	${OBJECTDIR}/_ext/home/cjg/NetBeansProjects/EchoesFrontend/CudaRt_execution.o \
	${OBJECTDIR}/CudaRt_version.o

# C Compiler Flags
CFLAGS=

# CC Compiler Flags
CCFLAGS=
CXXFLAGS=

# Fortran Compiler Flags
FFLAGS=

# Assembler Flags
ASFLAGS=

# Link Libraries and Options
LDLIBSOPTIONS=-Wl,-rpath ../EchoesUtil/dist/Debug/GNU-Linux-x86 -L../EchoesUtil/dist/Debug/GNU-Linux-x86 -lEchoesUtil

# Build Targets
.build-conf: ${BUILD_SUBPROJECTS}
	${MAKE}  -f nbproject/Makefile-Debug.mk dist/Debug/GNU-Linux-x86/libEchoesFrontend.so

dist/Debug/GNU-Linux-x86/libEchoesFrontend.so: ../EchoesUtil/dist/Debug/GNU-Linux-x86/libEchoesUtil.so

dist/Debug/GNU-Linux-x86/libEchoesFrontend.so: ${OBJECTFILES}
	${MKDIR} -p dist/Debug/GNU-Linux-x86
	${LINK.cc} -shared -o ${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/libEchoesFrontend.so -fPIC ${OBJECTFILES} ${LDLIBSOPTIONS} 

${OBJECTDIR}/CudaRt_internal.o: nbproject/Makefile-${CND_CONF}.mk CudaRt_internal.cpp 
	${MKDIR} -p ${OBJECTDIR}
	${RM} $@.d
	$(COMPILE.cc) -g -I../EchoesUtil -fPIC  -MMD -MP -MF $@.d -o ${OBJECTDIR}/CudaRt_internal.o CudaRt_internal.cpp

${OBJECTDIR}/CudaRt_device.o: nbproject/Makefile-${CND_CONF}.mk CudaRt_device.cpp 
	${MKDIR} -p ${OBJECTDIR}
	${RM} $@.d
	$(COMPILE.cc) -g -I../EchoesUtil -fPIC  -MMD -MP -MF $@.d -o ${OBJECTDIR}/CudaRt_device.o CudaRt_device.cpp

${OBJECTDIR}/CudaRt_texture.o: nbproject/Makefile-${CND_CONF}.mk CudaRt_texture.cpp 
	${MKDIR} -p ${OBJECTDIR}
	${RM} $@.d
	$(COMPILE.cc) -g -I../EchoesUtil -fPIC  -MMD -MP -MF $@.d -o ${OBJECTDIR}/CudaRt_texture.o CudaRt_texture.cpp

${OBJECTDIR}/CudaRt.o: nbproject/Makefile-${CND_CONF}.mk CudaRt.cpp 
	${MKDIR} -p ${OBJECTDIR}
	${RM} $@.d
	$(COMPILE.cc) -g -I../EchoesUtil -fPIC  -MMD -MP -MF $@.d -o ${OBJECTDIR}/CudaRt.o CudaRt.cpp

${OBJECTDIR}/CudaRt_opengl.o: nbproject/Makefile-${CND_CONF}.mk CudaRt_opengl.cpp 
	${MKDIR} -p ${OBJECTDIR}
	${RM} $@.d
	$(COMPILE.cc) -g -I../EchoesUtil -fPIC  -MMD -MP -MF $@.d -o ${OBJECTDIR}/CudaRt_opengl.o CudaRt_opengl.cpp

${OBJECTDIR}/CudaRt_thread.o: nbproject/Makefile-${CND_CONF}.mk CudaRt_thread.cpp 
	${MKDIR} -p ${OBJECTDIR}
	${RM} $@.d
	$(COMPILE.cc) -g -I../EchoesUtil -fPIC  -MMD -MP -MF $@.d -o ${OBJECTDIR}/CudaRt_thread.o CudaRt_thread.cpp

${OBJECTDIR}/Frontend.o: nbproject/Makefile-${CND_CONF}.mk Frontend.cpp 
	${MKDIR} -p ${OBJECTDIR}
	${RM} $@.d
	$(COMPILE.cc) -g -I../EchoesUtil -fPIC  -MMD -MP -MF $@.d -o ${OBJECTDIR}/Frontend.o Frontend.cpp

${OBJECTDIR}/CudaRt_memory.o: nbproject/Makefile-${CND_CONF}.mk CudaRt_memory.cpp 
	${MKDIR} -p ${OBJECTDIR}
	${RM} $@.d
	$(COMPILE.cc) -g -I../EchoesUtil -fPIC  -MMD -MP -MF $@.d -o ${OBJECTDIR}/CudaRt_memory.o CudaRt_memory.cpp

${OBJECTDIR}/CudaRt_event.o: nbproject/Makefile-${CND_CONF}.mk CudaRt_event.cpp 
	${MKDIR} -p ${OBJECTDIR}
	${RM} $@.d
	$(COMPILE.cc) -g -I../EchoesUtil -fPIC  -MMD -MP -MF $@.d -o ${OBJECTDIR}/CudaRt_event.o CudaRt_event.cpp

${OBJECTDIR}/CudaRt_stream.o: nbproject/Makefile-${CND_CONF}.mk CudaRt_stream.cpp 
	${MKDIR} -p ${OBJECTDIR}
	${RM} $@.d
	$(COMPILE.cc) -g -I../EchoesUtil -fPIC  -MMD -MP -MF $@.d -o ${OBJECTDIR}/CudaRt_stream.o CudaRt_stream.cpp

${OBJECTDIR}/CudaRt_error.o: nbproject/Makefile-${CND_CONF}.mk CudaRt_error.cpp 
	${MKDIR} -p ${OBJECTDIR}
	${RM} $@.d
	$(COMPILE.cc) -g -I../EchoesUtil -fPIC  -MMD -MP -MF $@.d -o ${OBJECTDIR}/CudaRt_error.o CudaRt_error.cpp

${OBJECTDIR}/_ext/home/cjg/NetBeansProjects/EchoesFrontend/CudaRt_execution.o: nbproject/Makefile-${CND_CONF}.mk /home/cjg/NetBeansProjects/EchoesFrontend/CudaRt_execution.cpp 
	${MKDIR} -p ${OBJECTDIR}/_ext/home/cjg/NetBeansProjects/EchoesFrontend
	${RM} $@.d
	$(COMPILE.cc) -g -I../EchoesUtil -fPIC  -MMD -MP -MF $@.d -o ${OBJECTDIR}/_ext/home/cjg/NetBeansProjects/EchoesFrontend/CudaRt_execution.o /home/cjg/NetBeansProjects/EchoesFrontend/CudaRt_execution.cpp

${OBJECTDIR}/CudaRt_version.o: nbproject/Makefile-${CND_CONF}.mk CudaRt_version.cpp 
	${MKDIR} -p ${OBJECTDIR}
	${RM} $@.d
	$(COMPILE.cc) -g -I../EchoesUtil -fPIC  -MMD -MP -MF $@.d -o ${OBJECTDIR}/CudaRt_version.o CudaRt_version.cpp

# Subprojects
.build-subprojects:
	cd ../EchoesUtil && ${MAKE}  -f Makefile CONF=Debug

# Clean Targets
.clean-conf: ${CLEAN_SUBPROJECTS}
	${RM} -r build/Debug
	${RM} dist/Debug/GNU-Linux-x86/libEchoesFrontend.so

# Subprojects
.clean-subprojects:
	cd ../EchoesUtil && ${MAKE}  -f Makefile CONF=Debug clean

# Enable dependency checking
.dep.inc: .depcheck-impl

include .dep.inc
