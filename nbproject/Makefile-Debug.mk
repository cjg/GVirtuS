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
AS=as

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
	${OBJECTDIR}/CudaRtHandler.o \
	${OBJECTDIR}/_ext/home/cjg/NetBeansProjects/EchoesBackend/CudaRtHandler_memory.o \
	${OBJECTDIR}/Backend.o \
	${OBJECTDIR}/Process.o \
	${OBJECTDIR}/_ext/home/cjg/NetBeansProjects/EchoesBackend/CudaRtHandler_thread.o \
	${OBJECTDIR}/_ext/home/cjg/NetBeansProjects/EchoesBackend/CudaRtHandler_device.o \
	${OBJECTDIR}/main.o \
	${OBJECTDIR}/CudaRtHandler_version.o \
	${OBJECTDIR}/_ext/home/cjg/NetBeansProjects/EchoesBackend/CudaRtHandler_execution.o \
	${OBJECTDIR}/_ext/home/cjg/NetBeansProjects/EchoesBackend/CudaRtHandler_error.o \
	${OBJECTDIR}/_ext/home/cjg/NetBeansProjects/EchoesBackend/CudaRtHandler_internal.o

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
LDLIBSOPTIONS=-Wl,-rpath ../EchoesUtil/dist/Debug/GNU-Linux-x86 -L../EchoesUtil/dist/Debug/GNU-Linux-x86 -lEchoesUtil -lexpat -lpthread

# Build Targets
.build-conf: ${BUILD_SUBPROJECTS}
	${MAKE}  -f nbproject/Makefile-Debug.mk dist/Debug/GNU-Linux-x86/echoesbackend

dist/Debug/GNU-Linux-x86/echoesbackend: ../EchoesUtil/dist/Debug/GNU-Linux-x86/libEchoesUtil.so

dist/Debug/GNU-Linux-x86/echoesbackend: ${OBJECTFILES}
	${MKDIR} -p dist/Debug/GNU-Linux-x86
	${LINK.cc} -o ${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/echoesbackend ${OBJECTFILES} ${LDLIBSOPTIONS} 

${OBJECTDIR}/CudaRtHandler.o: nbproject/Makefile-${CND_CONF}.mk CudaRtHandler.cpp 
	${MKDIR} -p ${OBJECTDIR}
	${RM} $@.d
	$(COMPILE.cc) -g -I../EchoesUtil -MMD -MP -MF $@.d -o ${OBJECTDIR}/CudaRtHandler.o CudaRtHandler.cpp

${OBJECTDIR}/_ext/home/cjg/NetBeansProjects/EchoesBackend/CudaRtHandler_memory.o: nbproject/Makefile-${CND_CONF}.mk /home/cjg/NetBeansProjects/EchoesBackend/CudaRtHandler_memory.cpp 
	${MKDIR} -p ${OBJECTDIR}/_ext/home/cjg/NetBeansProjects/EchoesBackend
	${RM} $@.d
	$(COMPILE.cc) -g -I../EchoesUtil -MMD -MP -MF $@.d -o ${OBJECTDIR}/_ext/home/cjg/NetBeansProjects/EchoesBackend/CudaRtHandler_memory.o /home/cjg/NetBeansProjects/EchoesBackend/CudaRtHandler_memory.cpp

${OBJECTDIR}/Backend.o: nbproject/Makefile-${CND_CONF}.mk Backend.cpp 
	${MKDIR} -p ${OBJECTDIR}
	${RM} $@.d
	$(COMPILE.cc) -g -I../EchoesUtil -MMD -MP -MF $@.d -o ${OBJECTDIR}/Backend.o Backend.cpp

${OBJECTDIR}/Process.o: nbproject/Makefile-${CND_CONF}.mk Process.cpp 
	${MKDIR} -p ${OBJECTDIR}
	${RM} $@.d
	$(COMPILE.cc) -g -I../EchoesUtil -MMD -MP -MF $@.d -o ${OBJECTDIR}/Process.o Process.cpp

${OBJECTDIR}/_ext/home/cjg/NetBeansProjects/EchoesBackend/CudaRtHandler_thread.o: nbproject/Makefile-${CND_CONF}.mk /home/cjg/NetBeansProjects/EchoesBackend/CudaRtHandler_thread.cpp 
	${MKDIR} -p ${OBJECTDIR}/_ext/home/cjg/NetBeansProjects/EchoesBackend
	${RM} $@.d
	$(COMPILE.cc) -g -I../EchoesUtil -MMD -MP -MF $@.d -o ${OBJECTDIR}/_ext/home/cjg/NetBeansProjects/EchoesBackend/CudaRtHandler_thread.o /home/cjg/NetBeansProjects/EchoesBackend/CudaRtHandler_thread.cpp

${OBJECTDIR}/_ext/home/cjg/NetBeansProjects/EchoesBackend/CudaRtHandler_device.o: nbproject/Makefile-${CND_CONF}.mk /home/cjg/NetBeansProjects/EchoesBackend/CudaRtHandler_device.cpp 
	${MKDIR} -p ${OBJECTDIR}/_ext/home/cjg/NetBeansProjects/EchoesBackend
	${RM} $@.d
	$(COMPILE.cc) -g -I../EchoesUtil -MMD -MP -MF $@.d -o ${OBJECTDIR}/_ext/home/cjg/NetBeansProjects/EchoesBackend/CudaRtHandler_device.o /home/cjg/NetBeansProjects/EchoesBackend/CudaRtHandler_device.cpp

${OBJECTDIR}/main.o: nbproject/Makefile-${CND_CONF}.mk main.cpp 
	${MKDIR} -p ${OBJECTDIR}
	${RM} $@.d
	$(COMPILE.cc) -g -I../EchoesUtil -MMD -MP -MF $@.d -o ${OBJECTDIR}/main.o main.cpp

${OBJECTDIR}/CudaRtHandler_version.o: nbproject/Makefile-${CND_CONF}.mk CudaRtHandler_version.cpp 
	${MKDIR} -p ${OBJECTDIR}
	${RM} $@.d
	$(COMPILE.cc) -g -I../EchoesUtil -MMD -MP -MF $@.d -o ${OBJECTDIR}/CudaRtHandler_version.o CudaRtHandler_version.cpp

${OBJECTDIR}/_ext/home/cjg/NetBeansProjects/EchoesBackend/CudaRtHandler_execution.o: nbproject/Makefile-${CND_CONF}.mk /home/cjg/NetBeansProjects/EchoesBackend/CudaRtHandler_execution.cpp 
	${MKDIR} -p ${OBJECTDIR}/_ext/home/cjg/NetBeansProjects/EchoesBackend
	${RM} $@.d
	$(COMPILE.cc) -g -I../EchoesUtil -MMD -MP -MF $@.d -o ${OBJECTDIR}/_ext/home/cjg/NetBeansProjects/EchoesBackend/CudaRtHandler_execution.o /home/cjg/NetBeansProjects/EchoesBackend/CudaRtHandler_execution.cpp

${OBJECTDIR}/_ext/home/cjg/NetBeansProjects/EchoesBackend/CudaRtHandler_error.o: nbproject/Makefile-${CND_CONF}.mk /home/cjg/NetBeansProjects/EchoesBackend/CudaRtHandler_error.cpp 
	${MKDIR} -p ${OBJECTDIR}/_ext/home/cjg/NetBeansProjects/EchoesBackend
	${RM} $@.d
	$(COMPILE.cc) -g -I../EchoesUtil -MMD -MP -MF $@.d -o ${OBJECTDIR}/_ext/home/cjg/NetBeansProjects/EchoesBackend/CudaRtHandler_error.o /home/cjg/NetBeansProjects/EchoesBackend/CudaRtHandler_error.cpp

${OBJECTDIR}/_ext/home/cjg/NetBeansProjects/EchoesBackend/CudaRtHandler_internal.o: nbproject/Makefile-${CND_CONF}.mk /home/cjg/NetBeansProjects/EchoesBackend/CudaRtHandler_internal.cpp 
	${MKDIR} -p ${OBJECTDIR}/_ext/home/cjg/NetBeansProjects/EchoesBackend
	${RM} $@.d
	$(COMPILE.cc) -g -I../EchoesUtil -MMD -MP -MF $@.d -o ${OBJECTDIR}/_ext/home/cjg/NetBeansProjects/EchoesBackend/CudaRtHandler_internal.o /home/cjg/NetBeansProjects/EchoesBackend/CudaRtHandler_internal.cpp

# Subprojects
.build-subprojects:
	cd ../EchoesUtil && ${MAKE}  -f Makefile CONF=Debug
	cd ../EchoesUtil && ${MAKE}  -f Makefile CONF=Debug
	cd ../EchoesUtil && ${MAKE}  -f Makefile CONF=Release

# Clean Targets
.clean-conf: ${CLEAN_SUBPROJECTS}
	${RM} -r build/Debug
	${RM} dist/Debug/GNU-Linux-x86/echoesbackend

# Subprojects
.clean-subprojects:
	cd ../EchoesUtil && ${MAKE}  -f Makefile CONF=Debug clean
	cd ../EchoesUtil && ${MAKE}  -f Makefile CONF=Debug clean
	cd ../EchoesUtil && ${MAKE}  -f Makefile CONF=Release clean

# Enable dependency checking
.dep.inc: .depcheck-impl

include .dep.inc
