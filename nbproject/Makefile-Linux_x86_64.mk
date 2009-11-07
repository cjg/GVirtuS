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
CND_CONF=Linux_x86_64
CND_DISTDIR=dist

# Include project Makefile
include Makefile

# Object Directory
OBJECTDIR=build/${CND_CONF}/${CND_PLATFORM}

# Object Files
OBJECTFILES= \
	${OBJECTDIR}/CudaRt_memory.o \
	${OBJECTDIR}/_ext/home/cjg/NetBeansProjects/EchoesFrontend/CudaRt_execution.o \
	${OBJECTDIR}/CudaRt_error.o \
	${OBJECTDIR}/CudaRt_device.o \
	${OBJECTDIR}/CudaRt.o \
	${OBJECTDIR}/CudaRt_thread.o \
	${OBJECTDIR}/Frontend.o

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
LDLIBSOPTIONS=-Wl,-rpath ../EchoesUtil/dist/Linux_x86_64/GNU-Linux-x86 -L../EchoesUtil/dist/Linux_x86_64/GNU-Linux-x86 -lEchoesUtil

# Build Targets
.build-conf: ${BUILD_SUBPROJECTS}
	${MAKE}  -f nbproject/Makefile-Linux_x86_64.mk dist/Linux_x86_64/GNU-Linux-x86/libEchoesFrontend.so

dist/Linux_x86_64/GNU-Linux-x86/libEchoesFrontend.so: ../EchoesUtil/dist/Linux_x86_64/GNU-Linux-x86/libEchoesUtil.so

dist/Linux_x86_64/GNU-Linux-x86/libEchoesFrontend.so: ${OBJECTFILES}
	${MKDIR} -p dist/Linux_x86_64/GNU-Linux-x86
	${LINK.cc} -shared -o ${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/libEchoesFrontend.so -fPIC ${OBJECTFILES} ${LDLIBSOPTIONS} 

${OBJECTDIR}/CudaRt_memory.o: nbproject/Makefile-${CND_CONF}.mk CudaRt_memory.cpp 
	${MKDIR} -p ${OBJECTDIR}
	${RM} $@.d
	$(COMPILE.cc) -g -Wall -D_CONFIG_FILE=\"/home/cjg/echoes.xml\" -I/opt/cuda/include -I../EchoesUtil -fPIC  -MMD -MP -MF $@.d -o ${OBJECTDIR}/CudaRt_memory.o CudaRt_memory.cpp

${OBJECTDIR}/_ext/home/cjg/NetBeansProjects/EchoesFrontend/CudaRt_execution.o: nbproject/Makefile-${CND_CONF}.mk /home/cjg/NetBeansProjects/EchoesFrontend/CudaRt_execution.cpp 
	${MKDIR} -p ${OBJECTDIR}/_ext/home/cjg/NetBeansProjects/EchoesFrontend
	${RM} $@.d
	$(COMPILE.cc) -g -Wall -D_CONFIG_FILE=\"/home/cjg/echoes.xml\" -I/opt/cuda/include -I../EchoesUtil -fPIC  -MMD -MP -MF $@.d -o ${OBJECTDIR}/_ext/home/cjg/NetBeansProjects/EchoesFrontend/CudaRt_execution.o /home/cjg/NetBeansProjects/EchoesFrontend/CudaRt_execution.cpp

${OBJECTDIR}/CudaRt_error.o: nbproject/Makefile-${CND_CONF}.mk CudaRt_error.cpp 
	${MKDIR} -p ${OBJECTDIR}
	${RM} $@.d
	$(COMPILE.cc) -g -Wall -D_CONFIG_FILE=\"/home/cjg/echoes.xml\" -I/opt/cuda/include -I../EchoesUtil -fPIC  -MMD -MP -MF $@.d -o ${OBJECTDIR}/CudaRt_error.o CudaRt_error.cpp

${OBJECTDIR}/CudaRt_device.o: nbproject/Makefile-${CND_CONF}.mk CudaRt_device.cpp 
	${MKDIR} -p ${OBJECTDIR}
	${RM} $@.d
	$(COMPILE.cc) -g -Wall -D_CONFIG_FILE=\"/home/cjg/echoes.xml\" -I/opt/cuda/include -I../EchoesUtil -fPIC  -MMD -MP -MF $@.d -o ${OBJECTDIR}/CudaRt_device.o CudaRt_device.cpp

${OBJECTDIR}/CudaRt.o: nbproject/Makefile-${CND_CONF}.mk CudaRt.cpp 
	${MKDIR} -p ${OBJECTDIR}
	${RM} $@.d
	$(COMPILE.cc) -g -Wall -D_CONFIG_FILE=\"/home/cjg/echoes.xml\" -I/opt/cuda/include -I../EchoesUtil -fPIC  -MMD -MP -MF $@.d -o ${OBJECTDIR}/CudaRt.o CudaRt.cpp

${OBJECTDIR}/CudaRt_thread.o: nbproject/Makefile-${CND_CONF}.mk CudaRt_thread.cpp 
	${MKDIR} -p ${OBJECTDIR}
	${RM} $@.d
	$(COMPILE.cc) -g -Wall -D_CONFIG_FILE=\"/home/cjg/echoes.xml\" -I/opt/cuda/include -I../EchoesUtil -fPIC  -MMD -MP -MF $@.d -o ${OBJECTDIR}/CudaRt_thread.o CudaRt_thread.cpp

${OBJECTDIR}/Frontend.o: nbproject/Makefile-${CND_CONF}.mk Frontend.cpp 
	${MKDIR} -p ${OBJECTDIR}
	${RM} $@.d
	$(COMPILE.cc) -g -Wall -D_CONFIG_FILE=\"/home/cjg/echoes.xml\" -I/opt/cuda/include -I../EchoesUtil -fPIC  -MMD -MP -MF $@.d -o ${OBJECTDIR}/Frontend.o Frontend.cpp

# Subprojects
.build-subprojects:
	cd ../EchoesUtil && ${MAKE}  -f Makefile CONF=Linux_x86_64

# Clean Targets
.clean-conf: ${CLEAN_SUBPROJECTS}
	${RM} -r build/Linux_x86_64
	${RM} dist/Linux_x86_64/GNU-Linux-x86/libEchoesFrontend.so

# Subprojects
.clean-subprojects:
	cd ../EchoesUtil && ${MAKE}  -f Makefile CONF=Linux_x86_64 clean

# Enable dependency checking
.dep.inc: .depcheck-impl

include .dep.inc
