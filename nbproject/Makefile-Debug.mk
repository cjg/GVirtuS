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
	${OBJECTDIR}/CudaRt.o \
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
LDLIBSOPTIONS=-Wl,-rpath ../EchoesUtil/dist/Debug/GNU-Linux-x86 -L../EchoesUtil/dist/Debug/GNU-Linux-x86 -lEchoesUtil

# Build Targets
.build-conf: ${BUILD_SUBPROJECTS}
	${MAKE}  -f nbproject/Makefile-Debug.mk dist/Debug/GNU-Linux-x86/libEchoesFrontend.so

dist/Debug/GNU-Linux-x86/libEchoesFrontend.so: ../EchoesUtil/dist/Debug/GNU-Linux-x86/libEchoesUtil.so

dist/Debug/GNU-Linux-x86/libEchoesFrontend.so: ${OBJECTFILES}
	${MKDIR} -p dist/Debug/GNU-Linux-x86
	${LINK.cc} -shared -o ${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/libEchoesFrontend.so -fPIC ${OBJECTFILES} ${LDLIBSOPTIONS} 

${OBJECTDIR}/CudaRt.o: nbproject/Makefile-${CND_CONF}.mk CudaRt.cpp 
	${MKDIR} -p ${OBJECTDIR}
	${RM} $@.d
	$(COMPILE.cc) -g -I../EchoesUtil -fPIC  -MMD -MP -MF $@.d -o ${OBJECTDIR}/CudaRt.o CudaRt.cpp

${OBJECTDIR}/Frontend.o: nbproject/Makefile-${CND_CONF}.mk Frontend.cpp 
	${MKDIR} -p ${OBJECTDIR}
	${RM} $@.d
	$(COMPILE.cc) -g -I../EchoesUtil -fPIC  -MMD -MP -MF $@.d -o ${OBJECTDIR}/Frontend.o Frontend.cpp

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
