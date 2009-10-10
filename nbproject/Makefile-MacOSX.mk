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
CND_CONF=MacOSX
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
LDLIBSOPTIONS=../EchoesUtil/dist/MacOSX/GNU-MacOSX/libEchoesUtil.dylib

# Build Targets
.build-conf: ${BUILD_SUBPROJECTS}
	${MAKE}  -f nbproject/Makefile-MacOSX.mk dist/MacOSX/GNU-Linux-x86/libEchoesFrontend.so

dist/MacOSX/GNU-Linux-x86/libEchoesFrontend.so: ../EchoesUtil/dist/MacOSX/GNU-MacOSX/libEchoesUtil.dylib

dist/MacOSX/GNU-Linux-x86/libEchoesFrontend.so: ${OBJECTFILES}
	${MKDIR} -p dist/MacOSX/GNU-Linux-x86
	${LINK.cc} -shared -o ${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/libEchoesFrontend.so -fPIC ${OBJECTFILES} ${LDLIBSOPTIONS} 

${OBJECTDIR}/CudaRt.o: nbproject/Makefile-${CND_CONF}.mk CudaRt.cpp 
	${MKDIR} -p ${OBJECTDIR}
	${RM} $@.d
	$(COMPILE.cc) -g -Wall -D_CONFIG_FILE=\"/Users/cjg/echoes.xml\" -I../EchoesUtil -fPIC  -MMD -MP -MF $@.d -o ${OBJECTDIR}/CudaRt.o CudaRt.cpp

${OBJECTDIR}/Frontend.o: nbproject/Makefile-${CND_CONF}.mk Frontend.cpp 
	${MKDIR} -p ${OBJECTDIR}
	${RM} $@.d
	$(COMPILE.cc) -g -Wall -D_CONFIG_FILE=\"/Users/cjg/echoes.xml\" -I../EchoesUtil -fPIC  -MMD -MP -MF $@.d -o ${OBJECTDIR}/Frontend.o Frontend.cpp

# Subprojects
.build-subprojects:
	cd ../EchoesUtil && ${MAKE}  -f Makefile CONF=MacOSX

# Clean Targets
.clean-conf: ${CLEAN_SUBPROJECTS}
	${RM} -r build/MacOSX
	${RM} dist/MacOSX/GNU-Linux-x86/libEchoesFrontend.so

# Subprojects
.clean-subprojects:
	cd ../EchoesUtil && ${MAKE}  -f Makefile CONF=MacOSX clean

# Enable dependency checking
.dep.inc: .depcheck-impl

include .dep.inc
