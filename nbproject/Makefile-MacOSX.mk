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
	${OBJECTDIR}/CudaRtHandler.o \
	${OBJECTDIR}/Backend.o \
	${OBJECTDIR}/Process.o \
	${OBJECTDIR}/main.o

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
	${MAKE}  -f nbproject/Makefile-MacOSX.mk dist/MacOSX/GNU-Linux-x86/echoesbackend

dist/MacOSX/GNU-Linux-x86/echoesbackend: ../EchoesUtil/dist/MacOSX/GNU-MacOSX/libEchoesUtil.dylib

dist/MacOSX/GNU-Linux-x86/echoesbackend: ${OBJECTFILES}
	${MKDIR} -p dist/MacOSX/GNU-Linux-x86
	${LINK.cc} -o ${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/echoesbackend ${OBJECTFILES} ${LDLIBSOPTIONS} 

${OBJECTDIR}/CudaRtHandler.o: nbproject/Makefile-${CND_CONF}.mk CudaRtHandler.cpp 
	${MKDIR} -p ${OBJECTDIR}
	${RM} $@.d
	$(COMPILE.cc) -g -Wall -I../EchoesUtil -MMD -MP -MF $@.d -o ${OBJECTDIR}/CudaRtHandler.o CudaRtHandler.cpp

${OBJECTDIR}/Backend.o: nbproject/Makefile-${CND_CONF}.mk Backend.cpp 
	${MKDIR} -p ${OBJECTDIR}
	${RM} $@.d
	$(COMPILE.cc) -g -Wall -I../EchoesUtil -MMD -MP -MF $@.d -o ${OBJECTDIR}/Backend.o Backend.cpp

${OBJECTDIR}/Process.o: nbproject/Makefile-${CND_CONF}.mk Process.cpp 
	${MKDIR} -p ${OBJECTDIR}
	${RM} $@.d
	$(COMPILE.cc) -g -Wall -I../EchoesUtil -MMD -MP -MF $@.d -o ${OBJECTDIR}/Process.o Process.cpp

${OBJECTDIR}/main.o: nbproject/Makefile-${CND_CONF}.mk main.cpp 
	${MKDIR} -p ${OBJECTDIR}
	${RM} $@.d
	$(COMPILE.cc) -g -Wall -I../EchoesUtil -MMD -MP -MF $@.d -o ${OBJECTDIR}/main.o main.cpp

# Subprojects
.build-subprojects:
	cd ../EchoesUtil && ${MAKE}  -f Makefile CONF=MacOSX

# Clean Targets
.clean-conf: ${CLEAN_SUBPROJECTS}
	${RM} -r build/MacOSX
	${RM} dist/MacOSX/GNU-Linux-x86/echoesbackend

# Subprojects
.clean-subprojects:
	cd ../EchoesUtil && ${MAKE}  -f Makefile CONF=MacOSX clean

# Enable dependency checking
.dep.inc: .depcheck-impl

include .dep.inc
