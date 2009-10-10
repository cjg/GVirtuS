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
CND_CONF=Linux_x86_64
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
LDLIBSOPTIONS=-Wl,-rpath ../EchoesUtils/dist/Linux_x86_64/GNU-Linux-x86 -L../EchoesUtils/dist/Linux_x86_64/GNU-Linux-x86 -lEchoesUtils

# Build Targets
.build-conf: ${BUILD_SUBPROJECTS}
	${MAKE}  -f nbproject/Makefile-Linux_x86_64.mk dist/Linux_x86_64/GNU-Linux-x86/libEchoesFrontend.so

dist/Linux_x86_64/GNU-Linux-x86/libEchoesFrontend.so: ../EchoesUtils/dist/Linux_x86_64/GNU-Linux-x86/libEchoesUtils.so

dist/Linux_x86_64/GNU-Linux-x86/libEchoesFrontend.so: ${OBJECTFILES}
	${MKDIR} -p dist/Linux_x86_64/GNU-Linux-x86
	${LINK.cc} -shared -o ${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/libEchoesFrontend.so -fPIC ${OBJECTFILES} ${LDLIBSOPTIONS} 

${OBJECTDIR}/CudaRt.o: nbproject/Makefile-${CND_CONF}.mk CudaRt.cpp 
	${MKDIR} -p ${OBJECTDIR}
	${RM} $@.d
	$(COMPILE.cc) -g -D_CONFIG_FILE=\"/home/cjg/echoes.xml\" -I/opt/cuda/include -I../EchoesUtils -fPIC  -MMD -MP -MF $@.d -o ${OBJECTDIR}/CudaRt.o CudaRt.cpp

${OBJECTDIR}/Frontend.o: nbproject/Makefile-${CND_CONF}.mk Frontend.cpp 
	${MKDIR} -p ${OBJECTDIR}
	${RM} $@.d
	$(COMPILE.cc) -g -D_CONFIG_FILE=\"/home/cjg/echoes.xml\" -I/opt/cuda/include -I../EchoesUtils -fPIC  -MMD -MP -MF $@.d -o ${OBJECTDIR}/Frontend.o Frontend.cpp

# Subprojects
.build-subprojects:
	cd ../EchoesUtils && ${MAKE}  -f Makefile CONF=Linux_x86_64

# Clean Targets
.clean-conf: ${CLEAN_SUBPROJECTS}
	${RM} -r build/Linux_x86_64
	${RM} dist/Linux_x86_64/GNU-Linux-x86/libEchoesFrontend.so

# Subprojects
.clean-subprojects:
	cd ../EchoesUtils && ${MAKE}  -f Makefile CONF=Linux_x86_64 clean

# Enable dependency checking
.dep.inc: .depcheck-impl

include .dep.inc
