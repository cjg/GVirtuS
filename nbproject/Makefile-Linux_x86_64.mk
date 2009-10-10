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
LDLIBSOPTIONS=-L/opt/cuda/lib -Wl,-rpath ../EchoesUtils/dist/Linux_x86_64/GNU-Linux-x86 -L../EchoesUtils/dist/Linux_x86_64/GNU-Linux-x86 -lEchoesUtils -lcudart

# Build Targets
.build-conf: ${BUILD_SUBPROJECTS}
	${MAKE}  -f nbproject/Makefile-Linux_x86_64.mk dist/Linux_x86_64/GNU-Linux-x86/echoesbackend

dist/Linux_x86_64/GNU-Linux-x86/echoesbackend: ../EchoesUtils/dist/Linux_x86_64/GNU-Linux-x86/libEchoesUtils.so

dist/Linux_x86_64/GNU-Linux-x86/echoesbackend: ${OBJECTFILES}
	${MKDIR} -p dist/Linux_x86_64/GNU-Linux-x86
	${LINK.cc} -o ${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/echoesbackend ${OBJECTFILES} ${LDLIBSOPTIONS} 

${OBJECTDIR}/CudaRtHandler.o: nbproject/Makefile-${CND_CONF}.mk CudaRtHandler.cpp 
	${MKDIR} -p ${OBJECTDIR}
	${RM} $@.d
	$(COMPILE.cc) -g -I../EchoesUtils -I/opt/cuda/include -MMD -MP -MF $@.d -o ${OBJECTDIR}/CudaRtHandler.o CudaRtHandler.cpp

${OBJECTDIR}/Backend.o: nbproject/Makefile-${CND_CONF}.mk Backend.cpp 
	${MKDIR} -p ${OBJECTDIR}
	${RM} $@.d
	$(COMPILE.cc) -g -I../EchoesUtils -I/opt/cuda/include -MMD -MP -MF $@.d -o ${OBJECTDIR}/Backend.o Backend.cpp

${OBJECTDIR}/Process.o: nbproject/Makefile-${CND_CONF}.mk Process.cpp 
	${MKDIR} -p ${OBJECTDIR}
	${RM} $@.d
	$(COMPILE.cc) -g -I../EchoesUtils -I/opt/cuda/include -MMD -MP -MF $@.d -o ${OBJECTDIR}/Process.o Process.cpp

${OBJECTDIR}/main.o: nbproject/Makefile-${CND_CONF}.mk main.cpp 
	${MKDIR} -p ${OBJECTDIR}
	${RM} $@.d
	$(COMPILE.cc) -g -I../EchoesUtils -I/opt/cuda/include -MMD -MP -MF $@.d -o ${OBJECTDIR}/main.o main.cpp

# Subprojects
.build-subprojects:
	cd ../EchoesUtils && ${MAKE}  -f Makefile CONF=Linux_x86_64

# Clean Targets
.clean-conf: ${CLEAN_SUBPROJECTS}
	${RM} -r build/Linux_x86_64
	${RM} dist/Linux_x86_64/GNU-Linux-x86/echoesbackend

# Subprojects
.clean-subprojects:
	cd ../EchoesUtils && ${MAKE}  -f Makefile CONF=Linux_x86_64 clean

# Enable dependency checking
.dep.inc: .depcheck-impl

include .dep.inc
