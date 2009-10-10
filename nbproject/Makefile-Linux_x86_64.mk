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
FC=gfortran
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
	${OBJECTDIR}/Communicator.o \
	${OBJECTDIR}/Observer.o \
	${OBJECTDIR}/TcpCommunicator.o \
	${OBJECTDIR}/Thread.o \
	${OBJECTDIR}/ConfigFile.o \
	${OBJECTDIR}/Mutex.o \
	${OBJECTDIR}/Observable.o \
	${OBJECTDIR}/AfUnixCommunicator.o

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
LDLIBSOPTIONS=-lpthread -lexpat

# Build Targets
.build-conf: ${BUILD_SUBPROJECTS}
	${MAKE}  -f nbproject/Makefile-Linux_x86_64.mk dist/Linux_x86_64/GNU-Linux-x86/libEchoesUtils.so

dist/Linux_x86_64/GNU-Linux-x86/libEchoesUtils.so: ${OBJECTFILES}
	${MKDIR} -p dist/Linux_x86_64/GNU-Linux-x86
	${LINK.cc} -shared -o ${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/libEchoesUtils.so -fPIC ${OBJECTFILES} ${LDLIBSOPTIONS} 

${OBJECTDIR}/Communicator.o: nbproject/Makefile-${CND_CONF}.mk Communicator.cpp 
	${MKDIR} -p ${OBJECTDIR}
	${RM} $@.d
	$(COMPILE.cc) -g -fPIC  -MMD -MP -MF $@.d -o ${OBJECTDIR}/Communicator.o Communicator.cpp

${OBJECTDIR}/Observer.o: nbproject/Makefile-${CND_CONF}.mk Observer.cpp 
	${MKDIR} -p ${OBJECTDIR}
	${RM} $@.d
	$(COMPILE.cc) -g -fPIC  -MMD -MP -MF $@.d -o ${OBJECTDIR}/Observer.o Observer.cpp

${OBJECTDIR}/TcpCommunicator.o: nbproject/Makefile-${CND_CONF}.mk TcpCommunicator.cpp
	${MKDIR} -p ${OBJECTDIR}
	${RM} $@.d
	$(COMPILE.cc) -g -fPIC  -MMD -MP -MF $@.d -o ${OBJECTDIR}/TcpCommunicator.o TcpCommunicator.cpp

${OBJECTDIR}/Thread.o: nbproject/Makefile-${CND_CONF}.mk Thread.cpp 
	${MKDIR} -p ${OBJECTDIR}
	${RM} $@.d
	$(COMPILE.cc) -g -fPIC  -MMD -MP -MF $@.d -o ${OBJECTDIR}/Thread.o Thread.cpp

${OBJECTDIR}/ConfigFile.o: nbproject/Makefile-${CND_CONF}.mk ConfigFile.cpp 
	${MKDIR} -p ${OBJECTDIR}
	${RM} $@.d
	$(COMPILE.cc) -g -fPIC  -MMD -MP -MF $@.d -o ${OBJECTDIR}/ConfigFile.o ConfigFile.cpp

${OBJECTDIR}/Mutex.o: nbproject/Makefile-${CND_CONF}.mk Mutex.cpp 
	${MKDIR} -p ${OBJECTDIR}
	${RM} $@.d
	$(COMPILE.cc) -g -fPIC  -MMD -MP -MF $@.d -o ${OBJECTDIR}/Mutex.o Mutex.cpp

${OBJECTDIR}/Observable.o: nbproject/Makefile-${CND_CONF}.mk Observable.cpp 
	${MKDIR} -p ${OBJECTDIR}
	${RM} $@.d
	$(COMPILE.cc) -g -fPIC  -MMD -MP -MF $@.d -o ${OBJECTDIR}/Observable.o Observable.cpp

${OBJECTDIR}/AfUnixCommunicator.o: nbproject/Makefile-${CND_CONF}.mk AfUnixCommunicator.cpp 
	${MKDIR} -p ${OBJECTDIR}
	${RM} $@.d
	$(COMPILE.cc) -g -fPIC  -MMD -MP -MF $@.d -o ${OBJECTDIR}/AfUnixCommunicator.o AfUnixCommunicator.cpp

# Subprojects
.build-subprojects:

# Clean Targets
.clean-conf:
	${RM} -r build/Linux_x86_64
	${RM} dist/Linux_x86_64/GNU-Linux-x86/libEchoesUtils.so

# Subprojects
.clean-subprojects:

# Enable dependency checking
.dep.inc: .depcheck-impl

include .dep.inc
