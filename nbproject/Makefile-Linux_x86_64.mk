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
	${OBJECTDIR}/Thread.o \
	${OBJECTDIR}/_ext/home/cjg/NetBeansProjects/EchoesUtil/VmciCommunicator.o \
	${OBJECTDIR}/Mutex.o \
	${OBJECTDIR}/_ext/home/cjg/NetBeansProjects/EchoesUtil/Buffer.o \
	${OBJECTDIR}/TcpCommunicator.o \
	${OBJECTDIR}/Observable.o \
	${OBJECTDIR}/AfUnixCommunicator.o \
	${OBJECTDIR}/Communicator.o \
	${OBJECTDIR}/Observer.o \
	${OBJECTDIR}/VMSocketCommunicator.o \
	${OBJECTDIR}/ConfigFile.o \
	${OBJECTDIR}/CudaUtil.o \
	${OBJECTDIR}/_ext/home/cjg/NetBeansProjects/EchoesUtil/Result.o

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
	${MAKE}  -f nbproject/Makefile-Linux_x86_64.mk dist/Linux_x86_64/GNU-Linux-x86/libEchoesUtil.so

dist/Linux_x86_64/GNU-Linux-x86/libEchoesUtil.so: ${OBJECTFILES}
	${MKDIR} -p dist/Linux_x86_64/GNU-Linux-x86
	${LINK.cc} -shared -o ${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/libEchoesUtil.so -fPIC ${OBJECTFILES} ${LDLIBSOPTIONS} 

${OBJECTDIR}/Thread.o: nbproject/Makefile-${CND_CONF}.mk Thread.cpp 
	${MKDIR} -p ${OBJECTDIR}
	${RM} $@.d
	$(COMPILE.cc) -g -Wall -I/opt/cuda/include -I/usr/lib/vmware/include -fPIC  -MMD -MP -MF $@.d -o ${OBJECTDIR}/Thread.o Thread.cpp

${OBJECTDIR}/_ext/home/cjg/NetBeansProjects/EchoesUtil/VmciCommunicator.o: nbproject/Makefile-${CND_CONF}.mk /home/cjg/NetBeansProjects/EchoesUtil/VmciCommunicator.cpp 
	${MKDIR} -p ${OBJECTDIR}/_ext/home/cjg/NetBeansProjects/EchoesUtil
	${RM} $@.d
	$(COMPILE.cc) -g -Wall -I/opt/cuda/include -I/usr/lib/vmware/include -fPIC  -MMD -MP -MF $@.d -o ${OBJECTDIR}/_ext/home/cjg/NetBeansProjects/EchoesUtil/VmciCommunicator.o /home/cjg/NetBeansProjects/EchoesUtil/VmciCommunicator.cpp

${OBJECTDIR}/Mutex.o: nbproject/Makefile-${CND_CONF}.mk Mutex.cpp 
	${MKDIR} -p ${OBJECTDIR}
	${RM} $@.d
	$(COMPILE.cc) -g -Wall -I/opt/cuda/include -I/usr/lib/vmware/include -fPIC  -MMD -MP -MF $@.d -o ${OBJECTDIR}/Mutex.o Mutex.cpp

${OBJECTDIR}/_ext/home/cjg/NetBeansProjects/EchoesUtil/Buffer.o: nbproject/Makefile-${CND_CONF}.mk /home/cjg/NetBeansProjects/EchoesUtil/Buffer.cpp 
	${MKDIR} -p ${OBJECTDIR}/_ext/home/cjg/NetBeansProjects/EchoesUtil
	${RM} $@.d
	$(COMPILE.cc) -g -Wall -I/opt/cuda/include -I/usr/lib/vmware/include -fPIC  -MMD -MP -MF $@.d -o ${OBJECTDIR}/_ext/home/cjg/NetBeansProjects/EchoesUtil/Buffer.o /home/cjg/NetBeansProjects/EchoesUtil/Buffer.cpp

${OBJECTDIR}/TcpCommunicator.o: nbproject/Makefile-${CND_CONF}.mk TcpCommunicator.cpp 
	${MKDIR} -p ${OBJECTDIR}
	${RM} $@.d
	$(COMPILE.cc) -g -Wall -I/opt/cuda/include -I/usr/lib/vmware/include -fPIC  -MMD -MP -MF $@.d -o ${OBJECTDIR}/TcpCommunicator.o TcpCommunicator.cpp

${OBJECTDIR}/Observable.o: nbproject/Makefile-${CND_CONF}.mk Observable.cpp 
	${MKDIR} -p ${OBJECTDIR}
	${RM} $@.d
	$(COMPILE.cc) -g -Wall -I/opt/cuda/include -I/usr/lib/vmware/include -fPIC  -MMD -MP -MF $@.d -o ${OBJECTDIR}/Observable.o Observable.cpp

${OBJECTDIR}/AfUnixCommunicator.o: nbproject/Makefile-${CND_CONF}.mk AfUnixCommunicator.cpp 
	${MKDIR} -p ${OBJECTDIR}
	${RM} $@.d
	$(COMPILE.cc) -g -Wall -I/opt/cuda/include -I/usr/lib/vmware/include -fPIC  -MMD -MP -MF $@.d -o ${OBJECTDIR}/AfUnixCommunicator.o AfUnixCommunicator.cpp

${OBJECTDIR}/Communicator.o: nbproject/Makefile-${CND_CONF}.mk Communicator.cpp 
	${MKDIR} -p ${OBJECTDIR}
	${RM} $@.d
	$(COMPILE.cc) -g -Wall -I/opt/cuda/include -I/usr/lib/vmware/include -fPIC  -MMD -MP -MF $@.d -o ${OBJECTDIR}/Communicator.o Communicator.cpp

${OBJECTDIR}/Observer.o: nbproject/Makefile-${CND_CONF}.mk Observer.cpp 
	${MKDIR} -p ${OBJECTDIR}
	${RM} $@.d
	$(COMPILE.cc) -g -Wall -I/opt/cuda/include -I/usr/lib/vmware/include -fPIC  -MMD -MP -MF $@.d -o ${OBJECTDIR}/Observer.o Observer.cpp

${OBJECTDIR}/VMSocketCommunicator.o: nbproject/Makefile-${CND_CONF}.mk VMSocketCommunicator.cpp 
	${MKDIR} -p ${OBJECTDIR}
	${RM} $@.d
	$(COMPILE.cc) -g -Wall -I/opt/cuda/include -I/usr/lib/vmware/include -fPIC  -MMD -MP -MF $@.d -o ${OBJECTDIR}/VMSocketCommunicator.o VMSocketCommunicator.cpp

${OBJECTDIR}/ConfigFile.o: nbproject/Makefile-${CND_CONF}.mk ConfigFile.cpp 
	${MKDIR} -p ${OBJECTDIR}
	${RM} $@.d
	$(COMPILE.cc) -g -Wall -I/opt/cuda/include -I/usr/lib/vmware/include -fPIC  -MMD -MP -MF $@.d -o ${OBJECTDIR}/ConfigFile.o ConfigFile.cpp

${OBJECTDIR}/CudaUtil.o: nbproject/Makefile-${CND_CONF}.mk CudaUtil.cpp 
	${MKDIR} -p ${OBJECTDIR}
	${RM} $@.d
	$(COMPILE.cc) -g -Wall -I/opt/cuda/include -I/usr/lib/vmware/include -fPIC  -MMD -MP -MF $@.d -o ${OBJECTDIR}/CudaUtil.o CudaUtil.cpp

${OBJECTDIR}/_ext/home/cjg/NetBeansProjects/EchoesUtil/Result.o: nbproject/Makefile-${CND_CONF}.mk /home/cjg/NetBeansProjects/EchoesUtil/Result.cpp 
	${MKDIR} -p ${OBJECTDIR}/_ext/home/cjg/NetBeansProjects/EchoesUtil
	${RM} $@.d
	$(COMPILE.cc) -g -Wall -I/opt/cuda/include -I/usr/lib/vmware/include -fPIC  -MMD -MP -MF $@.d -o ${OBJECTDIR}/_ext/home/cjg/NetBeansProjects/EchoesUtil/Result.o /home/cjg/NetBeansProjects/EchoesUtil/Result.cpp

# Subprojects
.build-subprojects:

# Clean Targets
.clean-conf:
	${RM} -r build/Linux_x86_64
	${RM} dist/Linux_x86_64/GNU-Linux-x86/libEchoesUtil.so

# Subprojects
.clean-subprojects:

# Enable dependency checking
.dep.inc: .depcheck-impl

include .dep.inc
