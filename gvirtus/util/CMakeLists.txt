cmake_minimum_required(VERSION 3.10.2)
project(util)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -fPIC")
set(CMAKE_CXX_STANDARD 17)

include_directories(SYSTEM ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(util_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/include/gvirtus-type.h
                          ${CMAKE_CURRENT_SOURCE_DIR}/src/AfUnixCommunicator.cpp
                          ${CMAKE_CURRENT_SOURCE_DIR}/include/AfUnixCommunicator.h
                          ${CMAKE_CURRENT_SOURCE_DIR}/src/Buffer.cpp
                          ${CMAKE_CURRENT_SOURCE_DIR}/include/Buffer.h
                          ${CMAKE_CURRENT_SOURCE_DIR}/src/Communicator.cpp
                          ${CMAKE_CURRENT_SOURCE_DIR}/include/Communicator.h
                          ${CMAKE_CURRENT_SOURCE_DIR}/src/ConfigFile.cpp
                          ${CMAKE_CURRENT_SOURCE_DIR}/include/ConfigFile.h
                          ${CMAKE_CURRENT_SOURCE_DIR}/src/MessageDispatcher.cpp
                          ${CMAKE_CURRENT_SOURCE_DIR}/include/MessageDispatcher.h
                          ${CMAKE_CURRENT_SOURCE_DIR}/src/Mutex.cpp
                          ${CMAKE_CURRENT_SOURCE_DIR}/include/Mutex.h
                          ${CMAKE_CURRENT_SOURCE_DIR}/src/Observable.cpp
                          ${CMAKE_CURRENT_SOURCE_DIR}/include/Observable.h
                          ${CMAKE_CURRENT_SOURCE_DIR}/src/Observer.cpp
                          ${CMAKE_CURRENT_SOURCE_DIR}/include/Observer.h
                          ${CMAKE_CURRENT_SOURCE_DIR}/src/Result.cpp
                          ${CMAKE_CURRENT_SOURCE_DIR}/include/Result.h
                          ${CMAKE_CURRENT_SOURCE_DIR}/src/ShmCommunicator.cpp
                          ${CMAKE_CURRENT_SOURCE_DIR}/include/ShmCommunicator.h
                          ${CMAKE_CURRENT_SOURCE_DIR}/include/Observer.h
                          ${CMAKE_CURRENT_SOURCE_DIR}/src/TcpCommunicator.cpp
                          ${CMAKE_CURRENT_SOURCE_DIR}/src/ZmqCommunicator.cpp
                          ${CMAKE_CURRENT_SOURCE_DIR}/include/ZmqCommunicator.h
                          ${CMAKE_CURRENT_SOURCE_DIR}/include/TcpCommunicator.h
                          ${CMAKE_CURRENT_SOURCE_DIR}/src/Thread.cpp
                          ${CMAKE_CURRENT_SOURCE_DIR}/include/Thread.h
                          ${CMAKE_CURRENT_SOURCE_DIR}/src/Util.cpp
                          ${CMAKE_CURRENT_SOURCE_DIR}/include/Util.h
                          ${CMAKE_CURRENT_SOURCE_DIR}/src/VMShmCommunicator.cpp
                          ${CMAKE_CURRENT_SOURCE_DIR}/include/VMShmCommunicator.h
                          ${CMAKE_CURRENT_SOURCE_DIR}/src/VMSocketCommunicator.cpp
                          ${CMAKE_CURRENT_SOURCE_DIR}/include/VMSocketCommunicator.h
                          ${CMAKE_CURRENT_SOURCE_DIR}/src/VirtioCommunicator.cpp
                          ${CMAKE_CURRENT_SOURCE_DIR}/include/VirtioCommunicator.h
                          # ${CMAKE_CURRENT_SOURCE_DIR}/VmciCommunicator.cpp
                          # ${CMAKE_CURRENT_SOURCE_DIR}/VmciCommunicator.h
                          ${CMAKE_CURRENT_SOURCE_DIR}/src/Decoder.cpp
                          ${CMAKE_CURRENT_SOURCE_DIR}/include/Encoder.h
                          ${CMAKE_CURRENT_SOURCE_DIR}/src/Encoder.cpp
                          ${CMAKE_CURRENT_SOURCE_DIR}/include/Decoder.h
)

file(GLOB UTIL_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/include/*.h*")
file(COPY ${UTIL_HEADERS} DESTINATION ${CMAKE_INSTALL_PREFIX}/include/gvirtus)
add_library(${PROJECT_NAME} STATIC ${util_SOURCE})
