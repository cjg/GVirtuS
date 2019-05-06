cmake_minimum_required(VERSION 3.10.2)
project(communicator)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -fPIC")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(SYSTEM ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(communicator_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/src/AfUnixCommunicator.cpp
                          ${CMAKE_CURRENT_SOURCE_DIR}/include/AfUnixCommunicator.h
                          ${CMAKE_CURRENT_SOURCE_DIR}/include/Communicator.h
                          ${CMAKE_CURRENT_SOURCE_DIR}/include/CommunicatorFactory.h
                          ${CMAKE_CURRENT_SOURCE_DIR}/src/ShmCommunicator.cpp
                          ${CMAKE_CURRENT_SOURCE_DIR}/include/ShmCommunicator.h
                          ${CMAKE_CURRENT_SOURCE_DIR}/src/Buffer.cpp
                          ${CMAKE_CURRENT_SOURCE_DIR}/include/Buffer.h
                          ${CMAKE_CURRENT_SOURCE_DIR}/src/Result.cpp
                          ${CMAKE_CURRENT_SOURCE_DIR}/include/Result.h
                          ${CMAKE_CURRENT_SOURCE_DIR}/src/TcpCommunicator.cpp
                          ${CMAKE_CURRENT_SOURCE_DIR}/src/ZmqCommunicator.cpp
                          ${CMAKE_CURRENT_SOURCE_DIR}/include/ZmqCommunicator.h
                          ${CMAKE_CURRENT_SOURCE_DIR}/include/TcpCommunicator.h
                          ${CMAKE_CURRENT_SOURCE_DIR}/src/VMShmCommunicator.cpp
                          ${CMAKE_CURRENT_SOURCE_DIR}/include/VMShmCommunicator.h
                          ${CMAKE_CURRENT_SOURCE_DIR}/src/VMSocketCommunicator.cpp
                          ${CMAKE_CURRENT_SOURCE_DIR}/include/VMSocketCommunicator.h
                          ${CMAKE_CURRENT_SOURCE_DIR}/src/VirtioCommunicator.cpp
                          ${CMAKE_CURRENT_SOURCE_DIR}/include/VirtioCommunicator.h
                          #${CMAKE_CURRENT_SOURCE_DIR}/VmciCommunicator.cpp
                          #${CMAKE_CURRENT_SOURCE_DIR}/VmciCommunicator.h
                          ${CMAKE_CURRENT_SOURCE_DIR}/src/UvwAdapter.cpp
                          ${CMAKE_CURRENT_SOURCE_DIR}/include/UvwAdapter.h
                          "include/Endpoint.h"
                          "include/EndpointFactory.h"
                          "src/EndpointFactory.cpp"
                          "include/Endpoint_Tcp.h"
                          "src/Endpoint_Tcp.cpp"
                          )

file(GLOB COMMUNICATOR_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/include/*.h*")
file(COPY ${COMMUNICATOR_HEADERS} DESTINATION ${CMAKE_INSTALL_PREFIX}/include/gvirtus/communicator)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_INSTALL_PREFIX}/lib)

add_library(${PROJECT_NAME} SHARED ${communicator_SOURCE})
set(LIBUV ${CMAKE_INSTALL_PREFIX}/external/lib/libuv.so)
target_link_libraries(${PROJECT_NAME} log4cplus zmq ${LIBUV})