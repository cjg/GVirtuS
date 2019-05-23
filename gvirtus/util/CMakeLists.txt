cmake_minimum_required(VERSION 3.10.2)
project(util)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -fPIC")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(SYSTEM ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(util_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/include/gvirtus-type.h
                          ${CMAKE_CURRENT_SOURCE_DIR}/src/MessageDispatcher.cpp
                          ${CMAKE_CURRENT_SOURCE_DIR}/include/MessageDispatcher.h
                          ${CMAKE_CURRENT_SOURCE_DIR}/src/Mutex.cpp
                          ${CMAKE_CURRENT_SOURCE_DIR}/include/Mutex.h
                          ${CMAKE_CURRENT_SOURCE_DIR}/src/Observable.cpp
                          ${CMAKE_CURRENT_SOURCE_DIR}/include/Observable.h
                          ${CMAKE_CURRENT_SOURCE_DIR}/src/Observer.cpp
                          ${CMAKE_CURRENT_SOURCE_DIR}/include/Observer.h
                          ${CMAKE_CURRENT_SOURCE_DIR}/include/Observer.h
                          ${CMAKE_CURRENT_SOURCE_DIR}/src/Util.cpp
                          ${CMAKE_CURRENT_SOURCE_DIR}/include/Util.h
                          ${CMAKE_CURRENT_SOURCE_DIR}/src/Decoder.cpp
                          ${CMAKE_CURRENT_SOURCE_DIR}/include/Encoder.h
                          ${CMAKE_CURRENT_SOURCE_DIR}/src/Encoder.cpp
                          ${CMAKE_CURRENT_SOURCE_DIR}/include/Decoder.h
                          "include/JSON.h"
                          "include/LD_Lib.h"
                          "include/SignalException.h"
                          "include/SignalState.h"
                          "src/SignalState.cpp"
        )

file(GLOB UTIL_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/include/*.h*")
file(COPY ${UTIL_HEADERS} DESTINATION ${CMAKE_INSTALL_PREFIX}/include/gvirtus/util)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

add_library(${PROJECT_NAME} STATIC ${util_SOURCE})
target_link_libraries(${PROJECT_NAME} stdc++fs ${CMAKE_DL_LIBS})
