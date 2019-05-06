cmake_minimum_required(VERSION 3.10.2)
project(gvirtus)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)



include_directories(SYSTEM ${CMAKE_INSTALL_PREFIX}/include/gvirtus)
################################
#   EXTERNAL LIBRARY
include(ExternalProject)
include_directories(SYSTEM ${CMAKE_INSTALL_PREFIX}/external/include)
link_directories(${CMAKE_INSTALL_PREFIX}/external/lib)
##
# libuv
ExternalProject_Add(libuv
        GIT_REPOSITORY https://github.com/libuv/libuv.git
        GIT_TAG v1.x
        CMAKE_ARGS "-DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}/external"
        TIMEOUT 360
        )

##
# uvw
ExternalProject_Add(uvw
        GIT_REPOSITORY https://github.com/skypjack/uvw.git
        CMAKE_ARGS "-DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}/external"
        TIMEOUT 360
        )

##
# nlohmann/json
ExternalProject_Add(nlohmann
        GIT_REPOSITORY https://github.com/nlohmann/json.git
        CMAKE_ARGS "-DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}/external"
        TIMEOUT 360
        )

add_dependencies(uvw libuv)
###################################

add_subdirectory(${CMAKE_SOURCE_DIR}/util ${CMAKE_BINARY_DIR}/util)
add_subdirectory(${CMAKE_SOURCE_DIR}/communicator ${CMAKE_BINARY_DIR}/communicator)
add_subdirectory(${CMAKE_SOURCE_DIR}/backend ${CMAKE_BINARY_DIR}/backend)
add_subdirectory(${CMAKE_SOURCE_DIR}/frontend ${CMAKE_BINARY_DIR}/frontend)
