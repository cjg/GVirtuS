cmake_minimum_required(VERSION 3.10.2)
project(gvirtus)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

get_filename_component(PARENT_DIR ../ ABSOLUTE)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PARENT_DIR}/cmake")

include_directories(SYSTEM ${CMAKE_INSTALL_PREFIX}/include/gvirtus)
################################

#   EXTERNAL LIBRARY
include(ExternalProject)
include_directories(SYSTEM ${CMAKE_INSTALL_PREFIX}/external/include)
link_directories(${CMAKE_INSTALL_PREFIX}/external/lib)

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


##
# uWebSockets
ExternalProject_Add(uWebSockets
        GIT_REPOSITORY https://github.com/xMentos5091/uWebSockets.git
        CMAKE_ARGS "-DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}/external"
        TIMEOUT 360
)

###################################

add_subdirectory(${CMAKE_SOURCE_DIR}/util ${CMAKE_BINARY_DIR}/util)
add_subdirectory(${CMAKE_SOURCE_DIR}/communicator ${CMAKE_BINARY_DIR}/communicator)
add_subdirectory(${CMAKE_SOURCE_DIR}/backend ${CMAKE_BINARY_DIR}/backend)
add_subdirectory(${CMAKE_SOURCE_DIR}/frontend ${CMAKE_BINARY_DIR}/frontend)
