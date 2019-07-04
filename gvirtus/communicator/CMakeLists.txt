cmake_minimum_required(VERSION 3.10.2)
project(communicator)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -fPIC")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(SYSTEM ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_INSTALL_PREFIX}/lib/communicator)
set(communicator_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/include/Communicator.h
                          ${CMAKE_CURRENT_SOURCE_DIR}/include/CommunicatorFactory.h
                          ${CMAKE_CURRENT_SOURCE_DIR}/src/Buffer.cpp
                          ${CMAKE_CURRENT_SOURCE_DIR}/include/Buffer.h
                          ${CMAKE_CURRENT_SOURCE_DIR}/src/Result.cpp
                          ${CMAKE_CURRENT_SOURCE_DIR}/include/Result.h
                          "include/Endpoint.h"
                          "include/EndpointFactory.h"
                          "src/EndpointFactory.cpp"
                          "include/Endpoint_Tcp.h"
                          "src/Endpoint_Tcp.cpp"
                          )

file(GLOB COMMUNICATOR_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/include/*.h*")
file(COPY ${COMMUNICATOR_HEADERS} DESTINATION ${CMAKE_INSTALL_PREFIX}/include/gvirtus/communicator)


add_library(communicator SHARED ${communicator_SOURCE})
target_link_libraries(communicator log4cplus)

include(FindLibUV)
if (LibUV_FOUND)
    message("LIBUV include dir: ${LibUV_INCLUDE_DIRS}")
    message("LIBUV libraries: ${LibUV_LIBRARIES}")
    include_directories(${LibUV_INCLUDE_DIRS})
    link_directories(${LibUV_LIBRARIES})
    list(APPEND LIB_LIST ${LibUV_LIBRARIES})
else ()
    message("LibUV not found")
endif ()



## UVW
add_library(tcp-communicator SHARED src/UvwAdapter.cpp)
target_link_libraries(tcp-communicator ${LibUV_LIBRARIES})



## uWS
set(LIBUS ${CMAKE_INSTALL_PREFIX}/external/lib/libuS.so)
include(FindZLIB)
message("ZLIB include dir: ${ZLIB_INCLUDE_DIR}")
message("ZLIB libraries: ${ZLIB_LIBRARIES}")
include_directories(${ZLIB_INCLUDE_DIR})
link_directories(${ZLIB_LIBRARIES})
list(APPEND LIB_LIST ${ZLIB_LIBRARIES})

add_library(http-communicator SHARED src/HTTPAdapter.cpp)
target_link_libraries(http-communicator ${LIBUS} ${ZLIB_LIBRARIES})

## TSL UWS
#add_library(http-communicator SHARED "" "")
##target_link_libraries(http-communicator ${LIBUS} ${ZLIB_LIBRARIES})

