cmake_minimum_required(VERSION 3.10.2)
project(util_cudart)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
set(CMAKE_CXX_STANDARD 17)

find_package(CUDA)
if(CUDA_FOUND)
    include_directories(SYSTEM ${CUDA_INCLUDE_DIRS})
endif()

get_filename_component(PARENT_DIR ../ ABSOLUTE)
include_directories(SYSTEM ${PARENT_DIR})
include_directories(SYSTEM ${CMAKE_INSTALL_PREFIX}/include/gvirtus)
include_directories(SYSTEM ${CMAKE_CURRENT_SOURCE_DIR}/include)

set(util_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/include/CudaUtil.h
                      ${CMAKE_CURRENT_SOURCE_DIR}/src/CudaUtil.cpp
)

file(GLOB UTIL_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/include/*.h*")
file(COPY ${UTIL_HEADERS} DESTINATION ${CMAKE_INSTALL_PREFIX}/include/gvirtus.cudautil)

add_library(cudautil STATIC ${util_SOURCE})
install(TARGETS cudautil DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/frontend)
install(TARGETS cudautil DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/backend)
