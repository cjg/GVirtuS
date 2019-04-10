cmake_minimum_required(VERSION 3.10.2)
project(gvirtus)

add_subdirectory(${CMAKE_SOURCE_DIR}/util ${CMAKE_BINARY_DIR}/util)
add_subdirectory(${CMAKE_SOURCE_DIR}/backend ${CMAKE_BINARY_DIR}/backend)
add_subdirectory(${CMAKE_SOURCE_DIR}/frontend ${CMAKE_BINARY_DIR}/frontend)
