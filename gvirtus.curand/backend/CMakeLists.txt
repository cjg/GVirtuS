cmake_minimum_required(VERSION 3.10.2)
project(curand-backend)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
set(CMAKE_CXX_STANDARD 17)

find_package(CUDA)
if(CUDA_FOUND)
    include_directories(SYSTEM ${CUDA_INCLUDE_DIRS})
endif()

get_filename_component(PARENT_DIR ../ ABSOLUTE)
include_directories(SYSTEM ${PARENT_DIR})
include_directories(SYSTEM ${CMAKE_INSTALL_PREFIX}/include/gvirtus)
include_directories(SYSTEM ${CMAKE_SOURCE_DIR}/include)

set(backend_SOURCE ${CMAKE_SOURCE_DIR}/include/CurandHandler.h
                      ${CMAKE_SOURCE_DIR}/src/CurandHandler.cpp
)

add_library(${PROJECT_NAME} SHARED ${backend_SOURCE})
install(TARGETS ${PROJECT_NAME} DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/backend)
target_link_libraries(${PROJECT_NAME} ${CUDA_LIBRARIES} rt curand log4cplus)
