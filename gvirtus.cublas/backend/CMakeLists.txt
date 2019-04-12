cmake_minimum_required(VERSION 3.10.2)
project(cublas-backend)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
set(CMAKE_CXX_STANDARD 17)

find_package(CUDA)
if(CUDA_FOUND)
    include_directories(SYSTEM ${CUDA_INCLUDE_DIRS})
endif()

include_directories(SYSTEM ${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(SYSTEM ${CMAKE_INSTALL_PREFIX}/include/gvirtus)

set(backend_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/include/CublasHandler.h
                              ${CMAKE_CURRENT_SOURCE_DIR}/src/CublasHandler.cpp
)

add_library(${PROJECT_NAME} SHARED ${backend_SOURCE})
install(TARGETS ${PROJECT_NAME} DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/backend)
target_link_libraries(${PROJECT_NAME} ${CUDA_LIBRARIES} cublas rt log4cplus)
