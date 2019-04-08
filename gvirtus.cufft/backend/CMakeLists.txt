cmake_minimum_required(VERSION 3.10.2)
project(cufft-backend)
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

set(backend_SOURCE ${CMAKE_SOURCE_DIR}/include/CufftHandler.h
                      ${CMAKE_SOURCE_DIR}/src/CufftHandler.cpp
)

add_library(${PROJECT_NAME} SHARED ${backend_SOURCE})
add_library(cufft-backend-static STATIC ${backend_SOURCE})
set_target_properties(cufft-backend-static PROPERTIES OUTPUT_NAME cufft-backend)
install(TARGETS ${PROJECT_NAME} DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/backend)
install(TARGETS cufft-backend-static DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/backend)

target_link_libraries(${PROJECT_NAME} ${CUDA_LIBRARIES} rt cufft log4cplus)
