cmake_minimum_required(VERSION 3.10.2)
project(cudnn-backend)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
set(CMAKE_CXX_STANDARD 17)

find_package(CUDA)
if(CUDA_FOUND)
	include_directories(SYSTEM ${CUDA_INCLUDE_DIRS})
	link_directories(${CUDA_LIBRARIES})
	link_directories(${CUDA_CUDNN_LIBRARIES})
endif()

get_filename_component(PARENT_DIR ../ ABSOLUTE)
include_directories(SYSTEM ${PARENT_DIR})
include_directories(SYSTEM ${CMAKE_INSTALL_PREFIX}/include/gvirtus)
include_directories(SYSTEM ${CMAKE_CURRENT_SOURCE_DIR}/include)

set(backend_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/include/CudnnHandler.h
                   ${CMAKE_CURRENT_SOURCE_DIR}/src/CudnnHandler.cpp

)

add_library(${PROJECT_NAME} SHARED ${backend_SOURCE})
install(TARGETS ${PROJECT_NAME} DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/backend)
target_link_libraries(${PROJECT_NAME} ${CUDA_LIBRARIES} rt ${CUDA_CUDNN_LIBRARIES} cudnn log4cplus)
