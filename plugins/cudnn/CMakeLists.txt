cmake_minimum_required(VERSION 3.17)
project(gvirtus-plugin-cudnn)

find_package(CUDA REQUIRED)

find_path(CUDNN_INCLUDE_DIRECTORY
        cudnn.h
        PATHS ${CUDA_INCLUDE_DIRS})
if(NOT CUDNN_INCLUDE_DIRECTORY)
    message(FATAL_ERROR "cudnn.h not found")
endif()
include_directories(${CUDNN_INCLUDE_DIRECTORY} ${CUDA_INCLUDE_DIRS})

get_filename_component(CUDA_LIBRARIES_PATH ${CUDA_CUDART_LIBRARY} DIRECTORY FALSE)
find_library(CUDNN_LIBRARY
        cudnn
        PATHS ${CUDA_LIBRARIES_PATH})
if(NOT CUDNN_LIBRARY)
    message(FATAL_ERROR "cudnn library not found")
endif()
message(STATUS "Found cuDNN library: ${CUDNN_LIBRARY}")

set(CUDNN_VERSION "${CUDNN_LIBRARY}")
while(IS_SYMLINK "${CUDNN_VERSION}")
    message(STATUS "${CUDNN_VERSION}")
    execute_process(COMMAND readlink "${CUDNN_VERSION}"
            OUTPUT_VARIABLE CUDNN_VERSION
            OUTPUT_STRIP_TRAILING_WHITESPACE)
endwhile()
string(REGEX REPLACE ".*\\.so\\.([0-9]+\\.[0-9]+\\.[0-9]+)" "\\1" CUDNN_VERSION "${CUDNN_VERSION}")

message(STATUS "Found cuDNN: ${CUDNN_VERSION} (${CUDNN_INCLUDE_DIRECTORY}/cudnn.h, ${CUDNN_LIBRARY})")

gvirtus_add_backend(cudnn ${CUDNN_VERSION}
        backend/CudnnHandler.cpp)
target_link_libraries(${PROJECT_NAME} ${CUDNN_LIBRARY})

gvirtus_add_frontend(cudnn ${CUDNN_VERSION}
        frontend/Cudnn.cpp
        frontend/Cudnn_helper.cpp
        frontend/CudnnFrontend.cpp)
