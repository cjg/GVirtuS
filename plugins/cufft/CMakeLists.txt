cmake_minimum_required(VERSION 3.17)
project(gvirtus-plugin-cufft)
find_package(CUDA REQUIRED)

find_path(CUFFT_INCLUDE_DIRECTORY
        cufft.h
        PATHS ${CUDA_INCLUDE_DIRS})
if(NOT CUFFT_INCLUDE_DIRECTORY)
    message(FATAL_ERROR "cufft.h not found")
endif()
include_directories(${CUFFT_INCLUDE_DIRECTORY} ${CUDA_INCLUDE_DIRS} util 3rdparty/include)

get_filename_component(CUDA_LIBRARIES_PATH ${CUDA_CUDART_LIBRARY} DIRECTORY FALSE)
find_library(CUFFT_LIBRARY
        cufft
        PATHS ${CUDA_LIBRARIES_PATH})
if(NOT CUFFT_LIBRARY)
    message(FATAL_ERROR "cufft library not found")
endif()
message(STATUS "Found cuFFT library: ${CUFFT_LIBRARY}")

set(CUFFT_VERSION "${CUFFT_LIBRARY}")
while(IS_SYMLINK "${CUFFT_VERSION}")
    message(STATUS "${CUFFT_VERSION}")
    execute_process(COMMAND readlink "${CUFFT_VERSION}"
            OUTPUT_VARIABLE CUFFT_VERSION
            OUTPUT_STRIP_TRAILING_WHITESPACE)
endwhile()
string(REGEX REPLACE ".*\\.so\\.([0-9]+\\.[0-9]+\\.[0-9]+)" "\\1" CUFFT_VERSION "${CUFFT_VERSION}")

message(STATUS "Found cuFFT: ${CUFFT_VERSION} (${CUFFT_INCLUDE_DIRECTORY}/cufft.h, ${CUFFT_LIBRARY})")

gvirtus_add_backend(cufft ${CUDA_VERSION}
        backend/CufftHandler.cpp)
target_link_libraries(${PROJECT_NAME} ${CUDA_cublas_LIBRARY} ${CUDA_CUDART_LIBRARY})

gvirtus_add_frontend(cufft ${CUDA_VERSION}
        frontend/Cufft.cpp
        frontend/CufftFrontend.cpp)
