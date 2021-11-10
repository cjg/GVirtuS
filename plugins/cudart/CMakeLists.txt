cmake_minimum_required(VERSION 3.17)
project(gvirtus-plugin-cudart)
find_package(CUDA REQUIRED)

find_path(CUDART_INCLUDE_DIRECTORY
        cuda_runtime_api.h
        PATHS ${CUDA_INCLUDE_DIRS})
if(NOT CUDART_INCLUDE_DIRECTORY)
    message(FATAL_ERROR "cuda_runtime_api.h not found")
endif()
include_directories(${CUDART_INCLUDE_DIRECTORY} ${CUDA_INCLUDE_DIRS} util 3rdparty/include)

get_filename_component(CUDA_LIBRARIES_PATH ${CUDA_CUDART_LIBRARY} DIRECTORY FALSE)
find_library(CUDART_LIBRARY
        cudart
        PATHS ${CUDA_LIBRARIES_PATH})
if(NOT CUDART_LIBRARY)
    message(FATAL_ERROR "cudart library not found")
endif()
message(STATUS "Found cudaRT library: ${CUDART_LIBRARY}")

set(CUDART_VERSION "${CUDART_LIBRARY}")
while(IS_SYMLINK "${CUDART_VERSION}")
    message(STATUS "${CUDART_VERSION}")
    execute_process(COMMAND readlink "${CUDART_VERSION}"
            OUTPUT_VARIABLE CUDART_VERSION
            OUTPUT_STRIP_TRAILING_WHITESPACE)
endwhile()
string(REGEX REPLACE ".*\\.so\\.([0-9]+\\.[0-9]+\\.[0-9]+)" "\\1" CUDART_VERSION "${CUDART_VERSION}")

message(STATUS "Found cudaRt: ${CUDART_VERSION} (${CUDART_INCLUDE_DIRECTORY}/cuda_runtime_api.h, ${CUDART_LIBRARY})")

gvirtus_add_backend(cudart ${CUDA_VERSION}
        backend/CudaRtHandler_texture.cpp
        backend/CudaRtHandler_event.cpp
        backend/CudaRtHandler_opengl.cpp
        backend/CudaRtHandler_surface.cpp
        backend/CudaRtHandler_thread.cpp
        backend/CudaRtHandler_internal.cpp
        backend/CudaRtHandler_occupancy.cpp
        backend/CudaRtHandler_execution.cpp
        backend/CudaRtHandler_device.cpp
        backend/CudaRtHandler_stream.cpp
        backend/CudaRtHandler_memory.cpp
        backend/CudaRtHandler_version.cpp
        backend/CudaRtHandler_error.cpp
        backend/CudaRtHandler.cpp
        util/CudaUtil.cpp)
target_link_libraries(${PROJECT_NAME} ${CUDA_CUDART_LIBRARY})

gvirtus_add_frontend(cudart ${CUDA_VERSION}
        frontend/CudaRt.cpp
        frontend/CudaRt_device.cpp
        frontend/CudaRt_error.cpp
        frontend/CudaRt_event.cpp
        frontend/CudaRt_execution.cpp
        frontend/CudaRtFrontend.cpp
        frontend/CudaRt_internal.cpp
        frontend/CudaRt_memory.cpp
        frontend/CudaRt_occupancy.cpp
        frontend/CudaRt_opengl.cpp
        frontend/CudaRt_stream.cpp
        frontend/CudaRt_surface.cpp
        frontend/CudaRt_texture.cpp
        frontend/CudaRt_thread.cpp
        frontend/CudaRt_version.cpp
        util/CudaUtil.cpp)

add_subdirectory(demo)