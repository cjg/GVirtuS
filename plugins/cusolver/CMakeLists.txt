cmake_minimum_required(VERSION 3.17)
project(gvirtus-plugin-cusolver)
find_package(CUDA REQUIRED)

find_path(CUSOLVER_INCLUDE_DIRECTORY
        cusolverDn.h
        PATHS ${CUDA_INCLUDE_DIRS})
if(NOT CUSOLVER_INCLUDE_DIRECTORY)
    message(FATAL_ERROR "cusolverDn.h not found")
endif()
include_directories(${CUSOLVER_INCLUDE_DIRECTORY} ${CUDA_INCLUDE_DIRS})

get_filename_component(CUDA_LIBRARIES_PATH ${CUDA_CUDART_LIBRARY} DIRECTORY FALSE)
find_library(CUSOLVER_LIBRARY
        cusolver
        PATHS ${CUDA_LIBRARIES_PATH})
if(NOT CUSOLVER_LIBRARY)
    message(FATAL_ERROR "cusolver library not found")
endif()
message(STATUS "Found cuSolver library: ${CUSOLVER_LIBRARY}")

set(CUSOLVER_VERSION "${CUSOLVER_LIBRARY}")
while(IS_SYMLINK "${CUSOLVER_VERSION}")
    message(STATUS "${CUSOLVER_VERSION}")
    execute_process(COMMAND readlink "${CUSOLVER_VERSION}"
            OUTPUT_VARIABLE CUSOLVER_VERSION
            OUTPUT_STRIP_TRAILING_WHITESPACE)
endwhile()
string(REGEX REPLACE ".*\\.so\\.([0-9]+\\.[0-9]+\\.[0-9]+)" "\\1" CUSOLVER_VERSION "${CUSOLVER_VERSION}")

message(STATUS "Found cuSolver: ${CUSOLVER_VERSION} (${CUSOLVER_INCLUDE_DIRECTORY}/cusolverDn.h, ${CUSOLVER_LIBRARY})")

gvirtus_add_backend(cusolver ${CUDA_VERSION}
        backend/CusolverHandler.cpp
)
target_link_libraries(${PROJECT_NAME} ${CUDA_cusolver_LIBRARY} ${CUDA_CUDART_LIBRARY})

gvirtus_add_frontend(cusolver ${CUDA_VERSION}
        frontend/Cusolver.cpp
        frontend/CusolverFrontend.cpp
        frontend/Cusolver_helper.cpp
)
