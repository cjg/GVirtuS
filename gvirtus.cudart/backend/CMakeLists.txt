cmake_minimum_required(VERSION 3.10.2)
project(cudart-backend)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
set(CMAKE_CXX_STANDARD 17)

include(FindCUDA)
message("CUDA include dir: ${CUDA_INCLUDE_DIRS}")
message("CUDA libraries: ${CUDA_LIBRARIES}")
include_directories(SYSTEM ${CUDA_INCLUDE_DIRS})
link_directories(${CUDA_LIBRARIES})
list(APPEND LIB_LIST ${CUDA_LIBRARIES})

get_filename_component(PARENT_DIR ../ ABSOLUTE)
include_directories(SYSTEM ${PARENT_DIR})
include_directories(SYSTEM ${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(SYSTEM ${CMAKE_INSTALL_PREFIX}/include/gvirtus)
include_directories(SYSTEM ${CMAKE_INSTALL_PREFIX}/include/gvirtus.cudautil)
include_directories(SYSTEM ${CMAKE_INSTALL_PREFIX}/external/include)

set(backend_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/include/CudaRtHandler.h
                              ${CMAKE_CURRENT_SOURCE_DIR}/src/CudaRtHandler.cpp
                              ${CMAKE_CURRENT_SOURCE_DIR}/src/CudaRtHandler_device.cpp
                              ${CMAKE_CURRENT_SOURCE_DIR}/src/CudaRtHandler_error.cpp
                              ${CMAKE_CURRENT_SOURCE_DIR}/src/CudaRtHandler_event.cpp
                              ${CMAKE_CURRENT_SOURCE_DIR}/src/CudaRtHandler_execution.cpp
                              ${CMAKE_CURRENT_SOURCE_DIR}/src/CudaRtHandler_internal.cpp
                              ${CMAKE_CURRENT_SOURCE_DIR}/src/CudaRtHandler_memory.cpp
                              ${CMAKE_CURRENT_SOURCE_DIR}/src/CudaRtHandler_opengl.cpp
                              ${CMAKE_CURRENT_SOURCE_DIR}/src/CudaRtHandler_stream.cpp
                              ${CMAKE_CURRENT_SOURCE_DIR}/src/CudaRtHandler_surface.cpp
                              ${CMAKE_CURRENT_SOURCE_DIR}/src/CudaRtHandler_texture.cpp
                              ${CMAKE_CURRENT_SOURCE_DIR}/src/CudaRtHandler_thread.cpp
                              ${CMAKE_CURRENT_SOURCE_DIR}/src/CudaRtHandler_version.cpp
                              ${CMAKE_CURRENT_SOURCE_DIR}/src/CudaRtHandler_occupancy.cpp
)

add_library(${PROJECT_NAME} SHARED ${backend_SOURCE})
install(TARGETS ${PROJECT_NAME} DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/backend)
target_link_libraries(${PROJECT_NAME} "-Wl,--whole-archive ${CMAKE_SOURCE_DIR}/util/libcudautil.a -Wl,--no-whole-archive" ${CUDA_LIBRARIES} rt log4cplus)
