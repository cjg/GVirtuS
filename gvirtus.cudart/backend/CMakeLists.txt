cmake_minimum_required(VERSION 3.10.2)
project(cudart-backend)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
set(CMAKE_CXX_STANDARD 17)

find_package(CUDA)
if(CUDA_FOUND)
    include_directories(SYSTEM ${CUDA_INCLUDE_DIRS})
endif()

get_filename_component(PARENT_DIR ../ ABSOLUTE)
include_directories(SYSTEM ${PARENT_DIR})
include_directories(SYSTEM ${PARENT_DIR}/util/include)
include_directories(SYSTEM ${CMAKE_INSTALL_PREFIX}/include/gvirtus)
include_directories(SYSTEM ${CMAKE_SOURCE_DIR}/include)

set(backend_SOURCE ${CMAKE_SOURCE_DIR}/include/CudaRtHandler.h
                              ${CMAKE_SOURCE_DIR}/include/CudaRtBackend.h
                              ${CMAKE_SOURCE_DIR}/src/CudaRtBackend.cpp
                              ${CMAKE_SOURCE_DIR}/src/CudaRtHandler.cpp
                              ${CMAKE_SOURCE_DIR}/src/CudaRtHandler_device.cpp
                              ${CMAKE_SOURCE_DIR}/src/CudaRtHandler_error.cpp
                              ${CMAKE_SOURCE_DIR}/src/CudaRtHandler_event.cpp
                              ${CMAKE_SOURCE_DIR}/src/CudaRtHandler_execution.cpp
                              ${CMAKE_SOURCE_DIR}/src/CudaRtHandler_internal.cpp
                              ${CMAKE_SOURCE_DIR}/src/CudaRtHandler_memory.cpp
                              ${CMAKE_SOURCE_DIR}/src/CudaRtHandler_opengl.cpp
                              ${CMAKE_SOURCE_DIR}/src/CudaRtHandler_stream.cpp
                              ${CMAKE_SOURCE_DIR}/src/CudaRtHandler_surface.cpp
                              ${CMAKE_SOURCE_DIR}/src/CudaRtHandler_texture.cpp
                              ${CMAKE_SOURCE_DIR}/src/CudaRtHandler_thread.cpp
                              ${CMAKE_SOURCE_DIR}/src/CudaRtHandler_version.cpp
                              ${CMAKE_SOURCE_DIR}/src/CudaRtHandler_occupancy.cpp
)

add_library(${PROJECT_NAME} SHARED ${backend_SOURCE})
add_library(cudart-backend-static STATIC ${backend_SOURCE})
set_target_properties(cudart-backend-static PROPERTIES OUTPUT_NAME cudart-backend)
install(TARGETS ${PROJECT_NAME} DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/backend)
install(TARGETS cudart-backend-static DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/backend)

target_link_libraries(${PROJECT_NAME} "-Wl,--whole-archive ${CMAKE_INSTALL_PREFIX}/lib/backend/libcudautil.a -Wl,--no-whole-archive" ${CUDA_LIBRARIES} rt log4cplus)
