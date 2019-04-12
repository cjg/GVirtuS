cmake_minimum_required(VERSION 3.10.2)
project(cudadr-backend)
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
include_directories(SYSTEM ${CMAKE_INSTALL_PREFIX}/include/gvirtus.cudautil)
include_directories(SYSTEM ${CMAKE_CURRENT_SOURCE_DIR}/include)

set(backend_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/include/CudaDrBackend.h
  ${CMAKE_CURRENT_SOURCE_DIR}/src/CudaDrBackend.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/CudaDrHandler.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/include/CudaDrHandler.h
  ${CMAKE_CURRENT_SOURCE_DIR}/src/CudaDrHandler_context.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/CudaDrHandler_device.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/CudaDrHandler_execution.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/CudaDrHandler_initialization.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/CudaDrHandler_memory.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/CudaDrHandler_module.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/CudaDrHandler_version.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/CudaDrHandler_stream.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/CudaDrHandler_event.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/CudaDrHandler_texture.cpp
)

add_library(${PROJECT_NAME} SHARED ${backend_SOURCE})
install(TARGETS ${PROJECT_NAME} DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/backend)
target_link_libraries(${PROJECT_NAME} "-Wl,--whole-archive ${CMAKE_INSTALL_PREFIX}/lib/backend/libcudautil.a -Wl,--no-whole-archive" ${CUDA_LIBRARIES} rt log4cplus cuda)
