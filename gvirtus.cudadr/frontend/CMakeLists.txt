cmake_minimum_required(VERSION 3.10.2)
project(cudadr-frontend)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -fPIC")
set(CMAKE_CXX_STANDARD 17)

get_filename_component(PARENT_DIR ../ ABSOLUTE)
include_directories(SYSTEM ${PARENT_DIR})
include_directories(SYSTEM ${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(SYSTEM ${CMAKE_INSTALL_PREFIX}/include/gvirtus)
include_directories(SYSTEM ${CMAKE_INSTALL_PREFIX}/include/gvirtus.cudautil)

set(frontend_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/src/CudaDr.cpp
                      ${CMAKE_CURRENT_SOURCE_DIR}/include/CudaDr.h
                      ${CMAKE_CURRENT_SOURCE_DIR}/include/CudaDrFrontend.h
                      ${CMAKE_CURRENT_SOURCE_DIR}/src/CudaDrFrontend.cpp
                      ${CMAKE_CURRENT_SOURCE_DIR}/src/CudaDr_context.cpp
                      ${CMAKE_CURRENT_SOURCE_DIR}/src/CudaDr_device.cpp
                      ${CMAKE_CURRENT_SOURCE_DIR}/src/CudaDr_execution.cpp
                      ${CMAKE_CURRENT_SOURCE_DIR}/src/CudaDr_initialization.cpp
                      ${CMAKE_CURRENT_SOURCE_DIR}/src/CudaDr_memory.cpp
                      ${CMAKE_CURRENT_SOURCE_DIR}/src/CudaDr_module.cpp
                      ${CMAKE_CURRENT_SOURCE_DIR}/src/CudaDr_version.cpp
                      ${CMAKE_CURRENT_SOURCE_DIR}/src/CudaDr_stream.cpp
                      ${CMAKE_CURRENT_SOURCE_DIR}/src/CudaDr_event.cpp
                      ${CMAKE_CURRENT_SOURCE_DIR}/src/CudaDr_texture.cpp
)

add_library(gvirtus-frontend SHARED IMPORTED)
set_target_properties(gvirtus-frontend PROPERTIES IMPORTED_LOCATION ${CMAKE_INSTALL_PREFIX}/lib/libgvirtus-frontend.so)

add_library(${PROJECT_NAME} SHARED ${frontend_SOURCE})
set_target_properties(${PROJECT_NAME} PROPERTIES OUTPUT_NAME cuda)
install(TARGETS ${PROJECT_NAME} DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/frontend)
target_link_libraries(${PROJECT_NAME} "-Wl,--whole-archive ${CMAKE_SOURCE_DIR}/util/libcudautil.a -Wl,--no-whole-archive" gvirtus-frontend rt log4cplus)
