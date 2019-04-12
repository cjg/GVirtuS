cmake_minimum_required(VERSION 3.10.2)
project(curand-frontend)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
set(CMAKE_CXX_STANDARD 17)

find_package(CUDA)
if(CUDA_FOUND)
    include_directories(SYSTEM ${CUDA_INCLUDE_DIRS})
endif()

include_directories(SYSTEM ${CMAKE_INSTALL_PREFIX}/include/gvirtus)
include_directories(SYSTEM ${CMAKE_CURRENT_SOURCE_DIR}/include)

set(frontend_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/include/CurandFrontend.h
                      ${CMAKE_CURRENT_SOURCE_DIR}/src/CurandFrontend.cpp
                      ${CMAKE_CURRENT_SOURCE_DIR}/src/Curand.cpp
)

add_library(gvirtus-frontend SHARED IMPORTED)
set_target_properties(gvirtus-frontend PROPERTIES IMPORTED_LOCATION ${CMAKE_INSTALL_PREFIX}/lib/libgvirtus-frontend.so)

add_library(${PROJECT_NAME} SHARED ${frontend_SOURCE})
set_target_properties(${PROJECT_NAME} PROPERTIES OUTPUT_NAME curand)
install(TARGETS ${PROJECT_NAME} DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/frontend)
target_link_libraries(${PROJECT_NAME} gvirtus-frontend rt log4cplus)
