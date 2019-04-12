cmake_minimum_required(VERSION 3.10.2)
project(cublas-frontend)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -fPIC")
set(CMAKE_CXX_STANDARD 17)

include_directories(SYSTEM ${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(SYSTEM ${CMAKE_INSTALL_PREFIX}/include/gvirtus)

set(frontend_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/include/CublasFrontend.h
                      ${CMAKE_CURRENT_SOURCE_DIR}/src/CublasFrontend.cpp
                      ${CMAKE_CURRENT_SOURCE_DIR}/src/Cublas_helper.cpp
                      ${CMAKE_CURRENT_SOURCE_DIR}/src/Cublas_level1.cpp
                      ${CMAKE_CURRENT_SOURCE_DIR}/src/Cublas_level2.cpp
                      ${CMAKE_CURRENT_SOURCE_DIR}/src/Cublas_level3.cpp
)

add_library(gvirtus-frontend SHARED IMPORTED)
set_target_properties(gvirtus-frontend PROPERTIES IMPORTED_LOCATION ${CMAKE_INSTALL_PREFIX}/lib/libgvirtus-frontend.so)

add_library(${PROJECT_NAME} SHARED ${frontend_SOURCE})
set_target_properties(${PROJECT_NAME} PROPERTIES OUTPUT_NAME cublas)
install(TARGETS ${PROJECT_NAME} DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/frontend)
target_link_libraries(${PROJECT_NAME} gvirtus-frontend rt log4cplus)
