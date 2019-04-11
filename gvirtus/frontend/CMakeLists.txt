cmake_minimum_required(VERSION 3.10.2)
project(gvirtus-frontend)

set(pluginsDir "-D_PLUGINS_DIR=\\\"${CMAKE_INSTALL_PREFIX}/lib/frontend\\\"")
set(configFile "-D_CONFIG_FILE=\\\"${CMAKE_INSTALL_PREFIX}/etc/gvirtus.properties\\\"")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -fPIC ${pluginsDir} ${configFile}")
set(CMAKE_CXX_STANDARD 17)

get_filename_component(PARENT_DIR ../ ABSOLUTE)
include_directories(SYSTEM ${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(SYSTEM ${CMAKE_INSTALL_PREFIX}/include/gvirtus)

set(frontend_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/src/Frontend.cpp
                            ${CMAKE_CURRENT_SOURCE_DIR}/include/Frontend.h
)

file(GLOB FRONTEND_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/include/*.h*")
file(COPY ${FRONTEND_HEADERS} DESTINATION ${CMAKE_INSTALL_PREFIX}/include/gvirtus)

add_library(${PROJECT_NAME} SHARED ${frontend_SOURCE})
install(TARGETS ${PROJECT_NAME} DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
target_link_libraries(${PROJECT_NAME} "-Wl,--export-dynamic -Wl,--whole-archive ${CMAKE_SOURCE_DIR}/util/libutil.a -Wl,--no-whole-archive" rt)
