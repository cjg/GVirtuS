cmake_minimum_required(VERSION 3.2)
project(gvirtus-backend)

set(pluginsDir "-D_PLUGINS_DIR=\\\"${CMAKE_INSTALL_PREFIX}/lib/backend\\\"")
set(configFile "-D_CONFIG_FILE=\\\"${CMAKE_INSTALL_PREFIX}/etc/gvirtus.properties\\\"")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -fPIC ${pluginsDir} ${configFile}")
set(CMAKE_CXX_STANDARD 17)

message(${CMAKE_INSTALL_PREFIX})
get_filename_component(PARENT_DIR ../ ABSOLUTE)
include_directories(SYSTEM ${PARENT_DIR}/util/include)
include_directories(SYSTEM ${CMAKE_SOURCE_DIR}/include)

set(backend_SOURCE ${CMAKE_SOURCE_DIR}/src/Backend.cpp
                            ${CMAKE_SOURCE_DIR}/include/Backend.h
                            ${CMAKE_SOURCE_DIR}/src/Handler.cpp
                            ${CMAKE_SOURCE_DIR}/include/Handler.h
                            ${CMAKE_SOURCE_DIR}/src/Process.cpp
                            ${CMAKE_SOURCE_DIR}/include/Process.h
                            ${CMAKE_SOURCE_DIR}/src/main.cpp
)

file(GLOB BACKEND_HEADERS "${CMAKE_SOURCE_DIR}/include/*.h*")
file(COPY ${BACKEND_HEADERS} DESTINATION ${CMAKE_INSTALL_PREFIX}/include/gvirtus)

add_executable(${PROJECT_NAME} ${backend_SOURCE})
install(TARGETS ${PROJECT_NAME} DESTINATION ${CMAKE_INSTALL_PREFIX}/bin)
target_link_libraries(${PROJECT_NAME} "-Wl,--export-dynamic -Wl,--whole-archive ${CMAKE_INSTALL_PREFIX}/lib/libutil.a -Wl,--no-whole-archive" rt log4cplus zmq ${CMAKE_DL_LIBS})
