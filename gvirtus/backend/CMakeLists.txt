cmake_minimum_required(VERSION 3.10.2)
project(gvirtus-backend)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(pluginsDir "-D_PLUGINS_DIR=\\\"${CMAKE_INSTALL_PREFIX}/lib/backend\\\"")
set(commsDir "-D_COMMS_DIR=\\\"${CMAKE_INSTALL_PREFIX}/lib/communicator\\\"")
set(configFile "-D_CONFIG_FILE=\\\"${CMAKE_INSTALL_PREFIX}/etc/gvirtus.properties\\\"")
set(configFileJSON "-D_CONFIG_FILE_JSON=\\\"${CMAKE_INSTALL_PREFIX}/etc/properties.json\\\"")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -fPIC ${pluginsDir} ${configFile} ${commsDir} ${configFileJSON}")

get_filename_component(PARENT_DIR ../ ABSOLUTE)
include_directories(SYSTEM ${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(SYSTEM ${CMAKE_INSTALL_PREFIX}/include/gvirtus)

set(backend_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/src/Backend.cpp
                            ${CMAKE_CURRENT_SOURCE_DIR}/include/Backend.h
                            ${CMAKE_CURRENT_SOURCE_DIR}/include/Handler.h
                            ${CMAKE_CURRENT_SOURCE_DIR}/src/Process.cpp
                            ${CMAKE_CURRENT_SOURCE_DIR}/include/Process.h
                            ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp
                          "include/Property.h"
                          "src/Property.cpp"
)

file(GLOB BACKEND_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/include/*.h*")
file(COPY ${BACKEND_HEADERS} DESTINATION ${CMAKE_INSTALL_PREFIX}/include/gvirtus)


add_executable(${PROJECT_NAME} ${backend_SOURCE})
add_dependencies(${PROJECT_NAME} nlohmann)
target_link_libraries(${PROJECT_NAME} "-Wl,--export-dynamic -Wl,--whole-archive ${CMAKE_SOURCE_DIR}/util/libutil.a -Wl,--no-whole-archive" rt log4cplus zmq ${CMAKE_DL_LIBS} stdc++fs communicator)

install(TARGETS ${PROJECT_NAME} DESTINATION ${CMAKE_INSTALL_PREFIX}/bin)