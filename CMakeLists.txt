cmake_minimum_required(VERSION 3.2)

project(GVirtuSpp)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
# Filesystem library
link_libraries(stdc++fs)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Allows you to use angle brackets instead of quotation marks in the include directives
set(INCLUDE_LOCATION ${CMAKE_SOURCE_DIR}/core/include)
include_directories(SYSTEM ${INCLUDE_LOCATION})

##############################################################################################
#   EXTERNAL LIBRARY
include(ExternalProject)
set(EXTERNAL_INSTALL_LOCATION ${CMAKE_SOURCE_DIR}/external)
set(EXTERNAL_INCLUDE_LOCATION ${EXTERNAL_INSTALL_LOCATION}/include)
include_directories(SYSTEM ${EXTERNAL_INCLUDE_LOCATION})
link_directories(${EXTERNAL_INSTALL_LOCATION}/lib)

##
# nlohmann-json
ExternalProject_Add(nlohmann
        GIT_REPOSITORY https://github.com/nlohmann/json.git
        CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${EXTERNAL_INSTALL_LOCATION}
        TIMEOUT 360
        )

##
# libuv
ExternalProject_Add(libuv
        GIT_REPOSITORY https://github.com/libuv/libuv.git
        GIT_TAG v1.x
        CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${EXTERNAL_INSTALL_LOCATION}
        TIMEOUT 360
        )

##
# uvw
ExternalProject_Add(uvw
        GIT_REPOSITORY https://github.com/skypjack/uvw.git
        CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${EXTERNAL_INSTALL_LOCATION}
        TIMEOUT 360
        )

### Fix uvw error
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/uvw-prefix/src/uvw-build/docs/html)
###

##############################################################################################
##############################################################################################
#   TESTING WITH
#   GOOGLE TEST
set(GOOGLE_TEST_LOCATION ${CMAKE_SOURCE_DIR}/unit_test)
set(BIN_TEST_LOCATION {GOOGLE_TEST_LOCATION}/bin)

ExternalProject_Add(googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${GOOGLE_TEST_LOCATION}
        TIMEOUT 360
        )

enable_testing()
include_directories(${GOOGLE_TEST_LOCATION} ${GOOGLE_TEST_LOCATION}/include)
set(SOURCE_TEST ${GOOGLE_TEST_LOCATION}/test_main.cpp)
add_executable(gtest ${SOURCE_TEST}
        unit_test/src/endpoint_test.cpp
        unit_test/src/property_test.cpp
        unit_test/src/json_test.cpp

        core/include/util/JSON.h
        core/src/communicator/endpoint/Endpoint.cpp
        core/src/backend/Property.cpp
        core/src/backend/Backend.cpp
        core/src/backend/Process.cpp
        core/src/communicator/UvwAdapter.cpp
        core/src/communicator/Communicator.cpp)

add_dependencies(gtest googletest nlohmann)
target_link_libraries(gtest ${GOOGLE_TEST_LOCATION}/lib64/libgtest.a ${GOOGLE_TEST_LOCATION}/lib64/libgtest_main.a)
install(TARGETS gtest DESTINATION ${GOOGLE_TEST_LOCATION})














































##################################################################################
add_executable(backendser core/main.cpp
        core/src/backend/Backend.cpp
        core/src/backend/Process.cpp
        core/src/backend/Property.cpp
        core/src/communicator/endpoint/EndpointTCP_IP.cpp
        core/include/communicator/endpoint/IEndpoint.h
        core/src/communicator/endpoint/Endpoint.cpp
        core/src/communicator/UvwAdapter.cpp
        core/src/communicator/Communicator.cpp
        core/include/communicator/ICommunicator.h

        )

add_dependencies(backendser nlohmann)
add_dependencies(uvw libuv)
add_dependencies(backendser libuv)
add_dependencies(backendser uvw)

set(LIBUV ${EXTERNAL_INSTALL_LOCATION}/lib64/libuv_a.a)
install(TARGETS backendser DESTINATION ${CMAKE_SOURCE_DIR}/core)
#
target_link_libraries(backendser ${LIBUV})
#
#add_executable(backendcl core/backendcl.cpp
#        core/src/backend/Backend.cpp
#        core/src/backend/Process.cpp
#        core/src/backend/Property.cpp
#        core/src/backend/Endpoint.cpp)
#
#add_dependencies(backendcl nlohmann)
#add_dependencies(uvw libuv)
#add_dependencies(backendcl libuv)
#add_dependencies(backendcl uvw)
#
#target_link_libraries(backendcl ${LIBUV})