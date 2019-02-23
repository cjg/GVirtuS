cmake_minimum_required(VERSION 3.12)

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

#
# nlohmann-json
ExternalProject_Add(nlohmann
        GIT_REPOSITORY https://github.com/nlohmann/json.git
        CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${EXTERNAL_INSTALL_LOCATION}
        TIMEOUT 360
        )

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
        core/src/backend/Endpoint.cpp
        core/src/backend/Property.cpp
        )
add_dependencies(gtest googletest nlohmann)
target_link_libraries(gtest ${GOOGLE_TEST_LOCATION}/lib64/libgtest.a ${GOOGLE_TEST_LOCATION}/lib64/libgtest_main.a)
install(TARGETS gtest DESTINATION ${GOOGLE_TEST_LOCATION})
