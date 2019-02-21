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

#   EXTERNAL LIBRARY
include(ExternalProject)
set(EXTERNAL_INSTALL_LOCATION ${CMAKE_SOURCE_DIR}/core/external)
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
# Unit test
set(UNIT_TEST_LOCATION ${CMAKE_SOURCE_DIR}/core/unit-test)
set(UNIT_TEST_BIN ${UNIT_TEST_LOCATION}/bin)

# Property class
add_executable(property-test
        core/unit-test/src/property-test.cpp
        core/src/backend/Property.cpp
        core/src/backend/Endpoint.cpp
        core/include/exception/Exception.h
        core/include/exception/PropertyException.h
        core/include/exception/EndpointException.h
        )
install(TARGETS property-test DESTINATION ${UNIT_TEST_BIN})

# JSON class
add_executable(json-test
        core/unit-test/src/json-test.cpp
        core/include/util/JSON.h
        core/src/backend/Property.cpp
        core/src/backend/Endpoint.cpp
        core/include/exception/Exception.h
        core/include/exception/PropertyException.h
        core/include/exception/EndpointException.h
        )
add_dependencies(json-test nlohmann)
install(TARGETS json-test DESTINATION ${UNIT_TEST_BIN})


