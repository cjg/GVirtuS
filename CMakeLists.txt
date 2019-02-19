cmake_minimum_required(VERSION 3.12)
project(GVirtuSpp)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Allows you to use angle brackets instead of quotation marks in the include directives
set(INCLUDE_LOCATION ${CMAKE_SOURCE_DIR}/core/include)
include_directories(SYSTEM ${INCLUDE_LOCATION})